# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate native task metadata, model assets, and ahead-of-time CPU kernels."""

import argparse
import dataclasses
import enum
import importlib.metadata
import json
import math
import shutil
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import warp as wp
from capture import export_physics
from pack import array_parts, pack_binary, prune_assets

from third_party.mjlab.oracle_util import configure_cache


def encode(value: Any) -> Any:
    """Serialize upstream configuration and tensor state into native metadata."""
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return encode(value.tolist())
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if dataclasses.is_dataclass(value):
        return {
            f.name: encode(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if callable(value):
        cls = value if hasattr(value, "__qualname__") else type(value)
        result = {"callable": cls.__module__ + "." + cls.__qualname__}
        if not isinstance(value, type) and not hasattr(value, "__qualname__"):
            result["state"] = {
                k: encode(v)
                for k, v in vars(value).items()
                if isinstance(
                    v, (torch.Tensor, np.ndarray, str, int, float, bool)
                )
            }
        return result
    if isinstance(value, float) and math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot export {type(value).__name__}")


def metadata(env: Any, task: str) -> dict[str, Any]:
    """Capture the resolved spaces, managers, sensors, and entity indexing."""

    def space_shape(space: Any) -> Any:
        if hasattr(space, "spaces"):
            return {k: space_shape(v) for k, v in space.spaces.items()}
        return list(space.shape)

    result: dict[str, Any] = {
        "id": task,
        "decimation": env.cfg.decimation,
        "physics_dt": env.physics_dt,
        "step_dt": env.step_dt,
        "max_episode_steps": env.max_episode_length,
        "episode_length_s": env.cfg.episode_length_s,
        "scale_rewards_by_dt": env.cfg.scale_rewards_by_dt,
        "action_size": env.single_action_space.shape[0],
        "observation_shapes": space_shape(env.single_observation_space),
        "viewer": encode(env.cfg.viewer),
        "terrain": encode(env.cfg.scene.terrain),
        "expanded_fields": sorted(env.sim.expanded_fields),
        "default_model_fields": {
            name: np.asarray(getattr(env.sim.mj_model, name)).tolist()
            for name in sorted(env.sim.expanded_fields)
        },
        "entities": {},
        "sensors": {},
    }
    terrain = env.scene.terrain
    if terrain is not None and terrain.terrain_origins is not None:
        generator = env.cfg.scene.terrain.terrain_generator
        from mjlab.terrains.heightfield_terrains import (
            HfRandomUniformTerrainCfg,
        )

        columns = list(generator.sub_terrains.values())
        random_heightfields = []
        model = env.sim.mj_model
        for geom in np.flatnonzero(
            model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD
        ):
            row = round(
                model.geom_pos[geom, 0] / generator.size[0]
                + (generator.num_rows - 1) / 2
            )
            col = round(
                model.geom_pos[geom, 1] / generator.size[1]
                + (len(columns) - 1) / 2
            )
            if isinstance(columns[col], HfRandomUniformTerrainCfg):
                terrain_cfg = columns[col]
                if terrain_cfg.downsampled_scale not in (
                    None,
                    terrain_cfg.horizontal_scale,
                ):
                    raise ValueError(
                        "the native builtin presets require interpolation at the original grid nodes"
                    )
                material = model.geom_matid[geom]
                random_heightfields.append({
                    "hfield": int(model.geom_dataid[geom]),
                    "geom": int(geom),
                    "row": row,
                    "column": col,
                    "texture": int(
                        model.mat_texid[
                            material, mujoco.mjtTextureRole.mjTEXROLE_RGB
                        ]
                    ),
                    "cfg": encode(terrain_cfg),
                })
        result["terrain_state"] = {
            "origins": encode(terrain.terrain_origins),
            "types": encode(terrain.terrain_types),
            "random_heightfields": random_heightfields,
        }
    for name, entity in env.scene.entities.items():
        result["entities"][name] = {
            "indexing": {
                f.name: encode(getattr(entity.indexing, f.name))
                for f in dataclasses.fields(entity.indexing)
                if f.name.endswith(("_ids", "_adr")) or f.name == "mocap_id"
            }
            | {"root_body_id": entity.indexing.root_body_id},
            "body_names": entity.body_names,
            "joint_names": entity.joint_names,
            "site_names": entity.site_names,
            "default_root_state": encode(entity.data.default_root_state),
            "default_joint_pos": encode(entity.data.default_joint_pos),
            "default_joint_vel": encode(entity.data.default_joint_vel),
            "soft_joint_pos_limits": encode(entity.data.soft_joint_pos_limits),
            "is_fixed_base": entity.data.is_fixed_base,
        }
    for name, sensor in env.scene.sensors.items():
        state = {}
        for key, value in vars(sensor).items():
            if isinstance(
                value, (torch.Tensor, np.ndarray, str, int, float, bool)
            ):
                state[key] = encode(value)
        if hasattr(sensor, "_slots"):
            state["slots"] = [
                {
                    f.name: encode(getattr(slot, f.name))
                    for f in dataclasses.fields(slot)
                    if f.name != "data_view"
                }
                for slot in sensor._slots
            ]
        if hasattr(sensor, "_frame_infos"):
            state["frames"] = encode(sensor._frame_infos)
        result["sensors"][name] = {
            "type": type(sensor).__name__,
            "cfg": encode(sensor.cfg),
            "state": state,
        }
    for name in ("reward", "termination", "event", "curriculum", "observation"):
        manager = getattr(env, name + "_manager")
        if name == "event":
            result[name] = {
                key: encode(manager.get_term_cfg(key)) for key in env.cfg.events
            }
        elif hasattr(manager, "_term_cfgs"):
            cfgs = manager._term_cfgs
            result[name] = encode(
                cfgs
                if isinstance(cfgs, dict)
                else dict(zip(manager._term_names, cfgs, strict=True))
            )
        elif hasattr(manager, "_group_obs_term_cfgs"):
            result[name] = {
                group: dict(
                    zip(
                        manager._group_obs_term_names[group],
                        encode(terms),
                        strict=True,
                    )
                )
                for group, terms in manager._group_obs_term_cfgs.items()
            }
        else:
            result[name] = encode(getattr(env.cfg, name + "s", {}))
    result["observation_groups"] = encode(env.cfg.observations)
    result["command"] = encode(env.cfg.commands)
    if "motion" in result["command"]:
        result["command"]["motion"]["motion_file"] = ""
    result["action"] = {}
    for name, term in env.action_manager._terms.items():
        result["action"][name] = {
            "cfg": encode(term.cfg),
            "state": {
                k: encode(v)
                for k, v in vars(term).items()
                if isinstance(
                    v, (torch.Tensor, np.ndarray, str, int, float, bool, slice)
                )
            },
        }
    return result


def main() -> None:
    """Export every builtin task from the pinned official registry."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--task", action="append")
    args = parser.parse_args()
    cache = tempfile.TemporaryDirectory(prefix="mjlab-export-")
    configure_cache(Path(cache.name))
    wp.config.kernel_cache_dir = str(args.cache or Path(cache.name))
    import mjlab
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import list_tasks, load_env_cfg
    from motion_fixture import generate_motion

    root = args.output
    for name in ("assets", "cpp", "testdata"):
        (root / name).mkdir(parents=True, exist_ok=True)
    registry = list_tasks()
    task_infos = []
    for index, task in enumerate(registry):
        if args.task and task not in args.task:
            continue
        cfg = load_env_cfg(task)
        cfg.scene.num_envs = 1
        cfg.seed = 0
        if "motion" in cfg.commands:
            fixture = root / "testdata/motion.npz"
            if not fixture.exists():
                generate_motion(cfg, fixture)
            cfg.commands["motion"].motion_file = str(fixture)
        folder = root / "assets" / str(index)
        folder.mkdir(parents=True, exist_ok=True)
        env = ManagerBasedRlEnv(cfg, device="cpu")
        context = env.sim._sensor_context
        scene_bounds = (
            None
            if context is None
            else (
                wp.clone(context.render_context.lower),
                wp.clone(context.render_context.upper),
            )
        )
        env.reset()
        info = metadata(env, task)
        exported = export_physics(env.sim, folder / "physics", scene_bounds)
        info["resources"] = exported["resources"]
        info["bindings"] = {
            k: {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "bytes": v.capacity,
            }
            for k, v in exported["bindings"].items()
        }
        (folder / "task.json").write_text(
            json.dumps(encode(info), indent=2, allow_nan=False) + "\n"
        )
        mujoco.mj_saveModel(env.sim.mj_model, str(folder / "model.mjb"))
        host_model = env.sim.mj_model
        pack_binary(
            folder / "model.mjb",
            (
                part
                for name in dir(host_model)
                if isinstance(value := getattr(host_model, name), np.ndarray)
                for part in array_parts(host_model, name, value)
            ),
        )
        pack_binary(
            folder / "physics.wrp",
            (
                part
                for name, value in exported["bindings"].items()
                for part in array_parts(
                    host_model, name.removeprefix("model."), value.numpy()
                )
            ),
        )
        for key, module in exported["modules"].items():
            # Object filenames include a CPU-feature suffix on x86; Warp's
            # generated source is named after the enclosing module directory.
            module_dir = Path(module["binary_path"]).parent
            source = (module_dir / (module_dir.name + ".cpp")).read_text()
            (root / "cpp" / (key + ".cc")).write_text(source)
        # Native builds compile the source with their own compiler; no JIT
        # objects, LLVM libraries, or Python modules belong in runtime assets.
        shutil.rmtree(folder / "physics_modules")
        task_infos.append({
            "id": task,
            "asset": str(index),
            "action_size": info["action_size"],
            "observation_shapes": info["observation_shapes"],
            "max_episode_steps": info["max_episode_steps"],
        })
        env.close()
        print("EXPORTED", task, flush=True)
    (root / "registry.json").write_text(json.dumps(task_infos, indent=2) + "\n")
    prune_assets(root / "assets")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("mujoco", "mujoco-warp", "warp-lang", "torch", "numpy")
    }
    source_metadata = (
        Path(mjlab.__file__).resolve().parents[2] / "pyproject.toml"
    )
    versions["mjlab"] = tomllib.loads(source_metadata.read_text())["project"][
        "version"
    ]
    (root / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    cache.cleanup()


if __name__ == "__main__":
    main()
