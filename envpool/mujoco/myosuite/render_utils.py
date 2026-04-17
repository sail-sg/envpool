# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared MyoSuite official-render helpers for tests and docs generation."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, cast

import gymnasium
import numpy as np

from envpool.mujoco.myosuite.config import (
    myosuite_expanded_entry,
    resolve_myosuite_model_path,
)
from envpool.mujoco.myosuite.metadata import MYOSUITE_SUITES
from envpool.mujoco.myosuite.oracle_utils import (
    load_oracle_class,
    prepare_oracle_imports,
    prepared_track_oracle_model_path,
)
from envpool.mujoco.myosuite.paths import myosuite_asset_root
from envpool.mujoco.myosuite.registration import MYOSUITE_PUBLIC_TASK_IDS
from envpool.python.glfw_context import preload_windows_gl_dlls
from envpool.registration import make_gymnasium

preload_windows_gl_dlls(strict=True)
prepare_oracle_imports(render=True)


_REORIENT_CLASS_NAMES = {
    "Geometries100EnvV0",
    "Geometries8EnvV0",
    "InDistribution",
    "OutofDistribution",
}


@dataclass(frozen=True)
class RenderCase:
    """One representative MyoSuite task rendered in docs and tests."""

    task_id: str
    label: str


@dataclass(frozen=True)
class RenderSequence:
    """Reset frame plus a short stepped render rollout for one task."""

    reset_envpool: np.ndarray
    reset_official: np.ndarray
    envpool_frames: tuple[np.ndarray, ...]
    official_frames: tuple[np.ndarray, ...]


class _RenderEarlyTerminationError(RuntimeError):
    """A candidate render seed terminated before enough frames were captured."""

    def __init__(self, task_id: str, seed: int, step_index: int):
        super().__init__(
            f"{task_id} terminated at step {step_index} before render capture "
            f"finished for seed {seed}"
        )
        self.task_id = task_id
        self.seed = seed
        self.step_index = step_index


@dataclass(frozen=True)
class _TrackReferenceSample:
    """One oracle TrackEnv reference sample consumed during a rollout."""

    time: float
    robot: np.ndarray
    robot_vel: np.ndarray | None
    object: np.ndarray


MYOSUITE_RENDER_COMPARE_CASES = (
    RenderCase("myoHandReorientID-v0", "HandReorientID"),
    RenderCase("myoLegWalk-v0", "LegWalk"),
    RenderCase("myoChallengeBimanual-v0", "ChallengeBimanual"),
    RenderCase("MyoHandAirplaneFly-v0", "TrackAirplaneFly"),
    RenderCase("myoLegHillyTerrainWalk-v0", "HillyTerrainWalk"),
)
MYOSUITE_RENDER_VALIDATE_TASK_IDS = tuple(MYOSUITE_PUBLIC_TASK_IDS)
MYOSUITE_RENDER_COMPARE_STEPS = 3
MYOSUITE_RENDER_RETRY_SEEDS = (
    11,
    23,
    37,
    53,
    71,
    89,
    107,
    131,
    151,
    173,
    197,
    223,
    257,
)
_MYODM_FIXED_TASK_IDS = frozenset(MYOSUITE_SUITES["myodm_fixed_ids"])


def official_render_thresholds(
    task_id: str,
) -> tuple[float, float] | None:
    """Return the render oracle thresholds for MyoSuite compare frames."""
    del task_id
    return (0.0, 0.0)


def _entry(task_id: str) -> dict[str, Any]:
    entry, _ = myosuite_expanded_entry(task_id)
    return entry


def _task_kwargs(task_id: str) -> dict[str, Any]:
    entry, variant_kwargs = myosuite_expanded_entry(task_id)
    return {**entry["kwargs"], **variant_kwargs}


def _tolist_array(value: Any) -> list[Any]:
    return np.asarray(value).tolist()


def _soccer_policy_id(policy: str) -> int:
    return {"block_ball": 0, "random": 1, "stationary": 2}[policy]


def _chasetag_policy_id(policy: str) -> int:
    return {
        "static_stationary": 0,
        "stationary": 1,
        "random": 2,
        "chase_player": 3,
        "repeller": 4,
    }[policy]


def _runtrack_osl_state_id(state: str) -> int:
    return {
        "e_stance": 0,
        "l_stance": 1,
        "e_swing": 2,
        "l_swing": 3,
    }[state]


def _asset_model_path(model_path: str) -> Path:
    path = Path(model_path)
    return path if path.is_absolute() else myosuite_asset_root() / model_path


def _action_shape(env: Any) -> tuple[int, ...]:
    sample = np.asarray(env.action_space.sample())
    if sample.ndim == 0:
        return (1,)
    return (1, *sample.shape)


def _seeded_actions(
    shape: tuple[int, ...], steps: int, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        rng.uniform(-0.9, 0.9, size=shape).astype(np.float32)
        for _ in range(steps)
    ]


def _zero_actions(shape: tuple[int, ...], steps: int) -> list[np.ndarray]:
    return [np.zeros(shape, dtype=np.float32) for _ in range(steps)]


def _hold_actions(
    env: Any, sync: dict[str, Any], steps: int
) -> list[np.ndarray]:
    import mujoco

    ctrl = np.asarray(sync["test_reset_ctrl"], dtype=np.float64)
    unwrapped = _unwrapped_env(env)
    model = unwrapped.sim.model
    raw = ctrl.copy()
    for index in range(model.nu):
        if (
            model.na != 0
            and model.actuator_dyntype[index] == mujoco.mjtDyn.mjDYN_MUSCLE
        ):
            activation = float(np.clip(ctrl[index], 1e-6, 1.0 - 1e-6))
            raw[index] = 0.5 - np.log(1.0 / activation - 1.0) / 5.0
    raw = np.clip(raw, -1.0, 1.0).astype(np.float32, copy=False)
    action = raw.reshape(_action_shape(env))
    return [action.copy() for _ in range(steps)]


def _render_envpool_array(env: Any) -> np.ndarray:
    frame = env.render()
    if frame is None:
        raise RuntimeError("EnvPool render() returned None")
    return cast(np.ndarray, frame)


def _unwrapped_env(env: Any) -> Any:
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _to_uint8_frame(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.max(array)) if array.size else 0.0
        if max_value <= 1.0:
            array = np.clip(array, 0.0, 1.0) * 255.0
        else:
            array = np.clip(array, 0.0, 255.0)
        return np.rint(array).astype(np.uint8)
    return np.clip(array, 0, 255).astype(np.uint8)


def _render_official_array(
    env: Any,
    *,
    width: int,
    height: int,
    camera_id: int,
    prefer_physics_render: bool = False,
) -> np.ndarray:
    import mujoco

    sim = getattr(_unwrapped_env(env), "sim", None)
    if sim is None:
        raise RuntimeError("official env does not expose sim")
    physics = getattr(sim, "sim", None)
    if (
        prefer_physics_render
        and physics is not None
        and hasattr(physics, "render")
    ):
        if hasattr(sim, "upload_height_field") and int(sim.model.nhfield) > 0:
            for hfield_id in range(int(sim.model.nhfield)):
                sim.upload_height_field(hfield_id)
        frame = physics.render(
            height=height,
            width=width,
            camera_id=camera_id,
        )
        if frame is None:
            raise RuntimeError("official physics.render() returned None")
        return _to_uint8_frame(frame)
    if hasattr(sim, "renderer"):
        if hasattr(sim, "upload_height_field") and int(sim.model.nhfield) > 0:
            if getattr(sim.renderer, "_renderer", None) is None:
                sim.renderer.setup_renderer(
                    sim.model.ptr, height=height, width=width
                )
            native_renderer = getattr(sim.renderer, "_renderer", None)
            mjr_context = getattr(native_renderer, "_mjr_context", None)
            gl_context = getattr(native_renderer, "_gl_context", None)
            if mjr_context is not None:
                if gl_context is not None:
                    gl_context.make_current()
                for hfield_id in range(int(sim.model.nhfield)):
                    mujoco.mjr_uploadHField(
                        sim.model.ptr,
                        mjr_context,
                        hfield_id,
                    )
            for hfield_id in range(int(sim.model.nhfield)):
                sim.upload_height_field(hfield_id)
        frame = sim.renderer.render_offscreen(
            width=width,
            height=height,
            camera_id=camera_id,
        )
        if frame is None:
            raise RuntimeError("official render_offscreen() returned None")
        return _to_uint8_frame(frame)
    if physics is not None and hasattr(physics, "render"):
        frame = physics.render(
            height=height,
            width=width,
            camera_id=camera_id,
        )
        if frame is None:
            raise RuntimeError("official physics.render() returned None")
        return _to_uint8_frame(frame)
    raise RuntimeError("official env does not expose a renderable sim")


def _ctor_accepts_kwarg(cls: Any, name: str) -> bool:
    sig = inspect.signature(cls.__init__)
    if name in sig.parameters:
        return True
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    )


def _dict_reference_init(
    reference: dict[str, Any], key: str, array_key: str
) -> np.ndarray:
    value = reference.get(key)
    if value is not None:
        return np.asarray(value, dtype=np.float64)
    base = np.asarray(reference[array_key], dtype=np.float64)
    if base.shape[0] == 2:
        return np.mean(base, axis=0)
    return base[0]


def _oracle_reference(reference: Any) -> Any:
    if isinstance(reference, str):
        return str(myosuite_asset_root() / reference)
    ref = dict(reference)
    time = np.asarray(ref["time"], dtype=np.float64)
    robot = np.asarray(ref["robot"], dtype=np.float64)
    obj = np.asarray(ref["object"], dtype=np.float64)
    robot_vel = (
        np.asarray(ref["robot_vel"], dtype=np.float64)
        if ref.get("robot_vel") is not None
        else None
    )
    robot_init = _dict_reference_init(ref, "robot_init", "robot")
    object_init = _dict_reference_init(ref, "object_init", "object")
    return {
        "time": time,
        "robot": robot,
        "robot_vel": robot_vel,
        "object": obj,
        "robot_init": robot_init,
        "object_init": object_init,
    }


@contextmanager
def _oracle_kwargs(task_id: str) -> Iterator[dict[str, Any]]:
    entry = _entry(task_id)
    kwargs = _task_kwargs(task_id)
    if entry["suite"] == "myodm":
        with prepared_track_oracle_model_path() as model_path:
            yield {
                "object_name": kwargs["object_name"],
                "model_path": model_path,
                "reference": _oracle_reference(kwargs["reference"]),
            }
        return
    if entry["suite"] == "myobase" and kwargs.get("edit_fn") is not None:
        edit_fn_name = str(kwargs.pop("edit_fn"))
        runtime_model_path = resolve_myosuite_model_path(
            str(kwargs["model_path"]), edit_fn_name
        )
        kwargs["model_path"] = str(_asset_model_path(runtime_model_path))
    for path_key in ("model_path", "init_pose_path"):
        if path_key in kwargs:
            kwargs[path_key] = str(_asset_model_path(kwargs[path_key]))
    yield kwargs


@contextmanager
def _make_oracle(
    task_id: str,
    *,
    seed: int,
    render_width: int,
    render_height: int,
    camera_id: int,
) -> Iterator[Any]:
    entry = _entry(task_id)
    oracle_cls = load_oracle_class(entry["entry_module"], entry["class_name"])
    with _oracle_kwargs(task_id) as kwargs:
        if _ctor_accepts_kwarg(oracle_cls, "render_mode"):
            kwargs["render_mode"] = "rgb_array"
        if camera_id != -1 and _ctor_accepts_kwarg(oracle_cls, "camera_id"):
            kwargs["camera_id"] = camera_id
        oracle = gymnasium.wrappers.TimeLimit(
            oracle_cls(seed=seed, **kwargs),
            max_episode_steps=int(entry["max_episode_steps"]),
        )
        try:
            yield oracle
        finally:
            oracle.close()


def _sync_reset_myobase(
    env: Any, task_id: str
) -> tuple[np.ndarray, dict[str, Any]]:
    obs, _ = env.reset()
    unwrapped = env.unwrapped
    sync: dict[str, Any] = {
        "test_reset_qpos": unwrapped.sim.data.qpos.copy().tolist(),
        "test_reset_qvel": unwrapped.sim.data.qvel.copy().tolist(),
        "test_reset_act": (
            unwrapped.sim.data.act.copy().tolist()
            if unwrapped.sim.model.na > 0
            else []
        ),
        "test_reset_qacc_warmstart": (
            unwrapped.sim.data.qacc_warmstart.copy().tolist()
        ),
    }
    entry = _entry(task_id)
    if entry["class_name"] == "KeyTurnEnvV0":
        obs_sim = getattr(unwrapped, "sim_obsd", unwrapped.sim)
        keyhead_sid = obs_sim.model.site_name2id("keyhead")
        key_body_id = int(obs_sim.model.site_bodyid[keyhead_sid])
        sync["test_key_body_pos"] = (
            obs_sim.model.body_pos[key_body_id].copy().tolist()
        )
    elif entry["class_name"] in {"ObjHoldFixedEnvV0", "ObjHoldRandomEnvV0"}:
        goal_sid = unwrapped.sim.model.site_name2id("goal")
        geom_id = unwrapped.sim.model.geom_name2id("object")
        sync["test_goal_pos"] = (
            unwrapped.sim.model.site_pos[goal_sid].copy().tolist()
        )
        sync["test_object_geom_size"] = (
            unwrapped.sim.model.geom_size[geom_id].copy().tolist()
        )
    elif entry["class_name"] in _REORIENT_CLASS_NAMES:
        model = unwrapped.sim.model
        target_body_id = model.body_name2id("target")
        obj_geom_id = model.geom_name2id("obj")
        target_geom_id = model.geom_name2id("target")
        top_geom_id = model.geom_name2id("top")
        bot_geom_id = model.geom_name2id("bot")
        target_top_geom_id = model.geom_name2id("t_top")
        target_bot_geom_id = model.geom_name2id("t_bot")
        object_body_id = model.body_name2id("Object")
        sync["test_target_body_quat"] = (
            model.body_quat[target_body_id].copy().tolist()
        )
        sync["test_object_geom_size"] = (
            model.geom_size[obj_geom_id].copy().tolist()
        )
        sync["test_target_geom_size"] = (
            model.geom_size[target_geom_id].copy().tolist()
        )
        sync["test_object_geom_rgba"] = (
            model.geom_rgba[obj_geom_id].copy().tolist()
        )
        sync["test_target_geom_rgba"] = (
            model.geom_rgba[target_geom_id].copy().tolist()
        )
        sync["test_object_geom_top_pos"] = (
            model.geom_pos[top_geom_id].copy().tolist()
        )
        sync["test_object_geom_bottom_pos"] = (
            model.geom_pos[bot_geom_id].copy().tolist()
        )
        sync["test_target_geom_top_pos"] = (
            model.geom_pos[target_top_geom_id].copy().tolist()
        )
        sync["test_target_geom_bottom_pos"] = (
            model.geom_pos[target_bot_geom_id].copy().tolist()
        )
        sync["test_object_body_mass"] = [float(model.body_mass[object_body_id])]
        sync["test_object_geom_type"] = int(model.geom_type[obj_geom_id])
        sync["test_target_geom_type"] = int(model.geom_type[target_geom_id])
        sync["test_object_geom_condim"] = int(model.geom_condim[obj_geom_id])
        sync["test_success_site_rgba"] = (
            model.site_rgba[unwrapped.success_indicator_sid].copy().tolist()
        )
    elif entry["class_name"] in {"PenTwirlFixedEnvV0", "PenTwirlRandomEnvV0"}:
        target_body_id = unwrapped.sim.model.body_name2id("target")
        sync["test_target_body_quat"] = (
            unwrapped.sim.model.body_quat[target_body_id].copy().tolist()
        )
    elif entry["class_name"] == "TerrainEnvV0":
        terrain_geom_id = unwrapped.sim.model.geom_name2id("terrain")
        hfield_id = int(unwrapped.sim.model.geom_dataid[terrain_geom_id])
        nrow = int(unwrapped.sim.model.hfield_nrow[hfield_id])
        ncol = int(unwrapped.sim.model.hfield_ncol[hfield_id])
        adr = int(unwrapped.sim.model.hfield_adr[hfield_id])
        sync["test_hfield_data"] = (
            unwrapped.sim.model
            .hfield_data[adr : adr + nrow * ncol]
            .copy()
            .tolist()
        )
        sync["test_terrain_geom_rgba"] = (
            unwrapped.sim.model.geom_rgba[terrain_geom_id].copy().tolist()
        )
        sync["test_terrain_geom_pos"] = (
            unwrapped.sim.model.geom_pos[terrain_geom_id].copy().tolist()
        )
        sync["test_terrain_geom_contype"] = int(
            unwrapped.sim.model.geom_contype[terrain_geom_id]
        )
        sync["test_terrain_geom_conaffinity"] = int(
            unwrapped.sim.model.geom_conaffinity[terrain_geom_id]
        )
    elif entry["class_name"] == "WalkEnvV0":
        terrain_geom_id = unwrapped.sim.model.geom_name2id("terrain")
        sync["test_terrain_geom_rgba"] = (
            unwrapped.sim.model.geom_rgba[terrain_geom_id].copy().tolist()
        )
        sync["test_terrain_geom_pos"] = (
            unwrapped.sim.model.geom_pos[terrain_geom_id].copy().tolist()
        )
        sync["test_terrain_geom_contype"] = int(
            unwrapped.sim.model.geom_contype[terrain_geom_id]
        )
        sync["test_terrain_geom_conaffinity"] = int(
            unwrapped.sim.model.geom_conaffinity[terrain_geom_id]
        )
    elif entry["class_name"] == "PoseEnvV0":
        sync["test_target_qpos"] = _tolist_array(unwrapped.target_jnt_value)
        if getattr(unwrapped, "weight_bodyname", None):
            body_id = unwrapped.sim.model.body_name2id(
                unwrapped.weight_bodyname
            )
            geom_id = unwrapped.sim.model.body_geomadr[body_id]
            sync["test_body_mass"] = [
                float(unwrapped.sim.model.body_mass[body_id])
            ]
            sync["test_geom_size0"] = [
                float(unwrapped.sim.model.geom_size[geom_id][0])
            ]
    elif entry["class_name"] == "ReachEnvV0":
        target_pos: list[float] = []
        for site_name in entry["kwargs"]["target_reach_range"]:
            site_id = unwrapped.sim.model.site_name2id(site_name + "_target")
            target_pos.extend(
                unwrapped.sim.model.site_pos[site_id].copy().tolist()
            )
        sync["test_target_pos"] = target_pos
    return np.asarray(obs, dtype=np.float64), sync


def _sync_reset_myochallenge(
    env: Any, task_id: str
) -> tuple[np.ndarray, dict[str, Any]]:
    obs, _ = env.reset()
    unwrapped = env.unwrapped
    sim = unwrapped.sim
    sync: dict[str, Any] = {
        "test_reset_qpos": sim.data.qpos.copy().tolist(),
        "test_reset_qvel": sim.data.qvel.copy().tolist(),
        "test_reset_act": (
            sim.data.act.copy().tolist() if sim.model.na > 0 else []
        ),
        "test_reset_qacc_warmstart": sim.data.qacc_warmstart.copy().tolist(),
    }
    entry = _entry(task_id)
    if entry["class_name"] == "ReorientEnvV0":
        sync["test_reset_act_dot"] = (
            sim.data.act_dot.copy().tolist() if sim.model.na > 0 else []
        )
        goal_bid = sim.model.body_name2id("target")
        target_gid = sim.model.geom_name2id("target_dice")
        object_bid = sim.model.body_name2id("Object")
        start_geom = sim.model.body_geomadr[object_bid]
        geom_count = sim.model.body_geomnum[object_bid]
        sync["test_goal_body_pos"] = (
            sim.model.body_pos[goal_bid].copy().tolist()
        )
        sync["test_goal_body_quat"] = (
            sim.model.body_quat[goal_bid].copy().tolist()
        )
        sync["test_target_geom_size"] = (
            sim.model.geom_size[target_gid].copy().tolist()
        )
        sync["test_object_geom_size"] = (
            sim.model
            .geom_size[start_geom : start_geom + geom_count]
            .reshape(-1)
            .tolist()
        )
        sync["test_object_geom_pos"] = (
            sim.model
            .geom_pos[start_geom : start_geom + geom_count]
            .reshape(-1)
            .tolist()
        )
        sync["test_object_geom_friction"] = (
            sim.model
            .geom_friction[start_geom : start_geom + geom_count]
            .reshape(-1)
            .tolist()
        )
        sync["test_object_body_mass"] = [float(sim.model.body_mass[object_bid])]
    elif entry["class_name"] == "RelocateEnvV0":
        sync["test_reset_act_dot"] = (
            sim.data.act_dot.copy().tolist() if sim.model.na > 0 else []
        )
        goal_bid = sim.model.body_name2id("target")
        goal_mocap_id = int(sim.model.body_mocapid[goal_bid])
        object_bid = sim.model.body_name2id("Object")
        start_geom = sim.model.body_geomadr[object_bid]
        geom_count = sim.model.body_geomnum[object_bid]
        sync["test_goal_body_pos"] = (
            sim.model.body_pos[goal_bid].copy().tolist()
        )
        sync["test_goal_body_quat"] = (
            sim.model.body_quat[goal_bid].copy().tolist()
        )
        if goal_mocap_id >= 0:
            sync["test_goal_mocap_pos"] = (
                sim.data.mocap_pos[goal_mocap_id].copy().tolist()
            )
            sync["test_goal_mocap_quat"] = (
                sim.data.mocap_quat[goal_mocap_id].copy().tolist()
            )
        sync["test_object_body_pos"] = (
            sim.model.body_pos[object_bid].copy().tolist()
        )
        sync["test_object_body_mass"] = [float(sim.model.body_mass[object_bid])]
        sync["test_object_geom_type"] = (
            sim.model
            .geom_type[start_geom : start_geom + geom_count]
            .astype(np.float64)
            .tolist()
        )
        sync["test_object_geom_size"] = (
            sim.model
            .geom_size[start_geom : start_geom + geom_count]
            .reshape(-1)
            .tolist()
        )
        sync["test_object_geom_pos"] = (
            sim.model
            .geom_pos[start_geom : start_geom + geom_count]
            .reshape(-1)
            .tolist()
        )
        sync["test_object_geom_quat"] = (
            sim.model
            .geom_quat[start_geom : start_geom + geom_count]
            .reshape(-1)
            .tolist()
        )
        sync["test_object_geom_rgba"] = (
            sim.model
            .geom_rgba[start_geom : start_geom + geom_count]
            .reshape(-1)
            .tolist()
        )
        sync["test_object_geom_friction"] = (
            sim.model
            .geom_friction[start_geom : start_geom + geom_count]
            .reshape(-1)
            .tolist()
        )
    elif entry["class_name"] == "BaodingEnvV1":
        sync["test_task"] = int(unwrapped.which_task.value)
        sync["test_ball1_starting_angle"] = float(
            unwrapped.ball_1_starting_angle
        )
        sync["test_ball2_starting_angle"] = float(
            unwrapped.ball_2_starting_angle
        )
        sync["test_x_radius"] = float(unwrapped.x_radius)
        sync["test_y_radius"] = float(unwrapped.y_radius)
        sync["test_goal_trajectory"] = (
            np.asarray(unwrapped.goal, dtype=np.float64).reshape(-1).tolist()
        )
        sync["test_object1_body_mass"] = [
            float(sim.model.body_mass[unwrapped.object1_bid])
        ]
        sync["test_object2_body_mass"] = [
            float(sim.model.body_mass[unwrapped.object2_bid])
        ]
        sync["test_object1_geom_size"] = (
            sim.model.geom_size[unwrapped.object1_gid].copy().tolist()
        )
        sync["test_object2_geom_size"] = (
            sim.model.geom_size[unwrapped.object2_gid].copy().tolist()
        )
        sync["test_object1_geom_friction"] = (
            sim.model.geom_friction[unwrapped.object1_gid].copy().tolist()
        )
        sync["test_object2_geom_friction"] = (
            sim.model.geom_friction[unwrapped.object2_gid].copy().tolist()
        )
    elif entry["class_name"] == "BimanualEnvV1":
        sync["test_start_pos"] = np.asarray(
            unwrapped.start_pos, dtype=np.float64
        ).tolist()
        sync["test_goal_pos"] = np.asarray(
            unwrapped.goal_pos, dtype=np.float64
        ).tolist()
        sync["test_object_body_mass"] = [
            float(sim.model.body_mass[unwrapped.obj_bid])
        ]
        sync["test_object_geom_size"] = (
            sim.model.geom_size[unwrapped.obj_gid].copy().tolist()
        )
        sync["test_object_geom_friction"] = (
            sim.model.geom_friction[unwrapped.obj_gid].copy().tolist()
        )
        obs = np.asarray(unwrapped.get_obs(), dtype=np.float64)
    elif entry["class_name"] == "RunTrack":
        terrain_geom_id = sim.model.geom_name2id("terrain")
        hfield_id = int(sim.model.geom_dataid[terrain_geom_id])
        nrow = int(sim.model.hfield_nrow[hfield_id])
        ncol = int(sim.model.hfield_ncol[hfield_id])
        adr = int(sim.model.hfield_adr[hfield_id])
        sync["test_hfield_data"] = (
            sim.model.hfield_data[adr : adr + nrow * ncol].copy().tolist()
        )
        sync["test_terrain_geom_rgba"] = (
            sim.model.geom_rgba[terrain_geom_id].copy().tolist()
        )
        sync["test_terrain_geom_pos"] = (
            sim.model.geom_pos[terrain_geom_id].copy().tolist()
        )
        sync["test_terrain_type"] = int(
            np.asarray(unwrapped.terrain_type).reshape(-1)[0]
        )
        sync["test_osl_state"] = _runtrack_osl_state_id(
            unwrapped.OSL_CTRL.STATE_MACHINE.current_state.get_name()
        )
        obs = np.asarray(unwrapped.get_obs(), dtype=np.float64)
    elif entry["class_name"] == "SoccerEnvV0":
        goalkeeper = unwrapped.goalkeeper
        sync["test_goalkeeper_pose"] = np.asarray(
            goalkeeper.get_goalkeeper_pose(), dtype=np.float64
        ).tolist()
        sync["test_goalkeeper_velocity"] = np.asarray(
            goalkeeper.goalkeeper_vel, dtype=np.float64
        ).tolist()
        sync["test_goalkeeper_noise_buffer"] = np.asarray(
            goalkeeper.noise_process.buffer, dtype=np.float64
        ).reshape(-1).tolist()
        sync["test_goalkeeper_noise_idx"] = int(goalkeeper.noise_process.idx)
        sync["test_goalkeeper_block_velocity"] = float(
            goalkeeper.block_velocity
        )
        sync["test_goalkeeper_policy"] = _soccer_policy_id(
            goalkeeper.goalkeeper_policy
        )
        # Soccer mutates the live goalkeeper after `reset()` has already built
        # the returned observation, so align against the post-reset live state.
        obs = np.asarray(unwrapped.get_obs(), dtype=np.float64)
    elif entry["class_name"] == "ChaseTagEnvV0":
        opponent = unwrapped.opponent
        terrain_geom_id = sim.model.geom_name2id("terrain")
        hfield_id = int(sim.model.geom_dataid[terrain_geom_id])
        sync["test_task"] = int(unwrapped.current_task.value)
        sync["test_hfield_data"] = (
            sim.model.hfield_data[
                int(sim.model.hfield_adr[hfield_id]) : int(sim.model.hfield_adr[hfield_id])
                + int(sim.model.hfield_nrow[hfield_id])
                * int(sim.model.hfield_ncol[hfield_id])
            ]
            .copy()
            .tolist()
        )
        sync["test_terrain_geom_rgba"] = (
            sim.model.geom_rgba[terrain_geom_id].copy().tolist()
        )
        sync["test_terrain_geom_pos"] = (
            sim.model.geom_pos[terrain_geom_id].copy().tolist()
        )
        sync["test_opponent_pose"] = np.asarray(
            opponent.get_opponent_pose(), dtype=np.float64
        ).tolist()
        sync["test_opponent_velocity"] = np.asarray(
            opponent.opponent_vel, dtype=np.float64
        ).tolist()
        sync["test_opponent_noise_buffer"] = np.asarray(
            opponent.noise_process.buffer, dtype=np.float64
        ).reshape(-1).tolist()
        sync["test_opponent_noise_idx"] = int(opponent.noise_process.idx)
        sync["test_chase_velocity"] = float(opponent.chase_velocity)
        sync["test_opponent_policy"] = _chasetag_policy_id(
            opponent.opponent_policy
        )
    elif entry["class_name"] not in {
        "RunTrack",
        "SoccerEnvV0",
        "ChaseTagEnvV0",
        "TableTennisEnvV0",
    }:
        raise ValueError(
            f"Unsupported MyoChallenge render sync task: {task_id}"
        )
    elif entry["class_name"] == "TableTennisEnvV0":
        sync["test_ball_body_pos"] = (
            sim.model.body_pos[unwrapped.id_info.ball_bid].copy().tolist()
        )
        sync["test_ball_geom_friction"] = (
            sim.model.geom_friction[unwrapped.id_info.ball_gid]
            .copy()
            .tolist()
        )
        sync["test_paddle_body_mass"] = [
            float(sim.model.body_mass[unwrapped.id_info.paddle_bid])
        ]
        sync["test_init_qpos"] = (
            np.asarray(unwrapped.init_qpos, dtype=np.float64).tolist()
        )
        sync["test_init_qvel"] = (
            np.asarray(unwrapped.init_qvel, dtype=np.float64).tolist()
        )
        sync["test_reset_act_dot"] = (
            sim.data.act_dot.copy().tolist() if sim.model.na > 0 else []
        )
    return np.asarray(obs, dtype=np.float64), sync


def _sync_reset_myodm(env: Any) -> tuple[np.ndarray, dict[str, Any]]:
    obs, _ = env.reset()
    unwrapped = env.unwrapped
    import mujoco

    integration_state = unwrapped.sim.sim.get_state(
        int(mujoco.mjtState.mjSTATE_INTEGRATION)
    )
    return np.asarray(obs, dtype=np.float64), {
        "test_reset_qpos": unwrapped.sim.data.qpos.copy().tolist(),
        "test_reset_qvel": unwrapped.sim.data.qvel.copy().tolist(),
        "test_reset_ctrl": unwrapped.sim.data.ctrl.copy().tolist(),
        "test_reset_act": (
            unwrapped.sim.data.act.copy().tolist()
            if unwrapped.sim.model.na > 0
            else []
        ),
        "test_reset_act_dot": (
            unwrapped.sim.data.act_dot.copy().tolist()
            if unwrapped.sim.model.na > 0
            else []
        ),
        "test_reset_qacc_warmstart": (
            unwrapped.sim.data.qacc_warmstart.copy().tolist()
        ),
        "test_reset_integration_state": integration_state.tolist(),
    }


def _oracle_reset_sync(
    env: Any, task_id: str
) -> tuple[np.ndarray, dict[str, Any]]:
    suite = _entry(task_id)["suite"]
    if suite == "myobase":
        return _sync_reset_myobase(env, task_id)
    if suite == "myochallenge":
        return _sync_reset_myochallenge(env, task_id)
    if suite == "myodm":
        return _sync_reset_myodm(env)
    raise ValueError(f"Unsupported MyoSuite suite for render sync: {suite}")


@contextmanager
def _record_track_reference_samples(
    env: Any,
) -> Iterator[list[_TrackReferenceSample]]:
    unwrapped = _unwrapped_env(env)
    ref = getattr(unwrapped, "ref", None)
    if ref is None or not hasattr(ref, "get_reference"):
        yield []
        return
    original_get_reference = ref.get_reference
    samples: list[_TrackReferenceSample] = []

    def wrapped_get_reference(time: Any) -> Any:
        reference = original_get_reference(time)
        samples.append(
            _TrackReferenceSample(
                time=float(time),
                robot=np.asarray(reference.robot, dtype=np.float64).copy(),
                robot_vel=(
                    None
                    if reference.robot_vel is None
                    else np.asarray(reference.robot_vel, dtype=np.float64).copy()
                ),
                object=np.asarray(reference.object, dtype=np.float64).copy(),
            )
        )
        return reference

    ref.get_reference = wrapped_get_reference
    try:
        yield samples
    finally:
        ref.get_reference = original_get_reference


def _track_reference_sync(
    samples: list[_TrackReferenceSample],
) -> dict[str, Any]:
    if not samples:
        return {}
    has_robot_vel = samples[0].robot_vel is not None
    return {
        "test_reference_time": [round(sample.time, 4) for sample in samples],
        "test_reference_robot": np.concatenate(
            [sample.robot for sample in samples]
        ).tolist(),
        "test_reference_robot_vel": (
            []
            if not has_robot_vel
            else np.concatenate(
                [cast(np.ndarray, sample.robot_vel) for sample in samples]
            ).tolist()
        ),
        "test_reference_object": np.concatenate(
            [sample.object for sample in samples]
        ).tolist(),
    }


def _make_render_envpool_env(
    task_id: str,
    *,
    seed: int,
    render_width: int,
    render_height: int,
    camera_id: int,
    sync: dict[str, Any],
) -> Any:
    return make_gymnasium(
        task_id,
        num_envs=1,
        seed=seed,
        render_mode="rgb_array",
        render_width=render_width,
        render_height=render_height,
        render_camera_id=camera_id,
        **sync,
    )


def _capture_render_sequence_once(
    task_id: str,
    *,
    steps: int,
    seed: int,
    render_width: int,
    render_height: int,
    camera_id: int,
    action_mode: str,
) -> RenderSequence:
    prefer_physics_render = False
    with _make_oracle(
        task_id,
        seed=seed,
        render_width=render_width,
        render_height=render_height,
        camera_id=camera_id,
    ) as oracle:
        suite = _entry(task_id)["suite"]
        with _record_track_reference_samples(oracle) as track_samples:
            _, sync = _oracle_reset_sync(oracle, task_id)
            reset_official = _render_official_array(
                oracle,
                width=render_width,
                height=render_height,
                camera_id=camera_id,
                prefer_physics_render=prefer_physics_render,
            ).copy()
            action_shape = _action_shape(oracle)
            if action_mode == "random":
                actions = _seeded_actions(action_shape, steps, seed + 97)
            elif action_mode == "hold":
                actions = _hold_actions(oracle, sync, steps)
            elif action_mode == "playback":
                actions = _zero_actions(action_shape, steps)
            elif action_mode == "zero":
                actions = _zero_actions(action_shape, steps)
            else:
                raise ValueError(f"Unsupported render action mode: {action_mode}")
            official_frames: list[np.ndarray] = []
            step_outcomes: list[tuple[bool, bool]] = []
            if suite == "myodm":
                if action_mode == "playback":
                    sync["test_playback_reference"] = True
                    for _ in range(steps):
                        _unwrapped_env(oracle).playback()
                        step_outcomes.append((False, False))
                        official_frames.append(
                            _render_official_array(
                                oracle,
                                width=render_width,
                                height=render_height,
                                camera_id=camera_id,
                                prefer_physics_render=prefer_physics_render,
                            ).copy()
                        )
                else:
                    for action in actions:
                        _, _, terminated0, truncated0, _ = oracle.step(action[0])
                        step_outcomes.append((bool(terminated0), bool(truncated0)))
                        official_frames.append(
                            _render_official_array(
                                oracle,
                                width=render_width,
                                height=render_height,
                                camera_id=camera_id,
                                prefer_physics_render=prefer_physics_render,
                            ).copy()
                        )
                if task_id not in _MYODM_FIXED_TASK_IDS:
                    sync.update(_track_reference_sync(track_samples))

        env = _make_render_envpool_env(
            task_id,
            seed=seed,
            render_width=render_width,
            render_height=render_height,
            camera_id=camera_id,
            sync=sync,
        )
        try:
            env.reset()
            reset_envpool = _render_envpool_array(env)[0].copy()
            envpool_frames: list[np.ndarray] = []
            if suite == "myodm":
                for step_index, (action, (terminated0, truncated0)) in enumerate(
                    zip(actions, step_outcomes, strict=True),
                    start=1,
                ):
                    _, _, terminated1, truncated1, _ = env.step(action)
                    if terminated0 != bool(terminated1[0]):
                        raise AssertionError(
                            f"{task_id} terminated mismatch at render step "
                            f"{step_index}"
                        )
                    if truncated0 != bool(truncated1[0]):
                        raise AssertionError(
                            f"{task_id} truncated mismatch at render step "
                            f"{step_index}"
                        )
                    if (
                        action_mode != "playback"
                        and (
                            step_index < steps
                            and (terminated0 or truncated0)
                        )
                    ):
                        raise _RenderEarlyTerminationError(
                            task_id, seed, step_index
                        )
                    envpool_frames.append(_render_envpool_array(env)[0].copy())
                return RenderSequence(
                    reset_envpool=reset_envpool,
                    reset_official=reset_official,
                    envpool_frames=tuple(envpool_frames),
                    official_frames=tuple(official_frames),
                )

            official_frames = []
            for step_index, action in enumerate(actions, start=1):
                _, _, terminated0, truncated0, _ = oracle.step(action[0])
                _, _, terminated1, truncated1, _ = env.step(action)
                if bool(terminated0) != bool(terminated1[0]):
                    raise AssertionError(
                        f"{task_id} terminated mismatch at render step "
                        f"{step_index}"
                    )
                if bool(truncated0) != bool(truncated1[0]):
                    raise AssertionError(
                        f"{task_id} truncated mismatch at render step "
                        f"{step_index}"
                    )
                if step_index < steps and (terminated0 or truncated0):
                    raise _RenderEarlyTerminationError(
                        task_id, seed, step_index
                    )
                envpool_frames.append(_render_envpool_array(env)[0].copy())
                official_frames.append(
                    _render_official_array(
                        oracle,
                        width=render_width,
                        height=render_height,
                        camera_id=camera_id,
                        prefer_physics_render=prefer_physics_render,
                    ).copy()
                )
            return RenderSequence(
                reset_envpool=reset_envpool,
                reset_official=reset_official,
                envpool_frames=tuple(envpool_frames),
                official_frames=tuple(official_frames),
            )
        finally:
            env.close()


def capture_render_sequence(
    task_id: str,
    *,
    steps: int = MYOSUITE_RENDER_COMPARE_STEPS,
    seed: int = 7,
    render_width: int = 192,
    render_height: int = 192,
    camera_id: int = -1,
    action_mode: str = "random",
    retry_seeds: tuple[int, ...] = (),
) -> RenderSequence:
    """Capture reset plus a short EnvPool-oracle render rollout."""
    last_early_termination: _RenderEarlyTerminationError | None = None
    action_modes = (action_mode,)
    if action_mode == "random":
        if _entry(task_id)["suite"] == "myodm":
            action_modes += ("hold", "zero", "playback")
        else:
            action_modes += ("zero",)
    for candidate_action_mode in action_modes:
        for candidate_seed in (seed, *retry_seeds):
            try:
                return _capture_render_sequence_once(
                    task_id,
                    steps=steps,
                    seed=candidate_seed,
                    render_width=render_width,
                    render_height=render_height,
                    camera_id=camera_id,
                    action_mode=candidate_action_mode,
                )
            except _RenderEarlyTerminationError as exc:
                last_early_termination = exc
                continue
    if last_early_termination is not None:
        raise RuntimeError(
            f"{task_id} terminated before {steps} render steps for all "
            f"candidate action modes {action_modes} and seeds "
            f"{(seed, *retry_seeds)}"
        ) from last_early_termination
    raise RuntimeError("capture_render_sequence exhausted without attempting a seed")
