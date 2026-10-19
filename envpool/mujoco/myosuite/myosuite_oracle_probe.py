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
"""Pinned official MyoSuite oracle helper.

This binary is used only by tests. It intentionally runs in a separate Python
process from EnvPool so the official MyoSuite dependencies can stay pinned to
the upstream v2.12.2 contract without replacing EnvPool's normal runtime deps.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib
import json
import os
import platform
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from envpool.mujoco.oracle import (
    configure_mujoco_package_shared_lib,
    runfiles_repository,
)
from envpool.mujoco.render_oracle import (
    configure_macos_mujoco_renderer as _configure_macos_mujoco_renderer,
)
from envpool.mujoco.render_oracle import (
    configure_windows_mujoco_renderer as _configure_windows_mujoco_renderer,
)
from envpool.python.glfw_context import preload_windows_gl_dlls

if platform.system() == "Windows":
    preload_windows_gl_dlls(strict=True)


def _configure_linux_mujoco_renderer(render: bool) -> None:
    """Force the pinned oracle onto EnvPool CI's headless EGL renderer."""
    if not render or platform.system() != "Linux":
        return

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")


def _link_or_copy_file(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _overlay_tree(
    source: Path,
    destination: Path,
    *,
    ignore: Any = None,
    prefer_directory_symlink: bool = True,
) -> None:
    if prefer_directory_symlink:
        try:
            os.symlink(source, destination, target_is_directory=True)
            return
        except OSError:
            # Some Bazel runfiles and Windows paths cannot be symlinked.
            pass
    shutil.copytree(
        source,
        destination,
        symlinks=True,
        copy_function=_link_or_copy_file,
        ignore=ignore,
    )


def _oracle_source_path() -> Path:
    source = runfiles_repository("myosuite_source") / "myosuite"
    if not (source / "__init__.py").is_file():
        raise RuntimeError(f"could not locate MyoSuite source at {source}")
    assembled = Path(tempfile.mkdtemp(prefix="myosuite-oracle-"))
    atexit.register(shutil.rmtree, assembled, ignore_errors=True)
    package = assembled / "myosuite"
    _overlay_tree(
        source,
        package,
        ignore=lambda _root, names: (
            {"simhive"} if "simhive" in names else set()
        ),
        prefer_directory_symlink=False,
    )
    simhive = package / "simhive"
    simhive.mkdir()
    for repo, name in (
        ("myosuite_mpl_sim", "MPL_sim"),
        ("myosuite_ycb_sim", "YCB_sim"),
        ("myosuite_furniture_sim", "furniture_sim"),
        ("myosuite_myo_sim", "myo_sim"),
        ("myosuite_object_sim", "object_sim"),
    ):
        repo_path = runfiles_repository(repo)
        if not repo_path.is_dir():
            raise RuntimeError(f"could not locate {repo_path}")
        _overlay_tree(repo_path, simhive / name)
    return assembled


def _import_official() -> tuple[Any, Any, Any]:
    warnings.filterwarnings("ignore")
    configure_mujoco_package_shared_lib()
    sys.path.insert(0, str(_oracle_source_path()))
    _configure_macos_mujoco_renderer()
    _configure_windows_mujoco_renderer()
    official_myosuite = importlib.import_module("myosuite")
    gym = importlib.import_module("myosuite.utils").gym
    gym_registry_specs = official_myosuite.gym_registry_specs
    return official_myosuite, gym_registry_specs, gym


def _reset_randomization_report(task_ids: list[str]) -> dict[str, Any]:
    """Measure the pinned oracle's reset contract without state injection.

    Model parameters matter as well as observations: a randomized mass or
    terrain can leave the initial observation unchanged. Do not include RNG
    keys, counters, solver caches, or other seed metadata as proof of variation.
    """
    official, _, gym = _import_official()
    data_keys = ("qpos", "qvel", "act", "mocap_pos", "mocap_quat")
    model_keys = (
        "site_pos",
        "site_quat",
        "site_size",
        "body_pos",
        "body_quat",
        "body_mass",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_friction",
        "hfield_data",
    )

    def trace_seed(task_id: str, seed: int) -> dict[str, list[str]]:
        # Leave this scope before creating the next environment: close() does
        # not release the MuJoCo data still held by unwrapped and NumPy views.
        env = gym.make(task_id, seed=seed)
        try:
            unwrapped = env.unwrapped
            trace: dict[str, list[str]] = {}
            for _ in range(8):
                obs, _ = env.reset()
                state = {
                    "obs": obs,
                    **{
                        key: getattr(unwrapped.mj_data, key)
                        for key in data_keys
                    },
                    **{
                        key: getattr(unwrapped.mj_model, key)
                        for key in model_keys
                    },
                }
                for key, value in state.items():
                    trace.setdefault(key, []).append(
                        hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
                    )
            return trace
        finally:
            env.close()

    reports = {}
    for task_id in task_ids:
        traces = [trace_seed(task_id, seed) for seed in (11, 12, 43, 44)]
        reports[task_id] = {
            key: (
                traces[0][key] != traces[2][key]
                or traces[1][key] != traces[3][key],
                any(len(set(trace[key])) > 1 for trace in traces),
                traces[0][key] != traces[1][key]
                or traces[2][key] != traces[3][key],
            )
            for key in traces[0]
        }
    return {"version": official.__version__, "tasks": reports}


def _space_report(task_ids: list[str]) -> dict[str, Any]:
    official_myosuite, gym_registry_specs, gym = _import_official()
    registry = gym_registry_specs()
    tasks: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        spec = registry[task_id]
        env = gym.make(task_id)
        try:
            tasks[task_id] = {
                "action_shape": list(env.action_space.shape),
                "action_dtype": str(env.action_space.dtype),
                "observation_dtype": str(env.observation_space.dtype),
                "max_episode_steps": int(spec.max_episode_steps),
                "observation_shape": list(env.observation_space.shape),
            }
        except Exception as exc:
            raise RuntimeError(f"oracle space failed for {task_id}") from exc
        finally:
            env.close()
    return {
        "ids": list(official_myosuite.myosuite_env_suite),
        "tasks": tasks,
        "version": official_myosuite.__version__,
    }


def _array(value: Any) -> np.ndarray:
    return np.asarray(value)


def _jsonable_array(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable_array(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable_array(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    array = _array(value)
    if array.ndim == 0:
        return array.item()
    if array.dtype == object:
        return [str(item) for item in array.ravel()]
    return array.tolist()


def _names_from_ids(model: Any, obj_type: Any, ids: list[int]) -> list[str]:
    import mujoco

    raw_model = model.ptr if hasattr(model, "ptr") else model
    return [
        mujoco.mj_id2name(raw_model, int(obj_type), int(obj_id))
        for obj_id in ids
    ]


def _metadata_report(task_ids: list[str]) -> dict[str, Any]:
    official_myosuite, _, gym = _import_official()
    import mujoco

    tasks: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        env = gym.make(task_id)
        try:
            unwrapped = env.unwrapped
            model = unwrapped.mj_model
            data = unwrapped.mj_data
            task: dict[str, Any] = {
                "action_shape": list(env.action_space.shape),
                "entry_class": type(unwrapped).__name__,
                "frame_skip": int(unwrapped.frame_skip),
                "init_qpos": _jsonable_array(unwrapped.init_qpos),
                "init_qvel": _jsonable_array(unwrapped.init_qvel),
                "model_nq": int(model.nq),
                "model_nv": int(model.nv),
                "model_na": int(model.na),
                "model_nu": int(model.nu),
                "obs_keys": list(unwrapped.obs_keys),
                "observation_shape": list(env.observation_space.shape),
                "rwd_keys_wt": dict(unwrapped.rwd_keys_wt),
            }
            for attr in (
                "far_th",
                "goal_th",
                "hip_period",
                "max_rot",
                "min_height",
                "pose_thd",
                "reset_type",
                "target_rot",
                "target_x_vel",
                "target_y_vel",
                "terrain",
                "variant",
            ):
                if hasattr(unwrapped, attr):
                    task[attr] = _jsonable_array(getattr(unwrapped, attr))
            if hasattr(unwrapped, "tip_sids"):
                task["tip_sites"] = _names_from_ids(
                    model, mujoco.mjtObj.mjOBJ_SITE, unwrapped.tip_sids
                )
            if hasattr(unwrapped, "target_sids"):
                task["target_sites"] = _names_from_ids(
                    model, mujoco.mjtObj.mjOBJ_SITE, unwrapped.target_sids
                )
            if hasattr(unwrapped, "target_jnt_ids"):
                task["target_joints"] = _names_from_ids(
                    model, mujoco.mjtObj.mjOBJ_JOINT, unwrapped.target_jnt_ids
                )
            for attr in (
                "target_jnt_range",
                "target_jnt_value",
                "target_reach_range",
            ):
                if hasattr(unwrapped, attr):
                    task[attr] = _jsonable_array(getattr(unwrapped, attr))
            task["initial_state"] = {
                "qpos": _jsonable_array(data.qpos),
                "qvel": _jsonable_array(data.qvel),
                "act": _jsonable_array(data.act) if model.na > 0 else [],
                "qacc_warmstart": _jsonable_array(data.qacc_warmstart),
                "site_pos": _jsonable_array(model.site_pos),
                "site_quat": _jsonable_array(model.site_quat),
                "body_pos": _jsonable_array(model.body_pos),
                "body_quat": _jsonable_array(model.body_quat),
            }
            env.reset(seed=0)
            task["reset_state"] = _state_report(unwrapped)
            tasks[task_id] = task
        finally:
            env.close()
    return {"tasks": tasks, "version": official_myosuite.__version__}


def _state_report(env: Any) -> dict[str, Any]:
    model = env.mj_model
    data = env.mj_data
    state = {
        "act": _jsonable_array(data.act) if model.na > 0 else [],
        "actuator_force": _jsonable_array(data.actuator_force),
        "actuator_length": _jsonable_array(data.actuator_length),
        "actuator_velocity": _jsonable_array(data.actuator_velocity),
        "ctrl": _jsonable_array(env.robot.mj_data.ctrl),
        "geom_xpos": _jsonable_array(data.geom_xpos),
        "geom_xmat": _jsonable_array(data.geom_xmat),
        "geom_rgba": _jsonable_array(model.geom_rgba),
        "qacc_warmstart": _jsonable_array(data.qacc_warmstart),
        "body_pos": _jsonable_array(model.body_pos),
        "body_quat": _jsonable_array(model.body_quat),
        "light_xdir": _jsonable_array(data.light_xdir),
        "light_xpos": _jsonable_array(data.light_xpos),
        "mocap_pos": _jsonable_array(data.mocap_pos),
        "mocap_quat": _jsonable_array(data.mocap_quat),
        "qpos": _jsonable_array(data.qpos),
        "qvel": _jsonable_array(data.qvel),
        "site_pos": _jsonable_array(model.site_pos),
        "site_quat": _jsonable_array(model.site_quat),
        "site_size": _jsonable_array(model.site_size),
        "site_xpos": _jsonable_array(data.site_xpos),
        "site_rgba": _jsonable_array(model.site_rgba),
        "time": float(data.time),
    }
    fatigue = getattr(env, "muscle_fatigue", None)
    if fatigue is not None:
        state.update({
            "fatigue_ma": _jsonable_array(fatigue._MA),
            "fatigue_mr": _jsonable_array(fatigue._MR),
            "fatigue_mf": _jsonable_array(fatigue._MF),
            "fatigue_tl": _jsonable_array(fatigue.TL),
            "fatigue_tauact": _jsonable_array(fatigue._tauact),
            "fatigue_taudeact": _jsonable_array(fatigue._taudeact),
            "fatigue_dt": float(fatigue._dt),
        })
    return state


def _state_array(
    state: dict[str, Any], key: str, shape: tuple[int, ...]
) -> np.ndarray | None:
    value = state.get(key)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    size = int(np.prod(shape, dtype=np.int64))
    if array.size < size:
        raise ValueError(
            f"sync state {key} has {array.size} values, expected {size}"
        )
    return array[:size].reshape(shape)


def _assign_sync_array(
    state: dict[str, Any], key: str, target: np.ndarray
) -> None:
    value = state.get(key)
    if value is None:
        return
    array = np.asarray(value, dtype=np.float64).ravel()
    target_flat = target.reshape(-1)
    count = min(array.size, target_flat.size)
    target_flat[:count] = array[:count]
    if count < target_flat.size:
        target_flat[count:] = 0.0


def _assign_sync_array_if_same_size(
    state: dict[str, Any], key: str, target: np.ndarray
) -> None:
    value = state.get(key)
    if value is None:
        return
    array = np.asarray(value, dtype=np.float64).ravel()
    if array.size != target.size:
        return
    target.reshape(-1)[:] = array


def _sync_osl_phase_from_qpos(env: Any) -> None:
    controller = getattr(env, "OSL_CTRL", None)
    if controller is None:
        return
    model = env.mj_model
    data = env.mj_data
    if model.nkey < 3:
        controller.reset("e_stance")
        controller.start()
        return
    qpos = np.asarray(data.qpos, dtype=np.float64)
    key_qpos = np.asarray(model.key_qpos, dtype=np.float64).reshape(
        model.nkey, model.nq
    )
    start = min(7, model.nq)
    distances = np.sum((key_qpos[:3, start:] - qpos[start:]) ** 2, axis=1)
    phase = "e_swing" if int(np.argmin(distances)) == 1 else "e_stance"
    controller.reset(phase)
    controller.start()


def _sync_baoding_goal_from_envpool_reset_state(
    env: Any, state: dict[str, Any]
) -> None:
    if not all(
        hasattr(env, attr)
        for attr in (
            "ball_1_starting_angle",
            "ball_2_starting_angle",
            "center_pos",
            "create_goal_trajectory",
            "x_radius",
            "y_radius",
        )
    ):
        return
    radius_x, radius_y, period, angle, direction = state[
        "baoding_goal_parameters"
    ]
    task_type = type(env.which_task)
    env.which_task = task_type({0: 0, -1: 1, 1: 2}[int(direction)])
    env.ball_1_starting_angle = angle
    env.ball_2_starting_angle = angle - np.pi
    env.center_pos = np.array([-0.0125, -0.07], dtype=np.float64)
    env.x_radius = radius_x
    env.y_radius = radius_y
    env.goal = env.create_goal_trajectory(
        time_step=float(getattr(env, "dt", 0.025)), time_period=period
    )
    env.counter = 0


def _sync_task_random_state(env: Any, state: dict[str, Any]) -> None:
    npc = getattr(env, "opponent", getattr(env, "goalkeeper", None))
    if npc is None and not hasattr(env, "ref"):
        return
    high, low, inc_high, inc_low, has_uint32, uint32 = state["task_rng0"]
    bitgen = np.random.PCG64()
    bitgen.state = {
        "bit_generator": "PCG64",
        "state": {
            "state": (high << 64) | low,
            "inc": (inc_high << 64) | inc_low,
        },
        "has_uint32": has_uint32,
        "uinteger": uint32,
    }
    rng = np.random.Generator(bitgen)
    if hasattr(env, "ref"):
        env.ref.np_random = rng
        env.ref.reset()
        return
    assert npc is not None
    import pink

    if hasattr(env, "current_task"):
        env.current_task = type(env.current_task)(int(state["chase_task"][0]))
    policy, speed, scale = state["npc_parameters"]
    npc.rng = rng
    # Synchronize only the initial RNG state, not future noise or transitions.
    # Both implementations subsequently advance their own colored-noise process.
    soccer = hasattr(env, "goalkeeper")
    npc.noise_process = pink.ColoredNoiseProcess(
        beta=2, size=(2, 10 if soccer else 2000), scale=scale, rng=npc.rng
    )
    policies = (
        "static_stationary",
        "stationary",
        "random",
        "repeller",
        "chase_player",
        "block_ball",
    )
    if soccer:
        npc.goalkeeper_policy = policies[int(policy)]
        npc.block_velocity = speed
        npc.goalkeeper_vel = np.zeros(2)
    else:
        npc.opponent_policy = policies[int(policy)]
        npc.chase_velocity = speed
        npc.opponent_vel = np.zeros(2)


def _sync_fatigue_hidden_state(env: Any, state: dict[str, Any]) -> None:
    fatigue = getattr(env, "muscle_fatigue", None)
    if fatigue is None:
        return
    _assign_sync_array(state, "fatigue_ma", fatigue._MA)
    _assign_sync_array(state, "fatigue_mr", fatigue._MR)
    _assign_sync_array(state, "fatigue_mf", fatigue._MF)
    _assign_sync_array(state, "fatigue_tl", fatigue.TL)


def _sync_collision_meshes(model: Any, state: dict[str, Any]) -> None:
    """Share contact-face order only after verifying equivalent geometry."""
    keys = (
        "mesh_polymap",
        "mesh_polynormal",
        "mesh_polyvert",
        "mesh_polyvertadr",
        "mesh_polyvertnum",
    )
    native = {
        key: np.asarray(state[key]).reshape(getattr(model, key).shape)
        for key in keys
    }

    def polygons(values: dict[str, Any]) -> list[tuple[int, ...]]:
        vertices = values["mesh_polyvert"]
        return [
            tuple(sorted(vertices[int(start) : int(start + count)]))
            for start, count in zip(
                values["mesh_polyvertadr"],
                values["mesh_polyvertnum"],
                strict=False,
            )
        ]

    official = {key: getattr(model, key) for key in keys}
    native_keys, official_keys = polygons(native), polygons(official)
    native_order = sorted(range(len(native_keys)), key=native_keys.__getitem__)
    official_order = sorted(
        range(len(official_keys)), key=official_keys.__getitem__
    )
    assert [native_keys[i] for i in native_order] == [
        official_keys[i] for i in official_order
    ], "different collision mesh polygons"
    np.testing.assert_allclose(
        native["mesh_polynormal"][native_order],
        official["mesh_polynormal"][official_order],
        rtol=0,
        atol=1e-12,
    )
    for key in keys:
        getattr(model, key)[:] = native[key]


def _sync_to_envpool_reset_state(env: Any, state: dict[str, Any]) -> np.ndarray:
    """Patch the official oracle to EnvPool's reset-time MuJoCo state once."""
    import mujoco

    model = env.mj_model
    data = env.mj_data

    _assign_sync_array(state, "site_pos", model.site_pos)
    _assign_sync_array(state, "site_quat", model.site_quat)
    _assign_sync_array(state, "site_size", model.site_size)
    _assign_sync_array(state, "site_rgba", model.site_rgba)
    _assign_sync_array(state, "body_pos", model.body_pos)
    _assign_sync_array(state, "body_quat", model.body_quat)
    _assign_sync_array(state, "body_mass", model.body_mass)
    _assign_sync_array(state, "geom_pos", model.geom_pos)
    _assign_sync_array(state, "geom_quat", model.geom_quat)
    _assign_sync_array(state, "geom_size", model.geom_size)
    _assign_sync_array(state, "geom_rgba", model.geom_rgba)
    _assign_sync_array(state, "geom_friction", model.geom_friction)
    _assign_sync_array_if_same_size(state, "geom_aabb", model.geom_aabb)
    _assign_sync_array_if_same_size(state, "geom_rbound", model.geom_rbound)
    _assign_sync_array_if_same_size(state, "geom_contype", model.geom_contype)
    _assign_sync_array_if_same_size(
        state, "geom_conaffinity", model.geom_conaffinity
    )
    _assign_sync_array_if_same_size(state, "geom_type", model.geom_type)
    _assign_sync_array_if_same_size(state, "geom_condim", model.geom_condim)
    _assign_sync_array(state, "hfield_data", model.hfield_data)
    if model.nmocap > 0:
        _assign_sync_array(state, "mocap_pos", data.mocap_pos)
        _assign_sync_array(state, "mocap_quat", data.mocap_quat)

    _sync_collision_meshes(model, state)
    qpos = _state_array(state, "qpos0", data.qpos.shape)
    qvel = _state_array(state, "qvel0", data.qvel.shape)
    act = _state_array(state, "act0", data.act.shape) if model.na > 0 else None
    data.time = 0.0
    if qpos is not None:
        data.qpos[:] = qpos
    if qvel is not None:
        data.qvel[:] = qvel
    if act is not None:
        data.act[:] = act

    _assign_sync_array(state, "ctrl", data.ctrl)
    mujoco.mj_forward(model, data)
    _sync_osl_phase_from_qpos(env)
    if getattr(env, "target_jnt_value", None) is not None:
        size = np.asarray(env.target_jnt_value).size
        env.target_jnt_value = np.asarray(state["target_jnt_value"][:size])
    _sync_baoding_goal_from_envpool_reset_state(env, state)
    _sync_task_random_state(env, state)
    _sync_fatigue_hidden_state(env, state)
    if hasattr(env, "start_bid") and hasattr(env, "goal_bid"):
        # Bimanual caches its sampled platform positions outside MuJoCo.
        env.start_pos = model.body_pos[env.start_bid].copy()
        env.goal_pos = model.body_pos[env.goal_bid].copy()
    _assign_sync_array(state, "qacc0", data.qacc)
    _assign_sync_array(state, "qacc_warmstart0", data.qacc_warmstart)
    # Robot advances its own MjData. Updating only the observation copy lets
    # get_obs() immediately overwrite the synchronized pose with the old reset.
    mujoco.mj_copyData(env.robot.mj_data, model, data)
    obs = env.get_obs()
    if hasattr(env, "last_ctrl"):
        env.last_ctrl = data.ctrl.copy()
    return obs


def _trace_info(info: dict[str, Any]) -> dict[str, Any]:
    scalar_info: dict[str, Any] = {}
    for key in ("rwd_dense", "rwd_sparse", "solved", "done", "time"):
        if key in info:
            scalar_info[key] = _jsonable_array(info[key])
    if "rwd_dict" in info:
        scalar_info["rwd_dict"] = {
            key: _jsonable_array(value)
            for key, value in info["rwd_dict"].items()
            if np.asarray(value).size <= 16
        }
    return scalar_info


def _render_frame(env: Any, width: int, height: int, camera_id: int) -> Any:
    import mujoco

    unwrapped = env.unwrapped
    mujoco.mj_forward(unwrapped.mj_model, unwrapped.mj_data)
    renderer = unwrapped.mj_renderer
    frame = renderer.render_offscreen(
        width=width,
        height=height,
        camera_id=camera_id,
    )
    return frame


def _next_action(
    rng: np.random.Generator,
    low: np.ndarray,
    high: np.ndarray,
    action_mode: str,
) -> np.ndarray:
    if action_mode == "random":
        return rng.uniform(low, high).astype(np.float32)
    if action_mode == "midpoint":
        return ((low + high) * 0.5).astype(np.float32)
    if action_mode == "zero":
        return np.clip(np.zeros_like(low), low, high).astype(np.float32)
    raise ValueError(f"unknown action mode: {action_mode}")


def _rollout_report(
    task_ids: list[str], steps: int, seed: int, action_mode: str
) -> dict[str, Any]:
    official_myosuite, _, gym = _import_official()
    rng = np.random.default_rng(seed + 17)
    tasks: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        env = gym.make(task_id)
        try:
            reset = env.reset(seed=seed)
            obs = reset[0] if isinstance(reset, tuple) else reset
            low = _array(env.action_space.low).astype(np.float32)
            high = _array(env.action_space.high).astype(np.float32)
            rewards: list[float] = []
            terminals: list[bool] = []
            truncateds: list[bool] = []
            obs_checksum = [float(_array(obs).astype(np.float64).sum())]
            for _ in range(steps):
                action = _next_action(rng, low, high, action_mode)
                step = env.step(action)
                obs = step[0]
                rewards.append(float(step[1]))
                terminals.append(bool(step[2]))
                truncateds.append(bool(step[3]) if len(step) > 4 else False)
                obs_checksum.append(float(_array(obs).astype(np.float64).sum()))
            tasks[task_id] = {
                "obs_checksum": obs_checksum,
                "rewards": rewards,
                "terminated": terminals,
                "truncated": truncateds,
            }
        finally:
            env.close()
    return {"tasks": tasks, "version": official_myosuite.__version__}


def _trace_report(
    task_ids: list[str],
    steps: int,
    seed: int,
    render: bool,
    render_width: int,
    render_height: int,
    camera_id: int,
    action_mode: str,
    sync_states: dict[str, Any] | None = None,
    trace_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    official_myosuite, _, gym = _import_official()
    import mujoco

    rng = np.random.default_rng(seed + 17)
    tasks: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        task_plan = trace_plan.get(task_id, {}) if trace_plan else {}
        planned_actions = task_plan.get("actions")
        planned_resets = task_plan.get("reset_before_step", [])
        planned_sync_states = task_plan.get("sync_states", [])
        env = gym.make(task_id)
        try:
            reset = env.reset(seed=seed)
            obs = reset[0] if isinstance(reset, tuple) else reset
            unwrapped = env.unwrapped
            if planned_sync_states:
                obs = _sync_to_envpool_reset_state(
                    unwrapped, planned_sync_states[0]
                )
            elif sync_states is not None and task_id in sync_states:
                obs = _sync_to_envpool_reset_state(
                    unwrapped, sync_states[task_id]
                )
            low = _array(env.action_space.low).astype(np.float32)
            high = _array(env.action_space.high).astype(np.float32)
            frames: list[Any] = []
            if render:
                frames.append(
                    _jsonable_array(
                        _render_frame(
                            env,
                            render_width,
                            render_height,
                            camera_id,
                        )
                    )
                )
            trace: dict[str, Any] = {
                "actions": [],
                "infos": [],
                "obs": [_jsonable_array(obs)],
                "observation_dtype": str(np.asarray(obs).dtype),
                "obs_fields": [
                    (
                        key,
                        int(np.asarray(unwrapped.obs_dict[key]).size),
                        np.asarray(unwrapped.obs_dict[key]).dtype.kind in "biu",
                    )
                    for key in unwrapped.obs_keys
                ],
                "reset_state": _state_report(unwrapped),
                "rewards": [],
                "states": [],
                "terminated": [],
                "truncated": [],
            }
            trace_steps = (
                len(planned_actions) if planned_actions is not None else steps
            )
            for step_id in range(trace_steps):
                if planned_actions is None:
                    action = _next_action(rng, low, high, action_mode)
                else:
                    action = np.asarray(
                        planned_actions[step_id], dtype=np.float32
                    )
                reset_before_step = step_id < len(planned_resets) and bool(
                    planned_resets[step_id]
                )
                trace["actions"].append(_jsonable_array(action))
                if reset_before_step:
                    reset = env.reset()
                    obs = reset[0] if isinstance(reset, tuple) else reset
                    if step_id + 1 < len(planned_sync_states):
                        obs = _sync_to_envpool_reset_state(
                            unwrapped, planned_sync_states[step_id + 1]
                        )
                    else:
                        mujoco.mj_forward(unwrapped.mj_model, unwrapped.mj_data)
                    trace["obs"].append(_jsonable_array(obs))
                    trace["rewards"].append(0.0)
                    trace["terminated"].append(False)
                    trace["truncated"].append(False)
                    trace["infos"].append({})
                else:
                    # Both sides receive identical float32-representable
                    # action values. Evaluate the reference at double precision
                    # rather than reproduce NumPy's float32 intermediate
                    # rounding in native muscle and robotic control code.
                    step = env.step(action.astype(np.float64))
                    obs = step[0]
                    trace["obs"].append(_jsonable_array(obs))
                    trace["rewards"].append(float(step[1]))
                    trace["terminated"].append(bool(step[2]))
                    trace["truncated"].append(
                        bool(step[3]) if len(step) > 4 else False
                    )
                    trace["infos"].append(_trace_info(step[-1]))
                state = _state_report(unwrapped)
                if hasattr(unwrapped, "last_ctrl"):
                    state["last_ctrl"] = _jsonable_array(unwrapped.last_ctrl)
                trace["states"].append(state)
                if render:
                    frames.append(
                        _jsonable_array(
                            _render_frame(
                                env,
                                render_width,
                                render_height,
                                camera_id,
                            )
                        )
                    )
                if planned_actions is None and (
                    trace["terminated"][-1] or trace["truncated"][-1]
                ):
                    break
            if render:
                trace["frames"] = frames
            tasks[task_id] = trace
        except Exception as exc:
            raise RuntimeError(f"oracle trace failed for {task_id}") from exc
        finally:
            env.close()
    return {"tasks": tasks, "version": official_myosuite.__version__}


def _reference_interpolation_report() -> dict[str, Any]:
    """Exercise the pinned reference interpolator at endpoints and midpoints."""
    _import_official()
    from myosuite.logger.reference_motion import ReferenceMotion

    motion = ReferenceMotion({
        "time": np.array([2.0, 4.0, 6.0]),
        "robot": np.array([[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]]),
        "robot_vel": np.array([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]]),
        "object": np.array([
            [0.0, 2.0, 4.0, 1.0, 0.0, 0.0, 0.0],
            [4.0, 6.0, 8.0, 1.0, 0.0, 0.0, 0.0],
            [8.0, 10.0, 12.0, 1.0, 0.0, 0.0, 0.0],
        ]),
    })
    frames = [motion.get_reference(t) for t in (2.0, 3.0, 4.0, 5.0, 6.0)]
    return {
        key: [getattr(frame, key).tolist() for frame in frames]
        for key in ("robot", "robot_vel", "object")
    }


def _challenge_regressions_report() -> dict[str, Any]:
    """Exercise the patched oracle's physical keeper and rally termination."""
    _, _, gym = _import_official()
    from myosuite.envs.myo.myochallenge import tabletennis_v0 as tabletennis

    commanded, physical = [], []
    for seed in range(8):
        keeper = gym.make("myoChallengeSoccerP2-v0", seed=seed)
        try:
            _, _info = keeper.reset()
            env = keeper.unwrapped
            commands, positions = [], []
            for step in range(5):
                if step:
                    keeper.step(np.zeros(keeper.action_space.shape, np.float32))
                commands.append(env.mj_data.mocap_pos[0].copy().tolist())
                positions.append(
                    env.robot.mj_data.body("goalkeeper").xpos.copy().tolist()
                )
            commanded.append(commands)
            physical.append(positions)
        finally:
            keeper.close()

    rally = gym.make("myoChallengeTableTennisP0-v0")
    outcomes = []
    try:
        env = rally.unwrapped
        labels = (
            tabletennis.PingpongContactLabels.PADDLE,
            tabletennis.PingpongContactLabels.OWN,
            tabletennis.PingpongContactLabels.OPPONENT,
        )
        for masks in (
            (0, 2, 2, 0, 1, 1, 0),
            (0, 2, 2, 0, 1, 1, 0, 4),
            (0, 4),
            (1, 0, 1),
            (2, 0, 2),
            (2, 2, 2),
            (1, 0, 4, 1),
        ):
            env.obs_dict["time"] = np.array([0.0])
            env.contact_trajectory = [
                {label for bit, label in enumerate(labels) if mask & (1 << bit)}
                for mask in masks
            ]
            result = tabletennis.evaluate_pingpong_trajectory(
                env.contact_trajectory
            )
            solved = result is None
            outcomes.append([solved, bool(env._get_done(0.8, solved))])
    finally:
        rally.close()
    return {"commanded": commanded, "physical": physical, "rallies": outcomes}


def main() -> None:
    """Run the requested pinned-oracle probe and write a JSON report."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "metadata",
            "space",
            "rollout",
            "trace",
            "reset_randomization",
            "reference_interpolation",
            "challenge_regressions",
        ),
        required=True,
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_width", type=int, default=64)
    parser.add_argument("--render_height", type=int, default=48)
    parser.add_argument("--camera_id", type=int, default=-1)
    parser.add_argument("--sync_state")
    parser.add_argument("--trace_plan")
    parser.add_argument(
        "--action_mode",
        choices=("random", "midpoint", "zero"),
        default="random",
    )
    parser.add_argument("--task_id", action="append", default=[])
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    _configure_linux_mujoco_renderer(args.render)

    sync_states = (
        json.loads(Path(args.sync_state).read_text())
        if args.sync_state is not None
        else None
    )
    trace_plan = (
        json.loads(Path(args.trace_plan).read_text())
        if args.trace_plan is not None
        else None
    )

    if args.mode == "challenge_regressions":
        report = _challenge_regressions_report()
    elif args.mode == "reference_interpolation":
        report = _reference_interpolation_report()
    elif args.mode == "reset_randomization":
        report = _reset_randomization_report(args.task_id)
    elif args.mode == "space":
        report = _space_report(args.task_id)
    elif args.mode == "rollout":
        report = _rollout_report(
            args.task_id, args.steps, args.seed, args.action_mode
        )
    elif args.mode == "trace":
        report = _trace_report(
            args.task_id,
            args.steps,
            args.seed,
            args.render,
            args.render_width,
            args.render_height,
            args.camera_id,
            args.action_mode,
            sync_states,
            trace_plan,
        )
    else:
        report = _metadata_report(args.task_id)
    Path(args.out).write_text(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
