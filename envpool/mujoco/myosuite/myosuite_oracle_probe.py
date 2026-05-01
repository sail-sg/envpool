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
the upstream v2.11.6 contract without replacing EnvPool's normal runtime deps.
"""

from __future__ import annotations

import argparse
import ctypes
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

from envpool.python.glfw_context import preload_windows_gl_dlls

if platform.system() == "Windows":
    preload_windows_gl_dlls(strict=True)


def _configure_macos_mujoco_renderer() -> None:
    """Match EnvPool's CGL context setup for official render-oracle tests."""
    if platform.system() != "Darwin":
        return

    import mujoco
    from mujoco import cgl as mujoco_cgl
    from mujoco import gl_context
    from mujoco.cgl import cgl
    from mujoco.rendering.classic import renderer as classic_renderer

    class _CglContext:
        def __init__(self, width: int, height: int) -> None:
            del width, height
            attrib = cgl.CGLPixelFormatAttribute
            profile = cgl.CGLOpenGLProfile
            attrib_values = (
                attrib.CGLPFAOpenGLProfile,
                profile.CGLOGLPVersion_Legacy,
                attrib.CGLPFAColorSize,
                24,
                attrib.CGLPFAAlphaSize,
                8,
                attrib.CGLPFADepthSize,
                24,
                attrib.CGLPFAStencilSize,
                8,
                attrib.CGLPFAAllowOfflineRenderers,
                0,
                0,  # terminator
            )
            attribs = (ctypes.c_int * len(attrib_values))(*attrib_values)
            self._pixel_format = cgl.CGLPixelFormatObj()
            num_pixel_formats = cgl.GLint()
            cgl.CGLChoosePixelFormat(
                attribs,
                ctypes.byref(self._pixel_format),
                ctypes.byref(num_pixel_formats),
            )
            if not self._pixel_format or num_pixel_formats.value == 0:
                raise RuntimeError("failed to create CGL pixel format")

            self._context = cgl.CGLContextObj()
            cgl.CGLCreateContext(
                self._pixel_format,
                0,
                ctypes.byref(self._context),
            )
            if not self._context:
                cgl.CGLReleasePixelFormat(self._pixel_format)
                self._pixel_format = None
                raise RuntimeError("failed to create CGL context")
            self._locked = False

        def make_current(self) -> None:
            cgl.CGLSetCurrentContext(self._context)
            if not self._locked:
                cgl.CGLLockContext(self._context)
                self._locked = True

        def free(self) -> None:
            if self._context:
                if self._locked:
                    cgl.CGLUnlockContext(self._context)
                    self._locked = False
                cgl.CGLSetCurrentContext(None)
                cgl.CGLReleaseContext(self._context)
                self._context = None
            if self._pixel_format:
                cgl.CGLReleasePixelFormat(self._pixel_format)
                self._pixel_format = None

        def __del__(self) -> None:
            self.free()

    gl_context.GLContext = _CglContext
    mujoco.gl_context.GLContext = _CglContext
    mujoco_cgl.GLContext = _CglContext
    classic_renderer.gl_context.GLContext = _CglContext


def _configure_linux_mujoco_renderer(render: bool) -> None:
    """Force the pinned oracle onto EnvPool CI's headless EGL renderer."""
    if not render or platform.system() != "Linux":
        return

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")


def _runfiles_root() -> Path:
    path = Path(__file__).absolute()
    for parent in (path, *path.parents):
        if parent.name.endswith(".runfiles"):
            return parent
    path = Path(__file__).resolve()
    runfiles_dir = os.environ.get("RUNFILES_DIR")
    if runfiles_dir:
        return Path(runfiles_dir)
    if "TEST_SRCDIR" in os.environ:
        return Path(os.environ["TEST_SRCDIR"])
    return path.parents[3]


def _oracle_source_path() -> Path:
    runfiles = _runfiles_root()
    source = runfiles / "myosuite_source/myosuite"
    if not (source / "__init__.py").is_file():
        raise RuntimeError(f"could not locate MyoSuite source at {source}")
    assembled = Path(tempfile.mkdtemp(prefix="myosuite-oracle-"))
    package = assembled / "myosuite"
    shutil.copytree(
        source,
        package,
        symlinks=False,
        ignore=lambda _root, names: (
            {"simhive"} if "simhive" in names else set()
        ),
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
        repo_path = runfiles / repo
        if not repo_path.is_dir():
            raise RuntimeError(f"could not locate {repo_path}")
        shutil.copytree(repo_path, simhive / name, symlinks=False)
    return assembled


def _import_official() -> tuple[Any, Any, Any]:
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(_oracle_source_path()))
    _configure_macos_mujoco_renderer()
    import myosuite as official_myosuite
    from myosuite.utils import gym

    gym_registry_specs = official_myosuite.gym_registry_specs
    return official_myosuite, gym_registry_specs, gym


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
                "max_episode_steps": int(spec.max_episode_steps),
                "observation_shape": list(env.observation_space.shape),
            }
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
            model = unwrapped.sim.model
            data = unwrapped.sim.data
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
    model = env.sim.model
    data = env.sim.data
    return {
        "act": _jsonable_array(data.act) if model.na > 0 else [],
        "ctrl": _jsonable_array(data.ctrl),
        "qacc_warmstart": _jsonable_array(data.qacc_warmstart),
        "body_pos": _jsonable_array(model.body_pos),
        "body_quat": _jsonable_array(model.body_quat),
        "mocap_pos": _jsonable_array(data.mocap_pos),
        "mocap_quat": _jsonable_array(data.mocap_quat),
        "qpos": _jsonable_array(data.qpos),
        "qvel": _jsonable_array(data.qvel),
        "site_pos": _jsonable_array(model.site_pos),
        "site_quat": _jsonable_array(model.site_quat),
        "time": float(data.time),
    }


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
    value = _state_array(state, key, target.shape)
    if value is not None:
        target[...] = value


def _sync_to_envpool_reset_state(env: Any, state: dict[str, Any]) -> np.ndarray:
    """Patch the official oracle to EnvPool's reset-time MuJoCo state once."""
    sim = env.sim
    model = sim.model
    data = sim.data

    _assign_sync_array(state, "site_pos", model.site_pos)
    _assign_sync_array(state, "site_quat", model.site_quat)
    _assign_sync_array(state, "body_pos", model.body_pos)
    _assign_sync_array(state, "body_quat", model.body_quat)
    if model.nmocap > 0:
        _assign_sync_array(state, "mocap_pos", data.mocap_pos)
        _assign_sync_array(state, "mocap_quat", data.mocap_quat)

    qpos = _state_array(state, "qpos0", data.qpos.shape)
    qvel = _state_array(state, "qvel0", data.qvel.shape)
    act = _state_array(state, "act0", data.act.shape) if model.na > 0 else None
    sim.set_state(time=0.0, qpos=qpos, qvel=qvel, act=act)

    _assign_sync_array(state, "ctrl", data.ctrl)
    _assign_sync_array(state, "qacc0", data.qacc)
    _assign_sync_array(state, "qacc_warmstart0", data.qacc_warmstart)
    if hasattr(env, "last_ctrl"):
        env.last_ctrl = data.ctrl.copy()
    sim.forward()
    return env.get_obs()


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
    return env.unwrapped.sim.renderer.render_offscreen(
        width=width,
        height=height,
        camera_id=camera_id,
    )


def _rollout_report(
    task_ids: list[str], steps: int, seed: int
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
                action = rng.uniform(low, high).astype(np.float32)
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
    sync_states: dict[str, Any] | None = None,
) -> dict[str, Any]:
    official_myosuite, _, gym = _import_official()
    rng = np.random.default_rng(seed + 17)
    tasks: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        env = gym.make(task_id)
        try:
            reset = env.reset(seed=seed)
            obs = reset[0] if isinstance(reset, tuple) else reset
            unwrapped = env.unwrapped
            if sync_states is not None and task_id in sync_states:
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
                            env, render_width, render_height, camera_id
                        )
                    )
                )
            trace: dict[str, Any] = {
                "actions": [],
                "infos": [],
                "obs": [_jsonable_array(obs)],
                "reset_state": _state_report(unwrapped),
                "rewards": [],
                "states": [],
                "terminated": [],
                "truncated": [],
            }
            for _ in range(steps):
                action = rng.uniform(low, high).astype(np.float32)
                step = env.step(action)
                obs = step[0]
                trace["actions"].append(_jsonable_array(action))
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
                                env, render_width, render_height, camera_id
                            )
                        )
                    )
            if render:
                trace["frames"] = frames
            tasks[task_id] = trace
        finally:
            env.close()
    return {"tasks": tasks, "version": official_myosuite.__version__}


def main() -> None:
    """Run the requested pinned-oracle probe and write a JSON report."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("metadata", "space", "rollout", "trace"),
        required=True,
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_width", type=int, default=64)
    parser.add_argument("--render_height", type=int, default=48)
    parser.add_argument("--camera_id", type=int, default=-1)
    parser.add_argument("--sync_state")
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

    if args.mode == "space":
        report = _space_report(args.task_id)
    elif args.mode == "rollout":
        report = _rollout_report(args.task_id, args.steps, args.seed)
    elif args.mode == "trace":
        report = _trace_report(
            args.task_id,
            args.steps,
            args.seed,
            args.render,
            args.render_width,
            args.render_height,
            args.camera_id,
            sync_states,
        )
    else:
        report = _metadata_report(args.task_id)
    Path(args.out).write_text(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
