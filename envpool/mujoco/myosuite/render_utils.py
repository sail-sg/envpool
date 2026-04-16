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

import ctypes
import inspect
import os
import platform
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, cast

import gymnasium
import numpy as np

import envpool.mujoco.myosuite.registration  # noqa: F401
from envpool.mujoco.myosuite.metadata import MYOSUITE_DIRECT_ENTRY_BY_ID
from envpool.mujoco.myosuite.oracle_utils import (
    load_oracle_class,
    prepared_track_oracle_model_path,
)
from envpool.mujoco.myosuite.paths import myosuite_asset_root
from envpool.python.glfw_context import preload_windows_gl_dlls
from envpool.registration import make_gymnasium

preload_windows_gl_dlls(strict=True)


def _configure_linux_mujoco_gl() -> None:
    if platform.system() != "Linux":
        return
    if os.environ.get("MUJOCO_GL"):
        if os.environ["MUJOCO_GL"] == "egl":
            os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        return
    for backend in ("egl", "osmesa"):
        env = dict(os.environ)
        env["MUJOCO_GL"] = backend
        if backend == "egl":
            env.setdefault("EGL_PLATFORM", "surfaceless")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import mujoco; "
                    "ctx = mujoco.GLContext(1, 1); "
                    "ctx.make_current(); "
                    "ctx.free()"
                ),
            ],
            env=env,
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            os.environ["MUJOCO_GL"] = backend
            if backend == "egl":
                os.environ.setdefault("EGL_PLATFORM", "surfaceless")
            return


_configure_linux_mujoco_gl()


def _configure_macos_dm_control_renderer() -> None:
    if platform.system() != "Darwin":
        return

    from dm_control import _render
    from dm_control._render import base as dm_control_render_base
    from dm_control._render import executor as dm_control_render_executor

    class _CglContext(dm_control_render_base.ContextBase):
        def __init__(self, max_width: int, max_height: int):
            super().__init__(
                max_width,
                max_height,
                dm_control_render_executor.PassthroughRenderExecutor,
            )

        def _platform_init(self, max_width: int, max_height: int) -> None:
            del max_width, max_height
            from mujoco.cgl import cgl

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
                attrib.CGLPFAMultisample,
                attrib.CGLPFASampleBuffers,
                1,
                attrib.CGLPFASample,
                4,
                attrib.CGLPFAAccelerated,
                0,
                0,
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

        def _platform_make_current(self) -> None:
            from mujoco.cgl import cgl

            cgl.CGLSetCurrentContext(self._context)
            if not self._locked:
                cgl.CGLLockContext(self._context)
                self._locked = True

        def _platform_free(self) -> None:
            from mujoco.cgl import cgl

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

    _render.Renderer = _CglContext
    _render.BACKEND = "cgl"
    _render.USING_GPU = True


_configure_macos_dm_control_renderer()


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
    require_pixel_match: bool = True


@dataclass(frozen=True)
class RenderSequence:
    """Reset frame plus a short stepped render rollout for one task."""

    reset_envpool: np.ndarray
    reset_official: np.ndarray
    envpool_frames: tuple[np.ndarray, ...]
    official_frames: tuple[np.ndarray, ...]


MYOSUITE_RENDER_COMPARE_CASES = (
    RenderCase("myoHandReorientID-v0", "HandReorientID"),
    RenderCase("myoLegWalk-v0", "LegWalk"),
    RenderCase("myoChallengeBimanual-v0", "ChallengeBimanual"),
    RenderCase("MyoHandAirplaneFly-v0", "TrackAirplaneFly"),
    RenderCase(
        "myoLegHillyTerrainWalk-v0",
        "HillyTerrainWalk",
        require_pixel_match=False,
    ),
)
MYOSUITE_RENDER_COMPARE_STEPS = 3


def official_render_thresholds(
    task_id: str,
) -> tuple[float, float] | None:
    """Return render oracle thresholds for tasks with stable pixel matching."""
    if task_id == "myoLegHillyTerrainWalk-v0":
        # Terrain heightfields are mutated at reset time through dm_control's GL
        # context, while MyoSuite's offscreen and physics render paths use
        # separate MuJoCo scene bootstraps. Keep terrain as a visual oracle in
        # docs and cover render semantics in tests instead of pretending it is a
        # stable pixel-exact gate.
        return None
    return (1.0, 0.06)


def _entry(task_id: str) -> dict[str, Any]:
    return MYOSUITE_DIRECT_ENTRY_BY_ID[task_id]


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
    if prefer_physics_render and physics is not None and hasattr(
        physics, "render"
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
    kwargs = dict(entry["kwargs"])
    if entry["suite"] == "myodm":
        with prepared_track_oracle_model_path() as model_path:
            yield {
                "object_name": kwargs["object_name"],
                "model_path": model_path,
                "reference": _oracle_reference(kwargs["reference"]),
            }
        return
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
    if entry["class_name"] in _REORIENT_CLASS_NAMES:
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
    else:
        raise ValueError(f"Unsupported MyoBase render sync task: {task_id}")
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
        goal_bid = sim.model.body_name2id("target")
        object_bid = sim.model.body_name2id("Object")
        start_geom = sim.model.body_geomadr[object_bid]
        geom_count = sim.model.body_geomnum[object_bid]
        sync["test_goal_body_pos"] = (
            sim.model.body_pos[goal_bid].copy().tolist()
        )
        sync["test_goal_body_quat"] = (
            sim.model.body_quat[goal_bid].copy().tolist()
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
    elif entry["class_name"] not in {
        "RunTrack",
        "SoccerEnvV0",
        "ChaseTagEnvV0",
        "TableTennisEnvV0",
    }:
        raise ValueError(
            f"Unsupported MyoChallenge render sync task: {task_id}"
        )
    return np.asarray(obs, dtype=np.float64), sync


def _sync_reset_myodm(env: Any) -> tuple[np.ndarray, dict[str, Any]]:
    obs, _ = env.reset()
    unwrapped = env.unwrapped
    return np.asarray(obs, dtype=np.float64), {
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


def capture_render_sequence(
    task_id: str,
    *,
    steps: int = MYOSUITE_RENDER_COMPARE_STEPS,
    seed: int = 7,
    render_width: int = 192,
    render_height: int = 192,
    camera_id: int = -1,
) -> RenderSequence:
    """Capture reset plus a short EnvPool-oracle render rollout."""
    prefer_physics_render = False
    with _make_oracle(
        task_id,
        seed=seed,
        render_width=render_width,
        render_height=render_height,
        camera_id=camera_id,
    ) as oracle:
        _, sync = _oracle_reset_sync(oracle, task_id)
        env = make_gymnasium(
            task_id,
            num_envs=1,
            seed=seed,
            render_mode="rgb_array",
            render_width=render_width,
            render_height=render_height,
            render_camera_id=camera_id,
            **sync,
        )
        try:
            env.reset()
            reset_envpool = _render_envpool_array(env)[0].copy()
            reset_official = _render_official_array(
                oracle,
                width=render_width,
                height=render_height,
                camera_id=camera_id,
                prefer_physics_render=prefer_physics_render,
            ).copy()
            actions = _seeded_actions(_action_shape(env), steps, seed + 97)
            envpool_frames: list[np.ndarray] = []
            official_frames: list[np.ndarray] = []
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
                if terminated0 or truncated0:
                    raise AssertionError(
                        f"{task_id} terminated before capturing step "
                        f"{step_index}"
                    )
            return RenderSequence(
                reset_envpool=reset_envpool,
                reset_official=reset_official,
                envpool_frames=tuple(envpool_frames),
                official_frames=tuple(official_frames),
            )
        finally:
            env.close()
