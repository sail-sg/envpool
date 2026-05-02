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
"""Tests for the batched MuJoCo render API."""

import ctypes
import os
import platform
import subprocess
import sys
from typing import Any, cast

import numpy as np
from absl.testing import absltest

from envpool.python.glfw_context import preload_windows_gl_dlls

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

import gymnasium as gym
import mujoco

import envpool.mujoco.gym.registration as reg
from envpool.registration import make_gym

_RENDER_STEPS = 3
_TASK_IDS = tuple(
    sorted(
        f"{task}-{version}"
        for task, versions, _ in reg.gym_mujoco_envs
        for version in versions
    )
)
_OFFICIAL_RENDER_TASK_IDS = tuple(
    task_id for task_id in _TASK_IDS if task_id.endswith("-v5")
)


def _configure_macos_official_renderer() -> None:
    if platform.system() != "Darwin":
        return

    from gymnasium.envs.mujoco import mujoco_rendering

    class _CglContext:
        def __init__(self, width: int, height: int):
            del width, height
            from mujoco.cgl import cgl

            attrib = cgl.CGLPixelFormatAttribute
            profile = cgl.CGLOpenGLProfile
            offline_attribs = (
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
                0,  # terminator
            )
            preferred_attribs = (
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
                0,  # terminator
            )
            self._pixel_format: Any = None
            if not self._choose_pixel_format(
                cgl, offline_attribs
            ) and not self._choose_pixel_format(cgl, preferred_attribs):
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

        def _choose_pixel_format(
            self, cgl: Any, attrib_values: tuple[int, ...]
        ) -> bool:
            attribs = (ctypes.c_int * len(attrib_values))(*attrib_values)
            pixel_format = cgl.CGLPixelFormatObj()
            num_pixel_formats = cgl.GLint()
            try:
                cgl.CGLChoosePixelFormat(
                    attribs,
                    ctypes.byref(pixel_format),
                    ctypes.byref(num_pixel_formats),
                )
            except cgl.CGLError:
                return False
            if not pixel_format or num_pixel_formats.value == 0:
                return False
            self._pixel_format = pixel_format
            return True

        def make_current(self) -> None:
            from mujoco.cgl import cgl

            cgl.CGLSetCurrentContext(self._context)
            # Mirror mujoco.cgl.GLContext so the official renderer uses the
            # same CGL lifecycle as EnvPool's native renderer.
            if not self._locked:
                cgl.CGLLockContext(self._context)
                self._locked = True

        def free(self) -> None:
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

        def __del__(self) -> None:
            self.free()

    def _import_cgl(width: int, height: int) -> Any:
        return _CglContext(width, height)

    mujoco_rendering._ALL_RENDERERS["cgl"] = _import_cgl
    os.environ["MUJOCO_GL"] = "cgl"


_configure_macos_official_renderer()


def _render_array(env: Any, env_ids: Any = None) -> np.ndarray:
    frame = env.render(env_ids=env_ids)
    assert frame is not None
    return cast(np.ndarray, frame)


def _render_official_array(env: gym.Env[Any, Any]) -> np.ndarray:
    base_env = cast(Any, env.unwrapped)
    viewer = base_env.mujoco_renderer.viewer
    if viewer is not None:
        viewer.make_context_current()
    return cast(np.ndarray, env.render())


def _official_viewer_debug(
    env: gym.Env[Any, Any],
    first_frame: np.ndarray,
    second_frame: np.ndarray,
) -> str:
    base_env = cast(Any, env.unwrapped)
    renderer = base_env.mujoco_renderer
    viewer = renderer.viewer
    if viewer is None:
        return "official viewer was not initialized"
    diff = np.abs(first_frame.astype(np.int16) - second_frame.astype(np.int16))
    return (
        "official render debug: "
        f"backend={getattr(viewer, 'backend', None)!r}, "
        f"camera_id={renderer.camera_id}, "
        f"default_cam_config={renderer.default_cam_config!r}, "
        f"cam_type={viewer.cam.type}, fixedcamid={viewer.cam.fixedcamid}, "
        f"azimuth={viewer.cam.azimuth:.3f}, "
        f"elevation={viewer.cam.elevation:.3f}, "
        f"distance={viewer.cam.distance:.3f}, "
        f"lookat={np.asarray(viewer.cam.lookat).tolist()}, "
        f"first_mean={float(first_frame.mean()):.3f}, "
        f"second_mean={float(second_frame.mean()):.3f}, "
        f"first_second_mean_delta={float(diff.mean()):.3f}, "
        f"first_second_max_delta={int(diff.max()) if diff.size else 0}"
    )


def _zero_action(space: Any, num_envs: int) -> np.ndarray:
    sample = np.asarray(space.sample())
    zero = np.zeros_like(sample)
    if sample.ndim == 0:
        return np.full((num_envs,), zero.item(), dtype=sample.dtype)
    return np.repeat(zero[np.newaxis, ...], num_envs, axis=0)


def _assert_frames_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    max_mean_abs_diff: float = 1.0,
    max_mismatch_ratio: float = 0.1,
) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"frame shapes differ: {actual.shape} != {expected.shape}"
        )
    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    if diff.size == 0:
        return
    mismatch_ratio = float(np.count_nonzero(diff)) / float(diff.size)
    mean_abs_diff = float(diff.mean())
    if mean_abs_diff > max_mean_abs_diff:
        raise AssertionError(
            "mean render delta "
            f"{mean_abs_diff:.3f} exceeded {max_mean_abs_diff:.3f}; "
            f"actual range=({int(actual.min())}, {int(actual.max())}) "
            f"mean={float(actual.mean()):.3f}, "
            f"expected range=({int(expected.min())}, {int(expected.max())}) "
            f"mean={float(expected.mean()):.3f}, "
            f"max_delta={int(diff.max())}"
        )
    if mismatch_ratio > max_mismatch_ratio:
        raise AssertionError(
            "render mismatch ratio "
            f"{mismatch_ratio:.4%} exceeded {max_mismatch_ratio:.4%}"
        )


def _reset_official_state(
    env: gym.Env[Any, Any], qpos: np.ndarray, qvel: np.ndarray
) -> None:
    base_env = cast(Any, env.unwrapped)
    mujoco.mj_resetData(base_env.model, base_env.data)
    base_env.set_state(qpos, qvel)


def _render_official_reset_frame(
    task_id: str,
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, str]:
    oracle = gym.make(
        task_id,
        render_mode="rgb_array",
        width=width,
        height=height,
    )
    try:
        oracle.reset(seed=0)
        _reset_official_state(oracle, qpos, qvel)
        frame = _render_official_array(oracle)
        frame_again = _render_official_array(oracle)
        return frame, _official_viewer_debug(oracle, frame, frame_again)
    finally:
        oracle.close()


class MujocoRenderTest(absltest.TestCase):
    """Render regression tests for Gym-style MuJoCo tasks."""

    def test_rgb_array_render_is_batch_consistent_and_state_invariant(
        self,
    ) -> None:
        """RGB renders should be batch-consistent and free of side effects."""
        env = make_gym(
            "Ant-v5",
            num_envs=2,
            render_mode="rgb_array",
            render_width=64,
            render_height=48,
        )
        try:
            env.reset()
            for step_idx in range(_RENDER_STEPS):
                frame0 = _render_array(env)
                frame1 = _render_array(env, env_ids=1)
                frames = _render_array(env, env_ids=[0, 1])
                frame0_again = _render_array(env)

                self.assertEqual(frame0.shape, (1, 48, 64, 3))
                self.assertEqual(frame1.shape, (1, 48, 64, 3))
                self.assertEqual(frame0.dtype, np.uint8)
                self.assertEqual(frames.shape, (2, 48, 64, 3))
                self.assertEqual(frames.dtype, np.uint8)
                _assert_frames_close(frame0[0], frames[0])
                _assert_frames_close(frame1[0], frames[1])
                _assert_frames_close(frame0, frame0_again)
                if step_idx + 1 < _RENDER_STEPS:
                    env.step(_zero_action(env.action_space, 2))
        finally:
            env.close()

    def test_render_succeeds_for_multiple_steps_for_all_tasks(self) -> None:
        """Every Gym-style MuJoCo task should render repeatedly."""
        for task_id in _TASK_IDS:
            with self.subTest(task_id=task_id):
                env = make_gym(
                    task_id,
                    num_envs=1,
                    seed=0,
                    render_mode="rgb_array",
                    render_width=96,
                    render_height=72,
                )
                try:
                    env.reset()
                    for step_idx in range(_RENDER_STEPS):
                        frame = _render_array(env)[0]
                        frame_again = _render_array(env)[0]
                        self.assertEqual(frame.shape, (72, 96, 3))
                        self.assertEqual(frame_again.shape, (72, 96, 3))
                        self.assertEqual(frame.dtype, np.uint8)
                        self.assertEqual(frame_again.dtype, np.uint8)
                        self.assertGreater(
                            int(frame.max()) - int(frame.min()), 0
                        )
                        self.assertGreater(
                            int(frame_again.max()) - int(frame_again.min()), 0
                        )
                        if step_idx + 1 < _RENDER_STEPS:
                            env.step(_zero_action(env.action_space, 1))
                finally:
                    env.close()

    def test_render_matches_official_reset_frame_for_all_available_tasks(
        self,
    ) -> None:
        """Rendered reset frames should match Gymnasium after state sync."""
        for task_id in _OFFICIAL_RENDER_TASK_IDS:
            with self.subTest(task_id=task_id):
                env = make_gym(
                    task_id,
                    num_envs=1,
                    seed=0,
                    render_mode="rgb_array",
                    render_width=96,
                    render_height=72,
                )
                try:
                    _, info = env.reset()
                    qpos = info["qpos0"][0]
                    qvel = info["qvel0"][0]
                    frame = _render_array(env)[0]
                    expected, official_debug = _render_official_reset_frame(
                        task_id,
                        qpos,
                        qvel,
                        width=96,
                        height=72,
                    )
                    try:
                        _assert_frames_close(
                            frame,
                            expected,
                            max_mean_abs_diff=8.0,
                            max_mismatch_ratio=0.12,
                        )
                    except AssertionError as exc:
                        raise AssertionError(
                            f"{exc}; {official_debug}"
                        ) from exc
                finally:
                    env.close()

    def test_human_render_uses_python_viewer(self) -> None:
        """Human mode should route rendered frames through the Python viewer."""
        env = make_gym(
            "Ant-v5",
            num_envs=1,
            render_mode="human",
            render_width=32,
            render_height=24,
        )
        shown: list[np.ndarray] = []
        cast(Any, env)._show_human_frame = lambda frame: shown.append(
            np.array(frame)
        )
        try:
            env.reset()
            result = env.render()

            self.assertIsNone(result)
            self.assertLen(shown, 1)
            self.assertEqual(shown[0].shape, (24, 32, 3))
            self.assertEqual(shown[0].dtype, np.uint8)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
