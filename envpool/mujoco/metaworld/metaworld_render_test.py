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
"""Render tests for native MetaWorld v3 tasks."""

from __future__ import annotations

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

import mujoco  # noqa: E402
from gymnasium.envs.mujoco import mujoco_rendering  # noqa: E402
from metaworld.env_dict import ALL_V3_ENVIRONMENTS  # noqa: E402

import envpool.mujoco.metaworld.registration as metaworld_registration  # noqa: E402
from envpool.registration import make_gymnasium  # noqa: E402

_RENDER_WIDTH = 64
_RENDER_HEIGHT = 48
# Official MetaWorld does not thread width/height through its constructors.
# Comparing at Gymnasium's default 480x480 source size avoids introducing
# extra offscreen-resize antialiasing differences before the pixel budget.
_OFFICIAL_RENDER_WIDTH = 480
_OFFICIAL_RENDER_HEIGHT = 480
_OFFICIAL_RENDER_MAX_MEAN_ABS_DIFF = 0.25
_OFFICIAL_RENDER_MAX_MISMATCH_RATIO = 0.005
_RENDER_STEPS = 3
_CAMERA_ID = 1  # MetaWorld's fixed "corner" camera.
_TASK_NAMES = tuple(metaworld_registration.metaworld_v3_envs)
_TASK_IDS = tuple(metaworld_registration.metaworld_v3_task_ids)


def _configure_macos_official_renderer() -> None:
    if platform.system() != "Darwin":
        return

    class _CglContext:
        def __init__(self, width: int, height: int) -> None:
            del width, height
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
            from mujoco.cgl import cgl

            cgl.CGLSetCurrentContext(self._context)
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


def _zero_action(space: Any, num_envs: int) -> np.ndarray:
    sample = np.asarray(space.sample())
    zero = np.zeros_like(sample)
    if sample.ndim == 0:
        return np.full((num_envs,), zero.item(), dtype=sample.dtype)
    return np.repeat(zero[np.newaxis, ...], num_envs, axis=0)


def _render_array(env: Any, env_ids: Any = None) -> np.ndarray:
    frame = env.render(env_ids=env_ids)
    assert frame is not None
    return cast(np.ndarray, frame)


def _make_oracle(task_name: str) -> Any:
    env = ALL_V3_ENVIRONMENTS[task_name](
        render_mode="rgb_array",
        camera_id=_CAMERA_ID,
    )
    env._set_task_called = True
    env._partially_observable = True
    env.mujoco_renderer.width = _OFFICIAL_RENDER_WIDTH
    env.mujoco_renderer.height = _OFFICIAL_RENDER_HEIGHT
    return env


def _sync_reset_state(oracle: Any, info: dict[str, Any]) -> None:
    rand_vec = np.asarray(info["rand_vec0"][0], dtype=np.float64)
    random_dim = int(oracle._random_reset_space.low.size)
    oracle._freeze_rand_vec = True
    oracle._last_rand_vec = rand_vec[:random_dim].copy()
    oracle.reset()

    qpos = np.asarray(info["qpos0"][0], dtype=np.float64)[
        : oracle.data.qpos.size
    ]
    qvel = np.asarray(info["qvel0"][0], dtype=np.float64)[
        : oracle.data.qvel.size
    ]
    oracle.set_state(qpos, qvel)
    oracle.data.mocap_pos[0] = np.asarray(
        info["mocap_pos0"][0], dtype=np.float64
    )
    oracle.data.mocap_quat[0] = np.asarray(
        info["mocap_quat0"][0], dtype=np.float64
    )
    oracle.data.qacc[:] = np.asarray(info["qacc0"][0], dtype=np.float64)[
        : oracle.data.qacc.size
    ]
    oracle.data.qacc_warmstart[:] = np.asarray(
        info["qacc_warmstart0"][0], dtype=np.float64
    )[: oracle.data.qacc_warmstart.size]
    mujoco.mj_forward(oracle.model, oracle.data)
    oracle.init_tcp = np.asarray(info["init_tcp0"][0], dtype=np.float64).copy()
    oracle.init_left_pad = np.asarray(
        info["init_left_pad0"][0], dtype=np.float64
    ).copy()
    oracle.init_right_pad = np.asarray(
        info["init_right_pad0"][0], dtype=np.float64
    ).copy()
    if hasattr(oracle, "_handle_init_pos"):
        oracle._handle_init_pos = oracle._get_pos_objects().copy()

    curr_obs = oracle._get_curr_obs_combined_no_goal()
    oracle._prev_obs = curr_obs.copy()
    obs = oracle._get_obs().astype(np.float64)
    oracle._last_stable_obs = obs.copy()


def _render_official_array(oracle: Any) -> np.ndarray:
    viewer = oracle.mujoco_renderer.viewer
    if viewer is not None:
        viewer.make_context_current()
    return cast(np.ndarray, oracle.render())


def _assert_frames_close_to_official(
    actual: np.ndarray, expected: np.ndarray, label: str
) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{label} frame shapes differ: {actual.shape} != {expected.shape}"
        )
    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    if diff.size == 0:
        return
    mismatch_ratio = float(np.count_nonzero(diff)) / float(diff.size)
    mean_abs_diff = float(diff.mean())
    if mean_abs_diff > _OFFICIAL_RENDER_MAX_MEAN_ABS_DIFF:
        raise AssertionError(
            f"{label} mean render delta {mean_abs_diff:.6f} exceeded "
            f"{_OFFICIAL_RENDER_MAX_MEAN_ABS_DIFF:.6f}; "
            f"max_abs_diff={int(diff.max())}, "
            f"mismatch_ratio={mismatch_ratio:.6%}, "
            f"actual_range=({int(actual.min())}, {int(actual.max())}), "
            f"expected_range=({int(expected.min())}, {int(expected.max())})"
        )
    if mismatch_ratio > _OFFICIAL_RENDER_MAX_MISMATCH_RATIO:
        raise AssertionError(
            f"{label} render mismatch ratio {mismatch_ratio:.6%} exceeded "
            f"{_OFFICIAL_RENDER_MAX_MISMATCH_RATIO:.6%}; "
            f"mean_abs_diff={mean_abs_diff:.6f}, "
            f"max_abs_diff={int(diff.max())}, "
            f"actual_range=({int(actual.min())}, {int(actual.max())}), "
            f"expected_range=({int(expected.min())}, {int(expected.max())})"
        )


class MetaWorldRenderTest(absltest.TestCase):
    """Render regression tests for native MetaWorld tasks."""

    def test_rgb_array_render_all_v3_tasks_reset_and_multistep(self) -> None:
        """Every v3 task should render nonblank reset and stepped frames."""
        for task_id in _TASK_IDS:
            with self.subTest(task_id=task_id):
                env = make_gymnasium(
                    task_id,
                    num_envs=1,
                    seed=0,
                    render_mode="rgb_array",
                    render_width=_RENDER_WIDTH,
                    render_height=_RENDER_HEIGHT,
                )
                try:
                    env.reset()
                    action = _zero_action(env.action_space, 1)
                    for step_idx in range(_RENDER_STEPS):
                        frame = _render_array(env)
                        self.assertEqual(
                            frame.shape,
                            (1, _RENDER_HEIGHT, _RENDER_WIDTH, 3),
                        )
                        self.assertEqual(frame.dtype, np.uint8)
                        self.assertGreater(
                            int(frame.max()) - int(frame.min()), 0
                        )
                        if step_idx + 1 < _RENDER_STEPS:
                            env.step(action)
                finally:
                    env.close()

    def test_rgb_array_render_is_batch_consistent_and_state_invariant(
        self,
    ) -> None:
        """Batch rendering should match env-id rendering without side effects."""
        env = make_gymnasium(
            "MetaWorld/Reach-v3",
            num_envs=2,
            seed=0,
            render_mode="rgb_array",
            render_width=_RENDER_WIDTH,
            render_height=_RENDER_HEIGHT,
        )
        try:
            env.reset()
            action = _zero_action(env.action_space, 2)
            for step_idx in range(_RENDER_STEPS):
                frame0 = _render_array(env)
                frame1 = _render_array(env, env_ids=1)
                frames = _render_array(env, env_ids=[0, 1])
                frame0_again = _render_array(env)

                self.assertEqual(
                    frame0.shape,
                    (1, _RENDER_HEIGHT, _RENDER_WIDTH, 3),
                )
                self.assertEqual(
                    frame1.shape,
                    (1, _RENDER_HEIGHT, _RENDER_WIDTH, 3),
                )
                self.assertEqual(
                    frames.shape,
                    (2, _RENDER_HEIGHT, _RENDER_WIDTH, 3),
                )
                self.assertEqual(frame0.dtype, np.uint8)
                self.assertEqual(frame1.dtype, np.uint8)
                self.assertEqual(frames.dtype, np.uint8)
                np.testing.assert_array_equal(frame0[0], frames[0])
                np.testing.assert_array_equal(frame1[0], frames[1])
                np.testing.assert_array_equal(frame0, frame0_again)
                if step_idx + 1 < _RENDER_STEPS:
                    env.step(action)
        finally:
            env.close()

    def test_rgb_array_render_matches_official_reset_frame(
        self,
    ) -> None:
        """EnvPool reset frames should closely match official MetaWorld."""
        for task_id, task_name in zip(_TASK_IDS, _TASK_NAMES, strict=True):
            with self.subTest(task_id=task_id):
                env = make_gymnasium(
                    task_id,
                    num_envs=1,
                    seed=7,
                    render_mode="rgb_array",
                    render_width=_OFFICIAL_RENDER_WIDTH,
                    render_height=_OFFICIAL_RENDER_HEIGHT,
                    render_camera_id=_CAMERA_ID,
                )
                oracle = _make_oracle(task_name)
                try:
                    _, info = env.reset()
                    _sync_reset_state(oracle, info)
                    _assert_frames_close_to_official(
                        _render_array(env)[0],
                        _render_official_array(oracle),
                        f"{task_id} reset",
                    )
                finally:
                    env.close()
                    oracle.close()


if __name__ == "__main__":
    absltest.main()
