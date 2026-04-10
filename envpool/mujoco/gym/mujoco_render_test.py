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

import os
import platform
from typing import Any, cast

import gymnasium as gym
import mujoco
import numpy as np
from absl.testing import absltest

from envpool.python.glfw_context import preload_windows_gl_dlls

preload_windows_gl_dlls(strict=True)

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

    def _import_cgl(width: int, height: int) -> Any:
        from mujoco.cgl import GLContext

        return GLContext(width, height)

    mujoco_rendering._ALL_RENDERERS.setdefault("cgl", _import_cgl)
    os.environ.setdefault("MUJOCO_GL", "cgl")


_configure_macos_official_renderer()


def _render_array(env: Any, env_ids: Any = None) -> np.ndarray:
    frame = env.render(env_ids=env_ids)
    assert frame is not None
    return cast(np.ndarray, frame)


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
            f"{mean_abs_diff:.3f} exceeded {max_mean_abs_diff:.3f}"
        )
    if mismatch_ratio > max_mismatch_ratio:
        raise AssertionError(
            "render mismatch ratio "
            f"{mismatch_ratio:.4%} exceeded {max_mismatch_ratio:.4%}"
        )


def _reset_official_state(
    env: gym.Env[Any, Any], qpos: np.ndarray, qvel: np.ndarray
) -> None:
    base_env = env.unwrapped
    mujoco.mj_resetData(base_env.model, base_env.data)
    base_env.set_state(qpos, qvel)


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
                oracle = gym.make(
                    task_id,
                    render_mode="rgb_array",
                    width=96,
                    height=72,
                )
                try:
                    _, info = env.reset()
                    oracle.reset(seed=0)
                    _reset_official_state(
                        oracle,
                        info["qpos0"][0],
                        info["qvel0"][0],
                    )
                    frame = _render_array(env)[0]
                    expected = cast(np.ndarray, oracle.render())
                    _assert_frames_close(
                        frame,
                        expected,
                        max_mean_abs_diff=8.0,
                        max_mismatch_ratio=0.12,
                    )
                finally:
                    env.close()
                    oracle.close()

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
