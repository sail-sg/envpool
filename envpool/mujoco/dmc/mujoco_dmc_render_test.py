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
"""Tests for the dm_control render path."""

from typing import Any, cast

import numpy as np
from absl.testing import absltest

from envpool.python.glfw_context import preload_windows_gl_dlls

preload_windows_gl_dlls(strict=True)

import envpool.mujoco.dmc.registration as reg
from envpool.registration import make_gym

_RENDER_STEPS = 3


def _task_map() -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for domain, task, _ in reg.dmc_mujoco_envs:
        domain_name = "".join(g[:1].upper() + g[1:] for g in domain.split("_"))
        task_name = "".join(g[:1].upper() + g[1:] for g in task.split("_"))
        result[f"{domain_name}{task_name}-v1"] = (domain, task)
    return result


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


class MujocoDmcRenderTest(absltest.TestCase):
    """Render regression tests for dm_control-backed MuJoCo tasks."""

    def test_rgb_array_render_is_batch_consistent(self) -> None:
        """Batch render output should match per-env renders for WalkerWalk."""
        env = make_gym(
            "WalkerWalk-v1",
            num_envs=2,
            seed=0,
            render_mode="rgb_array",
            render_width=96,
            render_height=72,
        )
        try:
            env.reset()
            for step_idx in range(_RENDER_STEPS):
                frame0 = _render_array(env)
                frame1 = _render_array(env, env_ids=1)
                frames = _render_array(env, env_ids=[0, 1])
                frame0_again = _render_array(env)
                self.assertEqual(frame0.shape, (1, 72, 96, 3))
                self.assertEqual(frame1.shape, (1, 72, 96, 3))
                self.assertEqual(frames.shape, (2, 72, 96, 3))
                self.assertEqual(frames.dtype, np.uint8)
                _assert_frames_close(frame0[0], frames[0])
                _assert_frames_close(frame1[0], frames[1])
                _assert_frames_close(frame0, frame0_again)
                if step_idx + 1 < _RENDER_STEPS:
                    env.step(_zero_action(env.action_space, 2))
        finally:
            env.close()

    def test_render_succeeds_for_all_tasks(self) -> None:
        """Every dm_control-backed task should render through the render path."""
        for task_id in sorted(_task_map()):
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


if __name__ == "__main__":
    absltest.main()
