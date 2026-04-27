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
"""Render tests for representative MyoSuite tasks."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from absl.testing import absltest

from envpool.mujoco.myosuite.render_utils import (
    MYOSUITE_RENDER_COMPARE_CASES,
    MYOSUITE_RENDER_COMPARE_STEPS,
    MYOSUITE_RENDER_RETRY_SEEDS,
    MYOSUITE_RENDER_VALIDATE_TASK_IDS,
    capture_render_sequence,
)
from envpool.registration import make_gymnasium

_RENDER_WIDTH = 96
_RENDER_HEIGHT = 72


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
    label: str = "frame",
    max_mean_abs_diff: float = 1.0,
    max_mismatch_ratio: float = 0.1,
) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{label} shapes differ: {actual.shape} != {expected.shape}"
        )
    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    if diff.size == 0:
        return
    mismatch_count = int(np.count_nonzero(diff))
    mismatch_ratio = float(mismatch_count) / float(diff.size)
    mean_abs_diff = float(diff.mean())
    max_abs_diff = int(diff.max())
    if mean_abs_diff > max_mean_abs_diff:
        raise AssertionError(
            f"{label} mean render delta "
            f"{mean_abs_diff:.6f} exceeded {max_mean_abs_diff:.6f}; "
            f"max delta {max_abs_diff}, mismatched values "
            f"{mismatch_count}/{diff.size} ({mismatch_ratio:.6%})"
        )
    if mismatch_ratio > max_mismatch_ratio:
        raise AssertionError(
            f"{label} render mismatch ratio "
            f"{mismatch_ratio:.6%} exceeded {max_mismatch_ratio:.6%}; "
            f"mean delta {mean_abs_diff:.6f}, max delta {max_abs_diff}, "
            f"mismatched values {mismatch_count}/{diff.size}"
        )


class MyoSuiteRenderTest(absltest.TestCase):
    """Checks public render semantics and representative oracle frames."""

    def test_render_is_batch_consistent_and_state_invariant(self) -> None:
        """Repeated renders should be batch-consistent and side-effect free."""
        env = make_gymnasium(
            "myoHandReorientID-v0",
            num_envs=2,
            seed=0,
            render_mode="rgb_array",
            render_width=_RENDER_WIDTH,
            render_height=_RENDER_HEIGHT,
        )
        try:
            env.reset()
            for step in range(MYOSUITE_RENDER_COMPARE_STEPS):
                frame0 = _render_array(env)
                frame1 = _render_array(env, env_ids=1)
                frames = _render_array(env, env_ids=[0, 1])
                frame0_again = _render_array(env)

                self.assertEqual(
                    frame0.shape, (1, _RENDER_HEIGHT, _RENDER_WIDTH, 3)
                )
                self.assertEqual(
                    frame1.shape, (1, _RENDER_HEIGHT, _RENDER_WIDTH, 3)
                )
                self.assertEqual(
                    frames.shape, (2, _RENDER_HEIGHT, _RENDER_WIDTH, 3)
                )
                self.assertEqual(frame0.dtype, np.uint8)
                _assert_frames_close(frame0[0], frames[0])
                _assert_frames_close(frame1[0], frames[1])
                _assert_frames_close(frame0, frame0_again)
                self.assertGreater(int(frame0.max()) - int(frame0.min()), 0)

                if step + 1 < MYOSUITE_RENDER_COMPARE_STEPS:
                    env.step(_zero_action(env.action_space, 2))
        finally:
            env.close()

    def test_representative_render_cases_stay_documented(self) -> None:
        """Representative doc cases remain part of the public render sweep."""
        covered = set(MYOSUITE_RENDER_VALIDATE_TASK_IDS)
        for render_case in MYOSUITE_RENDER_COMPARE_CASES:
            with self.subTest(task_id=render_case.task_id):
                self.assertIn(render_case.task_id, covered)

    def test_render_retry_skips_early_terminal_seed(self) -> None:
        """Render capture should retry when a candidate seed ends too early."""
        sequence = capture_render_sequence(
            "MyoHandAlarmclockFixed-v0",
            steps=MYOSUITE_RENDER_COMPARE_STEPS,
            seed=7,
            render_width=_RENDER_WIDTH,
            render_height=_RENDER_HEIGHT,
            action_mode="random",
            retry_seeds=MYOSUITE_RENDER_RETRY_SEEDS,
        )
        self.assertLen(sequence.envpool_frames, MYOSUITE_RENDER_COMPARE_STEPS)
        self.assertLen(sequence.official_frames, MYOSUITE_RENDER_COMPARE_STEPS)


if __name__ == "__main__":
    absltest.main()
