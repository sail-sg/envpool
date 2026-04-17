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
from absl import flags
from absl.testing import absltest

from envpool.mujoco.myosuite.render_utils import (
    MYOSUITE_RENDER_COMPARE_CASES,
    MYOSUITE_RENDER_COMPARE_STEPS,
    MYOSUITE_RENDER_RETRY_SEEDS,
    MYOSUITE_RENDER_VALIDATE_TASK_IDS,
    capture_render_sequence,
    official_render_thresholds,
)
from envpool.registration import make_gymnasium

_RENDER_WIDTH = 96
_RENDER_HEIGHT = 72

FLAGS = flags.FLAGS
flags.DEFINE_bool(
    "myosuite_render_run_surface",
    True,
    "Whether to run the full public MyoSuite render sweep.",
)
flags.DEFINE_integer(
    "myosuite_render_shard_index",
    0,
    "Zero-based shard index for the public MyoSuite render sweep.",
)
flags.DEFINE_integer(
    "myosuite_render_shard_count",
    1,
    "Total shard count for the public MyoSuite render sweep.",
)
flags.DEFINE_bool(
    "myosuite_render_include_doc_cases",
    True,
    "Whether to run the representative doc-case checks in this binary.",
)


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
    mismatch_ratio = float(np.count_nonzero(diff)) / float(diff.size)
    mean_abs_diff = float(diff.mean())
    if mean_abs_diff > max_mean_abs_diff:
        raise AssertionError(
            f"{label} mean render delta "
            f"{mean_abs_diff:.3f} exceeded {max_mean_abs_diff:.3f}"
        )
    if mismatch_ratio > max_mismatch_ratio:
        raise AssertionError(
            f"{label} render mismatch ratio "
            f"{mismatch_ratio:.4%} exceeded {max_mismatch_ratio:.4%}"
        )


def _selected_task_ids() -> tuple[str, ...]:
    shard_count = FLAGS.myosuite_render_shard_count
    shard_index = FLAGS.myosuite_render_shard_index
    if shard_count <= 0:
        raise ValueError("myosuite_render_shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            "myosuite_render_shard_index must satisfy "
            "0 <= index < shard_count"
        )
    return tuple(MYOSUITE_RENDER_VALIDATE_TASK_IDS[shard_index::shard_count])


class MyoSuiteRenderTest(absltest.TestCase):
    """Checks public render semantics and representative oracle frames."""

    def test_render_is_batch_consistent_and_state_invariant(self) -> None:
        """Repeated renders should be batch-consistent and side-effect free."""
        if not FLAGS.myosuite_render_include_doc_cases:
            self.skipTest("Representative render checks disabled for this shard.")
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

    def test_all_public_renders_match_official(self) -> None:
        """Every public MyoSuite task should match the oracle for 3 steps."""
        if not FLAGS.myosuite_render_run_surface:
            self.skipTest("Full-surface render sweep disabled for this target.")
        task_ids = _selected_task_ids()
        self.assertNotEmpty(task_ids)
        for task_id in task_ids:
            thresholds = official_render_thresholds(task_id)
            with self.subTest(task_id=task_id):
                sequence = capture_render_sequence(
                    task_id,
                    steps=MYOSUITE_RENDER_COMPARE_STEPS,
                    seed=7,
                    render_width=_RENDER_WIDTH,
                    render_height=_RENDER_HEIGHT,
                    action_mode="random",
                    retry_seeds=MYOSUITE_RENDER_RETRY_SEEDS,
                )
                max_mean_abs_diff, max_mismatch_ratio = thresholds
                _assert_frames_close(
                    sequence.reset_envpool,
                    sequence.reset_official,
                    label=f"{task_id} reset",
                    max_mean_abs_diff=max_mean_abs_diff,
                    max_mismatch_ratio=max_mismatch_ratio,
                )
                self.assertLen(
                    sequence.envpool_frames, MYOSUITE_RENDER_COMPARE_STEPS
                )
                for index, (envpool_frame, official_frame) in enumerate(
                    zip(
                        sequence.envpool_frames,
                        sequence.official_frames,
                        strict=True,
                    ),
                    start=1,
                ):
                    _assert_frames_close(
                        envpool_frame,
                        official_frame,
                        label=f"{task_id} step {index}",
                        max_mean_abs_diff=max_mean_abs_diff,
                        max_mismatch_ratio=max_mismatch_ratio,
                    )

    def test_representative_render_cases_stay_documented(self) -> None:
        """Representative doc cases remain part of the public render sweep."""
        if not FLAGS.myosuite_render_include_doc_cases:
            self.skipTest("Representative render checks disabled for this shard.")
        covered = set(MYOSUITE_RENDER_VALIDATE_TASK_IDS)
        for render_case in MYOSUITE_RENDER_COMPARE_CASES:
            with self.subTest(task_id=render_case.task_id):
                self.assertIn(render_case.task_id, covered)

    def test_render_retry_skips_early_terminal_seed(self) -> None:
        """Render capture should retry when a candidate seed ends too early."""
        if not FLAGS.myosuite_render_include_doc_cases:
            self.skipTest("Representative render checks disabled for this shard.")
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
