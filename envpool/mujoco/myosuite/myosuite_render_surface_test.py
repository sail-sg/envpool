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
"""Full-surface MyoSuite render oracle runner.

Bazel sharding and absltest both consume TEST_TOTAL_SHARDS/TEST_SHARD_INDEX.
Keep the exhaustive render sweep out of absltest so every Bazel shard executes
the env-ID slice selected here instead of sharding unittest methods.
"""

from __future__ import annotations

import contextlib
import io
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from absl import app, flags

from envpool.mujoco.myosuite.render_utils import (
    MYOSUITE_RENDER_COMPARE_STEPS,
    MYOSUITE_RENDER_RETRY_SEEDS,
    MYOSUITE_RENDER_VALIDATE_TASK_IDS,
    capture_render_sequence,
    official_render_thresholds,
)

_RENDER_WIDTH = 96
_RENDER_HEIGHT = 72
_CAPTURED_OUTPUT_TAIL_CHARS = 4000

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "myosuite_render_shard_index",
    0,
    "Zero-based shard index used only when Bazel sharding is disabled.",
)
flags.DEFINE_integer(
    "myosuite_render_shard_count",
    1,
    "Shard count used only when Bazel sharding is disabled.",
)


def _assert_frames_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    label: str,
    max_mean_abs_diff: float,
    max_mismatch_ratio: float,
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
    mismatch_detail = ""
    if mismatch_count:
        coords = np.argwhere(diff)[:5]
        examples = []
        for y, x, channel in coords:
            examples.append(
                f"({int(y)},{int(x)},{int(channel)}): "
                f"{int(actual[y, x, channel])}!={int(expected[y, x, channel])}"
            )
        mismatch_detail = "; first mismatches " + ", ".join(examples)
    if mean_abs_diff > max_mean_abs_diff:
        raise AssertionError(
            f"{label} mean render delta "
            f"{mean_abs_diff:.6f} exceeded {max_mean_abs_diff:.6f}; "
            f"max delta {max_abs_diff}, mismatched values "
            f"{mismatch_count}/{diff.size} ({mismatch_ratio:.6%})"
            f"{mismatch_detail}"
        )
    if mismatch_ratio > max_mismatch_ratio:
        raise AssertionError(
            f"{label} render mismatch ratio "
            f"{mismatch_ratio:.6%} exceeded {max_mismatch_ratio:.6%}; "
            f"mean delta {mean_abs_diff:.6f}, max delta {max_abs_diff}, "
            f"mismatched values {mismatch_count}/{diff.size}"
            f"{mismatch_detail}"
        )


def _shard_params() -> tuple[int, int]:
    bazel_shard_count = int(os.environ.get("TEST_TOTAL_SHARDS", "0"))
    if bazel_shard_count > 1:
        return int(os.environ.get("TEST_SHARD_INDEX", "0")), bazel_shard_count
    return (
        int(FLAGS.myosuite_render_shard_index),
        int(FLAGS.myosuite_render_shard_count),
    )


def _selected_task_ids() -> tuple[str, ...]:
    shard_index, shard_count = _shard_params()
    if shard_count <= 0:
        raise ValueError("TEST_TOTAL_SHARDS must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            "TEST_SHARD_INDEX must satisfy 0 <= index < TEST_TOTAL_SHARDS"
        )
    # The public IDs are generated in sorted upstream order. Keep contiguous
    # model variants together so macOS AGX does not compile shaders for many
    # unrelated MyoDM object models in one process.
    total = len(MYOSUITE_RENDER_VALIDATE_TASK_IDS)
    start = total * shard_index // shard_count
    end = total * (shard_index + 1) // shard_count
    return tuple(MYOSUITE_RENDER_VALIDATE_TASK_IDS[start:end])


def _captured_output_tail(captured_output: str) -> str:
    if not captured_output:
        return ""
    if len(captured_output) > _CAPTURED_OUTPUT_TAIL_CHARS:
        captured_output = captured_output[-_CAPTURED_OUTPUT_TAIL_CHARS:]
    return f"\nCaptured oracle output tail:\n{captured_output}"


def _check_task(task_id: str) -> None:
    max_mean_abs_diff, max_mismatch_ratio = official_render_thresholds(task_id)
    sequence = capture_render_sequence(
        task_id,
        steps=MYOSUITE_RENDER_COMPARE_STEPS,
        seed=7,
        render_width=_RENDER_WIDTH,
        render_height=_RENDER_HEIGHT,
        action_mode="random",
        retry_seeds=MYOSUITE_RENDER_RETRY_SEEDS,
    )
    _assert_frames_close(
        sequence.reset_envpool,
        sequence.reset_official,
        label=f"{task_id} reset",
        max_mean_abs_diff=max_mean_abs_diff,
        max_mismatch_ratio=max_mismatch_ratio,
    )
    if len(sequence.envpool_frames) != MYOSUITE_RENDER_COMPARE_STEPS:
        raise AssertionError(
            f"{task_id} captured {len(sequence.envpool_frames)} EnvPool "
            f"frames, expected {MYOSUITE_RENDER_COMPARE_STEPS}"
        )
    if len(sequence.official_frames) != MYOSUITE_RENDER_COMPARE_STEPS:
        raise AssertionError(
            f"{task_id} captured {len(sequence.official_frames)} official "
            f"frames, expected {MYOSUITE_RENDER_COMPARE_STEPS}"
        )
    for index, (envpool_frame, official_frame) in enumerate(
        zip(sequence.envpool_frames, sequence.official_frames, strict=True),
        start=1,
    ):
        _assert_frames_close(
            envpool_frame,
            official_frame,
            label=f"{task_id} step {index}",
            max_mean_abs_diff=max_mean_abs_diff,
            max_mismatch_ratio=max_mismatch_ratio,
        )


def main(argv: Sequence[str]) -> None:
    """Run this Bazel shard's public MyoSuite render comparisons."""
    if len(argv) > 1:
        raise app.UsageError("unexpected positional arguments")
    shard_status_file = os.environ.get("TEST_SHARD_STATUS_FILE")
    if shard_status_file:
        Path(shard_status_file).touch()
    task_ids = _selected_task_ids()
    shard_index, shard_count = _shard_params()
    if not task_ids:
        raise AssertionError("render surface shard selected no MyoSuite tasks")
    print(
        "Checking MyoSuite render shard "
        f"{shard_index}/{shard_count}: {len(task_ids)} task IDs",
        flush=True,
    )
    failures: list[str] = []
    for task_id in task_ids:
        print(f"Checking {task_id}", flush=True)
        captured_output = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured_output):
                with contextlib.redirect_stderr(captured_output):
                    _check_task(task_id)
        except Exception as exc:
            output_tail = _captured_output_tail(captured_output.getvalue())
            failures.append(
                f"{task_id}: {type(exc).__name__}: {exc}{output_tail}"
            )
    if failures:
        joined = "\n\n".join(failures)
        raise AssertionError(
            f"{len(failures)} MyoSuite render comparison(s) failed:\n\n{joined}"
        )


if __name__ == "__main__":
    app.run(main)
