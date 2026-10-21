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
"""Render smoke tests for native MyoSuite envs."""

from __future__ import annotations

import importlib
import os
import platform
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from absl.testing import absltest

from envpool.mujoco.myosuite.oracle import (
    run_oracle_probe,
    sampled_state_from_info,
)
from envpool.mujoco.myosuite.oracle import (
    sync_state_from_info as _sync_state_from_info,
)
from envpool.python.glfw_context import preload_windows_gl_dlls

if platform.system() == "Windows":
    preload_windows_gl_dlls(strict=True)

from envpool.mujoco.myosuite.tasks import (
    MYOSUITE_ORACLE_NUMPY2_BROKEN_IDS,
    MYOSUITE_TASKS,
)
from envpool.registration import make_gymnasium

importlib.import_module("envpool.mujoco.myosuite.registration")

_TASK_IDS = tuple(str(task["id"]) for task in MYOSUITE_TASKS)
_TASK_ID_SET = frozenset(_TASK_IDS)
# Render traces are the expensive part of this suite. The full 398-ID surface is
# covered by registry, space, reset/step, determinism, and generated docs. Keep
# CI render checks to a diagonal set that catches wrong-camera/model/scene
# regressions without retesting the full modifier cartesian product.
_ORACLE_RENDER_REPRESENTATIVE_TASK_IDS = frozenset({
    "MyoHandAirplaneFixed-v0",
    "MyoHandAirplaneFly-v0",
    "MyoHandCupPour-v0",
    "MyoHandHammerUse-v0",
    "MyoHandWatchRandom-v0",
    "motorFingerPoseFixed-v0",
    "myoArmReachFixed-v0",
    "myoElbowPose1D6MExoFixed-v0",
    "myoFingerReachFixed-v0",
    "myoFingerPoseFixed-v0",
    "myoHandReachFixed-v0",
    "myoHandPoseFixed-v0",
    "myoHandKeyTurnFixed-v0",
    "myoHandObjHoldFixed-v0",
    "myoHandPenTwirlFixed-v0",
    "myoHandReorient8-v0",
    "myoLegStandRandom-v0",
    "myoLegWalk-v0",
    "myoLegRoughTerrainWalk-v0",
    "myoFatiArmReachFixed-v0",
    "myoFatiHandReorient8-v0",
    "myoFatiLegWalk-v0",
    "myoSarcArmReachFixed-v0",
    "myoSarcHandReorient8-v0",
    "myoSarcLegWalk-v0",
    "myoChallengeBaodingP1-v1",
    "myoChallengeBimanual-v0",
    "myoChallengeChaseTagP1-v0",
    "myoChallengeDieReorientP1-v0",
    "myoChallengeOslRunFixed-v0",
    "myoChallengeRelocateP1-v0",
    "myoChallengeSoccerP1-v0",
    "myoChallengeTableTennisP0-v0",
    "myoFatiChallengeBimanual-v0",
    "myoSarcChallengeSoccerP2-v0",
})
_UNKNOWN_ORACLE_RENDER_REPRESENTATIVE_TASK_IDS = (
    _ORACLE_RENDER_REPRESENTATIVE_TASK_IDS - _TASK_ID_SET
)
if _UNKNOWN_ORACLE_RENDER_REPRESENTATIVE_TASK_IDS:
    raise ValueError(
        "unknown MyoSuite oracle render representatives: "
        f"{sorted(_UNKNOWN_ORACLE_RENDER_REPRESENTATIVE_TASK_IDS)}"
    )


def _render_task_allowlist_from_env() -> tuple[str, ...] | None:
    raw = os.environ.get("MYOSUITE_RENDER_TASK_IDS")
    if raw is None:
        return None
    task_ids = tuple(
        dict.fromkeys(
            task_id for task_id in raw.replace(",", " ").split() if task_id
        )
    )
    if not task_ids:
        raise ValueError("MYOSUITE_RENDER_TASK_IDS is set but empty")
    unknown = sorted(set(task_ids) - _TASK_ID_SET)
    if unknown:
        raise ValueError(f"unknown MYOSUITE_RENDER_TASK_IDS: {unknown}")
    return task_ids


_RENDER_TASK_ALLOWLIST = _render_task_allowlist_from_env()


def _filter_render_task_ids(task_ids: tuple[str, ...]) -> tuple[str, ...]:
    if _RENDER_TASK_ALLOWLIST is None:
        return task_ids
    return tuple(
        task_id for task_id in task_ids if task_id in _RENDER_TASK_ALLOWLIST
    )


def _native_render_task_ids() -> tuple[str, ...]:
    if _RENDER_TASK_ALLOWLIST is not None:
        return _filter_render_task_ids(_TASK_IDS)
    return tuple(
        task_id
        for task_id in _TASK_IDS
        if task_id in _ORACLE_RENDER_REPRESENTATIVE_TASK_IDS
    )


def _oracle_trace_task_ids() -> tuple[str, ...]:
    if _RENDER_TASK_ALLOWLIST is None:
        return tuple(
            task_id
            for task_id in _TASK_IDS
            if task_id in _ORACLE_RENDER_REPRESENTATIVE_TASK_IDS
            and task_id not in MYOSUITE_ORACLE_NUMPY2_BROKEN_IDS
        )
    return tuple(
        task_id
        for task_id in _filter_render_task_ids(_TASK_IDS)
        if task_id not in MYOSUITE_ORACLE_NUMPY2_BROKEN_IDS
    )


_NATIVE_RENDER_TASK_IDS = _native_render_task_ids()
_WIDTH = 64
_HEIGHT = 48
_ORACLE_RENDER_BATCH_SIZE = 8
# Catch wrong camera/model/scene regressions without chasing backend pixel noise.
_RENDER_BLOCK_SIZE = 4
_MAX_RENDER_MEAN_ABS_DIFF = 24.0
_MAX_RENDER_BLOCK_MEAN_ABS_DIFF = 24.0
_MAX_RENDER_LARGE_MISMATCH_RATIO = 0.50
_LARGE_RENDER_DELTA = 32


def _render_shard_task_ids(task_ids: tuple[str, ...]) -> tuple[str, ...]:
    total_shards = int(
        os.environ.get(
            "MYOSUITE_RENDER_TOTAL_SHARDS",
            os.environ.get("TEST_TOTAL_SHARDS", "1"),
        )
    )
    shard_index = int(
        os.environ.get(
            "MYOSUITE_RENDER_SHARD_INDEX",
            os.environ.get("TEST_SHARD_INDEX", "0"),
        )
    )
    shard_status_file = os.environ.get("TEST_SHARD_STATUS_FILE")
    if shard_status_file:
        Path(shard_status_file).touch()
    if total_shards <= 1:
        return task_ids
    if shard_index < 0 or shard_index >= total_shards:
        raise ValueError(f"invalid Bazel shard {shard_index} of {total_shards}")
    return tuple(
        task_id
        for index, task_id in enumerate(task_ids)
        if index % total_shards == shard_index
    )


def _task_batches(
    task_ids: tuple[str, ...],
    batch_size: int,
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        task_ids[start : start + batch_size]
        for start in range(0, len(task_ids), batch_size)
    )


_SHARDED_ORACLE_TRACE_TASK_IDS = _render_shard_task_ids(
    _oracle_trace_task_ids()
)
_SHARDED_NATIVE_RENDER_TASK_IDS = _render_shard_task_ids(
    _NATIVE_RENDER_TASK_IDS
)


def _oracle_trace(
    task_ids: tuple[str, ...],
    trace_plan: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    args = [
        "--mode",
        "trace",
        "--render",
        "--render_width",
        str(_WIDTH),
        "--render_height",
        str(_HEIGHT),
        "--action_mode",
        "midpoint",
        "--steps",
        "3",
        "--seed",
        "3",
    ]
    for task_id in task_ids:
        args.extend(["--task_id", task_id])
    return run_oracle_probe(args, trace_plan=trace_plan)["tasks"]


def _render(env: Any) -> np.ndarray:
    frame = env.render()
    if frame is None:
        raise AssertionError("MyoSuite render returned None")
    return frame


def _block_mean(frame: np.ndarray) -> np.ndarray:
    height, width, channels = frame.shape
    block = _RENDER_BLOCK_SIZE
    if height % block or width % block:
        raise AssertionError(
            f"render size {frame.shape} is not divisible by block {block}"
        )
    return frame.reshape(
        height // block,
        block,
        width // block,
        block,
        channels,
    ).mean(axis=(1, 3))


def _assert_render_aligned(
    test: absltest.TestCase,
    frame: np.ndarray,
    oracle_frame: np.ndarray,
    *,
    task_id: str,
    step_id: int,
) -> None:
    test.assertEqual(frame.shape, oracle_frame.shape)
    test.assertEqual(frame.dtype, np.uint8)
    test.assertEqual(oracle_frame.dtype, np.uint8)
    diff = np.abs(frame.astype(np.int16) - oracle_frame.astype(np.int16))
    max_abs = int(diff.max())
    mean_abs = float(np.mean(diff))
    block_diff = np.abs(_block_mean(frame) - _block_mean(oracle_frame))
    block_mean_abs = float(np.mean(block_diff))
    large_mismatch_ratio = float(
        np.mean(np.max(diff, axis=-1) > _LARGE_RENDER_DELTA)
    )
    if (
        mean_abs > _MAX_RENDER_MEAN_ABS_DIFF
        or block_mean_abs > _MAX_RENDER_BLOCK_MEAN_ABS_DIFF
        or large_mismatch_ratio > _MAX_RENDER_LARGE_MISMATCH_RATIO
    ):
        test.fail(
            f"{task_id} render step {step_id} drifted: "
            f"max_abs={max_abs}, mean_abs={mean_abs:.6f}, "
            f"block_mean_abs={block_mean_abs:.6f}, "
            f"large_mismatch_ratio={large_mismatch_ratio:.6f}"
        )


def _midpoint_action(env: Any) -> np.ndarray:
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    return ((low + high) * 0.5).astype(np.float32)


def _envpool_trace_record(
    task_id: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    env = make_gymnasium(
        task_id,
        num_envs=1,
        seed=3,
        render_mode="rgb_array",
        render_width=_WIDTH,
        render_height=_HEIGHT,
    )
    try:
        _, info = env.reset()
        frames = [_render(env)[0]]
        actions: list[list[float]] = []
        reset_before_step: list[bool] = []
        sync_states = [_sync_state_from_info(info)]
        sampled_states = []
        action = _midpoint_action(env)
        for _ in range(3):
            actions.append(action.tolist())
            *_, info = env.step(action[None, :])
            frames.append(_render(env)[0])
            step_info = info
            sampled_states.append(sampled_state_from_info(info))
            elapsed_step = int(np.asarray(step_info["elapsed_step"]).ravel()[0])
            reset_before_step.append(elapsed_step == 0)
            sync_states.append(
                _sync_state_from_info(step_info) if elapsed_step == 0 else {}
            )
        plan = {
            "actions": actions,
            "reset_before_step": reset_before_step,
            "sync_states": sync_states,
            "sampled_states": sampled_states,
        }
        return frames, plan
    finally:
        env.close()


class MyoSuiteRenderTest(absltest.TestCase):
    """Validate native MyoSuite RGB rendering after reset and steps."""

    def test_reset_and_first_three_step_render(self) -> None:
        """Representative tasks render through reset and first three steps."""
        for task_id in _SHARDED_NATIVE_RENDER_TASK_IDS:
            with self.subTest(task_id=task_id):
                env = make_gymnasium(
                    task_id,
                    num_envs=1,
                    seed=3,
                    render_mode="rgb_array",
                    render_width=_WIDTH,
                    render_height=_HEIGHT,
                )
                try:
                    env.reset()
                    frames = [_render(env)]
                    action = np.zeros(
                        (1, *env.action_space.shape), dtype=np.float32
                    )
                    for _ in range(3):
                        env.step(action)
                        frames.append(_render(env))
                    for frame in frames:
                        self.assertEqual(frame.shape, (1, _HEIGHT, _WIDTH, 3))
                        self.assertEqual(frame.dtype, np.uint8)
                        self.assertGreater(int(frame.max()), int(frame.min()))
                finally:
                    env.close()

    def test_official_trace_native_render_alignment(self) -> None:
        """Official render matches EnvPool reset and first 3 API frames."""
        for batch in _task_batches(
            _SHARDED_ORACLE_TRACE_TASK_IDS, _ORACLE_RENDER_BATCH_SIZE
        ):
            envpool_frames: dict[str, list[np.ndarray]] = {}
            trace_plan: dict[str, dict[str, Any]] = {}
            for task_id in batch:
                frames, plan = _envpool_trace_record(task_id)
                envpool_frames[task_id] = frames
                trace_plan[task_id] = plan
            oracle_tasks = _oracle_trace(batch, trace_plan)
            self.assertSetEqual(set(oracle_tasks), set(batch))
            for task_id in batch:
                with self.subTest(task_id=task_id):
                    oracle = oracle_tasks[task_id]
                    frames = envpool_frames[task_id]
                    oracle_frames = [
                        np.asarray(frame, dtype=np.uint8)
                        for frame in oracle["frames"]
                    ]
                    self.assertLen(frames, 4)
                    self.assertLen(oracle_frames, 4)
                    for step_id, (frame, oracle_frame) in enumerate(
                        zip(frames, oracle_frames, strict=True)
                    ):
                        _assert_render_aligned(
                            self,
                            frame,
                            oracle_frame,
                            task_id=task_id,
                            step_id=step_id,
                        )
                        self.assertGreater(int(frame.max()), int(frame.min()))


if __name__ == "__main__":
    unittest.main()
