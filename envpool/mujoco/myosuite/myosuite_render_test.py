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
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from absl.testing import absltest

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
_ORACLE_TRACE_TASK_IDS = tuple(
    task_id
    for task_id in _TASK_IDS
    if task_id not in MYOSUITE_ORACLE_NUMPY2_BROKEN_IDS
)
_NATIVE_ONLY_RENDER_TASK_IDS = tuple(
    task_id
    for task_id in _TASK_IDS
    if task_id in MYOSUITE_ORACLE_NUMPY2_BROKEN_IDS
)
_WIDTH = 64
_HEIGHT = 48
_ORACLE_RENDER_BATCH_SIZE = 32
_SYNC_STATE_KEYS = (
    "qpos0",
    "qvel0",
    "act0",
    "qacc0",
    "qacc_warmstart0",
    "ctrl",
    "site_pos",
    "site_quat",
    "site_size",
    "site_rgba",
    "body_pos",
    "body_quat",
    "body_mass",
    "geom_pos",
    "geom_quat",
    "geom_size",
    "geom_rgba",
    "geom_friction",
    "geom_aabb",
    "geom_rbound",
    "geom_contype",
    "geom_conaffinity",
    "geom_type",
    "geom_condim",
    "hfield_data",
    "mocap_pos",
    "mocap_quat",
    "fatigue_ma",
    "fatigue_mr",
    "fatigue_mf",
    "fatigue_tl",
)
_SYNC_STATE_SIZES = {
    "qpos0": "nq",
    "qvel0": "nv",
    "act0": "na",
    "qacc0": "nv",
    "qacc_warmstart0": "nv",
    "ctrl": "nu",
    "site_pos": "nsite3",
    "site_quat": "nsite4",
    "site_size": "nsite3",
    "site_rgba": "nsite4",
    "body_pos": "nbody3",
    "body_quat": "nbody4",
    "body_mass": "nbody",
    "geom_pos": "ngeom3",
    "geom_quat": "ngeom4",
    "geom_size": "ngeom3",
    "geom_rgba": "ngeom4",
    "geom_friction": "ngeom3",
    "geom_aabb": "ngeom6",
    "geom_rbound": "ngeom",
    "geom_contype": "ngeom",
    "geom_conaffinity": "ngeom",
    "geom_type": "ngeom",
    "geom_condim": "ngeom",
    "hfield_data": "nhfielddata",
    "mocap_pos": "nmocap3",
    "mocap_quat": "nmocap4",
    "fatigue_ma": "nu",
    "fatigue_mr": "nu",
    "fatigue_mf": "nu",
    "fatigue_tl": "nu",
}


def _shard_task_ids(task_ids: tuple[str, ...]) -> tuple[str, ...]:
    total_shards = int(os.environ.get("TEST_TOTAL_SHARDS", "1"))
    shard_index = int(os.environ.get("TEST_SHARD_INDEX", "0"))
    status_file = os.environ.get("TEST_SHARD_STATUS_FILE")
    if status_file:
        Path(status_file).touch(exist_ok=True)
    if total_shards <= 1:
        return task_ids
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


_SHARDED_ORACLE_TRACE_TASK_IDS = _shard_task_ids(_ORACLE_TRACE_TASK_IDS)
_SHARDED_NATIVE_ONLY_RENDER_TASK_IDS = _shard_task_ids(
    _NATIVE_ONLY_RENDER_TASK_IDS
)


def _oracle_probe_path() -> Path:
    runfiles = Path(os.environ["TEST_SRCDIR"])
    workspace = os.environ.get("TEST_WORKSPACE", "envpool")
    launcher_names = (
        ("myosuite_oracle_probe.exe", "myosuite_oracle_probe")
        if sys.platform == "win32"
        else ("myosuite_oracle_probe", "myosuite_oracle_probe.exe")
    )
    candidates = [
        runfiles / workspace / "envpool/mujoco" / launcher
        for launcher in launcher_names
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for launcher in launcher_names:
        matches = list(runfiles.rglob(launcher))
        if matches:
            return matches[0]
    raise RuntimeError(
        f"could not locate myosuite_oracle_probe under {runfiles}"
    )


def _oracle_trace(
    task_ids: tuple[str, ...],
    trace_plan: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out:
        out_path = Path(out.name)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as plan:
        plan_path = Path(plan.name)
    plan_path.write_text(json.dumps(trace_plan, sort_keys=True))
    cmd = [
        str(_oracle_probe_path()),
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
        "--out",
        str(out_path),
        "--trace_plan",
        str(plan_path),
    ]
    for task_id in task_ids:
        cmd.extend(["--task_id", task_id])
    env = os.environ.copy()
    env["ROBOHIVE_VERBOSITY"] = "SILENT"
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            env=env,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "MyoSuite oracle probe failed\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        return json.loads(out_path.read_text())["tasks"]
    finally:
        out_path.unlink(missing_ok=True)
        plan_path.unlink(missing_ok=True)


def _sync_state_from_info(info: dict[str, Any]) -> dict[str, Any]:
    dims = {
        "nq": int(np.asarray(info["model_nq"]).ravel()[0]),
        "nv": int(np.asarray(info["model_nv"]).ravel()[0]),
        "na": int(np.asarray(info["model_na"]).ravel()[0]),
        "nu": int(np.asarray(info["model_nu"]).ravel()[0]),
        "nsite": int(np.asarray(info["model_nsite"]).ravel()[0]),
        "nbody": int(np.asarray(info["model_nbody"]).ravel()[0]),
        "ngeom": int(np.asarray(info["model_ngeom"]).ravel()[0]),
        "nhfielddata": int(np.asarray(info["model_nhfielddata"]).ravel()[0]),
        "nmocap": int(np.asarray(info["model_nmocap"]).ravel()[0]),
    }
    dims.update({
        "nsite3": dims["nsite"] * 3,
        "nsite4": dims["nsite"] * 4,
        "nbody3": dims["nbody"] * 3,
        "nbody4": dims["nbody"] * 4,
        "ngeom3": dims["ngeom"] * 3,
        "ngeom4": dims["ngeom"] * 4,
        "ngeom6": dims["ngeom"] * 6,
        "nmocap3": dims["nmocap"] * 3,
        "nmocap4": dims["nmocap"] * 4,
    })
    sync_state = {}
    for key in _SYNC_STATE_KEYS:
        if key not in info:
            continue
        size = dims[_SYNC_STATE_SIZES[key]]
        sync_state[key] = (
            np.asarray(info[key][0], dtype=np.float64).ravel()[:size].tolist()
        )
    return sync_state


def _render(env: Any) -> np.ndarray:
    frame = env.render()
    if frame is None:
        raise AssertionError("MyoSuite render returned None")
    return frame


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
        action = _midpoint_action(env)
        for _ in range(3):
            actions.append(action.tolist())
            *_, info = env.step(action[None, :])
            frames.append(_render(env)[0])
            step_info = info
            elapsed_step = int(np.asarray(step_info["elapsed_step"]).ravel()[0])
            reset_before_step.append(elapsed_step == 0)
            sync_states.append(_sync_state_from_info(step_info))
        plan = {
            "actions": actions,
            "reset_before_step": reset_before_step,
            "sync_states": sync_states,
        }
        return frames, plan
    finally:
        env.close()


class MyoSuiteRenderTest(absltest.TestCase):
    """Validate native MyoSuite RGB rendering after reset and steps."""

    def test_reset_and_first_three_step_render(self) -> None:
        """Render oracle-skip tasks through reset and the first three steps."""
        for task_id in _SHARDED_NATIVE_ONLY_RENDER_TASK_IDS:
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

    def test_official_trace_native_render_bitwise(self) -> None:
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
                        np.testing.assert_array_equal(
                            frame,
                            oracle_frame,
                            err_msg=f"{task_id} render step {step_id}",
                        )
                        self.assertGreater(int(frame.max()), int(frame.min()))


if __name__ == "__main__":
    absltest.main()
