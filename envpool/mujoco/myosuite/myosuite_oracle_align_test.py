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
"""Oracle coverage checks for native MyoSuite envs."""

from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import numpy as np
from absl import logging
from absl.testing import absltest

from envpool.mujoco.myosuite.tasks import (
    MYOSUITE_ORACLE_NUMPY2_BROKEN_IDS,
    MYOSUITE_ORACLE_VERSION,
    MYOSUITE_TASKS,
    MyoSuiteTask,
)
from envpool.registration import make_gymnasium, make_spec

importlib.import_module("envpool.mujoco.myosuite.registration")

_ROLLOUT_STEPS = 128
_ORACLE_SPACE_BATCH_SIZE = 64
_ROLLOUT_BATCH_SIZE = 4
# Keep the expensive 128-step oracle traces to a diagonal sample across
# orthogonal task modifiers. Full registry/space/render coverage still checks
# every official ID; this set keeps muscle-condition/player-side combinations
# from growing as a cartesian product.
_ROLLOUT_TASK_IDS = frozenset({
    "MyoHandAirplaneFixed-v0",
    "MyoHandAirplaneFly-v0",
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
_BITWISE_ROLLOUT_TASK_IDS = frozenset({
    "myoFingerReachFixed-v0",
    "myoFingerPoseFixed-v0",
})
_LINUX_AARCH64_FINGER_ROLLOUT_RTOL = 1e-5
_LINUX_AARCH64_FINGER_ROLLOUT_ATOL = 1e-7
_EXPECTED_ORACLE_NUMPY2_BROKEN_IDS: frozenset[str] = frozenset()
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


def _assert_bitwise_rollout_obs(
    actual: np.ndarray,
    desired: np.ndarray,
    *,
    label: str,
) -> None:
    if sys.platform.startswith("linux") and platform.machine().lower() in {
        "aarch64",
        "arm64",
    }:
        # Linux aarch64 accumulates small float32 differences in these long
        # MuJoCo finger traces after tens of steps. Keep the residual scoped to
        # that platform and far below a semantically meaningful trajectory drift.
        try:
            np.testing.assert_allclose(
                actual,
                desired,
                rtol=_LINUX_AARCH64_FINGER_ROLLOUT_RTOL,
                atol=_LINUX_AARCH64_FINGER_ROLLOUT_ATOL,
            )
        except AssertionError as exc:
            raise AssertionError(f"{label}\n{exc}") from exc
        return
    np.testing.assert_array_equal(actual, desired, err_msg=label)


def _oracle_task_ids() -> tuple[str, ...]:
    return tuple(task["id"] for task in MYOSUITE_TASKS)


def _shard_task_ids(task_ids: tuple[str, ...]) -> tuple[str, ...]:
    total_shards = int(
        os.environ.get(
            "MYOSUITE_ORACLE_TOTAL_SHARDS",
            os.environ.get("TEST_TOTAL_SHARDS", "1"),
        )
    )
    shard_index = int(
        os.environ.get(
            "MYOSUITE_ORACLE_SHARD_INDEX",
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


def _oracle_rollout_task_ids() -> tuple[str, ...]:
    return _shard_task_ids(
        tuple(
            task_id
            for task_id in _oracle_task_ids()
            if task_id in _ROLLOUT_TASK_IDS
        )
    )


def _task_batches(
    task_ids: tuple[str, ...],
    batch_size: int,
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        task_ids[start : start + batch_size]
        for start in range(0, len(task_ids), batch_size)
    )


def _task_metadata_by_id() -> dict[str, MyoSuiteTask]:
    return {task["id"]: task for task in MYOSUITE_TASKS}


def _oracle_probe_path() -> Path:
    runfiles = Path(os.environ["TEST_SRCDIR"])
    workspace = os.environ.get("TEST_WORKSPACE", "envpool")
    launcher_names: tuple[str, ...] = (
        "myosuite_oracle_probe",
        "myosuite_oracle_probe.exe",
    )
    logical_suffixes = (
        tuple(f"envpool/mujoco/{launcher}" for launcher in launcher_names)
        + launcher_names
    )
    manifest = os.environ.get("RUNFILES_MANIFEST_FILE")
    if manifest:
        with Path(manifest).open(encoding="utf-8") as f:
            for line in f:
                logical, _, physical = line.rstrip("\n").partition(" ")
                logical = logical.replace("\\", "/")
                if any(logical.endswith(suffix) for suffix in logical_suffixes):
                    candidate = Path(physical or logical)
                    if candidate.is_file():
                        return candidate
    candidates = [
        runfiles / workspace / "envpool/mujoco" / launcher
        for launcher in launcher_names
    ]
    if sys.platform == "win32":
        candidates.extend(
            runfiles.parent / launcher for launcher in launcher_names
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for launcher in launcher_names:
        for match in runfiles.rglob(launcher):
            if match.is_file():
                return match
    raise RuntimeError(
        f"could not locate myosuite_oracle_probe under {runfiles}"
    )


def _oracle_probe_cmd() -> list[str]:
    path = _oracle_probe_path()
    if sys.platform == "win32" and path.suffix.lower() != ".exe":
        return [sys.executable, str(path)]
    return [str(path)]


def _run_oracle_probe(
    mode: str,
    task_ids: tuple[str, ...] = (),
    steps: int = _ROLLOUT_STEPS,
    sync_states: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out:
        out_path = Path(out.name)
    sync_path: Path | None = None
    cmd = _oracle_probe_cmd() + [
        "--mode",
        mode,
        "--out",
        str(out_path),
        "--steps",
        str(steps),
        "--seed",
        "5",
    ]
    for task_id in task_ids:
        cmd.extend(["--task_id", task_id])
    if sync_states is not None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as sync:
            sync_path = Path(sync.name)
        sync_path.write_text(json.dumps(sync_states, sort_keys=True))
        cmd.extend(["--sync_state", str(sync_path)])
    env = os.environ.copy()
    env["ROBOHIVE_VERBOSITY"] = "SILENT"
    try:
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                env=env,
                text=True,
            )
        except OSError as exc:
            raise RuntimeError(
                f"MyoSuite oracle probe failed to start\ncmd: {' '.join(cmd)}"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                "MyoSuite oracle probe failed\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        return cast(dict[str, Any], json.loads(out_path.read_text()))
    finally:
        out_path.unlink(missing_ok=True)
        if sync_path is not None:
            sync_path.unlink(missing_ok=True)


def _run_oracle_space_reports(
    task_ids: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for start in range(0, len(task_ids), _ORACLE_SPACE_BATCH_SIZE):
        batch = task_ids[start : start + _ORACLE_SPACE_BATCH_SIZE]
        report = _run_oracle_probe("space", batch)
        if report["version"] != MYOSUITE_ORACLE_VERSION:
            raise AssertionError(report["version"])
        tasks.update(cast(dict[str, dict[str, Any]], report["tasks"]))
    return tasks


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


class MyoSuiteOracleAlignTest(absltest.TestCase):
    """Validate native MyoSuite coverage against the pinned oracle surface."""

    def test_no_numpy2_oracle_failures_are_excluded(self) -> None:
        """Every pinned upstream ID is instantiable by the oracle."""
        self.assertSetEqual(
            MYOSUITE_ORACLE_NUMPY2_BROKEN_IDS,
            _EXPECTED_ORACLE_NUMPY2_BROKEN_IDS,
        )
        self.assertEmpty(MYOSUITE_ORACLE_NUMPY2_BROKEN_IDS)

    def test_pinned_official_registry_coverage(self) -> None:
        """Every pinned upstream registry ID must be represented locally."""
        report = _run_oracle_probe("space")
        self.assertEqual(report["version"], MYOSUITE_ORACLE_VERSION)
        official_ids = tuple(cast(list[str], report["ids"]))
        envpool_ids = tuple(task["id"] for task in MYOSUITE_TASKS)
        self.assertEqual(official_ids, envpool_ids)
        self.assertLen(official_ids, 398)

    def test_oracle_space_coverage(self) -> None:
        """Native spaces must match every official oracle env."""
        task_ids = _shard_task_ids(_oracle_task_ids())
        oracle_tasks = _run_oracle_space_reports(task_ids)
        self.assertLen(oracle_tasks, len(task_ids))
        task_metadata = _task_metadata_by_id()
        for task_id, oracle_task in oracle_tasks.items():
            with self.subTest(task_id=task_id):
                task = task_metadata[task_id]
                envpool_spec = make_spec(task_id)
                self.assertEqual(
                    tuple(oracle_task["observation_shape"]),
                    (task["obs_dim"],),
                )
                self.assertEqual(
                    tuple(oracle_task["action_shape"]),
                    (task["action_dim"],),
                )
                self.assertEqual(
                    oracle_task["max_episode_steps"],
                    task["max_episode_steps"],
                )
                self.assertEqual(
                    envpool_spec.observation_space.shape,
                    (task["obs_dim"],),
                )
                self.assertEqual(
                    envpool_spec.action_space.shape,
                    (task["action_dim"],),
                )
                self.assertEqual(
                    envpool_spec.config.max_episode_steps,
                    task["max_episode_steps"],
                )

    def test_oracle_rollout_surface(self) -> None:
        """Exercise nontrivial rollouts with oracle-generated actions."""
        rollout_task_ids = _oracle_rollout_task_ids()
        task_metadata = _task_metadata_by_id()
        for batch in _task_batches(rollout_task_ids, _ROLLOUT_BATCH_SIZE):
            envpools: dict[str, Any] = {}
            envpool_reset_obs: dict[str, np.ndarray] = {}
            sync_states: dict[str, dict[str, Any]] = {}
            try:
                for task_id in batch:
                    envpool = make_gymnasium(task_id, num_envs=1, seed=5)
                    envpool_obs, info = envpool.reset()
                    envpools[task_id] = envpool
                    envpool_reset_obs[task_id] = envpool_obs
                    sync_states[task_id] = _sync_state_from_info(info)

                report = _run_oracle_probe(
                    "trace", batch, sync_states=sync_states
                )
                oracle_tasks = cast(dict[str, dict[str, Any]], report["tasks"])
                self.assertSetEqual(set(oracle_tasks), set(batch))

                for task_id in batch:
                    task = task_metadata[task_id]
                    envpool = envpools[task_id]
                    envpool_obs = envpool_reset_obs[task_id]
                    oracle_task = oracle_tasks[task_id]
                    with self.subTest(task_id=task_id):
                        self.assertLen(oracle_task["obs"], _ROLLOUT_STEPS + 1)
                        self.assertLen(oracle_task["actions"], _ROLLOUT_STEPS)
                        self.assertEqual(
                            envpool_obs.shape, (1, task["obs_dim"])
                        )
                        if task_id in _BITWISE_ROLLOUT_TASK_IDS:
                            _assert_bitwise_rollout_obs(
                                envpool_obs[0].astype(np.float32),
                                np.asarray(
                                    oracle_task["obs"][0], dtype=np.float32
                                ),
                                label=f"{task_id} reset obs",
                            )

                        for step_id, action in enumerate(
                            oracle_task["actions"]
                        ):
                            action = np.asarray(action, dtype=np.float32)
                            envpool_step = envpool.step(action[None, :])
                            self.assertEqual(
                                envpool_step[0].shape, (1, task["obs_dim"])
                            )
                            self.assertEqual(envpool_step[1].shape, (1,))
                            self.assertEqual(envpool_step[2].shape, (1,))
                            self.assertEqual(envpool_step[3].shape, (1,))
                            if task_id in _BITWISE_ROLLOUT_TASK_IDS:
                                oracle_obs = np.asarray(
                                    oracle_task["obs"][step_id + 1],
                                    dtype=np.float32,
                                )
                                oracle_reward = np.asarray(
                                    oracle_task["rewards"][step_id],
                                    dtype=np.float32,
                                )
                                _assert_bitwise_rollout_obs(
                                    envpool_step[0][0].astype(np.float32),
                                    oracle_obs,
                                    label=f"{task_id} step {step_id} obs",
                                )
                                self.assertEqual(
                                    float(envpool_step[1][0]),
                                    float(oracle_reward),
                                    msg=f"{task_id} step {step_id} reward",
                                )
                                self.assertEqual(
                                    bool(envpool_step[2][0]),
                                    bool(oracle_task["terminated"][step_id]),
                                    msg=f"{task_id} step {step_id} terminated",
                                )
                                self.assertEqual(
                                    bool(envpool_step[3][0]),
                                    bool(oracle_task["truncated"][step_id]),
                                    msg=f"{task_id} step {step_id} truncated",
                                )
                            if bool(oracle_task["terminated"][step_id]) or bool(
                                oracle_task["truncated"][step_id]
                            ):
                                break
            finally:
                for envpool in envpools.values():
                    try:
                        envpool.close()
                    except Exception as exc:
                        logging.warning(
                            "ignored MyoSuite env close failure: %s", exc
                        )


if __name__ == "__main__":
    unittest.main()
