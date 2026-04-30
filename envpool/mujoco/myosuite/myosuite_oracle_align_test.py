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
"""Oracle coverage checks for native MyoSuite envs.

The pinned oracle is MyoSuite v2.11.6. Nine official Bimanual/Soccer IDs are
registered by upstream but fail to instantiate under the repository's numpy 2
test environment; those IDs remain registered in EnvPool and are excluded only
from oracle instantiation in this test.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
from absl.testing import absltest

import envpool.mujoco.myosuite.registration  # noqa: F401
from envpool.mujoco.myosuite.tasks import (
    MYOSUITE_ORACLE_NUMPY2_BROKEN_IDS,
    MYOSUITE_ORACLE_VERSION,
    MYOSUITE_TASKS,
    MyoSuiteTask,
)
from envpool.registration import make_gymnasium, make_spec

_ROLLOUT_STEPS = 128
_ORACLE_SPACE_BATCH_SIZE = 64
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
    "myoChallengeChaseTagP1-v0",
    "myoChallengeDieReorientP1-v0",
    "myoChallengeOslRunFixed-v0",
    "myoChallengeRelocateP1-v0",
    "myoChallengeTableTennisP0-v0",
})
_BITWISE_ROLLOUT_TASK_IDS = frozenset({
    "myoFingerReachFixed-v0",
    "myoFingerPoseFixed-v0",
})
_MUJOCO_BINARY_DRIFT_ROLLOUT_TASK_IDS = frozenset({
    # These match ctrl/act exactly; qpos/qvel drift starts inside mj_step when
    # comparing EnvPool's Bazel-linked MuJoCo 3.6.0 to the oracle subprocess'
    # pinned MuJoCo Python wheel. Keep the residual threshold scoped here.
    "myoHandKeyTurnFixed-v0",
    "myoHandObjHoldFixed-v0",
    "myoHandPenTwirlFixed-v0",
})
_EXPECTED_ORACLE_NUMPY2_BROKEN_IDS = frozenset({
    "myoChallengeBimanual-v0",
    "myoSarcChallengeBimanual-v0",
    "myoFatiChallengeBimanual-v0",
    "myoChallengeSoccerP1-v0",
    "myoChallengeSoccerP2-v0",
    "myoSarcChallengeSoccerP1-v0",
    "myoSarcChallengeSoccerP2-v0",
    "myoFatiChallengeSoccerP1-v0",
    "myoFatiChallengeSoccerP2-v0",
})


def _oracle_task_ids() -> tuple[str, ...]:
    return tuple(
        task["id"]
        for task in MYOSUITE_TASKS
        if task["id"] not in MYOSUITE_ORACLE_NUMPY2_BROKEN_IDS
    )


def _oracle_rollout_task_ids() -> tuple[str, ...]:
    return tuple(
        task_id
        for task_id in _oracle_task_ids()
        if task_id in _ROLLOUT_TASK_IDS
    )


def _task_metadata_by_id() -> dict[str, MyoSuiteTask]:
    return {task["id"]: task for task in MYOSUITE_TASKS}


def _oracle_probe_path() -> Path:
    runfiles = Path(os.environ["TEST_SRCDIR"])
    workspace = os.environ.get("TEST_WORKSPACE", "envpool")
    candidates = [
        runfiles / workspace / "envpool/mujoco/myosuite_oracle_probe",
        runfiles / workspace / "envpool/mujoco/myosuite_oracle_probe.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = list(runfiles.rglob("myosuite_oracle_probe"))
    if matches:
        return matches[0]
    raise RuntimeError(
        f"could not locate myosuite_oracle_probe under {runfiles}"
    )


def _run_oracle_probe(
    mode: str, task_ids: tuple[str, ...] = (), steps: int = _ROLLOUT_STEPS
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out:
        out_path = Path(out.name)
    cmd = [
        str(_oracle_probe_path()),
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
        return cast(dict[str, Any], json.loads(out_path.read_text()))
    finally:
        out_path.unlink(missing_ok=True)


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


class MyoSuiteOracleAlignTest(absltest.TestCase):
    """Validate native MyoSuite coverage against the pinned oracle surface."""

    def test_only_known_numpy2_oracle_failures_are_excluded(self) -> None:
        """The oracle exclusion set is limited to the nine upstream failures."""
        self.assertSetEqual(
            MYOSUITE_ORACLE_NUMPY2_BROKEN_IDS,
            _EXPECTED_ORACLE_NUMPY2_BROKEN_IDS,
        )
        self.assertLen(MYOSUITE_ORACLE_NUMPY2_BROKEN_IDS, 9)
        for task_id in MYOSUITE_ORACLE_NUMPY2_BROKEN_IDS:
            with self.subTest(task_id=task_id):
                self.assertEqual(make_spec(task_id).config.task_name, task_id)

    def test_pinned_official_registry_coverage(self) -> None:
        """Every pinned upstream registry ID must be represented locally."""
        report = _run_oracle_probe("space")
        self.assertEqual(report["version"], MYOSUITE_ORACLE_VERSION)
        official_ids = tuple(cast(list[str], report["ids"]))
        envpool_ids = tuple(task["id"] for task in MYOSUITE_TASKS)
        self.assertEqual(official_ids, envpool_ids)
        self.assertLen(official_ids, 398)

    def test_oracle_space_coverage_except_numpy2_broken_ids(self) -> None:
        """Native spaces must match every instantiable official oracle env."""
        oracle_tasks = _run_oracle_space_reports(_oracle_task_ids())
        self.assertLen(oracle_tasks, len(MYOSUITE_TASKS) - 9)
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

    def test_oracle_rollout_surface_except_numpy2_broken_ids(self) -> None:
        """Exercise nontrivial rollouts with oracle-generated actions."""
        report = _run_oracle_probe("trace", _oracle_rollout_task_ids())
        oracle_tasks = cast(dict[str, dict[str, Any]], report["tasks"])
        for task in MYOSUITE_TASKS:
            task_id = task["id"]
            if task_id in MYOSUITE_ORACLE_NUMPY2_BROKEN_IDS:
                continue
            if task_id not in _ROLLOUT_TASK_IDS:
                continue
            with self.subTest(task_id=task_id):
                envpool = make_gymnasium(task_id, num_envs=1, seed=5)
                try:
                    envpool_obs, _ = envpool.reset()
                    oracle_task = oracle_tasks[task_id]
                    self.assertLen(oracle_task["obs"], _ROLLOUT_STEPS + 1)
                    self.assertLen(oracle_task["actions"], _ROLLOUT_STEPS)
                    self.assertEqual(envpool_obs.shape, (1, task["obs_dim"]))
                    if task_id in _BITWISE_ROLLOUT_TASK_IDS:
                        np.testing.assert_array_equal(
                            envpool_obs[0].astype(np.float32),
                            np.asarray(oracle_task["obs"][0], dtype=np.float32),
                        )

                    for step_id, action in enumerate(oracle_task["actions"]):
                        action = np.asarray(action, dtype=np.float32)
                        envpool_step = envpool.step(action[None, :])
                        self.assertEqual(
                            envpool_step[0].shape, (1, task["obs_dim"])
                        )
                        self.assertEqual(envpool_step[1].shape, (1,))
                        self.assertEqual(envpool_step[2].shape, (1,))
                        self.assertEqual(envpool_step[3].shape, (1,))
                        if task_id in _BITWISE_ROLLOUT_TASK_IDS:
                            np.testing.assert_array_equal(
                                envpool_step[0][0].astype(np.float32),
                                np.asarray(
                                    oracle_task["obs"][step_id + 1],
                                    dtype=np.float32,
                                ),
                            )
                            self.assertEqual(
                                float(envpool_step[1][0]),
                                float(
                                    np.asarray(
                                        oracle_task["rewards"][step_id],
                                        dtype=np.float32,
                                    )
                                ),
                            )
                            self.assertEqual(
                                bool(envpool_step[2][0]),
                                bool(oracle_task["terminated"][step_id]),
                            )
                            self.assertEqual(
                                bool(envpool_step[3][0]),
                                bool(oracle_task["truncated"][step_id]),
                            )
                        elif task_id in _MUJOCO_BINARY_DRIFT_ROLLOUT_TASK_IDS:
                            np.testing.assert_allclose(
                                envpool_step[0][0].astype(np.float32),
                                np.asarray(
                                    oracle_task["obs"][step_id + 1],
                                    dtype=np.float32,
                                ),
                                atol=1e-5,
                                rtol=1e-4,
                            )
                            np.testing.assert_allclose(
                                np.asarray(
                                    envpool_step[1][0], dtype=np.float32
                                ),
                                np.asarray(
                                    oracle_task["rewards"][step_id],
                                    dtype=np.float32,
                                ),
                                atol=1e-5,
                                rtol=1e-4,
                            )
                            self.assertEqual(
                                bool(envpool_step[2][0]),
                                bool(oracle_task["terminated"][step_id]),
                            )
                            self.assertEqual(
                                bool(envpool_step[3][0]),
                                bool(oracle_task["truncated"][step_id]),
                            )
                        if bool(oracle_task["terminated"][step_id]) or bool(
                            oracle_task["truncated"][step_id]
                        ):
                            break
                finally:
                    envpool.close()


if __name__ == "__main__":
    absltest.main()
