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
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import numpy as np
from absl import logging
from absl.testing import absltest

from envpool.mujoco.myosuite.oracle import (
    sync_state_from_info as _sync_state_from_info,
)
from envpool.mujoco.myosuite.tasks import (
    MYOSUITE_ORACLE_VERSION,
    MYOSUITE_TASKS,
    MyoSuiteTask,
)
from envpool.registration import make_gymnasium, make_spec

importlib.import_module("envpool.mujoco.myosuite.registration")

_ROLLOUT_STEPS = 128
_ORACLE_SPACE_BATCH_SIZE = 64
_ROLLOUT_BATCH_SIZE = 4


def _field_budget(task_id: str, field: str) -> tuple[float, float | None]:
    """Peak and per-component time-mean budgets for sensitive contact traces."""
    # Complete macOS/Linux rollouts isolate these residuals to ordinary control
    # roundoff or MjSpec recompilation. Equal-control diagnostic replays remove
    # the muscle/contact residuals; permanent tests still drive external actions.
    # Keep all other fields tight and constrain persistent drift as well as peaks.
    budgets: dict[str, tuple[float, float]] = {}
    if task_id.startswith("MyoHand"):
        budgets = {"qv": (2e-4, 2e-5), "hand_qvel_err": (2e-4, 2e-5)}
        if task_id == "MyoHandFlashlight1On-v0":
            budgets.update({
                "qp": (3e-3, 2.5e-4),
                "hand_qpos_err": (3e-3, 2.5e-4),
                "qv": (0.1, 4e-3),
                "hand_qvel_err": (0.1, 4e-3),
                "obj_com_err": (6e-5, 4e-6),
                "reward": (6e-5, 6e-6),
            })
        elif task_id == "MyoHandWaterbottleLift-v0":
            budgets.update({
                "qp": (1.5e-4, 3e-5),
                "hand_qpos_err": (1.5e-4, 3e-5),
                "qv": (0.012, 1e-3),
                "hand_qvel_err": (0.012, 1e-3),
            })
    elif "HandKeyTurn" in task_id:
        budgets = {
            "hand_qpos": (2e-4, 4e-5),
            "hand_qvel": (2e-4, 4e-5),
            "key_qpos": (2e-4, 4e-5),
            "key_qvel": (2e-4, 4e-5),
            "reward": (2e-4, 4e-5),
        }
    elif task_id.startswith("myoSarcHandPose"):
        budgets = {
            "qpos": (1e-4, 2e-5),
            "qvel": (1e-4, 2e-5),
            "pose_err": (1e-4, 2e-5),
        }
        if task_id == "myoSarcHandPose5Fixed-v0":
            budgets.update({
                "qpos": (6e-3, 2.5e-3),
                "pose_err": (6e-3, 2.5e-3),
                "qvel": (1e-3, 1e-4),
                "reward": (2e-3, 5e-4),
            })
    elif task_id.startswith("myoFatiFingerReach"):
        # Stiff muscle dynamics: observed tip error <= 3.4 mm; joint mean
        # <= 0.0076 rad. Upstream's qvel observation is velocity multiplied by dt.
        budgets = {
            "qpos": (0.06, 0.01),
            "qvel": (0.3, 0.025),
            "tip_pos": (4e-3, 5e-4),
            "reach_err": (4e-3, 5e-4),
            "reward": (3.5e-3, 5e-4),
        }
    elif "ChallengeBimanual" in task_id:
        # Object contact gives a short angular-velocity spike (1.79, mean 0.0067)
        # while hand/prosthesis state stays tight and contact categories agree.
        budgets = {
            "object_qpos": (6e-3, 1e-3),
            "object_qvel": (2.0, 0.01),
            "reward": (2e-3, 5e-5),
        }
    elif "ChallengeTableTennis" in task_id:
        # Recompiled-model roundoff: paddle error <= 4 mm / 0.014 rad, with
        # transient velocity differences <= 0.21 m/s in the measured rollouts.
        budgets = {
            "body_qpos": (2e-4, 6e-5),
            "body_qvel": (0.02, 3e-4),
            "paddle_pos": (5e-3, 2e-3),
            "reach_err": (5e-3, 2e-3),
            "paddle_vel": (0.25, 0.015),
            "paddle_ori": (0.02, 5e-3),
            "reward": (2e-3, 4e-4),
        }
        if task_id == "myoFatiChallengeTableTennisP0-v0":
            # This stress case amplifies ~1e-14 model recompilation residuals.
            # Matching only those residuals and control roundoff in a diagnostic
            # removes all physics error. Keep the independent oracle here.
            budgets.update({
                "body_qpos": (0.01, 3e-3),
                "body_qvel": (0.75, 0.025),
                "paddle_pos": (0.08, 0.03),
                "reach_err": (0.08, 0.03),
                "paddle_vel": (1.0, 0.25),
                "paddle_ori": (0.06, 0.03),
                "reward": (0.015, 5e-3),
            })
    return budgets.get(field, (2e-5, None))


def _assert_rollout_values(
    actual: np.ndarray,
    desired: np.ndarray,
    *,
    label: str,
    budget: tuple[float, float | None] = (2e-5, None),
) -> None:
    peak, mean = budget
    np.testing.assert_allclose(
        actual, desired, rtol=1e-5, atol=peak, equal_nan=False, err_msg=label
    )
    if mean is not None:
        error = np.abs(actual - desired).mean(axis=0)
        np.testing.assert_array_less(error, mean, err_msg=f"{label} mean")


def _assert_observation_trace(
    task_id: str,
    actual: np.ndarray,
    desired: np.ndarray,
    fields: list[tuple[str, int, bool]],
) -> None:
    offset = 0
    for name, size, discrete in fields:
        selection = slice(offset, offset + size)
        label = f"{task_id} observation {name}"
        if discrete or name.startswith("touching_"):
            np.testing.assert_array_equal(
                actual[:, selection], desired[:, selection], err_msg=label
            )
        else:
            _assert_rollout_values(
                actual[:, selection],
                desired[:, selection],
                label=label,
                budget=_field_budget(task_id, name),
            )
        offset += size
    if offset != actual.shape[1]:
        raise AssertionError(f"{task_id}: incomplete observation layout")


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


class MyoSuiteOracleAlignTest(absltest.TestCase):
    """Validate native MyoSuite coverage against the pinned oracle surface."""

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
                self.assertTrue(
                    np.can_cast(
                        np.dtype(oracle_task["observation_dtype"]),
                        envpool_spec.observation_space.dtype,
                        casting="safe",
                    )
                )
                self.assertEqual(
                    envpool_spec.action_space.dtype,
                    np.dtype(oracle_task["action_dtype"]),
                )
                self.assertEqual(
                    envpool_spec.config.max_episode_steps,
                    task["max_episode_steps"],
                )

    def test_reference_interpolation(self) -> None:
        """The upstream interpolation patch preserves a linear trajectory."""
        if not _shard_task_ids(("reference",)):
            return
        report = _run_oracle_probe("reference_interpolation")
        np.testing.assert_array_equal(
            report["robot"], [[0, 2], [2, 4], [4, 6], [6, 8], [8, 10]]
        )
        np.testing.assert_array_equal(
            report["robot_vel"], [[1, 3], [3, 5], [5, 7], [7, 9], [9, 11]]
        )
        np.testing.assert_array_equal(
            report["object"],
            [[i, i + 2, i + 4, 1, 0, 0, 0] for i in (0, 2, 4, 6, 8)],
        )

    def test_challenge_regressions(self) -> None:
        """The patched oracle moves collisions and ends completed rallies."""
        if not _shard_task_ids(("challenge_regressions",)):
            return
        report = _run_oracle_probe("challenge_regressions")
        np.testing.assert_allclose(
            report["physical"], report["commanded"], rtol=0, atol=1e-14
        )
        positions = np.asarray(report["physical"])
        # The default policy mixture intentionally includes stationary keepers.
        self.assertGreater(
            np.linalg.norm(positions[:, -1] - positions[:, 0], axis=-1).max(),
            1e-4,
        )
        self.assertEqual(
            report["rallies"],
            [[False, False], [True, True]]
            + [[False, True]] * 4
            + [[True, True]],
        )

    def test_oracle_rollout_surface(self) -> None:
        """Compare every registered task through its first complete episode."""
        rollout_task_ids = _shard_task_ids(_oracle_task_ids())
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
                    "trace",
                    batch,
                    steps=max(
                        task_metadata[task_id]["max_episode_steps"]
                        for task_id in batch
                    ),
                    sync_states=sync_states,
                )
                oracle_tasks = cast(dict[str, dict[str, Any]], report["tasks"])
                self.assertSetEqual(set(oracle_tasks), set(batch))

                for task_id in batch:
                    task = task_metadata[task_id]
                    envpool = envpools[task_id]
                    envpool_obs = envpool_reset_obs[task_id]
                    oracle_task = oracle_tasks[task_id]
                    with self.subTest(task_id=task_id):
                        self.assertLen(
                            oracle_task["obs"], len(oracle_task["actions"]) + 1
                        )
                        self.assertTrue(
                            oracle_task["terminated"][-1]
                            or oracle_task["truncated"][-1]
                        )
                        self.assertEqual(
                            envpool_obs.shape, (1, task["obs_dim"])
                        )
                        _assert_rollout_values(
                            envpool_obs[0],
                            np.asarray(oracle_task["obs"][0]),
                            label=f"{task_id} reset obs",
                        )

                        self.assertEqual(
                            envpool_obs.dtype,
                            envpool.observation_space.dtype,
                        )
                        observations = []
                        rewards = []
                        sparse_rewards = []
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
                            self.assertEqual(
                                envpool_step[0].dtype, envpool_obs.dtype
                            )
                            observations.append(envpool_step[0][0].copy())
                            rewards.append(float(envpool_step[1][0]))
                            sparse_rewards.append(
                                float(envpool_step[4]["sparse"][0])
                            )
                            oracle_ctrl = oracle_task["states"][step_id]["ctrl"]
                            np.testing.assert_allclose(
                                envpool_step[4]["ctrl"][0, : len(oracle_ctrl)],
                                oracle_ctrl,
                                rtol=1e-10,
                                atol=1e-10,
                                equal_nan=False,
                                err_msg=f"{task_id} step {step_id} control",
                            )
                            oracle_info = oracle_task["infos"][step_id]
                            self.assertEqual(
                                bool(envpool_step[4]["solved"][0]),
                                bool(np.asarray(oracle_info["solved"]).item()),
                                msg=f"{task_id} step {step_id} solved",
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
                        _assert_observation_trace(
                            task_id,
                            np.asarray(observations),
                            np.asarray(oracle_task["obs"][1:]),
                            oracle_task["obs_fields"],
                        )
                        for name, actual, desired in (
                            ("reward", rewards, oracle_task["rewards"]),
                            (
                                "sparse reward",
                                sparse_rewards,
                                [
                                    np.asarray(info["rwd_sparse"]).item()
                                    for info in oracle_task["infos"]
                                ],
                            ),
                        ):
                            _assert_rollout_values(
                                np.asarray(actual),
                                np.asarray(desired),
                                label=f"{task_id} {name}",
                                budget=_field_budget(task_id, "reward"),
                            )
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
