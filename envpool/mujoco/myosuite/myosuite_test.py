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
"""Registry, smoke, and determinism tests for native MyoSuite envs."""

from __future__ import annotations

from typing import Any

import numpy as np
from absl.testing import absltest

import envpool.mujoco.myosuite.registration as myosuite_registration
from envpool.mujoco.myosuite.tasks import MYOSUITE_TASKS
from envpool.registration import list_all_envs, make_gymnasium, make_spec

_TASKS = tuple(MYOSUITE_TASKS)
_TASK_IDS = tuple(task["id"] for task in _TASKS)
_ORACLE_NUMPY2_BROKEN_TASKS = tuple(
    task for task in _TASKS if task["oracle_numpy2_broken"]
)
_DETERMINISM_STEPS = 8
_INFO_KEYS = {
    "task_id",
    "sparse",
    "solved",
    "oracle_numpy2_broken",
    "model_nq",
    "model_nv",
    "model_na",
}


class MyoSuiteTest(absltest.TestCase):
    """Validate registration, runtime surface, and determinism."""

    def _assert_info_equal(
        self, info0: dict[str, Any], info1: dict[str, Any]
    ) -> None:
        """Assert two vectorized EnvPool info dictionaries are equal."""
        self.assertEqual(info0.keys(), info1.keys())
        for key in info0:
            arr0 = np.asarray(info0[key])
            arr1 = np.asarray(info1[key])
            if arr0.dtype == object or arr1.dtype == object:
                self.assertEqual(arr0.shape, arr1.shape)
            else:
                np.testing.assert_array_equal(arr0, arr1, err_msg=f"info[{key}]")

    def test_generated_registry_matches_official_surface(self) -> None:
        """Generated metadata must cover all pinned official MyoSuite IDs."""
        self.assertLen(_TASKS, 398)
        self.assertEqual(
            tuple(myosuite_registration.myosuite_task_ids),
            _TASK_IDS,
        )
        registered = set(list_all_envs())
        for task in _TASKS:
            task_id = task["id"]
            alias = f"MyoSuite/{task_id}"
            with self.subTest(task_id=task_id):
                self.assertIn(task_id, registered)
                self.assertIn(alias, registered)
                spec = make_spec(task_id)
                alias_spec = make_spec(alias)
                self.assertEqual(
                    spec.observation_space.shape, (task["obs_dim"],)
                )
                self.assertEqual(spec.action_space.shape, (task["action_dim"],))
                self.assertEqual(
                    spec.config.max_episode_steps,
                    task["max_episode_steps"],
                )
                self.assertEqual(
                    alias_spec.observation_space.shape,
                    spec.observation_space.shape,
                )
                self.assertEqual(
                    alias_spec.action_space.shape,
                    spec.action_space.shape,
                )

    def test_reset_and_step_reference_surface(self) -> None:
        """Every registered task must reset and step with expected shapes."""
        for task in _TASKS:
            task_id = task["id"]
            with self.subTest(task_id=task_id):
                env = make_gymnasium(task_id, num_envs=2, seed=7)
                try:
                    obs, info = env.reset()
                    self.assertEqual(obs.shape, (2, task["obs_dim"]))
                    self.assertTrue(_INFO_KEYS.issubset(info.keys()))
                    action = np.zeros((2, task["action_dim"]), dtype=np.float32)
                    obs, rew, term, trunc, info = env.step(action)
                    self.assertEqual(obs.shape, (2, task["obs_dim"]))
                    self.assertEqual(rew.shape, (2,))
                    self.assertEqual(term.shape, (2,))
                    self.assertEqual(trunc.shape, (2,))
                    self.assertFalse(np.any(np.isnan(obs)))
                    self.assertFalse(np.any(np.isnan(rew)))
                    self.assertTrue(_INFO_KEYS.issubset(info.keys()))
                finally:
                    env.close()

    def test_oracle_skipped_tasks_keep_native_surface(self) -> None:
        """Oracle skips must not collapse the native environment surface."""
        self.assertLen(_ORACLE_NUMPY2_BROKEN_TASKS, 9)
        for task in _ORACLE_NUMPY2_BROKEN_TASKS:
            task_id = task["id"]
            with self.subTest(task_id=task_id):
                env = make_gymnasium(task_id, num_envs=1, seed=11)
                try:
                    obs, _ = env.reset()
                    self.assertFalse(np.all(obs == 0), task_id)
                    action = np.zeros((1, task["action_dim"]), dtype=np.float32)
                    obs, *_ = env.step(action)
                    self.assertFalse(np.all(obs == 0), task_id)
                finally:
                    env.close()

    def test_reference_surface_is_deterministic(self) -> None:
        """Same seed and action sequence must produce identical rollouts."""
        rng = np.random.default_rng(123)
        for task in _TASKS:
            task_id = task["id"]
            with self.subTest(task_id=task_id):
                actions = rng.uniform(
                    -1.0,
                    1.0,
                    size=(_DETERMINISM_STEPS, 2, task["action_dim"]),
                ).astype(np.float32)
                env0 = make_gymnasium(task_id, num_envs=2, seed=42)
                env1 = make_gymnasium(task_id, num_envs=2, seed=42)
                try:
                    obs0, info0 = env0.reset()
                    obs1, info1 = env1.reset()
                    np.testing.assert_array_equal(obs0, obs1)
                    self._assert_info_equal(info0, info1)
                    for action in actions:
                        step0 = env0.step(action)
                        step1 = env1.step(action)
                        for value0, value1 in zip(
                            step0[:4], step1[:4], strict=True
                        ):
                            np.testing.assert_array_equal(value0, value1)
                        self._assert_info_equal(step0[4], step1[4])
                finally:
                    env0.close()
                    env1.close()


if __name__ == "__main__":
    absltest.main()
