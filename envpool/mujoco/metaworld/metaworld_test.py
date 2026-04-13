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
"""Smoke and determinism tests for native MetaWorld v3 tasks."""

from __future__ import annotations

from typing import Any

import numpy as np
from absl.testing import absltest

import envpool.mujoco.metaworld.registration as metaworld_registration
from envpool.registration import list_all_envs, make_gymnasium, make_spec

_TASK_IDS = tuple(
    f"Meta-World/{task_name}"
    for task_name in metaworld_registration.metaworld_v3_envs
)
_INFO_KEYS = {
    "success",
    "near_object",
    "grasp_success",
    "grasp_reward",
    "in_place_reward",
    "obj_to_target",
    "unscaled_reward",
    "task_id",
}


class MetaWorldTest(absltest.TestCase):
    """Smoke and determinism tests for native MetaWorld tasks."""

    def _assert_info_equal(
        self, info0: dict[str, Any], info1: dict[str, Any]
    ) -> None:
        self.assertEqual(info0.keys(), info1.keys())
        for key in info0:
            self._assert_value_equal(info0[key], info1[key], f"info[{key}]")

    def _assert_value_equal(self, value0: Any, value1: Any, label: str) -> None:
        if isinstance(value0, dict):
            self.assertIsInstance(value1, dict)
            self.assertEqual(value0.keys(), value1.keys(), label)
            for key in value0:
                self._assert_value_equal(
                    value0[key], value1[key], f"{label}[{key}]"
                )
            return
        arr0 = np.asarray(value0)
        arr1 = np.asarray(value1)
        if arr0.dtype == object or arr1.dtype == object:
            self.assertEqual(arr0.shape, arr1.shape, label)
            for index in np.ndindex(arr0.shape):
                self._assert_value_equal(
                    arr0[index], arr1[index], f"{label}{index}"
                )
            return
        np.testing.assert_allclose(arr0, arr1, err_msg=label)

    def test_all_v3_tasks_are_registered(self) -> None:
        """Every v3 task should be registered with the shared space contract."""
        registered = set(list_all_envs())
        for task_id in _TASK_IDS:
            with self.subTest(task_id=task_id):
                self.assertIn(task_id, registered)
                spec = make_spec(task_id)
                self.assertEqual(spec.observation_space.shape, (39,))
                self.assertEqual(spec.action_space.shape, (4,))
                self.assertEqual(spec.config.max_episode_steps, 500)

    def test_unknown_task_name_fails_fast(self) -> None:
        """Overriding task_name with an unknown task should fail at construction."""
        with self.assertRaisesRegex(
            RuntimeError, "Unknown MetaWorld task_name: missing-v3"
        ):
            make_gymnasium(
                "Meta-World/reach-v3",
                task_name="missing-v3",
                num_envs=1,
            )

    def test_metaworld_namespace_alias_is_registered(self) -> None:
        """The historical EnvPool spelling should remain an alias."""
        env = make_gymnasium("MetaWorld/reach-v3", num_envs=1, seed=0)
        try:
            obs, _ = env.reset()
            self.assertEqual(obs.shape, (1, 39))
        finally:
            env.close()

    def test_reset_and_step_all_v3_tasks(self) -> None:
        """Every v3 task should reset and step without invalid outputs."""
        for task_id in _TASK_IDS:
            with self.subTest(task_id=task_id):
                env = make_gymnasium(task_id, num_envs=2, seed=0)
                try:
                    obs, info = env.reset()
                    self.assertEqual(obs.shape, (2, 39))
                    self.assertEqual(obs.dtype, np.float64)
                    self.assertTrue(_INFO_KEYS.issubset(info.keys()))
                    action = np.zeros((2, 4), dtype=np.float32)
                    obs, rew, term, trunc, info = env.step(action)
                    self.assertEqual(obs.shape, (2, 39))
                    self.assertEqual(rew.shape, (2,))
                    self.assertEqual(term.shape, (2,))
                    self.assertEqual(trunc.shape, (2,))
                    self.assertTrue(_INFO_KEYS.issubset(info.keys()))
                    self.assertFalse(np.any(np.isnan(obs)))
                    self.assertFalse(np.any(np.isnan(rew)))
                finally:
                    env.close()

    def test_all_v3_tasks_are_deterministic(self) -> None:
        """Same seed and same actions should produce identical rollouts."""
        rng = np.random.default_rng(123)
        actions = rng.uniform(-1.0, 1.0, size=(10, 2, 4)).astype(np.float32)
        for task_id in _TASK_IDS:
            with self.subTest(task_id=task_id):
                env0 = make_gymnasium(task_id, num_envs=2, seed=42)
                env1 = make_gymnasium(task_id, num_envs=2, seed=42)
                try:
                    obs0, info0 = env0.reset()
                    obs1, info1 = env1.reset()
                    np.testing.assert_allclose(obs0, obs1)
                    self._assert_info_equal(info0, info1)
                    for action in actions:
                        step0 = env0.step(action)
                        step1 = env1.step(action)
                        for value0, value1 in zip(
                            step0[:4], step1[:4], strict=True
                        ):
                            np.testing.assert_allclose(value0, value1)
                        self._assert_info_equal(step0[4], step1[4])
                finally:
                    env0.close()
                    env1.close()


if __name__ == "__main__":
    absltest.main()
