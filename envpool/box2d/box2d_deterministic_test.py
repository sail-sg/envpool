# Copyright 2022 Garena Online Private Limited
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
"""Unit tests for box2d environments deterministic check."""

from typing import Any

import numpy as np
from absl.testing import absltest

import envpool.box2d.registration  # noqa: F401
from envpool.registration import make_gym, make_spec

_TASK_IDS = (
    "CarRacing-v2",
    "CarRacing-v3",
    "BipedalWalker-v3",
    "BipedalWalkerHardcore-v3",
    "LunarLander-v2",
    "LunarLander-v3",
    "LunarLanderContinuous-v2",
    "LunarLanderContinuous-v3",
)


def _max_episode_steps(task_id: str, **kwargs: Any) -> int:
    return int(make_spec(task_id, **kwargs).config.max_episode_steps)


class _Box2dEnvPoolDeterministicTest(absltest.TestCase):
    def _assert_info_equal(
        self, info0: dict[str, Any], info1: dict[str, Any]
    ) -> None:
        self.assertEqual(info0.keys(), info1.keys())
        for key in info0:
            self._assert_value_equal(info0[key], info1[key], f"info[{key}]")

    def _assert_value_equal(self, value0: Any, value1: Any, label: str) -> None:
        if isinstance(value0, dict):
            self.assertIsInstance(value1, dict)
            self.assertEqual(value0.keys(), value1.keys())
            for key in value0:
                self._assert_value_equal(
                    value0[key],
                    value1[key],
                    f"{label}[{key}]",
                )
        else:
            arr0 = np.asarray(value0)
            arr1 = np.asarray(value1)
            if arr0.dtype == object or arr1.dtype == object:
                self.assertEqual(arr0.shape, arr1.shape, label)
                for index in np.ndindex(arr0.shape):
                    self._assert_value_equal(
                        arr0[index],
                        arr1[index],
                        f"{label}{index}",
                    )
                return
            np.testing.assert_allclose(value0, value1, err_msg=label)

    def run_deterministic_check(
        self,
        task_id: str,
        num_envs: int = 4,
        total: int | None = None,
        action_seed: int = 1,
        **kwargs: Any,
    ) -> None:
        if total is None:
            total = _max_episode_steps(task_id, **kwargs)
        env0 = make_gym(task_id, num_envs=num_envs, seed=0, **kwargs)
        env1 = make_gym(task_id, num_envs=num_envs, seed=0, **kwargs)
        env2 = make_gym(task_id, num_envs=num_envs, seed=1, **kwargs)
        act_space = env0.action_space
        act_space.seed(action_seed)
        try:
            obs0, info0 = env0.reset()
            obs1, info1 = env1.reset()
            obs2, _ = env2.reset()
            np.testing.assert_allclose(obs0, obs1)
            self._assert_info_equal(info0, info1)
            differs = not np.allclose(obs0, obs2)
            for _ in range(total):
                action = np.array([act_space.sample() for _ in range(num_envs)])
                obs0, rew0, term0, trunc0, info0 = env0.step(action)
                obs1, rew1, term1, trunc1, info1 = env1.step(action)
                obs2, rew2, term2, trunc2, _ = env2.step(action)
                np.testing.assert_allclose(obs0, obs1)
                np.testing.assert_allclose(rew0, rew1)
                np.testing.assert_array_equal(term0, term1)
                np.testing.assert_array_equal(trunc0, trunc1)
                self._assert_info_equal(info0, info1)
                differs = (
                    differs
                    or not np.allclose(obs0, obs2)
                    or not np.allclose(rew0, rew2)
                    or not np.array_equal(term0, term2)
                    or not np.array_equal(trunc0, trunc2)
                )
            self.assertTrue(
                differs, msg=f"expected seed divergence for {task_id}"
            )
        finally:
            env0.close()
            env1.close()
            env2.close()

    def test_registered_tasks(self) -> None:
        for task_id in _TASK_IDS:
            with self.subTest(task_id=task_id):
                self.run_deterministic_check(task_id)

    def test_car_racing_short_time_limit(self) -> None:
        self.run_deterministic_check("CarRacing-v3", max_episode_steps=3)


if __name__ == "__main__":
    absltest.main()
