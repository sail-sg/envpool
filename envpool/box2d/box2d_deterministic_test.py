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
from envpool.registration import make_gym


class _Box2dEnvPoolDeterministicTest(absltest.TestCase):

  def run_deterministic_check(
    self,
    task_id: str,
    num_envs: int = 4,
    **kwargs: Any,
  ) -> None:
    env0 = make_gym(task_id, num_envs=num_envs, seed=0, **kwargs)
    env1 = make_gym(task_id, num_envs=num_envs, seed=0, **kwargs)
    env2 = make_gym(task_id, num_envs=num_envs, seed=1, **kwargs)
    act_space = env0.action_space
    for _ in range(5000):
      action = np.array([act_space.sample() for _ in range(num_envs)])
      obs0, rew0, terminated, truncated, info0 = env0.step(action)
      obs1, rew1, terminated, truncated, info1 = env1.step(action)
      obs2, rew2, terminated, truncated, info2 = env2.step(action)
      np.testing.assert_allclose(obs0, obs1)
      self.assertFalse(np.allclose(obs0, obs2))

  def test_car_racing(self) -> None:
    self.run_deterministic_check("CarRacing-v2")
    self.run_deterministic_check("CarRacing-v2", max_episode_steps=3)

  def test_bipedal_walker(self) -> None:
    self.run_deterministic_check("BipedalWalker-v3")
    self.run_deterministic_check("BipedalWalkerHardcore-v3")

  def test_lunar_lander(self) -> None:
    self.run_deterministic_check("LunarLanderContinuous-v2")
    self.run_deterministic_check("LunarLander-v2")


if __name__ == "__main__":
  absltest.main()
