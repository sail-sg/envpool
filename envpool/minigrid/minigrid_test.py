# Copyright 2023 Garena Online Private Limited
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
"""Unit tests for minigrid environments check."""

from typing import Any

import gymnasium as gym
import minigrid
import numpy as np
from absl.testing import absltest
from absl import logging

import envpool.minigrid.registration  # noqa: F401
from envpool.registration import make_gym


class _MiniGridEnvPoolTest(absltest.TestCase):

  def test_deterministic_check(
    self,
    task_id: str = "MiniGrid-Empty-5x5-v0",
    num_envs: int = 1,
    total: int = 100000,
    **kwargs: Any,
  ) -> None:
    env0 = gym.make(task_id)
    env1 = make_gym(task_id, num_envs=num_envs, seed=0, **kwargs)
    obs0, info0 = env0.reset()
    obs1, info1 = env1.reset()
    np.testing.assert_allclose(obs0["image"], obs1["image"][0])
    done0 = False
    acts = []
    for i in range(total):
      act = env0.action_space.sample()
      acts.append(act)
      if done0:
        obs0, info0 = env0.reset()
        auto_reset = True
        term0 = trunc0 = False
      else:
        obs0, rew0, term0, trunc0, info0 = env0.step(act)
        auto_reset = False
      obs1, rew1, term1, trunc1, info1 = env1.step(np.array([act]))
      self.assertEqual(obs0["image"].shape, (7, 7, 3))
      self.assertEqual(obs1["image"].shape, (num_envs, 7, 7, 3))
      done0 = term0 | trunc0
      done1 = term1 | trunc1
      if not auto_reset:
        np.testing.assert_allclose(rew0, rew1[0], rtol=1e-6)
        np.testing.assert_allclose(done0, done1[0])
      np.testing.assert_allclose(obs0["image"], obs1["image"][0])


if __name__ == "__main__":
  absltest.main()
