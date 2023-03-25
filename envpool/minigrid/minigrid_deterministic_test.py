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

import numpy as np
from absl.testing import absltest

import envpool.minigrid.registration  # noqa: F401
from envpool.registration import make_gym


class _MiniGridEnvPoolDeterministicTest(absltest.TestCase):

  def run_deterministic_check(
    self,
    task_id: str,
    num_envs: int = 4,
    total: int = 5000,
    seed: int = 1,
    **kwargs: Any,
  ) -> None:
    env0 = make_gym(task_id, num_envs=num_envs, seed=0, **kwargs)
    env1 = make_gym(task_id, num_envs=num_envs, seed=0, **kwargs)
    env2 = make_gym(task_id, num_envs=num_envs, seed=1, **kwargs)
    act_space = env0.action_space
    act_space.seed(seed)
    same_count = 0
    for _ in range(total):
      action = np.array([act_space.sample() for _ in range(num_envs)])
      obs0, rew0, terminated, truncated, info0 = env0.step(action)
      obs1, rew1, terminated, truncated, info1 = env1.step(action)
      obs2, rew2, terminated, truncated, info2 = env2.step(action)
      np.testing.assert_allclose(obs0["image"], obs1["image"])
      np.testing.assert_allclose(obs0["direction"], obs1["direction"])
      # TODO: this may fail because the available state in minigrid env
      # is limited
      same_count += np.allclose(obs0["image"], obs2["image"]) and np.allclose(
        obs0["direction"], obs2["direction"]
      )
    assert same_count == 0, f"{same_count=}"

  def test_empty(self) -> None:
    self.run_deterministic_check("MiniGrid-Empty-Random-5x5-v0")
    self.run_deterministic_check("MiniGrid-Empty-Random-6x6-v0")


if __name__ == "__main__":
  absltest.main()
