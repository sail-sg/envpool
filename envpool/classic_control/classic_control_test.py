# Copyright 2021 Garena Online Private Limited
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
"""Unit tests for classic control environments."""

from typing import Any

import numpy as np
from absl.testing import absltest

from envpool.classic_control import (
  CartPoleEnvSpec,
  CartPoleGymEnvPool,
  PendulumEnvSpec,
  PendulumGymEnvPool,
)


class _ClassicControlEnvPoolTest(absltest.TestCase):

  def run_space_check(self, spec_cls: Any) -> None:
    """Check if envpool.observation_space == gym.make().observation_space."""
    # TODO(jiayi): wait for #27

  def run_deterministic_check(
    self, spec_cls: Any, envpool_cls: Any, obs_range: np.ndarray
  ) -> None:
    num_envs = 4
    env0 = envpool_cls(
      spec_cls(spec_cls.gen_config(num_envs=num_envs, seed=0))
    )
    env1 = envpool_cls(
      spec_cls(spec_cls.gen_config(num_envs=num_envs, seed=0))
    )
    env2 = envpool_cls(
      spec_cls(spec_cls.gen_config(num_envs=num_envs, seed=1))
    )
    act_space = env0.action_space
    for _ in range(5000):
      action = np.array([act_space.sample() for _ in range(num_envs)])
      obs0 = env0.step(action)[0]
      obs1 = env1.step(action)[0]
      obs2 = env2.step(action)[0]
      np.testing.assert_allclose(obs0, obs1)
      self.assertTrue(np.abs(obs0 - obs2).sum() > 0)
      self.assertTrue(np.all(np.abs(obs0) <= obs_range))
      self.assertTrue(np.all(np.abs(obs2) <= obs_range))

  def test_cartpole(self) -> None:
    fmax = np.finfo(np.float32).max
    obs_range = np.array([4.8, fmax, np.pi / 7.5, fmax])
    self.run_deterministic_check(
      CartPoleEnvSpec, CartPoleGymEnvPool, obs_range
    )

  def test_pendulum(self) -> None:
    obs_range = np.array([1.0, 1.0, 8.0])
    self.run_deterministic_check(
      PendulumEnvSpec, PendulumGymEnvPool, obs_range
    )


if __name__ == "__main__":
  absltest.main()
