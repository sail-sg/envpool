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

from envpool.box2d import (
  LunarLanderContinuousEnvSpec,
  LunarLanderContinuousGymEnvPool,
  LunarLanderDiscreteEnvSpec,
  LunarLanderDiscreteGymEnvPool,
)


class _Box2dEnvPoolDeterministicTest(absltest.TestCase):

  def run_deterministic_check(
    self,
    spec_cls: Any,
    envpool_cls: Any,
    num_envs: int = 4,
    **kwargs: Any,
  ) -> None:
    env0 = envpool_cls(
      spec_cls(spec_cls.gen_config(num_envs=num_envs, seed=0, **kwargs))
    )
    env1 = envpool_cls(
      spec_cls(spec_cls.gen_config(num_envs=num_envs, seed=0, **kwargs))
    )
    env2 = envpool_cls(
      spec_cls(spec_cls.gen_config(num_envs=num_envs, seed=1, **kwargs))
    )
    act_space = env0.action_space
    for _ in range(5000):
      action = np.array([act_space.sample() for _ in range(num_envs)])
      obs0 = env0.step(action)[0]
      obs1 = env1.step(action)[0]
      obs2 = env2.step(action)[0]
      np.testing.assert_allclose(obs0, obs1)
      self.assertFalse(np.allclose(obs0, obs2))

  def test_lunar_lander(self) -> None:
    self.run_deterministic_check(
      LunarLanderContinuousEnvSpec, LunarLanderContinuousGymEnvPool
    )
    self.run_deterministic_check(
      LunarLanderDiscreteEnvSpec, LunarLanderDiscreteGymEnvPool
    )


if __name__ == "__main__":
  absltest.main()
