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
"""Unit tests for box2d environments correctness check."""

from typing import Any, no_type_check

import gym
import numpy as np
from absl.testing import absltest

from envpool.box2d import (
  LunarLanderContinuousEnvSpec,
  LunarLanderContinuousGymEnvPool,
  LunarLanderDiscreteEnvSpec,
  LunarLanderDiscreteGymEnvPool,
)


class _Box2dEnvPoolCorrectnessTest(absltest.TestCase):

  @no_type_check
  def run_space_check(self, env0: gym.Env, env1: Any) -> None:
    """Check observation_space and action space."""
    obs0, obs1 = env0.observation_space, env1.observation_space
    np.testing.assert_allclose(obs0.shape, obs1.shape)
    act0, act1 = env0.action_space, env1.action_space
    if isinstance(act0, gym.spaces.Box):
      np.testing.assert_allclose(act0.low, act1.low)
      np.testing.assert_allclose(act0.high, act1.high)
    elif isinstance(act0, gym.spaces.Discrete):
      np.testing.assert_allclose(act0.n, act1.n)

  def test_lunar_lander_space(self) -> None:
    env0 = gym.make("LunarLander-v2")
    env1 = LunarLanderDiscreteGymEnvPool(
      LunarLanderDiscreteEnvSpec(LunarLanderDiscreteEnvSpec.gen_config())
    )
    self.run_space_check(env0, env1)

    env0 = gym.make("LunarLanderContinuous-v2")
    env1 = LunarLanderContinuousGymEnvPool(
      LunarLanderContinuousEnvSpec(LunarLanderContinuousEnvSpec.gen_config())
    )
    self.run_space_check(env0, env1)


if __name__ == "__main__":
  absltest.main()
