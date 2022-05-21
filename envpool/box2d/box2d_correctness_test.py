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
from absl import logging
from absl.testing import absltest

from envpool.box2d import (
  BipedalWalkerEnvSpec,
  BipedalWalkerGymEnvPool,
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

  def test_bipedal_walker_space(self) -> None:
    env0 = gym.make("BipedalWalker-v3")
    env1 = BipedalWalkerGymEnvPool(
      BipedalWalkerEnvSpec(BipedalWalkerEnvSpec.gen_config())
    )
    self.run_space_check(env0, env1)

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

  def heuristic_lunar_lander_policy(
    self, s: np.ndarray, continuous: bool
  ) -> np.ndarray:
    angle_targ = np.clip(s[0] * 0.5 + s[2] * 1.0, -0.4, 0.4)
    hover_targ = 0.55 * np.abs(s[0])
    angle_todo = (angle_targ - s[4]) * 0.5 - s[5] * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - s[3] * 0.5

    if s[6] or s[7]:
      angle_todo = 0
      hover_todo = -(s[3]) * 0.5

    if continuous:
      a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
      a = np.clip(a, -1, 1)
    else:
      a = 0
      if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
        a = 2
      elif angle_todo < -0.05:
        a = 3
      elif angle_todo > 0.05:
        a = 1
    return a

  def solve_lunar_lander(self, num_envs: int, continuous: bool) -> None:
    if continuous:
      env = LunarLanderContinuousGymEnvPool(
        LunarLanderContinuousEnvSpec(
          LunarLanderContinuousEnvSpec.gen_config(num_envs=num_envs)
        )
      )
    else:
      env = LunarLanderDiscreteGymEnvPool(
        LunarLanderDiscreteEnvSpec(
          LunarLanderDiscreteEnvSpec.gen_config(num_envs=num_envs)
        )
      )
    # each env run two episodes
    for _ in range(2):
      env_id = np.arange(num_envs)
      done = np.array([False] * num_envs)
      obs = env.reset(env_id)
      rewards = np.zeros(num_envs)
      while not np.all(done):
        action = np.array(
          [self.heuristic_lunar_lander_policy(s, continuous) for s in obs]
        )
        obs, rew, done, info = env.step(action, env_id)
        env_id = info["env_id"]
        rewards[env_id] += rew
        obs = obs[~done]
        env_id = env_id[~done]
      mean_reward = np.mean(rewards)
      logging.info(
        f"{continuous}, {np.mean(rewards):.6f} ± {np.std(rewards):.6f}"
      )
      # the following number is from gym's 1000 episode mean reward
      if continuous:  # 283.872619 ± 18.881830
        self.assertTrue(abs(mean_reward - 284) < 10, (continuous, mean_reward))
      else:  # 236.898334 ± 105.832610
        self.assertTrue(abs(mean_reward - 237) < 20, (continuous, mean_reward))

  def test_lunar_lander_correctness(self, num_envs: int = 30) -> None:
    self.solve_lunar_lander(num_envs, True)
    self.solve_lunar_lander(num_envs, False)


if __name__ == "__main__":
  absltest.main()
