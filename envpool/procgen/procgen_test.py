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
"""Unit tests for Procgen environments alignment & deterministic check."""
from typing import Any

import gym
import numpy as np
from absl import logging
from absl.testing import absltest

from envpool.procgen.registration import procgen_game_config
from envpool.registration import make_gym


class _ProcgenEnvPoolTest(absltest.TestCase):

  def gym_deterministic_check(
    self, task_id: str, num_envs: int = 4, total: int = 100
  ) -> None:
    env0 = make_gym(task_id, num_envs=num_envs, seed=0)
    env1 = make_gym(task_id, num_envs=num_envs, seed=0)
    env2 = make_gym(task_id, num_envs=num_envs, seed=1)
    act_space = env0.action_space
    for _ in range(total):
      action = np.array([act_space.sample() for _ in range(num_envs)])
      obs0 = env0.step(action)[0]
      obs1 = env1.step(action)[0]
      obs2 = env2.step(action)[0]
      np.testing.assert_allclose(obs0, obs1)
      self.assertFalse(np.allclose(obs0, obs2))

  def gym_align_check(
    self, game_name: str, spec_cls: Any, envpool_cls: Any
  ) -> None:
    logging.info(f"align check for gym {game_name}")
    num_env = 1
    for i in range(2):
      env_gym = envpool_cls(
        spec_cls(
          spec_cls.gen_config(num_envs=num_env, seed=i, game_name=game_name)
        )
      )
      env_procgen = gym.make(
        f"procgen:procgen-{game_name}-v0",
        rand_seed=i,
        use_generated_assets=True
      )
      env_gym.reset(np.arange(num_env, dtype=np.int32))
      env_procgen.reset()
      act_space = env_procgen.action_space
      envpool_done = False
      cnt = 1
      while not envpool_done:
        cnt += 1
        action = np.array([act_space.sample() for _ in range(num_env)])
        _, raw_reward, raw_done, _ = env_procgen.step(action[0])
        step_info = env_gym.step(action)
        envpool_reward, envpool_done = step_info[1], step_info[2]
        envpool_reward = envpool_reward[0]
        envpool_done = envpool_done[0]  # type: ignore
        # must die and earn reward same time aligned
        self.assertTrue(envpool_reward == raw_reward)
        self.assertTrue(raw_done == envpool_done)

  def test_gym_deterministic(self) -> None:
    for env_config in procgen_game_config:
      env_name = env_config[0]
      task_id = f"{env_name.capitalize()}Hard-v0"
      self.gym_deterministic_check(task_id)

  # def test_gym_align(self) -> None:
  #   # iterate over all procgen games to test Gym align
  #   for game in procgen_games_list:
  #     self.gym_align_check(game, ProcgenEnvSpec, ProcgenGymEnvPool)


if __name__ == "__main__":
  absltest.main()
