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

import time
from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from absl import logging
from absl.testing import absltest

import envpool.minigrid.registration  # noqa: F401
from envpool.registration import make_gym


class _MiniGridEnvPoolAlignTest(absltest.TestCase):

  def check_spec(
    self, spec0: gym.spaces.Space, spec1: gym.spaces.Space
  ) -> None:
    self.assertEqual(spec0.dtype, spec1.dtype)
    if isinstance(spec0, gym.spaces.Discrete):
      self.assertEqual(spec0.n, spec1.n)
    elif isinstance(spec0, gym.spaces.Box):
      np.testing.assert_allclose(spec0.low, spec1.low)
      np.testing.assert_allclose(spec0.high, spec1.high)

  def run_align_check(
    self,
    task_id: str,
    num_envs: int = 1,
    total: int = 10000,
    **kwargs: Any,
  ) -> None:
    env0 = gym.make(task_id)
    env1 = make_gym(task_id, num_envs=num_envs, seed=0, **kwargs)
    self.check_spec(
      env0.observation_space["direction"], env1.observation_space["direction"]
    )
    self.check_spec(
      env0.observation_space["image"], env1.observation_space["image"]
    )
    self.check_spec(env0.action_space, env1.action_space)
    done0 = True
    acts = []
    total_time_envpool = 0.0
    total_time_gym = 0.0
    for _ in range(total):
      act = env0.action_space.sample()
      acts.append(act)
      start = time.time()
      obs1, rew1, term1, trunc1, info1 = env1.step(np.array([act]))
      end = time.time()
      total_time_envpool += end - start
      start = time.time()
      if done0:
        obs0, info0 = env0.reset()
        auto_reset = True
        term0 = trunc0 = False
        env0.unwrapped.agent_pos = info1["agent_pos"][0]
        env0.unwrapped.agent_dir = obs1["direction"][0]
      else:
        obs0, rew0, term0, trunc0, info0 = env0.step(act)
        auto_reset = False
      end = time.time()
      total_time_gym += end - start
      self.assertEqual(obs0["image"].shape, (7, 7, 3))
      self.assertEqual(obs1["image"].shape, (num_envs, 7, 7, 3))
      done0 = term0 | trunc0
      done1 = term1 | trunc1
      if not auto_reset:
        np.testing.assert_allclose(obs0["direction"], obs1["direction"][0])
        np.testing.assert_allclose(obs0["image"], obs1["image"][0])
        np.testing.assert_allclose(rew0, rew1[0], rtol=1e-6)
        np.testing.assert_allclose(done0, done1[0])
        np.testing.assert_allclose(
          env0.unwrapped.agent_pos, info1["agent_pos"][0]
        )
    logging.info(f"{total_time_envpool=}")
    logging.info(f"{total_time_gym=}")

  def test_empty(self) -> None:
    self.run_align_check("MiniGrid-Empty-5x5-v0")
    self.run_align_check("MiniGrid-Empty-6x6-v0")
    self.run_align_check("MiniGrid-Empty-8x8-v0")
    self.run_align_check("MiniGrid-Empty-16x16-v0")
    self.run_align_check("MiniGrid-Empty-Random-5x5-v0")
    self.run_align_check("MiniGrid-Empty-Random-6x6-v0")


if __name__ == "__main__":
  absltest.main()
