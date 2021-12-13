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

import gym
import numpy as np
from absl.testing import absltest
from dm_env import TimeStep

from envpool.toy_text import (
  CatchDMEnvPool,
  CatchEnvSpec,
  CatchGymEnvPool,
  FrozenLakeEnvSpec,
  FrozenLakeGymEnvPool,
)


class _ToyTextEnvTest(absltest.TestCase):

  def test_catch(self) -> None:
    num_envs = 3
    row, col = 10, 5
    config = CatchEnvSpec.gen_config(num_envs=num_envs)
    spec = CatchEnvSpec(config)
    for env_type in ["dm", "gym"]:
      if env_type == "dm":
        e = CatchDMEnvPool(spec)
      else:
        e = CatchGymEnvPool(spec)
      # get a successful trajectory
      if env_type == "dm":
        obs = e.reset().observation.obs  # type: ignore
      else:
        obs = e.reset()
      assert obs.shape == (num_envs, row, col)
      ball_pos = np.where(obs[:, 0] == 1)[1]
      paddle_pos = np.where(obs[:, -1] == 1)[1]
      for t in range(row - 1):
        action = np.sign(ball_pos - paddle_pos) + 1
        if env_type == "dm":
          ts: TimeStep = e.step(action, np.arange(num_envs))
          obs, rew, done = ts.observation.obs, ts.reward, ts.last()
        else:
          obs, rew, done, _ = e.step(action, np.arange(num_envs))
        assert obs.shape == (num_envs, row, col)
        paddle_pos = np.where(obs[:, -1] == 1)[1]
        if t != row - 2:
          assert np.all(rew == 0) and np.all(~done)
        else:
          assert np.all(rew == 1) and np.all(done)
      # get a failure trajectory
      if env_type == "dm":
        obs = e.reset().observation.obs  # type: ignore
      else:
        obs = e.reset()
      assert obs.shape == (num_envs, row, col)
      ball_pos = np.where(obs[:, 0] == 1)[1]
      paddle_pos = np.where(obs[:, -1] == 1)[1]
      for t in range(row - 1):
        action = np.sign(paddle_pos - ball_pos) + 1
        action[action == 1] = 0
        if env_type == "dm":
          ts = e.step(action, np.arange(num_envs))
          obs, rew, done = ts.observation.obs, ts.reward, ts.last()
        else:
          obs, rew, done, _ = e.step(action, np.arange(num_envs))
        assert obs.shape == (num_envs, row, col)
        paddle_pos = np.where(obs[:, -1] == 1)[1]
        if t != row - 2:
          assert np.all(rew == 0) and np.all(~done)
        else:
          assert np.all(rew == -1) and np.all(done)

  def test_frozen_lake(self) -> None:
    for size in [4, 8]:
      config = FrozenLakeEnvSpec.gen_config(num_envs=1, size=size)
      spec = FrozenLakeEnvSpec(config)
      env = FrozenLakeGymEnvPool(spec)
      assert isinstance(env.observation_space, gym.spaces.Discrete)
      assert env.observation_space.n == size * size
      assert isinstance(env.action_space, gym.spaces.Discrete)
      assert env.action_space.n == 4
      if size == 4:
        ref = gym.make("FrozenLake-v1")
      else:
        ref = gym.make("FrozenLake8x8-v1")
      last_obs = env.reset()
      elapsed_step = 0
      for _ in range(10000):
        act = np.random.randint(4, size=(1,))
        obs, rew, done, info = env.step(act)
        flag = False
        for _ in range(20):
          ref.reset()
          ref._elapsed_steps = elapsed_step
          ref.unwrapped.s = int(last_obs[0])
          ref_obs, ref_rew, ref_done, ref_info = ref.step(int(act[0]))
          if ref_obs == obs[0]:
            if ref_rew == rew[0] and ref_done == done[0]:
              flag = True
            else:
              break
        assert flag
        last_obs = obs
        elapsed_step += 1
        if done:
          elapsed_step = 0
          last_obs = env.reset()


if __name__ == "__main__":
  absltest.main()
