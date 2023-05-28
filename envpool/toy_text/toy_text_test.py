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

from typing import no_type_check

import gym
import numpy as np
from absl import logging
from absl.testing import absltest
from dm_env import TimeStep

import envpool.toy_text.registration  # noqa: F401
from envpool.registration import make, make_gym


class _ToyTextEnvTest(absltest.TestCase):

  def test_catch(self) -> None:
    num_envs = 3
    row, col = 10, 5
    for env_type in ["dm", "gym"]:
      e = make("Catch-v0", env_type=env_type, num_envs=num_envs)
      # get a successful trajectory
      if env_type == "dm":
        obs = e.reset().observation.obs
      else:
        obs, _ = e.reset()
      assert obs.shape == (num_envs, row, col)
      ball_pos = np.where(obs[:, 0] == 1)[1]
      paddle_pos = np.where(obs[:, -1] == 1)[1]
      for t in range(row - 1):
        action = np.sign(ball_pos - paddle_pos) + 1
        if env_type == "dm":
          ts: TimeStep = e.step(action, np.arange(num_envs))
          obs, rew, done = ts.observation.obs, ts.reward, ts.last()
        else:
          obs, rew, terminated, truncated, _ = e.step(
            action, np.arange(num_envs)
          )
          done = np.logical_or(terminated, truncated)
        assert obs.shape == (num_envs, row, col)
        paddle_pos = np.where(obs[:, -1] == 1)[1]
        if t != row - 2:
          assert np.all(rew == 0) and np.all(~done)
        else:
          assert np.all(rew == 1) and np.all(done)
      # get a failure trajectory
      if env_type == "dm":
        obs = e.reset().observation.obs
      else:
        obs, _ = e.reset()
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
          obs, rew, terminated, truncated, _ = e.step(
            action, np.arange(num_envs)
          )
          done = np.logical_or(terminated, truncated)
        assert obs.shape == (num_envs, row, col)
        paddle_pos = np.where(obs[:, -1] == 1)[1]
        if t != row - 2:
          assert np.all(rew == 0) and np.all(~done)
        else:
          assert np.all(rew == -1) and np.all(done)

  @no_type_check
  def test_frozen_lake(self) -> None:
    for size in [4, 8]:
      if size == 4:
        env = make_gym("FrozenLake-v1")
        ref = gym.make("FrozenLake-v1")
      else:
        env = make_gym("FrozenLake8x8-v1")
        ref = gym.make("FrozenLake8x8-v1")
      logging.info(env)
      assert isinstance(env.observation_space, gym.spaces.Discrete)
      assert env.observation_space.n == size * size
      assert isinstance(env.action_space, gym.spaces.Discrete)
      assert env.action_space.n == 4
      last_obs, _ = env.reset()
      elapsed_step = 0
      for _ in range(1000):
        act = np.random.randint(4, size=(1,))
        obs, rew, terminated, truncated, info = env.step(act)
        done = np.logical_or(terminated, truncated)
        flag = False
        for _ in range(50):
          ref.reset()
          ref._elapsed_steps = elapsed_step
          ref.unwrapped.s = int(last_obs[0])
          ref_obs, ref_rew, ref_terminated, ref_truncated, ref_info = ref.step(
            int(act[0])
          )
          ref_done = np.logical_or(ref_terminated, ref_truncated)
          if ref_obs == obs[0]:
            if ref_rew == rew[0] and ref_done == done[0]:
              flag = True
            else:
              logging.info(
                f"At step {elapsed_step}, action {act}, {last_obs} -> {obs}, "
                f"ref: {ref_obs}, {ref_rew}, {ref_done}, {ref_info}, "
                f"but got {obs}, {rew}, {done}, {info}"
              )
              break
        assert flag
        last_obs = obs
        elapsed_step += 1
        if done:
          elapsed_step = 0
          last_obs, _ = env.reset()

  @no_type_check
  def test_taxi(self) -> None:
    env = make_gym("Taxi-v3")
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert env.observation_space.n == 500
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 6
    ref = gym.make("Taxi-v3")
    for _ in range(10):
      # random agent
      ref.reset()
      ref.unwrapped.s = env.reset()[0][0]
      done = [False]
      while not done[0]:
        act = np.random.randint(6, size=(1,))
        obs, rew, term, trunc, info = env.step(act)
        done = np.logical_or(term, trunc)
        ref_obs, ref_rew, ref_term, ref_trunc, ref_info = ref.step(int(act[0]))
        ref_done = np.logical_or(ref_term, ref_trunc)
        assert ref_obs == obs[0] and ref_rew == rew[0] and ref_done == done[0]
        assert ref_term == term[0] and ref_trunc == trunc[0]
    locs = ref.unwrapped.locs
    left_point = [(4, 4), (0, 3), (4, 2), (0, 1)]
    right_point = [(0, 0), (4, 1), (0, 2), (4, 3)]
    left = {0: 1, 1: -1, 2: 1, 3: -1, 4: 1}
    right = {0: -1, 1: 1, 2: -1, 3: 1, 4: -1}
    for _ in range(10):
      # 10IQ agent
      ref.reset()
      obs = ref.unwrapped.s = env.reset()[0][0]
      x, y, s, t = ref.unwrapped.decode(obs)
      actions = []
      for i, g in [[0, s], [1, t]]:
        # to (4, 0)
        while (x, y) != (4, 0):
          if (x, y) in left_point:
            y -= 1
            actions.append(3)
          else:
            x += left[y]
            actions.append(int(left[y] == -1))
        if (x, y) == locs[g]:
          actions.append(4 + i)
        else:
          # to (0, 4)
          while (x, y) != (0, 4):
            if (x, y) in right_point:
              y += 1
              actions.append(2)
            else:
              x += right[y]
              actions.append(int(right[y] == -1))
            if (x, y) == locs[g]:
              actions.append(4 + i)
              break
      while len(actions):
        a = actions.pop(0)
        ref_obs, ref_rew, ref_terminated, ref_truncated, ref_info = ref.step(a)
        obs, rew, terminated, truncated, info = env.step(np.array([a], int))
        assert ref_obs == obs[0] and ref_rew == rew[
          0] and ref_terminated == terminated[0] and ref_truncated == truncated[
            0]
      assert ref_rew == 20 and ref_done

  def test_nchain(self) -> None:
    num_envs = 100
    env = make_gym("NChain-v0", num_envs=num_envs)
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert env.observation_space.n == 5
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 2
    done = [False]
    env.reset()
    reward = 0
    while not done[0]:
      actions = np.random.randint(2, size=(num_envs,))
      obs, rew, terminated, truncated, info = env.step(actions)
      done = np.logical_or(terminated, truncated)
      reward += rew
    assert abs(np.mean(reward) - 1310) < 30 and abs(np.std(reward) - 78) < 15

  def test_cliffwalking(self) -> None:
    env = make_gym("CliffWalking-v0")
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert env.observation_space.n == 48
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 4
    ref = gym.make("CliffWalking-v0")
    for i in range(12):
      action = [0] * 4 + [1] * i + [2] * 4
      assert env.reset()[0] == ref.reset()[0]
      while len(action) > 0:
        a = action.pop(0)
        ref_obs, ref_rew, ref_terminated, ref_truncated, ref_info = ref.step(a)
        ref_done = np.logical_or(ref_terminated, ref_truncated)
        obs, rew, terminated, truncated, info = env.step(np.array([a], int))
        assert ref_obs == obs[0] and ref_rew == rew[
          0] and ref_terminated == terminated[0] and ref_truncated == truncated[
            0]
        if ref_done:
          break

  def test_blackjack(self) -> None:
    np.random.seed(0)
    num_envs = 100
    env = make_gym("Blackjack-v1", num_envs=num_envs)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (3,)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 2
    reward, rewards = np.zeros(num_envs), []
    for _ in range(1000):
      obs, rew, terminated, truncated, info = env.step(
        np.random.randint(2, size=(num_envs,))
      )
      done = np.logical_or(terminated, truncated)
      reward += rew
      if np.any(done):
        rewards += reward[done].tolist()
        reward[done] = 0
    assert abs(np.mean(rewards) + 0.395) < 0.05
    assert abs(np.std(rewards) - 0.89) < 0.05


if __name__ == "__main__":
  absltest.main()
