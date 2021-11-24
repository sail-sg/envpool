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

from typing import Any, Tuple

import gym
import numpy as np
from absl.testing import absltest
from dm_env import TimeStep

from envpool.classic_control import (
  CartPoleEnvSpec,
  CartPoleGymEnvPool,
  CatchDMEnvPool,
  CatchEnvSpec,
  CatchGymEnvPool,
  MountainCarContinuousEnvSpec,
  MountainCarContinuousGymEnvPool,
  MountainCarEnvSpec,
  MountainCarGymEnvPool,
  PendulumEnvSpec,
  PendulumGymEnvPool,
)


class _ClassicControlEnvPoolTest(absltest.TestCase):

  def run_space_check(self, spec_cls: Any) -> None:
    """Check if envpool.observation_space == gym.make().observation_space."""
    # TODO(jiayi): wait for #27

  def run_deterministic_check(
    self,
    spec_cls: Any,
    envpool_cls: Any,
    obs_range: Tuple[np.ndarray, np.ndarray],
    **kwargs: Any,
  ) -> None:
    num_envs = 4
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
    eps = np.finfo(np.float32).eps
    obs_min, obs_max = obs_range[0] - eps, obs_range[1] + eps
    for _ in range(5000):
      action = np.array([act_space.sample() for _ in range(num_envs)])
      obs0 = env0.step(action)[0]
      obs1 = env1.step(action)[0]
      obs2 = env2.step(action)[0]
      np.testing.assert_allclose(obs0, obs1)
      self.assertFalse(np.allclose(obs0, obs2))
      self.assertTrue(np.all(obs_min <= obs0), obs0)
      self.assertTrue(np.all(obs_min <= obs2), obs2)
      self.assertTrue(np.all(obs0 <= obs_max), obs0)
      self.assertTrue(np.all(obs2 <= obs_max), obs2)

  def run_align_check(self, env0: gym.Env, env1: Any, reset_fn: Any) -> None:
    for _ in range(10):
      reset_fn(env0, env1)
      d0 = False
      while not d0:
        a = env0.action_space.sample()
        o0, r0, d0, _ = env0.step(a)
        o1, r1, d1, _ = env1.step(np.array([a]), np.array([0]))
        np.testing.assert_allclose(o0, o1[0], atol=1e-6)
        np.testing.assert_allclose(r0, r1[0])
        np.testing.assert_allclose(d0, d1[0])

  def test_cartpole(self) -> None:
    fmax = np.finfo(np.float32).max
    obs_max = np.array([4.8, fmax, np.pi / 7.5, fmax])
    self.run_deterministic_check(
      CartPoleEnvSpec, CartPoleGymEnvPool, (-obs_max, obs_max)
    )

  def test_pendulum(self) -> None:
    obs_max = np.array([1.0, 1.0, 8.0])
    self.run_deterministic_check(
      PendulumEnvSpec, PendulumGymEnvPool, (-obs_max, obs_max)
    )

  def test_mountain_car(self) -> None:
    obs_min = np.array([-1.2, -0.07])
    obs_max = np.array([0.6, 0.07])

    self.run_deterministic_check(
      MountainCarEnvSpec,
      MountainCarGymEnvPool,
      (obs_min, obs_max),
    )

    self.run_deterministic_check(
      MountainCarContinuousEnvSpec,
      MountainCarContinuousGymEnvPool,
      (obs_min, obs_max),
    )

    def reset_fn(env0: gym.Env, env1: Any) -> None:
      env0.reset()
      obs = env1.reset()
      env0.unwrapped.state = obs[0]

    env0 = gym.make("MountainCar-v0")
    spec = MountainCarEnvSpec(MountainCarEnvSpec.gen_config())
    env1 = MountainCarGymEnvPool(spec)
    self.run_align_check(env0, env1, reset_fn)

    env0 = gym.make("MountainCarContinuous-v0")
    spec = MountainCarContinuousEnvSpec(
      MountainCarContinuousEnvSpec.gen_config()
    )
    env1 = MountainCarContinuousGymEnvPool(spec)
    self.run_align_check(env0, env1, reset_fn)


class _CatchEnvTest(absltest.TestCase):

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


if __name__ == "__main__":
  absltest.main()
