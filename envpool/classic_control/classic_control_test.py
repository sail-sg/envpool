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

from typing import Any, no_type_check

import gym
import numpy as np
from absl.testing import absltest

import envpool.classic_control.registration  # noqa: F401
from envpool.registration import make_gym


class _ClassicControlEnvPoolTest(absltest.TestCase):

  @no_type_check
  def run_space_check(self, env0: gym.Env, env1: Any) -> None:
    """Check if envpool.observation_space == gym.make().observation_space."""
    obs0, obs1 = env0.observation_space, env1.observation_space
    np.testing.assert_allclose(obs0.low, obs1.low)
    np.testing.assert_allclose(obs0.high, obs1.high)

  def run_deterministic_check(
    self,
    task_id: str,
    num_envs: int = 4,
    **kwargs: Any,
  ) -> None:
    env0 = make_gym(task_id, num_envs=num_envs, seed=0, **kwargs)
    env1 = make_gym(task_id, num_envs=num_envs, seed=0, **kwargs)
    env2 = make_gym(task_id, num_envs=num_envs, seed=1, **kwargs)
    act_space = env0.action_space
    eps = np.finfo(np.float32).eps
    obs_space = env0.observation_space
    obs_min, obs_max = obs_space.low - eps, obs_space.high + eps
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
    for i in range(10):
      np.random.seed(i)
      env0.action_space.seed(i)
      reset_fn(env0, env1)
      d0 = False
      while not d0:
        a = env0.action_space.sample()
        o0, r0, term0, trunc0, _ = env0.step(a)
        d0 = np.logical_or(term0, trunc0)
        o1, r1, term1, trunc1, _ = env1.step(np.array([a]), np.array([0]))
        d1 = np.logical_or(term1, trunc1)
        np.testing.assert_allclose(o0, o1[0], atol=1e-4)
        np.testing.assert_allclose(r0, r1[0])
        np.testing.assert_allclose(d0, d1[0])
        np.testing.assert_allclose(term0, term1[0])
        np.testing.assert_allclose(trunc0, trunc1[0])

  def test_cartpole(self) -> None:
    env0 = gym.make("CartPole-v1")
    env1 = make_gym("CartPole-v1")
    self.run_space_check(env0, env1)
    self.run_deterministic_check("CartPole-v1")

  def test_pendulum(self) -> None:
    env0 = gym.make("Pendulum-v1")
    env1 = make_gym("Pendulum-v1")
    self.run_space_check(env0, env1)
    self.run_deterministic_check("Pendulum-v1")

  def test_mountain_car(self) -> None:
    self.run_deterministic_check("MountainCar-v0")
    self.run_deterministic_check("MountainCarContinuous-v0")

    @no_type_check
    def reset_fn(env0: gym.Env, env1: Any) -> None:
      env0.reset()
      obs, _ = env1.reset()
      env0.unwrapped.state = obs[0]

    env0 = gym.make("MountainCar-v0")
    env1 = make_gym("MountainCar-v0")
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1, reset_fn)

    env0 = gym.make("MountainCarContinuous-v0")
    env1 = make_gym("MountainCarContinuous-v0")
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1, reset_fn)

  def test_acrobot(self) -> None:
    self.run_deterministic_check("Acrobot-v1")

    # in envpool we use float64 but gym use float32

    # def reset_fn(env0: gym.Env, env1: Any) -> None:
    #   env0.reset()
    #   obs, rew, done, info = env1.step(np.array([1]))
    #   env0.unwrapped.state = np.concatenate([info["state"][0], obs[0, -2:]])
    #   print(env0.unwrapped.state)

    env0 = gym.make("Acrobot-v1")
    env1 = make_gym("Acrobot-v1")
    self.run_space_check(env0, env1)
    # self.run_align_check(env0, env1, reset_fn)


if __name__ == "__main__":
  absltest.main()
