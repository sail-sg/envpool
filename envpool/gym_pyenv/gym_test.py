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
"""Unit tests for gym environments."""

import gym
import numpy as np
from absl.testing import absltest

from envpool.gym_pyenv import GymEnvPool, GymEnvSpec


class _GymEnvTest(absltest.TestCase):

  def test_cartpole(self) -> None:
    np.random.seed(0)
    num_envs = 10
    ref = [gym.make("CartPole-v0") for _ in range(num_envs)]
    spec = GymEnvSpec(
      GymEnvSpec.gen_config(
        num_envs=num_envs,
        env_fn=lambda: gym.make("CartPole-v0"),
        state_shape=ref[0].observation_space.shape,
        action_shape=[1],
      )
    )
    env = GymEnvPool(spec)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == ref[0].observation_space.shape
    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.action_space.shape == ref[0].action_space.shape
    for _ in range(1000):
      act = np.random.randint(2, size=(num_envs,))
      obs, rew, done, info = env.step(act)


if __name__ == "__main__":
  absltest.main()
