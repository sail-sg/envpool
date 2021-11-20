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

import numpy as np
from absl.testing import absltest

from envpool.classic_control import CartPoleEnvSpec, CartPoleGymEnvPool


class _CartPoleEnvPoolTest(absltest.TestCase):

  def test_seed(self) -> None:
    num_envs = 4
    config = CartPoleEnvSpec.gen_config(max_episode_steps=200, seed=0)
    spec = CartPoleEnvSpec(config)
    env0 = CartPoleGymEnvPool(spec)
    config = CartPoleEnvSpec.gen_config(max_episode_steps=200, seed=0)
    spec = CartPoleEnvSpec(config)
    env1 = CartPoleGymEnvPool(spec)
    config = CartPoleEnvSpec.gen_config(max_episode_steps=200, seed=1)
    spec = CartPoleEnvSpec(config)
    env2 = CartPoleGymEnvPool(spec)
    fmax = np.finfo(np.float32).max
    obs_range = np.array([4.8, fmax, np.pi / 7.5, fmax])
    for _ in range(1000):
      action = np.random.randint(2, size=(num_envs,))
      obs0 = env0.step(action)[0]
      obs1 = env1.step(action)[0]
      obs2 = env2.step(action)[0]
      np.testing.assert_allclose(obs0, obs1)
      self.assertTrue(np.abs(obs0 - obs2).sum() > 0)
      self.assertTrue(np.all(np.abs(obs0) < obs_range))
      self.assertTrue(np.all(np.abs(obs2) < obs_range))


if __name__ == "__main__":
  absltest.main()
