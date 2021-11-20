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
"""Test for envpool.make."""

import pprint

import dm_env
import gym
from absl.testing import absltest

import envpool


class _MakeTest(absltest.TestCase):

  def test_list_all_envs(self) -> None:
    pprint.pprint(envpool.list_all_envs())

  def test_make_atari(self) -> None:
    self.assertRaises(TypeError, envpool.make, "Pong-v5")
    spec = envpool.make_spec("Defender-v5")
    env_gym = envpool.make_gym("Defender-v5")
    env_dm = envpool.make_dm("Defender-v5")
    print(env_dm)
    print(env_gym)
    self.assertIsInstance(env_gym, gym.Env)
    self.assertIsInstance(env_dm, dm_env.Environment)
    self.assertEqual(spec.action_space.n, 18)
    self.assertEqual(env_gym.action_space.n, 18)
    self.assertEqual(env_dm.action_spec().num_values, 18)

  def test_make_classic(self) -> None:
    spec = envpool.make_spec("CartPole-v0")
    env_gym = envpool.make_gym("CartPole-v1")
    env_dm = envpool.make_dm("CartPole-v1")
    print(env_dm)
    print(env_gym)
    self.assertIsInstance(env_gym, gym.Env)
    self.assertIsInstance(env_dm, dm_env.Environment)
    # check reward threshold
    self.assertEqual(spec.reward_threshold, 195.0)
    self.assertEqual(env_gym.spec.reward_threshold, 475.0)


if __name__ == "__main__":
  absltest.main()
