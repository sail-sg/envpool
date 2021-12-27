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

  def test_version(self) -> None:
    print(envpool.__version__)

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

  def test_make_classic_and_toytext(self) -> None:
    classic = [
      "CartPole-v0", "CartPole-v1", "Pendulum-v0", "MountainCar-v0",
      "MountainCarContinuous-v0", "Acrobot-v1"
    ]
    toytext = ["Catch-v0", "FrozenLake-v1", "FrozenLake8x8-v1", "Taxi-v3"]
    for task_id in classic + toytext:
      envpool.make_spec(task_id)
      env_gym = envpool.make_gym(task_id)
      env_dm = envpool.make_dm(task_id)
      print(env_dm)
      print(env_gym)
      self.assertIsInstance(env_gym, gym.Env)
      self.assertIsInstance(env_dm, dm_env.Environment)
      env_dm.reset()
      env_gym.reset()


if __name__ == "__main__":
  absltest.main()
