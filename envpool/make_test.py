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
from typing import List

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

  def test_make_vizdoom(self) -> None:
    spec = envpool.make_spec("MyWayHome-v1")
    print(spec)
    env = envpool.make_gym("MyWayHome-v1")
    print(env)
    self.assertIsInstance(env, gym.Env)
    env.reset()

  def check_step(self, env_list: List[str]) -> None:
    for task_id in env_list:
      envpool.make_spec(task_id)
      env_gym = envpool.make_gym(task_id)
      env_dm = envpool.make_dm(task_id)
      print(env_dm)
      print(env_gym)
      self.assertIsInstance(env_gym, gym.Env)
      self.assertIsInstance(env_dm, dm_env.Environment)
      env_dm.reset()
      env_gym.reset()

  def test_make_classic(self) -> None:
    self.check_step(
      [
        "CartPole-v0",
        "CartPole-v1",
        "Pendulum-v0",
        "Pendulum-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "Acrobot-v1",
      ]
    )

  def test_make_toytext(self) -> None:
    self.check_step(
      [
        "Catch-v0",
        "FrozenLake-v1",
        "FrozenLake8x8-v1",
        "Taxi-v3",
        "NChain-v0",
        "CliffWalking-v0",
        "Blackjack-v1",
      ]
    )

  def test_make_box2d(self) -> None:
    self.check_step([
      "LunarLander-v2",
      "LunarLanderContinuous-v2",
    ])

  def test_make_mujoco_gym(self) -> None:
    self.check_step(
      [
        "Ant-v3",
        "Ant-v4",
        "HalfCheetah-v3",
        "HalfCheetah-v4",
        "Hopper-v3",
        "Hopper-v4",
        "Humanoid-v3",
        "Humanoid-v4",
        "HumanoidStandup-v2",
        "HumanoidStandup-v4",
        "InvertedDoublePendulum-v2",
        "InvertedDoublePendulum-v4",
        "InvertedPendulum-v2",
        "InvertedPendulum-v4",
        "Pusher-v2",
        "Pusher-v4",
        "Reacher-v2",
        "Reacher-v4",
        "Swimmer-v3",
        "Swimmer-v4",
        "Walker2d-v3",
        "Walker2d-v4",
      ]
    )

  def test_make_mujoco_dmc(self) -> None:
    self.check_step(
      [
        "AcrobotSwingup-v1",
        "AcrobotSwingupSparse-v1",
        "BallInCupCatch-v1",
        "CartpoleBalance-v1",
        "CartpoleBalanceSparse-v1",
        "CartpoleSwingup-v1",
        "CartpoleSwingupSparse-v1",
        "CartpoleThreePoles-v1",
        "CartpoleTwoPoles-v1",
        "CheetahRun-v1",
        "FingerSpin-v1",
        "FingerTurnEasy-v1",
        "FingerTurnHard-v1",
        "FishSwim-v1",
        "FishUpright-v1",
        "HopperHop-v1",
        "HopperStand-v1",
        "HumanoidRun-v1",
        "HumanoidRunPureState-v1",
        "HumanoidStand-v1",
        "HumanoidWalk-v1",
        "HumanoidCmuRun-v1",
        "HumanoidCmuStand-v1",
        "ManipulatorBringBall-v1",
        "ManipulatorBringPeg-v1",
        "ManipulatorInsertBall-v1",
        "ManipulatorInsertPeg-v1",
        "PendulumSwingup-v1",
        "PointMassEasy-v1",
        "PointMassHard-v1",
        "ReacherEasy-v1",
        "ReacherHard-v1",
        "SwimmerSwimmer6-v1",
        "SwimmerSwimmer15-v1",
        "WalkerRun-v1",
        "WalkerStand-v1",
        "WalkerWalk-v1",
      ]
    )


if __name__ == "__main__":
  absltest.main()
