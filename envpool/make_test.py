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
from pathlib import Path
from typing import get_type_hints

import dm_env
import gym
import gymnasium
from absl.testing import absltest

import envpool
import envpool.minigrid.registration  # noqa: F401
from envpool.python.protocol import (
    DMEnvPool,
    EnvPool,
    EnvSpec,
    GymEnvPool,
    GymnasiumEnvPool,
)


class _MakeTest(absltest.TestCase):
    def test_version(self) -> None:
        print(envpool.__version__)

    def test_public_typing_interface(self) -> None:
        self.assertTrue(Path(envpool.__file__).with_name("py.typed").is_file())
        self.assertContainsSubsequence(
            envpool.__all__,
            [
                "EnvSpec",
                "EnvPool",
                "DMEnvPool",
                "GymEnvPool",
                "GymnasiumEnvPool",
            ],
        )
        self.assertIs(envpool.EnvSpec, EnvSpec)
        self.assertIs(envpool.EnvPool, EnvPool)
        self.assertIs(envpool.DMEnvPool, DMEnvPool)
        self.assertIs(envpool.GymEnvPool, GymEnvPool)
        self.assertIs(envpool.GymnasiumEnvPool, GymnasiumEnvPool)
        self.assertIs(get_type_hints(envpool.make_spec)["return"], EnvSpec)
        self.assertIs(get_type_hints(envpool.make_dm)["return"], DMEnvPool)
        self.assertIs(get_type_hints(envpool.make_gym)["return"], GymEnvPool)
        self.assertIs(
            get_type_hints(envpool.make_gymnasium)["return"],
            GymnasiumEnvPool,
        )

    def test_list_all_envs(self) -> None:
        pprint.pprint(envpool.list_all_envs())

    def test_make_atari(self) -> None:
        self.assertRaises(TypeError, envpool.make, "Pong-v5")
        spec = envpool.make_spec("Defender-v5")
        env_gym = envpool.make_gym("Defender-v5")
        env_dm = envpool.make_dm("Defender-v5")
        env_gymnasium = envpool.make_gymnasium("Defender-v5")
        print(env_dm)
        print(env_gym)
        print(env_gym)
        self.assertIsInstance(env_gymnasium, gymnasium.Env)
        self.assertIsInstance(env_gym, gym.Env)
        self.assertIsInstance(env_dm, dm_env.Environment)
        self.assertEqual(spec.action_space.n, 18)
        self.assertEqual(env_gym.action_space.n, 18)
        self.assertEqual(env_dm.action_spec().num_values, 18)
        self.assertEqual(env_gymnasium.action_space.n, 18)
        # not work for wrong bin, see issue #146
        for wrong in ["Combat", "Joust", "MazeCraze", "Warlords"]:
            self.assertRaises(AssertionError, envpool.make_gym, f"{wrong}-v5")
            self.assertRaises(
                AssertionError, envpool.make_gymnasium, f"{wrong}-v5"
            )
        # invalid argument will raise AssertionError, see issue #214
        self.assertRaises(
            AssertionError, envpool.make_gym, "Pong-v5", seed=2**31
        )
        self.assertRaises(
            AssertionError, envpool.make_gymnasium, "Pong-v5", seed=2**31
        )

    def test_make_vizdoom(self) -> None:
        spec = envpool.make_spec("MyWayHome-v1")
        print(spec)
        env0 = envpool.make_gym("MyWayHome-v1")
        env1 = envpool.make_gymnasium("MyWayHome-v1")
        print(env0)
        print(env1)
        self.assertIsInstance(env0, gym.Env)
        self.assertIsInstance(env1, gymnasium.Env)
        env0.reset()
        env1.reset()

    def check_step(self, env_list: list[str]) -> None:
        for task_id in env_list:
            envpool.make_spec(task_id)
            env_gym = envpool.make_gym(task_id)
            env_dm = envpool.make_dm(task_id)
            env_gymnasium = envpool.make_gymnasium(task_id)
            print(env_dm)
            print(env_gym)
            print(env_gymnasium)
            self.assertIsInstance(env_gym, gym.Env)
            self.assertIsInstance(env_dm, dm_env.Environment)
            self.assertIsInstance(env_gymnasium, gymnasium.Env)
            env_dm.reset()
            env_gym.reset()
            env_gymnasium.reset()

    def test_make_classic(self) -> None:
        self.check_step([
            "CartPole-v0",
            "CartPole-v1",
            "Pendulum-v0",
            "Pendulum-v1",
            "MountainCar-v0",
            "MountainCarContinuous-v0",
            "Acrobot-v1",
        ])

    def test_make_toytext(self) -> None:
        self.check_step([
            "Catch-v0",
            "FrozenLake-v1",
            "FrozenLake8x8-v1",
            "Taxi-v3",
            "NChain-v0",
            "CliffWalking-v1",
            "CliffWalkingSlippery-v1",
            "CliffWalking-v0",
            "Blackjack-v1",
        ])

    def test_make_box2d(self) -> None:
        self.check_step([
            "CarRacing-v3",
            "CarRacing-v2",
            "BipedalWalker-v3",
            "BipedalWalkerHardcore-v3",
            "LunarLander-v3",
            "LunarLander-v2",
            "LunarLanderContinuous-v3",
            "LunarLanderContinuous-v2",
        ])

    def test_make_minigrid(self) -> None:
        task_ids = sorted(
            task_id
            for task_id in envpool.list_all_envs()
            if task_id.startswith("MiniGrid-")
        )
        self.assertLen(task_ids, 75)
        self.check_step(task_ids)

    def test_make_mujoco_gym(self) -> None:
        self.check_step([
            "Ant-v3",
            "Ant-v4",
            "Ant-v5",
            "HalfCheetah-v3",
            "HalfCheetah-v4",
            "HalfCheetah-v5",
            "Hopper-v3",
            "Hopper-v4",
            "Hopper-v5",
            "Humanoid-v3",
            "Humanoid-v4",
            "Humanoid-v5",
            "HumanoidStandup-v2",
            "HumanoidStandup-v4",
            "HumanoidStandup-v5",
            "InvertedDoublePendulum-v2",
            "InvertedDoublePendulum-v4",
            "InvertedDoublePendulum-v5",
            "InvertedPendulum-v2",
            "InvertedPendulum-v4",
            "InvertedPendulum-v5",
            "Pusher-v2",
            "Pusher-v4",
            "Pusher-v5",
            "Reacher-v2",
            "Reacher-v4",
            "Reacher-v5",
            "Swimmer-v3",
            "Swimmer-v4",
            "Swimmer-v5",
            "Walker2d-v3",
            "Walker2d-v4",
            "Walker2d-v5",
        ])

    def test_make_mujoco_dmc(self) -> None:
        self.check_step([
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
            "HumanoidCMURun-v1",
            "HumanoidCMUStand-v1",
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
        ])

    def test_make_procgen(self) -> None:
        self.check_step([
            "BigfishEasy-v0",
            "BigfishHard-v0",
            "BossfightEasy-v0",
            "BossfightHard-v0",
            "CaveflyerEasy-v0",
            "CaveflyerHard-v0",
            "CaveflyerMemory-v0",
            "ChaserEasy-v0",
            "ChaserHard-v0",
            "ChaserExtreme-v0",
            "ClimberEasy-v0",
            "ClimberHard-v0",
            "CoinrunEasy-v0",
            "CoinrunHard-v0",
            "DodgeballEasy-v0",
            "DodgeballHard-v0",
            "DodgeballExtreme-v0",
            "DodgeballMemory-v0",
            "FruitbotEasy-v0",
            "FruitbotHard-v0",
            "HeistEasy-v0",
            "HeistHard-v0",
            "HeistMemory-v0",
            "JumperEasy-v0",
            "JumperHard-v0",
            "JumperMemory-v0",
            "LeaperEasy-v0",
            "LeaperHard-v0",
            "LeaperExtreme-v0",
            "MazeEasy-v0",
            "MazeHard-v0",
            "MazeMemory-v0",
            "MinerEasy-v0",
            "MinerHard-v0",
            "MinerMemory-v0",
            "NinjaEasy-v0",
            "NinjaHard-v0",
            "PlunderEasy-v0",
            "PlunderHard-v0",
            "StarpilotEasy-v0",
            "StarpilotHard-v0",
            "StarpilotExtreme-v0",
        ])


if __name__ == "__main__":
    absltest.main()
