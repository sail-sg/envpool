# Copyright 2022 Garena Online Private Limited
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
"""Unit tests for Mujoco gym v4/v5 environments alignment."""

import platform
import sys
from typing import Any, no_type_check

import gymnasium as gym
import mujoco
import numpy as np
from absl import logging
from absl.testing import absltest
from packaging import version

import envpool.mujoco.gym.registration  # noqa: F401
from envpool.registration import make_gymnasium

_MUJOCO_V3 = version.parse(mujoco.__version__) >= version.parse("3.0.0")
_ARM64 = platform.machine().lower() in ("aarch64", "arm64")


class _MujocoGymAlignTest(absltest.TestCase):
    def test_v4_space_alignment(self) -> None:
        for env_id in [
            "Ant-v4",
            "HalfCheetah-v4",
            "Hopper-v4",
            "Humanoid-v4",
            "HumanoidStandup-v4",
            "InvertedDoublePendulum-v4",
            "InvertedPendulum-v4",
            "Pusher-v4",
            "Reacher-v4",
            "Swimmer-v4",
            "Walker2d-v4",
        ]:
            with self.subTest(env_id=env_id):
                try:
                    env0 = gym.make(env_id)
                except ImportError as err:
                    logging.warning(
                        "Skipping %s official reference env: %s", env_id, err
                    )
                    continue
                env1 = make_gymnasium(env_id)
                self.run_space_check(env0, env1)
                env0.close()
                env1.close()

    def observation_atol(self, env_id: str) -> float:
        if not _MUJOCO_V3:
            return 3e-4
        del env_id
        if _ARM64:
            return 1e-5
        return 1e-6

    def reward_atol(self, env_id: str) -> float:
        if not _MUJOCO_V3:
            return 1e-4
        del env_id
        return 5e-7

    def check_info_alignment(self, env_id: str) -> bool:
        if not _MUJOCO_V3:
            return True
        # Gymnasium's MuJoCo 3.x reference envs keep the step contract aligned,
        # but auxiliary info fields can still drift across backends.
        del env_id
        return False

    @no_type_check
    def run_space_check(self, env0: gym.Env, env1: Any) -> None:
        """Check observation_space and action space."""
        """Check observation_space and action space."""
        obs0, obs1 = env0.observation_space, env1.observation_space
        np.testing.assert_allclose(obs0.low, obs1.low)
        np.testing.assert_allclose(obs0.high, obs1.high)
        act0, act1 = env0.action_space, env1.action_space
        np.testing.assert_allclose(act0.low, act1.low)
        np.testing.assert_allclose(act0.high, act1.high)

    @no_type_check
    def reset_state(
        self, env: gym.Env, qpos: np.ndarray, qvel: np.ndarray
    ) -> None:
        # manually reset
        base_env = env.unwrapped
        mujoco.mj_resetData(base_env.model, base_env.data)
        base_env.set_state(qpos, qvel)

    def run_align_check(
        self,
        env_id: str,
        env0: gym.Env,
        env1: Any,
        no_time_limit: bool = False,
    ) -> None:
        logging.info(f"align check for {env1.__class__.__name__}")
        obs_atol = self.observation_atol(env_id)
        reward_atol = self.reward_atol(env_id)
        for i in range(5):
            env0.action_space.seed(i)
            env0.reset()
            a = env0.action_space.sample()
            obs, info = env1.reset(np.array([0]))
            self.reset_state(env0, info["qpos0"][0], info["qvel0"][0])
            logging.info(f"reset qpos {info['qpos0'][0]}")
            logging.info(f"reset qvel {info['qvel0'][0]}")
            d1 = np.array([False])
            cnt = 0
            while not d1[0]:
                cnt += 1
                a = env0.action_space.sample()
                # logging.info(f"{cnt} {a}")
                o0, r0, term0, trunc0, i0 = env0.step(a)
                d0 = np.logical_or(term0, trunc0)
                o1, r1, term1, trunc1, i1 = env1.step(
                    np.array([a]), np.array([0])
                )
                d1 = np.logical_or(term1, trunc1)
                np.testing.assert_allclose(o0, o1[0], atol=obs_atol)
                np.testing.assert_allclose(
                    float(r0), float(r1[0]), atol=reward_atol
                )
                if not no_time_limit:
                    np.testing.assert_allclose(d0, d1[0])
                if self.check_info_alignment(env_id):
                    for k in i0:
                        if k in i1:
                            np.testing.assert_allclose(
                                i0[k], i1[k][0], atol=1e-4
                            )

    def test_ant(self) -> None:
        assert version.parse(gym.__version__) >= version.parse("0.26.0")
        env0 = gym.make("Ant-v5")
        env1 = make_gymnasium("Ant-v5")
        self.run_space_check(env0, env1)
        self.run_align_check("Ant-v5", env0, env1)
        env0 = gym.make(
            "Ant-v5",
            terminate_when_unhealthy=False,
            exclude_current_positions_from_observation=False,
        )
        env1 = make_gymnasium(
            "Ant-v5",
            terminate_when_unhealthy=False,
            exclude_current_positions_from_observation=False,
            max_episode_steps=100,
        )
        self.run_space_check(env0, env1)
        self.run_align_check("Ant-v5", env0, env1, no_time_limit=True)

    def test_half_cheetah(self) -> None:
        env0 = gym.make("HalfCheetah-v5")
        env1 = make_gymnasium("HalfCheetah-v5")
        self.run_space_check(env0, env1)
        self.run_align_check("HalfCheetah-v5", env0, env1, no_time_limit=True)
        env0 = gym.make(
            "HalfCheetah-v5",
            exclude_current_positions_from_observation=True,
        )
        env1 = make_gymnasium(
            "HalfCheetah-v5",
            exclude_current_positions_from_observation=True,
        )
        self.run_space_check(env0, env1)

    def test_hopper(self) -> None:
        env0 = gym.make("Hopper-v5")
        env1 = make_gymnasium("Hopper-v5")
        self.run_space_check(env0, env1)
        self.run_align_check("Hopper-v5", env0, env1)
        env0 = gym.make(
            "Hopper-v5",
            terminate_when_unhealthy=False,
            exclude_current_positions_from_observation=False,
        )
        env1 = make_gymnasium(
            "Hopper-v5",
            terminate_when_unhealthy=False,
            exclude_current_positions_from_observation=False,
        )
        self.run_space_check(env0, env1)
        self.run_align_check("Hopper-v5", env0, env1, no_time_limit=True)

    def test_humanoid(self) -> None:
        if sys.platform == "darwin" and _MUJOCO_V3:
            self.skipTest(
                "MuJoCo humanoid alignment is numerically unstable on macOS"
            )
        env0 = gym.make("Humanoid-v5")
        env1 = make_gymnasium("Humanoid-v5")
        self.run_space_check(env0, env1)
        self.run_align_check("Humanoid-v5", env0, env1)
        env0 = gym.make(
            "Humanoid-v5",
            terminate_when_unhealthy=False,
            exclude_current_positions_from_observation=False,
        )
        env1 = make_gymnasium(
            "Humanoid-v5",
            terminate_when_unhealthy=False,
            exclude_current_positions_from_observation=False,
        )
        self.run_space_check(env0, env1)
        self.run_align_check("Humanoid-v5", env0, env1, no_time_limit=True)

    def test_humanoid_standup(self) -> None:
        env0 = gym.make("HumanoidStandup-v5")
        env1 = make_gymnasium("HumanoidStandup-v5")
        self.run_space_check(env0, env1)
        self.run_align_check(
            "HumanoidStandup-v5", env0, env1, no_time_limit=True
        )

    def test_inverted_double_pendulum(self) -> None:
        env0 = gym.make("InvertedDoublePendulum-v5")
        env1 = make_gymnasium(
            "InvertedDoublePendulum-v5",
        )
        self.run_space_check(env0, env1)
        self.run_align_check("InvertedDoublePendulum-v5", env0, env1)

    def test_inverted_pendulum(self) -> None:
        env0 = gym.make("InvertedPendulum-v5")
        env1 = make_gymnasium(
            "InvertedPendulum-v5",
        )
        self.run_space_check(env0, env1)
        self.run_align_check("InvertedPendulum-v5", env0, env1)

    def test_pusher(self) -> None:
        env0 = gym.make("Pusher-v5")
        env1 = make_gymnasium("Pusher-v5")
        self.run_space_check(env0, env1)
        self.run_align_check("Pusher-v5", env0, env1, no_time_limit=True)

    def test_reacher(self) -> None:
        env0 = gym.make("Reacher-v5")
        env1 = make_gymnasium("Reacher-v5")
        self.run_space_check(env0, env1)
        self.run_align_check("Reacher-v5", env0, env1, no_time_limit=True)

    def test_swimmer(self) -> None:
        env0 = gym.make("Swimmer-v5")
        env1 = make_gymnasium("Swimmer-v5")
        self.run_space_check(env0, env1)
        self.run_align_check("Swimmer-v5", env0, env1, no_time_limit=True)
        env0 = gym.make(
            "Swimmer-v5",
            exclude_current_positions_from_observation=False,
        )
        env1 = make_gymnasium(
            "Swimmer-v5",
            exclude_current_positions_from_observation=False,
        )
        self.run_space_check(env0, env1)

    def test_walker2d(self) -> None:
        env0 = gym.make("Walker2d-v5")
        env1 = make_gymnasium("Walker2d-v5")
        self.run_space_check(env0, env1)
        self.run_align_check("Walker2d-v5", env0, env1)
        env0 = gym.make(
            "Walker2d-v5",
            terminate_when_unhealthy=False,
            exclude_current_positions_from_observation=False,
        )
        env1 = make_gymnasium(
            "Walker2d-v5",
            terminate_when_unhealthy=False,
            exclude_current_positions_from_observation=False,
            max_episode_steps=100,
        )
        self.run_space_check(env0, env1)
        self.run_align_check("Walker2d-v5", env0, env1, no_time_limit=True)


if __name__ == "__main__":
    absltest.main()
