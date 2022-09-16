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
"""Unit tests for Mujoco gym v4 environments alignment."""

from typing import Any, no_type_check

import gym
import mjc_mwe
import numpy as np
from absl import logging
from absl.testing import absltest

from envpool.mujoco.gym import (
  GymAntEnvSpec,
  GymAntGymEnvPool,
  GymHalfCheetahEnvSpec,
  GymHalfCheetahGymEnvPool,
  GymHopperEnvSpec,
  GymHopperGymEnvPool,
  GymHumanoidEnvSpec,
  GymHumanoidGymEnvPool,
  GymHumanoidStandupEnvSpec,
  GymHumanoidStandupGymEnvPool,
  GymInvertedDoublePendulumEnvSpec,
  GymInvertedDoublePendulumGymEnvPool,
  GymInvertedPendulumEnvSpec,
  GymInvertedPendulumGymEnvPool,
  GymPusherEnvSpec,
  GymPusherGymEnvPool,
  GymReacherEnvSpec,
  GymReacherGymEnvPool,
  GymSwimmerEnvSpec,
  GymSwimmerGymEnvPool,
  GymWalker2dEnvSpec,
  GymWalker2dGymEnvPool,
)


class _MujocoGymAlignTest(absltest.TestCase):

  @no_type_check
  def run_space_check(self, env0: gym.Env, env1: Any) -> None:
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
    env._mujoco_bindings.mj_resetData(env.model, env.data)
    env.set_state(qpos, qvel)
    env._mujoco_bindings.mj_forward(env.model, env.data)

  def run_align_check(
    self, env0: gym.Env, env1: Any, no_time_limit: bool = False
  ) -> None:
    logging.info(f"align check for {env1.__class__.__name__}")
    for i in range(5):
      env0.action_space.seed(i)
      env0.reset()
      a = env0.action_space.sample()
      obs, info = env1.reset(np.array([0]))
      self.reset_state(env0, info["qpos0"][0], info["qvel0"][0])
      logging.info(f'reset qpos {info["qpos0"][0]}')
      logging.info(f'reset qvel {info["qvel0"][0]}')
      d1 = np.array([False])
      cnt = 0
      while not d1[0]:
        cnt += 1
        a = env0.action_space.sample()
        # logging.info(f"{cnt} {a}")
        o0, r0, d0, i0 = env0.step(a)
        o1, r1, d1, i1 = env1.step(np.array([a]), np.array([0]))
        np.testing.assert_allclose(o0, o1[0], atol=3e-4)
        np.testing.assert_allclose(r0, r1[0], atol=1e-4)
        if not no_time_limit:
          np.testing.assert_allclose(d0, d1[0])
        for k in i0:
          if k in i1:
            np.testing.assert_allclose(i0[k], i1[k][0], atol=1e-4)

  def test_ant(self) -> None:
    env0 = mjc_mwe.AntEnv()
    env1 = GymAntGymEnvPool(
      GymAntEnvSpec(GymAntEnvSpec.gen_config(gym_reset_return_info=True))
    )
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1)
    env0 = mjc_mwe.AntEnv(
      terminate_when_unhealthy=False,
      exclude_current_positions_from_observation=False,
    )
    env1 = GymAntGymEnvPool(
      GymAntEnvSpec(
        GymAntEnvSpec.gen_config(
          terminate_when_unhealthy=False,
          exclude_current_positions_from_observation=False,
          max_episode_steps=100,
          gym_reset_return_info=True,
        )
      )
    )
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1, no_time_limit=True)

  def test_half_cheetah(self) -> None:
    env0 = mjc_mwe.HalfCheetahEnv()
    env1 = GymHalfCheetahGymEnvPool(
      GymHalfCheetahEnvSpec(
        GymHalfCheetahEnvSpec.gen_config(gym_reset_return_info=True)
      )
    )
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1, no_time_limit=True)
    env0 = mjc_mwe.HalfCheetahEnv(
      exclude_current_positions_from_observation=True
    )
    env1 = GymHalfCheetahGymEnvPool(
      GymHalfCheetahEnvSpec(
        GymHalfCheetahEnvSpec.gen_config(
          exclude_current_positions_from_observation=True,
          gym_reset_return_info=True,
        )
      )
    )
    self.run_space_check(env0, env1)

  def test_hopper(self) -> None:
    env0 = mjc_mwe.HopperEnv()
    env1 = GymHopperGymEnvPool(
      GymHopperEnvSpec(
        GymHopperEnvSpec.gen_config(gym_reset_return_info=True)
      )
    )
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1)
    env0 = mjc_mwe.HopperEnv(
      terminate_when_unhealthy=False,
      exclude_current_positions_from_observation=False,
    )
    env1 = GymHopperGymEnvPool(
      GymHopperEnvSpec(
        GymHopperEnvSpec.gen_config(
          terminate_when_unhealthy=False,
          exclude_current_positions_from_observation=False,
          gym_reset_return_info=True,
        )
      )
    )
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1, no_time_limit=True)

  def test_humanoid(self) -> None:
    env0 = mjc_mwe.HumanoidEnv()
    env1 = GymHumanoidGymEnvPool(
      GymHumanoidEnvSpec(
        GymHumanoidEnvSpec.gen_config(gym_reset_return_info=True)
      )
    )
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1)
    env0 = mjc_mwe.HumanoidEnv(
      terminate_when_unhealthy=False,
      exclude_current_positions_from_observation=False,
    )
    env1 = GymHumanoidGymEnvPool(
      GymHumanoidEnvSpec(
        GymHumanoidEnvSpec.gen_config(
          terminate_when_unhealthy=False,
          exclude_current_positions_from_observation=False,
          gym_reset_return_info=True,
        )
      )
    )
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1, no_time_limit=True)

  def test_humanoid_standup(self) -> None:
    env0 = mjc_mwe.HumanoidStandupEnv()
    env1 = GymHumanoidStandupGymEnvPool(
      GymHumanoidStandupEnvSpec(
        GymHumanoidStandupEnvSpec.gen_config(gym_reset_return_info=True)
      )
    )
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1, no_time_limit=True)

  def test_inverted_double_pendulum(self) -> None:
    env0 = mjc_mwe.InvertedDoublePendulumEnv()
    env1 = GymInvertedDoublePendulumGymEnvPool(
      GymInvertedDoublePendulumEnvSpec(
        GymInvertedDoublePendulumEnvSpec.gen_config(
          gym_reset_return_info=True
        )
      )
    )
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1)

  def test_inverted_pendulum(self) -> None:
    env0 = mjc_mwe.InvertedPendulumEnv()
    env1 = GymInvertedPendulumGymEnvPool(
      GymInvertedPendulumEnvSpec(
        GymInvertedPendulumEnvSpec.gen_config(gym_reset_return_info=True)
      )
    )
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1)

  def test_pusher(self) -> None:
    env0 = mjc_mwe.PusherEnv()
    env1 = GymPusherGymEnvPool(
      GymPusherEnvSpec(
        GymPusherEnvSpec.gen_config(gym_reset_return_info=True)
      )
    )
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1, no_time_limit=True)

  def test_reacher(self) -> None:
    env0 = mjc_mwe.ReacherEnv()
    env1 = GymReacherGymEnvPool(
      GymReacherEnvSpec(
        GymReacherEnvSpec.gen_config(gym_reset_return_info=True)
      )
    )
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1, no_time_limit=True)

  def test_swimmer(self) -> None:
    env0 = mjc_mwe.SwimmerEnv()
    env1 = GymSwimmerGymEnvPool(
      GymSwimmerEnvSpec(
        GymSwimmerEnvSpec.gen_config(gym_reset_return_info=True)
      )
    )
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1, no_time_limit=True)
    env0 = mjc_mwe.SwimmerEnv(exclude_current_positions_from_observation=False)
    env1 = GymSwimmerGymEnvPool(
      GymSwimmerEnvSpec(
        GymSwimmerEnvSpec.gen_config(
          exclude_current_positions_from_observation=False,
          gym_reset_return_info=True,
        )
      )
    )
    self.run_space_check(env0, env1)

  def test_walker2d(self) -> None:
    env0 = mjc_mwe.Walker2dEnv()
    env1 = GymWalker2dGymEnvPool(
      GymWalker2dEnvSpec(
        GymWalker2dEnvSpec.gen_config(gym_reset_return_info=True)
      )
    )
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1)
    env0 = mjc_mwe.Walker2dEnv(
      terminate_when_unhealthy=False,
      exclude_current_positions_from_observation=False,
    )
    env1 = GymWalker2dGymEnvPool(
      GymWalker2dEnvSpec(
        GymWalker2dEnvSpec.gen_config(
          terminate_when_unhealthy=False,
          exclude_current_positions_from_observation=False,
          max_episode_steps=100,
          gym_reset_return_info=True,
        )
      )
    )
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1, no_time_limit=True)


if __name__ == "__main__":
  absltest.main()
