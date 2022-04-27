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
"""Unit tests for Mujoco environments."""

from typing import Any, no_type_check

import gym
import mjc_mwe
import numpy as np
from absl import logging
from absl.testing import absltest

from envpool.mujoco import (
  AntEnvSpec,
  AntGymEnvPool,
  HalfCheetahEnvSpec,
  HalfCheetahGymEnvPool,
  HopperEnvSpec,
  HopperGymEnvPool,
  HumanoidEnvSpec,
  HumanoidGymEnvPool,
  HumanoidStandupEnvSpec,
  HumanoidStandupGymEnvPool,
  InvertedDoublePendulumEnvSpec,
  InvertedDoublePendulumGymEnvPool,
  InvertedPendulumEnvSpec,
  InvertedPendulumGymEnvPool,
  PusherEnvSpec,
  PusherGymEnvPool,
  ReacherEnvSpec,
  ReacherGymEnvPool,
  SwimmerEnvSpec,
  SwimmerGymEnvPool,
  Walker2dEnvSpec,
  Walker2dGymEnvPool,
)


class _MujocoEnvPoolTest(absltest.TestCase):

  @no_type_check
  def run_space_check(self, env0: gym.Env, env1: Any) -> None:
    """Check if envpool.observation_space == gym.make().observation_space."""
    obs0, obs1 = env0.observation_space, env1.observation_space
    np.testing.assert_allclose(obs0.low, obs1.low)
    np.testing.assert_allclose(obs0.high, obs1.high)

  def run_deterministic_check(
    self,
    spec_cls: Any,
    envpool_cls: Any,
    num_envs: int = 4,
    **kwargs: Any,
  ) -> None:
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
    obs_space = env0.observation_space
    obs_min, obs_max = obs_space.low - eps, obs_space.high + eps
    for _ in range(3000):
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
      obs, _, _, info = env1.step(np.array([a]), np.array([0]))
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
    env1 = AntGymEnvPool(AntEnvSpec(AntEnvSpec.gen_config()))
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1)
    self.run_deterministic_check(AntEnvSpec, AntGymEnvPool)

  def test_half_cheetah(self) -> None:
    env0 = mjc_mwe.HalfCheetahEnv()
    env1 = HalfCheetahGymEnvPool(
      HalfCheetahEnvSpec(HalfCheetahEnvSpec.gen_config())
    )
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1, no_time_limit=True)
    self.run_deterministic_check(HalfCheetahEnvSpec, HalfCheetahGymEnvPool)

  def test_hopper(self) -> None:
    env0 = mjc_mwe.HopperEnv()
    env1 = HopperGymEnvPool(HopperEnvSpec(HopperEnvSpec.gen_config()))
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1)
    self.run_deterministic_check(HopperEnvSpec, HopperGymEnvPool)

  def test_humanoid(self) -> None:
    env0 = mjc_mwe.HumanoidEnv()
    env1 = HumanoidGymEnvPool(HumanoidEnvSpec(HumanoidEnvSpec.gen_config()))
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1)
    self.run_deterministic_check(HumanoidEnvSpec, HumanoidGymEnvPool)

  def test_humanoid_standup(self) -> None:
    env0 = mjc_mwe.HumanoidStandupEnv()
    env1 = HumanoidStandupGymEnvPool(
      HumanoidStandupEnvSpec(HumanoidStandupEnvSpec.gen_config())
    )
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1, no_time_limit=True)
    self.run_deterministic_check(
      HumanoidStandupEnvSpec, HumanoidStandupGymEnvPool
    )

  def test_inverted_double_pendulum(self) -> None:
    env0 = mjc_mwe.InvertedDoublePendulumEnv()
    env1 = InvertedDoublePendulumGymEnvPool(
      InvertedDoublePendulumEnvSpec(
        InvertedDoublePendulumEnvSpec.gen_config()
      )
    )
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1)
    self.run_deterministic_check(
      InvertedDoublePendulumEnvSpec, InvertedDoublePendulumGymEnvPool
    )

  def test_inverted_pendulum(self) -> None:
    env0 = mjc_mwe.InvertedPendulumEnv()
    env1 = InvertedPendulumGymEnvPool(
      InvertedPendulumEnvSpec(InvertedPendulumEnvSpec.gen_config())
    )
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1)
    self.run_deterministic_check(
      InvertedPendulumEnvSpec, InvertedPendulumGymEnvPool
    )

  def test_pusher(self) -> None:
    env0 = mjc_mwe.PusherEnv()
    env1 = PusherGymEnvPool(PusherEnvSpec(PusherEnvSpec.gen_config()))
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1, no_time_limit=True)
    self.run_deterministic_check(PusherEnvSpec, PusherGymEnvPool)

  def test_reacher(self) -> None:
    env0 = mjc_mwe.ReacherEnv()
    env1 = ReacherGymEnvPool(ReacherEnvSpec(ReacherEnvSpec.gen_config()))
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1, no_time_limit=True)
    self.run_deterministic_check(ReacherEnvSpec, ReacherGymEnvPool)

  def test_swimmer(self) -> None:
    env0 = mjc_mwe.SwimmerEnv()
    env1 = SwimmerGymEnvPool(SwimmerEnvSpec(SwimmerEnvSpec.gen_config()))
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1, no_time_limit=True)
    self.run_deterministic_check(SwimmerEnvSpec, SwimmerGymEnvPool)

  def test_walker2d(self) -> None:
    env0 = mjc_mwe.Walker2dEnv()
    env1 = Walker2dGymEnvPool(Walker2dEnvSpec(Walker2dEnvSpec.gen_config()))
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1, no_time_limit=True)
    self.run_deterministic_check(Walker2dEnvSpec, Walker2dGymEnvPool)


if __name__ == "__main__":
  absltest.main()
