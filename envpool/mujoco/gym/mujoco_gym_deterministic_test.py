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
"""Unit tests for Mujoco gym deterministic check."""

from typing import Any

import numpy as np
from absl.testing import absltest

from envpool.mujoco import GymAntEnvSpec as AntEnvSpec
from envpool.mujoco import GymAntGymEnvPool as AntGymEnvPool
from envpool.mujoco import GymHalfCheetahEnvSpec as HalfCheetahEnvSpec
from envpool.mujoco import GymHalfCheetahGymEnvPool as HalfCheetahGymEnvPool
from envpool.mujoco import GymHopperEnvSpec as HopperEnvSpec
from envpool.mujoco import GymHopperGymEnvPool as HopperGymEnvPool
from envpool.mujoco import GymHumanoidEnvSpec as HumanoidEnvSpec
from envpool.mujoco import GymHumanoidGymEnvPool as HumanoidGymEnvPool
from envpool.mujoco import GymHumanoidStandupEnvSpec as HumanoidStandupEnvSpec
from envpool.mujoco import (
  GymHumanoidStandupGymEnvPool as HumanoidStandupGymEnvPool,
)
from envpool.mujoco import (
  GymInvertedDoublePendulumEnvSpec as InvertedDoublePendulumEnvSpec,
)
from envpool.mujoco import (
  GymInvertedDoublePendulumGymEnvPool as InvertedDoublePendulumGymEnvPool,
)
from envpool.mujoco import (
  GymInvertedPendulumEnvSpec as InvertedPendulumEnvSpec,
)
from envpool.mujoco import (
  GymInvertedPendulumGymEnvPool as InvertedPendulumGymEnvPool,
)
from envpool.mujoco import GymPusherEnvSpec as PusherEnvSpec
from envpool.mujoco import GymPusherGymEnvPool as PusherGymEnvPool
from envpool.mujoco import GymReacherEnvSpec as ReacherEnvSpec
from envpool.mujoco import GymReacherGymEnvPool as ReacherGymEnvPool
from envpool.mujoco import GymSwimmerEnvSpec as SwimmerEnvSpec
from envpool.mujoco import GymSwimmerGymEnvPool as SwimmerGymEnvPool
from envpool.mujoco import GymWalker2dEnvSpec as Walker2dEnvSpec
from envpool.mujoco import GymWalker2dGymEnvPool as Walker2dGymEnvPool


class _MujocoEnvPoolDeterministicTest(absltest.TestCase):

  def check(self, spec_cls: Any, envpool_cls: Any, num_envs: int = 4) -> None:
    env0 = envpool_cls(
      spec_cls(spec_cls.gen_config(num_envs=num_envs, seed=0))
    )
    env1 = envpool_cls(
      spec_cls(spec_cls.gen_config(num_envs=num_envs, seed=0))
    )
    env2 = envpool_cls(
      spec_cls(spec_cls.gen_config(num_envs=num_envs, seed=1))
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

  def test_ant(self) -> None:
    self.check(AntEnvSpec, AntGymEnvPool)

  def test_half_cheetah(self) -> None:
    self.check(HalfCheetahEnvSpec, HalfCheetahGymEnvPool)

  def test_hopper(self) -> None:
    self.check(HopperEnvSpec, HopperGymEnvPool)

  def test_humanoid(self) -> None:
    self.check(HumanoidEnvSpec, HumanoidGymEnvPool)

  def test_humanoid_standup(self) -> None:
    self.check(HumanoidStandupEnvSpec, HumanoidStandupGymEnvPool)

  def test_inverted_double_pendulum(self) -> None:
    self.check(InvertedDoublePendulumEnvSpec, InvertedDoublePendulumGymEnvPool)

  def test_inverted_pendulum(self) -> None:
    self.check(InvertedPendulumEnvSpec, InvertedPendulumGymEnvPool)

  def test_pusher(self) -> None:
    self.check(PusherEnvSpec, PusherGymEnvPool)

  def test_reacher(self) -> None:
    self.check(ReacherEnvSpec, ReacherGymEnvPool)

  def test_swimmer(self) -> None:
    self.check(SwimmerEnvSpec, SwimmerGymEnvPool)

  def test_walker2d(self) -> None:
    self.check(Walker2dEnvSpec, Walker2dGymEnvPool)


if __name__ == "__main__":
  absltest.main()
