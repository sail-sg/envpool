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
"""Mujoco gym env in EnvPool."""

from envpool.mujoco.mujoco_gym_envpool import (
  _GymAntEnvPool,
  _GymAntEnvSpec,
  _GymHalfCheetahEnvPool,
  _GymHalfCheetahEnvSpec,
  _GymHopperEnvPool,
  _GymHopperEnvSpec,
  _GymHumanoidEnvPool,
  _GymHumanoidEnvSpec,
  _GymHumanoidStandupEnvPool,
  _GymHumanoidStandupEnvSpec,
  _GymInvertedDoublePendulumEnvPool,
  _GymInvertedDoublePendulumEnvSpec,
  _GymInvertedPendulumEnvPool,
  _GymInvertedPendulumEnvSpec,
  _GymPusherEnvPool,
  _GymPusherEnvSpec,
  _GymReacherEnvPool,
  _GymReacherEnvSpec,
  _GymSwimmerEnvPool,
  _GymSwimmerEnvSpec,
  _GymWalker2dEnvPool,
  _GymWalker2dEnvSpec,
)
from envpool.python.api import py_env

GymAntEnvSpec, GymAntDMEnvPool, GymAntGymEnvPool = py_env(
  _GymAntEnvSpec, _GymAntEnvPool
)
(
  GymHalfCheetahEnvSpec,
  GymHalfCheetahDMEnvPool,
  GymHalfCheetahGymEnvPool,
) = py_env(_GymHalfCheetahEnvSpec, _GymHalfCheetahEnvPool)
GymHopperEnvSpec, GymHopperDMEnvPool, GymHopperGymEnvPool = py_env(
  _GymHopperEnvSpec, _GymHopperEnvPool
)
GymHumanoidEnvSpec, GymHumanoidDMEnvPool, GymHumanoidGymEnvPool = py_env(
  _GymHumanoidEnvSpec, _GymHumanoidEnvPool
)
(
  GymHumanoidStandupEnvSpec,
  GymHumanoidStandupDMEnvPool,
  GymHumanoidStandupGymEnvPool,
) = py_env(_GymHumanoidStandupEnvSpec, _GymHumanoidStandupEnvPool)
(
  GymInvertedDoublePendulumEnvSpec,
  GymInvertedDoublePendulumDMEnvPool,
  GymInvertedDoublePendulumGymEnvPool,
) = py_env(
  _GymInvertedDoublePendulumEnvSpec, _GymInvertedDoublePendulumEnvPool
)
(
  GymInvertedPendulumEnvSpec,
  GymInvertedPendulumDMEnvPool,
  GymInvertedPendulumGymEnvPool,
) = py_env(_GymInvertedPendulumEnvSpec, _GymInvertedPendulumEnvPool)
GymPusherEnvSpec, GymPusherDMEnvPool, GymPusherGymEnvPool = py_env(
  _GymPusherEnvSpec, _GymPusherEnvPool
)
GymReacherEnvSpec, GymReacherDMEnvPool, GymReacherGymEnvPool = py_env(
  _GymReacherEnvSpec, _GymReacherEnvPool
)
GymSwimmerEnvSpec, GymSwimmerDMEnvPool, GymSwimmerGymEnvPool = py_env(
  _GymSwimmerEnvSpec, _GymSwimmerEnvPool
)
GymWalker2dEnvSpec, GymWalker2dDMEnvPool, GymWalker2dGymEnvPool = py_env(
  _GymWalker2dEnvSpec, _GymWalker2dEnvPool
)

__all__ = [
  "GymAntEnvSpec",
  "GymAntDMEnvPool",
  "GymAntGymEnvPool",
  "GymHalfCheetahEnvSpec",
  "GymHalfCheetahDMEnvPool",
  "GymHalfCheetahGymEnvPool",
  "GymHopperEnvSpec",
  "GymHopperDMEnvPool",
  "GymHopperGymEnvPool",
  "GymHumanoidEnvSpec",
  "GymHumanoidDMEnvPool",
  "GymHumanoidGymEnvPool",
  "GymHumanoidStandupEnvSpec",
  "GymHumanoidStandupDMEnvPool",
  "GymHumanoidStandupGymEnvPool",
  "GymInvertedDoublePendulumEnvSpec",
  "GymInvertedDoublePendulumDMEnvPool",
  "GymInvertedDoublePendulumGymEnvPool",
  "GymInvertedPendulumEnvSpec",
  "GymInvertedPendulumDMEnvPool",
  "GymInvertedPendulumGymEnvPool",
  "GymPusherEnvSpec",
  "GymPusherDMEnvPool",
  "GymPusherGymEnvPool",
  "GymReacherEnvSpec",
  "GymReacherDMEnvPool",
  "GymReacherGymEnvPool",
  "GymSwimmerEnvSpec",
  "GymSwimmerDMEnvPool",
  "GymSwimmerGymEnvPool",
  "GymWalker2dEnvSpec",
  "GymWalker2dDMEnvPool",
  "GymWalker2dGymEnvPool",
]
