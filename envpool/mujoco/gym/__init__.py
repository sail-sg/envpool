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

(GymAntEnvSpec, GymAntDMEnvPool, GymAntGymEnvPool,
 GymAntGymnasiumEnvPool) = py_env(_GymAntEnvSpec, _GymAntEnvPool)
(
  GymHalfCheetahEnvSpec,
  GymHalfCheetahDMEnvPool,
  GymHalfCheetahGymEnvPool,
  GymHalfCheetahGymnasiumEnvPool,
) = py_env(_GymHalfCheetahEnvSpec, _GymHalfCheetahEnvPool)
(
  GymHopperEnvSpec, GymHopperDMEnvPool, GymHopperGymEnvPool,
  GymHopperGymnasiumEnvPool
) = py_env(_GymHopperEnvSpec, _GymHopperEnvPool)
(
  GymHumanoidEnvSpec, GymHumanoidDMEnvPool, GymHumanoidGymEnvPool,
  GymHumanoidGymnasiumEnvPool
) = py_env(_GymHumanoidEnvSpec, _GymHumanoidEnvPool)
(
  GymHumanoidStandupEnvSpec,
  GymHumanoidStandupDMEnvPool,
  GymHumanoidStandupGymEnvPool,
  GymHumanoidStandupGymnasiumEnvPool,
) = py_env(_GymHumanoidStandupEnvSpec, _GymHumanoidStandupEnvPool)
(
  GymInvertedDoublePendulumEnvSpec,
  GymInvertedDoublePendulumDMEnvPool,
  GymInvertedDoublePendulumGymEnvPool,
  GymInvertedDoublePendulumGymnasiumEnvPool,
) = py_env(
  _GymInvertedDoublePendulumEnvSpec, _GymInvertedDoublePendulumEnvPool
)
(
  GymInvertedPendulumEnvSpec,
  GymInvertedPendulumDMEnvPool,
  GymInvertedPendulumGymEnvPool,
  GymInvertedPendulumGymnasiumEnvPool,
) = py_env(_GymInvertedPendulumEnvSpec, _GymInvertedPendulumEnvPool)
(
  GymPusherEnvSpec, GymPusherDMEnvPool, GymPusherGymEnvPool,
  GymPusherGymnasiumEnvPool
) = py_env(_GymPusherEnvSpec, _GymPusherEnvPool)
(
  GymReacherEnvSpec, GymReacherDMEnvPool, GymReacherGymEnvPool,
  GymReacherGymnasiumEnvPool
) = py_env(_GymReacherEnvSpec, _GymReacherEnvPool)
(
  GymSwimmerEnvSpec, GymSwimmerDMEnvPool, GymSwimmerGymEnvPool,
  GymSwimmerGymnasiumEnvPool
) = py_env(_GymSwimmerEnvSpec, _GymSwimmerEnvPool)
(
  GymWalker2dEnvSpec, GymWalker2dDMEnvPool, GymWalker2dGymEnvPool,
  GymWalker2dGymnasiumEnvPool
) = py_env(_GymWalker2dEnvSpec, _GymWalker2dEnvPool)

__all__ = [
  "GymAntEnvSpec",
  "GymAntDMEnvPool",
  "GymAntGymEnvPool",
  "GymnasiumAntGymEnvPool",
  "GymHalfCheetahEnvSpec",
  "GymHalfCheetahDMEnvPool",
  "GymHalfCheetahGymEnvPool",
  "GymHalfCheetahGymnasiumEnvPool",
  "GymHopperEnvSpec",
  "GymHopperDMEnvPool",
  "GymHopperGymEnvPool",
  "GymHopperGymnasiumEnvPool",
  "GymHumanoidEnvSpec",
  "GymHumanoidDMEnvPool",
  "GymHumanoidGymEnvPool",
  "GymHumanoidGymnasiumEnvPool",
  "GymHumanoidStandupEnvSpec",
  "GymHumanoidStandupDMEnvPool",
  "GymHumanoidStandupGymEnvPool",
  "GymHumanoidStandupGymnasiumEnvPool",
  "GymInvertedDoublePendulumEnvSpec",
  "GymInvertedDoublePendulumDMEnvPool",
  "GymInvertedDoublePendulumGymEnvPool",
  "GymInvertedDoublePendulumGymnasiumEnvPool",
  "GymInvertedPendulumEnvSpec",
  "GymInvertedPendulumDMEnvPool",
  "GymInvertedPendulumGymEnvPool",
  "GymInvertedPendulumGymnasiumEnvPool",
  "GymPusherEnvSpec",
  "GymPusherDMEnvPool",
  "GymPusherGymEnvPool",
  "GymPusherGymnasiumEnvPool",
  "GymReacherEnvSpec",
  "GymReacherDMEnvPool",
  "GymReacherGymEnvPool",
  "GymReacherGymnasiumEnvPool",
  "GymSwimmerEnvSpec",
  "GymSwimmerDMEnvPool",
  "GymSwimmerGymEnvPool",
  "GymSwimmerGymnasiumEnvPool",
  "GymWalker2dEnvSpec",
  "GymWalker2dDMEnvPool",
  "GymWalker2dGymEnvPool",
  "GymWalker2dGymnasiumEnvPool",
]
