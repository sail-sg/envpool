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
    _GymAntPixelEnvPool,
    _GymAntPixelEnvSpec,
    _GymHalfCheetahEnvPool,
    _GymHalfCheetahEnvSpec,
    _GymHalfCheetahPixelEnvPool,
    _GymHalfCheetahPixelEnvSpec,
    _GymHopperEnvPool,
    _GymHopperEnvSpec,
    _GymHopperPixelEnvPool,
    _GymHopperPixelEnvSpec,
    _GymHumanoidEnvPool,
    _GymHumanoidEnvSpec,
    _GymHumanoidPixelEnvPool,
    _GymHumanoidPixelEnvSpec,
    _GymHumanoidStandupEnvPool,
    _GymHumanoidStandupEnvSpec,
    _GymHumanoidStandupPixelEnvPool,
    _GymHumanoidStandupPixelEnvSpec,
    _GymInvertedDoublePendulumEnvPool,
    _GymInvertedDoublePendulumEnvSpec,
    _GymInvertedDoublePendulumPixelEnvPool,
    _GymInvertedDoublePendulumPixelEnvSpec,
    _GymInvertedPendulumEnvPool,
    _GymInvertedPendulumEnvSpec,
    _GymInvertedPendulumPixelEnvPool,
    _GymInvertedPendulumPixelEnvSpec,
    _GymPusherEnvPool,
    _GymPusherEnvSpec,
    _GymPusherPixelEnvPool,
    _GymPusherPixelEnvSpec,
    _GymReacherEnvPool,
    _GymReacherEnvSpec,
    _GymReacherPixelEnvPool,
    _GymReacherPixelEnvSpec,
    _GymSwimmerEnvPool,
    _GymSwimmerEnvSpec,
    _GymSwimmerPixelEnvPool,
    _GymSwimmerPixelEnvSpec,
    _GymWalker2dEnvPool,
    _GymWalker2dEnvSpec,
    _GymWalker2dPixelEnvPool,
    _GymWalker2dPixelEnvSpec,
)

from envpool.python.api import py_env

(
    GymAntEnvSpec,
    GymAntDMEnvPool,
    GymAntGymEnvPool,
    GymAntGymnasiumEnvPool,
) = py_env(_GymAntEnvSpec, _GymAntEnvPool)
(
    GymAntPixelEnvSpec,
    GymAntPixelDMEnvPool,
    GymAntPixelGymEnvPool,
    GymAntPixelGymnasiumEnvPool,
) = py_env(_GymAntPixelEnvSpec, _GymAntPixelEnvPool)
(
    GymHalfCheetahEnvSpec,
    GymHalfCheetahDMEnvPool,
    GymHalfCheetahGymEnvPool,
    GymHalfCheetahGymnasiumEnvPool,
) = py_env(_GymHalfCheetahEnvSpec, _GymHalfCheetahEnvPool)
(
    GymHalfCheetahPixelEnvSpec,
    GymHalfCheetahPixelDMEnvPool,
    GymHalfCheetahPixelGymEnvPool,
    GymHalfCheetahPixelGymnasiumEnvPool,
) = py_env(_GymHalfCheetahPixelEnvSpec, _GymHalfCheetahPixelEnvPool)
(
    GymHopperEnvSpec,
    GymHopperDMEnvPool,
    GymHopperGymEnvPool,
    GymHopperGymnasiumEnvPool,
) = py_env(_GymHopperEnvSpec, _GymHopperEnvPool)
(
    GymHopperPixelEnvSpec,
    GymHopperPixelDMEnvPool,
    GymHopperPixelGymEnvPool,
    GymHopperPixelGymnasiumEnvPool,
) = py_env(_GymHopperPixelEnvSpec, _GymHopperPixelEnvPool)
(
    GymHumanoidEnvSpec,
    GymHumanoidDMEnvPool,
    GymHumanoidGymEnvPool,
    GymHumanoidGymnasiumEnvPool,
) = py_env(_GymHumanoidEnvSpec, _GymHumanoidEnvPool)
(
    GymHumanoidPixelEnvSpec,
    GymHumanoidPixelDMEnvPool,
    GymHumanoidPixelGymEnvPool,
    GymHumanoidPixelGymnasiumEnvPool,
) = py_env(_GymHumanoidPixelEnvSpec, _GymHumanoidPixelEnvPool)
(
    GymHumanoidStandupEnvSpec,
    GymHumanoidStandupDMEnvPool,
    GymHumanoidStandupGymEnvPool,
    GymHumanoidStandupGymnasiumEnvPool,
) = py_env(_GymHumanoidStandupEnvSpec, _GymHumanoidStandupEnvPool)
(
    GymHumanoidStandupPixelEnvSpec,
    GymHumanoidStandupPixelDMEnvPool,
    GymHumanoidStandupPixelGymEnvPool,
    GymHumanoidStandupPixelGymnasiumEnvPool,
) = py_env(_GymHumanoidStandupPixelEnvSpec, _GymHumanoidStandupPixelEnvPool)
(
    GymInvertedDoublePendulumEnvSpec,
    GymInvertedDoublePendulumDMEnvPool,
    GymInvertedDoublePendulumGymEnvPool,
    GymInvertedDoublePendulumGymnasiumEnvPool,
) = py_env(_GymInvertedDoublePendulumEnvSpec, _GymInvertedDoublePendulumEnvPool)
(
    GymInvertedDoublePendulumPixelEnvSpec,
    GymInvertedDoublePendulumPixelDMEnvPool,
    GymInvertedDoublePendulumPixelGymEnvPool,
    GymInvertedDoublePendulumPixelGymnasiumEnvPool,
) = py_env(_GymInvertedDoublePendulumPixelEnvSpec, _GymInvertedDoublePendulumPixelEnvPool)
(
    GymInvertedPendulumEnvSpec,
    GymInvertedPendulumDMEnvPool,
    GymInvertedPendulumGymEnvPool,
    GymInvertedPendulumGymnasiumEnvPool,
) = py_env(_GymInvertedPendulumEnvSpec, _GymInvertedPendulumEnvPool)
(
    GymInvertedPendulumPixelEnvSpec,
    GymInvertedPendulumPixelDMEnvPool,
    GymInvertedPendulumPixelGymEnvPool,
    GymInvertedPendulumPixelGymnasiumEnvPool,
) = py_env(_GymInvertedPendulumPixelEnvSpec, _GymInvertedPendulumPixelEnvPool)
(
    GymPusherEnvSpec,
    GymPusherDMEnvPool,
    GymPusherGymEnvPool,
    GymPusherGymnasiumEnvPool,
) = py_env(_GymPusherEnvSpec, _GymPusherEnvPool)
(
    GymPusherPixelEnvSpec,
    GymPusherPixelDMEnvPool,
    GymPusherPixelGymEnvPool,
    GymPusherPixelGymnasiumEnvPool,
) = py_env(_GymPusherPixelEnvSpec, _GymPusherPixelEnvPool)
(
    GymReacherEnvSpec,
    GymReacherDMEnvPool,
    GymReacherGymEnvPool,
    GymReacherGymnasiumEnvPool,
) = py_env(_GymReacherEnvSpec, _GymReacherEnvPool)
(
    GymReacherPixelEnvSpec,
    GymReacherPixelDMEnvPool,
    GymReacherPixelGymEnvPool,
    GymReacherPixelGymnasiumEnvPool,
) = py_env(_GymReacherPixelEnvSpec, _GymReacherPixelEnvPool)
(
    GymSwimmerEnvSpec,
    GymSwimmerDMEnvPool,
    GymSwimmerGymEnvPool,
    GymSwimmerGymnasiumEnvPool,
) = py_env(_GymSwimmerEnvSpec, _GymSwimmerEnvPool)
(
    GymSwimmerPixelEnvSpec,
    GymSwimmerPixelDMEnvPool,
    GymSwimmerPixelGymEnvPool,
    GymSwimmerPixelGymnasiumEnvPool,
) = py_env(_GymSwimmerPixelEnvSpec, _GymSwimmerPixelEnvPool)
(
    GymWalker2dEnvSpec,
    GymWalker2dDMEnvPool,
    GymWalker2dGymEnvPool,
    GymWalker2dGymnasiumEnvPool,
) = py_env(_GymWalker2dEnvSpec, _GymWalker2dEnvPool)
(
    GymWalker2dPixelEnvSpec,
    GymWalker2dPixelDMEnvPool,
    GymWalker2dPixelGymEnvPool,
    GymWalker2dPixelGymnasiumEnvPool,
) = py_env(_GymWalker2dPixelEnvSpec, _GymWalker2dPixelEnvPool)

__all__ = [
    "GymAntEnvSpec",
    "GymAntDMEnvPool",
    "GymAntGymEnvPool",
    "GymAntGymnasiumEnvPool",
    "GymAntPixelEnvSpec",
    "GymAntPixelDMEnvPool",
    "GymAntPixelGymEnvPool",
    "GymAntPixelGymnasiumEnvPool",
    "GymHalfCheetahEnvSpec",
    "GymHalfCheetahDMEnvPool",
    "GymHalfCheetahGymEnvPool",
    "GymHalfCheetahGymnasiumEnvPool",
    "GymHalfCheetahPixelEnvSpec",
    "GymHalfCheetahPixelDMEnvPool",
    "GymHalfCheetahPixelGymEnvPool",
    "GymHalfCheetahPixelGymnasiumEnvPool",
    "GymHopperEnvSpec",
    "GymHopperDMEnvPool",
    "GymHopperGymEnvPool",
    "GymHopperGymnasiumEnvPool",
    "GymHopperPixelEnvSpec",
    "GymHopperPixelDMEnvPool",
    "GymHopperPixelGymEnvPool",
    "GymHopperPixelGymnasiumEnvPool",
    "GymHumanoidEnvSpec",
    "GymHumanoidDMEnvPool",
    "GymHumanoidGymEnvPool",
    "GymHumanoidGymnasiumEnvPool",
    "GymHumanoidPixelEnvSpec",
    "GymHumanoidPixelDMEnvPool",
    "GymHumanoidPixelGymEnvPool",
    "GymHumanoidPixelGymnasiumEnvPool",
    "GymHumanoidStandupEnvSpec",
    "GymHumanoidStandupDMEnvPool",
    "GymHumanoidStandupGymEnvPool",
    "GymHumanoidStandupGymnasiumEnvPool",
    "GymHumanoidStandupPixelEnvSpec",
    "GymHumanoidStandupPixelDMEnvPool",
    "GymHumanoidStandupPixelGymEnvPool",
    "GymHumanoidStandupPixelGymnasiumEnvPool",
    "GymInvertedDoublePendulumEnvSpec",
    "GymInvertedDoublePendulumDMEnvPool",
    "GymInvertedDoublePendulumGymEnvPool",
    "GymInvertedDoublePendulumGymnasiumEnvPool",
    "GymInvertedDoublePendulumPixelEnvSpec",
    "GymInvertedDoublePendulumPixelDMEnvPool",
    "GymInvertedDoublePendulumPixelGymEnvPool",
    "GymInvertedDoublePendulumPixelGymnasiumEnvPool",
    "GymInvertedPendulumEnvSpec",
    "GymInvertedPendulumDMEnvPool",
    "GymInvertedPendulumGymEnvPool",
    "GymInvertedPendulumGymnasiumEnvPool",
    "GymInvertedPendulumPixelEnvSpec",
    "GymInvertedPendulumPixelDMEnvPool",
    "GymInvertedPendulumPixelGymEnvPool",
    "GymInvertedPendulumPixelGymnasiumEnvPool",
    "GymPusherEnvSpec",
    "GymPusherDMEnvPool",
    "GymPusherGymEnvPool",
    "GymPusherGymnasiumEnvPool",
    "GymPusherPixelEnvSpec",
    "GymPusherPixelDMEnvPool",
    "GymPusherPixelGymEnvPool",
    "GymPusherPixelGymnasiumEnvPool",
    "GymReacherEnvSpec",
    "GymReacherDMEnvPool",
    "GymReacherGymEnvPool",
    "GymReacherGymnasiumEnvPool",
    "GymReacherPixelEnvSpec",
    "GymReacherPixelDMEnvPool",
    "GymReacherPixelGymEnvPool",
    "GymReacherPixelGymnasiumEnvPool",
    "GymSwimmerEnvSpec",
    "GymSwimmerDMEnvPool",
    "GymSwimmerGymEnvPool",
    "GymSwimmerGymnasiumEnvPool",
    "GymSwimmerPixelEnvSpec",
    "GymSwimmerPixelDMEnvPool",
    "GymSwimmerPixelGymEnvPool",
    "GymSwimmerPixelGymnasiumEnvPool",
    "GymWalker2dEnvSpec",
    "GymWalker2dDMEnvPool",
    "GymWalker2dGymEnvPool",
    "GymWalker2dGymnasiumEnvPool",
    "GymWalker2dPixelEnvSpec",
    "GymWalker2dPixelDMEnvPool",
    "GymWalker2dPixelGymEnvPool",
    "GymWalker2dPixelGymnasiumEnvPool",
]
