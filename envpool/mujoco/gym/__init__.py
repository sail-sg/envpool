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
    GymAntGymnasiumEnvPool,
) = py_env(_GymAntEnvSpec, _GymAntEnvPool)
(
    GymAntPixelEnvSpec,
    GymAntPixelDMEnvPool,
    GymAntPixelGymnasiumEnvPool,
) = py_env(_GymAntPixelEnvSpec, _GymAntPixelEnvPool)
(
    GymHalfCheetahEnvSpec,
    GymHalfCheetahDMEnvPool,
    GymHalfCheetahGymnasiumEnvPool,
) = py_env(_GymHalfCheetahEnvSpec, _GymHalfCheetahEnvPool)
(
    GymHalfCheetahPixelEnvSpec,
    GymHalfCheetahPixelDMEnvPool,
    GymHalfCheetahPixelGymnasiumEnvPool,
) = py_env(_GymHalfCheetahPixelEnvSpec, _GymHalfCheetahPixelEnvPool)
(
    GymHopperEnvSpec,
    GymHopperDMEnvPool,
    GymHopperGymnasiumEnvPool,
) = py_env(_GymHopperEnvSpec, _GymHopperEnvPool)
(
    GymHopperPixelEnvSpec,
    GymHopperPixelDMEnvPool,
    GymHopperPixelGymnasiumEnvPool,
) = py_env(_GymHopperPixelEnvSpec, _GymHopperPixelEnvPool)
(
    GymHumanoidEnvSpec,
    GymHumanoidDMEnvPool,
    GymHumanoidGymnasiumEnvPool,
) = py_env(_GymHumanoidEnvSpec, _GymHumanoidEnvPool)
(
    GymHumanoidPixelEnvSpec,
    GymHumanoidPixelDMEnvPool,
    GymHumanoidPixelGymnasiumEnvPool,
) = py_env(_GymHumanoidPixelEnvSpec, _GymHumanoidPixelEnvPool)
(
    GymHumanoidStandupEnvSpec,
    GymHumanoidStandupDMEnvPool,
    GymHumanoidStandupGymnasiumEnvPool,
) = py_env(_GymHumanoidStandupEnvSpec, _GymHumanoidStandupEnvPool)
(
    GymHumanoidStandupPixelEnvSpec,
    GymHumanoidStandupPixelDMEnvPool,
    GymHumanoidStandupPixelGymnasiumEnvPool,
) = py_env(_GymHumanoidStandupPixelEnvSpec, _GymHumanoidStandupPixelEnvPool)
(
    GymInvertedDoublePendulumEnvSpec,
    GymInvertedDoublePendulumDMEnvPool,
    GymInvertedDoublePendulumGymnasiumEnvPool,
) = py_env(_GymInvertedDoublePendulumEnvSpec, _GymInvertedDoublePendulumEnvPool)
(
    GymInvertedDoublePendulumPixelEnvSpec,
    GymInvertedDoublePendulumPixelDMEnvPool,
    GymInvertedDoublePendulumPixelGymnasiumEnvPool,
) = py_env(
    _GymInvertedDoublePendulumPixelEnvSpec,
    _GymInvertedDoublePendulumPixelEnvPool,
)
(
    GymInvertedPendulumEnvSpec,
    GymInvertedPendulumDMEnvPool,
    GymInvertedPendulumGymnasiumEnvPool,
) = py_env(_GymInvertedPendulumEnvSpec, _GymInvertedPendulumEnvPool)
(
    GymInvertedPendulumPixelEnvSpec,
    GymInvertedPendulumPixelDMEnvPool,
    GymInvertedPendulumPixelGymnasiumEnvPool,
) = py_env(_GymInvertedPendulumPixelEnvSpec, _GymInvertedPendulumPixelEnvPool)
(
    GymPusherEnvSpec,
    GymPusherDMEnvPool,
    GymPusherGymnasiumEnvPool,
) = py_env(_GymPusherEnvSpec, _GymPusherEnvPool)
(
    GymPusherPixelEnvSpec,
    GymPusherPixelDMEnvPool,
    GymPusherPixelGymnasiumEnvPool,
) = py_env(_GymPusherPixelEnvSpec, _GymPusherPixelEnvPool)
(
    GymReacherEnvSpec,
    GymReacherDMEnvPool,
    GymReacherGymnasiumEnvPool,
) = py_env(_GymReacherEnvSpec, _GymReacherEnvPool)
(
    GymReacherPixelEnvSpec,
    GymReacherPixelDMEnvPool,
    GymReacherPixelGymnasiumEnvPool,
) = py_env(_GymReacherPixelEnvSpec, _GymReacherPixelEnvPool)
(
    GymSwimmerEnvSpec,
    GymSwimmerDMEnvPool,
    GymSwimmerGymnasiumEnvPool,
) = py_env(_GymSwimmerEnvSpec, _GymSwimmerEnvPool)
(
    GymSwimmerPixelEnvSpec,
    GymSwimmerPixelDMEnvPool,
    GymSwimmerPixelGymnasiumEnvPool,
) = py_env(_GymSwimmerPixelEnvSpec, _GymSwimmerPixelEnvPool)
(
    GymWalker2dEnvSpec,
    GymWalker2dDMEnvPool,
    GymWalker2dGymnasiumEnvPool,
) = py_env(_GymWalker2dEnvSpec, _GymWalker2dEnvPool)
(
    GymWalker2dPixelEnvSpec,
    GymWalker2dPixelDMEnvPool,
    GymWalker2dPixelGymnasiumEnvPool,
) = py_env(_GymWalker2dPixelEnvSpec, _GymWalker2dPixelEnvPool)

__all__ = [
    "GymAntEnvSpec",
    "GymAntDMEnvPool",
    "GymAntGymnasiumEnvPool",
    "GymAntPixelEnvSpec",
    "GymAntPixelDMEnvPool",
    "GymAntPixelGymnasiumEnvPool",
    "GymHalfCheetahEnvSpec",
    "GymHalfCheetahDMEnvPool",
    "GymHalfCheetahGymnasiumEnvPool",
    "GymHalfCheetahPixelEnvSpec",
    "GymHalfCheetahPixelDMEnvPool",
    "GymHalfCheetahPixelGymnasiumEnvPool",
    "GymHopperEnvSpec",
    "GymHopperDMEnvPool",
    "GymHopperGymnasiumEnvPool",
    "GymHopperPixelEnvSpec",
    "GymHopperPixelDMEnvPool",
    "GymHopperPixelGymnasiumEnvPool",
    "GymHumanoidEnvSpec",
    "GymHumanoidDMEnvPool",
    "GymHumanoidGymnasiumEnvPool",
    "GymHumanoidPixelEnvSpec",
    "GymHumanoidPixelDMEnvPool",
    "GymHumanoidPixelGymnasiumEnvPool",
    "GymHumanoidStandupEnvSpec",
    "GymHumanoidStandupDMEnvPool",
    "GymHumanoidStandupGymnasiumEnvPool",
    "GymHumanoidStandupPixelEnvSpec",
    "GymHumanoidStandupPixelDMEnvPool",
    "GymHumanoidStandupPixelGymnasiumEnvPool",
    "GymInvertedDoublePendulumEnvSpec",
    "GymInvertedDoublePendulumDMEnvPool",
    "GymInvertedDoublePendulumGymnasiumEnvPool",
    "GymInvertedDoublePendulumPixelEnvSpec",
    "GymInvertedDoublePendulumPixelDMEnvPool",
    "GymInvertedDoublePendulumPixelGymnasiumEnvPool",
    "GymInvertedPendulumEnvSpec",
    "GymInvertedPendulumDMEnvPool",
    "GymInvertedPendulumGymnasiumEnvPool",
    "GymInvertedPendulumPixelEnvSpec",
    "GymInvertedPendulumPixelDMEnvPool",
    "GymInvertedPendulumPixelGymnasiumEnvPool",
    "GymPusherEnvSpec",
    "GymPusherDMEnvPool",
    "GymPusherGymnasiumEnvPool",
    "GymPusherPixelEnvSpec",
    "GymPusherPixelDMEnvPool",
    "GymPusherPixelGymnasiumEnvPool",
    "GymReacherEnvSpec",
    "GymReacherDMEnvPool",
    "GymReacherGymnasiumEnvPool",
    "GymReacherPixelEnvSpec",
    "GymReacherPixelDMEnvPool",
    "GymReacherPixelGymnasiumEnvPool",
    "GymSwimmerEnvSpec",
    "GymSwimmerDMEnvPool",
    "GymSwimmerGymnasiumEnvPool",
    "GymSwimmerPixelEnvSpec",
    "GymSwimmerPixelDMEnvPool",
    "GymSwimmerPixelGymnasiumEnvPool",
    "GymWalker2dEnvSpec",
    "GymWalker2dDMEnvPool",
    "GymWalker2dGymnasiumEnvPool",
    "GymWalker2dPixelEnvSpec",
    "GymWalker2dPixelDMEnvPool",
    "GymWalker2dPixelGymnasiumEnvPool",
]
