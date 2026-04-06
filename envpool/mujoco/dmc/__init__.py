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
"""Mujoco dm_control suite env in EnvPool."""

from envpool.mujoco.mujoco_dmc_envpool import (
    _DmcAcrobotEnvPool,
    _DmcAcrobotEnvSpec,
    _DmcAcrobotPixelEnvPool,
    _DmcAcrobotPixelEnvSpec,
    _DmcBallInCupEnvPool,
    _DmcBallInCupEnvSpec,
    _DmcBallInCupPixelEnvPool,
    _DmcBallInCupPixelEnvSpec,
    _DmcCartpoleEnvPool,
    _DmcCartpoleEnvSpec,
    _DmcCartpolePixelEnvPool,
    _DmcCartpolePixelEnvSpec,
    _DmcCheetahEnvPool,
    _DmcCheetahEnvSpec,
    _DmcCheetahPixelEnvPool,
    _DmcCheetahPixelEnvSpec,
    _DmcFingerEnvPool,
    _DmcFingerEnvSpec,
    _DmcFingerPixelEnvPool,
    _DmcFingerPixelEnvSpec,
    _DmcFishEnvPool,
    _DmcFishEnvSpec,
    _DmcFishPixelEnvPool,
    _DmcFishPixelEnvSpec,
    _DmcHopperEnvPool,
    _DmcHopperEnvSpec,
    _DmcHopperPixelEnvPool,
    _DmcHopperPixelEnvSpec,
    _DmcHumanoidCMUEnvPool,
    _DmcHumanoidCMUEnvSpec,
    _DmcHumanoidCMUPixelEnvPool,
    _DmcHumanoidCMUPixelEnvSpec,
    _DmcHumanoidEnvPool,
    _DmcHumanoidEnvSpec,
    _DmcHumanoidPixelEnvPool,
    _DmcHumanoidPixelEnvSpec,
    _DmcManipulatorEnvPool,
    _DmcManipulatorEnvSpec,
    _DmcManipulatorPixelEnvPool,
    _DmcManipulatorPixelEnvSpec,
    _DmcPendulumEnvPool,
    _DmcPendulumEnvSpec,
    _DmcPendulumPixelEnvPool,
    _DmcPendulumPixelEnvSpec,
    _DmcPointMassEnvPool,
    _DmcPointMassEnvSpec,
    _DmcPointMassPixelEnvPool,
    _DmcPointMassPixelEnvSpec,
    _DmcReacherEnvPool,
    _DmcReacherEnvSpec,
    _DmcReacherPixelEnvPool,
    _DmcReacherPixelEnvSpec,
    _DmcSwimmerEnvPool,
    _DmcSwimmerEnvSpec,
    _DmcSwimmerPixelEnvPool,
    _DmcSwimmerPixelEnvSpec,
    _DmcWalkerEnvPool,
    _DmcWalkerEnvSpec,
    _DmcWalkerPixelEnvPool,
    _DmcWalkerPixelEnvSpec,
)

from envpool.python.api import py_env

(
    DmcAcrobotEnvSpec,
    DmcAcrobotDMEnvPool,
    DmcAcrobotGymEnvPool,
    DmcAcrobotGymnasiumEnvPool,
) = py_env(_DmcAcrobotEnvSpec, _DmcAcrobotEnvPool)
(
    DmcAcrobotPixelEnvSpec,
    DmcAcrobotPixelDMEnvPool,
    DmcAcrobotPixelGymEnvPool,
    DmcAcrobotPixelGymnasiumEnvPool,
) = py_env(_DmcAcrobotPixelEnvSpec, _DmcAcrobotPixelEnvPool)
(
    DmcBallInCupEnvSpec,
    DmcBallInCupDMEnvPool,
    DmcBallInCupGymEnvPool,
    DmcBallInCupGymnasiumEnvPool,
) = py_env(_DmcBallInCupEnvSpec, _DmcBallInCupEnvPool)
(
    DmcBallInCupPixelEnvSpec,
    DmcBallInCupPixelDMEnvPool,
    DmcBallInCupPixelGymEnvPool,
    DmcBallInCupPixelGymnasiumEnvPool,
) = py_env(_DmcBallInCupPixelEnvSpec, _DmcBallInCupPixelEnvPool)
(
    DmcCartpoleEnvSpec,
    DmcCartpoleDMEnvPool,
    DmcCartpoleGymEnvPool,
    DmcCartpoleGymnasiumEnvPool,
) = py_env(_DmcCartpoleEnvSpec, _DmcCartpoleEnvPool)
(
    DmcCartpolePixelEnvSpec,
    DmcCartpolePixelDMEnvPool,
    DmcCartpolePixelGymEnvPool,
    DmcCartpolePixelGymnasiumEnvPool,
) = py_env(_DmcCartpolePixelEnvSpec, _DmcCartpolePixelEnvPool)
(
    DmcCheetahEnvSpec,
    DmcCheetahDMEnvPool,
    DmcCheetahGymEnvPool,
    DmcCheetahGymnasiumEnvPool,
) = py_env(_DmcCheetahEnvSpec, _DmcCheetahEnvPool)
(
    DmcCheetahPixelEnvSpec,
    DmcCheetahPixelDMEnvPool,
    DmcCheetahPixelGymEnvPool,
    DmcCheetahPixelGymnasiumEnvPool,
) = py_env(_DmcCheetahPixelEnvSpec, _DmcCheetahPixelEnvPool)
(
    DmcFingerEnvSpec,
    DmcFingerDMEnvPool,
    DmcFingerGymEnvPool,
    DmcFingerGymnasiumEnvPool,
) = py_env(_DmcFingerEnvSpec, _DmcFingerEnvPool)
(
    DmcFingerPixelEnvSpec,
    DmcFingerPixelDMEnvPool,
    DmcFingerPixelGymEnvPool,
    DmcFingerPixelGymnasiumEnvPool,
) = py_env(_DmcFingerPixelEnvSpec, _DmcFingerPixelEnvPool)
(
    DmcFishEnvSpec,
    DmcFishDMEnvPool,
    DmcFishGymEnvPool,
    DmcFishGymnasiumEnvPool,
) = py_env(_DmcFishEnvSpec, _DmcFishEnvPool)
(
    DmcFishPixelEnvSpec,
    DmcFishPixelDMEnvPool,
    DmcFishPixelGymEnvPool,
    DmcFishPixelGymnasiumEnvPool,
) = py_env(_DmcFishPixelEnvSpec, _DmcFishPixelEnvPool)
(
    DmcHopperEnvSpec,
    DmcHopperDMEnvPool,
    DmcHopperGymEnvPool,
    DmcHopperGymnasiumEnvPool,
) = py_env(_DmcHopperEnvSpec, _DmcHopperEnvPool)
(
    DmcHopperPixelEnvSpec,
    DmcHopperPixelDMEnvPool,
    DmcHopperPixelGymEnvPool,
    DmcHopperPixelGymnasiumEnvPool,
) = py_env(_DmcHopperPixelEnvSpec, _DmcHopperPixelEnvPool)
(
    DmcHumanoidEnvSpec,
    DmcHumanoidDMEnvPool,
    DmcHumanoidGymEnvPool,
    DmcHumanoidGymnasiumEnvPool,
) = py_env(_DmcHumanoidEnvSpec, _DmcHumanoidEnvPool)
(
    DmcHumanoidPixelEnvSpec,
    DmcHumanoidPixelDMEnvPool,
    DmcHumanoidPixelGymEnvPool,
    DmcHumanoidPixelGymnasiumEnvPool,
) = py_env(_DmcHumanoidPixelEnvSpec, _DmcHumanoidPixelEnvPool)
(
    DmcHumanoidCMUEnvSpec,
    DmcHumanoidCMUDMEnvPool,
    DmcHumanoidCMUGymEnvPool,
    DmcHumanoidCMUGymnasiumEnvPool,
) = py_env(_DmcHumanoidCMUEnvSpec, _DmcHumanoidCMUEnvPool)
(
    DmcHumanoidCMUPixelEnvSpec,
    DmcHumanoidCMUPixelDMEnvPool,
    DmcHumanoidCMUPixelGymEnvPool,
    DmcHumanoidCMUPixelGymnasiumEnvPool,
) = py_env(_DmcHumanoidCMUPixelEnvSpec, _DmcHumanoidCMUPixelEnvPool)
(
    DmcManipulatorEnvSpec,
    DmcManipulatorDMEnvPool,
    DmcManipulatorGymEnvPool,
    DmcManipulatorGymnasiumEnvPool,
) = py_env(_DmcManipulatorEnvSpec, _DmcManipulatorEnvPool)
(
    DmcManipulatorPixelEnvSpec,
    DmcManipulatorPixelDMEnvPool,
    DmcManipulatorPixelGymEnvPool,
    DmcManipulatorPixelGymnasiumEnvPool,
) = py_env(_DmcManipulatorPixelEnvSpec, _DmcManipulatorPixelEnvPool)
(
    DmcPendulumEnvSpec,
    DmcPendulumDMEnvPool,
    DmcPendulumGymEnvPool,
    DmcPendulumGymnasiumEnvPool,
) = py_env(_DmcPendulumEnvSpec, _DmcPendulumEnvPool)
(
    DmcPendulumPixelEnvSpec,
    DmcPendulumPixelDMEnvPool,
    DmcPendulumPixelGymEnvPool,
    DmcPendulumPixelGymnasiumEnvPool,
) = py_env(_DmcPendulumPixelEnvSpec, _DmcPendulumPixelEnvPool)
(
    DmcPointMassEnvSpec,
    DmcPointMassDMEnvPool,
    DmcPointMassGymEnvPool,
    DmcPointMassGymnasiumEnvPool,
) = py_env(_DmcPointMassEnvSpec, _DmcPointMassEnvPool)
(
    DmcPointMassPixelEnvSpec,
    DmcPointMassPixelDMEnvPool,
    DmcPointMassPixelGymEnvPool,
    DmcPointMassPixelGymnasiumEnvPool,
) = py_env(_DmcPointMassPixelEnvSpec, _DmcPointMassPixelEnvPool)
(
    DmcReacherEnvSpec,
    DmcReacherDMEnvPool,
    DmcReacherGymEnvPool,
    DmcReacherGymnasiumEnvPool,
) = py_env(_DmcReacherEnvSpec, _DmcReacherEnvPool)
(
    DmcReacherPixelEnvSpec,
    DmcReacherPixelDMEnvPool,
    DmcReacherPixelGymEnvPool,
    DmcReacherPixelGymnasiumEnvPool,
) = py_env(_DmcReacherPixelEnvSpec, _DmcReacherPixelEnvPool)
(
    DmcSwimmerEnvSpec,
    DmcSwimmerDMEnvPool,
    DmcSwimmerGymEnvPool,
    DmcSwimmerGymnasiumEnvPool,
) = py_env(_DmcSwimmerEnvSpec, _DmcSwimmerEnvPool)
(
    DmcSwimmerPixelEnvSpec,
    DmcSwimmerPixelDMEnvPool,
    DmcSwimmerPixelGymEnvPool,
    DmcSwimmerPixelGymnasiumEnvPool,
) = py_env(_DmcSwimmerPixelEnvSpec, _DmcSwimmerPixelEnvPool)
(
    DmcWalkerEnvSpec,
    DmcWalkerDMEnvPool,
    DmcWalkerGymEnvPool,
    DmcWalkerGymnasiumEnvPool,
) = py_env(_DmcWalkerEnvSpec, _DmcWalkerEnvPool)
(
    DmcWalkerPixelEnvSpec,
    DmcWalkerPixelDMEnvPool,
    DmcWalkerPixelGymEnvPool,
    DmcWalkerPixelGymnasiumEnvPool,
) = py_env(_DmcWalkerPixelEnvSpec, _DmcWalkerPixelEnvPool)

__all__ = [
    "DmcAcrobotEnvSpec",
    "DmcAcrobotDMEnvPool",
    "DmcAcrobotGymEnvPool",
    "DmcAcrobotGymnasiumEnvPool",
    "DmcAcrobotPixelEnvSpec",
    "DmcAcrobotPixelDMEnvPool",
    "DmcAcrobotPixelGymEnvPool",
    "DmcAcrobotPixelGymnasiumEnvPool",
    "DmcBallInCupEnvSpec",
    "DmcBallInCupDMEnvPool",
    "DmcBallInCupGymEnvPool",
    "DmcBallInCupGymnasiumEnvPool",
    "DmcBallInCupPixelEnvSpec",
    "DmcBallInCupPixelDMEnvPool",
    "DmcBallInCupPixelGymEnvPool",
    "DmcBallInCupPixelGymnasiumEnvPool",
    "DmcCartpoleEnvSpec",
    "DmcCartpoleDMEnvPool",
    "DmcCartpoleGymEnvPool",
    "DmcCartpoleGymnasiumEnvPool",
    "DmcCartpolePixelEnvSpec",
    "DmcCartpolePixelDMEnvPool",
    "DmcCartpolePixelGymEnvPool",
    "DmcCartpolePixelGymnasiumEnvPool",
    "DmcCheetahEnvSpec",
    "DmcCheetahDMEnvPool",
    "DmcCheetahGymEnvPool",
    "DmcCheetahGymnasiumEnvPool",
    "DmcCheetahPixelEnvSpec",
    "DmcCheetahPixelDMEnvPool",
    "DmcCheetahPixelGymEnvPool",
    "DmcCheetahPixelGymnasiumEnvPool",
    "DmcFingerEnvSpec",
    "DmcFingerDMEnvPool",
    "DmcFingerGymEnvPool",
    "DmcFingerGymnasiumEnvPool",
    "DmcFingerPixelEnvSpec",
    "DmcFingerPixelDMEnvPool",
    "DmcFingerPixelGymEnvPool",
    "DmcFingerPixelGymnasiumEnvPool",
    "DmcFishEnvSpec",
    "DmcFishDMEnvPool",
    "DmcFishGymEnvPool",
    "DmcFishGymnasiumEnvPool",
    "DmcFishPixelEnvSpec",
    "DmcFishPixelDMEnvPool",
    "DmcFishPixelGymEnvPool",
    "DmcFishPixelGymnasiumEnvPool",
    "DmcHopperEnvSpec",
    "DmcHopperDMEnvPool",
    "DmcHopperGymEnvPool",
    "DmcHopperGymnasiumEnvPool",
    "DmcHopperPixelEnvSpec",
    "DmcHopperPixelDMEnvPool",
    "DmcHopperPixelGymEnvPool",
    "DmcHopperPixelGymnasiumEnvPool",
    "DmcHumanoidEnvSpec",
    "DmcHumanoidDMEnvPool",
    "DmcHumanoidGymEnvPool",
    "DmcHumanoidGymnasiumEnvPool",
    "DmcHumanoidPixelEnvSpec",
    "DmcHumanoidPixelDMEnvPool",
    "DmcHumanoidPixelGymEnvPool",
    "DmcHumanoidPixelGymnasiumEnvPool",
    "DmcHumanoidCMUEnvSpec",
    "DmcHumanoidCMUDMEnvPool",
    "DmcHumanoidCMUGymEnvPool",
    "DmcHumanoidCMUGymnasiumEnvPool",
    "DmcHumanoidCMUPixelEnvSpec",
    "DmcHumanoidCMUPixelDMEnvPool",
    "DmcHumanoidCMUPixelGymEnvPool",
    "DmcHumanoidCMUPixelGymnasiumEnvPool",
    "DmcManipulatorEnvSpec",
    "DmcManipulatorDMEnvPool",
    "DmcManipulatorGymEnvPool",
    "DmcManipulatorGymnasiumEnvPool",
    "DmcManipulatorPixelEnvSpec",
    "DmcManipulatorPixelDMEnvPool",
    "DmcManipulatorPixelGymEnvPool",
    "DmcManipulatorPixelGymnasiumEnvPool",
    "DmcPendulumEnvSpec",
    "DmcPendulumDMEnvPool",
    "DmcPendulumGymEnvPool",
    "DmcPendulumGymnasiumEnvPool",
    "DmcPendulumPixelEnvSpec",
    "DmcPendulumPixelDMEnvPool",
    "DmcPendulumPixelGymEnvPool",
    "DmcPendulumPixelGymnasiumEnvPool",
    "DmcPointMassEnvSpec",
    "DmcPointMassDMEnvPool",
    "DmcPointMassGymEnvPool",
    "DmcPointMassGymnasiumEnvPool",
    "DmcPointMassPixelEnvSpec",
    "DmcPointMassPixelDMEnvPool",
    "DmcPointMassPixelGymEnvPool",
    "DmcPointMassPixelGymnasiumEnvPool",
    "DmcReacherEnvSpec",
    "DmcReacherDMEnvPool",
    "DmcReacherGymEnvPool",
    "DmcReacherGymnasiumEnvPool",
    "DmcReacherPixelEnvSpec",
    "DmcReacherPixelDMEnvPool",
    "DmcReacherPixelGymEnvPool",
    "DmcReacherPixelGymnasiumEnvPool",
    "DmcSwimmerEnvSpec",
    "DmcSwimmerDMEnvPool",
    "DmcSwimmerGymEnvPool",
    "DmcSwimmerGymnasiumEnvPool",
    "DmcSwimmerPixelEnvSpec",
    "DmcSwimmerPixelDMEnvPool",
    "DmcSwimmerPixelGymEnvPool",
    "DmcSwimmerPixelGymnasiumEnvPool",
    "DmcWalkerEnvSpec",
    "DmcWalkerDMEnvPool",
    "DmcWalkerGymEnvPool",
    "DmcWalkerGymnasiumEnvPool",
    "DmcWalkerPixelEnvSpec",
    "DmcWalkerPixelDMEnvPool",
    "DmcWalkerPixelGymEnvPool",
    "DmcWalkerPixelGymnasiumEnvPool",
]
