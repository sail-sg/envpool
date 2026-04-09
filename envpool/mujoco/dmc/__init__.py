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
    _DmcDogEnvPool,
    _DmcDogEnvSpec,
    _DmcDogPixelEnvPool,
    _DmcDogPixelEnvSpec,
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
    _DmcLqrEnvPool,
    _DmcLqrEnvSpec,
    _DmcLqrPixelEnvPool,
    _DmcLqrPixelEnvSpec,
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
    _DmcQuadrupedEnvPool,
    _DmcQuadrupedEnvSpec,
    _DmcQuadrupedPixelEnvPool,
    _DmcQuadrupedPixelEnvSpec,
    _DmcReacherEnvPool,
    _DmcReacherEnvSpec,
    _DmcReacherPixelEnvPool,
    _DmcReacherPixelEnvSpec,
    _DmcStackerEnvPool,
    _DmcStackerEnvSpec,
    _DmcStackerPixelEnvPool,
    _DmcStackerPixelEnvSpec,
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
    DmcAcrobotGymnasiumEnvPool,
) = py_env(_DmcAcrobotEnvSpec, _DmcAcrobotEnvPool)
(
    DmcAcrobotPixelEnvSpec,
    DmcAcrobotPixelDMEnvPool,
    DmcAcrobotPixelGymnasiumEnvPool,
) = py_env(_DmcAcrobotPixelEnvSpec, _DmcAcrobotPixelEnvPool)
(
    DmcBallInCupEnvSpec,
    DmcBallInCupDMEnvPool,
    DmcBallInCupGymnasiumEnvPool,
) = py_env(_DmcBallInCupEnvSpec, _DmcBallInCupEnvPool)
(
    DmcBallInCupPixelEnvSpec,
    DmcBallInCupPixelDMEnvPool,
    DmcBallInCupPixelGymnasiumEnvPool,
) = py_env(_DmcBallInCupPixelEnvSpec, _DmcBallInCupPixelEnvPool)
(
    DmcCartpoleEnvSpec,
    DmcCartpoleDMEnvPool,
    DmcCartpoleGymnasiumEnvPool,
) = py_env(_DmcCartpoleEnvSpec, _DmcCartpoleEnvPool)
(
    DmcCartpolePixelEnvSpec,
    DmcCartpolePixelDMEnvPool,
    DmcCartpolePixelGymnasiumEnvPool,
) = py_env(_DmcCartpolePixelEnvSpec, _DmcCartpolePixelEnvPool)
(
    DmcCheetahEnvSpec,
    DmcCheetahDMEnvPool,
    DmcCheetahGymnasiumEnvPool,
) = py_env(_DmcCheetahEnvSpec, _DmcCheetahEnvPool)
(
    DmcCheetahPixelEnvSpec,
    DmcCheetahPixelDMEnvPool,
    DmcCheetahPixelGymnasiumEnvPool,
) = py_env(_DmcCheetahPixelEnvSpec, _DmcCheetahPixelEnvPool)
(
    DmcDogEnvSpec,
    DmcDogDMEnvPool,
    DmcDogGymnasiumEnvPool,
) = py_env(_DmcDogEnvSpec, _DmcDogEnvPool)
(
    DmcDogPixelEnvSpec,
    DmcDogPixelDMEnvPool,
    DmcDogPixelGymnasiumEnvPool,
) = py_env(_DmcDogPixelEnvSpec, _DmcDogPixelEnvPool)
(
    DmcFingerEnvSpec,
    DmcFingerDMEnvPool,
    DmcFingerGymnasiumEnvPool,
) = py_env(_DmcFingerEnvSpec, _DmcFingerEnvPool)
(
    DmcFingerPixelEnvSpec,
    DmcFingerPixelDMEnvPool,
    DmcFingerPixelGymnasiumEnvPool,
) = py_env(_DmcFingerPixelEnvSpec, _DmcFingerPixelEnvPool)
(
    DmcFishEnvSpec,
    DmcFishDMEnvPool,
    DmcFishGymnasiumEnvPool,
) = py_env(_DmcFishEnvSpec, _DmcFishEnvPool)
(
    DmcFishPixelEnvSpec,
    DmcFishPixelDMEnvPool,
    DmcFishPixelGymnasiumEnvPool,
) = py_env(_DmcFishPixelEnvSpec, _DmcFishPixelEnvPool)
(
    DmcHopperEnvSpec,
    DmcHopperDMEnvPool,
    DmcHopperGymnasiumEnvPool,
) = py_env(_DmcHopperEnvSpec, _DmcHopperEnvPool)
(
    DmcHopperPixelEnvSpec,
    DmcHopperPixelDMEnvPool,
    DmcHopperPixelGymnasiumEnvPool,
) = py_env(_DmcHopperPixelEnvSpec, _DmcHopperPixelEnvPool)
(
    DmcHumanoidEnvSpec,
    DmcHumanoidDMEnvPool,
    DmcHumanoidGymnasiumEnvPool,
) = py_env(_DmcHumanoidEnvSpec, _DmcHumanoidEnvPool)
(
    DmcHumanoidPixelEnvSpec,
    DmcHumanoidPixelDMEnvPool,
    DmcHumanoidPixelGymnasiumEnvPool,
) = py_env(_DmcHumanoidPixelEnvSpec, _DmcHumanoidPixelEnvPool)
(
    DmcHumanoidCMUEnvSpec,
    DmcHumanoidCMUDMEnvPool,
    DmcHumanoidCMUGymnasiumEnvPool,
) = py_env(_DmcHumanoidCMUEnvSpec, _DmcHumanoidCMUEnvPool)
(
    DmcHumanoidCMUPixelEnvSpec,
    DmcHumanoidCMUPixelDMEnvPool,
    DmcHumanoidCMUPixelGymnasiumEnvPool,
) = py_env(_DmcHumanoidCMUPixelEnvSpec, _DmcHumanoidCMUPixelEnvPool)
(
    DmcLqrEnvSpec,
    DmcLqrDMEnvPool,
    DmcLqrGymnasiumEnvPool,
) = py_env(_DmcLqrEnvSpec, _DmcLqrEnvPool)
(
    DmcLqrPixelEnvSpec,
    DmcLqrPixelDMEnvPool,
    DmcLqrPixelGymnasiumEnvPool,
) = py_env(_DmcLqrPixelEnvSpec, _DmcLqrPixelEnvPool)
(
    DmcManipulatorEnvSpec,
    DmcManipulatorDMEnvPool,
    DmcManipulatorGymnasiumEnvPool,
) = py_env(_DmcManipulatorEnvSpec, _DmcManipulatorEnvPool)
(
    DmcManipulatorPixelEnvSpec,
    DmcManipulatorPixelDMEnvPool,
    DmcManipulatorPixelGymnasiumEnvPool,
) = py_env(_DmcManipulatorPixelEnvSpec, _DmcManipulatorPixelEnvPool)
(
    DmcPendulumEnvSpec,
    DmcPendulumDMEnvPool,
    DmcPendulumGymnasiumEnvPool,
) = py_env(_DmcPendulumEnvSpec, _DmcPendulumEnvPool)
(
    DmcPendulumPixelEnvSpec,
    DmcPendulumPixelDMEnvPool,
    DmcPendulumPixelGymnasiumEnvPool,
) = py_env(_DmcPendulumPixelEnvSpec, _DmcPendulumPixelEnvPool)
(
    DmcPointMassEnvSpec,
    DmcPointMassDMEnvPool,
    DmcPointMassGymnasiumEnvPool,
) = py_env(_DmcPointMassEnvSpec, _DmcPointMassEnvPool)
(
    DmcPointMassPixelEnvSpec,
    DmcPointMassPixelDMEnvPool,
    DmcPointMassPixelGymnasiumEnvPool,
) = py_env(_DmcPointMassPixelEnvSpec, _DmcPointMassPixelEnvPool)
(
    DmcQuadrupedEnvSpec,
    DmcQuadrupedDMEnvPool,
    DmcQuadrupedGymnasiumEnvPool,
) = py_env(_DmcQuadrupedEnvSpec, _DmcQuadrupedEnvPool)
(
    DmcQuadrupedPixelEnvSpec,
    DmcQuadrupedPixelDMEnvPool,
    DmcQuadrupedPixelGymnasiumEnvPool,
) = py_env(_DmcQuadrupedPixelEnvSpec, _DmcQuadrupedPixelEnvPool)
(
    DmcReacherEnvSpec,
    DmcReacherDMEnvPool,
    DmcReacherGymnasiumEnvPool,
) = py_env(_DmcReacherEnvSpec, _DmcReacherEnvPool)
(
    DmcReacherPixelEnvSpec,
    DmcReacherPixelDMEnvPool,
    DmcReacherPixelGymnasiumEnvPool,
) = py_env(_DmcReacherPixelEnvSpec, _DmcReacherPixelEnvPool)
(
    DmcStackerEnvSpec,
    DmcStackerDMEnvPool,
    DmcStackerGymnasiumEnvPool,
) = py_env(_DmcStackerEnvSpec, _DmcStackerEnvPool)
(
    DmcStackerPixelEnvSpec,
    DmcStackerPixelDMEnvPool,
    DmcStackerPixelGymnasiumEnvPool,
) = py_env(_DmcStackerPixelEnvSpec, _DmcStackerPixelEnvPool)
(
    DmcSwimmerEnvSpec,
    DmcSwimmerDMEnvPool,
    DmcSwimmerGymnasiumEnvPool,
) = py_env(_DmcSwimmerEnvSpec, _DmcSwimmerEnvPool)
(
    DmcSwimmerPixelEnvSpec,
    DmcSwimmerPixelDMEnvPool,
    DmcSwimmerPixelGymnasiumEnvPool,
) = py_env(_DmcSwimmerPixelEnvSpec, _DmcSwimmerPixelEnvPool)
(
    DmcWalkerEnvSpec,
    DmcWalkerDMEnvPool,
    DmcWalkerGymnasiumEnvPool,
) = py_env(_DmcWalkerEnvSpec, _DmcWalkerEnvPool)
(
    DmcWalkerPixelEnvSpec,
    DmcWalkerPixelDMEnvPool,
    DmcWalkerPixelGymnasiumEnvPool,
) = py_env(_DmcWalkerPixelEnvSpec, _DmcWalkerPixelEnvPool)

__all__ = [
    "DmcAcrobotEnvSpec",
    "DmcAcrobotDMEnvPool",
    "DmcAcrobotGymnasiumEnvPool",
    "DmcAcrobotPixelEnvSpec",
    "DmcAcrobotPixelDMEnvPool",
    "DmcAcrobotPixelGymnasiumEnvPool",
    "DmcBallInCupEnvSpec",
    "DmcBallInCupDMEnvPool",
    "DmcBallInCupGymnasiumEnvPool",
    "DmcBallInCupPixelEnvSpec",
    "DmcBallInCupPixelDMEnvPool",
    "DmcBallInCupPixelGymnasiumEnvPool",
    "DmcCartpoleEnvSpec",
    "DmcCartpoleDMEnvPool",
    "DmcCartpoleGymnasiumEnvPool",
    "DmcCartpolePixelEnvSpec",
    "DmcCartpolePixelDMEnvPool",
    "DmcCartpolePixelGymnasiumEnvPool",
    "DmcCheetahEnvSpec",
    "DmcCheetahDMEnvPool",
    "DmcCheetahGymnasiumEnvPool",
    "DmcCheetahPixelEnvSpec",
    "DmcCheetahPixelDMEnvPool",
    "DmcCheetahPixelGymnasiumEnvPool",
    "DmcFingerEnvSpec",
    "DmcFingerDMEnvPool",
    "DmcFingerGymnasiumEnvPool",
    "DmcFingerPixelEnvSpec",
    "DmcFingerPixelDMEnvPool",
    "DmcFingerPixelGymnasiumEnvPool",
    "DmcFishEnvSpec",
    "DmcFishDMEnvPool",
    "DmcFishGymnasiumEnvPool",
    "DmcFishPixelEnvSpec",
    "DmcFishPixelDMEnvPool",
    "DmcFishPixelGymnasiumEnvPool",
    "DmcHopperEnvSpec",
    "DmcHopperDMEnvPool",
    "DmcHopperGymnasiumEnvPool",
    "DmcHopperPixelEnvSpec",
    "DmcHopperPixelDMEnvPool",
    "DmcHopperPixelGymnasiumEnvPool",
    "DmcHumanoidEnvSpec",
    "DmcHumanoidDMEnvPool",
    "DmcHumanoidGymnasiumEnvPool",
    "DmcHumanoidPixelEnvSpec",
    "DmcHumanoidPixelDMEnvPool",
    "DmcHumanoidPixelGymnasiumEnvPool",
    "DmcHumanoidCMUEnvSpec",
    "DmcHumanoidCMUDMEnvPool",
    "DmcHumanoidCMUGymnasiumEnvPool",
    "DmcHumanoidCMUPixelEnvSpec",
    "DmcHumanoidCMUPixelDMEnvPool",
    "DmcHumanoidCMUPixelGymnasiumEnvPool",
    "DmcManipulatorEnvSpec",
    "DmcManipulatorDMEnvPool",
    "DmcManipulatorGymnasiumEnvPool",
    "DmcManipulatorPixelEnvSpec",
    "DmcManipulatorPixelDMEnvPool",
    "DmcManipulatorPixelGymnasiumEnvPool",
    "DmcPendulumEnvSpec",
    "DmcPendulumDMEnvPool",
    "DmcPendulumGymnasiumEnvPool",
    "DmcPendulumPixelEnvSpec",
    "DmcPendulumPixelDMEnvPool",
    "DmcPendulumPixelGymnasiumEnvPool",
    "DmcPointMassEnvSpec",
    "DmcPointMassDMEnvPool",
    "DmcPointMassGymnasiumEnvPool",
    "DmcPointMassPixelEnvSpec",
    "DmcPointMassPixelDMEnvPool",
    "DmcPointMassPixelGymnasiumEnvPool",
    "DmcReacherEnvSpec",
    "DmcReacherDMEnvPool",
    "DmcReacherGymnasiumEnvPool",
    "DmcReacherPixelEnvSpec",
    "DmcReacherPixelDMEnvPool",
    "DmcReacherPixelGymnasiumEnvPool",
    "DmcSwimmerEnvSpec",
    "DmcSwimmerDMEnvPool",
    "DmcSwimmerGymnasiumEnvPool",
    "DmcSwimmerPixelEnvSpec",
    "DmcSwimmerPixelDMEnvPool",
    "DmcSwimmerPixelGymnasiumEnvPool",
    "DmcWalkerEnvSpec",
    "DmcWalkerDMEnvPool",
    "DmcWalkerGymnasiumEnvPool",
    "DmcWalkerPixelEnvSpec",
    "DmcWalkerPixelDMEnvPool",
    "DmcWalkerPixelGymnasiumEnvPool",
]
