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
  _DmcBallInCupEnvPool,
  _DmcBallInCupEnvSpec,
  _DmcCartpoleEnvPool,
  _DmcCartpoleEnvSpec,
  _DmcCheetahEnvPool,
  _DmcCheetahEnvSpec,
  _DmcFingerEnvPool,
  _DmcFingerEnvSpec,
  _DmcFishEnvPool,
  _DmcFishEnvSpec,
  _DmcHopperEnvPool,
  _DmcHopperEnvSpec,
  _DmcHumanoidCMUEnvPool,
  _DmcHumanoidCMUEnvSpec,
  _DmcHumanoidEnvPool,
  _DmcHumanoidEnvSpec,
  _DmcManipulatorEnvPool,
  _DmcManipulatorEnvSpec,
  _DmcPendulumEnvPool,
  _DmcPendulumEnvSpec,
  _DmcPointMassEnvPool,
  _DmcPointMassEnvSpec,
  _DmcReacherEnvPool,
  _DmcReacherEnvSpec,
  _DmcSwimmerEnvPool,
  _DmcSwimmerEnvSpec,
  _DmcWalkerEnvPool,
  _DmcWalkerEnvSpec,
)
from envpool.python.api import py_env

(
  DmcAcrobotEnvSpec, DmcAcrobotDMEnvPool, DmcAcrobotGymEnvPool,
  DmcAcrobotGymnasiumEnvPool
) = py_env(_DmcAcrobotEnvSpec, _DmcAcrobotEnvPool)
(
  DmcBallInCupEnvSpec, DmcBallInCupDMEnvPool, DmcBallInCupGymEnvPool,
  DmcBallInCupGymnasiumEnvPool
) = py_env(_DmcBallInCupEnvSpec, _DmcBallInCupEnvPool)
(
  DmcCartpoleEnvSpec, DmcCartpoleDMEnvPool, DmcCartpoleGymEnvPool,
  DmcCartpoleGymnasiumEnvPool
) = py_env(_DmcCartpoleEnvSpec, _DmcCartpoleEnvPool)
(
  DmcCheetahEnvSpec, DmcCheetahDMEnvPool, DmcCheetahGymEnvPool,
  DmcCheetahGymnasiumEnvPool
) = py_env(_DmcCheetahEnvSpec, _DmcCheetahEnvPool)
(
  DmcFingerEnvSpec, DmcFingerDMEnvPool, DmcFingerGymEnvPool,
  DmcFingerGymnasiumEnvPool
) = py_env(_DmcFingerEnvSpec, _DmcFingerEnvPool)
(DmcFishEnvSpec, DmcFishDMEnvPool, DmcFishGymEnvPool,
 DmcFishGymnasiumEnvPool) = py_env(_DmcFishEnvSpec, _DmcFishEnvPool)
(
  DmcHopperEnvSpec, DmcHopperDMEnvPool, DmcHopperGymEnvPool,
  DmcHopperGymnasiumEnvPool
) = py_env(_DmcHopperEnvSpec, _DmcHopperEnvPool)
(
  DmcHumanoidEnvSpec, DmcHumanoidDMEnvPool, DmcHumanoidGymEnvPool,
  DmcHumanoidGymnasiumEnvPool
) = py_env(_DmcHumanoidEnvSpec, _DmcHumanoidEnvPool)
(
  DmcHumanoidCMUEnvSpec,
  DmcHumanoidCMUDMEnvPool,
  DmcHumanoidCMUGymEnvPool,
  DmcHumanoidCMUGymnasiumEnvPool,
) = py_env(_DmcHumanoidCMUEnvSpec, _DmcHumanoidCMUEnvPool)
(
  DmcManipulatorEnvSpec,
  DmcManipulatorDMEnvPool,
  DmcManipulatorGymEnvPool,
  DmcManipulatorGymnasiumEnvPool,
) = py_env(_DmcManipulatorEnvSpec, _DmcManipulatorEnvPool)
(
  DmcPendulumEnvSpec, DmcPendulumDMEnvPool, DmcPendulumGymEnvPool,
  DmcPendulumGymnasiumEnvPool
) = py_env(_DmcPendulumEnvSpec, _DmcPendulumEnvPool)
(
  DmcPointMassEnvSpec, DmcPointMassDMEnvPool, DmcPointMassGymEnvPool,
  DmcPointMassGymnasiumEnvPool
) = py_env(_DmcPointMassEnvSpec, _DmcPointMassEnvPool)
(
  DmcReacherEnvSpec, DmcReacherDMEnvPool, DmcReacherGymEnvPool,
  DmcReacherGymnasiumEnvPool
) = py_env(_DmcReacherEnvSpec, _DmcReacherEnvPool)
(
  DmcSwimmerEnvSpec, DmcSwimmerDMEnvPool, DmcSwimmerGymEnvPool,
  DmcSwimmerGymnasiumEnvPool
) = py_env(_DmcSwimmerEnvSpec, _DmcSwimmerEnvPool)
(
  DmcWalkerEnvSpec, DmcWalkerDMEnvPool, DmcWalkerGymEnvPool,
  DmcWalkerGymnasiumEnvPool
) = py_env(_DmcWalkerEnvSpec, _DmcWalkerEnvPool)

__all__ = [
  "DmcAcrobotEnvSpec",
  "DmcAcrobotDMEnvPool",
  "DmcAcrobotGymEnvPool",
  "DmcAcrobotGymnasiumEnvPool",
  "DmcBallInCupEnvSpec",
  "DmcBallInCupDMEnvPool",
  "DmcBallInCupGymEnvPool",
  "DmcBallInCupGymnasiumEnvPool",
  "DmcCartpoleEnvSpec",
  "DmcCartpoleDMEnvPool",
  "DmcCartpoleGymEnvPool",
  "DmcCartpoleGymnasiumEnvPool",
  "DmcCheetahEnvSpec",
  "DmcCheetahDMEnvPool",
  "DmcCheetahGymEnvPool",
  "DmcCheetahGymnasiumEnvPool",
  "DmcFingerEnvSpec",
  "DmcFingerDMEnvPool",
  "DmcFingerGymEnvPool",
  "DmcFingerGymnasiumEnvPool",
  "DmcFishEnvSpec",
  "DmcFishDMEnvPool",
  "DmcFishGymEnvPool",
  "DmcFishGymnasiumEnvPool",
  "DmcHopperEnvSpec",
  "DmcHopperDMEnvPool",
  "DmcHopperGymEnvPool",
  "DmcHopperGymnasiumEnvPool",
  "DmcHumanoidEnvSpec",
  "DmcHumanoidDMEnvPool",
  "DmcHumanoidGymEnvPool",
  "DmcHumanoidGymnasiumEnvPool",
  "DmcHumanoidCMUEnvSpec",
  "DmcHumanoidCMUDMEnvPool",
  "DmcHumanoidCMUGymEnvPool",
  "DmcHumanoidCMUGymnasiumEnvPool",
  "DmcManipulatorEnvSpec",
  "DmcManipulatorDMEnvPool",
  "DmcManipulatorGymEnvPool",
  "DmcManipulatorGymnasiumEnvPool",
  "DmcPendulumEnvSpec",
  "DmcPendulumDMEnvPool",
  "DmcPendulumGymEnvPool",
  "DmcPendulumGymnasiumEnvPool",
  "DmcPointMassEnvSpec",
  "DmcPointMassDMEnvPool",
  "DmcPointMassGymEnvPool",
  "DmcPointMassGymnasiumEnvPool",
  "DmcReacherEnvSpec",
  "DmcReacherDMEnvPool",
  "DmcReacherGymEnvPool",
  "DmcReacherGymnasiumEnvPool",
  "DmcSwimmerEnvSpec",
  "DmcSwimmerDMEnvPool",
  "DmcSwimmerGymEnvPool",
  "DmcSwimmerGymnasiumEnvPool",
  "DmcWalkerEnvSpec",
  "DmcWalkerDMEnvPool",
  "DmcWalkerGymEnvPool",
  "DmcWalkerGymnasiumEnvPool",
]
