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

DmcAcrobotEnvSpec, DmcAcrobotDMEnvPool, DmcAcrobotGymEnvPool = py_env(
  _DmcAcrobotEnvSpec, _DmcAcrobotEnvPool
)
DmcBallInCupEnvSpec, DmcBallInCupDMEnvPool, DmcBallInCupGymEnvPool = py_env(
  _DmcBallInCupEnvSpec, _DmcBallInCupEnvPool
)
DmcCartpoleEnvSpec, DmcCartpoleDMEnvPool, DmcCartpoleGymEnvPool = py_env(
  _DmcCartpoleEnvSpec, _DmcCartpoleEnvPool
)
DmcCheetahEnvSpec, DmcCheetahDMEnvPool, DmcCheetahGymEnvPool = py_env(
  _DmcCheetahEnvSpec, _DmcCheetahEnvPool
)
DmcFingerEnvSpec, DmcFingerDMEnvPool, DmcFingerGymEnvPool = py_env(
  _DmcFingerEnvSpec, _DmcFingerEnvPool
)
DmcFishEnvSpec, DmcFishDMEnvPool, DmcFishGymEnvPool = py_env(
  _DmcFishEnvSpec, _DmcFishEnvPool
)
DmcHopperEnvSpec, DmcHopperDMEnvPool, DmcHopperGymEnvPool = py_env(
  _DmcHopperEnvSpec, _DmcHopperEnvPool
)
DmcHumanoidEnvSpec, DmcHumanoidDMEnvPool, DmcHumanoidGymEnvPool = py_env(
  _DmcHumanoidEnvSpec, _DmcHumanoidEnvPool
)
(
  DmcManipulatorEnvSpec,
  DmcManipulatorDMEnvPool,
  DmcManipulatorGymEnvPool,
) = py_env(_DmcManipulatorEnvSpec, _DmcManipulatorEnvPool)
DmcPendulumEnvSpec, DmcPendulumDMEnvPool, DmcPendulumGymEnvPool = py_env(
  _DmcPendulumEnvSpec, _DmcPendulumEnvPool
)
DmcPointMassEnvSpec, DmcPointMassDMEnvPool, DmcPointMassGymEnvPool = py_env(
  _DmcPointMassEnvSpec, _DmcPointMassEnvPool
)
DmcReacherEnvSpec, DmcReacherDMEnvPool, DmcReacherGymEnvPool = py_env(
  _DmcReacherEnvSpec, _DmcReacherEnvPool
)
DmcSwimmerEnvSpec, DmcSwimmerDMEnvPool, DmcSwimmerGymEnvPool = py_env(
  _DmcSwimmerEnvSpec, _DmcSwimmerEnvPool
)
DmcWalkerEnvSpec, DmcWalkerDMEnvPool, DmcWalkerGymEnvPool = py_env(
  _DmcWalkerEnvSpec, _DmcWalkerEnvPool
)

__all__ = [
  "DmcAcrobotEnvSpec",
  "DmcAcrobotDMEnvPool",
  "DmcAcrobotGymEnvPool",
  "DmcBallInCupEnvSpec",
  "DmcBallInCupDMEnvPool",
  "DmcBallInCupGymEnvPool",
  "DmcCartpoleEnvSpec",
  "DmcCartpoleDMEnvPool",
  "DmcCartpoleGymEnvPool",
  "DmcCheetahEnvSpec",
  "DmcCheetahDMEnvPool",
  "DmcCheetahGymEnvPool",
  "DmcFingerEnvSpec",
  "DmcFingerDMEnvPool",
  "DmcFingerGymEnvPool",
  "DmcFishEnvSpec",
  "DmcFishDMEnvPool",
  "DmcFishGymEnvPool",
  "DmcHopperEnvSpec",
  "DmcHopperDMEnvPool",
  "DmcHopperGymEnvPool",
  "DmcHumanoidEnvSpec",
  "DmcHumanoidDMEnvPool",
  "DmcHumanoidGymEnvPool",
  "DmcManipulatorEnvSpec",
  "DmcManipulatorDMEnvPool",
  "DmcManipulatorGymEnvPool",
  "DmcPendulumEnvSpec",
  "DmcPendulumDMEnvPool",
  "DmcPendulumGymEnvPool",
  "DmcPointMassEnvSpec",
  "DmcPointMassDMEnvPool",
  "DmcPointMassGymEnvPool",
  "DmcReacherEnvSpec",
  "DmcReacherDMEnvPool",
  "DmcReacherGymEnvPool",
  "DmcSwimmerEnvSpec",
  "DmcSwimmerDMEnvPool",
  "DmcSwimmerGymEnvPool",
  "DmcWalkerEnvSpec",
  "DmcWalkerDMEnvPool",
  "DmcWalkerGymEnvPool",
]
