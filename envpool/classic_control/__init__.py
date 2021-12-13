# Copyright 2021 Garena Online Private Limited
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
"""Classic control env in EnvPool."""

from envpool.python.api import py_env

from .classic_control_envpool import (
  _AcrobotEnvPool,
  _AcrobotEnvSpec,
  _CartPoleEnvPool,
  _CartPoleEnvSpec,
  _MountainCarContinuousEnvPool,
  _MountainCarContinuousEnvSpec,
  _MountainCarEnvPool,
  _MountainCarEnvSpec,
  _PendulumEnvPool,
  _PendulumEnvSpec,
)

CartPoleEnvSpec, CartPoleDMEnvPool, CartPoleGymEnvPool = py_env(
  _CartPoleEnvSpec, _CartPoleEnvPool
)

PendulumEnvSpec, PendulumDMEnvPool, PendulumGymEnvPool = py_env(
  _PendulumEnvSpec, _PendulumEnvPool
)

(MountainCarEnvSpec, MountainCarDMEnvPool,
 MountainCarGymEnvPool) = py_env(_MountainCarEnvSpec, _MountainCarEnvPool)

(
  MountainCarContinuousEnvSpec, MountainCarContinuousDMEnvPool,
  MountainCarContinuousGymEnvPool
) = py_env(_MountainCarContinuousEnvSpec, _MountainCarContinuousEnvPool)

AcrobotEnvSpec, AcrobotDMEnvPool, AcrobotGymEnvPool = py_env(
  _AcrobotEnvSpec, _AcrobotEnvPool
)

__all__ = [
  "CartPoleEnvSpec",
  "CartPoleDMEnvPool",
  "CartPoleGymEnvPool",
  "PendulumEnvSpec",
  "PendulumDMEnvPool",
  "PendulumGymEnvPool",
  "MountainCarEnvSpec",
  "MountainCarDMEnvPool",
  "MountainCarGymEnvPool",
  "MountainCarContinuousEnvSpec",
  "MountainCarContinuousDMEnvPool",
  "MountainCarContinuousGymEnvPool",
  "AcrobotEnvSpec",
  "AcrobotDMEnvPool",
  "AcrobotGymEnvPool",
]
