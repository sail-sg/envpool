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
"""Mujoco env in EnvPool."""

from envpool.python.api import py_env

from .mujoco_envpool import (
  _AntEnvPool,
  _AntEnvSpec,
  _HalfCheetahEnvPool,
  _HalfCheetahEnvSpec,
  _HopperEnvPool,
  _HopperEnvSpec,
  _HumanoidEnvPool,
  _HumanoidEnvSpec,
  _HumanoidStandupEnvPool,
  _HumanoidStandupEnvSpec,
  _InvertedDoublePendulumEnvPool,
  _InvertedDoublePendulumEnvSpec,
  _InvertedPendulumEnvPool,
  _InvertedPendulumEnvSpec,
  _PusherEnvPool,
  _PusherEnvSpec,
  _ReacherEnvPool,
  _ReacherEnvSpec,
  _SwimmerEnvPool,
  _SwimmerEnvSpec,
  _Walker2dEnvPool,
  _Walker2dEnvSpec,
)

AntEnvSpec, AntDMEnvPool, AntGymEnvPool = py_env(_AntEnvSpec, _AntEnvPool)

HalfCheetahEnvSpec, HalfCheetahDMEnvPool, HalfCheetahGymEnvPool = py_env(
  _HalfCheetahEnvSpec, _HalfCheetahEnvPool
)

HopperEnvSpec, HopperDMEnvPool, HopperGymEnvPool = py_env(
  _HopperEnvSpec, _HopperEnvPool
)

HumanoidEnvSpec, HumanoidDMEnvPool, HumanoidGymEnvPool = py_env(
  _HumanoidEnvSpec, _HumanoidEnvPool
)

(
  HumanoidStandupEnvSpec,
  HumanoidStandupDMEnvPool,
  HumanoidStandupGymEnvPool,
) = py_env(_HumanoidStandupEnvSpec, _HumanoidStandupEnvPool)

(
  InvertedDoublePendulumEnvSpec,
  InvertedDoublePendulumDMEnvPool,
  InvertedDoublePendulumGymEnvPool,
) = py_env(_InvertedDoublePendulumEnvSpec, _InvertedDoublePendulumEnvPool)

(
  InvertedPendulumEnvSpec,
  InvertedPendulumDMEnvPool,
  InvertedPendulumGymEnvPool,
) = py_env(_InvertedPendulumEnvSpec, _InvertedPendulumEnvPool)

PusherEnvSpec, PusherDMEnvPool, PusherGymEnvPool = py_env(
  _PusherEnvSpec, _PusherEnvPool
)

ReacherEnvSpec, ReacherDMEnvPool, ReacherGymEnvPool = py_env(
  _ReacherEnvSpec, _ReacherEnvPool
)

SwimmerEnvSpec, SwimmerDMEnvPool, SwimmerGymEnvPool = py_env(
  _SwimmerEnvSpec, _SwimmerEnvPool
)

Walker2dEnvSpec, Walker2dDMEnvPool, Walker2dGymEnvPool = py_env(
  _Walker2dEnvSpec, _Walker2dEnvPool
)

__all__ = [
  "AntEnvSpec",
  "AntDMEnvPool",
  "AntGymEnvPool",
  "HalfCheetahEnvSpec",
  "HalfCheetahDMEnvPool",
  "HalfCheetahGymEnvPool",
  "HopperEnvSpec",
  "HopperDMEnvPool",
  "HopperGymEnvPool",
  "HumanoidEnvSpec",
  "HumanoidDMEnvPool",
  "HumanoidGymEnvPool",
  "HumanoidStandupEnvSpec",
  "HumanoidStandupDMEnvPool",
  "HumanoidStandupGymEnvPool",
  "InvertedDoublePendulumEnvSpec",
  "InvertedDoublePendulumDMEnvPool",
  "InvertedDoublePendulumGymEnvPool"
  "InvertedPendulumEnvSpec",
  "InvertedPendulumDMEnvPool",
  "InvertedPendulumGymEnvPool",
  "PusherEnvSpec",
  "PusherDMEnvPool",
  "PusherGymEnvPool",
  "ReacherEnvSpec",
  "ReacherDMEnvPool",
  "ReacherGymEnvPool",
  "SwimmerEnvSpec",
  "SwimmerDMEnvPool",
  "SwimmerGymEnvPool",
  "Walker2dEnvSpec",
  "Walker2dDMEnvPool",
  "Walker2dGymEnvPool",
]
