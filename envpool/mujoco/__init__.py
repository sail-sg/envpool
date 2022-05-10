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

from .mujoco_gym_envpool import _AntEnvPool as _GymAntEnvPool
from .mujoco_gym_envpool import _AntEnvSpec as _GymAntEnvSpec
from .mujoco_gym_envpool import _HalfCheetahEnvPool as _GymHalfCheetahEnvPool
from .mujoco_gym_envpool import _HalfCheetahEnvSpec as _GymHalfCheetahEnvSpec
from .mujoco_gym_envpool import _HopperEnvPool as _GymHopperEnvPool
from .mujoco_gym_envpool import _HopperEnvSpec as _GymHopperEnvSpec
from .mujoco_gym_envpool import _HumanoidEnvPool as _GymHumanoidEnvPool
from .mujoco_gym_envpool import _HumanoidEnvSpec as _GymHumanoidEnvSpec
from .mujoco_gym_envpool import (
  _HumanoidStandupEnvPool as _GymHumanoidStandupEnvPool,
)
from .mujoco_gym_envpool import (
  _HumanoidStandupEnvSpec as _GymHumanoidStandupEnvSpec,
)
from .mujoco_gym_envpool import (
  _InvertedDoublePendulumEnvPool as _GymInvertedDoublePendulumEnvPool,
)
from .mujoco_gym_envpool import (
  _InvertedDoublePendulumEnvSpec as _GymInvertedDoublePendulumEnvSpec,
)
from .mujoco_gym_envpool import (
  _InvertedPendulumEnvPool as _GymInvertedPendulumEnvPool,
)
from .mujoco_gym_envpool import (
  _InvertedPendulumEnvSpec as _GymInvertedPendulumEnvSpec,
)
from .mujoco_gym_envpool import _PusherEnvPool as _GymPusherEnvPool
from .mujoco_gym_envpool import _PusherEnvSpec as _GymPusherEnvSpec
from .mujoco_gym_envpool import _ReacherEnvPool as _GymReacherEnvPool
from .mujoco_gym_envpool import _ReacherEnvSpec as _GymReacherEnvSpec
from .mujoco_gym_envpool import _SwimmerEnvPool as _GymSwimmerEnvPool
from .mujoco_gym_envpool import _SwimmerEnvSpec as _GymSwimmerEnvSpec
from .mujoco_gym_envpool import _Walker2dEnvPool as _GymWalker2dEnvPool
from .mujoco_gym_envpool import _Walker2dEnvSpec as _GymWalker2dEnvSpec

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
