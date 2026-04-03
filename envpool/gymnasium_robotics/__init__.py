# Copyright 2026 Garena Online Private Limited
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
"""Gymnasium-Robotics envs implemented by native EnvPool C++ backends."""

from envpool.gymnasium_robotics.gymnasium_robotics_envpool import (
    _GymnasiumRoboticsAdroitEnvPool,
    _GymnasiumRoboticsAdroitEnvSpec,
    _GymnasiumRoboticsFetchEnvPool,
    _GymnasiumRoboticsFetchEnvSpec,
    _GymnasiumRoboticsHandEnvPool,
    _GymnasiumRoboticsHandEnvSpec,
    _GymnasiumRoboticsKitchenEnvPool,
    _GymnasiumRoboticsKitchenEnvSpec,
    _GymnasiumRoboticsPointMazeEnvPool,
    _GymnasiumRoboticsPointMazeEnvSpec,
)

from envpool.python.api import py_env

(
    GymnasiumRoboticsFetchEnvSpec,
    GymnasiumRoboticsFetchDMEnvPool,
    GymnasiumRoboticsFetchGymEnvPool,
    GymnasiumRoboticsFetchGymnasiumEnvPool,
) = py_env(_GymnasiumRoboticsFetchEnvSpec, _GymnasiumRoboticsFetchEnvPool)

(
    GymnasiumRoboticsHandEnvSpec,
    GymnasiumRoboticsHandDMEnvPool,
    GymnasiumRoboticsHandGymEnvPool,
    GymnasiumRoboticsHandGymnasiumEnvPool,
) = py_env(_GymnasiumRoboticsHandEnvSpec, _GymnasiumRoboticsHandEnvPool)

(
    GymnasiumRoboticsAdroitEnvSpec,
    GymnasiumRoboticsAdroitDMEnvPool,
    GymnasiumRoboticsAdroitGymEnvPool,
    GymnasiumRoboticsAdroitGymnasiumEnvPool,
) = py_env(_GymnasiumRoboticsAdroitEnvSpec, _GymnasiumRoboticsAdroitEnvPool)

(
    GymnasiumRoboticsPointMazeEnvSpec,
    GymnasiumRoboticsPointMazeDMEnvPool,
    GymnasiumRoboticsPointMazeGymEnvPool,
    GymnasiumRoboticsPointMazeGymnasiumEnvPool,
) = py_env(
    _GymnasiumRoboticsPointMazeEnvSpec,
    _GymnasiumRoboticsPointMazeEnvPool,
)

(
    GymnasiumRoboticsKitchenEnvSpec,
    GymnasiumRoboticsKitchenDMEnvPool,
    GymnasiumRoboticsKitchenGymEnvPool,
    GymnasiumRoboticsKitchenGymnasiumEnvPool,
) = py_env(_GymnasiumRoboticsKitchenEnvSpec, _GymnasiumRoboticsKitchenEnvPool)

__all__ = [
    "GymnasiumRoboticsFetchEnvSpec",
    "GymnasiumRoboticsFetchDMEnvPool",
    "GymnasiumRoboticsFetchGymEnvPool",
    "GymnasiumRoboticsFetchGymnasiumEnvPool",
    "GymnasiumRoboticsHandEnvSpec",
    "GymnasiumRoboticsHandDMEnvPool",
    "GymnasiumRoboticsHandGymEnvPool",
    "GymnasiumRoboticsHandGymnasiumEnvPool",
    "GymnasiumRoboticsAdroitEnvSpec",
    "GymnasiumRoboticsAdroitDMEnvPool",
    "GymnasiumRoboticsAdroitGymEnvPool",
    "GymnasiumRoboticsAdroitGymnasiumEnvPool",
    "GymnasiumRoboticsPointMazeEnvSpec",
    "GymnasiumRoboticsPointMazeDMEnvPool",
    "GymnasiumRoboticsPointMazeGymEnvPool",
    "GymnasiumRoboticsPointMazeGymnasiumEnvPool",
    "GymnasiumRoboticsKitchenEnvSpec",
    "GymnasiumRoboticsKitchenDMEnvPool",
    "GymnasiumRoboticsKitchenGymEnvPool",
    "GymnasiumRoboticsKitchenGymnasiumEnvPool",
]
