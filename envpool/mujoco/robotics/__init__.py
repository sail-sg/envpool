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

from envpool.mujoco.robotics_envpool import (
    _GymnasiumRoboticsAdroitEnvPool,
    _GymnasiumRoboticsAdroitEnvSpec,
    _GymnasiumRoboticsAdroitPixelEnvPool,
    _GymnasiumRoboticsAdroitPixelEnvSpec,
    _GymnasiumRoboticsFetchEnvPool,
    _GymnasiumRoboticsFetchEnvSpec,
    _GymnasiumRoboticsFetchPixelEnvPool,
    _GymnasiumRoboticsFetchPixelEnvSpec,
    _GymnasiumRoboticsHandEnvPool,
    _GymnasiumRoboticsHandEnvSpec,
    _GymnasiumRoboticsHandPixelEnvPool,
    _GymnasiumRoboticsHandPixelEnvSpec,
    _GymnasiumRoboticsKitchenEnvPool,
    _GymnasiumRoboticsKitchenEnvSpec,
    _GymnasiumRoboticsKitchenPixelEnvPool,
    _GymnasiumRoboticsKitchenPixelEnvSpec,
    _GymnasiumRoboticsPointMazeEnvPool,
    _GymnasiumRoboticsPointMazeEnvSpec,
    _GymnasiumRoboticsPointMazePixelEnvPool,
    _GymnasiumRoboticsPointMazePixelEnvSpec,
)

from envpool.python.api import py_env

(
    GymnasiumRoboticsFetchEnvSpec,
    GymnasiumRoboticsFetchDMEnvPool,
    GymnasiumRoboticsFetchGymnasiumEnvPool,
) = py_env(_GymnasiumRoboticsFetchEnvSpec, _GymnasiumRoboticsFetchEnvPool)
(
    GymnasiumRoboticsFetchPixelEnvSpec,
    GymnasiumRoboticsFetchPixelDMEnvPool,
    GymnasiumRoboticsFetchPixelGymnasiumEnvPool,
) = py_env(
    _GymnasiumRoboticsFetchPixelEnvSpec, _GymnasiumRoboticsFetchPixelEnvPool
)
(
    GymnasiumRoboticsHandEnvSpec,
    GymnasiumRoboticsHandDMEnvPool,
    GymnasiumRoboticsHandGymnasiumEnvPool,
) = py_env(_GymnasiumRoboticsHandEnvSpec, _GymnasiumRoboticsHandEnvPool)
(
    GymnasiumRoboticsHandPixelEnvSpec,
    GymnasiumRoboticsHandPixelDMEnvPool,
    GymnasiumRoboticsHandPixelGymnasiumEnvPool,
) = py_env(
    _GymnasiumRoboticsHandPixelEnvSpec, _GymnasiumRoboticsHandPixelEnvPool
)
(
    GymnasiumRoboticsAdroitEnvSpec,
    GymnasiumRoboticsAdroitDMEnvPool,
    GymnasiumRoboticsAdroitGymnasiumEnvPool,
) = py_env(_GymnasiumRoboticsAdroitEnvSpec, _GymnasiumRoboticsAdroitEnvPool)
(
    GymnasiumRoboticsAdroitPixelEnvSpec,
    GymnasiumRoboticsAdroitPixelDMEnvPool,
    GymnasiumRoboticsAdroitPixelGymnasiumEnvPool,
) = py_env(
    _GymnasiumRoboticsAdroitPixelEnvSpec, _GymnasiumRoboticsAdroitPixelEnvPool
)
(
    GymnasiumRoboticsPointMazeEnvSpec,
    GymnasiumRoboticsPointMazeDMEnvPool,
    GymnasiumRoboticsPointMazeGymnasiumEnvPool,
) = py_env(
    _GymnasiumRoboticsPointMazeEnvSpec, _GymnasiumRoboticsPointMazeEnvPool
)
(
    GymnasiumRoboticsPointMazePixelEnvSpec,
    GymnasiumRoboticsPointMazePixelDMEnvPool,
    GymnasiumRoboticsPointMazePixelGymnasiumEnvPool,
) = py_env(
    _GymnasiumRoboticsPointMazePixelEnvSpec,
    _GymnasiumRoboticsPointMazePixelEnvPool,
)
(
    GymnasiumRoboticsKitchenEnvSpec,
    GymnasiumRoboticsKitchenDMEnvPool,
    GymnasiumRoboticsKitchenGymnasiumEnvPool,
) = py_env(_GymnasiumRoboticsKitchenEnvSpec, _GymnasiumRoboticsKitchenEnvPool)
(
    GymnasiumRoboticsKitchenPixelEnvSpec,
    GymnasiumRoboticsKitchenPixelDMEnvPool,
    GymnasiumRoboticsKitchenPixelGymnasiumEnvPool,
) = py_env(
    _GymnasiumRoboticsKitchenPixelEnvSpec, _GymnasiumRoboticsKitchenPixelEnvPool
)

__all__ = [
    "GymnasiumRoboticsFetchEnvSpec",
    "GymnasiumRoboticsFetchDMEnvPool",
    "GymnasiumRoboticsFetchGymnasiumEnvPool",
    "GymnasiumRoboticsFetchPixelEnvSpec",
    "GymnasiumRoboticsFetchPixelDMEnvPool",
    "GymnasiumRoboticsFetchPixelGymnasiumEnvPool",
    "GymnasiumRoboticsHandEnvSpec",
    "GymnasiumRoboticsHandDMEnvPool",
    "GymnasiumRoboticsHandGymnasiumEnvPool",
    "GymnasiumRoboticsHandPixelEnvSpec",
    "GymnasiumRoboticsHandPixelDMEnvPool",
    "GymnasiumRoboticsHandPixelGymnasiumEnvPool",
    "GymnasiumRoboticsAdroitEnvSpec",
    "GymnasiumRoboticsAdroitDMEnvPool",
    "GymnasiumRoboticsAdroitGymnasiumEnvPool",
    "GymnasiumRoboticsAdroitPixelEnvSpec",
    "GymnasiumRoboticsAdroitPixelDMEnvPool",
    "GymnasiumRoboticsAdroitPixelGymnasiumEnvPool",
    "GymnasiumRoboticsPointMazeEnvSpec",
    "GymnasiumRoboticsPointMazeDMEnvPool",
    "GymnasiumRoboticsPointMazeGymnasiumEnvPool",
    "GymnasiumRoboticsPointMazePixelEnvSpec",
    "GymnasiumRoboticsPointMazePixelDMEnvPool",
    "GymnasiumRoboticsPointMazePixelGymnasiumEnvPool",
    "GymnasiumRoboticsKitchenEnvSpec",
    "GymnasiumRoboticsKitchenDMEnvPool",
    "GymnasiumRoboticsKitchenGymnasiumEnvPool",
    "GymnasiumRoboticsKitchenPixelEnvSpec",
    "GymnasiumRoboticsKitchenPixelDMEnvPool",
    "GymnasiumRoboticsKitchenPixelGymnasiumEnvPool",
]
