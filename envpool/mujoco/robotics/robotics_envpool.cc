// Copyright 2026 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "envpool/core/py_envpool.h"
#include "envpool/mujoco/metaworld/metaworld_env.h"
#include "envpool/mujoco/robotics/adroit.h"
#include "envpool/mujoco/robotics/fetch.h"
#include "envpool/mujoco/robotics/hand.h"
#include "envpool/mujoco/robotics/kitchen.h"
#include "envpool/mujoco/robotics/point_maze.h"

using GymnasiumRoboticsFetchEnvSpec =
    PyEnvSpec<gymnasium_robotics::FetchEnvSpec>;
using GymnasiumRoboticsFetchEnvPool =
    PyEnvPool<gymnasium_robotics::FetchEnvPool>;
using GymnasiumRoboticsFetchPixelEnvSpec =
    PyEnvSpec<gymnasium_robotics::FetchPixelEnvSpec>;
using GymnasiumRoboticsFetchPixelEnvPool =
    PyEnvPool<gymnasium_robotics::FetchPixelEnvPool>;

using GymnasiumRoboticsHandEnvSpec = PyEnvSpec<gymnasium_robotics::HandEnvSpec>;
using GymnasiumRoboticsHandEnvPool = PyEnvPool<gymnasium_robotics::HandEnvPool>;
using GymnasiumRoboticsHandPixelEnvSpec =
    PyEnvSpec<gymnasium_robotics::HandPixelEnvSpec>;
using GymnasiumRoboticsHandPixelEnvPool =
    PyEnvPool<gymnasium_robotics::HandPixelEnvPool>;

using GymnasiumRoboticsAdroitEnvSpec =
    PyEnvSpec<gymnasium_robotics::AdroitEnvSpec>;
using GymnasiumRoboticsAdroitEnvPool =
    PyEnvPool<gymnasium_robotics::AdroitEnvPool>;
using GymnasiumRoboticsAdroitPixelEnvSpec =
    PyEnvSpec<gymnasium_robotics::AdroitPixelEnvSpec>;
using GymnasiumRoboticsAdroitPixelEnvPool =
    PyEnvPool<gymnasium_robotics::AdroitPixelEnvPool>;

using GymnasiumRoboticsPointMazeEnvSpec =
    PyEnvSpec<gymnasium_robotics::PointMazeEnvSpec>;
using GymnasiumRoboticsPointMazeEnvPool =
    PyEnvPool<gymnasium_robotics::PointMazeEnvPool>;
using GymnasiumRoboticsPointMazePixelEnvSpec =
    PyEnvSpec<gymnasium_robotics::PointMazePixelEnvSpec>;
using GymnasiumRoboticsPointMazePixelEnvPool =
    PyEnvPool<gymnasium_robotics::PointMazePixelEnvPool>;

using GymnasiumRoboticsKitchenEnvSpec =
    PyEnvSpec<gymnasium_robotics::KitchenEnvSpec>;
using GymnasiumRoboticsKitchenEnvPool =
    PyEnvPool<gymnasium_robotics::KitchenEnvPool>;
using GymnasiumRoboticsKitchenPixelEnvSpec =
    PyEnvSpec<gymnasium_robotics::KitchenPixelEnvSpec>;
using GymnasiumRoboticsKitchenPixelEnvPool =
    PyEnvPool<gymnasium_robotics::KitchenPixelEnvPool>;

using MetaWorldEnvSpec = PyEnvSpec<metaworld::MetaWorldEnvSpec>;
using MetaWorldEnvPool = PyEnvPool<metaworld::MetaWorldEnvPool>;
using MetaWorldPixelEnvSpec = PyEnvSpec<metaworld::MetaWorldPixelEnvSpec>;
using MetaWorldPixelEnvPool = PyEnvPool<metaworld::MetaWorldPixelEnvPool>;

PYBIND11_MODULE(robotics_envpool, m) {
  REGISTER(m, GymnasiumRoboticsFetchEnvSpec, GymnasiumRoboticsFetchEnvPool)
  REGISTER(m, GymnasiumRoboticsFetchPixelEnvSpec,
           GymnasiumRoboticsFetchPixelEnvPool)
  REGISTER(m, GymnasiumRoboticsHandEnvSpec, GymnasiumRoboticsHandEnvPool)
  REGISTER(m, GymnasiumRoboticsHandPixelEnvSpec,
           GymnasiumRoboticsHandPixelEnvPool)
  REGISTER(m, GymnasiumRoboticsAdroitEnvSpec, GymnasiumRoboticsAdroitEnvPool)
  REGISTER(m, GymnasiumRoboticsAdroitPixelEnvSpec,
           GymnasiumRoboticsAdroitPixelEnvPool)
  REGISTER(m, GymnasiumRoboticsPointMazeEnvSpec,
           GymnasiumRoboticsPointMazeEnvPool)
  REGISTER(m, GymnasiumRoboticsPointMazePixelEnvSpec,
           GymnasiumRoboticsPointMazePixelEnvPool)
  REGISTER(m, GymnasiumRoboticsKitchenEnvSpec, GymnasiumRoboticsKitchenEnvPool)
  REGISTER(m, GymnasiumRoboticsKitchenPixelEnvSpec,
           GymnasiumRoboticsKitchenPixelEnvPool)
  REGISTER(m, MetaWorldEnvSpec, MetaWorldEnvPool)
  REGISTER(m, MetaWorldPixelEnvSpec, MetaWorldPixelEnvPool)
}
