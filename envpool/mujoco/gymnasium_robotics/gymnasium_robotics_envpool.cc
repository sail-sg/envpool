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
#include "envpool/mujoco/gymnasium_robotics/adroit.h"
#include "envpool/mujoco/gymnasium_robotics/fetch.h"
#include "envpool/mujoco/gymnasium_robotics/hand.h"
#include "envpool/mujoco/gymnasium_robotics/kitchen.h"
#include "envpool/mujoco/gymnasium_robotics/point_maze.h"

using GymnasiumRoboticsFetchEnvSpec =
    PyEnvSpec<gymnasium_robotics::FetchEnvSpec>;
using GymnasiumRoboticsFetchEnvPool =
    PyEnvPool<gymnasium_robotics::FetchEnvPool>;
using GymnasiumRoboticsHandEnvSpec = PyEnvSpec<gymnasium_robotics::HandEnvSpec>;
using GymnasiumRoboticsHandEnvPool = PyEnvPool<gymnasium_robotics::HandEnvPool>;
using GymnasiumRoboticsAdroitEnvSpec =
    PyEnvSpec<gymnasium_robotics::AdroitEnvSpec>;
using GymnasiumRoboticsAdroitEnvPool =
    PyEnvPool<gymnasium_robotics::AdroitEnvPool>;
using GymnasiumRoboticsPointMazeEnvSpec =
    PyEnvSpec<gymnasium_robotics::PointMazeEnvSpec>;
using GymnasiumRoboticsPointMazeEnvPool =
    PyEnvPool<gymnasium_robotics::PointMazeEnvPool>;
using GymnasiumRoboticsKitchenEnvSpec =
    PyEnvSpec<gymnasium_robotics::KitchenEnvSpec>;
using GymnasiumRoboticsKitchenEnvPool =
    PyEnvPool<gymnasium_robotics::KitchenEnvPool>;

PYBIND11_MODULE(gymnasium_robotics_envpool, m) {
  REGISTER(m, GymnasiumRoboticsFetchEnvSpec, GymnasiumRoboticsFetchEnvPool)
  REGISTER(m, GymnasiumRoboticsHandEnvSpec, GymnasiumRoboticsHandEnvPool)
  REGISTER(m, GymnasiumRoboticsAdroitEnvSpec, GymnasiumRoboticsAdroitEnvPool)
  REGISTER(m, GymnasiumRoboticsPointMazeEnvSpec,
           GymnasiumRoboticsPointMazeEnvPool)
  REGISTER(m, GymnasiumRoboticsKitchenEnvSpec, GymnasiumRoboticsKitchenEnvPool)
}
