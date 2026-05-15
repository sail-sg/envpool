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
#include "envpool/mujoco/playground/py_envpool_register.h"
#include "envpool/mujoco/playground/spot.h"

using PlaygroundSpotJoystickEnvSpec =
    PyEnvSpec<mujoco_playground::PlaygroundSpotJoystickEnvSpec>;
using PlaygroundSpotJoystickEnvPool =
    PyEnvPool<mujoco_playground::PlaygroundSpotJoystickEnvPool>;
using PlaygroundSpotJoystickPixelEnvSpec =
    PyEnvSpec<mujoco_playground::PlaygroundSpotJoystickPixelEnvSpec>;
using PlaygroundSpotJoystickPixelEnvPool =
    PyEnvPool<mujoco_playground::PlaygroundSpotJoystickPixelEnvPool>;
using PlaygroundSpotGetupEnvSpec =
    PyEnvSpec<mujoco_playground::PlaygroundSpotGetupEnvSpec>;
using PlaygroundSpotGetupEnvPool =
    PyEnvPool<mujoco_playground::PlaygroundSpotGetupEnvPool>;
using PlaygroundSpotGetupPixelEnvSpec =
    PyEnvSpec<mujoco_playground::PlaygroundSpotGetupPixelEnvSpec>;
using PlaygroundSpotGetupPixelEnvPool =
    PyEnvPool<mujoco_playground::PlaygroundSpotGetupPixelEnvPool>;
using PlaygroundSpotGaitEnvSpec =
    PyEnvSpec<mujoco_playground::PlaygroundSpotGaitEnvSpec>;
using PlaygroundSpotGaitEnvPool =
    PyEnvPool<mujoco_playground::PlaygroundSpotGaitEnvPool>;
using PlaygroundSpotGaitPixelEnvSpec =
    PyEnvSpec<mujoco_playground::PlaygroundSpotGaitPixelEnvSpec>;
using PlaygroundSpotGaitPixelEnvPool =
    PyEnvPool<mujoco_playground::PlaygroundSpotGaitPixelEnvPool>;

void RegisterPlaygroundSpot(pybind11::module_& m) {
  REGISTER(m, PlaygroundSpotJoystickEnvSpec, PlaygroundSpotJoystickEnvPool)
  REGISTER(m, PlaygroundSpotJoystickPixelEnvSpec,
           PlaygroundSpotJoystickPixelEnvPool)
  REGISTER(m, PlaygroundSpotGetupEnvSpec, PlaygroundSpotGetupEnvPool)
  REGISTER(m, PlaygroundSpotGetupPixelEnvSpec, PlaygroundSpotGetupPixelEnvPool)
  REGISTER(m, PlaygroundSpotGaitEnvSpec, PlaygroundSpotGaitEnvPool)
  REGISTER(m, PlaygroundSpotGaitPixelEnvSpec, PlaygroundSpotGaitPixelEnvPool)
}
