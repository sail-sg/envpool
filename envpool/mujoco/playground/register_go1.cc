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
#include "envpool/mujoco/playground/go1.h"
#include "envpool/mujoco/playground/py_envpool_register.h"

using PlaygroundGo1EnvSpec = PyEnvSpec<mujoco_playground::PlaygroundGo1EnvSpec>;
using PlaygroundGo1EnvPool = PyEnvPool<mujoco_playground::PlaygroundGo1EnvPool>;
using PlaygroundGo1PixelEnvSpec =
    PyEnvSpec<mujoco_playground::PlaygroundGo1PixelEnvSpec>;
using PlaygroundGo1PixelEnvPool =
    PyEnvPool<mujoco_playground::PlaygroundGo1PixelEnvPool>;
using PlaygroundGo1GetupEnvSpec =
    PyEnvSpec<mujoco_playground::PlaygroundGo1GetupEnvSpec>;
using PlaygroundGo1GetupEnvPool =
    PyEnvPool<mujoco_playground::PlaygroundGo1GetupEnvPool>;
using PlaygroundGo1GetupPixelEnvSpec =
    PyEnvSpec<mujoco_playground::PlaygroundGo1GetupPixelEnvSpec>;
using PlaygroundGo1GetupPixelEnvPool =
    PyEnvPool<mujoco_playground::PlaygroundGo1GetupPixelEnvPool>;
using PlaygroundGo1HandstandEnvSpec =
    PyEnvSpec<mujoco_playground::PlaygroundGo1HandstandEnvSpec>;
using PlaygroundGo1HandstandEnvPool =
    PyEnvPool<mujoco_playground::PlaygroundGo1HandstandEnvPool>;
using PlaygroundGo1HandstandPixelEnvSpec =
    PyEnvSpec<mujoco_playground::PlaygroundGo1HandstandPixelEnvSpec>;
using PlaygroundGo1HandstandPixelEnvPool =
    PyEnvPool<mujoco_playground::PlaygroundGo1HandstandPixelEnvPool>;

void RegisterPlaygroundGo1(pybind11::module_& m) {
  REGISTER(m, PlaygroundGo1EnvSpec, PlaygroundGo1EnvPool)
  REGISTER(m, PlaygroundGo1PixelEnvSpec, PlaygroundGo1PixelEnvPool)
  REGISTER(m, PlaygroundGo1GetupEnvSpec, PlaygroundGo1GetupEnvPool)
  REGISTER(m, PlaygroundGo1GetupPixelEnvSpec, PlaygroundGo1GetupPixelEnvPool)
  REGISTER(m, PlaygroundGo1HandstandEnvSpec, PlaygroundGo1HandstandEnvPool)
  REGISTER(m, PlaygroundGo1HandstandPixelEnvSpec,
           PlaygroundGo1HandstandPixelEnvPool)
}
