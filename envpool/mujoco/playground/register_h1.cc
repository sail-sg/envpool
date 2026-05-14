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
#include "envpool/mujoco/playground/h1.h"
#include "envpool/mujoco/playground/py_envpool_register.h"

using PlaygroundH1EnvSpec = PyEnvSpec<mujoco_playground::PlaygroundH1EnvSpec>;
using PlaygroundH1EnvPool = PyEnvPool<mujoco_playground::PlaygroundH1EnvPool>;
using PlaygroundH1PixelEnvSpec =
    PyEnvSpec<mujoco_playground::PlaygroundH1PixelEnvSpec>;
using PlaygroundH1PixelEnvPool =
    PyEnvPool<mujoco_playground::PlaygroundH1PixelEnvPool>;

void RegisterPlaygroundH1(pybind11::module_& m) {
  REGISTER(m, PlaygroundH1EnvSpec, PlaygroundH1EnvPool)
  REGISTER(m, PlaygroundH1PixelEnvSpec, PlaygroundH1PixelEnvPool)
}
