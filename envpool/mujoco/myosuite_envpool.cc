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
#include "envpool/mujoco/myosuite/myobase.h"

using MyoSuitePoseEnvSpec = PyEnvSpec<myosuite_envpool::MyoSuitePoseEnvSpec>;
using MyoSuitePoseEnvPool = PyEnvPool<myosuite_envpool::MyoSuitePoseEnvPool>;
using MyoSuitePosePixelEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoSuitePosePixelEnvSpec>;
using MyoSuitePosePixelEnvPool =
    PyEnvPool<myosuite_envpool::MyoSuitePosePixelEnvPool>;

using MyoSuiteReachEnvSpec = PyEnvSpec<myosuite_envpool::MyoSuiteReachEnvSpec>;
using MyoSuiteReachEnvPool = PyEnvPool<myosuite_envpool::MyoSuiteReachEnvPool>;
using MyoSuiteReachPixelEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoSuiteReachPixelEnvSpec>;
using MyoSuiteReachPixelEnvPool =
    PyEnvPool<myosuite_envpool::MyoSuiteReachPixelEnvPool>;

PYBIND11_MODULE(myosuite_envpool, m) {
  REGISTER(m, MyoSuitePoseEnvSpec, MyoSuitePoseEnvPool)
  REGISTER(m, MyoSuitePosePixelEnvSpec, MyoSuitePosePixelEnvPool)
  REGISTER(m, MyoSuiteReachEnvSpec, MyoSuiteReachEnvPool)
  REGISTER(m, MyoSuiteReachPixelEnvSpec, MyoSuiteReachPixelEnvPool)
}
