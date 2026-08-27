// Copyright 2026 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "envpool/craftax/craftax.h"

#include "envpool/core/py_envpool.h"

using CraftaxSymbolicEnvSpec =
    PyEnvSpec<craftax::CraftaxEnv<false, false>::Spec>;
using CraftaxSymbolicEnvPool =
    PyEnvPool<AsyncEnvPool<craftax::CraftaxEnv<false, false>>>;
using CraftaxPixelsEnvSpec = PyEnvSpec<craftax::CraftaxEnv<false, true>::Spec>;
using CraftaxPixelsEnvPool =
    PyEnvPool<AsyncEnvPool<craftax::CraftaxEnv<false, true>>>;
using CraftaxClassicSymbolicEnvSpec =
    PyEnvSpec<craftax::CraftaxEnv<true, false>::Spec>;
using CraftaxClassicSymbolicEnvPool =
    PyEnvPool<AsyncEnvPool<craftax::CraftaxEnv<true, false>>>;
using CraftaxClassicPixelsEnvSpec =
    PyEnvSpec<craftax::CraftaxEnv<true, true>::Spec>;
using CraftaxClassicPixelsEnvPool =
    PyEnvPool<AsyncEnvPool<craftax::CraftaxEnv<true, true>>>;

PYBIND11_MODULE(craftax_envpool, m) {
  REGISTER(m, CraftaxSymbolicEnvSpec, CraftaxSymbolicEnvPool)
  REGISTER(m, CraftaxPixelsEnvSpec, CraftaxPixelsEnvPool)
  REGISTER(m, CraftaxClassicSymbolicEnvSpec, CraftaxClassicSymbolicEnvPool)
  REGISTER(m, CraftaxClassicPixelsEnvSpec, CraftaxClassicPixelsEnvPool)
}
