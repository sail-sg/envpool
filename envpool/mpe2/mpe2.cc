// Copyright 2026 Garena Online Private Limited
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "envpool/mpe2/mpe2.h"

#include "envpool/core/py_envpool.h"

using MPE2DiscreteEnvSpec = PyEnvSpec<mpe2::MPE2EnvSpec<false>>;
using MPE2DiscreteEnvPool = PyEnvPool<AsyncEnvPool<mpe2::MPE2Env<false>>>;
using MPE2ContinuousEnvSpec = PyEnvSpec<mpe2::MPE2EnvSpec<true>>;
using MPE2ContinuousEnvPool = PyEnvPool<AsyncEnvPool<mpe2::MPE2Env<true>>>;

PYBIND11_MODULE(mpe2_envpool, m) {
  REGISTER(m, MPE2DiscreteEnvSpec, MPE2DiscreteEnvPool)
  REGISTER(m, MPE2ContinuousEnvSpec, MPE2ContinuousEnvPool)
  pybind11::list tasks;
  for (const auto& task : mpe2::kTasks) {
    tasks.append(pybind11::make_tuple(task.name, task.id, task.alias));
  }
  m.attr("TASKS") = tasks;
}
