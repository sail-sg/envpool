// Copyright 2021 Garena Online Private Limited
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

#include "envpool/classic_control/cartpole.h"
#include "envpool/classic_control/pendulum.h"
#include "envpool/core/py_envpool.h"

typedef PyEnvSpec<classic_control::CartPoleEnvSpec> CartPoleEnvSpec;
typedef PyEnvPool<classic_control::CartPoleEnvPool> CartPoleEnvPool;

typedef PyEnvSpec<classic_control::PendulumEnvSpec> PendulumEnvSpec;
typedef PyEnvPool<classic_control::PendulumEnvPool> PendulumEnvPool;

PYBIND11_MODULE(classic_control_envpool, m) {
  REGISTER(m, CartPoleEnvSpec, CartPoleEnvPool)
  REGISTER(m, PendulumEnvSpec, PendulumEnvPool)
}
