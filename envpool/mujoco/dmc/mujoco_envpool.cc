// Copyright 2022 Garena Online Private Limited
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
#include "envpool/mujoco/dmc/cheetah.h"
#include "envpool/mujoco/dmc/hopper.h"

using DmcHopperEnvSpec = PyEnvSpec<mujoco_dmc::HopperEnvSpec>;
using DmcHopperEnvPool = PyEnvPool<mujoco_dmc::HopperEnvPool>;

using DmcCheetahEnvSpec = PyEnvSpec<mujoco_dmc::CheetahEnvSpec>;
using DmcCheetahEnvPool = PyEnvPool<mujoco_dmc::CheetahEnvPool>;

PYBIND11_MODULE(mujoco_dmc_envpool, m) {
  REGISTER(m, DmcCheetahEnvSpec, DmcCheetahEnvPool)
  REGISTER(m, DmcHopperEnvSpec, DmcHopperEnvPool)
}
