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
#include "envpool/mujoco/myosuite/myodm.h"

namespace myosuite_envpool {

void RegisterMyoSuiteDmBindings(py::module_& module) {
  using MyoDMTrackEnvSpec = PyEnvSpec<::myosuite_envpool::MyoDMTrackEnvSpec>;
  using MyoDMTrackEnvPool = PyEnvPool<::myosuite_envpool::MyoDMTrackEnvPool>;
  using MyoDMTrackPixelEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoDMTrackPixelEnvSpec>;
  using MyoDMTrackPixelEnvPool =
      PyEnvPool<::myosuite_envpool::MyoDMTrackPixelEnvPool>;

  REGISTER(module, MyoDMTrackEnvSpec, MyoDMTrackEnvPool)
  REGISTER(module, MyoDMTrackPixelEnvSpec, MyoDMTrackPixelEnvPool)
}

}  // namespace myosuite_envpool
