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
#include "envpool/mujoco/myosuite/myobase_extended.h"

namespace myosuite_envpool {

void RegisterMyoSuiteBaseBindings(py::module_& module) {
  using MyoSuitePoseEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoSuitePoseEnvSpec>;
  using MyoSuitePoseEnvPool =
      PyEnvPool<::myosuite_envpool::MyoSuitePoseEnvPool>;
  using MyoSuitePosePixelEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoSuitePosePixelEnvSpec>;
  using MyoSuitePosePixelEnvPool =
      PyEnvPool<::myosuite_envpool::MyoSuitePosePixelEnvPool>;

  using MyoSuiteReachEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoSuiteReachEnvSpec>;
  using MyoSuiteReachEnvPool =
      PyEnvPool<::myosuite_envpool::MyoSuiteReachEnvPool>;
  using MyoSuiteReachPixelEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoSuiteReachPixelEnvSpec>;
  using MyoSuiteReachPixelEnvPool =
      PyEnvPool<::myosuite_envpool::MyoSuiteReachPixelEnvPool>;

  using MyoSuiteKeyTurnEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoSuiteKeyTurnEnvSpec>;
  using MyoSuiteKeyTurnEnvPool =
      PyEnvPool<::myosuite_envpool::MyoSuiteKeyTurnEnvPool>;
  using MyoSuiteKeyTurnPixelEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoSuiteKeyTurnPixelEnvSpec>;
  using MyoSuiteKeyTurnPixelEnvPool =
      PyEnvPool<::myosuite_envpool::MyoSuiteKeyTurnPixelEnvPool>;

  using MyoSuiteObjHoldEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoSuiteObjHoldEnvSpec>;
  using MyoSuiteObjHoldEnvPool =
      PyEnvPool<::myosuite_envpool::MyoSuiteObjHoldEnvPool>;
  using MyoSuiteObjHoldPixelEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoSuiteObjHoldPixelEnvSpec>;
  using MyoSuiteObjHoldPixelEnvPool =
      PyEnvPool<::myosuite_envpool::MyoSuiteObjHoldPixelEnvPool>;

  using MyoSuiteTorsoEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoSuiteTorsoEnvSpec>;
  using MyoSuiteTorsoEnvPool =
      PyEnvPool<::myosuite_envpool::MyoSuiteTorsoEnvPool>;
  using MyoSuiteTorsoPixelEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoSuiteTorsoPixelEnvSpec>;
  using MyoSuiteTorsoPixelEnvPool =
      PyEnvPool<::myosuite_envpool::MyoSuiteTorsoPixelEnvPool>;

  using MyoSuitePenTwirlEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoSuitePenTwirlEnvSpec>;
  using MyoSuitePenTwirlEnvPool =
      PyEnvPool<::myosuite_envpool::MyoSuitePenTwirlEnvPool>;
  using MyoSuitePenTwirlPixelEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoSuitePenTwirlPixelEnvSpec>;
  using MyoSuitePenTwirlPixelEnvPool =
      PyEnvPool<::myosuite_envpool::MyoSuitePenTwirlPixelEnvPool>;

  using MyoSuiteReorientEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoSuiteReorientEnvSpec>;
  using MyoSuiteReorientEnvPool =
      PyEnvPool<::myosuite_envpool::MyoSuiteReorientEnvPool>;
  using MyoSuiteReorientPixelEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoSuiteReorientPixelEnvSpec>;
  using MyoSuiteReorientPixelEnvPool =
      PyEnvPool<::myosuite_envpool::MyoSuiteReorientPixelEnvPool>;

  using MyoSuiteWalkEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoSuiteWalkEnvSpec>;
  using MyoSuiteWalkEnvPool =
      PyEnvPool<::myosuite_envpool::MyoSuiteWalkEnvPool>;
  using MyoSuiteWalkPixelEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoSuiteWalkPixelEnvSpec>;
  using MyoSuiteWalkPixelEnvPool =
      PyEnvPool<::myosuite_envpool::MyoSuiteWalkPixelEnvPool>;

  using MyoSuiteTerrainEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoSuiteTerrainEnvSpec>;
  using MyoSuiteTerrainEnvPool =
      PyEnvPool<::myosuite_envpool::MyoSuiteTerrainEnvPool>;
  using MyoSuiteTerrainPixelEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoSuiteTerrainPixelEnvSpec>;
  using MyoSuiteTerrainPixelEnvPool =
      PyEnvPool<::myosuite_envpool::MyoSuiteTerrainPixelEnvPool>;

  REGISTER(module, MyoSuitePoseEnvSpec, MyoSuitePoseEnvPool)
  REGISTER(module, MyoSuitePosePixelEnvSpec, MyoSuitePosePixelEnvPool)
  REGISTER(module, MyoSuiteReachEnvSpec, MyoSuiteReachEnvPool)
  REGISTER(module, MyoSuiteReachPixelEnvSpec, MyoSuiteReachPixelEnvPool)
  REGISTER(module, MyoSuiteKeyTurnEnvSpec, MyoSuiteKeyTurnEnvPool)
  REGISTER(module, MyoSuiteKeyTurnPixelEnvSpec, MyoSuiteKeyTurnPixelEnvPool)
  REGISTER(module, MyoSuiteObjHoldEnvSpec, MyoSuiteObjHoldEnvPool)
  REGISTER(module, MyoSuiteObjHoldPixelEnvSpec, MyoSuiteObjHoldPixelEnvPool)
  REGISTER(module, MyoSuiteTorsoEnvSpec, MyoSuiteTorsoEnvPool)
  REGISTER(module, MyoSuiteTorsoPixelEnvSpec, MyoSuiteTorsoPixelEnvPool)
  REGISTER(module, MyoSuitePenTwirlEnvSpec, MyoSuitePenTwirlEnvPool)
  REGISTER(module, MyoSuitePenTwirlPixelEnvSpec, MyoSuitePenTwirlPixelEnvPool)
  REGISTER(module, MyoSuiteReorientEnvSpec, MyoSuiteReorientEnvPool)
  REGISTER(module, MyoSuiteReorientPixelEnvSpec, MyoSuiteReorientPixelEnvPool)
  REGISTER(module, MyoSuiteWalkEnvSpec, MyoSuiteWalkEnvPool)
  REGISTER(module, MyoSuiteWalkPixelEnvSpec, MyoSuiteWalkPixelEnvPool)
  REGISTER(module, MyoSuiteTerrainEnvSpec, MyoSuiteTerrainEnvPool)
  REGISTER(module, MyoSuiteTerrainPixelEnvSpec, MyoSuiteTerrainPixelEnvPool)
}

}  // namespace myosuite_envpool
