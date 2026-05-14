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

PYBIND11_MODULE(playground_envpool, m) {
  RegisterPlaygroundAloha(m);
  RegisterPlaygroundApollo(m);
  RegisterPlaygroundBarkour(m);
  RegisterPlaygroundBerkeleyHumanoid(m);
  RegisterPlaygroundG1(m);
  RegisterPlaygroundGo1(m);
  RegisterPlaygroundH1(m);
  RegisterPlaygroundHand(m);
  RegisterPlaygroundOp3(m);
  RegisterPlaygroundPanda(m);
  RegisterPlaygroundPandaRobotiq(m);
  RegisterPlaygroundSpot(m);
  RegisterPlaygroundT1(m);
}
