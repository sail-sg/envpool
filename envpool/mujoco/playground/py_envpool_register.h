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

#ifndef ENVPOOL_MUJOCO_PLAYGROUND_PY_ENVPOOL_REGISTER_H_
#define ENVPOOL_MUJOCO_PLAYGROUND_PY_ENVPOOL_REGISTER_H_

#include <pybind11/pybind11.h>

void RegisterPlaygroundAloha(pybind11::module_& m);
void RegisterPlaygroundApollo(pybind11::module_& m);
void RegisterPlaygroundBarkour(pybind11::module_& m);
void RegisterPlaygroundBerkeleyHumanoid(pybind11::module_& m);
void RegisterPlaygroundG1(pybind11::module_& m);
void RegisterPlaygroundGo1(pybind11::module_& m);
void RegisterPlaygroundH1(pybind11::module_& m);
void RegisterPlaygroundHand(pybind11::module_& m);
void RegisterPlaygroundOp3(pybind11::module_& m);
void RegisterPlaygroundPanda(pybind11::module_& m);
void RegisterPlaygroundPandaRobotiq(pybind11::module_& m);
void RegisterPlaygroundSpot(pybind11::module_& m);
void RegisterPlaygroundT1(pybind11::module_& m);

#endif  // ENVPOOL_MUJOCO_PLAYGROUND_PY_ENVPOOL_REGISTER_H_
