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

#include <type_traits>

#include "envpool/mujoco/myosuite/myosuite_env.h"

namespace {

static_assert(
    std::is_same_v<myosuite::MyoSuiteEnvPool::Spec, myosuite::MyoSuiteEnvSpec>);
static_assert(std::is_same_v<myosuite::MyoSuitePixelEnvPool::Spec,
                             myosuite::MyoSuitePixelEnvSpec>);

}  // namespace
