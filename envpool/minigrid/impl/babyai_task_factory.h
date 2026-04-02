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

#ifndef ENVPOOL_MINIGRID_IMPL_BABYAI_TASK_FACTORY_H_
#define ENVPOOL_MINIGRID_IMPL_BABYAI_TASK_FACTORY_H_

#include <memory>

#include "envpool/minigrid/impl/babyai_env.h"

namespace minigrid {

std::unique_ptr<MiniGridTask> MakeBabyAIGoToTask(
    const BabyAITaskConfig& config);
std::unique_ptr<MiniGridTask> MakeBabyAIPickupTask(
    const BabyAITaskConfig& config);
std::unique_ptr<MiniGridTask> MakeBabyAIOpenTask(
    const BabyAITaskConfig& config);
std::unique_ptr<MiniGridTask> MakeBabyAIUnlockTask(
    const BabyAITaskConfig& config);

}  // namespace minigrid

#endif  // ENVPOOL_MINIGRID_IMPL_BABYAI_TASK_FACTORY_H_
