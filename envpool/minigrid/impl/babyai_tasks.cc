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

#include <memory>

#include "envpool/minigrid/impl/babyai_task_factory.h"

namespace minigrid {

std::unique_ptr<MiniGridTask> MakeBabyAITask(const BabyAITaskConfig& config) {
  if (auto task = MakeBabyAIGoToTask(config); task != nullptr) {
    return task;
  }
  if (auto task = MakeBabyAIPickupTask(config); task != nullptr) {
    return task;
  }
  if (auto task = MakeBabyAIOpenTask(config); task != nullptr) {
    return task;
  }
  return MakeBabyAIUnlockTask(config);
}

}  // namespace minigrid
