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

#include "envpool/minigrid/impl/minigrid_env.h"

#include <string>
#include <vector>

#include "envpool/minigrid/impl/minigrid_task_factory.h"
#include "envpool/minigrid/minigrid.h"

namespace minigrid {

MiniGridEnv::MiniGridEnv(const Spec& spec, int env_id)
    : Env<MiniGridEnvSpec>(spec, env_id) {
  task_ = MakeMiniGridTask(spec.config);
  task_->SetGenerator(&gen_);
}

bool MiniGridEnv::IsDone() { return task_->IsDone(); }

void MiniGridEnv::Reset() {
  task_->Reset();
  WriteState(0.0f);
}

void MiniGridEnv::Step(const Action& action) {
  WriteState(task_->Step(static_cast<Act>(action["action"_])));
}

std::pair<int, int> MiniGridEnv::RenderSize(int width, int height) const {
  return task_->RenderSize(width, height);
}

void MiniGridEnv::Render(int width, int height, int /*camera_id*/,
                         unsigned char* rgb) {
  task_->Render(width, height, rgb);
}

MiniGridDebugState MiniGridEnv::DebugState() const {
  return task_->DebugState();
}

int MiniGridEnv::CurrentMaxEpisodeSteps() const { return task_->MaxSteps(); }

void MiniGridEnv::WriteState(float reward) {
  auto state = Allocate();
  task_->GenImage(state["obs:image"_]);
  task_->WriteMission(state["obs:mission"_]);
  state["obs:direction"_] = task_->AgentDir();
  state["reward"_] = reward;
  state["info:agent_pos"_](0) = task_->AgentPos().first;
  state["info:agent_pos"_](1) = task_->AgentPos().second;
  state["info:mission_id"_] = task_->MissionId();
}

std::vector<MiniGridDebugState> MiniGridEnvPool::DebugStates(
    const std::vector<int>& env_ids) const {
  std::vector<MiniGridDebugState> states;
  states.reserve(env_ids.size());
  for (int env_id : env_ids) {
    states.emplace_back(envs_[env_id]->DebugState());
  }
  return states;
}

}  // namespace minigrid
