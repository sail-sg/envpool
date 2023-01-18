// Copyright 2023 Garena Online Private Limited
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

#include "envpool/minigrid/impl/minigrid_empty_env.h"

namespace minigrid {

MiniGridEmptyEnv::MiniGridEmptyEnv(int size,
                                   std::pair<int, int> agent_start_pos, int agent_start_dir,
                                   int max_steps, int agent_view_size) {
  width_ = size;
  height_ = size;
  agent_start_pos_ = agent_start_pos;
  agent_start_dir_ = agent_start_dir;
  see_through_walls_ = true;
  max_steps_ = max_steps;
  agent_view_size_ = agent_view_size;
}

void MiniGridEmptyEnv::GenGrid() {
  grid_.clear();
  for (int i = 0; i < height_; ++i) {
    std::vector<WorldObj> temp_vec;
    for (int j = 0; j < width_; ++j) {
      temp_vec.emplace_back(WorldObj(kEmpty));
    }
    grid_.emplace_back(temp_vec);
  }
  // generate the surrounding walls
  for (int i = 0; i < width_; ++i) {
    grid_[0][i] = WorldObj(kWall, kGrey);
    grid_[height_ - 1][i] = WorldObj(kWall, kGrey);
  }
  for (int i = 0; i < height_; ++i) {
    grid_[i][0] = WorldObj(kWall, kGrey);
    grid_[i][width_ - 1] = WorldObj(kWall, kGrey);
  }
  // place a goal square in the bottom-right corner
  grid_[height_ - 2][width_ - 2] = WorldObj(kGoal, kGreen);
  // place the agent
  if (agent_start_pos_.first == -1) {
    PlaceAgent(1, 1, width_ - 2, height_ - 2);
  } else {
    agent_pos_ = agent_start_pos_;
    agent_dir_ = agent_start_dir_;
  }
}

}  // namespace minigrid