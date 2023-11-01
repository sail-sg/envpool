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

#include "envpool/minigrid/impl/minigrid_doorkey_env.h"

#include <utility>
#include <vector>

namespace minigrid {

MiniGridDoorKeyEnv::MiniGridDoorKeyEnv(int size,
                                       std::pair<int, int> agent_start_pos,
                                       int agent_start_dir, int max_steps,
                                       int agent_view_size) {
  width_ = size;
  height_ = size;
  agent_start_pos_ = agent_start_pos;
  agent_start_dir_ = agent_start_dir;
  see_through_walls_ = false;
  max_steps_ = max_steps;
  agent_view_size_ = agent_view_size;
}

void MiniGridDoorKeyEnv::GenGrid() {
  grid_.clear();
  for (int i = 0; i < height_; ++i) {
    std::vector<WorldObj> temp_vec(width_);
    for (int j = 0; j < width_; ++j) {
      temp_vec[j] = WorldObj(kEmpty);
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
  // generate a vertical splitting wall
  std::uniform_int_distribution<> x_dist(2, width_ - 3);
  int x = x_dist(*gen_ref_);
  for (int y = 0; y < height_; ++y) {
    grid_[y][x] = WorldObj(kWall, kGrey);
  }
  // place the agent at a random position and orientation
  // on the left side of the splitting wall
  PlaceAgent(1, 1, x - 1, height_ - 2);
  // place a door in the wall
  std::uniform_int_distribution<> y_dist(1, height_ - 3);
  int door_idx = y_dist(*gen_ref_);
  grid_[door_idx][x] = WorldObj(kDoor, kYellow);
  grid_[door_idx][x].SetDoorLocked(true);
  grid_[door_idx][x].SetDoorOpen(false);
  // place a yellow key on the left side
  auto pos = PlaceObject(1, 1, x - 1, height_ - 2);
  grid_[pos.second][pos.first] = WorldObj(kKey, kYellow);
}

}  // namespace minigrid
