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

#include "envpool/minigrid/impl/minigrid_env.h"

namespace minigrid {

void MiniGridEnv::MiniGridReset() {
  GenGrid();
  step_count_ = 0;
  CHECK(agent_pos_.first >= 0 && agent_pos_.second >= 0);
  CHECK(agent_dir_ >= 0);
  CHECK(grid_[agent_pos_.second][agent_pos_.first].GetType() == kEmpty);
  carrying_ = WorldObj(kEmpty);
}

void MiniGridEnv::PlaceAgent(int start_x, int start_y, int end_x, int end_y) {
  // Place an object at an empty position in the grid
  end_x = (end_x == -1) ? width_ - 1 : end_x;
  end_y = (end_y == -1) ? height_ - 1 : end_y;
  CHECK(start_x <= end_x && start_y <= end_y);
  std::uniform_int_distribution<> x_dist(start_x, end_x + 1);
  std::uniform_int_distribution<> y_dist(start_y, end_y + 1);
  while (true) {
    int x = x_dist(*gen_ref_);
    int y = y_dist(*gen_ref_);
    if (grid_[y][x].GetType() != kEmpty) continue;
    agent_pos_.first = x;
    agent_pos_.second = y;
    break;
  }
  // Randomly select a direction
  if (agent_start_dir_ == -1) {
    std::uniform_int_distribution<> dir_dist(0, 4);
    agent_dir_ = dir_dist(*gen_ref_);
  }
}

void MiniGridEnv::GenImage(Array& obs) {
  // Get the extents of the square set of tiles visible to the agent
  // Note: the bottom extent indices are not include in the set
  int topX, topY;
  if (agent_dir_ == 0) {
    topX = agent_pos_.first;
    topY = agent_pos_.second - (agent_view_size_ / 2);
  } else if (agent_dir_ == 1) {
    topX = agent_pos_.first - (agent_view_size_ / 2);
    topY = agent_pos_.second;
  } else if (agent_dir_ == 2) {
    topX = agent_pos_.first - agent_view_size_ + 1;
    topY = agent_pos_.second - (agent_view_size_ / 2);
  } else if (agent_dir_ == 3) {
    topX = agent_pos_.first - (agent_view_size_ / 2);
    topY = agent_pos_.second - agent_view_size_ + 1;
  } else
    CHECK(false);

  // Generate the sub-grid observed by the agent
  std::vector<std::vector<WorldObj>> agent_view_grid;
  for (int i = 0; i < agent_view_size_; ++i) {
    std::vector<WorldObj> temp_vec;
    for (int j = 0; j < agent_view_size_; ++j) {
      int x = topX + j;
      int y = topY + i;
      if (x >= 0 && x < width_ && y >= 0 && y < height_) {
        temp_vec.emplace_back(WorldObj(grid_[y][x].GetType()));
      } else {
        temp_vec.emplace_back(WorldObj(kWall));
      }
    }
    agent_view_grid.emplace_back(temp_vec);
  }
  // Rotate the agent view grid to relatively facing up
  for (int i = 0; i < agent_dir_ + 1; ++i) {
    // Rotate counter-clockwise
    std::vector<std::vector<WorldObj>> copy_grid =
        agent_view_grid;  // This is a deep copy
    for (int y = 0; y < agent_view_size_; ++y) {
      for (int x = 0; x < agent_view_size_; ++x) {
        copy_grid[agent_view_size_ - 1 - x][y] = agent_view_grid[y][x];
      }
    }
    agent_view_grid = copy_grid;
  }
  // Process occluders and visibility
  // Note that this incurs some performance cost
  int agent_pos_x = agent_view_size_ / 2;
  int agent_pos_y = agent_view_size_ - 1;
  bool vis_mask[agent_view_size_ * agent_view_size_];
  if (!see_through_walls_) {
    // TODO: process_vis
    memset(vis_mask, 0, sizeof(vis_mask));

  } else {
    memset(vis_mask, 1, sizeof(vis_mask));
  }
  // Let the agent see what it's carrying
  if (carrying_.GetType() != kEmpty)
    agent_view_grid[agent_pos_y][agent_pos_x] = carrying_;
  // TODO: Transpose the partially observable view into obs_
  for (int y = 0; y < agent_view_size_; ++y) {
    for (int x = 0; x < agent_view_size_; ++x) {
      if (vis_mask[y * agent_view_size_ + x] == true) {
        // Turn over to align with the python library
        obs(x, y, 0) = static_cast<uint8_t>(agent_view_grid[y][x].GetType());
        obs(x, y, 1) = static_cast<uint8_t>(agent_view_grid[y][x].GetColor());
        obs(x, y, 2) = static_cast<uint8_t>(agent_view_grid[y][x].GetState());
      }
    }
  }
}

}  // namespace minigrid