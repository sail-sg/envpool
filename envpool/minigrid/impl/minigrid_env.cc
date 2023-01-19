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
  done_ = false;
  CHECK(agent_pos_.first >= 0 && agent_pos_.second >= 0);
  CHECK(agent_dir_ >= 0);
  CHECK(grid_[agent_pos_.second][agent_pos_.first].GetType() == kEmpty);
  carrying_ = WorldObj(kEmpty);
}

float MiniGridEnv::MiniGridStep(Act act) {
  step_count_ += 1;
  float reward = 0.0;
  // Get the position in front of the agent
  std::pair<int, int> fwd_pos = agent_pos_;
  switch(agent_dir_) {
    case 0:
      fwd_pos.first += 1;
      break;
    case 1: 
      fwd_pos.second += 1;
      break;
    case 2:
      fwd_pos.first -= 1;
      break;
    case 3:
      fwd_pos.second -= 1;
      break;
    default: 
      CHECK(false);
      break;
  }
  CHECK(fwd_pos.first >= 0 );
  CHECK(fwd_pos.first < width_);
  CHECK(fwd_pos.second >= 0);
  CHECK(fwd_pos.second < height_);
  // Get the forward cell object
  if (act == kLeft) {
    agent_dir_ -= 1;
    if (agent_dir_ < 0) {
      agent_dir_ += 4;
    }
  } else if (act == kRight) {
    agent_dir_ = (agent_dir_ + 1) % 4;
  } else if (act == kForward) {
    if (grid_[fwd_pos.second][fwd_pos.first].CanOverlap()) {
      agent_pos_ = fwd_pos;
    }
    if (grid_[fwd_pos.second][fwd_pos.first].GetType() == kGoal) {
      done_ = true;
      reward = 1 - 0.9 * ((float)step_count_ / max_steps_);
    } else if (grid_[fwd_pos.second][fwd_pos.first].GetType() == kLava) {
      done_ = true;
    }
  } else if (act == kPickup) {
    if (carrying_.GetType() == kEmpty && grid_[fwd_pos.second][fwd_pos.first].CanPickup()) {
        carrying_ = grid_[fwd_pos.second][fwd_pos.first];
        grid_[fwd_pos.second][fwd_pos.first] = WorldObj(kEmpty);
    }
  } else if (act == kDrop) {
    if (carrying_.GetType() != kEmpty && grid_[fwd_pos.second][fwd_pos.first].GetType() == kEmpty) {
      grid_[fwd_pos.second][fwd_pos.first] = carrying_;
      carrying_ = WorldObj(kEmpty);
    }
  } else if (act == kToggle) {
    WorldObj obj = grid_[fwd_pos.second][fwd_pos.first];
    if (obj.GetType() == kDoor) {
      if (obj.GetDoorLocked()) {
        // If the agent has the right key to open the door
        if (carrying_.GetType() == kKey && carrying_.GetColor() == obj.GetColor()) {
          grid_[fwd_pos.second][fwd_pos.first].SetDoorOpen(true);
        }
      } else {
        grid_[fwd_pos.second][fwd_pos.first].SetDoorOpen(!obj.GetDoorOpen());
      }
    } else if (obj.GetType() == kBox) {
      // TODO: box
    }
  } else if (act != kDone) {
    CHECK(false);
  }  
  if (step_count_ >= max_steps_) {
    done_ = true;
  }
  return reward;
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
    // TODO: Process_vis
    memset(vis_mask, 0, sizeof(vis_mask));
    vis_mask[agent_pos_y * agent_view_size_ + agent_pos_x] = true;
  } else {
    memset(vis_mask, 1, sizeof(vis_mask));
  }
  // Let the agent see what it's carrying
  if (carrying_.GetType() != kEmpty) {
    agent_view_grid[agent_pos_y][agent_pos_x] = carrying_;
  } else {
    agent_view_grid[agent_pos_y][agent_pos_x] = WorldObj(kEmpty);
  }
  for (int y = 0; y < agent_view_size_; ++y) {
    for (int x = 0; x < agent_view_size_; ++x) {
      if (vis_mask[y * agent_view_size_ + x] == true) {
        // Transpose to align with the python library
        obs(x, y, 0) = static_cast<uint8_t>(agent_view_grid[y][x].GetType());
        obs(x, y, 1) = static_cast<uint8_t>(agent_view_grid[y][x].GetColor());
        obs(x, y, 2) = static_cast<uint8_t>(agent_view_grid[y][x].GetState());
      }
    }
  }
}

}  // namespace minigrid