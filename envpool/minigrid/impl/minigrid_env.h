/*
 * Copyright 2023 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENVPOOL_MINIGRID_IMPL_MINIGRID_ENV_H_
#define ENVPOOL_MINIGRID_IMPL_MINIGRID_ENV_H_

#include <random>
#include <utility>
#include <vector>

#include "envpool/core/array.h"
#include "envpool/minigrid/impl/utils.h"

namespace minigrid {

class MiniGridEnv {
 protected:
  int width_;
  int height_;
  int max_steps_{100};
  int step_count_{0};
  int agent_view_size_{7};
  bool see_through_walls_{false};
  bool done_{true};
  std::pair<int, int> agent_start_pos_;
  std::pair<int, int> agent_pos_;
  int agent_start_dir_;
  int agent_dir_;
  std::mt19937* gen_ref_;
  std::vector<std::vector<WorldObj>> grid_;
  WorldObj carrying_;

 public:
  MiniGridEnv() { carrying_ = WorldObj(kEmpty); }
  void MiniGridReset();
  float MiniGridStep(Act act);
  void PlaceAgent(int start_x = 0, int start_y = 0, int end_x = -1,
                  int end_y = -1);
  void GenImage(const Array& obs);
  virtual void GenGrid() {}
};

}  // namespace minigrid

#endif  // ENVPOOL_MINIGRID_IMPL_MINIGRID_ENV_H_
