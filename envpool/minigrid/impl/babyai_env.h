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

#ifndef ENVPOOL_MINIGRID_IMPL_BABYAI_ENV_H_
#define ENVPOOL_MINIGRID_IMPL_BABYAI_ENV_H_

#include <memory>
#include <string>

#include "envpool/minigrid/impl/minigrid_env.h"

namespace minigrid {

inline constexpr int kBabyAIMissionBytes = 512;

struct BabyAITaskConfig {
  std::string env_name;
  int room_size{8};
  int num_rows{3};
  int num_cols{3};
  int num_dists{18};
  float locked_room_prob{0.5f};
  bool locations{true};
  bool unblocking{true};
  bool implicit_unlock{true};
  std::string action_kinds{"goto,pickup,open,putnext"};
  std::string instr_kinds{"action,and,seq"};
  bool doors_open{false};
  bool debug{false};
  std::string select_by;
  std::string first_color;
  std::string second_color;
  bool strict{false};
  int num_doors{2};
  int num_objs{8};
  int objs_per_room{4};
  bool start_carrying{false};
  bool distractors{false};
  Type obj_type{kBall};
  int max_steps{0};
  int mission_bytes{kBabyAIMissionBytes};
};

std::unique_ptr<MiniGridTask> MakeBabyAITask(const BabyAITaskConfig& config);

}  // namespace minigrid

#endif  // ENVPOOL_MINIGRID_IMPL_BABYAI_ENV_H_
