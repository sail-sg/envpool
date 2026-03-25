/*
 * Copyright 2026 Garena Online Private Limited
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

#ifndef ENVPOOL_MINIGRID_MINIGRID_H_
#define ENVPOOL_MINIGRID_MINIGRID_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/minigrid/impl/minigrid_env.h"

namespace minigrid {

class MiniGridEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "env_name"_.Bind(std::string("empty")), "size"_.Bind(8),
        "width"_.Bind(9), "height"_.Bind(7),
        "agent_start_pos"_.Bind(std::pair<int, int>(1, 1)),
        "agent_start_dir"_.Bind(0), "agent_view_size"_.Bind(7),
        "action_max"_.Bind(6), "num_crossings"_.Bind(1),
        "obstacle_type"_.Bind(std::string("lava")), "strip2_row"_.Bind(2),
        "n_obstacles"_.Bind(4), "num_objs"_.Bind(3),
        "random_length"_.Bind(false), "room_size"_.Bind(6), "num_rows"_.Bind(3),
        "num_cols"_.Bind(3), "obj_type"_.Bind(std::string("ball")),
        "min_num_rooms"_.Bind(2), "max_num_rooms"_.Bind(6),
        "max_room_size"_.Bind(10), "key_in_box"_.Bind(true),
        "blocked"_.Bind(true), "agent_room"_.Bind(std::pair<int, int>(1, 1)),
        "num_quarters"_.Bind(4), "num_rooms_visited"_.Bind(25));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    int agent_view_size = conf["agent_view_size"_];
    int bound = std::max({conf["size"_], conf["width"_], conf["height"_], 25});
    return MakeDict(
        "obs:direction"_.Bind(Spec<int>({-1}, {0, 3})),
        "obs:image"_.Bind(
            Spec<uint8_t>({agent_view_size, agent_view_size, 3}, {0, 255})),
        "obs:mission"_.Bind(Spec<uint8_t>({kMissionBytes}, {0, 255})),
        "info:agent_pos"_.Bind(Spec<int>({2}, {0, bound})),
        "info:mission_id"_.Bind(Spec<int>({-1}, {-1, 1024})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, conf["action_max"_]})));
  }
};

using MiniGridEnvSpec = EnvSpec<MiniGridEnvFns>;

class MiniGridEnv : public Env<MiniGridEnvSpec> {
 public:
  MiniGridEnv(const Spec& spec, int env_id);

  bool IsDone() override;
  void Reset() override;
  void Step(const Action& action) override;

  [[nodiscard]] MiniGridDebugState DebugState() const;

 private:
  std::unique_ptr<MiniGridTask> task_;

  void WriteState(float reward);
};

class MiniGridEnvPool : public AsyncEnvPool<MiniGridEnv> {
 public:
  using AsyncEnvPool<MiniGridEnv>::AsyncEnvPool;

  [[nodiscard]] std::vector<MiniGridDebugState> DebugStates(
      const std::vector<int>& env_ids) const;
};

}  // namespace minigrid

#endif  // ENVPOOL_MINIGRID_MINIGRID_H_
