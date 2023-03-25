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

#ifndef ENVPOOL_MINIGRID_EMPTY_H_
#define ENVPOOL_MINIGRID_EMPTY_H_

#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/minigrid/impl/minigrid_empty_env.h"
#include "envpool/minigrid/impl/minigrid_env.h"

namespace minigrid {

class EmptyEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("size"_.Bind(8),
                    "agent_start_pos"_.Bind(std::pair<int, int>(1, 1)),
                    "agent_start_dir"_.Bind(0), "agent_view_size"_.Bind(7));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    int agent_view_size = conf["agent_view_size"_];
    int size = conf["size"_];
    return MakeDict("obs:direction"_.Bind(Spec<int>({-1}, {0, 3})),
                    "obs:image"_.Bind(Spec<uint8_t>(
                        {agent_view_size, agent_view_size, 3}, {0, 255})),
                    "info:agent_pos"_.Bind(Spec<int>({2}, {0, size})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 6})));
  }
};

using EmptyEnvSpec = EnvSpec<EmptyEnvFns>;
using FrameSpec = Spec<uint8_t>;

class EmptyEnv : public Env<EmptyEnvSpec>, public MiniGridEmptyEnv {
 public:
  EmptyEnv(const Spec& spec, int env_id)
      : Env<EmptyEnvSpec>(spec, env_id),
        MiniGridEmptyEnv(spec.config["size"_], spec.config["agent_start_pos"_],
                         spec.config["agent_start_dir"_],
                         spec.config["max_episode_steps"_],
                         spec.config["agent_view_size"_]) {
    gen_ref_ = &gen_;
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    MiniGridReset();
    WriteState(0.0);
  }

  void Step(const Action& action) override {
    int act = action["action"_];
    WriteState(MiniGridStep(static_cast<Act>(act)));
  }

 private:
  void WriteState(float reward) {
    State state = Allocate();
    GenImage(state["obs:image"_]);
    state["obs:direction"_] = agent_dir_;
    state["reward"_] = reward;
    state["info:agent_pos"_](0) = agent_pos_.first;
    state["info:agent_pos"_](1) = agent_pos_.second;
  }
};

using EmptyEnvPool = AsyncEnvPool<EmptyEnv>;

}  // namespace minigrid

#endif  // ENVPOOL_MINIGRID_EMPTY_H_
