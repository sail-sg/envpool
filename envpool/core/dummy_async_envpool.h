/*
 * Copyright 2021 Garena Online Private Limited
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

#ifndef ENVPOOL_CORE_DUMMY_ASYNC_ENVPOOL_H_
#define ENVPOOL_CORE_DUMMY_ASYNC_ENVPOOL_H_

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace dummy {

class DummyAsyncEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("state_num"_.bind(10), "action_num"_.bind(6));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.bind(Spec<int>({-1, conf["state_num"_]})),
                    "reward"_.bind(Spec<float>({-1})),
                    "info:players.done"_.bind(Spec<bool>({-1})),
                    "info:players.id"_.bind(Spec<int>({-1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("players.action"_.bind(Spec<int>({-1})),
                    "players.id"_.bind(Spec<int>({-1})));
  }
};

typedef class EnvSpec<DummyAsyncEnvFns> DummyAsyncEnvSpec;

class DummyAsyncEnv : public Env<DummyAsyncEnvSpec> {
 protected:
  int state_, max_num_players_;

 public:
  DummyAsyncEnv(const Spec& spec, int env_id)
      : Env<DummyAsyncEnvSpec>(spec, env_id),
        state_(0),
        max_num_players_(spec.config["max_num_players"_]) {
    if (seed_ < 1) {
      seed_ = 1;
    }
  }
  void Reset() override {
    state_ = 0;
    int num_players =
        max_num_players_ <= 1 ? 1 : state_ % (max_num_players_ - 1) + 1;
    auto state = Allocate(num_players);
    for (int i = 0; i < num_players; ++i) {
      state["info:players.id"_][i] = i;
      state["info:players.done"_][i] = IsDone();
      state["obs"_](i, 0) = state_;
      state["obs"_](i, 1) = 0;
      state["reward"_][i] = -i;
    }
  }
  void Step(const Action& action) override {
    ++state_;
    int num_players =
        max_num_players_ <= 1 ? 1 : state_ % (max_num_players_ - 1) + 1;
    auto state = Allocate(num_players);
    int action_num = action["players.env_id"_].Shape(0);
    for (int i = 0; i < action_num; ++i) {
      if (static_cast<int>(action["players.env_id"_][i]) != env_id_) {
        action_num = 0;
      }
    }
    for (int i = 0; i < num_players; ++i) {
      state["info:players.id"_][i] = i;
      state["info:players.done"_][i] = IsDone();
      state["obs"_](i, 0) = state_;
      state["obs"_](i, 1) = action_num;
      state["reward"_][i] = -i;
    }
  }
  bool IsDone() override { return state_ >= seed_; }
};

typedef AsyncEnvPool<DummyAsyncEnv> DummyAsyncEnvPool;

}  // namespace dummy

#endif  // ENVPOOL_CORE_DUMMY_ASYNC_ENVPOOL_H_
