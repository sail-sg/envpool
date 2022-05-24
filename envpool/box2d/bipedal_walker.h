/*
 * Copyright 2022 Garena Online Private Limited
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

#ifndef ENVPOOL_BOX2D_BIPEDAL_WALKER_H_
#define ENVPOOL_BOX2D_BIPEDAL_WALKER_H_

#include "envpool/box2d/bipedal_walker_env.h"
#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace box2d {

class BipedalWalkerEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.Bind(1600),
                    "reward_threshold"_.Bind(300.0), "hardcore"_.Bind(false));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
#ifdef ENVPOOL_TEST
    return MakeDict("obs"_.Bind(Spec<float>({24})),
                    "info:scroll"_.Bind(Spec<float>({-1})),
                    "info:path2"_.Bind(Spec<float>({199, 2, 2})),
                    "info:path4"_.Bind(
                        Spec<Container<float>>({-1}, Spec<float>({-1, 4, 2}))),
                    "info:path5"_.Bind(Spec<float>({1, 5, 2})));
#else
    return MakeDict("obs"_.Bind(Spec<float>({24})));
#endif
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<float>({4}, {-1.0, 1.0})));
  }
};

using BipedalWalkerEnvSpec = EnvSpec<BipedalWalkerEnvFns>;

class BipedalWalkerEnv : public Env<BipedalWalkerEnvSpec>,
                         public BipedalWalkerBox2dEnv {
 public:
  BipedalWalkerEnv(const Spec& spec, int env_id)
      : Env<BipedalWalkerEnvSpec>(spec, env_id),
        BipedalWalkerBox2dEnv(spec.config["hardcore"_],
                              spec.config["max_episode_steps"_]) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    BipedalWalkerReset(&gen_);
    WriteState();
  }

  void Step(const Action& action) override {
    BipedalWalkerStep(&gen_, action["action"_][0], action["action"_][1],
                      action["action"_][2], action["action"_][3]);
    WriteState();
  }

 private:
  void WriteState() {
    State state = Allocate();
    state["reward"_] = reward_;
    state["obs"_].Assign(obs_.begin(), obs_.size());
#ifdef ENVPOOL_TEST
    state["info:scroll"_] = scroll_;
    state["info:path2"_].Assign(path2_.data(), path2_.size());
    state["info:path5"_].Assign(path5_.data(), path5_.size());

    Container<float>& path4 = state["info:path4"_];
    auto* array = new TArray<float>(::Spec<float>({path4_.size() / 8, 4, 2}));
    array->Assign(path4_.data(), path4_.size());
    path4.reset(array);
#endif
  }
};

using BipedalWalkerEnvPool = AsyncEnvPool<BipedalWalkerEnv>;

}  // namespace box2d

#endif  // ENVPOOL_BOX2D_BIPEDAL_WALKER_H_
