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

#ifndef ENVPOOL_BOX2D_LUNAR_LANDER_CONTINUOUS_H_
#define ENVPOOL_BOX2D_LUNAR_LANDER_CONTINUOUS_H_

#include "envpool/box2d/lunar_lander_env.h"
#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace box2d {

class LunarLanderContinuousEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.Bind(1000),
                    "reward_threshold"_.Bind(200.0));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.Bind(Spec<float>({8})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<float>({2}, {-1.0, 1.0})));
  }
};

using LunarLanderContinuousEnvSpec = EnvSpec<LunarLanderContinuousEnvFns>;

class LunarLanderContinuousEnv : public Env<LunarLanderContinuousEnvSpec>,
                                 public LunarLanderBox2dEnv {
 public:
  LunarLanderContinuousEnv(const Spec& spec, int env_id)
      : Env<LunarLanderContinuousEnvSpec>(spec, env_id),
        LunarLanderBox2dEnv(true, spec.config["max_episode_steps"_]) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    LunarLanderReset(&gen_);
    WriteState();
  }

  void Step(const Action& action) override {
    float action0 = action["action"_][0];
    float action1 = action["action"_][1];
    LunarLanderStep(&gen_, 0, action0, action1);
    WriteState();
  }

 private:
  void WriteState() {
    State state = Allocate();
    state["reward"_] = reward_;
    state["obs"_].Assign(obs_.begin(), obs_.size());
  }
};

using LunarLanderContinuousEnvPool = AsyncEnvPool<LunarLanderContinuousEnv>;

}  // namespace box2d

#endif  // ENVPOOL_BOX2D_LUNAR_LANDER_CONTINUOUS_H_
