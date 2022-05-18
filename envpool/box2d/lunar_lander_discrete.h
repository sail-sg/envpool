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

#ifndef ENVPOOL_BOX2D_LUNAR_LANDER_DISCRETE_H_
#define ENVPOOL_BOX2D_LUNAR_LANDER_DISCRETE_H_

#include "envpool/box2d/lunar_lander.h"
#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace box2d {

class LunarLanderDiscreteEnvFns {
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
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 3})));
  }
};

using LunarLanderDiscreteEnvSpec = EnvSpec<LunarLanderDiscreteEnvFns>;

class LunarLanderDiscreteEnv : public Env<LunarLanderDiscreteEnvSpec>,
                               public LunarLanderEnv {
 public:
  LunarLanderDiscreteEnv(const Spec& spec, int env_id)
      : Env<LunarLanderDiscreteEnvSpec>(spec, env_id),
        LunarLanderEnv(false, spec.config["max_episode_steps"_]) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    LunarLanderReset(&gen_);
    WriteState();
  }

  void Step(const Action& action) override {
    int act = action["action"_];
    LunarLanderStep(&gen_, act, 0, 0);
    WriteState();
  }

 private:
  void WriteState() {
    State state = Allocate();
    state["reward"_] = reward_;
    state["obs"_].Assign(obs_.begin(), obs_.size());
  }
};

using LunarLanderDiscreteEnvPool = AsyncEnvPool<LunarLanderDiscreteEnv>;

}  // namespace box2d

#endif  // ENVPOOL_BOX2D_LUNAR_LANDER_DISCRETE_H_
