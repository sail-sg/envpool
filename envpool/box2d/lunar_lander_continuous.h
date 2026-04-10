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

#include <algorithm>
#include <array>

#include "envpool/box2d/lunar_lander_env.h"
#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace box2d {

class LunarLanderContinuousEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("reward_threshold"_.Bind(200.0));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
#ifdef ENVPOOL_TEST
    return MakeDict("obs"_.Bind(Spec<float>({8})),
                    "info:sky_polys"_.Bind(Spec<float>({10, 4, 2})),
                    "info:lander_state"_.Bind(Spec<float>({7})),
                    "info:leg_states"_.Bind(Spec<float>({2, 7})),
                    "info:ground_contact"_.Bind(Spec<float>({2})),
                    "info:prev_shaping"_.Bind(Spec<float>({-1})),
                    "info:game_over"_.Bind(Spec<float>({-1})),
                    "info:last_dispersion"_.Bind(Spec<float>({2})),
                    "info:initial_force"_.Bind(Spec<float>({2})));
#else
    return MakeDict("obs"_.Bind(Spec<float>({8})));
#endif
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
    auto state = Allocate();
    state["reward"_] = reward_;
    state["obs"_].Assign(obs_.data(), obs_.size());
#ifdef ENVPOOL_TEST
    auto sky_poly = SkyPolyState();
    state["info:sky_polys"_].Assign(sky_poly.data(), sky_poly.size());
    auto lander_state = BodyState(lander_);
    state["info:lander_state"_].Assign(lander_state.data(),
                                       lander_state.size());
    std::array<float, 14> leg_states;
    for (int i = 0; i < 2; ++i) {
      auto leg_state = BodyState(legs_[i]);
      std::copy(leg_state.begin(), leg_state.end(),
                leg_states.begin() + i * leg_state.size());
    }
    state["info:leg_states"_].Assign(leg_states.data(), leg_states.size());
    state["info:ground_contact"_].Assign(ground_contact_.data(),
                                         ground_contact_.size());
    state["info:prev_shaping"_] = prev_shaping_;
    state["info:game_over"_] = done_ ? 1.0f : 0.0f;
    state["info:last_dispersion"_].Assign(last_dispersion_.data(),
                                          last_dispersion_.size());
    state["info:initial_force"_].Assign(initial_force_.data(),
                                        initial_force_.size());
#endif
  }
};

using LunarLanderContinuousEnvPool = AsyncEnvPool<LunarLanderContinuousEnv>;

}  // namespace box2d

#endif  // ENVPOOL_BOX2D_LUNAR_LANDER_CONTINUOUS_H_
