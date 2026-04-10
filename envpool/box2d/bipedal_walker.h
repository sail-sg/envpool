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

#include <algorithm>
#include <array>
#include <vector>

#include "envpool/box2d/bipedal_walker_env.h"
#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace box2d {

class BipedalWalkerEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("reward_threshold"_.Bind(300.0), "hardcore"_.Bind(false));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
#ifdef ENVPOOL_TEST
    return MakeDict("obs"_.Bind(Spec<float>({24})),
                    "info:scroll"_.Bind(Spec<float>({-1})),
                    "info:path2"_.Bind(Spec<float>({199, 2, 2})),
                    "info:path4"_.Bind(
                        Spec<Container<float>>({-1}, Spec<float>({-1, 4, 2}))),
                    "info:path5"_.Bind(Spec<float>({1, 5, 2})),
                    "info:cloud_poly"_.Bind(Spec<float>({10, 5, 2})),
                    "info:hull_state"_.Bind(Spec<float>({7})),
                    "info:leg_states"_.Bind(Spec<float>({4, 7})),
                    "info:joint_states"_.Bind(Spec<float>({4, 4})),
                    "info:ground_contact"_.Bind(Spec<float>({4})),
                    "info:prev_shaping"_.Bind(Spec<float>({-1})),
                    "info:game_over"_.Bind(Spec<float>({-1})),
                    "info:initial_force"_.Bind(Spec<float>({-1})));
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
    auto state = Allocate();
    state["reward"_] = reward_;
    state["obs"_].Assign(obs_.data(), obs_.size());
#ifdef ENVPOOL_TEST
    state["info:scroll"_] = scroll_;
    state["info:path2"_].Assign(terrain_edge_path2_.data(),
                                terrain_edge_path2_.size());
    state["info:path5"_].Assign(hull_path5_.data(), hull_path5_.size());
    auto cloud_poly = CloudPolyState();
    state["info:cloud_poly"_].Assign(cloud_poly.data(), cloud_poly.size());
    auto hull_state = BodyState(hull_);
    state["info:hull_state"_].Assign(hull_state.data(), hull_state.size());
    std::array<float, 28> leg_states;
    for (int i = 0; i < 4; ++i) {
      auto leg_state = BodyState(legs_[i]);
      std::copy(leg_state.begin(), leg_state.end(),
                leg_states.begin() + i * leg_state.size());
    }
    state["info:leg_states"_].Assign(leg_states.data(), leg_states.size());
    std::array<float, 16> joint_states;
    for (int i = 0; i < 4; ++i) {
      joint_states[i * 4 + 0] = joints_[i]->GetJointAngle();
      joint_states[i * 4 + 1] = joints_[i]->GetJointSpeed();
      joint_states[i * 4 + 2] = joints_[i]->GetMotorSpeed();
      joint_states[i * 4 + 3] = joints_[i]->GetMaxMotorTorque();
    }
    state["info:joint_states"_].Assign(joint_states.data(),
                                       joint_states.size());
    state["info:ground_contact"_].Assign(ground_contact_.data(),
                                         ground_contact_.size());
    state["info:prev_shaping"_] = prev_shaping_;
    state["info:game_over"_] = done_ ? 1.0f : 0.0f;
    state["info:initial_force"_] = initial_force_;

    Container<float>& path4 = state["info:path4"_];
    std::vector<float> path4_data;
    path4_data.reserve(terrain_poly_path4_.size() + leg_path4_.size());
    path4_data.insert(path4_data.end(), terrain_poly_path4_.begin(),
                      terrain_poly_path4_.end());
    path4_data.insert(path4_data.end(), leg_path4_.begin(), leg_path4_.end());
    auto* array = new TArray<float>(::Spec<float>(
        std::vector<int>{static_cast<int>(path4_data.size()) / 8, 4, 2}));
    array->Assign(path4_data.data(), path4_data.size());
    path4.reset(array);
#endif
  }
};

using BipedalWalkerEnvPool = AsyncEnvPool<BipedalWalkerEnv>;

}  // namespace box2d

#endif  // ENVPOOL_BOX2D_BIPEDAL_WALKER_H_
