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
// https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py

#ifndef ENVPOOL_BOX2D_CAR_RACING_H_
#define ENVPOOL_BOX2D_CAR_RACING_H_

#include <array>
#include <vector>

#include "car_racing_env.h"
#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace box2d {

class CarRacingEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("reward_threshold"_.Bind(900.0),
                    "lap_complete_percent"_.Bind(0.95f));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
#ifdef ENVPOOL_TEST
    return MakeDict(
        "obs"_.Bind(Spec<uint8_t>({96, 96, 3}, {0, 255})),
        "info:tile_visited_count"_.Bind(Spec<int>({-1})),
        "info:car_fuel_spent"_.Bind(Spec<float>({-1})),
        "info:car_gas"_.Bind(Spec<float>({-1, 2})),
        "info:car_steer"_.Bind(Spec<float>({-1, 2})),
        "info:car_brake"_.Bind(Spec<float>({-1, 4})),
        "info:track"_.Bind(Spec<Container<float>>({-1}, Spec<float>({-1, 4}))),
        "info:road_poly"_.Bind(
            Spec<Container<float>>({-1}, Spec<float>({-1, 4, 2}))),
        "info:road_color"_.Bind(
            Spec<Container<float>>({-1}, Spec<float>({-1, 3}))),
        "info:car_body_states"_.Bind(Spec<float>({5, 7})),
        "info:car_wheel_states"_.Bind(Spec<float>({4, 5})),
        "info:car_reward"_.Bind(Spec<float>({-1})),
        "info:car_prev_reward"_.Bind(Spec<float>({-1})),
        "info:car_time"_.Bind(Spec<float>({-1})),
        "info:new_lap"_.Bind(Spec<float>({-1})));
#else
    return MakeDict("obs"_.Bind(Spec<uint8_t>({96, 96, 3}, {0, 255})));
#endif
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<float>({3}, {{-1, 0, 0}, {1, 1, 1}})));
  }
};

using CarRacingEnvSpec = EnvSpec<CarRacingEnvFns>;

class CarRacingEnv : public Env<CarRacingEnvSpec>, public CarRacingBox2dEnv {
 public:
  CarRacingEnv(const Spec& spec, int env_id)
      : Env<CarRacingEnvSpec>(spec, env_id),
        CarRacingBox2dEnv(spec.config["max_episode_steps"_],
                          spec.config["lap_complete_percent"_]) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    CarRacingReset(&gen_);
    WriteState();
  }

  void Step(const Action& action) override {
    CarRacingStep(&gen_, action["action"_][0], action["action"_][1],
                  action["action"_][2]);
    WriteState();
  }

 private:
  void WriteState() {
    auto state = Allocate();
    state["reward"_] = step_reward_;
    CreateImageArray();
    state["obs"_].Assign(img_array_.data, 96 * 96 * 3);
#ifdef ENVPOOL_TEST
    state["info:tile_visited_count"_] = tile_visited_count_;
    state["info:car_fuel_spent"_] = car_->GetFuelSpent();
    auto car_gas = car_->GetGas();
    auto car_steer = car_->GetSteer();
    auto car_brake = car_->GetBrake();
    state["info:car_gas"_].Assign(car_gas.data(), car_gas.size());
    state["info:car_steer"_].Assign(car_steer.data(), car_steer.size());
    state["info:car_brake"_].Assign(car_brake.data(), car_brake.size());
    auto track = TrackState();
    Container<float>& track_container = state["info:track"_];
    auto* track_array = new TArray<float>(
        ::Spec<float>(std::vector<int>{static_cast<int>(track.size()) / 4, 4}));
    track_array->Assign(track.data(), track.size());
    track_container.reset(track_array);

    auto road_poly = RoadPolyState();
    Container<float>& road_poly_container = state["info:road_poly"_];
    auto* road_poly_array = new TArray<float>(::Spec<float>(
        std::vector<int>{static_cast<int>(road_poly.size()) / 8, 4, 2}));
    road_poly_array->Assign(road_poly.data(), road_poly.size());
    road_poly_container.reset(road_poly_array);

    auto road_color = RoadColorState();
    Container<float>& road_color_container = state["info:road_color"_];
    auto* road_color_array = new TArray<float>(::Spec<float>(
        std::vector<int>{static_cast<int>(road_color.size()) / 3, 3}));
    road_color_array->Assign(road_color.data(), road_color.size());
    road_color_container.reset(road_color_array);

    auto body_states = CarBodyStates();
    auto wheel_states = CarWheelStates();
    state["info:car_body_states"_].Assign(body_states.data(),
                                          body_states.size());
    state["info:car_wheel_states"_].Assign(wheel_states.data(),
                                           wheel_states.size());
    state["info:car_reward"_] = reward_;
    state["info:car_prev_reward"_] = prev_reward_;
    state["info:car_time"_] = t_;
    state["info:new_lap"_] = new_lap_ ? 1.0f : 0.0f;
#endif
  }
};

using CarRacingEnvPool = AsyncEnvPool<CarRacingEnv>;

}  // namespace box2d

#endif  // ENVPOOL_BOX2D_CAR_RACING_H_
