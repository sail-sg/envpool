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
// https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py

#ifndef ENVPOOL_TOY_TEXT_TAXI_H_
#define ENVPOOL_TOY_TEXT_TAXI_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace toy_text {

class TaxiEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("reward_threshold"_.Bind(8.0), "is_rainy"_.Bind(false),
                    "rainy_probability"_.Bind(0.8),
                    "fickle_passenger"_.Bind(false),
                    "fickle_probability"_.Bind(0.3));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.Bind(Spec<int>({-1}, {0, 499})),
                    "info:prob"_.Bind(Spec<float>({-1})),
                    "info:action_mask"_.Bind(Spec<std::int8_t>({6}, {0, 1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 5})));
  }
};

using TaxiEnvSpec = EnvSpec<TaxiEnvFns>;

class TaxiEnv : public Env<TaxiEnvSpec> {
 protected:
  int x_, y_, s_, t_, max_episode_steps_, elapsed_step_;
  std::uniform_int_distribution<> dist_car_, dist_loc_;
  std::uniform_real_distribution<double> uniform_;
  bool is_rainy_, fickle_passenger_, fickle_step_{false};
  double rainy_probability_, fickle_probability_;
  bool done_{true};
  std::vector<std::vector<int>> loc_;
  std::vector<std::string> map_, loc_map_;

 public:
  TaxiEnv(const Spec& spec, int env_id)
      : Env<TaxiEnvSpec>(spec, env_id),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        dist_car_(0, 3),
        dist_loc_(0, 4),
        uniform_(0.0, 1.0),
        is_rainy_(spec.config["is_rainy"_]),
        fickle_passenger_(spec.config["fickle_passenger"_]),
        rainy_probability_(spec.config["rainy_probability"_]),
        fickle_probability_(spec.config["fickle_probability"_]),
        loc_({{0, 0}, {0, 4}, {4, 0}, {4, 3}}),
        map_({"|:|::|", "|:|::|", "|::::|", "||:|:|", "||:|:|"}),
        loc_map_({"0   1", "     ", "     ", "     ", "2  3 "}) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    x_ = dist_loc_(gen_);
    y_ = dist_loc_(gen_);
    s_ = dist_car_(gen_);
    do {
      t_ = dist_car_(gen_);
    } while (s_ == t_);
    fickle_step_ = fickle_passenger_ && uniform_(gen_) < fickle_probability_;
    done_ = false;
    elapsed_step_ = 0;
    WriteState(0.0, 1.0f);
  }

  void Step(const Action& action) override {
    done_ = (++elapsed_step_ >= max_episode_steps_);
    const int requested_action = action["action"_];
    const int previous_x = x_;
    const int previous_y = y_;
    const int previous_passenger = s_;
    float probability = 1.0f;
    int act = requested_action;
    if (is_rainy_ && requested_action < 4) {
      static constexpr std::array<std::array<int, 3>, 4> k_directions = {
          {{0, 2, 3}, {1, 3, 2}, {2, 1, 0}, {3, 0, 1}}};
      const double sample = uniform_(gen_);
      const double lateral_probability = (1.0 - rainy_probability_) / 2.0;
      const int direction = sample < rainy_probability_ ? 0
                            : sample < rainy_probability_ + lateral_probability
                                ? 1
                                : 2;
      probability = static_cast<float>(direction == 0 ? rainy_probability_
                                                      : lateral_probability);
      if (CanMove(requested_action)) {
        act = k_directions[requested_action][direction];
      }
    }
    float reward = -1.0;
    if (act == 0) {
      if (x_ < 4) {
        ++x_;
      }
    } else if (act == 1) {
      if (x_ > 0) {
        --x_;
      }
    } else if (act == 2) {
      if (map_[x_][y_ + 1] == ':') {
        ++y_;
      }
    } else if (act == 3) {
      if (map_[x_][y_] == ':') {
        --y_;
      }
    } else if (act == 4) {
      // pick up
      if (s_ < 4 && x_ == loc_[s_][0] && y_ == loc_[s_][1]) {
        s_ = 4;
      } else {
        reward = -10.0;
      }
    } else {
      // drop off
      if (s_ == 4 && x_ == loc_[t_][0] && y_ == loc_[t_][1]) {
        s_ = t_;
        done_ = true;
        reward = 20.0;
      } else if (s_ == 4 && loc_map_[x_][y_] != ' ') {
        s_ = loc_map_[x_][y_] - '0';
      } else {
        reward = -10.0;
      }
    }
    if (fickle_step_ && previous_passenger == 4 &&
        (x_ != previous_x || y_ != previous_y)) {
      fickle_step_ = false;
      int next_destination;
      do {
        next_destination = dist_car_(gen_);
      } while (next_destination == t_);
      t_ = next_destination;
    }
    WriteState(reward, probability);
  }

 private:
  [[nodiscard]] bool CanMove(int action) const {
    if (action == 0) {
      return x_ < 4;
    }
    if (action == 1) {
      return x_ > 0;
    }
    if (action == 2) {
      return y_ < 4 && map_[x_][y_ + 1] == ':';
    }
    return y_ > 0 && map_[x_][y_] == ':';
  }

  void WriteState(float reward, float probability) {
    auto state = Allocate();
    state["obs"_] = ((x_ * 5 + y_) * 5 + s_) * 4 + t_;
    state["reward"_] = reward;
    state["info:prob"_] = probability;
    for (int action = 0; action < 4; ++action) {
      state["info:action_mask"_][action] = CanMove(action);
    }
    state["info:action_mask"_][4] =
        s_ < 4 && x_ == loc_[s_][0] && y_ == loc_[s_][1];
    state["info:action_mask"_][5] = s_ == 4 && loc_map_[x_][y_] != ' ';
  }
};

using TaxiEnvPool = AsyncEnvPool<TaxiEnv>;

}  // namespace toy_text

#endif  // ENVPOOL_TOY_TEXT_TAXI_H_
