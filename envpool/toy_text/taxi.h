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
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace toy_text {

class TaxiEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.bind(200),
                    "reward_threshold"_.bind(8.0f));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.bind(Spec<int>({-1}, {0, 499})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.bind(Spec<int>({-1}, {0, 5})));
  }
};

typedef class EnvSpec<TaxiEnvFns> TaxiEnvSpec;

class TaxiEnv : public Env<TaxiEnvSpec> {
 protected:
  int x_, y_, s_, t_, max_episode_steps_, elapsed_step_;
  std::uniform_int_distribution<> dist_car_, dist_loc_;
  bool done_;
  std::vector<std::vector<int>> loc_;
  std::vector<std::string> map_, loc_map_;

 public:
  TaxiEnv(const Spec& spec, int env_id)
      : Env<TaxiEnvSpec>(spec, env_id),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        dist_car_(0, 3),
        dist_loc_(0, 4),
        done_(true),
        loc_({{0, 0}, {0, 4}, {4, 0}, {4, 3}}),
        map_({"|:|::|", "|:|::|", "|::::|", "||:|:|", "||:|:|"}),
        loc_map_({"0   1", "     ", "     ", "     ", "2  3 "}) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    x_ = dist_loc_(gen_);
    y_ = dist_loc_(gen_);
    s_ = dist_car_(gen_);
    t_ = dist_car_(gen_);
    done_ = false;
    elapsed_step_ = 0;
    State state = Allocate();
    WriteObs(state, 0.0f);
  }

  void Step(const Action& action) override {
    done_ = (++elapsed_step_ >= max_episode_steps_);
    int act = action["action"_];
    float reward = -1.0f;
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
        reward = -10.0f;
      }
    } else {
      // drop off
      if (s_ == 4 && x_ == loc_[t_][0] && y_ == loc_[t_][1]) {
        s_ = t_;
        done_ = true;
        reward = 20.0f;
      } else if (s_ == 4 && loc_map_[x_][y_] != ' ') {
        s_ = loc_map_[x_][y_] - '0';
      } else {
        reward = -10.0f;
      }
    }
    State state = Allocate();
    WriteObs(state, reward);
  }

 private:
  void WriteObs(State& state, float reward) {  // NOLINT
    state["obs"_] = ((x_ * 5 + y_) * 5 + s_) * 4 + t_;
    state["reward"_] = reward;
  }
};

typedef AsyncEnvPool<TaxiEnv> TaxiEnvPool;

}  // namespace toy_text

#endif  // ENVPOOL_TOY_TEXT_TAXI_H_
