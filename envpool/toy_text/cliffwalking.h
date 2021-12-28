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
// https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py

#ifndef ENVPOOL_TOY_TEXT_CLIFFWALKING_H_
#define ENVPOOL_TOY_TEXT_CLIFFWALKING_H_

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace toy_text {

class CliffWalkingEnvFns {
 public:
  static decltype(auto) DefaultConfig() { return MakeDict(); }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.bind(Spec<int>({-1}, {0, 47})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.bind(Spec<int>({-1}, {0, 3})));
  }
};

typedef class EnvSpec<CliffWalkingEnvFns> CliffWalkingEnvSpec;

class CliffWalkingEnv : public Env<CliffWalkingEnvSpec> {
 protected:
  int x_, y_;
  bool done_;

 public:
  CliffWalkingEnv(const Spec& spec, int env_id)
      : Env<CliffWalkingEnvSpec>(spec, env_id), done_(true) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    x_ = 3;
    y_ = 0;
    done_ = false;
    State state = Allocate();
    WriteObs(state, 0.0f);
  }

  void Step(const Action& action) override {
    int act = action["action"_];
    float reward = -1.0f;
    if (act == 0) {
      --x_;
    } else if (act == 1) {
      ++y_;
    } else if (act == 2) {
      ++x_;
    } else {
      --y_;
    }
    x_ = std::min(3, std::max(0, x_));
    y_ = std::min(11, std::max(0, y_));
    if (x_ == 3 && y_ > 0 && y_ < 11) {
      reward = -100.0f;
      x_ = 3;
      y_ = 0;
    }
    if (x_ == 3 && y_ == 11) {
      done_ = true;
    }
    State state = Allocate();
    WriteObs(state, reward);
  }

 private:
  void WriteObs(State& state, float reward) {  // NOLINT
    state["obs"_] = x_ * 12 + y_;
    state["reward"_] = reward;
  }
};

typedef AsyncEnvPool<CliffWalkingEnv> CliffWalkingEnvPool;

}  // namespace toy_text

#endif  // ENVPOOL_TOY_TEXT_CLIFFWALKING_H_
