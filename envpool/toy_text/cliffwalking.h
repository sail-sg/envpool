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

#include <array>
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
  static decltype(auto) DefaultConfig() {
    return MakeDict("is_slippery"_.Bind(false));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.Bind(Spec<int>({-1}, {0, 47})),
                    "info:prob"_.Bind(Spec<float>({-1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 3})));
  }
};

using CliffWalkingEnvSpec = EnvSpec<CliffWalkingEnvFns>;

class CliffWalkingEnv : public Env<CliffWalkingEnvSpec> {
 protected:
  int x_, y_;
  bool is_slippery_;
  bool done_{true};

 public:
  CliffWalkingEnv(const Spec& spec, int env_id)
      : Env<CliffWalkingEnvSpec>(spec, env_id),
        is_slippery_(spec.config["is_slippery"_]) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    x_ = 3;
    y_ = 0;
    done_ = false;
    WriteState(0.0, 1.0f);
  }

  void Step(const Action& action) override {
    int act = SampleAction(action["action"_]);
    float reward = -1.0;
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
      reward = -100.0;
      x_ = 3;
      y_ = 0;
    }
    if (x_ == 3 && y_ == 11) {
      done_ = true;
    }
    WriteState(reward, is_slippery_ ? 1.0f / 3.0f : 1.0f);
  }

 private:
  int SampleAction(int intended_action) {
    if (!is_slippery_) {
      return intended_action;
    }
    static constexpr std::array<int, 3> kOffsets = {-1, 0, 1};
    std::uniform_int_distribution<int> dist(0, kOffsets.size() - 1);
    return (intended_action + kOffsets[dist(gen_)] + 4) % 4;
  }

  void WriteState(float reward, float prob) {
    auto state = Allocate();
    state["obs"_] = x_ * 12 + y_;
    state["reward"_] = reward;
    state["info:prob"_] = prob;
  }
};

using CliffWalkingEnvPool = AsyncEnvPool<CliffWalkingEnv>;

}  // namespace toy_text

#endif  // ENVPOOL_TOY_TEXT_CLIFFWALKING_H_
