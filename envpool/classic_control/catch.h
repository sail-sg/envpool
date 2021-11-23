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

#ifndef ENVPOOL_CLASSIC_CONTROL_CATCH_H_
#define ENVPOOL_CLASSIC_CONTROL_CATCH_H_

#include <cmath>
#include <random>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace classic_control {

class CatchEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("height"_.bind(10), "width"_.bind(5));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.bind(
        Spec<float>({conf["height"_], conf["width"_]}, {0.0f, 1.0f})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.bind(Spec<int>({-1}, {0, 2})));
  }
};

typedef class EnvSpec<CatchEnvFns> CatchEnvSpec;

class CatchEnv : public Env<CatchEnvSpec> {
 protected:
  int x_, y_, height_, width_, paddle_;
  std::uniform_int_distribution<> dist_;
  bool done_;

 public:
  CatchEnv(const Spec& spec, int env_id)
      : Env<CatchEnvSpec>(spec, env_id),
        height_(spec.config["height"_]),
        width_(spec.config["width"_]),
        dist_(0, width_ - 1),
        done_(true) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    x_ = 0;
    y_ = dist_(gen_);
    paddle_ = width_ / 2;
    done_ = false;
    State state = Allocate();
    WriteObs(state, 0.0f);
  }

  void Step(const Action& action) override {
    int act = action["action"_];
    float reward = 0.0f;
    paddle_ += act - 1;
    if (paddle_ < 0) paddle_ = 0;
    if (paddle_ >= width_) paddle_ = width_ - 1;
    if (++x_ == height_ - 1) {
      done_ = true;
      reward = y_ == paddle_ ? 1.0f : -1.0f;
    }
    State state = Allocate();
    WriteObs(state, reward);
  }

 private:
  void WriteObs(State& state, float reward) {  // NOLINT
    state["obs"_](x_, y_) = 1.0f;
    state["obs"_](height_ - 1, paddle_) = 1.0f;
    state["reward"_] = reward;
  }
};

typedef AsyncEnvPool<CatchEnv> CatchEnvPool;

}  // namespace classic_control

#endif  // ENVPOOL_CLASSIC_CONTROL_CATCH_H_
