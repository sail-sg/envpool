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
// https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

#ifndef ENVPOOL_CLASSIC_CONTROL_CARTPOLE_H_
#define ENVPOOL_CLASSIC_CONTROL_CARTPOLE_H_

#include <cmath>
#include <limits>
#include <random>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace classic_control {

class CartPoleEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("reward_threshold"_.Bind(195.0));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    float fmax = std::numeric_limits<float>::max();
    return MakeDict(
        "obs"_.Bind(Spec<float>({4}, {{-4.8, -fmax, -M_PI / 7.5, -fmax},
                                      {4.8, fmax, M_PI / 7.5, fmax}})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 1})));
  }
};

using CartPoleEnvSpec = EnvSpec<CartPoleEnvFns>;

class CartPoleEnv : public Env<CartPoleEnvSpec> {
 protected:
  const double kGravity = 9.8;
  const double kMassCart = 1.0;
  const double kMassPole = 0.1;
  const double kMassTotal = kMassCart + kMassPole;
  const double kLength = 0.5;
  const double kMassPoleLength = kMassPole * kLength;
  const double kForceMag = 10.0;
  const double kTau = 0.02;
  const double kThetaThresholdRadians = 12 * 2 * M_PI / 360;
  const double kXThreshold = 2.4;
  const double kInitRange = 0.05;
  int max_episode_steps_, elapsed_step_;
  double x_, x_dot_, theta_, theta_dot_;
  std::uniform_real_distribution<> dist_;
  bool done_;

 public:
  CartPoleEnv(const Spec& spec, int env_id)
      : Env<CartPoleEnvSpec>(spec, env_id),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        elapsed_step_(max_episode_steps_ + 1),
        dist_(-kInitRange, kInitRange),
        done_(true) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    x_ = dist_(gen_);
    x_dot_ = dist_(gen_);
    theta_ = dist_(gen_);
    theta_dot_ = dist_(gen_);
    done_ = false;
    elapsed_step_ = 0;
    WriteState(0.0);
  }

  void Step(const Action& action) override {
    done_ = (++elapsed_step_ >= max_episode_steps_);
    int act = action["action"_];
    double force = act == 1 ? kForceMag : -kForceMag;
    double costheta = std::cos(theta_);
    double sintheta = std::sin(theta_);
    double temp =
        (force + kMassPoleLength * theta_dot_ * theta_dot_ * sintheta) /
        kMassTotal;
    double theta_acc =
        (kGravity * sintheta - costheta * temp) /
        (kLength * (4.0 / 3.0 - kMassPole * costheta * costheta / kMassTotal));
    double x_acc = temp - kMassPoleLength * theta_acc * costheta / kMassTotal;

    x_ += kTau * x_dot_;
    x_dot_ += kTau * x_acc;
    theta_ += kTau * theta_dot_;
    theta_dot_ += kTau * theta_acc;
    if (x_ < -kXThreshold || x_ > kXThreshold ||
        theta_ < -kThetaThresholdRadians || theta_ > kThetaThresholdRadians) {
      done_ = true;
    }
    WriteState(1.0);
  }

 private:
  void WriteState(float reward) {
    State state = Allocate();
    state["obs"_][0] = static_cast<float>(x_);
    state["obs"_][1] = static_cast<float>(x_dot_);
    state["obs"_][2] = static_cast<float>(theta_);
    state["obs"_][3] = static_cast<float>(theta_dot_);
    state["reward"_] = reward;
  }
};

using CartPoleEnvPool = AsyncEnvPool<CartPoleEnv>;

}  // namespace classic_control

#endif  // ENVPOOL_CLASSIC_CONTROL_CARTPOLE_H_
