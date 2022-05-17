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
// https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

#ifndef ENVPOOL_CLASSIC_CONTROL_PENDULUM_H_
#define ENVPOOL_CLASSIC_CONTROL_PENDULUM_H_

#include <cmath>
#include <random>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace classic_control {

class PendulumEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.Bind(200), "version"_.Bind(0));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict(
        "obs"_.Bind(Spec<float>({3}, {{-1.0, -1.0, -8.0}, {1.0, 1.0, 8.0}})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<float>({-1, 1}, {-2.0, 2.0})));
  }
};

using PendulumEnvSpec = EnvSpec<PendulumEnvFns>;

class PendulumEnv : public Env<PendulumEnvSpec> {
 protected:
  const double kPi = std::acos(-1);
  const double kMaxSpeed = 8;
  const double kMaxTorque = 2;
  const double kDt = 0.05;
  const double kGravity = 10;

  int max_episode_steps_, elapsed_step_;
  int version_;
  double theta_, theta_dot_;
  std::uniform_real_distribution<> dist_, dist_dot_;
  bool done_;

 public:
  PendulumEnv(const Spec& spec, int env_id)
      : Env<PendulumEnvSpec>(spec, env_id),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        version_(spec.config["version"_]),
        elapsed_step_(max_episode_steps_ + 1),
        dist_(-kPi, kPi),
        dist_dot_(-1, 1),
        done_(true) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    theta_ = dist_(gen_);
    theta_dot_ = dist_dot_(gen_);
    done_ = false;
    elapsed_step_ = 0;
    WriteState(0.0);
  }

  void Step(const Action& action) override {
    done_ = (++elapsed_step_ >= max_episode_steps_);
    float act = action["action"_];
    double u = act < -kMaxTorque  ? -kMaxTorque
               : act > kMaxTorque ? kMaxTorque
                                  : act;
    double cost =
        theta_ * theta_ + 0.1 * theta_dot_ * theta_dot_ + 0.001 * u * u;
    double new_theta_dot =
        theta_dot_ + 3 * (kGravity / 2 * std::sin(theta_) + u) * kDt;
    if (version_ == 0) {
      theta_ += new_theta_dot * kDt;
    }
    theta_dot_ = new_theta_dot < -kMaxSpeed  ? -kMaxSpeed
                 : new_theta_dot > kMaxSpeed ? kMaxSpeed
                                             : new_theta_dot;
    if (version_ == 1) {
      theta_ += new_theta_dot * kDt;
    }
    while (theta_ < -kPi) {
      theta_ += kPi * 2;
    }
    while (theta_ >= kPi) {
      theta_ -= kPi * 2;
    }
    WriteState(static_cast<float>(-cost));
  }

 private:
  void WriteState(float reward) {
    State state = Allocate();
    state["obs"_][0] = static_cast<float>(std::cos(theta_));
    state["obs"_][1] = static_cast<float>(std::sin(theta_));
    state["obs"_][2] = static_cast<float>(theta_dot_);
    state["reward"_] = reward;
  }
};

using PendulumEnvPool = AsyncEnvPool<PendulumEnv>;

}  // namespace classic_control

#endif  // ENVPOOL_CLASSIC_CONTROL_PENDULUM_H_
