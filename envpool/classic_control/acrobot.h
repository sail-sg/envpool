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
// https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py

#ifndef ENVPOOL_CLASSIC_CONTROL_ACROBOT_H_
#define ENVPOOL_CLASSIC_CONTROL_ACROBOT_H_

#include <cmath>
#include <random>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace classic_control {

class AcrobotEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.Bind(500),
                    "reward_threshold"_.Bind(-100.0));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.Bind(Spec<float>(
                        {6}, {{-1.0, -1.0, -1.0, -1.0, -4 * M_PI, -9 * M_PI},
                              {1.0, 1.0, 1.0, 1.0, 4 * M_PI, 9 * M_PI}})),
                    "info:state"_.Bind(Spec<float>({2})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 2})));
  }
};

using AcrobotEnvSpec = EnvSpec<AcrobotEnvFns>;

class AcrobotEnv : public Env<AcrobotEnvSpec> {
  struct V5 {
    double s0_{0}, s1_{0}, s2_{0}, s3_{0}, s4_{0};
    V5() = default;
    V5(double s0, double s1, double s2, double s3, double s4)
        : s0_(s0), s1_(s1), s2_(s2), s3_(s3), s4_(s4) {}
    V5 operator+(const V5& v) const {
      return V5(s0_ + v.s0_, s1_ + v.s1_, s2_ + v.s2_, s3_ + v.s3_,
                s4_ + v.s4_);
    }
    V5 operator*(double v) const {
      return V5(s0_ * v, s1_ * v, s2_ * v, s3_ * v, s4_ * v);
    }
  };

 protected:
  const double kPi = std::acos(-1);
  const double kG = 9.8;
  const double kDt = 0.2;
  const double kL = 1.0;
  const double kM = 1.0;
  const double kLC = 0.5;
  const double kI = 1.0;
  const double kMaxVel1 = 4 * kPi;
  const double kMaxVel2 = 9 * kPi;
  const double kInitRange = 0.1;

  int max_episode_steps_, elapsed_step_;
  V5 s_;
  std::uniform_real_distribution<> dist_;
  bool done_;

 public:
  AcrobotEnv(const Spec& spec, int env_id)
      : Env<AcrobotEnvSpec>(spec, env_id),
        max_episode_steps_(spec.config_["max_episode_steps"_]),
        elapsed_step_(max_episode_steps_ + 1),
        dist_(-kInitRange, kInitRange),
        done_(true) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    s_.s0_ = dist_(gen_);
    s_.s1_ = dist_(gen_);
    s_.s2_ = dist_(gen_);
    s_.s3_ = dist_(gen_);
    s_.s4_ = 0;
    done_ = false;
    elapsed_step_ = 0;
    WriteState(0.0);
  }

  void Step(const Action& action) override {
    done_ = (++elapsed_step_ >= max_episode_steps_);
    int act = action["action"_];
    float reward = -1.0;

    s_.s4_ = act - 1;
    s_ = Rk4(s_);
    while (s_.s0_ < -kPi) {
      s_.s0_ += kPi * 2;
    }
    while (s_.s1_ < -kPi) {
      s_.s1_ += kPi * 2;
    }
    while (s_.s0_ >= kPi) {
      s_.s0_ -= kPi * 2;
    }
    while (s_.s1_ >= kPi) {
      s_.s1_ -= kPi * 2;
    }
    if (s_.s2_ < -kMaxVel1) {
      s_.s2_ = -kMaxVel1;
    }
    if (s_.s3_ < -kMaxVel2) {
      s_.s3_ = -kMaxVel2;
    }
    if (s_.s2_ > kMaxVel1) {
      s_.s2_ = kMaxVel1;
    }
    if (s_.s3_ > kMaxVel2) {
      s_.s3_ = kMaxVel2;
    }
    if (-std::cos(s_.s0_) - std::cos(s_.s0_ + s_.s1_) > 1) {
      done_ = true;
      reward = 0.0;
    }

    WriteState(reward);
  }

 private:
  V5 Rk4(V5 y0) {
    V5 k1 = Derivs(y0, 0);
    V5 k2 = Derivs(y0 + k1 * (kDt / 2), kDt / 2);
    V5 k3 = Derivs(y0 + k2 * (kDt / 2), kDt / 2);
    V5 k4 = Derivs(y0 + k3 * kDt, kDt);
    return y0 + (k1 + k2 * 2 + k3 * 2 + k4) * (kDt / 6.0);
  }

  [[nodiscard]] V5 Derivs(V5 s, double t) const {
    double theta1 = s.s0_;
    double theta2 = s.s1_;
    double dtheta1 = s.s2_;
    double dtheta2 = s.s3_;
    double a = s.s4_;
    double d1 = kM * kLC * kLC +
                kM * (kL * kL + kLC * kLC + 2 * kL * kLC * std::cos(theta2)) +
                kI * 2;
    double d2 = kM * (kLC * kLC + kL * kLC * std::cos(theta2)) + kI;
    double phi2 = kM * kLC * kG * std::cos(theta1 + theta2 - kPi / 2);
    double phi1 =
        -(dtheta2 + 2 * dtheta1) * kM * kL * kLC * dtheta2 * std::sin(theta2) +
        kM * (kLC + kL) * kG * std::cos(theta1 - kPi / 2) + phi2;
    double ddtheta2 =
        (a + d2 / d1 * phi1 -
         kM * kL * kLC * dtheta1 * dtheta1 * std::sin(theta2) - phi2) /
        (kM * kLC * kLC + kI - d2 * d2 / d1);
    double ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;
    return V5(dtheta1, dtheta2, ddtheta1, ddtheta2, 0);
  }

  void WriteState(float reward) {
    State state = Allocate();
    state["obs"_][0] = static_cast<float>(std::cos(s_.s0_));
    state["obs"_][1] = static_cast<float>(std::sin(s_.s0_));
    state["obs"_][2] = static_cast<float>(std::cos(s_.s1_));
    state["obs"_][3] = static_cast<float>(std::sin(s_.s1_));
    state["obs"_][4] = static_cast<float>(s_.s2_);
    state["obs"_][5] = static_cast<float>(s_.s3_);
    state["info:state"_][0] = static_cast<float>(s_.s0_);
    state["info:state"_][1] = static_cast<float>(s_.s1_);
    state["reward"_] = reward;
  }
};

using AcrobotEnvPool = AsyncEnvPool<AcrobotEnv>;

}  // namespace classic_control

#endif  // ENVPOOL_CLASSIC_CONTROL_ACROBOT_H_
