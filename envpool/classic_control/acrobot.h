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
    return MakeDict("max_episode_steps"_.bind(500),
                    "reward_threshold"_.bind(-100.0));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    // TODO(jiayi): specify range with [1, 1, 1, 1, 4 * np.pi, 9 * np.pi]
    return MakeDict("obs"_.bind(Spec<float>({6})),
                    "info:state"_.bind(Spec<float>({2})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.bind(Spec<int>({-1}, {0, 2})));
  }
};

typedef class EnvSpec<AcrobotEnvFns> AcrobotEnvSpec;

class AcrobotEnv : public Env<AcrobotEnvSpec> {
  struct V5 {
    double s0, s1, s2, s3, s4;
    V5() : s0(0), s1(0), s2(0), s3(0), s4(0) {}
    V5(double s0, double s1, double s2, double s3, double s4)
        : s0(s0), s1(s1), s2(s2), s3(s3), s4(s4) {}
    V5 operator+(const V5& v) const {
      return V5(s0 + v.s0, s1 + v.s1, s2 + v.s2, s3 + v.s3, s4 + v.s4);
    }
    V5 operator*(double v) {
      return V5(s0 * v, s1 * v, s2 * v, s3 * v, s4 * v);
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
        max_episode_steps_(spec.config["max_episode_steps"_]),
        elapsed_step_(max_episode_steps_ + 1),
        dist_(-kInitRange, kInitRange),
        done_(true) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    s_.s0 = dist_(gen_);
    s_.s1 = dist_(gen_);
    s_.s2 = dist_(gen_);
    s_.s3 = dist_(gen_);
    s_.s4 = 0;
    done_ = false;
    elapsed_step_ = 0;
    State state = Allocate();
    WriteObs(state, 0.0f);
  }

  void Step(const Action& action) override {
    done_ = (++elapsed_step_ >= max_episode_steps_);
    int act = action["action"_];
    double reward = -1;

    s_.s4 = act - 1;
    s_ = rk4(s_);
    while (s_.s0 < -kPi) s_.s0 += kPi * 2;
    while (s_.s1 < -kPi) s_.s1 += kPi * 2;
    while (s_.s0 >= kPi) s_.s0 -= kPi * 2;
    while (s_.s1 >= kPi) s_.s1 -= kPi * 2;
    if (s_.s2 < -kMaxVel1) s_.s2 = -kMaxVel1;
    if (s_.s3 < -kMaxVel2) s_.s3 = -kMaxVel2;
    if (s_.s2 > kMaxVel1) s_.s2 = kMaxVel1;
    if (s_.s3 > kMaxVel2) s_.s3 = kMaxVel2;
    if (-std::cos(s_.s0) - std::cos(s_.s0 + s_.s1) > 1) {
      done_ = true;
      reward = 0;
    }

    State state = Allocate();
    WriteObs(state, reward);
  }

 private:
  V5 rk4(V5 y0) {
    V5 k1 = derivs(y0, 0);
    V5 k2 = derivs(y0 + k1 * (kDt / 2), kDt / 2);
    V5 k3 = derivs(y0 + k2 * (kDt / 2), kDt / 2);
    V5 k4 = derivs(y0 + k3 * kDt, kDt);
    return y0 + (k1 + k2 * 2 + k3 * 2 + k4) * (kDt / 6.0);
  }

  V5 derivs(V5 s, double t) {
    double theta1 = s.s0;
    double theta2 = s.s1;
    double dtheta1 = s.s2;
    double dtheta2 = s.s3;
    double a = s.s4;
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

  void WriteObs(State& state, float reward) {  // NOLINT
    state["obs"_][0] = static_cast<float>(std::cos(s_.s0));
    state["obs"_][1] = static_cast<float>(std::sin(s_.s0));
    state["obs"_][2] = static_cast<float>(std::cos(s_.s1));
    state["obs"_][3] = static_cast<float>(std::sin(s_.s1));
    state["obs"_][4] = static_cast<float>(s_.s2);
    state["obs"_][5] = static_cast<float>(s_.s3);
    state["info:state"_][0] = static_cast<float>(s_.s0);
    state["info:state"_][1] = static_cast<float>(s_.s1);
    state["reward"_] = reward;
  }
};

typedef AsyncEnvPool<AcrobotEnv> AcrobotEnvPool;

}  // namespace classic_control

#endif  // ENVPOOL_CLASSIC_CONTROL_ACROBOT_H_
