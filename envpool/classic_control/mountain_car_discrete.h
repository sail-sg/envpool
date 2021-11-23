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
// https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py

#ifndef ENVPOOL_CLASSIC_CONTROL_MOUNTAIN_CAR_DISCRETE_H_
#define ENVPOOL_CLASSIC_CONTROL_MOUNTAIN_CAR_DISCRETE_H_

#include <cmath>
#include <random>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace classic_control {

class MountainCarDiscreteEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.bind(200),
                    "reward_threshold"_.bind(-110.0));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    // TODO(jiayi): specify range with [-1.2, -0.07] ~ [0.6, 0.07]
    return MakeDict("obs"_.bind(Spec<float>({2}, {-1.2, 0.6})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.bind(Spec<int>({-1}, {0, 2})));
  }
};

typedef class EnvSpec<MountainCarDiscreteEnvFns> MountainCarDiscreteEnvSpec;

class MountainCarDiscreteEnv : public Env<MountainCarDiscreteEnvSpec> {
 protected:
  const double kMinPos = -1.2;
  const double kMaxPos = 0.6;
  const double kMaxSpeed = 0.07;
  const double kForce = 0.001;
  const double kGoalPos = 0.5;
  const double kGoalVel = 0;
  const double kGravity = 0.0025;
  int max_episode_steps_, elapsed_step_;
  double pos_, vel_;
  std::uniform_real_distribution<> dist_;
  bool done_;

 public:
  MountainCarDiscreteEnv(const Spec& spec, int env_id)
      : Env<MountainCarDiscreteEnvSpec>(spec, env_id),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        elapsed_step_(max_episode_steps_ + 1),
        dist_(-0.6, -0.4),
        done_(true) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    pos_ = dist_(gen_);
    vel_ = 0.0f;
    done_ = false;
    elapsed_step_ = 0;
    State state = Allocate();
    WriteObs(state, 0.0f);
  }

  void Step(const Action& action) override {
    done_ = (++elapsed_step_ >= max_episode_steps_);
    double act = static_cast<int>(action["action"_]) - 1;
    vel_ += act * kForce - std::cos(3 * pos_) * kGravity;
    if (vel_ < -kMaxSpeed) {
      vel_ = -kMaxSpeed;
    } else if (vel_ > kMaxSpeed) {
      vel_ = kMaxSpeed;
    }
    pos_ += vel_;
    if (pos_ < kMinPos) {
      pos_ = kMinPos;
    } else if (pos_ > kMaxPos) {
      pos_ = kMaxPos;
    }
    if (pos_ == kMinPos && vel_ < 0) {
      vel_ = 0;
    }
    if (pos_ >= kGoalPos && vel_ >= kGoalVel) {
      done_ = true;
    }
    State state = Allocate();
    WriteObs(state, -1.0f);
  }

 private:
  void WriteObs(State& state, float reward) {  // NOLINT
    state["obs"_][0] = static_cast<float>(pos_);
    state["obs"_][1] = static_cast<float>(vel_);
    state["reward"_] = reward;
  }
};

typedef AsyncEnvPool<MountainCarDiscreteEnv> MountainCarDiscreteEnvPool;

}  // namespace classic_control

#endif  // ENVPOOL_CLASSIC_CONTROL_MOUNTAIN_CAR_DISCRETE_H_
