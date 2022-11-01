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
// https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py

#ifndef ENVPOOL_CLASSIC_CONTROL_MOUNTAIN_CAR_CONTINUOUS_H_
#define ENVPOOL_CLASSIC_CONTROL_MOUNTAIN_CAR_CONTINUOUS_H_

#include <cmath>
#include <random>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace classic_control {

class MountainCarContinuousEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("reward_threshold"_.Bind(90.0));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict(
        "obs"_.Bind(Spec<float>({2}, {{-1.2, -0.07}, {0.6, 0.07}})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<float>({-1, 1}, {-1.0, 1.0})));
  }
};

using MountainCarContinuousEnvSpec = EnvSpec<MountainCarContinuousEnvFns>;

class MountainCarContinuousEnv : public Env<MountainCarContinuousEnvSpec> {
 protected:
  const double kMinPos = -1.2;
  const double kMaxPos = 0.6;
  const double kMaxSpeed = 0.07;
  const double kPower = 0.0015;
  const double kGoalPos = 0.45;
  const double kGoalVel = 0;
  const double kGravity = 0.0025;
  int max_episode_steps_, elapsed_step_;
  double pos_, vel_;
  std::uniform_real_distribution<> dist_;
  bool done_;

 public:
  MountainCarContinuousEnv(const Spec& spec, int env_id)
      : Env<MountainCarContinuousEnvSpec>(spec, env_id),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        elapsed_step_(max_episode_steps_ + 1),
        dist_(-0.6, -0.4),
        done_(true) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    pos_ = dist_(gen_);
    vel_ = 0.0;
    done_ = false;
    elapsed_step_ = 0;
    WriteState(0.0);
  }

  void Step(const Action& action) override {
    done_ = (++elapsed_step_ >= max_episode_steps_);
    double act = static_cast<float>(action["action"_]);
    double reward = -0.1 * act * act;
    if (act < -1) {
      act = -1;
    } else if (act > 1) {
      act = 1;
    }
    vel_ += act * kPower - std::cos(3 * pos_) * kGravity;
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
      reward += 100;
    }
    WriteState(static_cast<float>(reward));
  }

 private:
  void WriteState(float reward) {
    State state = Allocate();
    state["obs"_][0] = static_cast<float>(pos_);
    state["obs"_][1] = static_cast<float>(vel_);
    state["reward"_] = reward;
  }
};

using MountainCarContinuousEnvPool = AsyncEnvPool<MountainCarContinuousEnv>;

}  // namespace classic_control

#endif  // ENVPOOL_CLASSIC_CONTROL_MOUNTAIN_CAR_CONTINUOUS_H_
