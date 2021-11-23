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
// https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py

#ifndef ENVPOOL_CLASSIC_CONTROL_MOUNTAIN_CAR_H_
#define ENVPOOL_CLASSIC_CONTROL_MOUNTAIN_CAR_H_

#include <cmath>
#include <random>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace classic_control {

class MountainCarEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.bind(200),
                    "reward_threshold"_.bind(-110.0),
                    "is_continuous"_.bind(false));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    // TODO(jiayi): specify range with [-1.2, -0.07] ~ [0.6, 0.07]
    return MakeDict("obs"_.bind(Spec<float>({2}, {-1.2, 0.6})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    if (conf["is_continuous"_]) {
      return MakeDict("action"_.bind(Spec<float>({-1}, {-1.0f, 1.0f})));
    } else {
      return MakeDict("action"_.bind(Spec<int>({-1}, {0, 2})));
    }
  }
};

typedef class EnvSpec<MountainCarEnvFns> MountainCarEnvSpec;

class MountainCarEnv : public Env<MountainCarEnvSpec> {
 protected:
  const double kMinPos = -1.2;
  const double kMaxPos = 0.6;
  const double kMaxSpeed = 0.07;
  const double kPower = 0.0015;
  const double kGoalVel = 0;
  const double kForce = 0.001;
  const double kGravity = 0.0025;
  bool is_continuous_, done_;
  int max_episode_steps_, elapsed_step_;
  double pos_, vel_, goal_pos_;
  std::uniform_real_distribution<> dist_;

 public:
  MountainCarEnv(const Spec& spec, int env_id)
      : Env<MountainCarEnvSpec>(spec, env_id),
        is_continuous_(spec.config["is_continuous"_]),
        done_(true),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        dist_(-0.6, -0.4),
        goal_pos_(is_continuous_ ? 0.45 : 0.5) {}

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
    double act;
    if (is_continuous_) {
      act = static_cast<float>(action["action"_]);
    } else {
      act = static_cast<int>(action["action"_]) - 1;
    }
    float reward;
    if (is_continuous_) {
      reward = -0.1 * act * act;
      if (act < -1) {
        act = -1;
      } else if (act > 1) {
        act = 1;
      }
      vel_ += act * kPower - std::cos(3 * pos_) * kGravity;
    } else {
      reward = -1;
      vel_ += act * kForce - std::cos(3 * pos_) * kGravity;
    }
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
    if (pos_ >= goal_pos_ && vel_ >= kGoalVel) {
      done_ = true;
      if (is_continuous_) {
        reward += 100;
      }
    }
    State state = Allocate();
    WriteObs(state, reward);
  }

 private:
  void WriteObs(State& state, float reward) {  // NOLINT
    state["obs"_][0] = static_cast<float>(pos_);
    state["obs"_][1] = static_cast<float>(vel_);
    state["reward"_] = reward;
  }
};

typedef AsyncEnvPool<MountainCarEnv> MountainCarEnvPool;

}  // namespace classic_control

#endif  // ENVPOOL_CLASSIC_CONTROL_MOUNTAIN_CAR_H_
