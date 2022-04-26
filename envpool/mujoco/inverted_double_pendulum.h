/*
 * Copyright 2022 Garena Online Private Limited
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

#ifndef ENVPOOL_MUJOCO_INVERTED_DOUBLE_PENDULUM_H_
#define ENVPOOL_MUJOCO_INVERTED_DOUBLE_PENDULUM_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/mujoco_env.h"

namespace mujoco {

class InvertedDoublePendulumEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "max_episode_steps"_.bind(1000), "reward_threshold"_.bind(9100.0),
        "frame_skip"_.bind(5), "post_constraint"_.bind(true),
        "healthy_reward"_.bind(10.0), "healthy_z_max"_.bind(1),

        "ob_min"_.bind(-10), "ob_max"_.bind(10), "dist_penalty_1"_.bind(0.01),
        "dist_penalty_2"_.bind(2), "vel_penalty_1"_.bind(0.001),
        "vel_penalty_2"_.bind(0.00003), "reset_noise_scale"_.bind(0.1));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict("obs"_.bind(Spec<mjtNum>({11}, {-inf, inf})),
                    // TODO(jiayi): remove these two lines for speed
                    "info:qpos0"_.bind(Spec<mjtNum>({2})),
                    "info:qvel0"_.bind(Spec<mjtNum>({2})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.bind(Spec<mjtNum>({-1, 1}, {-1.0f, 1.0f})));
  }
};

typedef class EnvSpec<InvertedDoublePendulumEnvFns>
    InvertedDoublePendulumEnvSpec;

class InvertedDoublePendulumEnv : public Env<InvertedDoublePendulumEnvSpec>,
                                  public MujocoEnv {
 protected:
  int max_episode_steps_, elapsed_step_;
  mjtNum healthy_reward_, healthy_z_max_, ob_min_, ob_max_, dist_penalty_1_,
      dist_penalty_2_, vel_penalty_1_, vel_penalty_2_;
  std::unique_ptr<mjtNum> qpos0_, qvel0_;  // for align check
  std::uniform_real_distribution<> dist_qpos_;
  std::normal_distribution<> dist_qvel_;
  bool done_;

 public:
  InvertedDoublePendulumEnv(const Spec& spec, int env_id)
      : Env<InvertedDoublePendulumEnvSpec>(spec, env_id),
        MujocoEnv(spec.config["base_path"_] +
                      "/mujoco/assets/inverted_double_pendulum.xml",
                  spec.config["frame_skip"_], spec.config["post_constraint"_]),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        elapsed_step_(max_episode_steps_ + 1),
        healthy_reward_(spec.config["healthy_reward"_]),
        healthy_z_max_(spec.config["healthy_z_max"_]),
        ob_min_(spec.config["ob_min"_]),
        ob_max_(spec.config["ob_max"_]),
        dist_penalty_1_(spec.config["dist_penalty_1"_]),
        dist_penalty_2_(spec.config["dist_penalty_2"_]),
        vel_penalty_1_(spec.config["vel_penalty_1"_]),
        vel_penalty_2_(spec.config["vel_penalty_2"_]),
        qpos0_(new mjtNum[model_->nq]),
        qvel0_(new mjtNum[model_->nv]),
        dist_qpos_(-spec.config["reset_noise_scale"_],
                   spec.config["reset_noise_scale"_]),
        done_(true) {}

  void MujocoResetModel() {
    for (int i = 0; i < model_->nq; ++i) {
      data_->qpos[i] = qpos0_.get()[i] = init_qpos_[i] + dist_qpos_(gen_);
    }
    for (int i = 0; i < model_->nv; ++i) {
      data_->qvel[i] = qvel0_.get()[i] = init_qvel_[i] + dist_qvel_(gen_);
    }
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    MujocoReset();
    WriteObs(0.0f);
  }

  void Step(const Action& action) override {
    // step
    MujocoStep(static_cast<mjtNum*>(action["action"_].data()));

    // dist_penalty
    mjtNum x = data_->site_xpos[0][0];
    mjtNum y = data_->site_xpos[0][2] - dist_penalty_2_;
    mjtNum dist_penalty = dist_penalty_1_ * x * x + y * y;
    // vel_penalty

    mjtNum m = data_->qvel[1];
    mjtNum n = data_->qvel[2];
    mjtNum vel_penalty = vel_penalty_1_ * m * m + vel_penalty_2_ * n * n;

    // alive_bonus
    // reward and done
    ++elapsed_step_;
    float reward = healthy_reward - dist_penalty - vel_penalty done_ =
                       !IsHealthy() || (elapsed_step_ >= max_episode_steps_);
    WriteObs(1.0f);
  }

 private:
  bool IsHealthy() {
    if (ddata_->site_xpos[0][2] > healthy_z_max_) {
      return false;
    }
    return true;
  }

  void WriteObs(float reward) {
    State state = Allocate();
    state["reward"_] = reward;
    // obs
    mjtNum* obs = static_cast<mjtNum*>(state["obs"_].data());
    *(obs++) = std::data_->qpos[0];
    *(obs++) = std::sin(data_->qpos[1]);
    *(obs++) = std::sin(data_->qpos[2]);
    *(obs++) = std::cos(data_->qpos[1]);
    *(obs++) = std::cos(data_->qpos[2]);
    for (int i = 0; i < model_->nv; ++i) {
      mjtNum x = data_->qvel[i];
      x = std::min(ob_max_, x);
      x = std::max(ob_min_, x);
      *(obs++) = x;
    }
    for (int i = 0; i < 3; ++i) {
      mjtNum x = data_->qfrc_constraint[i];
      x = std::min(ob_max_, x);
      x = std::max(ob_min_, x);
      *(obs++) = x;
    }
    // info
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
    state["info:qvel0"_].Assign(qvel0_.get(), model_->nv);
  }
};

typedef AsyncEnvPool<InvertedDoublePendulumEnv> InvertedDoublePendulumEnvPool;

}  // namespace mujoco

#endif  // ENVPOOL_MUJOCO_INVERTED_DOUBLE_PENDULUM_H_
