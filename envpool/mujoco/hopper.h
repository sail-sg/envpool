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

#ifndef ENVPOOL_MUJOCO_HOPPER_H_
#define ENVPOOL_MUJOCO_HOPPER_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/mujoco_env.h"

namespace mujoco {

class HopperEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "max_episode_steps"_.bind(1000), "reward_threshold"_.bind(6000.0),
        "frame_skip"_.bind(4), "post_constraint"_.bind(true),
        "ctrl_cost_weight"_.bind(1e-3), "healthy_reward"_.bind(1.0),
        "velocity_min"_.bind(-10.0), "velocity_max"_.bind(10.0),
        "healthy_state_min"_.bind(-100.0), "healthy_state_max"_.bind(100.0),
        "healthy_angle_min"_.bind(-0.2), "healthy_angle_max"_.bind(0.2),
        "healthy_z_min"_.bind(0.7), "reset_noise_scale"_.bind(5e-3));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict("obs"_.bind(Spec<mjtNum>({11}, {-inf, inf})),
                    "info:x_position"_.bind(Spec<mjtNum>({-1})),
                    "info:x_velocity"_.bind(Spec<mjtNum>({-1})),
                    // TODO(jiayi): remove these two lines for speed
                    "info:qpos0"_.bind(Spec<mjtNum>({6})),
                    "info:qvel0"_.bind(Spec<mjtNum>({6})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.bind(Spec<mjtNum>({-1, 3}, {-1.0f, 1.0f})));
  }
};

typedef class EnvSpec<HopperEnvFns> HopperEnvSpec;

class HopperEnv : public Env<HopperEnvSpec>, public MujocoEnv {
 protected:
  int max_episode_steps_, elapsed_step_;
  mjtNum ctrl_cost_weight_, healthy_reward_, healthy_z_min_;
  mjtNum velocity_min_, velocity_max_;
  mjtNum healthy_state_min_, healthy_state_max_;
  mjtNum healthy_angle_min_, healthy_angle_max_;
  std::unique_ptr<mjtNum> qpos0_, qvel0_;  // for align check
  std::uniform_real_distribution<> dist_;
  bool done_;

 public:
  HopperEnv(const Spec& spec, int env_id)
      : Env<HopperEnvSpec>(spec, env_id),
        MujocoEnv(spec.config["base_path"_] + "/mujoco/assets/hopper.xml",
                  spec.config["frame_skip"_], spec.config["post_constraint"_]),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        elapsed_step_(max_episode_steps_ + 1),
        ctrl_cost_weight_(spec.config["ctrl_cost_weight"_]),
        healthy_reward_(spec.config["healthy_reward"_]),
        healthy_z_min_(spec.config["healthy_z_min"_]),
        velocity_min_(spec.config["velocity_min"_]),
        velocity_max_(spec.config["velocity_max"_]),
        healthy_state_min_(spec.config["healthy_state_min"_]),
        healthy_state_max_(spec.config["healthy_state_max"_]),
        healthy_angle_min_(spec.config["healthy_angle_min"_]),
        healthy_angle_max_(spec.config["healthy_angle_max"_]),
        qpos0_(new mjtNum[model_->nq]),
        qvel0_(new mjtNum[model_->nv]),
        dist_(-spec.config["reset_noise_scale"_],
              spec.config["reset_noise_scale"_]),
        done_(true) {}

  void MujocoResetModel() {
    for (int i = 0; i < model_->nq; ++i) {
      data_->qpos[i] = qpos0_.get()[i] = init_qpos_[i] + dist_(gen_);
    }
    for (int i = 0; i < model_->nv; ++i) {
      data_->qvel[i] = qvel0_.get()[i] = init_qvel_[i] + dist_(gen_);
    }
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    MujocoReset();
    WriteObs(0.0f, 0, 0);
  }

  void Step(const Action& action) override {
    // step
    mjtNum* act = static_cast<mjtNum*>(action["action"_].data());
    mjtNum x_before = data_->qpos[0];
    MujocoStep(act);
    mjtNum x_after = data_->qpos[0];

    // ctrl_cost
    mjtNum ctrl_cost = 0.0;
    for (int i = 0; i < model_->nu; ++i) {
      ctrl_cost += ctrl_cost_weight_ * act[i] * act[i];
    }
    // xv
    mjtNum dt = frame_skip_ * model_->opt.timestep;
    mjtNum xv = (x_after - x_before) / dt;
    // reward and done
    float reward = xv + healthy_reward_ - ctrl_cost;
    ++elapsed_step_;
    done_ = !IsHealthy() || (elapsed_step_ >= max_episode_steps_);
    WriteObs(reward, xv, x_after);
  }

 private:
  bool IsHealthy() {
    mjtNum z = data_->qpos[1], angle = data_->qpos[2];
    if (angle <= healthy_angle_min_ || angle >= healthy_angle_max_ ||
        z <= healthy_z_min_) {
      return false;
    }
    for (int i = 2; i < model_->nq; ++i) {
      if (data_->qpos[i] <= healthy_state_min_ ||
          data_->qpos[i] >= healthy_state_max_) {
        return false;
      }
    }
    for (int i = 0; i < model_->nv; ++i) {
      if (data_->qvel[i] <= healthy_state_min_ ||
          data_->qvel[i] >= healthy_state_max_) {
        return false;
      }
    }
    return true;
  }

  void WriteObs(float reward, mjtNum xv, mjtNum x_after) {  // NOLINT
    State state = Allocate();
    state["reward"_] = reward;
    // obs
    mjtNum* obs = static_cast<mjtNum*>(state["obs"_].data());
    for (int i = 1; i < model_->nq; ++i) {
      *(obs++) = data_->qpos[i];
    }
    for (int i = 0; i < model_->nv; ++i) {
      mjtNum x = data_->qvel[i];
      x = std::min(velocity_max_, x);
      x = std::max(velocity_min_, x);
      *(obs++) = x;
    }
    // info
    state["info:x_position"_] = x_after;
    state["info:x_velocity"_] = xv;
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
    state["info:qvel0"_].Assign(qvel0_.get(), model_->nv);
  }
};

typedef AsyncEnvPool<HopperEnv> HopperEnvPool;

}  // namespace mujoco

#endif  // ENVPOOL_MUJOCO_HOPPER_H_
