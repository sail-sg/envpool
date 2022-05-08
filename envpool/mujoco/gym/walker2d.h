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

#ifndef ENVPOOL_MUJOCO_GYM_WALKER2D_H_
#define ENVPOOL_MUJOCO_GYM_WALKER2D_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/gym/mujoco_env.h"

namespace mujoco {

class Walker2dEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "max_episode_steps"_.Bind(1000), "frame_skip"_.Bind(4),
        "post_constraint"_.Bind(true), "ctrl_cost_weight"_.Bind(0.001),
        "terminate_when_unhealthy"_.Bind(true),
        "exclude_current_positions_from_observation"_.Bind(true),
        "forward_reward_weight"_.Bind(1.0), "healthy_reward"_.Bind(1.0),
        "healthy_z_min"_.Bind(0.8), "healthy_z_max"_.Bind(2.0),
        "healthy_angle_min"_.Bind(-1.0), "healthy_angle_max"_.Bind(1.0),
        "velocity_min"_.Bind(-10.0), "velocity_max"_.Bind(10.0),
        "reset_noise_scale"_.Bind(0.005));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    bool no_pos = conf["exclude_current_positions_from_observation"_];
    return MakeDict("obs"_.Bind(Spec<mjtNum>({no_pos ? 17 : 18}, {-inf, inf})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.Bind(Spec<mjtNum>({9})),
                    "info:qvel0"_.Bind(Spec<mjtNum>({9})),
#endif
                    "info:x_position"_.Bind(Spec<mjtNum>({-1})),
                    "info:x_velocity"_.Bind(Spec<mjtNum>({-1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 6}, {-1.0, 1.0})));
  }
};

using Walker2dEnvSpec = EnvSpec<Walker2dEnvFns>;

class Walker2dEnv : public Env<Walker2dEnvSpec>, public MujocoEnv {
 protected:
  bool terminate_when_unhealthy_, no_pos_;
  mjtNum ctrl_cost_weight_, forward_reward_weight_;
  mjtNum healthy_reward_, healthy_z_min_, healthy_z_max_;
  mjtNum healthy_angle_min_, healthy_angle_max_;
  mjtNum velocity_min_, velocity_max_;
  std::uniform_real_distribution<> dist_;

 public:
  Walker2dEnv(const Spec& spec, int env_id)
      : Env<Walker2dEnvSpec>(spec, env_id),
        MujocoEnv(spec.config["base_path"_] + "/mujoco/assets_gym/walker2d.xml",
                  spec.config["frame_skip"_], spec.config["post_constraint"_],
                  spec.config["max_episode_steps"_]),
        terminate_when_unhealthy_(spec.config["terminate_when_unhealthy"_]),
        no_pos_(spec.config["exclude_current_positions_from_observation"_]),
        ctrl_cost_weight_(spec.config["ctrl_cost_weight"_]),
        forward_reward_weight_(spec.config["forward_reward_weight"_]),
        healthy_reward_(spec.config["healthy_reward"_]),
        healthy_z_min_(spec.config["healthy_z_min"_]),
        healthy_z_max_(spec.config["healthy_z_max"_]),
        healthy_angle_min_(spec.config["healthy_angle_min"_]),
        healthy_angle_max_(spec.config["healthy_angle_max"_]),
        velocity_min_(spec.config["velocity_min"_]),
        velocity_max_(spec.config["velocity_max"_]),
        dist_(-spec.config["reset_noise_scale"_],
              spec.config["reset_noise_scale"_]) {}

  void MujocoResetModel() override {
    for (int i = 0; i < model_->nq; ++i) {
      data_->qpos[i] = qpos0_[i] = init_qpos_[i] + dist_(gen_);
    }
    for (int i = 0; i < model_->nv; ++i) {
      data_->qvel[i] = qvel0_[i] = init_qvel_[i] + dist_(gen_);
    }
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    MujocoReset();
    WriteState(0.0, 0.0, 0.0);
  }

  void Step(const Action& action) override {
    // step
    mjtNum* act = static_cast<mjtNum*>(action["action"_].Data());
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
    mjtNum healthy_reward =
        terminate_when_unhealthy_ || IsHealthy() ? healthy_reward_ : 0.0;
    auto reward = static_cast<float>(xv * forward_reward_weight_ +
                                     healthy_reward - ctrl_cost);
    ++elapsed_step_;
    done_ = (terminate_when_unhealthy_ ? !IsHealthy() : false) ||
            (elapsed_step_ >= max_episode_steps_);
    WriteState(reward, xv, x_after);
  }

 private:
  bool IsHealthy() {
    if (data_->qpos[1] < healthy_z_min_ || data_->qpos[1] > healthy_z_max_) {
      return false;
    }
    if (data_->qpos[2] < healthy_angle_min_ ||
        data_->qpos[2] > healthy_angle_max_) {
      return false;
    }
    return true;
  }

  void WriteState(float reward, mjtNum xv, mjtNum x_after) {
    State state = Allocate();
    state["reward"_] = reward;
    // obs
    mjtNum* obs = static_cast<mjtNum*>(state["obs"_].Data());
    for (int i = no_pos_ ? 1 : 0; i < model_->nq; ++i) {
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
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_, model_->nq);
    state["info:qvel0"_].Assign(qvel0_, model_->nv);
#endif
  }
};

using Walker2dEnvPool = AsyncEnvPool<Walker2dEnv>;

}  // namespace mujoco

#endif  // ENVPOOL_MUJOCO_GYM_WALKER2D_H_
