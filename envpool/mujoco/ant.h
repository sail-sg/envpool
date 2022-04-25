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

#ifndef ENVPOOL_MUJOCO_ANT_H_
#define ENVPOOL_MUJOCO_ANT_H_

#include <algorithm>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/mujoco_env.h"

namespace mujoco {

class AntEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "max_episode_steps"_.bind(1000), "reward_threshold"_.bind(6000.0),
        "ctrl_cost_weight"_.bind(0.5), "contact_cost_weight"_.bind(5e-4),
        "healthy_reward"_.bind(1.0), "healthy_z_min"_.bind(0.2),
        "healthy_z_max"_.bind(1.0), "contact_force_min"_.bind(-1.0),
        "contact_force_max"_.bind(1.0), "reset_noise_scale"_.bind(0.1),
        "frame_skip"_.bind(5));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.bind(Spec<mjtNum>({111})),
                    "info:reward_forward"_.bind(Spec<mjtNum>({-1})),
                    "info:reward_ctrl"_.bind(Spec<mjtNum>({-1})),
                    "info:reward_contact"_.bind(Spec<mjtNum>({-1})),
                    "info:reward_survive"_.bind(Spec<mjtNum>({-1})),
                    "info:x_position"_.bind(Spec<mjtNum>({-1})),
                    "info:y_position"_.bind(Spec<mjtNum>({-1})),
                    "info:distance_from_origin"_.bind(Spec<mjtNum>({-1})),
                    "info:x_velocity"_.bind(Spec<mjtNum>({-1})),
                    "info:y_velocity"_.bind(Spec<mjtNum>({-1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.bind(Spec<mjtNum>({-1, 8}, {-1.0f, 1.0f})));
  }
};

typedef class EnvSpec<AntEnvFns> AntEnvSpec;

class AntEnv : public Env<AntEnvSpec>, public MujocoEnv {
 protected:
  int max_episode_steps_, elapsed_step_;
  mjtNum ctrl_cost_weight_, contact_cost_weight_, healthy_reward_;
  mjtNum healthy_z_min_, healthy_z_max_;
  mjtNum contact_force_min_, contact_force_max_;
  std::uniform_real_distribution<> dist_qpos_;
  std::normal_distribution<> dist_qvel_;
  bool done_;

 public:
  AntEnv(const Spec& spec, int env_id)
      : Env<AntEnvSpec>(spec, env_id),
        MujocoEnv(spec.config["base_path"_] + "/mujoco/assets/ant.xml",
                  spec.config["frame_skip"_]),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        elapsed_step_(max_episode_steps_ + 1),
        ctrl_cost_weight_(spec.config["ctrl_cost_weight"_]),
        contact_cost_weight_(spec.config["contact_cost_weight"_]),
        healthy_reward_(spec.config["healthy_reward"_]),
        healthy_z_min_(spec.config["healthy_z_min"_]),
        healthy_z_max_(spec.config["healthy_z_max"_]),
        contact_force_min_(spec.config["contact_force_min"_]),
        contact_force_max_(spec.config["contact_force_max"_]),
        dist_qpos_(-spec.config["reset_noise_scale"_],
                   spec.config["reset_noise_scale"_]),
        dist_qvel_(0, spec.config["reset_noise_scale"_]),
        done_(true) {}

  void MujocoResetModel() {
    for (int i = 0; i < model_->nq; ++i) {
      data_->qpos[i] = init_qpos_[i] + dist_qpos_(gen_);
    }
    for (int i = 0; i < model_->nv; ++i) {
      data_->qvel[i] = init_qvel_[i] + dist_qvel_(gen_);
    }
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    MujocoReset();
    WriteObs(0.0f, 0, 0, 0, 0, 0, 0);
  }

  void Step(const Action& action) override {
    // step
    mjtNum* act = static_cast<mjtNum*>(action["action"_].data());
    mjtNum x_before = data_->xpos[3], y_before = data_->xpos[4];
    MujocoStep(act);
    mjtNum x_after = data_->xpos[3], y_after = data_->xpos[4];

    // ctrl_cost
    mjtNum ctrl_cost = 0.0;
    for (int i = 0; i < model_->nu; ++i) {
      ctrl_cost += ctrl_cost_weight_ * act[i] * act[i];
    }
    // xv and yv
    mjtNum dt = frame_skip_ * model_->opt.timestep;
    mjtNum xv = (x_after - x_before) / dt;
    mjtNum yv = (y_after - y_before) / dt;
    // contact cost
    mjtNum contact_cost = 0.0;
    for (int i = 0; i < 6 * model_->nbody; ++i) {
      mjtNum x = data_->cfrc_ext[i];
      x = std::min(contact_force_max_, x);
      x = std::max(contact_force_min_, x);
      contact_cost += contact_cost_weight_ * x * x;
    }

    // reward and done
    float reward = xv + healthy_reward_ - ctrl_cost - contact_cost;
    ++elapsed_step_;
    done_ = !IsHealthy() || (elapsed_step_ >= max_episode_steps_);
    WriteObs(reward, xv, yv, ctrl_cost, contact_cost, x_after, y_after);
  }

 private:
  bool IsHealthy() {
    if (healthy_z_min_ <= data_->qpos[2] && data_->qpos[2] <= healthy_z_max_) {
      for (int i = 0; i < model_->nq; ++i) {
        if (!std::isfinite(data_->qpos[i])) {
          return false;
        }
      }
      for (int i = 0; i < model_->nv; ++i) {
        if (!std::isfinite(data_->qvel[i])) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  void WriteObs(float reward, mjtNum xv, mjtNum yv, mjtNum ctrl_cost,
                mjtNum contact_cost, mjtNum x_after,
                mjtNum y_after) {  // NOLINT
    State state = Allocate();
    state["reward"_] = reward;
    // obs
    mjtNum* obs = static_cast<mjtNum*>(state["obs"_].data());
    for (int i = 2; i < model_->nq; ++i) {
      *(obs++) = data_->qpos[i];
    }
    for (int i = 0; i < model_->nv; ++i) {
      *(obs++) = data_->qvel[i];
    }
    for (int i = 0; i < 6 * model_->nbody; ++i) {
      mjtNum x = data_->cfrc_ext[i];
      x = std::min(contact_force_max_, x);
      x = std::max(contact_force_min_, x);
      *(obs++) = x;
    }
    // info
    state["info:reward_forward"_] = xv;
    state["info:reward_ctrl"_] = -ctrl_cost;
    state["info:reward_contact"_] = -contact_cost;
    state["info:reward_survive"_] = healthy_reward_;
    state["info:x_position"_] = x_after;
    state["info:y_position"_] = y_after;
    state["info:distance_from_origin"_] =
        std::sqrt(x_after * x_after + y_after * y_after);
    state["info:x_velocity"_] = xv;
    state["info:y_velocity"_] = yv;
  }
};

typedef AsyncEnvPool<AntEnv> AntEnvPool;

}  // namespace mujoco

#endif  // ENVPOOL_MUJOCO_ANT_H_
