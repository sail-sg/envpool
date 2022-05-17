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

#ifndef ENVPOOL_MUJOCO_GYM_HUMANOID_H_
#define ENVPOOL_MUJOCO_GYM_HUMANOID_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/gym/mujoco_env.h"

namespace mujoco_gym {

class HumanoidEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "max_episode_steps"_.Bind(1000), "frame_skip"_.Bind(5),
        "post_constraint"_.Bind(true), "forward_reward_weight"_.Bind(1.25),
        "terminate_when_unhealthy"_.Bind(true),
        "exclude_current_positions_from_observation"_.Bind(true),
        "ctrl_cost_weight"_.Bind(0.1), "contact_cost_weight"_.Bind(5e-7),
        "contact_cost_max"_.Bind(10.0), "healthy_reward"_.Bind(5.0),
        "healthy_z_min"_.Bind(1.0), "healthy_z_max"_.Bind(2.0),
        "reset_noise_scale"_.Bind(1e-2));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    bool no_pos = conf["exclude_current_positions_from_observation"_];
    return MakeDict(
        "obs"_.Bind(Spec<mjtNum>({no_pos ? 376 : 378}, {-inf, inf})),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({24})),
        "info:qvel0"_.Bind(Spec<mjtNum>({23})),
#endif
        "info:reward_linvel"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_quadctrl"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_alive"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_impact"_.Bind(Spec<mjtNum>({-1})),
        "info:x_position"_.Bind(Spec<mjtNum>({-1})),
        "info:y_position"_.Bind(Spec<mjtNum>({-1})),
        "info:distance_from_origin"_.Bind(Spec<mjtNum>({-1})),
        "info:x_velocity"_.Bind(Spec<mjtNum>({-1})),
        "info:y_velocity"_.Bind(Spec<mjtNum>({-1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 17}, {-0.4, 0.4})));
  }
};

using HumanoidEnvSpec = EnvSpec<HumanoidEnvFns>;

class HumanoidEnv : public Env<HumanoidEnvSpec>, public MujocoEnv {
 protected:
  bool terminate_when_unhealthy_, no_pos_;
  mjtNum ctrl_cost_weight_, contact_cost_weight_, contact_cost_max_;
  mjtNum forward_reward_weight_, healthy_reward_;
  mjtNum healthy_z_min_, healthy_z_max_;
  std::uniform_real_distribution<> dist_;

 public:
  HumanoidEnv(const Spec& spec, int env_id)
      : Env<HumanoidEnvSpec>(spec, env_id),
        MujocoEnv(spec.config["base_path"_] + "/mujoco/assets_gym/humanoid.xml",
                  spec.config["frame_skip"_], spec.config["post_constraint"_],
                  spec.config["max_episode_steps"_]),
        terminate_when_unhealthy_(spec.config["terminate_when_unhealthy"_]),
        no_pos_(spec.config["exclude_current_positions_from_observation"_]),
        ctrl_cost_weight_(spec.config["ctrl_cost_weight"_]),
        contact_cost_weight_(spec.config["contact_cost_weight"_]),
        contact_cost_max_(spec.config["contact_cost_max"_]),
        forward_reward_weight_(spec.config["forward_reward_weight"_]),
        healthy_reward_(spec.config["healthy_reward"_]),
        healthy_z_min_(spec.config["healthy_z_min"_]),
        healthy_z_max_(spec.config["healthy_z_max"_]),
        dist_(-spec.config["reset_noise_scale"_],
              spec.config["reset_noise_scale"_]) {}

  void MujocoResetModel() override {
    for (int i = 0; i < model_->nq; ++i) {
      data_->qpos[i] = init_qpos_[i] + dist_(gen_);
    }
    for (int i = 0; i < model_->nv; ++i) {
      data_->qvel[i] = init_qvel_[i] + dist_(gen_);
    }
#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_, data_->qpos, sizeof(mjtNum) * model_->nq);
    std::memcpy(qvel0_, data_->qvel, sizeof(mjtNum) * model_->nv);
#endif
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    MujocoReset();
    WriteState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  }

  void Step(const Action& action) override {
    // step
    mjtNum* act = static_cast<mjtNum*>(action["action"_].Data());
    const auto& before = GetMassCenter();
    MujocoStep(act);
    const auto& after = GetMassCenter();

    // ctrl_cost
    mjtNum ctrl_cost = 0.0;
    for (int i = 0; i < model_->nu; ++i) {
      ctrl_cost += ctrl_cost_weight_ * act[i] * act[i];
    }
    // xv and yv
    mjtNum dt = frame_skip_ * model_->opt.timestep;
    mjtNum xv = (after[0] - before[0]) / dt;
    mjtNum yv = (after[1] - before[1]) / dt;
    // contact cost
    mjtNum contact_cost = 0.0;
    for (int i = 0; i < 6 * model_->nbody; ++i) {
      mjtNum x = data_->cfrc_ext[i];
      contact_cost += contact_cost_weight_ * x * x;
    }
    contact_cost = std::min(contact_cost, contact_cost_max_);

    // reward and done
    mjtNum healthy_reward =
        terminate_when_unhealthy_ || IsHealthy() ? healthy_reward_ : 0.0;
    auto reward = static_cast<float>(xv * forward_reward_weight_ +
                                     healthy_reward - ctrl_cost - contact_cost);
    ++elapsed_step_;
    done_ = (terminate_when_unhealthy_ ? !IsHealthy() : false) ||
            (elapsed_step_ >= max_episode_steps_);
    WriteState(reward, xv, yv, ctrl_cost, contact_cost, after[0], after[1],
               healthy_reward);
  }

 private:
  bool IsHealthy() {
    return healthy_z_min_ < data_->qpos[2] && data_->qpos[2] < healthy_z_max_;
  }

  std::array<mjtNum, 2> GetMassCenter() {
    mjtNum mass_sum = 0.0;
    mjtNum mass_x = 0.0;
    mjtNum mass_y = 0.0;
    for (int i = 0; i < model_->nbody; ++i) {
      mjtNum mass = model_->body_mass[i];
      mass_sum += mass;
      mass_x += mass * data_->xipos[i * 3 + 0];
      mass_y += mass * data_->xipos[i * 3 + 1];
    }
    return {mass_x / mass_sum, mass_y / mass_sum};
  }

  void WriteState(float reward, mjtNum xv, mjtNum yv, mjtNum ctrl_cost,
                  mjtNum contact_cost, mjtNum x_after, mjtNum y_after,
                  mjtNum healthy_reward) {
    State state = Allocate();
    state["reward"_] = reward;
    // obs
    mjtNum* obs = static_cast<mjtNum*>(state["obs"_].Data());
    for (int i = no_pos_ ? 2 : 0; i < model_->nq; ++i) {
      *(obs++) = data_->qpos[i];
    }
    for (int i = 0; i < model_->nv; ++i) {
      *(obs++) = data_->qvel[i];
    }
    for (int i = 0; i < 10 * model_->nbody; ++i) {
      *(obs++) = data_->cinert[i];
    }
    for (int i = 0; i < 6 * model_->nbody; ++i) {
      *(obs++) = data_->cvel[i];
    }
    for (int i = 0; i < model_->nv; ++i) {
      *(obs++) = data_->qfrc_actuator[i];
    }
    for (int i = 0; i < 6 * model_->nbody; ++i) {
      *(obs++) = data_->cfrc_ext[i];
    }
    // info
    state["info:reward_linvel"_] = xv * forward_reward_weight_;
    state["info:reward_quadctrl"_] = -ctrl_cost;
    state["info:reward_impact"_] = -contact_cost;
    state["info:reward_alive"_] = healthy_reward;
    state["info:x_position"_] = x_after;
    state["info:y_position"_] = y_after;
    state["info:distance_from_origin"_] =
        std::sqrt(x_after * x_after + y_after * y_after);
    state["info:x_velocity"_] = xv;
    state["info:y_velocity"_] = yv;
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_, model_->nq);
    state["info:qvel0"_].Assign(qvel0_, model_->nv);
#endif
  }
};

using HumanoidEnvPool = AsyncEnvPool<HumanoidEnv>;

}  // namespace mujoco_gym

#endif  // ENVPOOL_MUJOCO_GYM_HUMANOID_H_
