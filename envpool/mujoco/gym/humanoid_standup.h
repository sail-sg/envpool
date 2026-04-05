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

#ifndef ENVPOOL_MUJOCO_GYM_HUMANOID_STANDUP_H_
#define ENVPOOL_MUJOCO_GYM_HUMANOID_STANDUP_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/gym/mujoco_env.h"

namespace mujoco_gym {

class HumanoidStandupEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "frame_skip"_.Bind(5), "frame_stack"_.Bind(1),
        "post_constraint"_.Bind(true), "forward_reward_weight"_.Bind(1.0),
        "exclude_current_positions_from_observation"_.Bind(true),
        "exclude_worldbody_observations"_.Bind(false),
        "exclude_root_actuator_forces"_.Bind(false),
        "xml_file"_.Bind(std::string("humanoidstandup.xml")),
        "ctrl_cost_weight"_.Bind(0.1), "contact_cost_weight"_.Bind(5e-7),
        "contact_cost_max"_.Bind(10.0), "healthy_reward"_.Bind(1.0),
        "reset_noise_scale"_.Bind(1e-2));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    bool no_pos = conf["exclude_current_positions_from_observation"_];
    int obs_n = no_pos ? 376 : 378;
    if (conf["exclude_worldbody_observations"_]) {
      obs_n -= 10 + 6 + 6;
    }
    if (conf["exclude_root_actuator_forces"_]) {
      obs_n -= 6;
    }
    return MakeDict(
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({24})),
        "info:qvel0"_.Bind(Spec<mjtNum>({23})),
#endif
        "obs"_.Bind(StackSpec(Spec<mjtNum>({obs_n}, {-inf, inf}),
                              conf["frame_stack"_])),
        "info:reward_linup"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_quadctrl"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_alive"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_impact"_.Bind(Spec<mjtNum>({-1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 17}, {-0.4, 0.4})));
  }
};

using HumanoidStandupEnvSpec = EnvSpec<HumanoidStandupEnvFns>;
using HumanoidStandupPixelEnvFns =
PixelObservationEnvFns<HumanoidStandupEnvFns>;
using HumanoidStandupPixelEnvSpec = EnvSpec<HumanoidStandupPixelEnvFns>;

template <typename EnvSpecT, bool kFromPixels>
class HumanoidStandupEnvBase : public Env<EnvSpecT>, public MujocoEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  bool no_pos_;
  bool exclude_worldbody_observations_, exclude_root_actuator_forces_;
  mjtNum ctrl_cost_weight_, contact_cost_weight_, contact_cost_max_;
  mjtNum forward_reward_weight_, healthy_reward_;
  std::uniform_real_distribution<> dist_;

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  HumanoidStandupEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        MujocoEnv(
            std::string(spec.config["base_path"_]) + "/mujoco/assets_gym/" +
                std::string(spec.config["xml_file"_]),
            spec.config["frame_skip"_], spec.config["post_constraint"_],
            spec.config["max_episode_steps"_], spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        no_pos_(spec.config["exclude_current_positions_from_observation"_]),
        exclude_worldbody_observations_(
            spec.config["exclude_worldbody_observations"_]),
        exclude_root_actuator_forces_(
            spec.config["exclude_root_actuator_forces"_]),
        ctrl_cost_weight_(spec.config["ctrl_cost_weight"_]),
        contact_cost_weight_(spec.config["contact_cost_weight"_]),
        contact_cost_max_(spec.config["contact_cost_max"_]),
        forward_reward_weight_(spec.config["forward_reward_weight"_]),
        healthy_reward_(spec.config["healthy_reward"_]),
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
    WriteState(0.0, 0.0, 0.0, 0.0, true);
  }

  void Step(const Action& action) override {
    // step
    mjtNum* act = static_cast<mjtNum*>(action["action"_].Data());
    MujocoStep(act);

    // ctrl_cost
    mjtNum ctrl_cost = 0.0;
    for (int i = 0; i < model_->nu; ++i) {
      ctrl_cost += ctrl_cost_weight_ * act[i] * act[i];
    }
    // xv
    mjtNum xv = data_->qpos[2] / model_->opt.timestep;
    // contact cost
    mjtNum contact_cost = 0.0;
    for (int i = 0; i < 6 * model_->nbody; ++i) {
      mjtNum x = data_->cfrc_ext[i];
      contact_cost += contact_cost_weight_ * x * x;
    }
    contact_cost = std::min(contact_cost, contact_cost_max_);

    // reward and done
    auto reward =
        static_cast<float>(xv * forward_reward_weight_ + healthy_reward_ -
                           ctrl_cost - contact_cost);
    done_ = (++elapsed_step_ >= max_episode_steps_);
    WriteState(reward, xv, ctrl_cost, contact_cost, false);
  }

 private:
  void WriteState(float reward, mjtNum xv, mjtNum ctrl_cost,
                  mjtNum contact_cost, bool reset) {
    auto state = Allocate();
    state["reward"_] = reward;
    // obs
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation(&obs_pixels, reset);
    } else {
      auto obs_state = state["obs"_];
      mjtNum* obs = PrepareObservation(&obs_state);
      for (int i = no_pos_ ? 2 : 0; i < model_->nq; ++i) {
        *(obs++) = data_->qpos[i];
      }
      for (int i = 0; i < model_->nv; ++i) {
        *(obs++) = data_->qvel[i];
      }
      int start_body = exclude_worldbody_observations_ ? 1 : 0;
      for (int i = start_body; i < model_->nbody; ++i) {
        for (int j = 0; j < 10; ++j) {
          *(obs++) = data_->cinert[i * 10 + j];
        }
      }
      for (int i = start_body; i < model_->nbody; ++i) {
        for (int j = 0; j < 6; ++j) {
          *(obs++) = data_->cvel[i * 6 + j];
        }
      }
      for (int i = exclude_root_actuator_forces_ ? 6 : 0; i < model_->nv; ++i) {
        *(obs++) = data_->qfrc_actuator[i];
      }
      for (int i = start_body; i < model_->nbody; ++i) {
        for (int j = 0; j < 6; ++j) {
          *(obs++) = data_->cfrc_ext[i * 6 + j];
        }
      }
      CommitObservation(&obs_state, reset);
    }
    state["info:reward_linup"_] = xv * forward_reward_weight_;
    state["info:reward_quadctrl"_] = -ctrl_cost;
    state["info:reward_impact"_] = -contact_cost;
    state["info:reward_alive"_] = healthy_reward_;
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_, model_->nq);
    state["info:qvel0"_].Assign(qvel0_, model_->nv);
#endif
  }
};
using HumanoidStandupEnv =
HumanoidStandupEnvBase<HumanoidStandupEnvSpec, false>;
using HumanoidStandupPixelEnv =
HumanoidStandupEnvBase<HumanoidStandupPixelEnvSpec, true>;
using HumanoidStandupEnvPool = AsyncEnvPool<HumanoidStandupEnv>;
using HumanoidStandupPixelEnvPool = AsyncEnvPool<HumanoidStandupPixelEnv>;

}  // namespace mujoco_gym

#endif  // ENVPOOL_MUJOCO_GYM_HUMANOID_STANDUP_H_
