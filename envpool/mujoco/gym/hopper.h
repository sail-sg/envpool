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

#ifndef ENVPOOL_MUJOCO_GYM_HOPPER_H_
#define ENVPOOL_MUJOCO_GYM_HOPPER_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/gym/mujoco_env.h"

namespace mujoco_gym {

class HopperEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(6000.0), "frame_skip"_.Bind(4),
        "frame_stack"_.Bind(1), "post_constraint"_.Bind(true),
        "terminate_when_unhealthy"_.Bind(true),
        "legacy_healthy_reward"_.Bind(true),
        "exclude_current_positions_from_observation"_.Bind(true),
        "xml_file"_.Bind(std::string("hopper.xml")),
        "gymnasium_v5_render_camera"_.Bind(false),
        "ctrl_cost_weight"_.Bind(1e-3), "forward_reward_weight"_.Bind(1.0),
        "healthy_reward"_.Bind(1.0), "velocity_min"_.Bind(-10.0),
        "velocity_max"_.Bind(10.0), "healthy_state_min"_.Bind(-100.0),
        "healthy_state_max"_.Bind(100.0), "healthy_angle_min"_.Bind(-0.2),
        "healthy_angle_max"_.Bind(0.2), "healthy_z_min"_.Bind(0.7),
        "reset_noise_scale"_.Bind(5e-3));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    bool no_pos = conf["exclude_current_positions_from_observation"_];
    return MakeDict(
        "obs"_.Bind(StackSpec(Spec<mjtNum>({no_pos ? 11 : 12}, {-inf, inf}),
                              conf["frame_stack"_])),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({6})),
        "info:qvel0"_.Bind(Spec<mjtNum>({6})),
#endif
        "info:x_position"_.Bind(Spec<mjtNum>({-1})),
        "info:x_velocity"_.Bind(Spec<mjtNum>({-1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 3}, {-1.0, 1.0})));
  }
};

using HopperEnvSpec = EnvSpec<HopperEnvFns>;
using HopperPixelEnvFns = PixelObservationEnvFns<HopperEnvFns>;
using HopperPixelEnvSpec = EnvSpec<HopperPixelEnvFns>;

template <typename EnvSpecT, bool kFromPixels>
class HopperEnvBase : public Env<EnvSpecT>, public MujocoEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  bool terminate_when_unhealthy_, no_pos_;
  bool legacy_healthy_reward_;
  bool gymnasium_v5_render_camera_;
  mjtNum ctrl_cost_weight_, forward_reward_weight_;
  mjtNum healthy_reward_, healthy_z_min_;
  mjtNum velocity_min_, velocity_max_;
  mjtNum healthy_state_min_, healthy_state_max_;
  mjtNum healthy_angle_min_, healthy_angle_max_;
  std::uniform_real_distribution<> dist_;

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  HopperEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        MujocoEnv(
            std::string(spec.config["base_path"_]) + "/mujoco/assets_gym/" +
                std::string(spec.config["xml_file"_]),
            spec.config["frame_skip"_], spec.config["post_constraint"_],
            spec.config["max_episode_steps"_], spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        terminate_when_unhealthy_(spec.config["terminate_when_unhealthy"_]),
        no_pos_(spec.config["exclude_current_positions_from_observation"_]),
        legacy_healthy_reward_(spec.config["legacy_healthy_reward"_]),
        gymnasium_v5_render_camera_(spec.config["gymnasium_v5_render_camera"_]),
        ctrl_cost_weight_(spec.config["ctrl_cost_weight"_]),
        forward_reward_weight_(spec.config["forward_reward_weight"_]),
        healthy_reward_(spec.config["healthy_reward"_]),
        healthy_z_min_(spec.config["healthy_z_min"_]),
        velocity_min_(spec.config["velocity_min"_]),
        velocity_max_(spec.config["velocity_max"_]),
        healthy_state_min_(spec.config["healthy_state_min"_]),
        healthy_state_max_(spec.config["healthy_state_max"_]),
        healthy_angle_min_(spec.config["healthy_angle_min"_]),
        healthy_angle_max_(spec.config["healthy_angle_max"_]),
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

  bool RenderCamera(mjvCamera* camera) override {
    if (!gymnasium_v5_render_camera_) {
      return false;
    }
    camera->trackbodyid = 2;
    camera->distance = 3.0;
    camera->lookat[0] = 0.0;
    camera->lookat[1] = 0.0;
    camera->lookat[2] = 1.15;
    camera->elevation = -20.0;
    ApplyGymnasiumDefaultCameraId(camera);
    return true;
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    MujocoReset();
    WriteState(0.0, 0.0, 0.0, true);
  }

  void Step(const Action& action) override {
    // step
    auto* act = static_cast<mjtNum*>(action["action"_].Data());
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
    bool is_healthy = IsHealthy();
    bool give_healthy_reward = is_healthy;
    if (legacy_healthy_reward_) {
      give_healthy_reward = terminate_when_unhealthy_ || is_healthy;
    }
    mjtNum healthy_reward = give_healthy_reward ? healthy_reward_ : 0.0;
    auto reward = static_cast<float>(xv * forward_reward_weight_ +
                                     healthy_reward - ctrl_cost);
    ++elapsed_step_;
    done_ = (terminate_when_unhealthy_ ? !is_healthy : false) ||
            (elapsed_step_ >= max_episode_steps_);
    WriteState(reward, xv, x_after, false);
  }

 private:
  bool IsHealthy() {
    mjtNum z = data_->qpos[1];
    mjtNum angle = data_->qpos[2];
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

  void WriteState(float reward, mjtNum xv, mjtNum x_after, bool reset) {
    auto state = Allocate();
    state["reward"_] = reward;
    // obs
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation(&obs_pixels, reset);
    } else {
      auto obs_state = state["obs"_];
      mjtNum* obs = PrepareObservation(&obs_state);
      for (int i = no_pos_ ? 1 : 0; i < model_->nq; ++i) {
        *(obs++) = data_->qpos[i];
      }
      for (int i = 0; i < model_->nv; ++i) {
        mjtNum x = data_->qvel[i];
        x = std::min(velocity_max_, x);
        x = std::max(velocity_min_, x);
        *(obs++) = x;
      }
      CommitObservation(&obs_state, reset);
    }
    state["info:x_position"_] = x_after;
    state["info:x_velocity"_] = xv;
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_, model_->nq);
    state["info:qvel0"_].Assign(qvel0_, model_->nv);
#endif
  }
};

using HopperEnv = HopperEnvBase<HopperEnvSpec, false>;
using HopperPixelEnv = HopperEnvBase<HopperPixelEnvSpec, true>;
using HopperEnvPool = AsyncEnvPool<HopperEnv>;
using HopperPixelEnvPool = AsyncEnvPool<HopperPixelEnv>;

}  // namespace mujoco_gym

#endif  // ENVPOOL_MUJOCO_GYM_HOPPER_H_
