// Copyright 2026 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ENVPOOL_MUJOCO_PLAYGROUND_H1_H_
#define ENVPOOL_MUJOCO_PLAYGROUND_H1_H_

#include <mujoco.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/frame_stack.h"
#include "envpool/mujoco/playground/mujoco_env.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mujoco_playground {

using envpool::mujoco::PixelObservationEnvFns;
using envpool::mujoco::RenderCameraIdOrDefault;
using envpool::mujoco::RenderHeightOrDefault;
using envpool::mujoco::RenderWidthOrDefault;
using envpool::mujoco::StackSpec;

constexpr int kH1ActionDim = 19;
constexpr int kH1Feet = 2;
constexpr int kH1FootSensors = 3;
constexpr int kH1JointHistoryDim = 19;
constexpr int kH1InplaceHistoryLen = 3;
constexpr int kH1JoystickHistoryLen = 1;
constexpr int kH1InplaceBaseObsDim = 63;
constexpr int kH1JoystickBaseObsDim = 66;
constexpr int kH1InplaceStateDim = 186;
constexpr int kH1JoystickStateDim = 113;
constexpr int kH1MaxStateDim = kH1InplaceStateDim;
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
constexpr const char* kH1FeetSites[kH1Feet] = {"left_foot", "right_foot"};
inline const std::array<const char*, kH1FootSensors>& H1LeftFootSensors() {
  static constexpr std::array<const char*, kH1FootSensors> k_names = {
      "left_foot1_floor_found", "left_foot2_floor_found",
      "left_foot3_floor_found"};
  return k_names;
}

inline const std::array<const char*, kH1FootSensors>& H1RightFootSensors() {
  static constexpr std::array<const char*, kH1FootSensors> k_names = {
      "right_foot1_floor_found", "right_foot2_floor_found",
      "right_foot3_floor_found"};
  return k_names;
}

inline const std::array<int, 15>& H1HxIdxs() {
  static constexpr std::array<int, 15> k_idxs = {0,  1,  4,  5,  6,  9,  10, 11,
                                                 12, 13, 14, 15, 16, 17, 18};
  return k_idxs;
}

inline const std::array<mjtNum, 15>& H1HxWeights() {
  static constexpr std::array<mjtNum, 15> k_weights = {5.0, 5.0, 5.0, 5.0, 5.0,
                                                       5.0, 2.0, 1.0, 1.0, 1.0,
                                                       1.0, 1.0, 1.0, 1.0, 1.0};
  return k_weights;
}

class PlaygroundH1EnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "frame_stack"_.Bind(1),
        "task_name"_.Bind(std::string("H1JoystickGaitTracking")),
        "ctrl_dt"_.Bind(0.02), "sim_dt"_.Bind(0.004),
        "early_termination"_.Bind(true), "action_scale"_.Bind(0.3),
        "history_len"_.Bind(kH1JoystickHistoryLen),
        "obs_noise_level"_.Bind(0.6), "noise_joint_pos"_.Bind(0.01),
        "noise_joint_vel"_.Bind(1.5), "noise_gyro"_.Bind(0.2),
        "noise_gravity"_.Bind(0.05), "feet_phase_scale"_.Bind(5.0),
        "tracking_lin_vel_scale"_.Bind(3.5),
        "tracking_ang_vel_scale"_.Bind(0.75), "feet_air_time_scale"_.Bind(0.0),
        "ang_vel_xy_scale"_.Bind(-0.0), "lin_vel_z_scale"_.Bind(-0.0),
        "pose_scale"_.Bind(-2.5), "foot_slip_scale"_.Bind(-0.0),
        "action_rate_scale"_.Bind(-0.01), "ang_vel_scale"_.Bind(-0.5),
        "lin_vel_scale"_.Bind(-0.5), "tracking_sigma"_.Bind(0.5),
        "lin_vel_x_min"_.Bind(-1.5), "lin_vel_x_max"_.Bind(1.5),
        "lin_vel_y_min"_.Bind(-0.5), "lin_vel_y_max"_.Bind(0.5),
        "ang_vel_yaw_min"_.Bind(-1.0), "ang_vel_yaw_max"_.Bind(1.0),
        "lin_vel_threshold"_.Bind(0.1), "ang_vel_threshold"_.Bind(0.1),
        "gait_frequency_min"_.Bind(0.5), "gait_frequency_max"_.Bind(2.0),
        "gait_count"_.Bind(1), "foot_height_min"_.Bind(0.08),
        "foot_height_max"_.Bind(0.4));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    const int state_dim = conf["history_len"_] == kH1InplaceHistoryLen
                              ? kH1InplaceStateDim
                              : kH1JoystickStateDim;
    return MakeDict(
        "obs:state"_.Bind(StackSpec(Spec<mjtNum>({state_dim}, {-inf, inf}),
                                    conf["frame_stack"_])),
        "info:terminated"_.Bind(Spec<bool>({-1})),
        "info:command"_.Bind(Spec<mjtNum>({-1, 3})),
        "info:reward_feet_phase"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_tracking_lin_vel"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_tracking_ang_vel"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_feet_air_time"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_ang_vel_xy"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_lin_vel_z"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_pose"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_foot_slip"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_action_rate"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_ang_vel"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_lin_vel"_.Bind(Spec<mjtNum>({-1}))
#ifdef ENVPOOL_TEST
            ,
        "info:qpos"_.Bind(Spec<mjtNum>({64})),
        "info:qvel"_.Bind(Spec<mjtNum>({64})),
        "info:ctrl"_.Bind(Spec<mjtNum>({64})),
        "info:qacc"_.Bind(Spec<mjtNum>({64})),
        "info:qacc_warmstart"_.Bind(Spec<mjtNum>({64})),
        "info:xfrc_applied"_.Bind(Spec<mjtNum>({256})),
        "info:site_xpos"_.Bind(Spec<mjtNum>({512})),
        "info:site_xmat"_.Bind(Spec<mjtNum>({2048})),
        "info:actuator_force"_.Bind(Spec<mjtNum>({kH1ActionDim})),
        "info:sensordata"_.Bind(Spec<mjtNum>({256})),
        "info:last_act"_.Bind(Spec<mjtNum>({kH1ActionDim})),
        "info:last_last_act"_.Bind(Spec<mjtNum>({kH1ActionDim})),
        "info:motor_targets"_.Bind(Spec<mjtNum>({kH1ActionDim})),
        "info:qpos_error_history"_.Bind(
            Spec<mjtNum>({kH1InplaceHistoryLen * kH1JointHistoryDim})),
        "info:qvel_history"_.Bind(
            Spec<mjtNum>({kH1InplaceHistoryLen * kH1JointHistoryDim})),
        "info:swing_peak"_.Bind(Spec<mjtNum>({kH1Feet})),
        "info:feet_air_time"_.Bind(Spec<mjtNum>({kH1Feet})),
        "info:last_contact"_.Bind(Spec<bool>({kH1Feet})),
        "info:lin_vel"_.Bind(Spec<mjtNum>({3})),
        "info:ang_vel"_.Bind(Spec<mjtNum>({3})),
        "info:gait_freq"_.Bind(Spec<mjtNum>({-1})),
        "info:gait"_.Bind(Spec<int>({-1})),
        "info:phase"_.Bind(Spec<mjtNum>({kH1Feet})),
        "info:phase_dt"_.Bind(Spec<mjtNum>({-1})),
        "info:foot_height"_.Bind(Spec<mjtNum>({-1})),
        "info:step"_.Bind(Spec<int>({-1})),
        "info:left_contact"_.Bind(Spec<bool>({-1})),
        "info:right_contact"_.Bind(Spec<bool>({-1}))
#endif
    );  // NOLINT
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<mjtNum>({-1, kH1ActionDim}, {-1.0, 1.0})));
  }
};

using H1Aliases = PlaygroundEnvAliases<PlaygroundH1EnvFns>;
using PlaygroundH1EnvSpec = H1Aliases::Spec;
using PlaygroundH1PixelEnvFns = H1Aliases::PixelFns;
using PlaygroundH1PixelEnvSpec = H1Aliases::PixelSpec;

template <typename EnvSpecT, bool kFromPixels>
class PlaygroundH1EnvBase : public Env<EnvSpecT>, public PlaygroundMujocoEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  bool is_inplace_{false};
  int n_substeps_{5};
  int history_len_{1};
  int state_dim_{kH1JoystickStateDim};
  int base_obs_dim_{kH1JoystickBaseObsDim};
  std::vector<mjtNum> init_qpos_;
  std::array<mjtNum, kH1ActionDim> default_pose_{};
  std::array<mjtNum, kH1ActionDim> lowers_{};
  std::array<mjtNum, kH1ActionDim> uppers_{};
  std::array<mjtNum, kH1ActionDim> hx_default_pose_{};
  std::array<mjtNum, kH1ActionDim> last_act_{};
  std::array<mjtNum, kH1ActionDim> last_last_act_{};
  std::array<mjtNum, kH1ActionDim> motor_targets_{};
  std::array<mjtNum, 3> command_{};
  std::array<mjtNum, kH1InplaceHistoryLen * kH1JointHistoryDim>
      qpos_error_history_{};
  std::array<mjtNum, kH1InplaceHistoryLen * kH1JointHistoryDim> qvel_history_{};
  std::array<mjtNum, kH1MaxStateDim> state_obs_{};
  std::array<mjtNum, kH1Feet> swing_peak_{};
  std::array<mjtNum, kH1Feet> feet_air_time_{};
  std::array<bool, kH1Feet> last_contact_{};
  std::array<mjtNum, 3> lin_vel_{};
  std::array<mjtNum, 3> ang_vel_{};
  std::array<mjtNum, kH1Feet> phase_{};
  std::array<int, kH1Feet> feet_site_ids_{};
  std::array<int, kH1Feet> foot_linvel_sensor_adrs_{};
  std::array<int, kH1FootSensors> left_floor_sensor_ids_{};
  std::array<int, kH1FootSensors> right_floor_sensor_ids_{};
  std::array<int, kH1FootSensors> left_floor_sensor_adrs_{};
  std::array<int, kH1FootSensors> right_floor_sensor_adrs_{};
  int gyro_adr_{-1};
  int upvector_adr_{-1};
  int global_linvel_adr_{-1};
  int global_angvel_adr_{-1};
  int local_linvel_adr_{-1};
  int step_{0};
  int gait_{0};
  mjtNum gait_freq_{0.0};
  mjtNum phase_dt_{0.0};
  mjtNum foot_height_{0.0};
  bool left_contact_{false};
  bool right_contact_{false};
  mjtNum reward_feet_phase_{0.0};
  mjtNum reward_tracking_lin_vel_{0.0};
  mjtNum reward_tracking_ang_vel_{0.0};
  mjtNum reward_feet_air_time_{0.0};
  mjtNum reward_ang_vel_xy_{0.0};
  mjtNum reward_lin_vel_z_{0.0};
  mjtNum reward_pose_{0.0};
  mjtNum reward_foot_slip_{0.0};
  mjtNum reward_action_rate_{0.0};
  mjtNum reward_ang_vel_{0.0};
  mjtNum reward_lin_vel_{0.0};
  std::uniform_real_distribution<mjtNum> unit_uniform_{0.0, 1.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  PlaygroundH1EnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        PlaygroundMujocoEnv(H1XmlPath(spec.config["base_path"_]),
                            spec.config["max_episode_steps"_],
                            spec.config["frame_stack"_],
                            RenderWidthOrDefault<kFromPixels>(spec.config),
                            RenderHeightOrDefault<kFromPixels>(spec.config),
                            RenderCameraIdOrDefault<kFromPixels>(spec.config)) {
    const std::string task_name = spec.config["task_name"_];
    if (task_name == "H1InplaceGaitTracking") {
      is_inplace_ = true;
    } else if (task_name != "H1JoystickGaitTracking") {
      throw std::runtime_error("Unsupported H1 task_name " + task_name);
    }
    history_len_ = spec.config["history_len"_];
    if (history_len_ != kH1JoystickHistoryLen &&
        history_len_ != kH1InplaceHistoryLen) {
      throw std::runtime_error("Unsupported H1 history_len.");
    }
    state_dim_ = is_inplace_ ? kH1InplaceStateDim : kH1JoystickStateDim;
    base_obs_dim_ = is_inplace_ ? kH1InplaceBaseObsDim : kH1JoystickBaseObsDim;
    if (model_->nq < 7 + kH1ActionDim || model_->nv < 6 + kH1ActionDim ||
        model_->nu != kH1ActionDim) {
      throw std::runtime_error("Unexpected H1 model dimensions.");
    }
    n_substeps_ = static_cast<int>(
        std::llround(spec.config["ctrl_dt"_] / spec.config["sim_dt"_]));
    model_->opt.timestep = spec.config["sim_dt"_];
    const int home_id = mj_name2id(model_, mjOBJ_KEY, "home");
    if (home_id < 0) {
      throw std::runtime_error("H1 model is missing home key.");
    }
    init_qpos_.assign(model_->key_qpos + home_id * model_->nq,
                      model_->key_qpos + (home_id + 1) * model_->nq);
    std::copy(init_qpos_.begin() + 7, init_qpos_.begin() + 7 + kH1ActionDim,
              default_pose_.begin());
    for (int i = 0; i < kH1ActionDim; ++i) {
      lowers_[i] = model_->actuator_ctrlrange[i * 2];
      uppers_[i] = model_->actuator_ctrlrange[i * 2 + 1];
    }
    for (int i = 0; i < 15; ++i) {
      hx_default_pose_[i] = default_pose_[H1HxIdxs()[i]];
    }
    for (int i = 0; i < kH1Feet; ++i) {
      feet_site_ids_[i] = RequireId(mjOBJ_SITE, kH1FeetSites[i]);
      foot_linvel_sensor_adrs_[i] =
          SensorAdr(std::string(kH1FeetSites[i]) + "_global_linvel");
    }
    for (int i = 0; i < kH1FootSensors; ++i) {
      left_floor_sensor_ids_[i] =
          RequireId(mjOBJ_SENSOR, H1LeftFootSensors()[i]);
      right_floor_sensor_ids_[i] =
          RequireId(mjOBJ_SENSOR, H1RightFootSensors()[i]);
      left_floor_sensor_adrs_[i] =
          model_->sensor_adr[left_floor_sensor_ids_[i]];
      right_floor_sensor_adrs_[i] =
          model_->sensor_adr[right_floor_sensor_ids_[i]];
    }
    gyro_adr_ = SensorAdr("gyro");
    upvector_adr_ = SensorAdr("upvector");
    global_linvel_adr_ = SensorAdr("global_linvel");
    global_angvel_adr_ = SensorAdr("global_angvel");
    local_linvel_adr_ = SensorAdr("local_linvel");
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    has_cached_render_ = false;
    done_ = false;
    terminated_ = false;
    elapsed_step_ = 0;
    step_ = 0;
    ResetRewards();
    std::fill(last_act_.begin(), last_act_.end(), 0.0);
    std::fill(last_last_act_.begin(), last_last_act_.end(), 0.0);
    std::fill(motor_targets_.begin(), motor_targets_.end(), 0.0);
    std::fill(command_.begin(), command_.end(), 0.0);
    std::fill(qpos_error_history_.begin(), qpos_error_history_.end(), 0.0);
    std::fill(qvel_history_.begin(), qvel_history_.end(), 0.0);
    std::fill(state_obs_.begin(), state_obs_.end(), 0.0);
    std::fill(swing_peak_.begin(), swing_peak_.end(), 0.0);
    std::fill(feet_air_time_.begin(), feet_air_time_.end(), 0.0);
    std::fill(last_contact_.begin(), last_contact_.end(), false);
    std::fill(lin_vel_.begin(), lin_vel_.end(), 0.0);
    std::fill(ang_vel_.begin(), ang_vel_.end(), 0.0);

    mj_resetData(model_, data_);
    std::copy(init_qpos_.begin(), init_qpos_.end(), data_->qpos);
    std::fill(data_->qvel, data_->qvel + model_->nv, 0.0);
    std::fill(data_->ctrl, data_->ctrl + model_->nu, 0.0);
    mj_forward(model_, data_);

    gait_freq_ = Uniform(spec_.config["gait_frequency_min"_],
                         spec_.config["gait_frequency_max"_]);
    phase_dt_ = 2.0 * M_PI * Dt() * gait_freq_;
    gait_ = static_cast<int>(std::floor(
        Uniform(0.0, static_cast<mjtNum>(spec_.config["gait_count"_]))));
    if (gait_ >= spec_.config["gait_count"_]) {
      gait_ = spec_.config["gait_count"_] - 1;
    }
    phase_[0] = 0.0;
    phase_[1] = gait_ == 0 ? M_PI : 0.0;
    foot_height_ = Uniform(spec_.config["foot_height_min"_],
                           spec_.config["foot_height_max"_]);
    if (!is_inplace_) {
      SampleCommand();
    }
    const std::array<bool, kH1Feet> contact = FootContactByAdr();
    left_contact_ = contact[0];
    right_contact_ = contact[1];
    UpdateObs(contact, /*add_noise=*/true);
    WriteState(0.0, true);
  }

  void Step(const Action& action) override {
    has_cached_render_ = false;
    const auto* act = static_cast<const mjtNum*>(action["action"_].Data());
    ResetRewards();

    if (is_inplace_) {
      for (int i = 0; i < kH1ActionDim; ++i) {
        const mjtNum target =
            default_pose_[i] + act[i] * spec_.config["action_scale"_];
        motor_targets_[i] = std::clamp(target, lowers_[i], uppers_[i]);
        data_->ctrl[i] = motor_targets_[i];
      }
    } else {
      for (int i = 0; i < kH1ActionDim; ++i) {
        const mjtNum target =
            data_->ctrl[i] + act[i] * spec_.config["action_scale"_];
        motor_targets_[i] = std::clamp(target, lowers_[i], uppers_[i]);
        data_->ctrl[i] = motor_targets_[i];
      }
    }
    for (int i = 0; i < n_substeps_; ++i) {
      mj_step(model_, data_);
    }

    const std::array<bool, kH1Feet> contact = FootContactForStep();
    std::array<bool, kH1Feet> first_contact{};
    if (!is_inplace_) {
      for (int i = 0; i < kH1Feet; ++i) {
        first_contact[i] =
            feet_air_time_[i] > 0.0 && (contact[i] || last_contact_[i]);
        feet_air_time_[i] += Dt();
      }
    }
    for (int i = 0; i < kH1Feet; ++i) {
      const mjtNum z = data_->site_xpos[feet_site_ids_[i] * 3 + 2];
      swing_peak_[i] = std::max(swing_peak_[i], z);
    }

    UpdateObs(contact, /*add_noise=*/true);
    terminated_ = GetTermination();
    mjtNum reward = 0.0;
    if (is_inplace_) {
      reward = ComputeInplaceReward();
    } else {
      reward = ComputeJoystickReward(act, first_contact, contact);
    }

    std::copy(last_act_.begin(), last_act_.end(), last_last_act_.begin());
    std::copy(act, act + kH1ActionDim, last_act_.begin());
    const mjtNum phase0 = phase_[0] + phase_dt_;
    const mjtNum phase1 = phase_[1] + phase_dt_;
    phase_[0] = WrapPhase(phase0);
    phase_[1] = WrapPhase(phase1);
    for (int i = 0; i < kH1Feet; ++i) {
      if (contact[i]) {
        swing_peak_[i] = 0.0;
      }
    }
    if (is_inplace_) {
      left_contact_ = contact[0];
      right_contact_ = contact[1];
    } else {
      for (int i = 0; i < kH1Feet; ++i) {
        if (contact[i]) {
          feet_air_time_[i] = 0.0;
        }
        last_contact_[i] = contact[i];
      }
      ++step_;
      if (step_ > 500) {
        SampleCommand();
      }
      if (terminated_ || step_ > 500) {
        step_ = 0;
      }
    }

    ++elapsed_step_;
    done_ = terminated_ || elapsed_step_ >= max_episode_steps_;
    WriteState(static_cast<float>(reward), false);
  }

 private:
  static std::string H1XmlPath(const std::string& base_path) {
    return base_path +
           "/mujoco/playground/assets/mujoco_playground/_src/locomotion/h1/"
           "xmls/scene_mjx_feetonly.xml";
  }

  mjtNum Dt() const { return spec_.config["ctrl_dt"_]; }

  int RequireId(mjtObj obj_type, const std::string& name) const {
    const int id = mj_name2id(model_, obj_type, name.c_str());
    if (id < 0) {
      throw std::runtime_error("Missing MuJoCo object: " + name);
    }
    return id;
  }

  int SensorAdr(const std::string& name) const {
    const int id = RequireId(mjOBJ_SENSOR, name);
    return model_->sensor_adr[id];
  }

  mjtNum Uniform(mjtNum low, mjtNum high) {
    return low + (high - low) * unit_uniform_(gen_);
  }

  static mjtNum WrapPhase(mjtNum value) {
    return std::fmod(value + M_PI, 2.0 * M_PI) - M_PI;
  }

  static mjtNum Bezier(mjtNum x) { return x * x * x + 3.0 * x * x * (1 - x); }

  static mjtNum GaitRz(mjtNum phase, mjtNum swing_height) {
    const mjtNum x = (phase + M_PI) / (2.0 * M_PI);
    if (x <= 0.5) {
      return swing_height * Bezier(2.0 * x);
    }
    return swing_height + (0.0 - swing_height) * Bezier(2.0 * x - 1.0);
  }

  void ResetRewards() {
    reward_feet_phase_ = 0.0;
    reward_tracking_lin_vel_ = 0.0;
    reward_tracking_ang_vel_ = 0.0;
    reward_feet_air_time_ = 0.0;
    reward_ang_vel_xy_ = 0.0;
    reward_lin_vel_z_ = 0.0;
    reward_pose_ = 0.0;
    reward_foot_slip_ = 0.0;
    reward_action_rate_ = 0.0;
    reward_ang_vel_ = 0.0;
    reward_lin_vel_ = 0.0;
  }

  void SampleCommand() {
    command_[0] =
        Uniform(spec_.config["lin_vel_x_min"_], spec_.config["lin_vel_x_max"_]);
    command_[1] =
        Uniform(spec_.config["lin_vel_y_min"_], spec_.config["lin_vel_y_max"_]);
    command_[2] = Uniform(spec_.config["ang_vel_yaw_min"_],
                          spec_.config["ang_vel_yaw_max"_]);
    if (std::abs(command_[0]) < spec_.config["lin_vel_threshold"_]) {
      command_[0] = 0.0;
    }
    if (std::abs(command_[1]) < spec_.config["lin_vel_threshold"_]) {
      command_[1] = 0.0;
    }
    if (std::abs(command_[2]) < spec_.config["ang_vel_threshold"_]) {
      command_[2] = 0.0;
    }
  }

  bool GetTermination() const {
    bool joint_limit = false;
    for (int i = 0; i < kH1ActionDim; ++i) {
      const mjtNum q = data_->qpos[7 + i];
      joint_limit = joint_limit || q < lowers_[i] || q > uppers_[i];
    }
    const bool fall = data_->sensordata[upvector_adr_ + 2] < 0.85;
    if (spec_.config["early_termination"_]) {
      return joint_limit || fall;
    }
    return joint_limit;
  }

  std::array<bool, kH1Feet> FootContactByAdr() const {
    return {AnySensorPositive(left_floor_sensor_adrs_),
            AnySensorPositive(right_floor_sensor_adrs_)};
  }

  std::array<bool, kH1Feet> FootContactForStep() const {
    return {AnySensorPositive(left_floor_sensor_ids_),
            AnySensorPositive(right_floor_sensor_ids_)};
  }

  template <std::size_t N>
  bool AnySensorPositive(const std::array<int, N>& indices) const {
    return std::any_of(indices.begin(), indices.end(), [this](int index) {
      return data_->sensordata[index] > 0.0;
    });
  }

  mjtNum Noise(mjtNum scale) {
    return spec_.config["obs_noise_level"_] * scale * Uniform(-1.0, 1.0);
  }

  void RollHistory(
      std::array<mjtNum, kH1InplaceHistoryLen * kH1JointHistoryDim>* history) {
    const int history_dim = history_len_ * kH1JointHistoryDim;
    for (int i = history_dim - 1; i >= kH1JointHistoryDim; --i) {
      (*history)[i] = (*history)[i - kH1JointHistoryDim];
    }
  }

  void UpdateObs(const std::array<bool, kH1Feet>& contact, bool add_noise) {
    std::array<mjtNum, kH1JoystickBaseObsDim> base_obs{};
    int index = 0;
    for (int i = 0; i < 3; ++i) {
      base_obs[index++] =
          data_->sensordata[gyro_adr_ + i] +
          (add_noise ? Noise(spec_.config["noise_gyro"_]) : 0.0);
    }
    for (int i = 0; i < 3; ++i) {
      base_obs[index++] =
          data_->sensordata[upvector_adr_ + i] +
          (add_noise ? Noise(spec_.config["noise_gravity"_]) : 0.0);
    }
    for (int i = 0; i < kH1ActionDim; ++i) {
      base_obs[index++] =
          data_->qpos[7 + i] - default_pose_[i] +
          (add_noise ? Noise(spec_.config["noise_joint_pos"_]) : 0.0);
    }
    for (int i = 0; i < kH1ActionDim; ++i) {
      base_obs[index++] =
          data_->qvel[6 + i] +
          (add_noise ? Noise(spec_.config["noise_joint_vel"_]) : 0.0);
    }
    for (int i = 0; i < kH1ActionDim; ++i) {
      base_obs[index++] = last_act_[i];
    }
    if (!is_inplace_) {
      for (int i = 0; i < 3; ++i) {
        base_obs[index++] = command_[i];
      }
    }

    RollHistory(&qvel_history_);
    RollHistory(&qpos_error_history_);
    for (int i = 0; i < kH1ActionDim; ++i) {
      qvel_history_[i] = data_->qvel[6 + i];
      qpos_error_history_[i] = data_->qpos[7 + i] - motor_targets_[i];
    }

    std::fill(state_obs_.begin(), state_obs_.end(), 0.0);
    int out = 0;
    for (int i = 0; i < base_obs_dim_; ++i) {
      state_obs_[out++] = base_obs[i];
    }
    const int history_dim = history_len_ * kH1JointHistoryDim;
    for (int i = 0; i < history_dim; ++i) {
      state_obs_[out++] = qvel_history_[i];
    }
    for (int i = 0; i < history_dim; ++i) {
      state_obs_[out++] = qpos_error_history_[i];
    }
    state_obs_[out++] = contact[0] ? 1.0 : 0.0;
    state_obs_[out++] = contact[1] ? 1.0 : 0.0;
    state_obs_[out++] = std::cos(phase_[0]);
    state_obs_[out++] = std::cos(phase_[1]);
    state_obs_[out++] = std::sin(phase_[0]);
    state_obs_[out++] = std::sin(phase_[1]);
    state_obs_[out++] = gait_freq_;
    state_obs_[out++] = static_cast<mjtNum>(gait_);
    state_obs_[out++] = foot_height_;
  }

  mjtNum CostPose() const {
    mjtNum cost = 0.0;
    for (int i = 0; i < 15; ++i) {
      const mjtNum delta = data_->qpos[7 + H1HxIdxs()[i]] - hx_default_pose_[i];
      cost += delta * delta * H1HxWeights()[i];
    }
    return cost;
  }

  mjtNum RewardFeetPhase(mjtNum denominator) const {
    mjtNum error = 0.0;
    for (int foot = 0; foot < kH1Feet; ++foot) {
      const mjtNum foot_z = data_->site_xpos[feet_site_ids_[foot] * 3 + 2];
      const mjtNum rz = GaitRz(phase_[foot], foot_height_);
      const mjtNum delta = foot_z - rz;
      error += delta * delta;
    }
    return std::exp(-error / denominator);
  }

  mjtNum ComputeInplaceReward() {
    reward_feet_phase_ =
        RewardFeetPhase(0.1) * spec_.config["feet_phase_scale"_];
    const mjtNum ang_x = data_->sensordata[global_angvel_adr_ + 0];
    const mjtNum ang_y = data_->sensordata[global_angvel_adr_ + 1];
    const mjtNum ang_z = data_->sensordata[global_angvel_adr_ + 2];
    reward_ang_vel_ = (ang_x * ang_x + ang_y * ang_y + ang_z * ang_z) *
                      spec_.config["ang_vel_scale"_];
    const mjtNum lin_x = data_->sensordata[global_linvel_adr_ + 0];
    const mjtNum lin_y = data_->sensordata[global_linvel_adr_ + 1];
    reward_lin_vel_ =
        (lin_x * lin_x + lin_y * lin_y) * spec_.config["lin_vel_scale"_];
    reward_pose_ = CostPose() * spec_.config["pose_scale"_];
    const mjtNum neg = reward_ang_vel_ + reward_lin_vel_ + reward_pose_;
    return reward_feet_phase_ * std::exp(0.2 * neg) * Dt();
  }

  mjtNum ComputeJoystickReward(const mjtNum* act,
                               const std::array<bool, kH1Feet>& first_contact,
                               const std::array<bool, kH1Feet>& contact) {
    const mjtNum lin_x_error =
        (command_[0] - data_->sensordata[local_linvel_adr_ + 0]) *
        (command_[0] - data_->sensordata[local_linvel_adr_ + 0]);
    const mjtNum lin_y_error =
        (command_[1] - data_->sensordata[local_linvel_adr_ + 1]) *
        (command_[1] - data_->sensordata[local_linvel_adr_ + 1]);
    reward_tracking_lin_vel_ = std::exp(-(lin_x_error + lin_y_error) /
                                        spec_.config["tracking_sigma"_]) *
                               spec_.config["tracking_lin_vel_scale"_];

    const mjtNum yaw_error = command_[2] - data_->sensordata[gyro_adr_ + 2];
    reward_tracking_ang_vel_ =
        std::exp(-(yaw_error * yaw_error) / spec_.config["tracking_sigma"_]) *
        spec_.config["tracking_ang_vel_scale"_];

    reward_feet_phase_ =
        RewardFeetPhase(0.01) * spec_.config["feet_phase_scale"_];
    mjtNum feet_air_time = 0.0;
    for (int foot = 0; foot < kH1Feet; ++foot) {
      if (first_contact[foot]) {
        feet_air_time += feet_air_time_[foot] - 0.1;
      }
    }
    const mjtNum cmd_norm =
        std::sqrt(command_[0] * command_[0] + command_[1] * command_[1]);
    reward_feet_air_time_ = (cmd_norm > 0.05 ? feet_air_time : 0.0) *
                            spec_.config["feet_air_time_scale"_];

    const mjtNum ang_x = data_->sensordata[global_angvel_adr_ + 0];
    const mjtNum ang_y = data_->sensordata[global_angvel_adr_ + 1];
    reward_ang_vel_xy_ =
        (ang_x * ang_x + ang_y * ang_y) * spec_.config["ang_vel_xy_scale"_];
    const mjtNum lin_z = data_->sensordata[global_linvel_adr_ + 2];
    reward_lin_vel_z_ =
        (gait_ > 0 ? lin_z * lin_z : 0.0) * spec_.config["lin_vel_z_scale"_];
    reward_pose_ = CostPose() * spec_.config["pose_scale"_];

    mjtNum foot_slip = 0.0;
    for (int foot = 0; foot < kH1Feet; ++foot) {
      const int adr = foot_linvel_sensor_adrs_[foot];
      const mjtNum vx = data_->sensordata[adr + 0];
      const mjtNum vy = data_->sensordata[adr + 1];
      if (contact[foot]) {
        foot_slip += vx * vx + vy * vy;
      }
    }
    reward_foot_slip_ = foot_slip * spec_.config["foot_slip_scale"_];

    mjtNum action_rate = 0.0;
    for (int i = 0; i < kH1ActionDim; ++i) {
      const mjtNum c1 = last_act_[i] - last_last_act_[i];
      const mjtNum c2 = last_act_[i] - 2.0 * last_last_act_[i] + act[i];
      action_rate += c1 * c1 + c2 * c2;
    }
    reward_action_rate_ = action_rate * spec_.config["action_rate_scale"_];

    mjtNum reward = (reward_tracking_lin_vel_ + reward_tracking_ang_vel_ +
                     reward_feet_phase_ + reward_feet_air_time_ +
                     reward_ang_vel_xy_ + reward_lin_vel_z_ + reward_pose_ +
                     reward_foot_slip_ + reward_action_rate_) *
                    Dt();
    return std::max(reward, static_cast<mjtNum>(0.0));
  }

  void WriteState(float reward, bool reset) {
    auto state = Allocate();
    state["reward"_] = reward;
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      auto obs_state = state["obs:state"_];
      mjtNum* obs = PrepareObservation("obs:state", &obs_state);
      std::copy(state_obs_.begin(), state_obs_.begin() + state_dim_, obs);
      CommitObservation("obs:state", &obs_state, reset);
    }
    state["info:terminated"_] = terminated_;
    state["info:command"_].Assign(command_.data(), command_.size());
    state["info:reward_feet_phase"_] = reward_feet_phase_;
    state["info:reward_tracking_lin_vel"_] = reward_tracking_lin_vel_;
    state["info:reward_tracking_ang_vel"_] = reward_tracking_ang_vel_;
    state["info:reward_feet_air_time"_] = reward_feet_air_time_;
    state["info:reward_ang_vel_xy"_] = reward_ang_vel_xy_;
    state["info:reward_lin_vel_z"_] = reward_lin_vel_z_;
    state["info:reward_pose"_] = reward_pose_;
    state["info:reward_foot_slip"_] = reward_foot_slip_;
    state["info:reward_action_rate"_] = reward_action_rate_;
    state["info:reward_ang_vel"_] = reward_ang_vel_;
    state["info:reward_lin_vel"_] = reward_lin_vel_;
#ifdef ENVPOOL_TEST
    std::array<mjtNum, 64> pad{};
    std::copy(data_->qpos, data_->qpos + model_->nq, pad.begin());
    state["info:qpos"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(data_->qvel, data_->qvel + model_->nv, pad.begin());
    state["info:qvel"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(data_->ctrl, data_->ctrl + model_->nu, pad.begin());
    state["info:ctrl"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(data_->qacc, data_->qacc + model_->nv, pad.begin());
    state["info:qacc"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(data_->qacc_warmstart, data_->qacc_warmstart + model_->nv,
              pad.begin());
    state["info:qacc_warmstart"_].Assign(pad.data(), pad.size());
    std::array<mjtNum, 256> force_pad{};
    std::copy(data_->xfrc_applied, data_->xfrc_applied + model_->nbody * 6,
              force_pad.begin());
    state["info:xfrc_applied"_].Assign(force_pad.data(), force_pad.size());
    std::array<mjtNum, 512> site_pos_pad{};
    std::copy(data_->site_xpos, data_->site_xpos + model_->nsite * 3,
              site_pos_pad.begin());
    state["info:site_xpos"_].Assign(site_pos_pad.data(), site_pos_pad.size());
    std::array<mjtNum, 2048> site_mat_pad{};
    std::copy(data_->site_xmat, data_->site_xmat + model_->nsite * 9,
              site_mat_pad.begin());
    state["info:site_xmat"_].Assign(site_mat_pad.data(), site_mat_pad.size());
    state["info:actuator_force"_].Assign(data_->actuator_force, kH1ActionDim);
    std::array<mjtNum, 256> sensor_pad{};
    std::copy(data_->sensordata, data_->sensordata + model_->nsensordata,
              sensor_pad.begin());
    state["info:sensordata"_].Assign(sensor_pad.data(), sensor_pad.size());
    state["info:last_act"_].Assign(last_act_.data(), last_act_.size());
    state["info:last_last_act"_].Assign(last_last_act_.data(),
                                        last_last_act_.size());
    state["info:motor_targets"_].Assign(motor_targets_.data(),
                                        motor_targets_.size());
    state["info:qpos_error_history"_].Assign(qpos_error_history_.data(),
                                             qpos_error_history_.size());
    state["info:qvel_history"_].Assign(qvel_history_.data(),
                                       qvel_history_.size());
    state["info:swing_peak"_].Assign(swing_peak_.data(), swing_peak_.size());
    state["info:feet_air_time"_].Assign(feet_air_time_.data(),
                                        feet_air_time_.size());
    state["info:last_contact"_].Assign(last_contact_.data(),
                                       last_contact_.size());
    for (int i = 0; i < 3; ++i) {
      lin_vel_[i] = data_->sensordata[global_linvel_adr_ + i];
      ang_vel_[i] = data_->sensordata[global_angvel_adr_ + i];
    }
    state["info:lin_vel"_].Assign(lin_vel_.data(), lin_vel_.size());
    state["info:ang_vel"_].Assign(ang_vel_.data(), ang_vel_.size());
    state["info:gait_freq"_] = gait_freq_;
    state["info:gait"_] = gait_;
    state["info:phase"_].Assign(phase_.data(), phase_.size());
    state["info:phase_dt"_] = phase_dt_;
    state["info:foot_height"_] = foot_height_;
    state["info:step"_] = step_;
    state["info:left_contact"_] = left_contact_;
    state["info:right_contact"_] = right_contact_;
#endif
  }
};

template <typename Spec, bool kFromPixels>
using H1Base = PlaygroundH1EnvBase<Spec, kFromPixels>;
using H1Env = H1Base<PlaygroundH1EnvSpec, false>;
using H1PixelEnv = H1Base<PlaygroundH1PixelEnvSpec, true>;
using PlaygroundH1Env = H1Env;
using PlaygroundH1PixelEnv = H1PixelEnv;
using PlaygroundH1EnvPool = PlaygroundEnvPoolT<PlaygroundH1Env>;
using PlaygroundH1PixelEnvPool = PlaygroundEnvPoolT<PlaygroundH1PixelEnv>;

}  // namespace mujoco_playground

#endif  // ENVPOOL_MUJOCO_PLAYGROUND_H1_H_
