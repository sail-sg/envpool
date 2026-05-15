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

#ifndef ENVPOOL_MUJOCO_PLAYGROUND_T1_H_
#define ENVPOOL_MUJOCO_PLAYGROUND_T1_H_

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

constexpr int kT1ActionDim = 23;
constexpr int kT1Feet = 2;
constexpr int kT1FootSensors = 4;
constexpr int kT1ObsDim = 85;
constexpr int kT1PrivilegedStateDim = 180;
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
constexpr const char* kT1FeetSites[kT1Feet] = {"left_foot", "right_foot"};
inline const std::array<const char*, kT1FootSensors>& T1LeftFootSensors() {
  static constexpr std::array<const char*, kT1FootSensors> k_names = {
      "left_foot_1_floor_found", "left_foot_2_floor_found",
      "left_foot_3_floor_found", "left_foot_4_floor_found"};
  return k_names;
}

inline const std::array<const char*, kT1FootSensors>& T1RightFootSensors() {
  static constexpr std::array<const char*, kT1FootSensors> k_names = {
      "right_foot_1_floor_found", "right_foot_2_floor_found",
      "right_foot_3_floor_found", "right_foot_4_floor_found"};
  return k_names;
}

class PlaygroundT1EnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "frame_stack"_.Bind(1),
        "task_name"_.Bind(std::string("T1JoystickFlatTerrain")),
        "ctrl_dt"_.Bind(0.02), "sim_dt"_.Bind(0.002), "action_scale"_.Bind(1.0),
        "soft_joint_pos_limit_factor"_.Bind(0.95), "noise_level"_.Bind(1.0),
        "noise_joint_pos"_.Bind(0.03), "noise_joint_vel"_.Bind(1.5),
        "noise_gravity"_.Bind(0.05), "noise_linvel"_.Bind(0.1),
        "noise_gyro"_.Bind(0.2), "tracking_lin_vel_scale"_.Bind(1.0),
        "tracking_ang_vel_scale"_.Bind(0.5), "lin_vel_z_scale"_.Bind(0.0),
        "ang_vel_xy_scale"_.Bind(-0.15), "orientation_scale"_.Bind(-1.0),
        "base_height_scale"_.Bind(0.0), "torques_scale"_.Bind(0.0),
        "action_rate_scale"_.Bind(0.0), "energy_scale"_.Bind(0.0),
        "dof_acc_scale"_.Bind(0.0), "dof_vel_scale"_.Bind(0.0),
        "feet_clearance_scale"_.Bind(0.0), "feet_air_time_scale"_.Bind(2.0),
        "feet_slip_scale"_.Bind(-0.25), "feet_height_scale"_.Bind(0.0),
        "feet_phase_scale"_.Bind(1.0), "stand_still_scale"_.Bind(0.0),
        "alive_scale"_.Bind(0.25), "termination_scale"_.Bind(0.0),
        "collision_scale"_.Bind(-1.0), "joint_deviation_knee_scale"_.Bind(-0.1),
        "joint_deviation_hip_scale"_.Bind(-0.1),
        "dof_pos_limits_scale"_.Bind(-1.0), "pose_scale"_.Bind(-1.0),
        "feet_distance_scale"_.Bind(-1.0), "tracking_sigma"_.Bind(0.25),
        "max_foot_height"_.Bind(0.12), "base_height_target"_.Bind(0.665),
        "push_enable"_.Bind(true), "push_interval_min"_.Bind(5.0),
        "push_interval_max"_.Bind(10.0), "push_magnitude_min"_.Bind(0.1),
        "push_magnitude_max"_.Bind(1.0), "lin_vel_x_min"_.Bind(-1.0),
        "lin_vel_x_max"_.Bind(1.0), "lin_vel_y_min"_.Bind(-0.8),
        "lin_vel_y_max"_.Bind(0.8), "ang_vel_yaw_min"_.Bind(-1.0),
        "ang_vel_yaw_max"_.Bind(1.0));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs:state"_.Bind(StackSpec(Spec<mjtNum>({kT1ObsDim}, {-inf, inf}),
                                    conf["frame_stack"_])),
        "obs:privileged_state"_.Bind(
            StackSpec(Spec<mjtNum>({kT1PrivilegedStateDim}, {-inf, inf}),
                      conf["frame_stack"_])),
        "info:terminated"_.Bind(Spec<bool>({-1})),
        "info:command"_.Bind(Spec<mjtNum>({-1, 3})),
        "info:reward_tracking_lin_vel"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_tracking_ang_vel"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_lin_vel_z"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_ang_vel_xy"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_orientation"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_base_height"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_torques"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_action_rate"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_energy"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_dof_acc"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_dof_vel"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_feet_clearance"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_feet_air_time"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_feet_slip"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_feet_height"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_feet_phase"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_stand_still"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_alive"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_termination"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_collision"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_joint_deviation_knee"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_joint_deviation_hip"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_dof_pos_limits"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_pose"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_feet_distance"_.Bind(Spec<mjtNum>({-1}))
#ifdef ENVPOOL_TEST
            ,
        "info:qpos0"_.Bind(Spec<mjtNum>({64})),
        "info:qvel0"_.Bind(Spec<mjtNum>({64})),
        "info:ctrl0"_.Bind(Spec<mjtNum>({64})),
        "info:qpos"_.Bind(Spec<mjtNum>({64})),
        "info:qvel"_.Bind(Spec<mjtNum>({64})),
        "info:ctrl"_.Bind(Spec<mjtNum>({64})),
        "info:qacc"_.Bind(Spec<mjtNum>({64})),
        "info:qacc_warmstart"_.Bind(Spec<mjtNum>({64})),
        "info:xfrc_applied"_.Bind(Spec<mjtNum>({256})),
        "info:last_act"_.Bind(Spec<mjtNum>({kT1ActionDim})),
        "info:last_last_act"_.Bind(Spec<mjtNum>({kT1ActionDim})),
        "info:motor_targets"_.Bind(Spec<mjtNum>({kT1ActionDim})),
        "info:actuator_force"_.Bind(Spec<mjtNum>({kT1ActionDim})),
        "info:feet_air_time"_.Bind(Spec<mjtNum>({kT1Feet})),
        "info:last_contact"_.Bind(Spec<bool>({kT1Feet})),
        "info:swing_peak"_.Bind(Spec<mjtNum>({kT1Feet})),
        "info:phase"_.Bind(Spec<mjtNum>({2})),
        "info:phase_dt"_.Bind(Spec<mjtNum>({1})),
        "info:push"_.Bind(Spec<mjtNum>({2})),
        "info:push_step"_.Bind(Spec<int>({-1})),
        "info:push_interval_steps"_.Bind(Spec<int>({-1})),
        "info:step"_.Bind(Spec<int>({-1})),
        "info:site_xpos"_.Bind(Spec<mjtNum>({256})),
        "info:site_xmat"_.Bind(Spec<mjtNum>({256})),
        "info:sensordata"_.Bind(Spec<mjtNum>({256}))
#endif
    );  // NOLINT
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<mjtNum>({-1, kT1ActionDim}, {-1.0, 1.0})));
  }
};

using T1Aliases = PlaygroundEnvAliases<PlaygroundT1EnvFns>;
using PlaygroundT1EnvSpec = T1Aliases::Spec;
using PlaygroundT1PixelEnvFns = T1Aliases::PixelFns;
using PlaygroundT1PixelEnvSpec = T1Aliases::PixelSpec;

template <typename EnvSpecT, bool kFromPixels>
class PlaygroundT1EnvBase : public Env<EnvSpecT>, public PlaygroundMujocoEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  int n_substeps_{10};
  std::vector<mjtNum> init_qpos_;
  std::vector<mjtNum> reset_qpos_;
  std::vector<mjtNum> reset_qvel_;
  std::vector<mjtNum> reset_ctrl_;
  std::array<mjtNum, kT1ActionDim> default_pose_{};
  std::array<mjtNum, kT1ActionDim> qpos_noise_scale_{};
  std::array<mjtNum, kT1ActionDim> soft_lowers_{};
  std::array<mjtNum, kT1ActionDim> soft_uppers_{};
  std::array<mjtNum, kT1ActionDim> weights_{};
  std::array<int, 4> hip_indices_{};
  std::array<int, 2> knee_indices_{};
  std::array<mjtNum, kT1ActionDim> last_act_{};
  std::array<mjtNum, kT1ActionDim> last_last_act_{};
  std::array<mjtNum, kT1ActionDim> motor_targets_{};
  std::array<mjtNum, 3> command_{};
  std::array<mjtNum, kT1Feet> feet_air_time_{};
  std::array<mjtNum, kT1Feet> swing_peak_{};
  std::array<bool, kT1Feet> last_contact_{};
  std::array<mjtNum, 2> phase_{};
  std::array<mjtNum, 1> phase_dt_{};
  std::array<mjtNum, 2> push_{};
  std::array<int, kT1Feet> feet_site_ids_{};
  std::array<int, kT1Feet> feet_linvel_sensor_adrs_{};
  std::array<int, kT1FootSensors> left_floor_sensor_ids_{};
  std::array<int, kT1FootSensors> right_floor_sensor_ids_{};
  int pelvis_imu_site_id_{-1};
  int gyro_adr_{-1};
  int local_linvel_adr_{-1};
  int accelerometer_adr_{-1};
  int upvector_adr_{-1};
  int global_linvel_adr_{-1};
  int global_angvel_adr_{-1};
  int left_foot_right_foot_sensor_id_{-1};
  int step_{0};
  int push_step_{0};
  int push_interval_steps_{1};
  mjtNum reward_tracking_lin_vel_{0.0};
  mjtNum reward_tracking_ang_vel_{0.0};
  mjtNum reward_lin_vel_z_{0.0};
  mjtNum reward_ang_vel_xy_{0.0};
  mjtNum reward_orientation_{0.0};
  mjtNum reward_base_height_{0.0};
  mjtNum reward_torques_{0.0};
  mjtNum reward_action_rate_{0.0};
  mjtNum reward_energy_{0.0};
  mjtNum reward_dof_acc_{0.0};
  mjtNum reward_dof_vel_{0.0};
  mjtNum reward_feet_clearance_{0.0};
  mjtNum reward_feet_air_time_{0.0};
  mjtNum reward_feet_slip_{0.0};
  mjtNum reward_feet_height_{0.0};
  mjtNum reward_feet_phase_{0.0};
  mjtNum reward_stand_still_{0.0};
  mjtNum reward_alive_{0.0};
  mjtNum reward_termination_{0.0};
  mjtNum reward_collision_{0.0};
  mjtNum reward_joint_deviation_knee_{0.0};
  mjtNum reward_joint_deviation_hip_{0.0};
  mjtNum reward_dof_pos_limits_{0.0};
  mjtNum reward_pose_{0.0};
  mjtNum reward_feet_distance_{0.0};
  std::uniform_real_distribution<mjtNum> unit_uniform_{0.0, 1.0};
  std::uniform_real_distribution<mjtNum> noise_uniform_{-1.0, 1.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  PlaygroundT1EnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        PlaygroundMujocoEnv(
            T1XmlPath(spec.config["base_path"_], spec.config["task_name"_]),
            spec.config["max_episode_steps"_], spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)) {
    const std::string task_name = spec.config["task_name"_];
    if (task_name != "T1JoystickFlatTerrain" &&
        task_name != "T1JoystickRoughTerrain") {
      throw std::runtime_error("Unsupported T1 task_name " + task_name);
    }
    if (model_->nq < 7 + kT1ActionDim || model_->nv < 6 + kT1ActionDim ||
        model_->nu != kT1ActionDim) {
      throw std::runtime_error("Unexpected T1 dimensions.");
    }
    n_substeps_ = static_cast<int>(
        std::llround(spec.config["ctrl_dt"_] / spec.config["sim_dt"_]));
    model_->opt.timestep = spec.config["sim_dt"_];
    model_->opt.ccd_iterations = 10;

    const int home_key = mj_name2id(model_, mjOBJ_KEY, "home");
    if (home_key < 0) {
      throw std::runtime_error("T1 model is missing home key.");
    }
    init_qpos_.assign(model_->key_qpos + home_key * model_->nq,
                      model_->key_qpos + (home_key + 1) * model_->nq);
    std::copy(init_qpos_.begin() + 7, init_qpos_.begin() + 7 + kT1ActionDim,
              default_pose_.begin());
    std::copy(default_pose_.begin(), default_pose_.end(),
              motor_targets_.begin());

    for (int i = 0; i < kT1ActionDim; ++i) {
      const mjtNum lower = model_->jnt_range[(i + 1) * 2];
      const mjtNum upper = model_->jnt_range[(i + 1) * 2 + 1];
      const mjtNum center = (lower + upper) * 0.5;
      const mjtNum range = upper - lower;
      soft_lowers_[i] =
          center - 0.5 * range * spec.config["soft_joint_pos_limit_factor"_];
      soft_uppers_[i] =
          center + 0.5 * range * spec.config["soft_joint_pos_limit_factor"_];
    }
    qpos_noise_scale_.fill(spec.config["noise_joint_pos"_]);
    weights_ = {1.0, 1.0, 0.1,  1.0, 1.0, 1.0,  0.1, 1.0, 1.0,  1.0, 1.0, 0.01,
                1.0, 1.0, 0.01, 1.0, 1.0, 0.01, 1.0, 1.0, 0.01, 1.0, 1.0};
    hip_indices_ = {
        JointQposIndex("Left_Hip_Roll"), JointQposIndex("Left_Hip_Yaw"),
        JointQposIndex("Right_Hip_Roll"), JointQposIndex("Right_Hip_Yaw")};
    knee_indices_ = {JointQposIndex("Left_Knee_Pitch"),
                     JointQposIndex("Right_Knee_Pitch")};

    pelvis_imu_site_id_ = RequireId(mjOBJ_SITE, "imu");
    gyro_adr_ = SensorAdr("gyro");
    local_linvel_adr_ = SensorAdr("local_linvel");
    accelerometer_adr_ = SensorAdr("accelerometer");
    upvector_adr_ = SensorAdr("upvector");
    global_linvel_adr_ = SensorAdr("global_linvel");
    global_angvel_adr_ = SensorAdr("global_angvel");
    for (int i = 0; i < kT1Feet; ++i) {
      feet_site_ids_[i] = RequireId(mjOBJ_SITE, kT1FeetSites[i]);
      feet_linvel_sensor_adrs_[i] =
          SensorAdr(std::string(kT1FeetSites[i]) + "_global_linvel");
    }
    for (int i = 0; i < kT1FootSensors; ++i) {
      left_floor_sensor_ids_[i] =
          RequireId(mjOBJ_SENSOR, T1LeftFootSensors()[i]);
      right_floor_sensor_ids_[i] =
          RequireId(mjOBJ_SENSOR, T1RightFootSensors()[i]);
    }
    left_foot_right_foot_sensor_id_ =
        RequireId(mjOBJ_SENSOR, "left_foot_right_foot_found");
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    has_cached_render_ = false;
    done_ = false;
    terminated_ = false;
    elapsed_step_ = 0;
    step_ = 0;
    push_step_ = 0;
    ResetRewards();
    std::fill(last_act_.begin(), last_act_.end(), 0.0);
    std::fill(last_last_act_.begin(), last_last_act_.end(), 0.0);
    std::fill(feet_air_time_.begin(), feet_air_time_.end(), 0.0);
    std::fill(last_contact_.begin(), last_contact_.end(), false);
    std::fill(swing_peak_.begin(), swing_peak_.end(), 0.0);
    std::fill(push_.begin(), push_.end(), 0.0);

    mj_resetData(model_, data_);
    std::copy(init_qpos_.begin(), init_qpos_.end(), data_->qpos);
    data_->qpos[0] += Uniform(-0.5, 0.5);
    data_->qpos[1] += Uniform(-0.5, 0.5);
    ApplyYaw(Uniform(-3.14, 3.14));
    for (int i = 0; i < kT1ActionDim; ++i) {
      data_->qpos[7 + i] *= Uniform(0.5, 1.5);
    }
    std::fill(data_->qvel, data_->qvel + model_->nv, 0.0);
    for (int i = 0; i < 6; ++i) {
      data_->qvel[i] = Uniform(-0.5, 0.5);
    }
    std::copy(data_->qpos + 7, data_->qpos + 7 + kT1ActionDim, data_->ctrl);
    std::copy(data_->ctrl, data_->ctrl + kT1ActionDim, motor_targets_.begin());
    mj_forward(model_, data_);
    reset_qpos_.assign(data_->qpos, data_->qpos + model_->nq);
    reset_qvel_.assign(data_->qvel, data_->qvel + model_->nv);
    reset_ctrl_.assign(data_->ctrl, data_->ctrl + model_->nu);

    const mjtNum gait_freq = Uniform(1.25, 1.75);
    phase_dt_[0] = 2.0 * M_PI * Dt() * gait_freq;
    phase_[0] = 0.0;
    phase_[1] = M_PI;
    SampleCommand();
    const mjtNum push_interval = Uniform(spec_.config["push_interval_min"_],
                                         spec_.config["push_interval_max"_]);
    push_interval_steps_ = static_cast<int>(std::round(push_interval / Dt()));

    std::array<bool, kT1Feet> contact = FootContact();
    UpdateObs(contact, /*add_noise=*/true);
    WriteState(0.0, true);
  }

  void Step(const Action& action) override {
    has_cached_render_ = false;
    const auto* act = static_cast<const mjtNum*>(action["action"_].Data());
    ApplyPush();
    for (int i = 0; i < kT1ActionDim; ++i) {
      motor_targets_[i] =
          default_pose_[i] + act[i] * spec_.config["action_scale"_];
      data_->ctrl[i] = motor_targets_[i];
    }
    for (int i = 0; i < n_substeps_; ++i) {
      mj_step(model_, data_);
    }

    const std::array<bool, kT1Feet> contact = FootContact();
    std::array<bool, kT1Feet> first_contact{};
    for (int i = 0; i < kT1Feet; ++i) {
      const bool contact_filt = contact[i] || last_contact_[i];
      first_contact[i] = feet_air_time_[i] > 0.0 && contact_filt;
      feet_air_time_[i] += Dt();
      const mjtNum foot_z = data_->site_xpos[feet_site_ids_[i] * 3 + 2];
      swing_peak_[i] = std::max(swing_peak_[i], foot_z);
    }

    UpdateObs(contact, /*add_noise=*/true);
    terminated_ = GetTermination();
    ComputeRewards(act, first_contact, contact);
    mjtNum reward =
        reward_tracking_lin_vel_ + reward_tracking_ang_vel_ +
        reward_lin_vel_z_ + reward_ang_vel_xy_ + reward_orientation_ +
        reward_base_height_ + reward_torques_ + reward_action_rate_ +
        reward_energy_ + reward_dof_acc_ + reward_dof_vel_ +
        reward_feet_clearance_ + reward_feet_air_time_ + reward_feet_slip_ +
        reward_feet_height_ + reward_feet_phase_ + reward_stand_still_ +
        reward_alive_ + reward_termination_ + reward_collision_ +
        reward_joint_deviation_knee_ + reward_joint_deviation_hip_ +
        reward_dof_pos_limits_ + reward_pose_ + reward_feet_distance_;
    reward = std::clamp(reward * Dt(), static_cast<mjtNum>(0.0),
                        static_cast<mjtNum>(10000.0));

    ++step_;
    ++push_step_;
    for (int i = 0; i < kT1Feet; ++i) {
      feet_air_time_[i] = contact[i] ? 0.0 : feet_air_time_[i];
      last_contact_[i] = contact[i];
      swing_peak_[i] = contact[i] ? 0.0 : swing_peak_[i];
    }
    UpdatePhase();
    std::copy(last_act_.begin(), last_act_.end(), last_last_act_.begin());
    std::copy(act, act + kT1ActionDim, last_act_.begin());
    if (step_ > 500) {
      SampleCommand();
    }
    if (terminated_ || step_ > 500) {
      step_ = 0;
    }
    ++elapsed_step_;
    done_ = terminated_ || elapsed_step_ >= max_episode_steps_;
    WriteState(static_cast<float>(reward), false);
  }

 private:
  static std::string T1XmlPath(const std::string& base_path,
                               const std::string& task_name) {
    const bool rough = task_name == "T1JoystickRoughTerrain";
    return base_path +
           "/mujoco/playground/assets/mujoco_playground/_src/locomotion/"
           "t1/xmls/scene_mjx_feetonly_" +
           std::string(rough ? "rough" : "flat") + "_terrain.xml";
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

  int JointQposIndex(const std::string& name) const {
    const int id = RequireId(mjOBJ_JOINT, name);
    return model_->jnt_qposadr[id] - 7;
  }

  mjtNum Uniform(mjtNum low, mjtNum high) {
    return low + (high - low) * unit_uniform_(gen_);
  }

  mjtNum Noise(mjtNum scale) {
    return spec_.config["noise_level"_] * scale * noise_uniform_(gen_);
  }

  void ApplyYaw(mjtNum yaw) {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const mjtNum yaw_quat[4] = {std::cos(0.5 * yaw), 0.0, 0.0,
                                std::sin(0.5 * yaw)};
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const mjtNum old[4] = {data_->qpos[3], data_->qpos[4], data_->qpos[5],
                           data_->qpos[6]};
    data_->qpos[3] = old[0] * yaw_quat[0] - old[1] * yaw_quat[1] -
                     old[2] * yaw_quat[2] - old[3] * yaw_quat[3];
    data_->qpos[4] = old[0] * yaw_quat[1] + old[1] * yaw_quat[0] +
                     old[2] * yaw_quat[3] - old[3] * yaw_quat[2];
    data_->qpos[5] = old[0] * yaw_quat[2] - old[1] * yaw_quat[3] +
                     old[2] * yaw_quat[0] + old[3] * yaw_quat[1];
    data_->qpos[6] = old[0] * yaw_quat[3] + old[1] * yaw_quat[2] -
                     old[2] * yaw_quat[1] + old[3] * yaw_quat[0];
  }

  void ResetRewards() {
    reward_tracking_lin_vel_ = 0.0;
    reward_tracking_ang_vel_ = 0.0;
    reward_lin_vel_z_ = 0.0;
    reward_ang_vel_xy_ = 0.0;
    reward_orientation_ = 0.0;
    reward_base_height_ = 0.0;
    reward_torques_ = 0.0;
    reward_action_rate_ = 0.0;
    reward_energy_ = 0.0;
    reward_dof_acc_ = 0.0;
    reward_dof_vel_ = 0.0;
    reward_feet_clearance_ = 0.0;
    reward_feet_air_time_ = 0.0;
    reward_feet_slip_ = 0.0;
    reward_feet_height_ = 0.0;
    reward_feet_phase_ = 0.0;
    reward_stand_still_ = 0.0;
    reward_alive_ = 0.0;
    reward_termination_ = 0.0;
    reward_collision_ = 0.0;
    reward_joint_deviation_knee_ = 0.0;
    reward_joint_deviation_hip_ = 0.0;
    reward_dof_pos_limits_ = 0.0;
    reward_pose_ = 0.0;
    reward_feet_distance_ = 0.0;
  }

  void SampleCommand() {
    command_[0] =
        Uniform(spec_.config["lin_vel_x_min"_], spec_.config["lin_vel_x_max"_]);
    command_[1] =
        Uniform(spec_.config["lin_vel_y_min"_], spec_.config["lin_vel_y_max"_]);
    command_[2] = Uniform(spec_.config["ang_vel_yaw_min"_],
                          spec_.config["ang_vel_yaw_max"_]);
    if (unit_uniform_(gen_) < 0.1) {
      std::fill(command_.begin(), command_.end(), 0.0);
    }
  }

  void ApplyPush() {
    std::fill(push_.begin(), push_.end(), 0.0);
    if (!spec_.config["push_enable"_] || push_interval_steps_ <= 0) {
      return;
    }
    if ((push_step_ + 1) % push_interval_steps_ != 0) {
      return;
    }
    const mjtNum theta = Uniform(0.0, 2.0 * M_PI);
    const mjtNum magnitude = Uniform(spec_.config["push_magnitude_min"_],
                                     spec_.config["push_magnitude_max"_]);
    push_[0] = std::cos(theta) * magnitude;
    push_[1] = std::sin(theta) * magnitude;
    data_->qvel[0] += push_[0];
    data_->qvel[1] += push_[1];
  }

  void UpdatePhase() {
    const mjtNum command_norm =
        std::sqrt(command_[0] * command_[0] + command_[1] * command_[1] +
                  command_[2] * command_[2]);
    for (int i = 0; i < kT1Feet; ++i) {
      const mjtNum phase_next = phase_[i] + phase_dt_[0];
      phase_[i] = std::fmod(phase_next + M_PI, 2.0 * M_PI) - M_PI;
      if (command_norm <= 0.01) {
        phase_[i] = M_PI;
      }
    }
  }

  std::array<bool, kT1Feet> FootContact() const {
    std::array<bool, kT1Feet> contact{};
    for (int i = 0; i < kT1FootSensors; ++i) {
      contact[0] =
          contact[0] ||
          data_->sensordata[model_->sensor_adr[left_floor_sensor_ids_[i]]] >
              0.0;
      contact[1] =
          contact[1] ||
          data_->sensordata[model_->sensor_adr[right_floor_sensor_ids_[i]]] >
              0.0;
    }
    return contact;
  }

  std::array<mjtNum, 3> SiteGravity() const {
    const mjtNum* xmat = data_->site_xmat + pelvis_imu_site_id_ * 9;
    return {-xmat[6], -xmat[7], -xmat[8]};
  }

  bool GetTermination() const {
    bool has_nan = false;
    for (int i = 0; i < model_->nq; ++i) {
      has_nan = has_nan || std::isnan(data_->qpos[i]);
    }
    for (int i = 0; i < model_->nv; ++i) {
      has_nan = has_nan || std::isnan(data_->qvel[i]);
    }
    return data_->sensordata[upvector_adr_ + 2] < 0.0 || has_nan;
  }

  static mjtNum Bezier(mjtNum x) { return x * x * x + 3.0 * x * x * (1 - x); }

  mjtNum FootPhaseTarget(mjtNum phase) const {
    const mjtNum x = (phase + M_PI) / (2.0 * M_PI);
    const mjtNum height = spec_.config["max_foot_height"_];
    if (x <= 0.5) {
      return height * Bezier(2.0 * x);
    }
    return height + (0.0 - height) * Bezier(2.0 * x - 1.0);
  }

  void UpdateObs(const std::array<bool, kT1Feet>& contact, bool add_noise) {
    std::array<mjtNum, 3> gravity = SiteGravity();
    std::array<mjtNum, kT1ObsDim> obs{};
    int index = 0;
    for (int i = 0; i < 3; ++i) {
      const mjtNum value = data_->sensordata[local_linvel_adr_ + i];
      obs[index++] =
          value + (add_noise ? Noise(spec_.config["noise_linvel"_]) : 0.0);
    }
    for (int i = 0; i < 3; ++i) {
      const mjtNum value = data_->sensordata[gyro_adr_ + i];
      obs[index++] =
          value + (add_noise ? Noise(spec_.config["noise_gyro"_]) : 0.0);
    }
    for (int i = 0; i < 3; ++i) {
      obs[index++] = gravity[i] +
                     (add_noise ? Noise(spec_.config["noise_gravity"_]) : 0.0);
    }
    for (int i = 0; i < 3; ++i) {
      obs[index++] = command_[i];
    }
    for (int i = 0; i < kT1ActionDim; ++i) {
      const mjtNum value = data_->qpos[7 + i];
      obs[index++] = value - default_pose_[i] +
                     (add_noise ? Noise(qpos_noise_scale_[i]) : 0.0);
    }
    for (int i = 0; i < kT1ActionDim; ++i) {
      const mjtNum value = data_->qvel[6 + i];
      obs[index++] =
          value + (add_noise ? Noise(spec_.config["noise_joint_vel"_]) : 0.0);
    }
    for (int i = 0; i < kT1ActionDim; ++i) {
      obs[index++] = last_act_[i];
    }
    obs[index++] = std::cos(phase_[0]);
    obs[index++] = std::cos(phase_[1]);
    obs[index++] = std::sin(phase_[0]);
    obs[index++] = std::sin(phase_[1]);

    state_obs_ = obs;
    std::array<mjtNum, kT1PrivilegedStateDim> privileged{};
    index = 0;
    for (mjtNum value : obs) {
      privileged[index++] = value;
    }
    for (int i = 0; i < 3; ++i) {
      privileged[index++] = data_->sensordata[gyro_adr_ + i];
    }
    for (int i = 0; i < 3; ++i) {
      privileged[index++] = data_->sensordata[accelerometer_adr_ + i];
    }
    for (int i = 0; i < 3; ++i) {
      privileged[index++] = gravity[i];
    }
    for (int i = 0; i < 3; ++i) {
      privileged[index++] = data_->sensordata[local_linvel_adr_ + i];
    }
    for (int i = 0; i < 3; ++i) {
      privileged[index++] = data_->sensordata[global_angvel_adr_ + i];
    }
    for (int i = 0; i < kT1ActionDim; ++i) {
      privileged[index++] = data_->qpos[7 + i] - default_pose_[i];
    }
    for (int i = 0; i < kT1ActionDim; ++i) {
      privileged[index++] = data_->qvel[6 + i];
    }
    privileged[index++] = data_->qpos[2];
    for (int i = 0; i < kT1ActionDim; ++i) {
      privileged[index++] = data_->actuator_force[i];
    }
    for (bool value : contact) {
      privileged[index++] = value ? 1.0 : 0.0;
    }
    for (int foot = 0; foot < kT1Feet; ++foot) {
      const int adr = feet_linvel_sensor_adrs_[foot];
      for (int i = 0; i < 3; ++i) {
        privileged[index++] = data_->sensordata[adr + i];
      }
    }
    for (mjtNum value : feet_air_time_) {
      privileged[index++] = value;
    }
    privileged_obs_ = privileged;
  }

  void ComputeRewards(const mjtNum* act,
                      const std::array<bool, kT1Feet>& first_contact,
                      const std::array<bool, kT1Feet>& contact) {
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
    const mjtNum lin_z = data_->sensordata[local_linvel_adr_ + 2];
    reward_lin_vel_z_ = lin_z * lin_z * spec_.config["lin_vel_z_scale"_];
    const mjtNum ang_x = data_->sensordata[gyro_adr_ + 0];
    const mjtNum ang_y = data_->sensordata[gyro_adr_ + 1];
    reward_ang_vel_xy_ =
        (ang_x * ang_x + ang_y * ang_y) * spec_.config["ang_vel_xy_scale"_];
    const mjtNum up_x = data_->sensordata[upvector_adr_ + 0];
    const mjtNum up_y = data_->sensordata[upvector_adr_ + 1];
    reward_orientation_ =
        (up_x * up_x + up_y * up_y) * spec_.config["orientation_scale"_];
    const mjtNum base_height_delta =
        data_->qpos[2] - spec_.config["base_height_target"_];
    reward_base_height_ = base_height_delta * base_height_delta *
                          spec_.config["base_height_scale"_];

    mjtNum torque_abs = 0.0;
    mjtNum energy = 0.0;
    for (int i = 0; i < kT1ActionDim; ++i) {
      torque_abs += std::abs(data_->actuator_force[i]);
      energy +=
          std::abs(data_->qvel[6 + i]) * std::abs(data_->actuator_force[i]);
    }
    reward_torques_ = torque_abs * spec_.config["torques_scale"_];
    reward_energy_ = energy * spec_.config["energy_scale"_];
    mjtNum dof_acc = 0.0;
    mjtNum dof_vel = 0.0;
    for (int i = 0; i < kT1ActionDim; ++i) {
      dof_acc += data_->qacc[6 + i] * data_->qacc[6 + i];
      dof_vel += data_->qvel[6 + i] * data_->qvel[6 + i];
    }
    reward_dof_acc_ = dof_acc * spec_.config["dof_acc_scale"_];
    reward_dof_vel_ = dof_vel * spec_.config["dof_vel_scale"_];

    mjtNum action_rate = 0.0;
    for (int i = 0; i < kT1ActionDim; ++i) {
      const mjtNum delta = act[i] - last_act_[i];
      action_rate += delta * delta;
    }
    reward_action_rate_ = action_rate * spec_.config["action_rate_scale"_];

    mjtNum feet_clearance = 0.0;
    mjtNum feet_height = 0.0;
    mjtNum feet_phase_error = 0.0;
    for (int foot = 0; foot < kT1Feet; ++foot) {
      const int adr = feet_linvel_sensor_adrs_[foot];
      const mjtNum vx = data_->sensordata[adr + 0];
      const mjtNum vy = data_->sensordata[adr + 1];
      const mjtNum vel_norm = std::sqrt(std::sqrt(vx * vx + vy * vy));
      const mjtNum foot_z = data_->site_xpos[feet_site_ids_[foot] * 3 + 2];
      feet_clearance +=
          std::abs(foot_z - spec_.config["max_foot_height"_]) * vel_norm;
      const mjtNum height_error =
          swing_peak_[foot] / spec_.config["max_foot_height"_] - 1.0;
      feet_height +=
          height_error * height_error * (first_contact[foot] ? 1.0 : 0.0);
      const mjtNum phase_error = foot_z - FootPhaseTarget(phase_[foot]);
      feet_phase_error += phase_error * phase_error;
    }
    reward_feet_clearance_ =
        feet_clearance * spec_.config["feet_clearance_scale"_];
    reward_feet_height_ = feet_height * spec_.config["feet_height_scale"_];
    const mjtNum cmd_norm =
        std::sqrt(command_[0] * command_[0] + command_[1] * command_[1] +
                  command_[2] * command_[2]);
    reward_feet_phase_ =
        std::exp(-feet_phase_error / 0.01) * spec_.config["feet_phase_scale"_];

    const mjtNum body_vel_norm =
        std::sqrt(data_->sensordata[global_linvel_adr_ + 0] *
                      data_->sensordata[global_linvel_adr_ + 0] +
                  data_->sensordata[global_linvel_adr_ + 1] *
                      data_->sensordata[global_linvel_adr_ + 1]);
    int num_contacts = 0;
    for (bool value : contact) {
      num_contacts += value ? 1 : 0;
    }
    reward_feet_slip_ =
        body_vel_norm * num_contacts * spec_.config["feet_slip_scale"_];

    mjtNum feet_air_time = 0.0;
    for (int i = 0; i < kT1Feet; ++i) {
      const mjtNum air_time =
          std::min(feet_air_time_[i] - 0.2, static_cast<mjtNum>(0.3));
      feet_air_time += (first_contact[i] ? air_time : 0.0);
    }
    if (cmd_norm <= 0.1) {
      feet_air_time = 0.0;
    }
    reward_feet_air_time_ =
        feet_air_time * spec_.config["feet_air_time_scale"_];

    mjtNum stand_still = 0.0;
    for (int i = 0; i < kT1ActionDim; ++i) {
      stand_still += std::abs(data_->qpos[7 + i] - default_pose_[i]);
    }
    reward_stand_still_ = (cmd_norm < 0.1 ? stand_still : 0.0) *
                          spec_.config["stand_still_scale"_];
    reward_alive_ = spec_.config["alive_scale"_];
    reward_termination_ =
        (terminated_ ? 1.0 : 0.0) * spec_.config["termination_scale"_];
    const bool collision =
        data_->sensordata[model_->sensor_adr[left_foot_right_foot_sensor_id_]] >
        0.0;
    reward_collision_ =
        (collision ? 1.0 : 0.0) * spec_.config["collision_scale"_];

    mjtNum joint_dev_hip = 0.0;
    for (int i = 0; i < 4; ++i) {
      const int index = hip_indices_[i];
      joint_dev_hip += std::abs(data_->qpos[7 + index] - default_pose_[index]);
    }
    if (std::abs(command_[1]) <= 0.1) {
      joint_dev_hip = 0.0;
    }
    reward_joint_deviation_hip_ =
        joint_dev_hip * spec_.config["joint_deviation_hip_scale"_];

    mjtNum joint_dev_knee = 0.0;
    for (int i : knee_indices_) {
      joint_dev_knee += std::abs(data_->qpos[7 + i] - default_pose_[i]);
    }
    reward_joint_deviation_knee_ =
        joint_dev_knee * spec_.config["joint_deviation_knee_scale"_];

    mjtNum dof_limits = 0.0;
    mjtNum pose = 0.0;
    for (int i = 0; i < kT1ActionDim; ++i) {
      const mjtNum q = data_->qpos[7 + i];
      dof_limits += std::max(soft_lowers_[i] - q, static_cast<mjtNum>(0.0));
      dof_limits += std::max(q - soft_uppers_[i], static_cast<mjtNum>(0.0));
      const mjtNum pose_delta = q - default_pose_[i];
      pose += pose_delta * pose_delta * weights_[i];
    }
    reward_dof_pos_limits_ = dof_limits * spec_.config["dof_pos_limits_scale"_];
    reward_pose_ = pose * spec_.config["pose_scale"_];

    const mjtNum* left_pos = data_->site_xpos + feet_site_ids_[0] * 3;
    const mjtNum* right_pos = data_->site_xpos + feet_site_ids_[1] * 3;
    const mjtNum* base_xmat = data_->site_xmat + pelvis_imu_site_id_ * 9;
    const mjtNum base_yaw = std::atan2(base_xmat[3], base_xmat[0]);
    const mjtNum feet_distance =
        std::abs(std::cos(base_yaw) * (left_pos[1] - right_pos[1]) -
                 std::sin(base_yaw) * (left_pos[0] - right_pos[0]));
    const mjtNum feet_distance_cost =
        std::clamp(static_cast<mjtNum>(0.2) - feet_distance,
                   static_cast<mjtNum>(0.0), static_cast<mjtNum>(0.1));
    reward_feet_distance_ =
        feet_distance_cost * spec_.config["feet_distance_scale"_];
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
      std::copy(state_obs_.begin(), state_obs_.end(), obs);
      CommitObservation("obs:state", &obs_state, reset);
      auto obs_privileged = state["obs:privileged_state"_];
      mjtNum* privileged =
          PrepareObservation("obs:privileged_state", &obs_privileged);
      std::copy(privileged_obs_.begin(), privileged_obs_.end(), privileged);
      CommitObservation("obs:privileged_state", &obs_privileged, reset);
    }
    state["info:terminated"_] = terminated_;
    state["info:command"_].Assign(command_.data(), command_.size());
    state["info:reward_tracking_lin_vel"_] = reward_tracking_lin_vel_;
    state["info:reward_tracking_ang_vel"_] = reward_tracking_ang_vel_;
    state["info:reward_lin_vel_z"_] = reward_lin_vel_z_;
    state["info:reward_ang_vel_xy"_] = reward_ang_vel_xy_;
    state["info:reward_orientation"_] = reward_orientation_;
    state["info:reward_base_height"_] = reward_base_height_;
    state["info:reward_torques"_] = reward_torques_;
    state["info:reward_action_rate"_] = reward_action_rate_;
    state["info:reward_energy"_] = reward_energy_;
    state["info:reward_dof_acc"_] = reward_dof_acc_;
    state["info:reward_dof_vel"_] = reward_dof_vel_;
    state["info:reward_feet_clearance"_] = reward_feet_clearance_;
    state["info:reward_feet_air_time"_] = reward_feet_air_time_;
    state["info:reward_feet_slip"_] = reward_feet_slip_;
    state["info:reward_feet_height"_] = reward_feet_height_;
    state["info:reward_feet_phase"_] = reward_feet_phase_;
    state["info:reward_stand_still"_] = reward_stand_still_;
    state["info:reward_alive"_] = reward_alive_;
    state["info:reward_termination"_] = reward_termination_;
    state["info:reward_collision"_] = reward_collision_;
    state["info:reward_joint_deviation_knee"_] = reward_joint_deviation_knee_;
    state["info:reward_joint_deviation_hip"_] = reward_joint_deviation_hip_;
    state["info:reward_dof_pos_limits"_] = reward_dof_pos_limits_;
    state["info:reward_pose"_] = reward_pose_;
    state["info:reward_feet_distance"_] = reward_feet_distance_;
#ifdef ENVPOOL_TEST
    std::array<mjtNum, 64> pad{};
    std::copy(data_->qpos, data_->qpos + model_->nq, pad.begin());
    state["info:qpos"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(reset_qpos_.begin(), reset_qpos_.end(), pad.begin());
    state["info:qpos0"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(data_->qvel, data_->qvel + model_->nv, pad.begin());
    state["info:qvel"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(reset_qvel_.begin(), reset_qvel_.end(), pad.begin());
    state["info:qvel0"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(data_->ctrl, data_->ctrl + model_->nu, pad.begin());
    state["info:ctrl"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(reset_ctrl_.begin(), reset_ctrl_.end(), pad.begin());
    state["info:ctrl0"_].Assign(pad.data(), pad.size());
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
    state["info:last_act"_].Assign(last_act_.data(), last_act_.size());
    state["info:last_last_act"_].Assign(last_last_act_.data(),
                                        last_last_act_.size());
    state["info:motor_targets"_].Assign(motor_targets_.data(),
                                        motor_targets_.size());
    state["info:actuator_force"_].Assign(data_->actuator_force, kT1ActionDim);
    state["info:feet_air_time"_].Assign(feet_air_time_.data(),
                                        feet_air_time_.size());
    state["info:last_contact"_].Assign(last_contact_.data(),
                                       last_contact_.size());
    state["info:swing_peak"_].Assign(swing_peak_.data(), swing_peak_.size());
    state["info:phase"_].Assign(phase_.data(), phase_.size());
    state["info:phase_dt"_].Assign(phase_dt_.data(), phase_dt_.size());
    state["info:push"_].Assign(push_.data(), push_.size());
    state["info:push_step"_] = push_step_;
    state["info:push_interval_steps"_] = push_interval_steps_;
    state["info:step"_] = step_;
    std::array<mjtNum, 256> site_pad{};
    std::copy(data_->site_xpos, data_->site_xpos + model_->nsite * 3,
              site_pad.begin());
    state["info:site_xpos"_].Assign(site_pad.data(), site_pad.size());
    site_pad.fill(0.0);
    std::copy(data_->site_xmat, data_->site_xmat + model_->nsite * 9,
              site_pad.begin());
    state["info:site_xmat"_].Assign(site_pad.data(), site_pad.size());
    std::array<mjtNum, 256> sensor_pad{};
    std::copy(data_->sensordata, data_->sensordata + model_->nsensordata,
              sensor_pad.begin());
    state["info:sensordata"_].Assign(sensor_pad.data(), sensor_pad.size());
#endif
  }

  std::array<mjtNum, kT1ObsDim> state_obs_{};
  std::array<mjtNum, kT1PrivilegedStateDim> privileged_obs_{};
};

template <typename Spec, bool kFromPixels>
using T1Base = PlaygroundT1EnvBase<Spec, kFromPixels>;
using T1Env = T1Base<PlaygroundT1EnvSpec, false>;
using T1PixelEnv = T1Base<PlaygroundT1PixelEnvSpec, true>;
using PlaygroundT1Env = T1Env;
using PlaygroundT1PixelEnv = T1PixelEnv;
using PlaygroundT1EnvPool = PlaygroundEnvPoolT<PlaygroundT1Env>;
using PlaygroundT1PixelEnvPool = PlaygroundEnvPoolT<PlaygroundT1PixelEnv>;

}  // namespace mujoco_playground

#endif  // ENVPOOL_MUJOCO_PLAYGROUND_T1_H_
