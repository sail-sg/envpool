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

#ifndef ENVPOOL_MUJOCO_PLAYGROUND_SPOT_H_
#define ENVPOOL_MUJOCO_PLAYGROUND_SPOT_H_

#include <mujoco.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
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

constexpr int kSpotActionDim = 12;
constexpr int kSpotFeet = 4;
constexpr int kSpotHistoryLen = 3;
constexpr int kSpotJoystickStateDim = 81;
constexpr int kSpotJoystickPrivilegedStateDim = 167;
constexpr int kSpotGetupStateDim = 30;
constexpr int kSpotGaitStateDim = 69;
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
constexpr const char* kSpotFeetSites[kSpotFeet] = {"FL", "FR", "HL", "HR"};
inline const std::array<const char*, kSpotFeet>& SpotFootLinvelSensors() {
  static constexpr std::array<const char*, kSpotFeet> k_names = {
      "FL_global_linvel", "FR_global_linvel", "HL_global_linvel",
      "HR_global_linvel"};
  return k_names;
}

inline const std::array<const char*, kSpotFeet>& SpotFootPosSensors() {
  static constexpr std::array<const char*, kSpotFeet> k_names = {
      "FL_pos", "FR_pos", "HL_pos", "HR_pos"};
  return k_names;
}
constexpr std::array<std::array<mjtNum, kSpotFeet>, 5> kSpotGaitPhases = {{
    {0.0, M_PI, M_PI, 0.0},
    {0.0, 0.5 * M_PI, M_PI, 1.5 * M_PI},
    {0.0, M_PI, 0.0, M_PI},
    {0.0, 0.0, M_PI, M_PI},
    {0.0, 0.0, 0.0, 0.0},
}};

inline decltype(auto) SpotDefaultConfig(const std::string& task_name) {
  return MakeDict(
      "frame_stack"_.Bind(1), "task_name"_.Bind(task_name),
      "ctrl_dt"_.Bind(0.02), "sim_dt"_.Bind(0.004), "kp"_.Bind(300.0),
      "kd"_.Bind(1.0), "early_termination"_.Bind(true),
      "action_scale"_.Bind(0.3), "history_len"_.Bind(kSpotHistoryLen),
      "noise_level"_.Bind(1.0), "noise_joint_pos"_.Bind(0.05),
      "noise_gyro"_.Bind(0.1), "noise_gravity"_.Bind(0.03),
      "noise_feet_pos_x"_.Bind(0.01), "noise_feet_pos_y"_.Bind(0.005),
      "noise_feet_pos_z"_.Bind(0.02), "drop_from_height_prob"_.Bind(0.6),
      "settle_time"_.Bind(0.5), "tracking_lin_vel_scale"_.Bind(1.5),
      "tracking_ang_vel_scale"_.Bind(0.8), "lin_vel_z_scale"_.Bind(-2.0),
      "ang_vel_xy_scale"_.Bind(-0.05), "orientation_scale"_.Bind(-5.0),
      "termination_scale"_.Bind(-1.0), "posture_scale"_.Bind(1.0),
      "torques_scale"_.Bind(-0.0002), "action_rate_scale"_.Bind(-0.01),
      "energy_scale"_.Bind(-0.001), "feet_slip_scale"_.Bind(-0.1),
      "feet_clearance_scale"_.Bind(-2.0), "feet_height_scale"_.Bind(-0.1),
      "feet_air_time_scale"_.Bind(0.1), "feet_phase_scale"_.Bind(0.0),
      "hip_splay_scale"_.Bind(0.0), "torso_height_scale"_.Bind(0.0),
      "stand_still_scale"_.Bind(0.0), "tracking_sigma"_.Bind(0.25),
      "max_foot_height"_.Bind(0.12), "pert_enable"_.Bind(false),
      "kick_wait_time_min"_.Bind(1.0), "kick_wait_time_max"_.Bind(3.0),
      "kick_duration_min"_.Bind(0.05), "kick_duration_max"_.Bind(0.2),
      "kick_velocity_min"_.Bind(0.0), "kick_velocity_max"_.Bind(3.0),
      "lin_vel_x_min"_.Bind(-1.0), "lin_vel_x_max"_.Bind(1.0),
      "lin_vel_y_min"_.Bind(-0.8), "lin_vel_y_max"_.Bind(0.8),
      "ang_vel_yaw_min"_.Bind(-1.0), "ang_vel_yaw_max"_.Bind(1.0),
      "gait_frequency_min"_.Bind(0.5), "gait_frequency_max"_.Bind(4.0),
      "gait_count"_.Bind(5), "foot_height_min"_.Bind(0.08),
      "foot_height_max"_.Bind(0.4));
}

class PlaygroundSpotJoystickEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return SpotDefaultConfig("SpotFlatTerrainJoystick");
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs:state"_.Bind(
            StackSpec(Spec<mjtNum>({kSpotJoystickStateDim}, {-inf, inf}),
                      conf["frame_stack"_])),
        "obs:privileged_state"_.Bind(StackSpec(
            Spec<mjtNum>({kSpotJoystickPrivilegedStateDim}, {-inf, inf}),
            conf["frame_stack"_])),
        "info:terminated"_.Bind(Spec<bool>({-1})),
        "info:command"_.Bind(Spec<mjtNum>({-1, 3})),
        "info:reward_tracking_lin_vel"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_tracking_ang_vel"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_feet_phase"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_lin_vel_z"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_ang_vel_xy"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_hip_splay"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_orientation"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_termination"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_posture"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_torques"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_action_rate"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_energy"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_feet_slip"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_feet_clearance"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_feet_height"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_feet_air_time"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_torso_height"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_stand_still"_.Bind(Spec<mjtNum>({-1}))
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
        "info:actuator_force"_.Bind(Spec<mjtNum>({kSpotActionDim})),
        "info:sensordata"_.Bind(Spec<mjtNum>({256})),
        "info:last_act"_.Bind(Spec<mjtNum>({kSpotActionDim})),
        "info:last_last_act"_.Bind(Spec<mjtNum>({kSpotActionDim})),
        "info:motor_targets"_.Bind(Spec<mjtNum>({kSpotActionDim})),
        "info:qpos_error_history"_.Bind(
            Spec<mjtNum>({kSpotHistoryLen * kSpotActionDim})),
        "info:swing_peak"_.Bind(Spec<mjtNum>({kSpotFeet})),
        "info:feet_air_time"_.Bind(Spec<mjtNum>({kSpotFeet})),
        "info:last_contact"_.Bind(Spec<bool>({kSpotFeet})),
        "info:gait_freq"_.Bind(Spec<mjtNum>({-1})),
        "info:gait"_.Bind(Spec<int>({-1})),
        "info:phase"_.Bind(Spec<mjtNum>({kSpotFeet})),
        "info:phase_dt"_.Bind(Spec<mjtNum>({-1})),
        "info:foot_height"_.Bind(Spec<mjtNum>({-1})),
        "info:step"_.Bind(Spec<int>({-1}))
#endif
    );  // NOLINT
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<mjtNum>({-1, kSpotActionDim}, {-1.0, 1.0})));
  }
};

class PlaygroundSpotGetupEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return SpotDefaultConfig("SpotGetup");
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict("obs:state"_.Bind(StackSpec(
                        Spec<mjtNum>({kSpotGetupStateDim}, {-inf, inf}),
                        conf["frame_stack"_])),
                    "info:terminated"_.Bind(Spec<bool>({-1})),
                    "info:command"_.Bind(Spec<mjtNum>({-1, 3})),
                    "info:reward_orientation"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_torso_height"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_posture"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_stand_still"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_torques"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_action_rate"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_tracking_lin_vel"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_tracking_ang_vel"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_feet_phase"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_lin_vel_z"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_ang_vel_xy"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_hip_splay"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_termination"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_energy"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_feet_slip"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_feet_clearance"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_feet_height"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_feet_air_time"_.Bind(Spec<mjtNum>({-1}))
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
                    "info:actuator_force"_.Bind(Spec<mjtNum>({kSpotActionDim})),
                    "info:sensordata"_.Bind(Spec<mjtNum>({256})),
                    "info:last_act"_.Bind(Spec<mjtNum>({kSpotActionDim})),
                    "info:last_last_act"_.Bind(Spec<mjtNum>({kSpotActionDim})),
                    "info:motor_targets"_.Bind(Spec<mjtNum>({kSpotActionDim})),
                    "info:qpos_error_history"_.Bind(
                        Spec<mjtNum>({kSpotHistoryLen * kSpotActionDim})),
                    "info:swing_peak"_.Bind(Spec<mjtNum>({kSpotFeet})),
                    "info:feet_air_time"_.Bind(Spec<mjtNum>({kSpotFeet})),
                    "info:last_contact"_.Bind(Spec<bool>({kSpotFeet})),
                    "info:gait_freq"_.Bind(Spec<mjtNum>({-1})),
                    "info:gait"_.Bind(Spec<int>({-1})),
                    "info:phase"_.Bind(Spec<mjtNum>({kSpotFeet})),
                    "info:phase_dt"_.Bind(Spec<mjtNum>({-1})),
                    "info:foot_height"_.Bind(Spec<mjtNum>({-1})),
                    "info:step"_.Bind(Spec<int>({-1}))
#endif
    );  // NOLINT
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return PlaygroundSpotJoystickEnvFns::ActionSpec(conf);
  }
};

class PlaygroundSpotGaitEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return SpotDefaultConfig("SpotJoystickGaitTracking");
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict("obs:state"_.Bind(StackSpec(
                        Spec<mjtNum>({kSpotGaitStateDim}, {-inf, inf}),
                        conf["frame_stack"_])),
                    "info:terminated"_.Bind(Spec<bool>({-1})),
                    "info:command"_.Bind(Spec<mjtNum>({-1, 3})),
                    "info:reward_tracking_lin_vel"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_tracking_ang_vel"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_feet_phase"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_lin_vel_z"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_ang_vel_xy"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_hip_splay"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_orientation"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_torso_height"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_posture"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_stand_still"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_torques"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_action_rate"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_termination"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_energy"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_feet_slip"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_feet_clearance"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_feet_height"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_feet_air_time"_.Bind(Spec<mjtNum>({-1}))
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
                    "info:actuator_force"_.Bind(Spec<mjtNum>({kSpotActionDim})),
                    "info:sensordata"_.Bind(Spec<mjtNum>({256})),
                    "info:last_act"_.Bind(Spec<mjtNum>({kSpotActionDim})),
                    "info:last_last_act"_.Bind(Spec<mjtNum>({kSpotActionDim})),
                    "info:motor_targets"_.Bind(Spec<mjtNum>({kSpotActionDim})),
                    "info:qpos_error_history"_.Bind(
                        Spec<mjtNum>({kSpotHistoryLen * kSpotActionDim})),
                    "info:swing_peak"_.Bind(Spec<mjtNum>({kSpotFeet})),
                    "info:feet_air_time"_.Bind(Spec<mjtNum>({kSpotFeet})),
                    "info:last_contact"_.Bind(Spec<bool>({kSpotFeet})),
                    "info:gait_freq"_.Bind(Spec<mjtNum>({-1})),
                    "info:gait"_.Bind(Spec<int>({-1})),
                    "info:phase"_.Bind(Spec<mjtNum>({kSpotFeet})),
                    "info:phase_dt"_.Bind(Spec<mjtNum>({-1})),
                    "info:foot_height"_.Bind(Spec<mjtNum>({-1})),
                    "info:step"_.Bind(Spec<int>({-1}))
#endif
    );  // NOLINT
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return PlaygroundSpotJoystickEnvFns::ActionSpec(conf);
  }
};

using SpotJoystickAliases = PlaygroundEnvAliases<PlaygroundSpotJoystickEnvFns>;
using SpotGetupAliases = PlaygroundEnvAliases<PlaygroundSpotGetupEnvFns>;
using SpotGaitAliases = PlaygroundEnvAliases<PlaygroundSpotGaitEnvFns>;
using PlaygroundSpotJoystickEnvSpec = SpotJoystickAliases::Spec;
using PlaygroundSpotJoystickPixelEnvFns = SpotJoystickAliases::PixelFns;
using PlaygroundSpotJoystickPixelEnvSpec = SpotJoystickAliases::PixelSpec;
using PlaygroundSpotGetupEnvSpec = SpotGetupAliases::Spec;
using PlaygroundSpotGetupPixelEnvFns = SpotGetupAliases::PixelFns;
using PlaygroundSpotGetupPixelEnvSpec = SpotGetupAliases::PixelSpec;
using PlaygroundSpotGaitEnvSpec = SpotGaitAliases::Spec;
using PlaygroundSpotGaitPixelEnvFns = SpotGaitAliases::PixelFns;
using PlaygroundSpotGaitPixelEnvSpec = SpotGaitAliases::PixelSpec;

template <typename EnvSpecT, bool kFromPixels>
class PlaygroundSpotEnvBase : public Env<EnvSpecT>, public PlaygroundMujocoEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  int n_substeps_{5};
  bool is_getup_{false};
  bool is_gait_{false};
  int state_dim_{kSpotJoystickStateDim};
  std::vector<mjtNum> init_qpos_;
  std::array<mjtNum, kSpotActionDim> default_pose_{};
  std::array<mjtNum, kSpotActionDim> lowers_{};
  std::array<mjtNum, kSpotActionDim> uppers_{};
  std::array<mjtNum, kSpotActionDim> last_act_{};
  std::array<mjtNum, kSpotActionDim> last_last_act_{};
  std::array<mjtNum, kSpotActionDim> motor_targets_{};
  std::array<mjtNum, 3> command_{};
  std::array<mjtNum, kSpotHistoryLen * kSpotActionDim> qpos_error_history_{};
  std::array<mjtNum, kSpotFeet> feet_air_time_{};
  std::array<mjtNum, kSpotFeet> swing_peak_{};
  std::array<bool, kSpotFeet> last_contact_{};
  std::array<mjtNum, kSpotFeet> phase_{};
  mjtNum phase_dt_{0.0};
  mjtNum gait_freq_{0.0};
  int gait_{0};
  mjtNum foot_height_{0.0};
  int step_{0};
  int torso_body_id_{-1};
  int imu_site_id_{-1};
  std::array<int, kSpotFeet> feet_site_ids_{};
  std::array<int, kSpotFeet> feet_floor_sensor_adrs_{};
  std::array<int, kSpotFeet> feet_linvel_sensor_adrs_{};
  std::array<int, kSpotFeet> feet_pos_sensor_adrs_{};
  int gyro_adr_{-1};
  int local_linvel_adr_{-1};
  int accelerometer_adr_{-1};
  int upvector_adr_{-1};
  int global_linvel_adr_{-1};
  int global_angvel_adr_{-1};
  mjtNum z_des_{0.0};
  std::array<mjtNum, 3> up_vec_{0.0, 0.0, 1.0};
  std::array<mjtNum, 4> hx_default_pose_{};
  std::array<mjtNum, kSpotJoystickStateDim> joystick_obs_{};
  std::array<mjtNum, kSpotJoystickPrivilegedStateDim> privileged_obs_{};
  std::array<mjtNum, kSpotGetupStateDim> getup_obs_{};
  std::array<mjtNum, kSpotGaitStateDim> gait_obs_{};
  mjtNum reward_tracking_lin_vel_{0.0};
  mjtNum reward_tracking_ang_vel_{0.0};
  mjtNum reward_lin_vel_z_{0.0};
  mjtNum reward_ang_vel_xy_{0.0};
  mjtNum reward_orientation_{0.0};
  mjtNum reward_termination_{0.0};
  mjtNum reward_posture_{0.0};
  mjtNum reward_torques_{0.0};
  mjtNum reward_action_rate_{0.0};
  mjtNum reward_energy_{0.0};
  mjtNum reward_feet_slip_{0.0};
  mjtNum reward_feet_clearance_{0.0};
  mjtNum reward_feet_height_{0.0};
  mjtNum reward_feet_air_time_{0.0};
  mjtNum reward_feet_phase_{0.0};
  mjtNum reward_hip_splay_{0.0};
  mjtNum reward_torso_height_{0.0};
  mjtNum reward_stand_still_{0.0};
  std::uniform_real_distribution<mjtNum> unit_uniform_{0.0, 1.0};
  std::uniform_real_distribution<mjtNum> noise_uniform_{-1.0, 1.0};
  std::normal_distribution<mjtNum> normal_{0.0, 1.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  PlaygroundSpotEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        PlaygroundMujocoEnv(
            SpotXmlPath(spec.config["base_path"_], spec.config["task_name"_]),
            spec.config["max_episode_steps"_], spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)) {
    (void)env_id;
    const std::string task_name = spec.config["task_name"_];
    is_getup_ = task_name == "SpotGetup";
    is_gait_ = task_name == "SpotJoystickGaitTracking";
    if (!is_getup_ && !is_gait_ && task_name != "SpotFlatTerrainJoystick") {
      throw std::runtime_error("Unsupported Spot task_name " + task_name);
    }
    if (is_getup_) {
      state_dim_ = kSpotGetupStateDim;
    } else if (is_gait_) {
      state_dim_ = kSpotGaitStateDim;
    } else {
      state_dim_ = kSpotJoystickStateDim;
    }
    if (model_->nq < 7 + kSpotActionDim || model_->nv < 6 + kSpotActionDim ||
        model_->nu != kSpotActionDim) {
      throw std::runtime_error("Unexpected Spot model dimensions.");
    }
    model_->opt.timestep = spec.config["sim_dt"_];
    for (int i = 6; i < model_->nv; ++i) {
      model_->dof_damping[i] = spec.config["kd"_];
    }
    for (int i = 0; i < model_->nu; ++i) {
      model_->actuator_gainprm[i * mjNGAIN + 0] = spec.config["kp"_];
      model_->actuator_biasprm[i * mjNBIAS + 1] = -spec.config["kp"_];
    }
    n_substeps_ = static_cast<int>(
        std::llround(spec.config["ctrl_dt"_] / spec.config["sim_dt"_]));

    const int home_key = RequireId(mjOBJ_KEY, "home");
    init_qpos_.assign(model_->key_qpos + home_key * model_->nq,
                      model_->key_qpos + (home_key + 1) * model_->nq);
    std::copy(init_qpos_.begin() + 7, init_qpos_.begin() + 7 + kSpotActionDim,
              default_pose_.begin());
    for (int i = 0; i < kSpotActionDim; ++i) {
      lowers_[i] = model_->actuator_ctrlrange[i * 2];
      uppers_[i] = model_->actuator_ctrlrange[i * 2 + 1];
    }
    z_des_ = init_qpos_[2];
    torso_body_id_ = RequireId(mjOBJ_BODY, "body");
    imu_site_id_ = RequireId(mjOBJ_SITE, "imu");
    gyro_adr_ = SensorAdr("gyro");
    local_linvel_adr_ = SensorAdr("local_linvel");
    accelerometer_adr_ = SensorAdr("accelerometer");
    upvector_adr_ = SensorAdr("upvector");
    global_linvel_adr_ = SensorAdr("global_linvel");
    global_angvel_adr_ = SensorAdr("global_angvel");
    for (int i = 0; i < kSpotFeet; ++i) {
      feet_site_ids_[i] = RequireId(mjOBJ_SITE, kSpotFeetSites[i]);
      feet_floor_sensor_adrs_[i] =
          SensorAdr(std::string(kSpotFeetSites[i]) + "_floor_found");
      feet_linvel_sensor_adrs_[i] = SensorAdr(SpotFootLinvelSensors()[i]);
      if (!is_getup_) {
        feet_pos_sensor_adrs_[i] = SensorAdr(SpotFootPosSensors()[i]);
      }
    }
    hx_default_pose_ = {default_pose_[0], default_pose_[3], default_pose_[6],
                        default_pose_[9]};
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
    std::fill(feet_air_time_.begin(), feet_air_time_.end(), 0.0);
    std::fill(swing_peak_.begin(), swing_peak_.end(), 0.0);
    std::fill(last_contact_.begin(), last_contact_.end(), false);

    mj_resetData(model_, data_);
    if (is_getup_) {
      ResetGetupQpos();
    } else {
      std::copy(init_qpos_.begin(), init_qpos_.end(), data_->qpos);
    }
    std::fill(data_->qvel, data_->qvel + model_->nv, 0.0);
    std::fill(data_->ctrl, data_->ctrl + model_->nu, 0.0);
    std::fill(data_->xfrc_applied, data_->xfrc_applied + model_->nbody * 6,
              0.0);
    mj_forward(model_, data_);
    if (is_getup_) {
      const int settle_steps = static_cast<int>(
          std::llround(spec_.config["settle_time"_] / spec_.config["sim_dt"_]));
      for (int i = 0; i < kSpotActionDim; ++i) {
        data_->ctrl[i] = data_->qpos[7 + i];
      }
      for (int i = 0; i < settle_steps; ++i) {
        mj_step(model_, data_);
      }
      data_->time = 0.0;
      UpdateGetupObs(/*add_noise=*/true);
    } else if (is_gait_) {
      SampleCommand();
      SampleGait();
      const auto contact = FootContact();
      UpdateGaitObs(contact, /*add_noise=*/true);
    } else {
      SampleCommand();
      const auto contact = FootContact();
      UpdateJoystickObs(contact, /*add_noise=*/true);
    }
    WriteState(0.0, true);
  }

  void Step(const Action& action) override {
    has_cached_render_ = false;
    const auto* act = static_cast<const mjtNum*>(action["action"_].Data());
    ResetRewards();
    if (is_getup_) {
      StepGetup(act);
    } else if (is_gait_) {
      StepGait(act);
    } else {
      StepJoystick(act);
    }
  }

 private:
  static std::string SpotXmlPath(const std::string& base_path,
                                 const std::string& task_name) {
    const bool getup = task_name == "SpotGetup";
    return base_path +
           "/mujoco/playground/assets/mujoco_playground/_src/locomotion/"
           "spot/xmls/" +
           std::string(getup ? "scene_mjx_flat_terrain.xml"
                             : "scene_mjx_feetonly_flat_terrain.xml");
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

  mjtNum Noise(mjtNum scale) {
    return spec_.config["noise_level"_] * scale * noise_uniform_(gen_);
  }

  void ResetGetupQpos() {
    if (Uniform(0.0, 1.0) >= spec_.config["drop_from_height_prob"_]) {
      std::copy(init_qpos_.begin(), init_qpos_.end(), data_->qpos);
      return;
    }
    std::fill(data_->qpos, data_->qpos + model_->nq, 0.0);
    data_->qpos[2] = 1.0;
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum quat[4] = {normal_(gen_), normal_(gen_), normal_(gen_),
                      normal_(gen_)};
    const mjtNum norm = std::sqrt(quat[0] * quat[0] + quat[1] * quat[1] +
                                  quat[2] * quat[2] + quat[3] * quat[3]) +
                        1e-6;
    for (int i = 0; i < 4; ++i) {
      data_->qpos[3 + i] = quat[i] / norm;
    }
    for (int i = 0; i < kSpotActionDim; ++i) {
      data_->qpos[7 + i] = Uniform(lowers_[i], uppers_[i]);
    }
  }

  void SampleCommand() {
    command_[0] =
        Uniform(spec_.config["lin_vel_x_min"_], spec_.config["lin_vel_x_max"_]);
    command_[1] =
        Uniform(spec_.config["lin_vel_y_min"_], spec_.config["lin_vel_y_max"_]);
    command_[2] = Uniform(spec_.config["ang_vel_yaw_min"_],
                          spec_.config["ang_vel_yaw_max"_]);
    if (Uniform(0.0, 1.0) < 0.1) {
      command_ = {0.0, 0.0, 0.0};
    }
  }

  void SampleGait() {
    gait_freq_ = Uniform(spec_.config["gait_frequency_min"_],
                         spec_.config["gait_frequency_max"_]);
    phase_dt_ = 2.0 * M_PI * Dt() * gait_freq_;
    gait_ = static_cast<int>(std::floor(
        Uniform(0.0, static_cast<mjtNum>(spec_.config["gait_count"_]))));
    if (gait_ >= spec_.config["gait_count"_]) {
      gait_ = spec_.config["gait_count"_] - 1;
    }
    phase_ = kSpotGaitPhases[gait_];
    foot_height_ = Uniform(spec_.config["foot_height_min"_],
                           spec_.config["foot_height_max"_]);
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

  std::array<bool, kSpotFeet> FootContact() const {
    std::array<bool, kSpotFeet> contact{};
    for (int i = 0; i < kSpotFeet; ++i) {
      contact[i] = data_->sensordata[feet_floor_sensor_adrs_[i]] > 0.0;
    }
    return contact;
  }

  void RollHistory() {
    for (int i = kSpotHistoryLen * kSpotActionDim - 1; i >= kSpotActionDim;
         --i) {
      qpos_error_history_[i] = qpos_error_history_[i - kSpotActionDim];
    }
  }

  void UpdateJoystickObs(const std::array<bool, kSpotFeet>& contact,
                         bool add_noise) {
    std::array<mjtNum, kSpotActionDim> noisy_joint_angles{};
    std::array<mjtNum, kSpotFeet * 3> feet_pos{};
    std::array<mjtNum, kSpotFeet * 3> noisy_feet_pos{};
    for (int i = 0; i < kSpotActionDim; ++i) {
      noisy_joint_angles[i] =
          data_->qpos[7 + i] +
          (add_noise ? Noise(spec_.config["noise_joint_pos"_]) : 0.0);
    }
    for (int foot = 0; foot < kSpotFeet; ++foot) {
      const int adr = feet_pos_sensor_adrs_[foot];
      // NOLINTNEXTLINE(modernize-avoid-c-arrays)
      const mjtNum scales[3] = {spec_.config["noise_feet_pos_x"_],
                                spec_.config["noise_feet_pos_y"_],
                                spec_.config["noise_feet_pos_z"_]};
      for (int i = 0; i < 3; ++i) {
        const int index = foot * 3 + i;
        feet_pos[index] = data_->sensordata[adr + i];
        noisy_feet_pos[index] =
            feet_pos[index] + (add_noise ? Noise(scales[i]) : 0.0);
      }
    }
    RollHistory();
    for (int i = 0; i < kSpotActionDim; ++i) {
      qpos_error_history_[i] = noisy_joint_angles[i] - motor_targets_[i];
    }

    int out = 0;
    for (int i = 0; i < 3; ++i) {
      joystick_obs_[out++] =
          data_->sensordata[gyro_adr_ + i] +
          (add_noise ? Noise(spec_.config["noise_gyro"_]) : 0.0);
    }
    for (int i = 0; i < 3; ++i) {
      joystick_obs_[out++] =
          data_->sensordata[upvector_adr_ + i] +
          (add_noise ? Noise(spec_.config["noise_gravity"_]) : 0.0);
    }
    for (int i = 0; i < kSpotActionDim; ++i) {
      joystick_obs_[out++] = noisy_joint_angles[i] - default_pose_[i];
    }
    for (mjtNum value : qpos_error_history_) {
      joystick_obs_[out++] = value;
    }
    for (mjtNum value : noisy_feet_pos) {
      joystick_obs_[out++] = value;
    }
    for (mjtNum value : last_act_) {
      joystick_obs_[out++] = value;
    }
    for (mjtNum value : command_) {
      joystick_obs_[out++] = value;
    }

    out = 0;
    for (mjtNum value : joystick_obs_) {
      privileged_obs_[out++] = value;
    }
    for (int i = 0; i < 3; ++i) {
      privileged_obs_[out++] = data_->sensordata[gyro_adr_ + i];
    }
    for (int i = 0; i < 3; ++i) {
      privileged_obs_[out++] = data_->sensordata[accelerometer_adr_ + i];
    }
    for (int i = 0; i < 3; ++i) {
      privileged_obs_[out++] = data_->sensordata[upvector_adr_ + i];
    }
    for (int i = 0; i < 3; ++i) {
      privileged_obs_[out++] = data_->sensordata[local_linvel_adr_ + i];
    }
    for (int i = 0; i < 3; ++i) {
      privileged_obs_[out++] = data_->sensordata[global_angvel_adr_ + i];
    }
    for (int i = 0; i < kSpotActionDim; ++i) {
      privileged_obs_[out++] = data_->qpos[7 + i] - default_pose_[i];
    }
    for (mjtNum value : feet_pos) {
      privileged_obs_[out++] = value;
    }
    for (int i = 0; i < kSpotActionDim; ++i) {
      privileged_obs_[out++] = data_->qvel[6 + i];
    }
    for (int i = 0; i < kSpotActionDim; ++i) {
      privileged_obs_[out++] = data_->actuator_force[i];
    }
    for (bool value : last_contact_) {
      privileged_obs_[out++] = value ? 1.0 : 0.0;
    }
    for (int foot = 0; foot < kSpotFeet; ++foot) {
      const int adr = feet_linvel_sensor_adrs_[foot];
      for (int i = 0; i < 3; ++i) {
        privileged_obs_[out++] = data_->sensordata[adr + i];
      }
    }
    for (mjtNum value : feet_air_time_) {
      privileged_obs_[out++] = value;
    }
    for (int i = 0; i < 3; ++i) {
      privileged_obs_[out++] = data_->xfrc_applied[torso_body_id_ * 6 + i];
    }
  }

  void UpdateGaitObs(const std::array<bool, kSpotFeet>& contact,
                     bool add_noise) {
    std::array<mjtNum, kSpotActionDim> noisy_joint_angles{};
    for (int i = 0; i < kSpotActionDim; ++i) {
      noisy_joint_angles[i] =
          data_->qpos[7 + i] +
          (add_noise ? Noise(spec_.config["noise_joint_pos"_]) : 0.0);
    }
    RollHistory();
    for (int i = 0; i < kSpotActionDim; ++i) {
      qpos_error_history_[i] = noisy_joint_angles[i] - motor_targets_[i];
    }
    int out = 0;
    for (int i = 0; i < 3; ++i) {
      gait_obs_[out++] = data_->sensordata[gyro_adr_ + i] +
                         (add_noise ? Noise(spec_.config["noise_gyro"_]) : 0.0);
    }
    for (int i = 0; i < 3; ++i) {
      gait_obs_[out++] =
          data_->sensordata[upvector_adr_ + i] +
          (add_noise ? Noise(spec_.config["noise_gravity"_]) : 0.0);
    }
    for (int i = 0; i < kSpotActionDim; ++i) {
      gait_obs_[out++] = noisy_joint_angles[i];
    }
    for (mjtNum value : qpos_error_history_) {
      gait_obs_[out++] = value;
    }
    for (bool value : contact) {
      gait_obs_[out++] = value ? 1.0 : 0.0;
    }
    for (mjtNum value : phase_) {
      gait_obs_[out++] = std::cos(value);
    }
    for (mjtNum value : phase_) {
      gait_obs_[out++] = std::sin(value);
    }
    gait_obs_[out++] = gait_freq_;
    gait_obs_[out++] = static_cast<mjtNum>(gait_);
    gait_obs_[out++] = foot_height_;
  }

  void UpdateGetupObs(bool add_noise) {
    int out = 0;
    for (int i = 0; i < 3; ++i) {
      getup_obs_[out++] =
          data_->sensordata[gyro_adr_ + i] +
          (add_noise ? Noise(spec_.config["noise_gyro"_]) : 0.0);
    }
    for (int i = 0; i < 3; ++i) {
      getup_obs_[out++] =
          data_->sensordata[upvector_adr_ + i] +
          (add_noise ? Noise(spec_.config["noise_gravity"_]) : 0.0);
    }
    for (int i = 0; i < kSpotActionDim; ++i) {
      getup_obs_[out++] =
          data_->qpos[7 + i] - default_pose_[i] +
          (add_noise ? Noise(spec_.config["noise_joint_pos"_]) : 0.0);
    }
    for (mjtNum value : last_act_) {
      getup_obs_[out++] = value;
    }
  }

  bool JoystickTermination() const {
    if (!spec_.config["early_termination"_]) {
      return false;
    }
    return data_->sensordata[upvector_adr_ + 2] < 0.85;
  }

  bool JointLimitTermination() const {
    bool done = false;
    for (int i = 0; i < kSpotActionDim; ++i) {
      const mjtNum q = data_->qpos[7 + i];
      done = done || q < lowers_[i] || q > uppers_[i];
    }
    return done;
  }

  void StepJoystick(const mjtNum* act) {
    for (int i = 0; i < kSpotActionDim; ++i) {
      const mjtNum target =
          default_pose_[i] + act[i] * spec_.config["action_scale"_];
      motor_targets_[i] = std::clamp(target, lowers_[i], uppers_[i]);
      data_->ctrl[i] = motor_targets_[i];
    }
    for (int i = 0; i < n_substeps_; ++i) {
      mj_step(model_, data_);
    }
    const auto contact = FootContact();
    std::array<bool, kSpotFeet> first_contact{};
    for (int i = 0; i < kSpotFeet; ++i) {
      first_contact[i] =
          feet_air_time_[i] > 0.0 && (contact[i] || last_contact_[i]);
      feet_air_time_[i] += Dt();
      const mjtNum z = data_->site_xpos[feet_site_ids_[i] * 3 + 2];
      swing_peak_[i] = std::max(swing_peak_[i], z);
    }
    UpdateJoystickObs(contact, /*add_noise=*/true);
    terminated_ = JoystickTermination();
    const mjtNum reward = ComputeJoystickReward(act, first_contact, contact);
    std::copy(last_act_.begin(), last_act_.end(), last_last_act_.begin());
    std::copy(act, act + kSpotActionDim, last_act_.begin());
    ++step_;
    if (step_ > 200) {
      SampleCommand();
    }
    if (terminated_ || step_ > 200) {
      step_ = 0;
    }
    for (int i = 0; i < kSpotFeet; ++i) {
      if (contact[i]) {
        feet_air_time_[i] = 0.0;
        swing_peak_[i] = 0.0;
      }
      last_contact_[i] = contact[i];
    }
    ++elapsed_step_;
    done_ = terminated_ || elapsed_step_ >= max_episode_steps_;
    WriteState(static_cast<float>(reward), false);
  }

  void StepGait(const mjtNum* act) {
    for (int i = 0; i < kSpotActionDim; ++i) {
      const mjtNum target =
          default_pose_[i] + act[i] * spec_.config["action_scale"_];
      motor_targets_[i] = std::clamp(target, lowers_[i], uppers_[i]);
      data_->ctrl[i] = motor_targets_[i];
    }
    for (int i = 0; i < n_substeps_; ++i) {
      mj_step(model_, data_);
    }
    const auto contact = FootContact();
    for (int i = 0; i < kSpotFeet; ++i) {
      const mjtNum z = data_->site_xpos[feet_site_ids_[i] * 3 + 2];
      swing_peak_[i] = std::max(swing_peak_[i], z);
    }
    UpdateGaitObs(contact, /*add_noise=*/true);
    terminated_ = JoystickTermination();
    const mjtNum reward = ComputeGaitReward();
    std::copy(last_act_.begin(), last_act_.end(), last_last_act_.begin());
    std::copy(act, act + kSpotActionDim, last_act_.begin());
    ++step_;
    for (mjtNum& value : phase_) {
      value = WrapPhase(value + phase_dt_);
    }
    if (step_ > 200) {
      SampleCommand();
    }
    if (terminated_ || step_ > 200) {
      step_ = 0;
    }
    for (int i = 0; i < kSpotFeet; ++i) {
      last_contact_[i] = contact[i];
      if (contact[i]) {
        swing_peak_[i] = 0.0;
      }
    }
    ++elapsed_step_;
    done_ = terminated_ || elapsed_step_ >= max_episode_steps_;
    WriteState(static_cast<float>(reward), false);
  }

  void StepGetup(const mjtNum* act) {
    for (int i = 0; i < kSpotActionDim; ++i) {
      const mjtNum target =
          data_->qpos[7 + i] + act[i] * spec_.config["action_scale"_];
      motor_targets_[i] = std::clamp(target, lowers_[i], uppers_[i]);
      data_->ctrl[i] = motor_targets_[i];
    }
    for (int i = 0; i < n_substeps_; ++i) {
      mj_step(model_, data_);
    }
    UpdateGetupObs(/*add_noise=*/true);
    terminated_ = JointLimitTermination();
    const mjtNum reward = ComputeGetupReward(act);
    std::copy(last_act_.begin(), last_act_.end(), last_last_act_.begin());
    std::copy(act, act + kSpotActionDim, last_act_.begin());
    ++elapsed_step_;
    done_ = terminated_ || elapsed_step_ >= max_episode_steps_;
    WriteState(static_cast<float>(reward), false);
  }

  void ResetRewards() {
    reward_tracking_lin_vel_ = 0.0;
    reward_tracking_ang_vel_ = 0.0;
    reward_lin_vel_z_ = 0.0;
    reward_ang_vel_xy_ = 0.0;
    reward_orientation_ = 0.0;
    reward_termination_ = 0.0;
    reward_posture_ = 0.0;
    reward_torques_ = 0.0;
    reward_action_rate_ = 0.0;
    reward_energy_ = 0.0;
    reward_feet_slip_ = 0.0;
    reward_feet_clearance_ = 0.0;
    reward_feet_height_ = 0.0;
    reward_feet_air_time_ = 0.0;
    reward_feet_phase_ = 0.0;
    reward_hip_splay_ = 0.0;
    reward_torso_height_ = 0.0;
    reward_stand_still_ = 0.0;
  }

  mjtNum ComputeJoystickReward(const mjtNum* act,
                               const std::array<bool, kSpotFeet>& first_contact,
                               const std::array<bool, kSpotFeet>& contact) {
    const mjtNum lin_x_err =
        command_[0] - data_->sensordata[local_linvel_adr_ + 0];
    const mjtNum lin_y_err =
        command_[1] - data_->sensordata[local_linvel_adr_ + 1];
    reward_tracking_lin_vel_ =
        std::exp(-(lin_x_err * lin_x_err + lin_y_err * lin_y_err) /
                 spec_.config["tracking_sigma"_]) *
        spec_.config["tracking_lin_vel_scale"_];
    const mjtNum yaw_err = command_[2] - data_->sensordata[gyro_adr_ + 2];
    reward_tracking_ang_vel_ =
        std::exp(-(yaw_err * yaw_err) / spec_.config["tracking_sigma"_]) *
        spec_.config["tracking_ang_vel_scale"_];
    const mjtNum lin_z = data_->sensordata[global_linvel_adr_ + 2];
    reward_lin_vel_z_ = lin_z * lin_z * spec_.config["lin_vel_z_scale"_];
    const mjtNum ang_x = data_->sensordata[global_angvel_adr_ + 0];
    const mjtNum ang_y = data_->sensordata[global_angvel_adr_ + 1];
    reward_ang_vel_xy_ =
        (ang_x * ang_x + ang_y * ang_y) * spec_.config["ang_vel_xy_scale"_];
    const mjtNum up_x = data_->sensordata[upvector_adr_ + 0];
    const mjtNum up_y = data_->sensordata[upvector_adr_ + 1];
    reward_orientation_ =
        (up_x * up_x + up_y * up_y) * spec_.config["orientation_scale"_];
    reward_termination_ =
        (terminated_ ? 1.0 : 0.0) * spec_.config["termination_scale"_];
    mjtNum posture_cost = 0.0;
    for (int i = 0; i < kSpotActionDim; ++i) {
      const mjtNum delta = data_->qpos[7 + i] - default_pose_[i];
      posture_cost += delta * delta;
    }
    const mjtNum cmd_norm =
        std::sqrt(command_[0] * command_[0] + command_[1] * command_[1] +
                  command_[2] * command_[2]);
    reward_posture_ = std::exp((cmd_norm < 0.01 ? -10.0 : 0.0) * posture_cost) *
                      spec_.config["posture_scale"_];
    mjtNum torques_l2 = 0.0;
    mjtNum torques_l1 = 0.0;
    mjtNum energy = 0.0;
    for (int i = 0; i < kSpotActionDim; ++i) {
      torques_l2 += data_->actuator_force[i] * data_->actuator_force[i];
      torques_l1 += std::abs(data_->actuator_force[i]);
      energy +=
          std::abs(data_->qvel[6 + i]) * std::abs(data_->actuator_force[i]);
    }
    reward_torques_ =
        (std::sqrt(torques_l2) + torques_l1) * spec_.config["torques_scale"_];
    mjtNum action_rate = 0.0;
    for (int i = 0; i < kSpotActionDim; ++i) {
      const mjtNum c1 = act[i] - last_act_[i];
      const mjtNum c2 = act[i] - 2.0 * last_act_[i] + last_last_act_[i];
      action_rate += c1 * c1 + c2 * c2;
    }
    reward_action_rate_ = action_rate * spec_.config["action_rate_scale"_];
    reward_energy_ = energy * spec_.config["energy_scale"_];
    mjtNum feet_slip = 0.0;
    mjtNum feet_clearance = 0.0;
    mjtNum feet_height = 0.0;
    for (int foot = 0; foot < kSpotFeet; ++foot) {
      const int adr = feet_linvel_sensor_adrs_[foot];
      const mjtNum vx = data_->sensordata[adr + 0];
      const mjtNum vy = data_->sensordata[adr + 1];
      feet_slip += (vx * vx + vy * vy) * (contact[foot] ? 1.0 : 0.0);
      const mjtNum vel_norm = std::sqrt(std::sqrt(vx * vx + vy * vy));
      const mjtNum z = data_->site_xpos[feet_site_ids_[foot] * 3 + 2];
      feet_clearance +=
          std::abs(z - spec_.config["max_foot_height"_]) * vel_norm;
      const mjtNum height_error =
          swing_peak_[foot] / spec_.config["max_foot_height"_] - 1.0;
      feet_height +=
          height_error * height_error * (first_contact[foot] ? 1.0 : 0.0);
    }
    reward_feet_slip_ = feet_slip * spec_.config["feet_slip_scale"_];
    reward_feet_clearance_ =
        feet_clearance * spec_.config["feet_clearance_scale"_];
    reward_feet_height_ = (cmd_norm >= 0.01 ? feet_height : 0.0) *
                          spec_.config["feet_height_scale"_];
    mjtNum feet_air_time = 0.0;
    for (int i = 0; i < kSpotFeet; ++i) {
      feet_air_time +=
          (feet_air_time_[i] - 0.1) * (first_contact[i] ? 1.0 : 0.0);
    }
    reward_feet_air_time_ = (cmd_norm >= 0.01 ? feet_air_time : 0.0) *
                            spec_.config["feet_air_time_scale"_];
    mjtNum reward =
        (reward_tracking_lin_vel_ + reward_tracking_ang_vel_ +
         reward_lin_vel_z_ + reward_ang_vel_xy_ + reward_orientation_ +
         reward_termination_ + reward_posture_ + reward_torques_ +
         reward_action_rate_ + reward_energy_ + reward_feet_slip_ +
         reward_feet_clearance_ + reward_feet_height_ + reward_feet_air_time_) *
        Dt();
    return std::clamp(reward, static_cast<mjtNum>(0.0),
                      static_cast<mjtNum>(10000.0));
  }

  mjtNum RewardFeetPhase(mjtNum denom) const {
    mjtNum error = 0.0;
    for (int foot = 0; foot < kSpotFeet; ++foot) {
      const mjtNum z = data_->site_xpos[feet_site_ids_[foot] * 3 + 2];
      const mjtNum target = GaitRz(phase_[foot], foot_height_);
      const mjtNum delta = z - target;
      error += delta * delta;
    }
    return std::exp(-error / denom);
  }

  mjtNum ComputeGaitReward() {
    reward_tracking_lin_vel_ =
        std::exp(
            -((command_[0] - data_->sensordata[local_linvel_adr_ + 0]) *
                  (command_[0] - data_->sensordata[local_linvel_adr_ + 0]) +
              (command_[1] - data_->sensordata[local_linvel_adr_ + 1]) *
                  (command_[1] - data_->sensordata[local_linvel_adr_ + 1])) /
            spec_.config["tracking_sigma"_]) *
        spec_.config["tracking_lin_vel_scale"_];
    const mjtNum yaw_err = command_[2] - data_->sensordata[gyro_adr_ + 2];
    reward_tracking_ang_vel_ =
        std::exp(-(yaw_err * yaw_err) / spec_.config["tracking_sigma"_]) *
        spec_.config["tracking_ang_vel_scale"_];
    reward_feet_phase_ =
        RewardFeetPhase(0.1) * spec_.config["feet_phase_scale"_];
    const mjtNum lin_z = data_->sensordata[global_linvel_adr_ + 2];
    reward_lin_vel_z_ =
        (gait_ > 2 ? lin_z * lin_z : 0.0) * spec_.config["lin_vel_z_scale"_];
    const mjtNum ang_x = data_->sensordata[global_angvel_adr_ + 0];
    const mjtNum ang_y = data_->sensordata[global_angvel_adr_ + 1];
    reward_ang_vel_xy_ =
        (ang_x * ang_x + ang_y * ang_y) * spec_.config["ang_vel_xy_scale"_];
    mjtNum hip_splay = 0.0;
    for (int index : {0, 3, 6, 9}) {
      const int slot = index / 3;
      const mjtNum delta = data_->qpos[7 + index] - hx_default_pose_[slot];
      hip_splay += delta * delta;
    }
    reward_hip_splay_ = hip_splay * spec_.config["hip_splay_scale"_];
    const mjtNum pos = reward_tracking_lin_vel_ + reward_tracking_ang_vel_ +
                       reward_feet_phase_;
    const mjtNum neg =
        reward_ang_vel_xy_ + reward_lin_vel_z_ + reward_hip_splay_;
    return pos * std::exp(0.2 * neg) * Dt();
  }

  mjtNum ComputeGetupReward(const mjtNum* act) {
    const mjtNum dx = up_vec_[0] - data_->sensordata[upvector_adr_ + 0];
    const mjtNum dy = up_vec_[1] - data_->sensordata[upvector_adr_ + 1];
    const mjtNum dz = up_vec_[2] - data_->sensordata[upvector_adr_ + 2];
    const mjtNum ori_error = dx * dx + dy * dy + dz * dz;
    const bool is_upright = ori_error < 0.01;
    reward_orientation_ =
        std::exp(-2.0 * ori_error) * spec_.config["orientation_scale"_];
    const mjtNum height_error =
        std::clamp((z_des_ - data_->qpos[2]) / z_des_, 0.0, 1.0);
    const bool is_at_height = height_error < 0.005;
    reward_torso_height_ =
        (1.0 - height_error) * spec_.config["torso_height_scale"_];
    mjtNum posture_cost = 0.0;
    for (int i = 0; i < kSpotActionDim; ++i) {
      const mjtNum delta = data_->qpos[7 + i] - default_pose_[i];
      posture_cost += delta * delta;
    }
    const bool gate = is_upright && is_at_height;
    reward_posture_ = (gate ? std::exp(-0.5 * posture_cost) : 0.0) *
                      spec_.config["posture_scale"_];
    mjtNum action_cost = 0.0;
    mjtNum action_rate = 0.0;
    mjtNum torques_l2 = 0.0;
    mjtNum torques_l1 = 0.0;
    for (int i = 0; i < kSpotActionDim; ++i) {
      action_cost += act[i] * act[i];
      const mjtNum c1 = act[i] - last_act_[i];
      const mjtNum c2 = act[i] - 2.0 * last_act_[i] + last_last_act_[i];
      action_rate += c1 * c1 + c2 * c2;
      torques_l2 += data_->actuator_force[i] * data_->actuator_force[i];
      torques_l1 += std::abs(data_->actuator_force[i]);
    }
    reward_stand_still_ = (gate ? std::exp(-0.5 * action_cost) : 0.0) *
                          spec_.config["stand_still_scale"_];
    reward_action_rate_ = action_rate * spec_.config["action_rate_scale"_];
    reward_torques_ =
        (std::sqrt(torques_l2) + torques_l1) * spec_.config["torques_scale"_];
    const mjtNum reward =
        (reward_orientation_ + reward_torso_height_ + reward_posture_ +
         reward_stand_still_ + reward_action_rate_ + reward_torques_) *
        Dt();
    return std::clamp(reward, static_cast<mjtNum>(0.0),
                      static_cast<mjtNum>(10000.0));
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
      if (is_getup_) {
        std::copy(getup_obs_.begin(), getup_obs_.end(), obs);
      } else if (is_gait_) {
        std::copy(gait_obs_.begin(), gait_obs_.end(), obs);
      } else {
        std::copy(joystick_obs_.begin(), joystick_obs_.end(), obs);
      }
      if constexpr (std::is_same_v<EnvSpecT, PlaygroundSpotJoystickEnvSpec>) {
        auto obs_privileged = state["obs:privileged_state"_];
        mjtNum* privileged =
            PrepareObservation("obs:privileged_state", &obs_privileged);
        std::copy(privileged_obs_.begin(), privileged_obs_.end(), privileged);
        CommitObservation("obs:privileged_state", &obs_privileged, reset);
      }
      CommitObservation("obs:state", &obs_state, reset);
    }
    state["info:terminated"_] = terminated_;
    state["info:command"_].Assign(command_.data(), command_.size());
    state["info:reward_tracking_lin_vel"_] = reward_tracking_lin_vel_;
    state["info:reward_tracking_ang_vel"_] = reward_tracking_ang_vel_;
    state["info:reward_feet_phase"_] = reward_feet_phase_;
    state["info:reward_lin_vel_z"_] = reward_lin_vel_z_;
    state["info:reward_ang_vel_xy"_] = reward_ang_vel_xy_;
    state["info:reward_hip_splay"_] = reward_hip_splay_;
    state["info:reward_orientation"_] = reward_orientation_;
    state["info:reward_termination"_] = reward_termination_;
    state["info:reward_posture"_] = reward_posture_;
    state["info:reward_torques"_] = reward_torques_;
    state["info:reward_action_rate"_] = reward_action_rate_;
    state["info:reward_energy"_] = reward_energy_;
    state["info:reward_feet_slip"_] = reward_feet_slip_;
    state["info:reward_feet_clearance"_] = reward_feet_clearance_;
    state["info:reward_feet_height"_] = reward_feet_height_;
    state["info:reward_feet_air_time"_] = reward_feet_air_time_;
    state["info:reward_torso_height"_] = reward_torso_height_;
    state["info:reward_stand_still"_] = reward_stand_still_;
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
    state["info:actuator_force"_].Assign(data_->actuator_force, kSpotActionDim);
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
    state["info:swing_peak"_].Assign(swing_peak_.data(), swing_peak_.size());
    state["info:feet_air_time"_].Assign(feet_air_time_.data(),
                                        feet_air_time_.size());
    state["info:last_contact"_].Assign(last_contact_.data(),
                                       last_contact_.size());
    state["info:gait_freq"_] = gait_freq_;
    state["info:gait"_] = gait_;
    state["info:phase"_].Assign(phase_.data(), phase_.size());
    state["info:phase_dt"_] = phase_dt_;
    state["info:foot_height"_] = foot_height_;
    state["info:step"_] = step_;
#endif
  }
};

template <typename Spec, bool kFromPixels>
using SpotBase = PlaygroundSpotEnvBase<Spec, kFromPixels>;
using SpotJoystickEnv = SpotBase<PlaygroundSpotJoystickEnvSpec, false>;
using SpotJoystickPixelEnv = SpotBase<PlaygroundSpotJoystickPixelEnvSpec, true>;
using SpotGetupEnv = SpotBase<PlaygroundSpotGetupEnvSpec, false>;
using SpotGetupPixelEnv = SpotBase<PlaygroundSpotGetupPixelEnvSpec, true>;
using SpotGaitEnv = SpotBase<PlaygroundSpotGaitEnvSpec, false>;
using SpotGaitPixelEnv = SpotBase<PlaygroundSpotGaitPixelEnvSpec, true>;
using PlaygroundSpotJoystickEnv = SpotJoystickEnv;
using PlaygroundSpotJoystickPixelEnv = SpotJoystickPixelEnv;
using PlaygroundSpotGetupEnv = SpotGetupEnv;
using PlaygroundSpotGetupPixelEnv = SpotGetupPixelEnv;
using PlaygroundSpotGaitEnv = SpotGaitEnv;
using PlaygroundSpotGaitPixelEnv = SpotGaitPixelEnv;
using SpotJoystickEnvPool = PlaygroundEnvPoolT<PlaygroundSpotJoystickEnv>;
using SpotJoystickPixelEnvPool = PlaygroundEnvPoolT<SpotJoystickPixelEnv>;
using SpotGetupEnvPool = PlaygroundEnvPoolT<PlaygroundSpotGetupEnv>;
using SpotGetupPixelEnvPool = PlaygroundEnvPoolT<PlaygroundSpotGetupPixelEnv>;
using SpotGaitEnvPool = PlaygroundEnvPoolT<PlaygroundSpotGaitEnv>;
using SpotGaitPixelEnvPool = PlaygroundEnvPoolT<PlaygroundSpotGaitPixelEnv>;
using PlaygroundSpotJoystickEnvPool = SpotJoystickEnvPool;
using PlaygroundSpotJoystickPixelEnvPool = SpotJoystickPixelEnvPool;
using PlaygroundSpotGetupEnvPool = SpotGetupEnvPool;
using PlaygroundSpotGetupPixelEnvPool = SpotGetupPixelEnvPool;
using PlaygroundSpotGaitEnvPool = SpotGaitEnvPool;
using PlaygroundSpotGaitPixelEnvPool = SpotGaitPixelEnvPool;

}  // namespace mujoco_playground

#endif  // ENVPOOL_MUJOCO_PLAYGROUND_SPOT_H_
