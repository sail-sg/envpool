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

#ifndef ENVPOOL_MUJOCO_PLAYGROUND_HAND_H_
#define ENVPOOL_MUJOCO_PLAYGROUND_HAND_H_

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

constexpr int kHandQDim = 16;
constexpr int kLeapActionDim = 16;
constexpr int kAeroActionDim = 7;
constexpr int kHandMaxActionDim = kLeapActionDim;
constexpr int kLeapRotateStateDim = 32;
constexpr int kLeapRotatePrivilegedDim = 105;
constexpr int kLeapReorientStateDim = 57;
constexpr int kLeapReorientPrivilegedDim = 128;
constexpr int kAeroRotateStateDim = 14;
constexpr int kAeroRotatePrivilegedDim = 81;
constexpr int kHandMaxStateDim = kLeapReorientStateDim;
constexpr int kHandMaxPrivilegedDim = kLeapReorientPrivilegedDim;
constexpr int kHandMaxFingertipScalars = 15;
constexpr int kAeroTendonSensorCount = 6;
constexpr int kAeroJointSensorCount = 1;

class PlaygroundHandEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "frame_stack"_.Bind(1),
        "task_name"_.Bind(std::string("LeapCubeRotateZAxis")),
        "ctrl_dt"_.Bind(0.05), "sim_dt"_.Bind(0.01), "action_scale"_.Bind(0.6),
        "ema_alpha"_.Bind(1.0), "history_len"_.Bind(1),
        "noise_level"_.Bind(1.0), "noise_joint_pos"_.Bind(0.05),
        "noise_tendon_length"_.Bind(0.005), "noise_cube_pos"_.Bind(0.02),
        "noise_cube_ori"_.Bind(0.1), "random_ori_injection_prob"_.Bind(0.0),
        "success_threshold"_.Bind(0.1), "success_reward"_.Bind(0.0),
        "pert_enable"_.Bind(0.0), "angvel_scale"_.Bind(1.0),
        "linvel_scale"_.Bind(0.0), "pose_scale"_.Bind(0.0),
        "torques_scale"_.Bind(0.0), "energy_scale"_.Bind(0.0),
        "termination_scale"_.Bind(-100.0), "action_rate_scale"_.Bind(0.0),
        "orientation_scale"_.Bind(0.0), "position_scale"_.Bind(0.0),
        "hand_pose_scale"_.Bind(0.0), "joint_vel_scale"_.Bind(0.0),
        "aero_action_scale0"_.Bind(0.02), "aero_action_scale1"_.Bind(0.02),
        "aero_action_scale2"_.Bind(0.02), "aero_action_scale3"_.Bind(0.02),
        "aero_action_scale4"_.Bind(0.7), "aero_action_scale5"_.Bind(0.003),
        "aero_action_scale6"_.Bind(0.012));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    const std::string task_name = conf["task_name"_];
    int state_dim = kLeapRotateStateDim;
    int privileged_dim = kLeapRotatePrivilegedDim;
    if (task_name == "LeapCubeReorient") {
      state_dim = kLeapReorientStateDim;
      privileged_dim = kLeapReorientPrivilegedDim;
    } else if (task_name == "AeroCubeRotateZAxis") {
      state_dim = kAeroRotateStateDim;
      privileged_dim = kAeroRotatePrivilegedDim;
    }
    return MakeDict(
        "obs:state"_.Bind(StackSpec(Spec<mjtNum>({state_dim}, {-inf, inf}),
                                    conf["frame_stack"_])),
        "obs:privileged_state"_.Bind(StackSpec(
            Spec<mjtNum>({privileged_dim}, {-inf, inf}), conf["frame_stack"_])),
        "info:terminated"_.Bind(Spec<bool>({-1})),
        "info:success"_.Bind(Spec<mjtNum>({-1})),
        "info:steps_since_last_success"_.Bind(Spec<int>({-1})),
        "info:success_count"_.Bind(Spec<int>({-1})),
        "info:reward_angvel"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_linvel"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_pose"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_torques"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_energy"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_termination"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_action_rate"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_orientation"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_position"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_hand_pose"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_joint_vel"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_success"_.Bind(Spec<mjtNum>({-1}))
#ifdef ENVPOOL_TEST
            ,
        "info:qpos"_.Bind(Spec<mjtNum>({64})),
        "info:qvel"_.Bind(Spec<mjtNum>({64})),
        "info:ctrl"_.Bind(Spec<mjtNum>({64})),
        "info:qacc"_.Bind(Spec<mjtNum>({64})),
        "info:qacc_warmstart"_.Bind(Spec<mjtNum>({64})),
        "info:xfrc_applied"_.Bind(Spec<mjtNum>({512})),
        "info:mocap_pos"_.Bind(Spec<mjtNum>({64})),
        "info:mocap_quat"_.Bind(Spec<mjtNum>({64})),
        "info:xpos"_.Bind(Spec<mjtNum>({512})),
        "info:xmat"_.Bind(Spec<mjtNum>({2048})),
        "info:xquat"_.Bind(Spec<mjtNum>({512})),
        "info:site_xpos"_.Bind(Spec<mjtNum>({512})),
        "info:site_xmat"_.Bind(Spec<mjtNum>({2048})),
        "info:sensordata"_.Bind(Spec<mjtNum>({512})),
        "info:actuator_force"_.Bind(Spec<mjtNum>({64})),
        "info:qfrc_actuator"_.Bind(Spec<mjtNum>({64})),
        "info:last_act"_.Bind(Spec<mjtNum>({kHandMaxActionDim})),
        "info:last_last_act"_.Bind(Spec<mjtNum>({kHandMaxActionDim})),
        "info:motor_targets"_.Bind(Spec<mjtNum>({kHandMaxActionDim})),
        "info:last_cube_angvel"_.Bind(Spec<mjtNum>({3})),
        "info:obs_history"_.Bind(Spec<mjtNum>({128})),
        "info:qpos_error_history"_.Bind(Spec<mjtNum>({64})),
        "info:cube_pos_error_history"_.Bind(Spec<mjtNum>({32})),
        "info:cube_ori_error_history"_.Bind(Spec<mjtNum>({64})),
        "info:goal_quat_dquat"_.Bind(Spec<mjtNum>({3})),
        "info:pert_dir"_.Bind(Spec<mjtNum>({6})),
        "info:step"_.Bind(Spec<int>({-1}))
#endif
    );  // NOLINT
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    const std::string task_name = conf["task_name"_];
    const int action_dim =
        task_name == "AeroCubeRotateZAxis" ? kAeroActionDim : kLeapActionDim;
    return MakeDict(
        "action"_.Bind(Spec<mjtNum>({-1, action_dim}, {-1.0, 1.0})));
  }
};

using PlaygroundHandEnvSpec = EnvSpec<PlaygroundHandEnvFns>;
using PlaygroundHandPixelEnvFns = PixelObservationEnvFns<PlaygroundHandEnvFns>;
using PlaygroundHandPixelEnvSpec = EnvSpec<PlaygroundHandPixelEnvFns>;

template <typename EnvSpecT, bool kFromPixels>
class PlaygroundHandEnvBase : public Env<EnvSpecT>, public PlaygroundMujocoEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  bool is_aero_{false};
  bool is_reorient_{false};
  int action_dim_{kLeapActionDim};
  int state_dim_{kLeapRotateStateDim};
  int privileged_dim_{kLeapRotatePrivilegedDim};
  int n_substeps_{5};
  std::vector<mjtNum> init_qpos_;
  std::vector<mjtNum> init_ctrl_;
  std::vector<mjtNum> init_mpos_;
  std::vector<mjtNum> init_mquat_;
  std::array<int, kHandQDim> hand_qposadr_{};
  std::array<int, kHandQDim> hand_dofadr_{};
  std::array<mjtNum, kHandQDim> default_pose_{};
  std::array<mjtNum, kHandQDim> lowers_{};
  std::array<mjtNum, kHandQDim> uppers_{};
  std::array<mjtNum, kAeroActionDim> aero_action_scale_{};
  std::array<int, kAeroTendonSensorCount> aero_tendon_sensor_adrs_{};
  std::array<int, kAeroJointSensorCount> aero_joint_sensor_adrs_{};
  std::array<int, kHandMaxFingertipScalars / 3> fingertip_sensor_adrs_{};
  int fingertip_count_{4};
  int cube_body_id_{-1};
  int cube_qposadr_{-1};
  int goal_mocap_id_{-1};
  int cube_position_sensor_adr_{-1};
  int cube_orientation_sensor_adr_{-1};
  int cube_linvel_sensor_adr_{-1};
  int cube_angvel_sensor_adr_{-1};
  int palm_position_sensor_adr_{-1};
  int cube_goal_orientation_sensor_adr_{-1};

  std::array<mjtNum, kHandMaxActionDim> last_act_{};
  std::array<mjtNum, kHandMaxActionDim> last_last_act_{};
  std::array<mjtNum, kHandMaxActionDim> motor_targets_{};
  std::array<mjtNum, 3> last_cube_angvel_{};
  std::vector<mjtNum> obs_history_;
  std::vector<mjtNum> qpos_error_history_;
  std::vector<mjtNum> cube_pos_error_history_;
  std::vector<mjtNum> cube_ori_error_history_;
  std::array<mjtNum, 3> goal_quat_dquat_{};
  std::array<mjtNum, 6> pert_dir_{};
  int step_{0};
  int steps_since_last_success_{0};
  int success_count_{0};
  mjtNum success_{0.0};

  std::array<mjtNum, kHandMaxStateDim> obs_{};
  std::array<mjtNum, kHandMaxPrivilegedDim> privileged_obs_{};
  mjtNum reward_angvel_{0.0};
  mjtNum reward_linvel_{0.0};
  mjtNum reward_pose_{0.0};
  mjtNum reward_torques_{0.0};
  mjtNum reward_energy_{0.0};
  mjtNum reward_termination_{0.0};
  mjtNum reward_action_rate_{0.0};
  mjtNum reward_orientation_{0.0};
  mjtNum reward_position_{0.0};
  mjtNum reward_hand_pose_{0.0};
  mjtNum reward_joint_vel_{0.0};
  mjtNum reward_success_{0.0};

  std::uniform_real_distribution<mjtNum> unit_uniform_{0.0, 1.0};
  std::normal_distribution<mjtNum> normal_{0.0, 1.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  PlaygroundHandEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        PlaygroundMujocoEnv(
            XmlPath(spec.config["base_path"_], spec.config["task_name"_]),
            spec.config["max_episode_steps"_], spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config),
            envpool::mujoco::CameraPolicy::kDmControl) {
    const std::string task_name = spec.config["task_name"_];
    is_aero_ = task_name == "AeroCubeRotateZAxis";
    is_reorient_ = task_name == "LeapCubeReorient";
    if (!is_aero_ && !is_reorient_ && task_name != "LeapCubeRotateZAxis") {
      throw std::runtime_error("Unsupported Playground hand task: " +
                               task_name);
    }
    action_dim_ = is_aero_ ? kAeroActionDim : kLeapActionDim;
    state_dim_ = kLeapRotateStateDim;
    privileged_dim_ = kLeapRotatePrivilegedDim;
    if (is_reorient_) {
      state_dim_ = kLeapReorientStateDim;
      privileged_dim_ = kLeapReorientPrivilegedDim;
    } else if (is_aero_) {
      state_dim_ = kAeroRotateStateDim;
      privileged_dim_ = kAeroRotatePrivilegedDim;
    }
    if (model_->nu != action_dim_) {
      throw std::runtime_error("Unexpected Playground hand action dimension.");
    }
    n_substeps_ = static_cast<int>(
        std::llround(spec.config["ctrl_dt"_] / spec.config["sim_dt"_]));
    model_->opt.timestep = spec.config["sim_dt"_];
    if (!is_aero_) {
      model_->opt.ccd_iterations = 10;
    }
    aero_action_scale_ = {
        spec.config["aero_action_scale0"_], spec.config["aero_action_scale1"_],
        spec.config["aero_action_scale2"_], spec.config["aero_action_scale3"_],
        spec.config["aero_action_scale4"_], spec.config["aero_action_scale5"_],
        spec.config["aero_action_scale6"_],
    };
    InitModelIds();
    const int history_len = spec.config["history_len"_];
    obs_history_.assign(
        history_len * (is_aero_ ? kAeroRotateStateDim : kLeapRotateStateDim),
        0.0);
    qpos_error_history_.assign(history_len * kHandQDim, 0.0);
    cube_pos_error_history_.assign(history_len * 3, 0.0);
    cube_ori_error_history_.assign(history_len * 6, 0.0);
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    has_cached_render_ = false;
    done_ = false;
    terminated_ = false;
    elapsed_step_ = 0;
    step_ = 0;
    steps_since_last_success_ = 0;
    success_count_ = 0;
    success_ = 0.0;
    ResetRewardInfo();
    mj_resetData(model_, data_);
    std::copy(init_qpos_.begin(), init_qpos_.end(), data_->qpos);
    std::fill(data_->qvel, data_->qvel + model_->nv, 0.0);
    std::copy(init_ctrl_.begin(), init_ctrl_.end(), data_->ctrl);
    std::fill(data_->xfrc_applied, data_->xfrc_applied + model_->nbody * 6,
              0.0);
    RestoreHomeMocap();

    if (is_reorient_) {
      const auto goal_quat = RandomQuat();
      for (int i = 0; i < 4; ++i) {
        data_->mocap_quat[4 * goal_mocap_id_ + i] = goal_quat[i];
      }
    } else {
      for (int i = 0; i < 3; ++i) {
        data_->mocap_pos[3 * goal_mocap_id_ + i] = -100.0;
      }
    }

    for (int i = 0; i < kHandQDim; ++i) {
      data_->qpos[hand_qposadr_[i]] = std::clamp(
          default_pose_[i] + 0.1 * normal_(gen_), lowers_[i], uppers_[i]);
      motor_targets_[i] = data_->qpos[hand_qposadr_[i]];
    }
    if (is_aero_) {
      for (int i = 0; i < action_dim_; ++i) {
        motor_targets_[i] = init_ctrl_[i];
        data_->ctrl[i] = init_ctrl_[i];
      }
    } else {
      for (int i = 0; i < action_dim_; ++i) {
        data_->ctrl[i] = motor_targets_[i];
      }
    }

    const auto start_quat = RandomQuat();
    data_->qpos[cube_qposadr_ + 0] = 0.1 + Uniform(-0.01, 0.01);
    data_->qpos[cube_qposadr_ + 1] = Uniform(-0.01, 0.01);
    data_->qpos[cube_qposadr_ + 2] = 0.05 + Uniform(-0.01, 0.01);
    for (int i = 0; i < 4; ++i) {
      data_->qpos[cube_qposadr_ + 3 + i] = start_quat[i];
    }

    mj_forward(model_, data_);
    last_act_.fill(0.0);
    last_last_act_.fill(0.0);
    last_cube_angvel_.fill(0.0);
    std::fill(obs_history_.begin(), obs_history_.end(), 0.0);
    std::fill(qpos_error_history_.begin(), qpos_error_history_.end(), 0.0);
    std::fill(cube_pos_error_history_.begin(), cube_pos_error_history_.end(),
              0.0);
    std::fill(cube_ori_error_history_.begin(), cube_ori_error_history_.end(),
              0.0);
    goal_quat_dquat_.fill(0.0);
    pert_dir_.fill(0.0);

    if (is_reorient_) {
      FillReorientObs();
    } else {
      FillRotateObs();
    }
    WriteState(0.0, true);
  }

  void Step(const Action& action) override {
    has_cached_render_ = false;
    const auto* act = static_cast<const mjtNum*>(action["action"_].Data());
    std::array<mjtNum, kHandMaxActionDim> raw_action{};
    for (int i = 0; i < action_dim_; ++i) {
      raw_action[i] = act[i];
    }
    ResetRewardInfo();
    if (is_reorient_) {
      StepReorient(raw_action);
    } else {
      StepRotate(raw_action);
    }
  }

 private:
  static std::string XmlPath(const std::string& base_path,
                             const std::string& task_name) {
    const char* family =
        task_name == "AeroCubeRotateZAxis" ? "aero_hand" : "leap_hand";
    return base_path +
           "/mujoco/playground/assets/mujoco_playground/_src/"
           "manipulation/" +
           family + "/xmls/scene_mjx_cube.xml";
  }

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

  mjtNum* SensorData(const std::string& name) const {
    return data_->sensordata + SensorAdr(name);
  }

  mjtNum Uniform(mjtNum low, mjtNum high) {
    return low + (high - low) * unit_uniform_(gen_);
  }

  std::array<mjtNum, 4> RandomQuat() {
    const mjtNum u = unit_uniform_(gen_);
    const mjtNum v = unit_uniform_(gen_);
    const mjtNum w = unit_uniform_(gen_);
    std::array<mjtNum, 4> quat = {
        std::sqrt(1.0 - u) * std::sin(2.0 * M_PI * v),
        std::sqrt(1.0 - u) * std::cos(2.0 * M_PI * v),
        std::sqrt(u) * std::sin(2.0 * M_PI * w),
        std::sqrt(u) * std::cos(2.0 * M_PI * w),
    };
    mju_normalize4(quat.data());
    return quat;
  }

  void InitModelIds() {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const char* leap_joints[kHandQDim] = {
        "if_mcp", "if_rot", "if_pip", "if_dip", "mf_mcp", "mf_rot",
        "mf_pip", "mf_dip", "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
        "th_cmc", "th_axl", "th_mcp", "th_ipl"};
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const char* aero_joints[kHandQDim] = {
        "right_index_mcp_flex",  "right_index_pip",      "right_index_dip",
        "right_middle_mcp_flex", "right_middle_pip",     "right_middle_dip",
        "right_ring_mcp_flex",   "right_ring_pip",       "right_ring_dip",
        "right_pinky_mcp_flex",  "right_pinky_pip",      "right_pinky_dip",
        "right_thumb_cmc_abd",   "right_thumb_cmc_flex", "right_thumb_mcp",
        "right_thumb_ip"};
    const char** joints = is_aero_ ? aero_joints : leap_joints;
    for (int i = 0; i < kHandQDim; ++i) {
      const int joint_id = RequireId(mjOBJ_JOINT, joints[i]);
      hand_qposadr_[i] = model_->jnt_qposadr[joint_id];
      hand_dofadr_[i] = model_->jnt_dofadr[joint_id];
    }

    cube_body_id_ = RequireId(mjOBJ_BODY, "cube");
    cube_qposadr_ = model_->jnt_qposadr[model_->body_jntadr[cube_body_id_]];
    const int goal_body = RequireId(mjOBJ_BODY, "goal");
    goal_mocap_id_ = model_->body_mocapid[goal_body];
    if (goal_mocap_id_ < 0) {
      throw std::runtime_error("Playground hand goal body is not mocap.");
    }

    cube_position_sensor_adr_ = SensorAdr("cube_position");
    cube_orientation_sensor_adr_ = SensorAdr("cube_orientation");
    cube_linvel_sensor_adr_ = SensorAdr("cube_linvel");
    cube_angvel_sensor_adr_ = SensorAdr("cube_angvel");
    palm_position_sensor_adr_ = SensorAdr("palm_position");
    cube_goal_orientation_sensor_adr_ = SensorAdr("cube_goal_orientation");
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const char* leap_tips[4] = {"th_tip_position", "if_tip_position",
                                "mf_tip_position", "rf_tip_position"};
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const char* aero_tips[5] = {"if_tip_position", "mf_tip_position",
                                "rf_tip_position", "pf_tip_position",
                                "th_tip_position"};
    fingertip_count_ = is_aero_ ? 5 : 4;
    for (int i = 0; i < fingertip_count_; ++i) {
      fingertip_sensor_adrs_[i] =
          SensorAdr(is_aero_ ? aero_tips[i] : leap_tips[i]);
    }
    if (is_aero_) {
      // NOLINTNEXTLINE(modernize-avoid-c-arrays)
      const char* tendon_sensors[kAeroTendonSensorCount] = {
          "len_if", "len_mf", "len_rf", "len_pf", "len_th1", "len_th2"};
      // NOLINTNEXTLINE(modernize-avoid-c-arrays)
      const char* joint_sensors[kAeroJointSensorCount] = {"len_th_abd"};
      for (int i = 0; i < kAeroTendonSensorCount; ++i) {
        aero_tendon_sensor_adrs_[i] = SensorAdr(tendon_sensors[i]);
      }
      for (int i = 0; i < kAeroJointSensorCount; ++i) {
        aero_joint_sensor_adrs_[i] = SensorAdr(joint_sensors[i]);
      }
    }

    const int key_id = RequireId(mjOBJ_KEY, "home");
    init_qpos_.assign(model_->key_qpos + key_id * model_->nq,
                      model_->key_qpos + (key_id + 1) * model_->nq);
    init_ctrl_.assign(model_->key_ctrl + key_id * model_->nu,
                      model_->key_ctrl + (key_id + 1) * model_->nu);
    init_mpos_.assign(model_->key_mpos + key_id * model_->nmocap * 3,
                      model_->key_mpos + (key_id + 1) * model_->nmocap * 3);
    init_mquat_.assign(model_->key_mquat + key_id * model_->nmocap * 4,
                       model_->key_mquat + (key_id + 1) * model_->nmocap * 4);
    for (int i = 0; i < kHandQDim; ++i) {
      default_pose_[i] = init_qpos_[hand_qposadr_[i]];
      if (is_aero_) {
        const int joint_id = model_->dof_jntid[hand_dofadr_[i]];
        lowers_[i] = model_->jnt_range[2 * joint_id];
        uppers_[i] = model_->jnt_range[2 * joint_id + 1];
      } else {
        lowers_[i] = model_->actuator_ctrlrange[2 * i];
        uppers_[i] = model_->actuator_ctrlrange[2 * i + 1];
      }
    }
    for (int i = 0; i < action_dim_; ++i) {
      motor_targets_[i] = init_ctrl_[i];
    }
  }

  void RestoreHomeMocap() {
    for (int i = 0; i < model_->nmocap * 3; ++i) {
      data_->mocap_pos[i] =
          i < static_cast<int>(init_mpos_.size()) ? init_mpos_[i] : 0.0;
    }
    for (int i = 0; i < model_->nmocap * 4; ++i) {
      if (i < static_cast<int>(init_mquat_.size())) {
        data_->mocap_quat[i] = init_mquat_[i];
      } else {
        data_->mocap_quat[i] = i % 4 == 0 ? 1.0 : 0.0;
      }
    }
  }

  void ResetRewardInfo() {
    reward_angvel_ = 0.0;
    reward_linvel_ = 0.0;
    reward_pose_ = 0.0;
    reward_torques_ = 0.0;
    reward_energy_ = 0.0;
    reward_termination_ = 0.0;
    reward_action_rate_ = 0.0;
    reward_orientation_ = 0.0;
    reward_position_ = 0.0;
    reward_hand_pose_ = 0.0;
    reward_joint_vel_ = 0.0;
    reward_success_ = 0.0;
  }

  const mjtNum* CubePos() const {
    return data_->sensordata + cube_position_sensor_adr_;
  }

  const mjtNum* CubeQuat() const {
    return data_->sensordata + cube_orientation_sensor_adr_;
  }

  const mjtNum* CubeLinvel() const {
    return data_->sensordata + cube_linvel_sensor_adr_;
  }

  const mjtNum* CubeAngvel() const {
    return data_->sensordata + cube_angvel_sensor_adr_;
  }

  const mjtNum* PalmPos() const {
    return data_->sensordata + palm_position_sensor_adr_;
  }

  const mjtNum* GoalQuat() const {
    return data_->sensordata + cube_goal_orientation_sensor_adr_;
  }

  static mjtNum Norm3(const mjtNum* value) {
    return std::sqrt(value[0] * value[0] + value[1] * value[1] +
                     value[2] * value[2]);
  }

  static mjtNum Norm3(const std::array<mjtNum, 3>& value) {
    return std::sqrt(value[0] * value[0] + value[1] * value[1] +
                     value[2] * value[2]);
  }

  static mjtNum L1Norm3(const mjtNum* value) {
    return std::abs(value[0]) + std::abs(value[1]) + std::abs(value[2]);
  }

  static mjtNum F32(mjtNum value) {
    return static_cast<mjtNum>(static_cast<float>(value));
  }

  static mjtNum SumSquares(const std::array<mjtNum, kHandMaxActionDim>& lhs,
                           const std::array<mjtNum, kHandMaxActionDim>& rhs,
                           int count) {
    mjtNum sum = 0.0;
    for (int i = 0; i < count; ++i) {
      const mjtNum d = lhs[i] - rhs[i];
      sum += d * d;
    }
    return sum;
  }

  static mjtNum ToleranceLinear(mjtNum x, mjtNum lower, mjtNum upper,
                                mjtNum margin) {
    if (lower <= x && x <= upper) {
      return 1.0;
    }
    const mjtNum d = (x < lower ? lower - x : x - upper) / margin;
    const mjtNum scaled = d * 0.9;
    return std::abs(scaled) < 1.0 ? 1.0 - scaled : 0.0;
  }

  static void RollHistory(std::vector<mjtNum>* history, int frame_dim,
                          const mjtNum* value) {
    for (int i = static_cast<int>(history->size()) - 1; i >= frame_dim; --i) {
      (*history)[i] = (*history)[i - frame_dim];
    }
    for (int i = 0; i < frame_dim; ++i) {
      (*history)[i] = value[i];
    }
  }

  mjtNum OrientationError() const {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum inv_goal[4];
    mju_negQuat(inv_goal, GoalQuat());
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum quat_diff[4];
    mju_mulQuat(quat_diff, CubeQuat(), inv_goal);
    mju_normalize4(quat_diff);
    const mjtNum vec_norm =
        std::sqrt(quat_diff[1] * quat_diff[1] + quat_diff[2] * quat_diff[2] +
                  quat_diff[3] * quat_diff[3]);
    return 2.0 * std::asin(std::min<mjtNum>(vec_norm, 1.0));
  }

  void QuatDiffMatTail(mjtNum* out) const {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum inv_goal[4];
    mju_negQuat(inv_goal, GoalQuat());
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum quat_diff[4];
    mju_mulQuat(quat_diff, CubeQuat(), inv_goal);
    mju_normalize4(quat_diff);
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum mat[9];
    mju_quat2Mat(mat, quat_diff);
    for (int i = 0; i < 6; ++i) {
      out[i] = mat[i + 3];
    }
  }

  bool HasNaN() const {
    return std::any_of(data_->qpos, data_->qpos + model_->nq,
                       [](mjtNum q) { return std::isnan(q); }) ||
           std::any_of(data_->qvel, data_->qvel + model_->nv,
                       [](mjtNum qvel) { return std::isnan(qvel); });
  }

  bool Termination() const { return CubePos()[2] < -0.05 || HasNaN(); }

  void ApplyRotateControl(const std::array<mjtNum, kHandMaxActionDim>& action) {
    if (is_aero_) {
      for (int i = 0; i < action_dim_; ++i) {
        motor_targets_[i] = init_ctrl_[i] + action[i] * aero_action_scale_[i];
        data_->ctrl[i] = motor_targets_[i];
      }
      return;
    }
    const mjtNum action_scale = spec_.config["action_scale"_];
    for (int i = 0; i < action_dim_; ++i) {
      motor_targets_[i] = default_pose_[i] + action[i] * action_scale;
      data_->ctrl[i] = motor_targets_[i];
    }
  }

  void ApplyReorientControl(
      const std::array<mjtNum, kHandMaxActionDim>& action) {
    if (spec_.config["pert_enable"_] == 0.0) {
      std::fill(data_->xfrc_applied, data_->xfrc_applied + model_->nbody * 6,
                0.0);
    }
    std::array<mjtNum, kHandMaxActionDim> target{};
    const mjtNum action_scale = spec_.config["action_scale"_];
    const mjtNum ema_alpha = spec_.config["ema_alpha"_];
    for (int i = 0; i < action_dim_; ++i) {
      target[i] = std::clamp(data_->ctrl[i] + action[i] * action_scale,
                             lowers_[i], uppers_[i]);
      target[i] = ema_alpha * target[i] + (1.0 - ema_alpha) * motor_targets_[i];
      motor_targets_[i] = target[i];
      data_->ctrl[i] = target[i];
    }
  }

  void StepRotate(const std::array<mjtNum, kHandMaxActionDim>& action) {
    success_ = 0.0;
    ApplyRotateControl(action);
    for (int i = 0; i < n_substeps_; ++i) {
      mj_step(model_, data_);
    }
    FillRotateObs();
    terminated_ = Termination();
    ComputeRotateReward(action, terminated_);
    mjtNum reward =
        (reward_angvel_ + reward_linvel_ + reward_pose_ + reward_torques_ +
         reward_energy_ + reward_termination_ + reward_action_rate_) *
        spec_.config["ctrl_dt"_];
    last_last_act_ = last_act_;
    last_act_ = action;
    for (int i = 0; i < 3; ++i) {
      last_cube_angvel_[i] = CubeAngvel()[i];
    }
    ++elapsed_step_;
    done_ = terminated_ || elapsed_step_ >= max_episode_steps_;
    WriteState(static_cast<float>(reward), false);
  }

  void StepReorient(const std::array<mjtNum, kHandMaxActionDim>& action) {
    ApplyReorientControl(action);
    for (int i = 0; i < n_substeps_; ++i) {
      mj_step(model_, data_);
    }
    const bool success =
        OrientationError() < spec_.config["success_threshold"_];
    success_ = success ? 1.0 : 0.0;
    steps_since_last_success_ = success ? 0 : (steps_since_last_success_ + 1);
    if (success) {
      ++success_count_;
    }
    terminated_ = Termination();
    FillReorientObs();
    ComputeReorientReward(action, terminated_);
    mjtNum reward = (reward_orientation_ + reward_position_ +
                     reward_termination_ + reward_hand_pose_ +
                     reward_action_rate_ + reward_joint_vel_ + reward_energy_) *
                    spec_.config["ctrl_dt"_];
    reward_success_ = success ? 1.0 : 0.0;
    reward += reward_success_ * spec_.config["success_reward"_];
    UpdateGoalQuat(success);
    ++step_;
    last_last_act_ = last_act_;
    last_act_ = action;
    ++elapsed_step_;
    done_ = terminated_ || elapsed_step_ >= max_episode_steps_;
    WriteState(static_cast<float>(reward), false);
  }

  void ComputeRotateReward(const std::array<mjtNum, kHandMaxActionDim>& action,
                           bool done) {
    const mjtNum* cube_pos = CubePos();
    const mjtNum* palm_pos = PalmPos();
    std::array<mjtNum, 3> cube_pos_error = {palm_pos[0] - cube_pos[0],
                                            palm_pos[1] - cube_pos[1],
                                            palm_pos[2] - cube_pos[2]};
    (void)cube_pos_error;
    reward_angvel_ = CubeAngvel()[2] * spec_.config["angvel_scale"_];
    reward_linvel_ = L1Norm3(CubeLinvel()) * spec_.config["linvel_scale"_];
    reward_termination_ =
        (done ? 1.0 : 0.0) * spec_.config["termination_scale"_];
    reward_action_rate_ = SumSquares(action, last_act_, action_dim_) *
                          spec_.config["action_rate_scale"_];
    mjtNum pose = 0.0;
    for (int i = 0; i < kHandQDim; ++i) {
      const mjtNum d = data_->qpos[hand_qposadr_[i]] - default_pose_[i];
      pose += d * d;
    }
    reward_pose_ = pose * spec_.config["pose_scale"_];
    mjtNum torques = 0.0;
    for (int i = 0; i < action_dim_; ++i) {
      torques += data_->actuator_force[i] * data_->actuator_force[i];
    }
    reward_torques_ = torques * spec_.config["torques_scale"_];
    mjtNum energy = 0.0;
    for (int i = 0; i < kHandQDim; ++i) {
      const mjtNum force = is_aero_ ? data_->qfrc_actuator[hand_dofadr_[i]]
                                    : data_->actuator_force[i];
      energy += std::abs(data_->qvel[hand_dofadr_[i]]) * std::abs(force);
    }
    reward_energy_ = energy * spec_.config["energy_scale"_];
  }

  void ComputeReorientReward(
      const std::array<mjtNum, kHandMaxActionDim>& action, bool done) {
    std::array<mjtNum, 3> cube_pos_error = {PalmPos()[0] - CubePos()[0],
                                            PalmPos()[1] - CubePos()[1],
                                            PalmPos()[2] - CubePos()[2]};
    const mjtNum cube_pose_mse = Norm3(cube_pos_error);
    reward_position_ = ToleranceLinear(cube_pose_mse, 0.0, 0.02, 0.05) *
                       spec_.config["position_scale"_];
    reward_orientation_ = ToleranceLinear(OrientationError(), 0.0, 0.2, M_PI) *
                          spec_.config["orientation_scale"_];
    reward_termination_ =
        (done ? 1.0 : 0.0) * spec_.config["termination_scale"_];
    mjtNum hand_pose = 0.0;
    for (int i = 0; i < kHandQDim; ++i) {
      const mjtNum d = data_->qpos[hand_qposadr_[i]] - default_pose_[i];
      hand_pose += d * d;
    }
    reward_hand_pose_ = hand_pose * spec_.config["hand_pose_scale"_];
    mjtNum c1 = 0.0;
    mjtNum c2 = 0.0;
    for (int i = 0; i < action_dim_; ++i) {
      const mjtNum d1 = action[i] - last_act_[i];
      const mjtNum d2 = action[i] - 2.0 * last_act_[i] + last_last_act_[i];
      c1 += d1 * d1;
      c2 += d2 * d2;
    }
    reward_action_rate_ = (c1 + c2) * spec_.config["action_rate_scale"_];
    mjtNum joint_vel = 0.0;
    static constexpr mjtNum k_max_velocity = 5.0;
    static constexpr mjtNum k_vel_tolerance = 1.0;
    for (int i = 0; i < kHandQDim; ++i) {
      const mjtNum scaled =
          data_->qvel[hand_dofadr_[i]] / (k_max_velocity - k_vel_tolerance);
      joint_vel += scaled * scaled;
    }
    reward_joint_vel_ = joint_vel * spec_.config["joint_vel_scale"_];
    mjtNum energy = 0.0;
    for (int i = 0; i < kHandQDim; ++i) {
      energy += std::abs(data_->qvel[hand_dofadr_[i]]) *
                std::abs(data_->actuator_force[i]);
    }
    reward_energy_ = energy * spec_.config["energy_scale"_];
  }

  void UpdateGoalQuat(bool success) {
    if (success) {
      for (int i = 0; i < 3; ++i) {
        goal_quat_dquat_[i] = 3.0 + Uniform(-2.0, 2.0);
      }
    } else {
      for (mjtNum& value : goal_quat_dquat_) {
        value *= 0.8;
      }
    }
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum quat[4];
    std::copy(data_->mocap_quat + 4 * goal_mocap_id_,
              data_->mocap_quat + 4 * goal_mocap_id_ + 4, quat);
    mju_quatIntegrate(quat, goal_quat_dquat_.data(),
                      2.0 * spec_.config["ctrl_dt"_]);
    for (int i = 0; i < 4; ++i) {
      data_->mocap_quat[4 * goal_mocap_id_ + i] = quat[i];
    }
  }

  void FillRotateObs() {
    std::array<mjtNum, kLeapRotateStateDim> current{};
    int index = 0;
    if (is_aero_) {
      for (int i = 0; i < kAeroTendonSensorCount; ++i) {
        current[index++] = F32(data_->sensordata[aero_tendon_sensor_adrs_[i]]) +
                           NoiseUniform(spec_.config["noise_tendon_length"_]);
      }
      for (int i = 0; i < kAeroJointSensorCount; ++i) {
        current[index++] = F32(data_->sensordata[aero_joint_sensor_adrs_[i]]) +
                           NoiseUniform(spec_.config["noise_joint_pos"_]);
      }
      for (int i = 0; i < action_dim_; ++i) {
        current[index++] = last_act_[i];
      }
      RollHistory(&obs_history_, kAeroRotateStateDim, current.data());
      FillAeroRotatePrivileged(current.data());
      std::copy(obs_history_.begin(), obs_history_.end(), obs_.begin());
      return;
    }
    for (int i = 0; i < kHandQDim; ++i) {
      current[index++] = data_->qpos[hand_qposadr_[i]] +
                         NoiseUniform(spec_.config["noise_joint_pos"_]);
    }
    for (int i = 0; i < action_dim_; ++i) {
      current[index++] = last_act_[i];
    }
    RollHistory(&obs_history_, kLeapRotateStateDim, current.data());
    FillLeapRotatePrivileged(current.data());
    std::copy(obs_history_.begin(), obs_history_.end(), obs_.begin());
  }

  mjtNum NoiseUniform(mjtNum scale) {
    return (2.0 * unit_uniform_(gen_) - 1.0) * spec_.config["noise_level"_] *
           scale;
  }

  void FillLeapRotatePrivileged(const mjtNum* current_state) {
    int p = 0;
    for (int i = 0; i < state_dim_; ++i) {
      privileged_obs_[p++] = obs_history_[i];
    }
    for (int i = 0; i < kHandQDim; ++i) {
      privileged_obs_[p++] = data_->qpos[hand_qposadr_[i]];
    }
    for (int i = 0; i < kHandQDim; ++i) {
      privileged_obs_[p++] = data_->qvel[hand_dofadr_[i]];
    }
    for (int i = 0; i < action_dim_; ++i) {
      privileged_obs_[p++] = data_->actuator_force[i];
    }
    FillFingertips(&p);
    FillCubeCommonPrivileged(&p);
    (void)current_state;
  }

  void FillAeroRotatePrivileged(const mjtNum* current_state) {
    int p = 0;
    for (int i = 0; i < state_dim_; ++i) {
      privileged_obs_[p++] = obs_history_[i];
    }
    for (int i = 0; i < kHandQDim; ++i) {
      privileged_obs_[p++] = data_->qpos[hand_qposadr_[i]];
    }
    for (int i = 0; i < kHandQDim; ++i) {
      privileged_obs_[p++] = data_->qvel[hand_dofadr_[i]];
    }
    for (int i = 0; i < action_dim_; ++i) {
      privileged_obs_[p++] = data_->actuator_force[i];
    }
    FillFingertips(&p);
    FillCubeCommonPrivileged(&p);
    (void)current_state;
  }

  void FillReorientObs() {
    std::array<mjtNum, kLeapReorientStateDim> state{};
    std::array<mjtNum, kHandQDim> noisy_joint{};
    for (int i = 0; i < kHandQDim; ++i) {
      noisy_joint[i] = data_->qpos[hand_qposadr_[i]] +
                       NoiseUniform(spec_.config["noise_joint_pos"_]);
    }
    std::array<mjtNum, kHandQDim> qpos_error{};
    for (int i = 0; i < kHandQDim; ++i) {
      qpos_error[i] = noisy_joint[i] - motor_targets_[i];
    }
    RollHistory(&qpos_error_history_, kHandQDim, qpos_error.data());

    std::array<mjtNum, 3> noisy_cube_pos = {
        CubePos()[0] + NoiseUniform(spec_.config["noise_cube_pos"_]),
        CubePos()[1] + NoiseUniform(spec_.config["noise_cube_pos"_]),
        CubePos()[2] + NoiseUniform(spec_.config["noise_cube_pos"_])};
    std::array<mjtNum, 4> noisy_cube_quat = {CubeQuat()[0], CubeQuat()[1],
                                             CubeQuat()[2], CubeQuat()[3]};
    if (spec_.config["noise_level"_] != 0.0 &&
        spec_.config["noise_cube_ori"_] != 0.0) {
      for (int i = 0; i < 4; ++i) {
        noisy_cube_quat[i] += normal_(gen_) * spec_.config["noise_level"_] *
                              spec_.config["noise_cube_ori"_];
      }
      mju_normalize4(noisy_cube_quat.data());
    }
    if (spec_.config["random_ori_injection_prob"_] > 0.0 &&
        unit_uniform_(gen_) < spec_.config["noise_level"_] *
                                  spec_.config["random_ori_injection_prob"_]) {
      const auto random_quat = RandomQuat();
      noisy_cube_pos = {Uniform(-0.5, 0.5), Uniform(-0.5, 0.5),
                        Uniform(-0.5, 0.5)};
      noisy_cube_quat = random_quat;
    }

    std::array<mjtNum, 3> cube_pos_error = {PalmPos()[0] - noisy_cube_pos[0],
                                            PalmPos()[1] - noisy_cube_pos[1],
                                            PalmPos()[2] - noisy_cube_pos[2]};
    RollHistory(&cube_pos_error_history_, 3, cube_pos_error.data());

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum inv_goal[4];
    mju_negQuat(inv_goal, GoalQuat());
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum quat_diff[4];
    mju_mulQuat(quat_diff, noisy_cube_quat.data(), inv_goal);
    mju_normalize4(quat_diff);
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum mat[9];
    mju_quat2Mat(mat, quat_diff);
    std::array<mjtNum, 6> xmat_diff{};
    for (int i = 0; i < 6; ++i) {
      xmat_diff[i] = mat[i + 3];
    }
    RollHistory(&cube_ori_error_history_, 6, xmat_diff.data());

    int index = 0;
    for (int i = 0; i < kHandQDim; ++i) {
      state[index++] = noisy_joint[i];
    }
    for (mjtNum value : qpos_error_history_) {
      state[index++] = value;
    }
    for (mjtNum value : cube_pos_error_history_) {
      state[index++] = value;
    }
    for (mjtNum value : cube_ori_error_history_) {
      state[index++] = value;
    }
    for (int i = 0; i < action_dim_; ++i) {
      state[index++] = last_act_[i];
    }
    std::copy(state.begin(), state.begin() + state_dim_, obs_.begin());
    FillReorientPrivileged(state.data());
  }

  void FillReorientPrivileged(const mjtNum* state) {
    int p = 0;
    for (int i = 0; i < state_dim_; ++i) {
      privileged_obs_[p++] = state[i];
    }
    for (int i = 0; i < kHandQDim; ++i) {
      privileged_obs_[p++] = data_->qpos[hand_qposadr_[i]];
    }
    for (int i = 0; i < kHandQDim; ++i) {
      privileged_obs_[p++] = data_->qvel[hand_dofadr_[i]];
    }
    FillFingertips(&p);
    for (int i = 0; i < 3; ++i) {
      privileged_obs_[p++] = PalmPos()[i] - CubePos()[i];
    }
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum xmat_diff[6];
    QuatDiffMatTail(xmat_diff);
    for (mjtNum value : xmat_diff) {
      privileged_obs_[p++] = value;
    }
    for (int i = 0; i < 3; ++i) {
      privileged_obs_[p++] = CubeLinvel()[i];
    }
    for (int i = 0; i < 3; ++i) {
      privileged_obs_[p++] = CubeAngvel()[i];
    }
    for (int i = 0; i < 6; ++i) {
      privileged_obs_[p++] = pert_dir_[i];
    }
    for (int i = 0; i < 6; ++i) {
      privileged_obs_[p++] = data_->xfrc_applied[cube_body_id_ * 6 + i];
    }
  }

  void FillFingertips(int* p) {
    for (int i = 0; i < fingertip_count_; ++i) {
      const mjtNum* tip = data_->sensordata + fingertip_sensor_adrs_[i];
      for (int j = 0; j < 3; ++j) {
        privileged_obs_[(*p)++] = tip[j];
      }
    }
  }

  void FillCubeCommonPrivileged(int* p) {
    for (int i = 0; i < 3; ++i) {
      privileged_obs_[(*p)++] = PalmPos()[i] - CubePos()[i];
    }
    for (int i = 0; i < 4; ++i) {
      privileged_obs_[(*p)++] = CubeQuat()[i];
    }
    for (int i = 0; i < 3; ++i) {
      privileged_obs_[(*p)++] = CubeAngvel()[i];
    }
    for (int i = 0; i < 3; ++i) {
      privileged_obs_[(*p)++] = CubeLinvel()[i];
    }
  }

  template <std::size_t N>
  static void CopyPadded(const mjtNum* src, int count,
                         std::array<mjtNum, N>* dst) {
    dst->fill(0.0);
    const int n = std::min<int>(count, static_cast<int>(N));
    std::copy(src, src + n, dst->begin());
  }

  template <std::size_t N>
  static void CopyVectorPadded(const std::vector<mjtNum>& src,
                               std::array<mjtNum, N>* dst) {
    dst->fill(0.0);
    const int n = std::min<int>(src.size(), static_cast<int>(N));
    std::copy(src.begin(), src.begin() + n, dst->begin());
  }

  void WriteState(float reward, bool reset) {
    auto state = Allocate();
    state["reward"_] = reward;
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      auto obs_state = state["obs:state"_];
      mjtNum* obs_state_ptr = PrepareObservation("obs:state", &obs_state);
      std::copy(obs_.begin(), obs_.begin() + state_dim_, obs_state_ptr);
      CommitObservation("obs:state", &obs_state, reset);
      auto obs_privileged = state["obs:privileged_state"_];
      mjtNum* privileged =
          PrepareObservation("obs:privileged_state", &obs_privileged);
      std::copy(privileged_obs_.begin(),
                privileged_obs_.begin() + privileged_dim_, privileged);
      CommitObservation("obs:privileged_state", &obs_privileged, reset);
    }
    state["info:terminated"_] = terminated_;
    state["info:success"_] = success_;
    state["info:steps_since_last_success"_] = steps_since_last_success_;
    state["info:success_count"_] = success_count_;
    state["info:reward_angvel"_] = reward_angvel_;
    state["info:reward_linvel"_] = reward_linvel_;
    state["info:reward_pose"_] = reward_pose_;
    state["info:reward_torques"_] = reward_torques_;
    state["info:reward_energy"_] = reward_energy_;
    state["info:reward_termination"_] = reward_termination_;
    state["info:reward_action_rate"_] = reward_action_rate_;
    state["info:reward_orientation"_] = reward_orientation_;
    state["info:reward_position"_] = reward_position_;
    state["info:reward_hand_pose"_] = reward_hand_pose_;
    state["info:reward_joint_vel"_] = reward_joint_vel_;
    state["info:reward_success"_] = reward_success_;
#ifdef ENVPOOL_TEST
    std::array<mjtNum, 64> pad64{};
    CopyPadded(data_->qpos, model_->nq, &pad64);
    state["info:qpos"_].Assign(pad64.data(), pad64.size());
    CopyPadded(data_->qvel, model_->nv, &pad64);
    state["info:qvel"_].Assign(pad64.data(), pad64.size());
    CopyPadded(data_->ctrl, model_->nu, &pad64);
    state["info:ctrl"_].Assign(pad64.data(), pad64.size());
    CopyPadded(data_->qacc, model_->nv, &pad64);
    state["info:qacc"_].Assign(pad64.data(), pad64.size());
    CopyPadded(data_->qacc_warmstart, model_->nv, &pad64);
    state["info:qacc_warmstart"_].Assign(pad64.data(), pad64.size());
    CopyPadded(data_->mocap_pos, model_->nmocap * 3, &pad64);
    state["info:mocap_pos"_].Assign(pad64.data(), pad64.size());
    CopyPadded(data_->mocap_quat, model_->nmocap * 4, &pad64);
    state["info:mocap_quat"_].Assign(pad64.data(), pad64.size());
    CopyPadded(data_->actuator_force, model_->nu, &pad64);
    state["info:actuator_force"_].Assign(pad64.data(), pad64.size());
    CopyPadded(data_->qfrc_actuator, model_->nv, &pad64);
    state["info:qfrc_actuator"_].Assign(pad64.data(), pad64.size());
    state["info:last_act"_].Assign(last_act_.data(), last_act_.size());
    state["info:last_last_act"_].Assign(last_last_act_.data(),
                                        last_last_act_.size());
    state["info:motor_targets"_].Assign(motor_targets_.data(),
                                        motor_targets_.size());
    state["info:last_cube_angvel"_].Assign(last_cube_angvel_.data(),
                                           last_cube_angvel_.size());
    std::array<mjtNum, 128> pad128{};
    CopyVectorPadded(obs_history_, &pad128);
    state["info:obs_history"_].Assign(pad128.data(), pad128.size());
    CopyVectorPadded(qpos_error_history_, &pad64);
    state["info:qpos_error_history"_].Assign(pad64.data(), pad64.size());
    std::array<mjtNum, 32> pad32{};
    CopyVectorPadded(cube_pos_error_history_, &pad32);
    state["info:cube_pos_error_history"_].Assign(pad32.data(), pad32.size());
    CopyVectorPadded(cube_ori_error_history_, &pad64);
    state["info:cube_ori_error_history"_].Assign(pad64.data(), pad64.size());
    state["info:goal_quat_dquat"_].Assign(goal_quat_dquat_.data(),
                                          goal_quat_dquat_.size());
    state["info:pert_dir"_].Assign(pert_dir_.data(), pert_dir_.size());
    state["info:step"_] = step_;
    std::array<mjtNum, 512> pad512{};
    CopyPadded(data_->xfrc_applied, model_->nbody * 6, &pad512);
    state["info:xfrc_applied"_].Assign(pad512.data(), pad512.size());
    CopyPadded(data_->xpos, model_->nbody * 3, &pad512);
    state["info:xpos"_].Assign(pad512.data(), pad512.size());
    CopyPadded(data_->xquat, model_->nbody * 4, &pad512);
    state["info:xquat"_].Assign(pad512.data(), pad512.size());
    CopyPadded(data_->site_xpos, model_->nsite * 3, &pad512);
    state["info:site_xpos"_].Assign(pad512.data(), pad512.size());
    CopyPadded(data_->sensordata, model_->nsensordata, &pad512);
    state["info:sensordata"_].Assign(pad512.data(), pad512.size());
    std::array<mjtNum, 2048> pad2048{};
    CopyPadded(data_->xmat, model_->nbody * 9, &pad2048);
    state["info:xmat"_].Assign(pad2048.data(), pad2048.size());
    CopyPadded(data_->site_xmat, model_->nsite * 9, &pad2048);
    state["info:site_xmat"_].Assign(pad2048.data(), pad2048.size());
#endif
  }
};

using PlaygroundHandEnv = PlaygroundHandEnvBase<PlaygroundHandEnvSpec, false>;
using PlaygroundHandPixelEnv =
    PlaygroundHandEnvBase<PlaygroundHandPixelEnvSpec, true>;
using PlaygroundHandEnvPool = AsyncEnvPool<PlaygroundHandEnv>;
using PlaygroundHandPixelEnvPool = AsyncEnvPool<PlaygroundHandPixelEnv>;

}  // namespace mujoco_playground

#endif  // ENVPOOL_MUJOCO_PLAYGROUND_HAND_H_
