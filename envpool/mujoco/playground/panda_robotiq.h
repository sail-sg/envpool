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

#ifndef ENVPOOL_MUJOCO_PLAYGROUND_PANDA_ROBOTIQ_H_
#define ENVPOOL_MUJOCO_PLAYGROUND_PANDA_ROBOTIQ_H_

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

constexpr int kPandaRobotiqActionDim = 7;
constexpr int kPandaRobotiqCtrlDim = 8;
constexpr int kPandaRobotiqObsDim = 48;
constexpr int kPandaRobotiqArmJoints = 7;
constexpr int kPandaRobotiqNormalSensors = 6;
constexpr int kPandaRobotiqContactSensors = 3;

class PlaygroundPandaRobotiqEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "frame_stack"_.Bind(1),
        "task_name"_.Bind(std::string("PandaRobotiqPushCube")),
        "ctrl_dt"_.Bind(0.005), "sim_dt"_.Bind(0.005),
        "action_scale"_.Bind(0.1), "action_history_len"_.Bind(5),
        "obs_history_len"_.Bind(30), "action_min_delay"_.Bind(1),
        "action_max_delay"_.Bind(3), "obs_min_delay"_.Bind(6),
        "obs_max_delay"_.Bind(12), "noise_obj_pos"_.Bind(0.015),
        "noise_obj_angle"_.Bind(7.5), "noise_robot_qpos"_.Bind(0.1),
        "noise_robot_qvel"_.Bind(0.1), "noise_eef_pos"_.Bind(0.02),
        "noise_eef_angle"_.Bind(5.0), "termination_reward"_.Bind(-50.0),
        "success_reward"_.Bind(500.0), "success_wait_reward"_.Bind(3.0),
        "success_step_count"_.Bind(30), "gripper_box_scale"_.Bind(2.0),
        "box_target_scale"_.Bind(8.0), "box_orientation_scale"_.Bind(6.0),
        "gripper_collision_side_scale"_.Bind(1.0),
        "robot_target_qpos_scale"_.Bind(0.75), "joint_vel_scale"_.Bind(1.0),
        "joint_vel_limit_scale"_.Bind(3.0), "total_command_scale"_.Bind(-0.1),
        "action_rate_scale"_.Bind(-0.1));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs:state"_.Bind(
            StackSpec(Spec<mjtNum>({kPandaRobotiqObsDim}, {-inf, inf}),
                      conf["frame_stack"_])),
        "info:terminated"_.Bind(Spec<bool>({-1})),
        "info:out_of_bounds"_.Bind(Spec<mjtNum>({-1})),
        "info:success"_.Bind(Spec<mjtNum>({-1})),
        "info:success_1"_.Bind(Spec<mjtNum>({-1})),
        "info:success_2"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_gripper_box"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_box_target"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_box_orientation"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_gripper_collision_side"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_robot_target_qpos"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_joint_vel"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_joint_vel_limit"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_total_command"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_action_rate"_.Bind(Spec<mjtNum>({-1}))
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
        "info:last_action"_.Bind(Spec<mjtNum>({kPandaRobotiqActionDim})),
        "info:action_history"_.Bind(Spec<mjtNum>({64})),
        "info:obs_history"_.Bind(Spec<mjtNum>({2048})),
        "info:success_step_count"_.Bind(Spec<int>({-1})),
        "info:prev_step_success"_.Bind(Spec<int>({-1})),
        "info:curriculum_id"_.Bind(Spec<int>({-1}))
#endif
    );  // NOLINT
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(
        Spec<mjtNum>({-1, kPandaRobotiqActionDim}, {-1.0, 1.0})));
  }
};

using PandaRobotiqAliases = PlaygroundEnvAliases<PlaygroundPandaRobotiqEnvFns>;
using PlaygroundPandaRobotiqEnvSpec = PandaRobotiqAliases::Spec;
using PlaygroundPandaRobotiqPixelEnvFns = PandaRobotiqAliases::PixelFns;
using PlaygroundPandaRobotiqPixelEnvSpec = PandaRobotiqAliases::PixelSpec;

template <typename EnvSpecT, bool kFromPixels>
class PlaygroundPandaRobotiqEnvBase : public Env<EnvSpecT>,
                                      public PlaygroundMujocoEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  int n_substeps_{1};
  std::vector<mjtNum> init_qpos_;
  std::vector<mjtNum> init_ctrl_;
  std::array<int, kPandaRobotiqArmJoints> arm_qposadr_{};
  std::array<mjtNum, 3> init_obj_pos_{};
  std::array<mjtNum, 4> init_obj_quat_{};
  std::array<mjtNum, kPandaRobotiqCtrlDim> lowers_{};
  std::array<mjtNum, kPandaRobotiqCtrlDim> uppers_{};
  std::array<mjtNum, kPandaRobotiqActionDim> last_action_{};
  std::array<mjtNum, kPandaRobotiqObsDim> obs_{};
  std::vector<mjtNum> action_history_;
  std::vector<mjtNum> obs_history_;
  int gripper_site_id_{-1};
  int obj_body_id_{-1};
  int obj_qposadr_{-1};
  int mocap_target_id_{-1};
  int wall_geom_id_{-1};
  int floor_geom_id_{-1};
  std::array<int, kPandaRobotiqNormalSensors> normal_sensor_adrs_{};
  std::array<int, kPandaRobotiqNormalSensors> normal_sensor_dims_{};
  std::array<int, kPandaRobotiqContactSensors> wall_sensor_adrs_{};
  std::array<int, kPandaRobotiqContactSensors> floor_sensor_adrs_{};
  int success_step_count_{0};
  int prev_step_success_{0};
  int curriculum_id_{0};
  mjtNum out_of_bounds_{0.0};
  mjtNum success_{0.0};
  mjtNum success_1_{0.0};
  mjtNum success_2_{0.0};
  mjtNum reward_gripper_box_{0.0};
  mjtNum reward_box_target_{0.0};
  mjtNum reward_box_orientation_{0.0};
  mjtNum reward_gripper_collision_side_{0.0};
  mjtNum reward_robot_target_qpos_{0.0};
  mjtNum reward_joint_vel_{0.0};
  mjtNum reward_joint_vel_limit_{0.0};
  mjtNum reward_total_command_{0.0};
  mjtNum reward_action_rate_{0.0};
  std::uniform_real_distribution<mjtNum> unit_uniform_{0.0, 1.0};
  std::normal_distribution<mjtNum> normal_{0.0, 1.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  PlaygroundPandaRobotiqEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        PlaygroundMujocoEnv(XmlPath(spec.config["base_path"_]),
                            spec.config["max_episode_steps"_],
                            spec.config["frame_stack"_],
                            RenderWidthOrDefault<kFromPixels>(spec.config),
                            RenderHeightOrDefault<kFromPixels>(spec.config),
                            RenderCameraIdOrDefault<kFromPixels>(spec.config),
                            envpool::mujoco::CameraPolicy::kDmControl) {
    if (spec.config["task_name"_] != "PandaRobotiqPushCube") {
      throw std::runtime_error("Unsupported Panda Robotiq task_name.");
    }
    if (model_->nu != kPandaRobotiqCtrlDim) {
      throw std::runtime_error("Unexpected Panda Robotiq ctrl dimension.");
    }
    n_substeps_ = static_cast<int>(
        std::llround(spec.config["ctrl_dt"_] / spec.config["sim_dt"_]));
    model_->opt.timestep = spec.config["sim_dt"_];
    model_->opt.ccd_iterations = 10;
    InitModelIds();
    action_history_.assign(
        spec.config["action_history_len"_] * kPandaRobotiqActionDim, 0.0);
    obs_history_.assign(spec.config["obs_history_len"_] * kPandaRobotiqObsDim,
                        0.0);
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    has_cached_render_ = false;
    done_ = false;
    terminated_ = false;
    elapsed_step_ = 0;
    ResetRewards();
    mj_resetData(model_, data_);
    std::copy(init_qpos_.begin(), init_qpos_.end(), data_->qpos);
    std::fill(data_->qvel, data_->qvel + model_->nv, 0.0);
    std::copy(init_ctrl_.begin(), init_ctrl_.end(), data_->ctrl);
    for (int i = 0; i < model_->nmocap; ++i) {
      data_->mocap_pos[3 * i + 0] = 0.0;
      data_->mocap_pos[3 * i + 1] = 0.0;
      data_->mocap_pos[3 * i + 2] = 0.0;
      data_->mocap_quat[4 * i + 0] = 1.0;
      data_->mocap_quat[4 * i + 1] = 0.0;
      data_->mocap_quat[4 * i + 2] = 0.0;
      data_->mocap_quat[4 * i + 3] = 0.0;
    }

    const auto box_pos = RandomTargetPos(0.15);
    const auto box_quat = RandomTargetQuat(360.0);
    const auto target_pos = RandomTargetPos(0.05);
    const auto target_delta_quat = RandomTargetQuat(45.0);
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum target_quat[4];
    mju_mulQuat(target_quat, box_quat.data(), target_delta_quat.data());
    for (int i = 0; i < 3; ++i) {
      data_->qpos[obj_qposadr_ + i] = box_pos[i];
      data_->mocap_pos[3 * mocap_target_id_ + i] = target_pos[i];
    }
    data_->qpos[obj_qposadr_ + 2] = init_obj_pos_[2];
    for (int i = 0; i < 4; ++i) {
      data_->qpos[obj_qposadr_ + 3 + i] = box_quat[i];
      data_->mocap_quat[4 * mocap_target_id_ + i] = target_quat[i];
    }
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    static constexpr mjtNum k_jnt_range[kPandaRobotiqArmJoints][2] = {
        {-2.8973, 2.8973},  {-1.7628, 1.7628}, {-2.8973, 2.8973},
        {-3.0718, -0.0698}, {-2.8973, 2.8973}, {-0.0175, 3.7525},
        {-2.8973, 2.8973}};
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    static constexpr mjtNum k_init_limit[kPandaRobotiqArmJoints] = {
        0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3};
    for (int i = 0; i < kPandaRobotiqArmJoints; ++i) {
      data_->qpos[arm_qposadr_[i]] +=
          0.3 * Uniform(k_jnt_range[i][0] * k_init_limit[i],
                        k_jnt_range[i][1] * k_init_limit[i]);
    }
    mj_forward(model_, data_);
    last_action_.fill(0.0);
    std::fill(action_history_.begin(), action_history_.end(), 0.0);
    std::fill(obs_history_.begin(), obs_history_.end(), 0.0);
    success_step_count_ = 0;
    prev_step_success_ = 0;
    curriculum_id_ = 0;
    UpdateSingleObs();
    WriteState(0.0, true);
  }

  void Step(const Action& action) override {
    has_cached_render_ = false;
    const auto* act = static_cast<const mjtNum*>(action["action"_].Data());
    std::array<mjtNum, kPandaRobotiqActionDim> raw_action{};
    for (int i = 0; i < kPandaRobotiqActionDim; ++i) {
      raw_action[i] = act[i];
    }
    RollActionHistory(raw_action);
    const int delay = SampleDelay(spec_.config["action_min_delay"_],
                                  spec_.config["action_max_delay"_],
                                  spec_.config["action_history_len"_]);
    std::array<mjtNum, kPandaRobotiqActionDim> delayed_action{};
    for (int i = 0; i < kPandaRobotiqActionDim; ++i) {
      delayed_action[i] = action_history_[delay * kPandaRobotiqActionDim + i];
    }
    ApplyControl(delayed_action);
    for (int i = 0; i < n_substeps_; ++i) {
      mj_step(model_, data_);
    }

    const mjtNum dense_reward = ComputeReward(raw_action);
    const bool termination = Termination();
    mjtNum reward = dense_reward;
    reward += spec_.config["termination_reward"_] * (termination ? 1.0 : 0.0);
    const bool success_wait = UpdateSuccess();
    reward += spec_.config["success_wait_reward"_] * (success_wait ? 1.0 : 0.0);
    reward += spec_.config["success_reward"_] * success_;
    reward *= spec_.config["ctrl_dt"_];
    if (success_ > 0.0) {
      ResetTargetAfterSuccess();
    }

    out_of_bounds_ = termination ? 1.0 : 0.0;
    terminated_ = termination || HasNaN();
    done_ = terminated_;
    UpdateDelayedObs();
    last_action_ = delayed_action;
    ++elapsed_step_;
    done_ = done_ || elapsed_step_ >= max_episode_steps_;
    WriteState(static_cast<float>(reward), false);
  }

 private:
  static std::string XmlPath(const std::string& base_path) {
    return base_path +
           "/mujoco/playground/assets/mujoco_playground/_src/"
           "manipulation/franka_emika_panda_robotiq/xmls/"
           "scene_panda_robotiq_cube.xml";
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

  int SensorDim(const std::string& name) const {
    const int id = RequireId(mjOBJ_SENSOR, name);
    return model_->sensor_dim[id];
  }

  mjtNum Uniform(mjtNum low, mjtNum high) {
    return low + (high - low) * unit_uniform_(gen_);
  }

  int SampleDelay(int min_delay, int max_delay, int history_len) {
    if (max_delay <= min_delay) {
      return std::clamp(min_delay, 0, history_len - 1);
    }
    std::uniform_int_distribution<int> dist(min_delay, max_delay - 1);
    return std::clamp(dist(gen_), 0, history_len - 1);
  }

  void InitModelIds() {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const char* arm_joints[kPandaRobotiqArmJoints] = {
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"};
    for (int i = 0; i < kPandaRobotiqArmJoints; ++i) {
      const int joint_id = RequireId(mjOBJ_JOINT, arm_joints[i]);
      arm_qposadr_[i] = model_->jnt_qposadr[joint_id];
    }
    gripper_site_id_ = RequireId(mjOBJ_SITE, "gripper");
    obj_body_id_ = RequireId(mjOBJ_BODY, "box");
    obj_qposadr_ = model_->jnt_qposadr[model_->body_jntadr[obj_body_id_]];
    const int mocap_body = RequireId(mjOBJ_BODY, "mocap_target");
    mocap_target_id_ = model_->body_mocapid[mocap_body];
    wall_geom_id_ = RequireId(mjOBJ_GEOM, "wall");
    floor_geom_id_ = RequireId(mjOBJ_GEOM, "floor");
    const int key_id = RequireId(mjOBJ_KEY, "home");
    init_qpos_.assign(model_->key_qpos + key_id * model_->nq,
                      model_->key_qpos + (key_id + 1) * model_->nq);
    init_ctrl_.assign(model_->key_ctrl + key_id * model_->nu,
                      model_->key_ctrl + (key_id + 1) * model_->nu);
    for (int i = 0; i < 3; ++i) {
      init_obj_pos_[i] = init_qpos_[obj_qposadr_ + i];
    }
    for (int i = 0; i < 4; ++i) {
      init_obj_quat_[i] = init_qpos_[obj_qposadr_ + 3 + i];
    }
    for (int i = 0; i < kPandaRobotiqCtrlDim; ++i) {
      lowers_[i] = model_->actuator_ctrlrange[2 * i];
      uppers_[i] = model_->actuator_ctrlrange[2 * i + 1];
    }
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const char* normal_sensors[kPandaRobotiqNormalSensors] = {
        "left_coupler_col_1_box_normal",  "left_coupler_col_2_box_normal",
        "left_follower_pad2_box_normal",  "right_coupler_col_1_box_normal",
        "right_coupler_col_2_box_normal", "right_follower_pad2_box_normal"};
    for (int i = 0; i < kPandaRobotiqNormalSensors; ++i) {
      normal_sensor_adrs_[i] = SensorAdr(normal_sensors[i]);
      normal_sensor_dims_[i] = SensorDim(normal_sensors[i]);
    }
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const char* hand_geoms[kPandaRobotiqContactSensors] = {
        "left_finger_pad", "right_finger_pad", "hand_capsule"};
    for (int i = 0; i < kPandaRobotiqContactSensors; ++i) {
      wall_sensor_adrs_[i] =
          SensorAdr(std::string("wall_") + hand_geoms[i] + "_found");
      floor_sensor_adrs_[i] =
          SensorAdr(std::string("floor_") + hand_geoms[i] + "_found");
    }
  }

  void ResetRewards() {
    out_of_bounds_ = 0.0;
    success_ = 0.0;
    success_1_ = 0.0;
    success_2_ = 0.0;
    reward_gripper_box_ = 0.0;
    reward_box_target_ = 0.0;
    reward_box_orientation_ = 0.0;
    reward_gripper_collision_side_ = 0.0;
    reward_robot_target_qpos_ = 0.0;
    reward_joint_vel_ = 0.0;
    reward_joint_vel_limit_ = 0.0;
    reward_total_command_ = 0.0;
    reward_action_rate_ = 0.0;
  }

  std::array<mjtNum, 3> RandomTargetPos(mjtNum offset) {
    std::array<mjtNum, 3> pos = {
        Uniform(init_obj_pos_[0] - offset * 0.4,
                init_obj_pos_[0] + offset * 0.4),
        Uniform(init_obj_pos_[1] - offset, init_obj_pos_[1] + offset),
        Uniform(init_obj_pos_[2] - 0.005, init_obj_pos_[2] + 0.005)};
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    static constexpr mjtNum k_min[3] = {0.4, -0.2, -0.005};
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    static constexpr mjtNum k_max[3] = {0.65, 0.2, 0.04};
    for (int i = 0; i < 3; ++i) {
      pos[i] = std::clamp(pos[i], k_min[i], k_max[i]);
    }
    return pos;
  }

  std::array<mjtNum, 4> RandomTargetQuat(mjtNum max_angle_deg) {
    const mjtNum theta = Uniform(0.0, max_angle_deg * M_PI / 180.0);
    return {std::cos(0.5 * theta), 0.0, 0.0, std::sin(0.5 * theta)};
  }

  void RollActionHistory(
      const std::array<mjtNum, kPandaRobotiqActionDim>& action) {
    for (int i = static_cast<int>(action_history_.size()) - 1;
         i >= kPandaRobotiqActionDim; --i) {
      action_history_[i] = action_history_[i - kPandaRobotiqActionDim];
    }
    for (int i = 0; i < kPandaRobotiqActionDim; ++i) {
      action_history_[i] = action[i];
    }
  }

  void ApplyControl(const std::array<mjtNum, kPandaRobotiqActionDim>& action) {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    static constexpr mjtNum k_gear[kPandaRobotiqActionDim] = {
        150.0, 150.0, 150.0, 150.0, 20.0, 20.0, 20.0};
    const mjtNum action_scale = spec_.config["action_scale"_];
    for (int i = 0; i < kPandaRobotiqActionDim; ++i) {
      const mjtNum torque_limit = 8.0 / k_gear[i];
      data_->ctrl[i] =
          std::clamp(action[i] * action_scale, -torque_limit, torque_limit);
    }
    data_->ctrl[7] = 0.82;
    for (int i = 0; i < kPandaRobotiqCtrlDim; ++i) {
      data_->ctrl[i] = std::clamp(data_->ctrl[i], lowers_[i], uppers_[i]);
    }
  }

  static mjtNum Norm3(const mjtNum* values) {
    return std::sqrt(values[0] * values[0] + values[1] * values[1] +
                     values[2] * values[2]);
  }

  static mjtNum NormArray(const std::array<mjtNum, 3>& values) {
    return std::sqrt(values[0] * values[0] + values[1] * values[1] +
                     values[2] * values[2]);
  }

  static mjtNum NormAction(
      const std::array<mjtNum, kPandaRobotiqActionDim>& action) {
    mjtNum sum = 0.0;
    for (mjtNum value : action) {
      sum += value * value;
    }
    return std::sqrt(sum);
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

  static mjtNum ToleranceReciprocal(mjtNum x, mjtNum lower, mjtNum upper,
                                    mjtNum margin) {
    if (lower <= x && x <= upper) {
      return 1.0;
    }
    const mjtNum d = (x < lower ? lower - x : x - upper) / margin;
    return 1.0 / (std::abs(d) * 9.0 + 1.0);
  }

  mjtNum OrientationError(const mjtNum* object_quat,
                          const mjtNum* target_quat) const {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum inv_target[4];
    mju_negQuat(inv_target, target_quat);
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum quat_diff[4];
    mju_mulQuat(quat_diff, object_quat, inv_target);
    mju_normalize4(quat_diff);
    const mjtNum vec_norm =
        std::sqrt(quat_diff[1] * quat_diff[1] + quat_diff[2] * quat_diff[2] +
                  quat_diff[3] * quat_diff[3]);
    return 2.0 * std::asin(std::min<mjtNum>(vec_norm, 1.0));
  }

  mjtNum ComputeReward(
      const std::array<mjtNum, kPandaRobotiqActionDim>& action) {
    const mjtNum* target_pos = data_->mocap_pos + 3 * mocap_target_id_;
    const mjtNum* target_quat = data_->mocap_quat + 4 * mocap_target_id_;
    const mjtNum* box_pos = data_->xpos + 3 * obj_body_id_;
    const mjtNum* box_quat = data_->xquat + 4 * obj_body_id_;
    std::array<mjtNum, 3> side_dir = {box_pos[0] - target_pos[0],
                                      box_pos[1] - target_pos[1],
                                      box_pos[2] - target_pos[2]};
    const mjtNum side_norm = NormArray(side_dir);
    if (side_norm > 1e-3) {
      for (mjtNum& value : side_dir) {
        value = value / side_norm * 0.1;
      }
    } else {
      side_dir = {0.0, 0.0, 0.0};
    }
    const mjtNum* gripper_pos = data_->site_xpos + 3 * gripper_site_id_;
    const std::array<mjtNum, 3> side_delta = {
        side_dir[0] + box_pos[0] - gripper_pos[0],
        side_dir[1] + box_pos[1] - gripper_pos[1],
        side_dir[2] + box_pos[2] - gripper_pos[2]};
    reward_gripper_box_ = ToleranceLinear(NormArray(side_delta), 0.0, 0.1, 1.0);
    const mjtNum box_xy_dist =
        std::sqrt((box_pos[0] - target_pos[0]) * (box_pos[0] - target_pos[0]) +
                  (box_pos[1] - target_pos[1]) * (box_pos[1] - target_pos[1]));
    reward_box_target_ = ToleranceReciprocal(box_xy_dist, 0.0, 0.005, 0.4);
    reward_box_orientation_ = ToleranceReciprocal(
        OrientationError(box_quat, target_quat), 0.0, 0.2, M_PI);
    std::array<mjtNum, 3> normal = {0.0, 0.0, 0.0};
    for (int i = 0; i < kPandaRobotiqNormalSensors; ++i) {
      for (int j = 0; j < std::min(3, normal_sensor_dims_[i]); ++j) {
        normal[j] += data_->sensordata[normal_sensor_adrs_[i] + j];
      }
    }
    for (mjtNum& value : normal) {
      value /= kPandaRobotiqNormalSensors;
    }
    const mjtNum normal_norm = NormArray(normal);
    if (normal_norm > 0.0) {
      for (mjtNum& value : normal) {
        value /= normal_norm;
      }
    }
    reward_gripper_collision_side_ =
        std::sqrt(normal[0] * normal[0] + normal[1] * normal[1]);
    mjtNum arm_err = 0.0;
    for (int i = 0; i < kPandaRobotiqArmJoints; ++i) {
      const mjtNum delta =
          data_->qpos[arm_qposadr_[i]] - init_qpos_[arm_qposadr_[i]];
      arm_err += delta * delta;
    }
    reward_robot_target_qpos_ =
        ToleranceLinear(std::sqrt(arm_err), 0.0, 0.5, 4.5);
    mjtNum joint_vel_norm = 0.0;
    for (int i = 0; i < kPandaRobotiqArmJoints; ++i) {
      joint_vel_norm += data_->qvel[i] * data_->qvel[i];
    }
    reward_joint_vel_ =
        ToleranceReciprocal(std::sqrt(joint_vel_norm), 0.0, 0.5, 2.0);
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    static constexpr mjtNum k_jnt_vel_range[kPandaRobotiqArmJoints][2] = {
        {-2.1750, 2.1750}, {-2.1750, 2.1750}, {-2.1750, 2.1750},
        {-2.1750, 2.1750}, {-2.6100, 2.6100}, {-2.6100, 2.6100},
        {-2.6100, 2.6100}};
    bool near_vel_limit = false;
    for (int i = 0; i < kPandaRobotiqArmJoints; ++i) {
      near_vel_limit |=
          data_->qvel[arm_qposadr_[i]] > k_jnt_vel_range[i][1] * 0.9;
      near_vel_limit |=
          data_->qvel[arm_qposadr_[i]] < k_jnt_vel_range[i][0] * 0.9;
    }
    reward_joint_vel_limit_ = near_vel_limit ? 0.0 : 1.0;
    reward_total_command_ = NormAction(action);
    std::array<mjtNum, kPandaRobotiqActionDim> action_delta{};
    for (int i = 0; i < kPandaRobotiqActionDim; ++i) {
      action_delta[i] = action[i] - last_action_[i];
    }
    reward_action_rate_ = NormAction(action_delta);

    const mjtNum gripper_box = reward_gripper_box_;
    const mjtNum box_target = reward_box_target_;
    const mjtNum box_orientation = reward_box_orientation_;
    const mjtNum gripper_collision_side = reward_gripper_collision_side_;
    const mjtNum robot_target_qpos = reward_robot_target_qpos_;
    const mjtNum joint_vel = reward_joint_vel_;
    const mjtNum joint_vel_limit = reward_joint_vel_limit_;
    const mjtNum total_command = reward_total_command_;
    const mjtNum action_rate = reward_action_rate_;
    mjtNum scaled =
        gripper_box * spec_.config["gripper_box_scale"_] +
        box_target * spec_.config["box_target_scale"_] +
        box_orientation * spec_.config["box_orientation_scale"_] +
        gripper_collision_side * spec_.config["gripper_collision_side_scale"_] +
        robot_target_qpos * spec_.config["robot_target_qpos_scale"_] +
        joint_vel * spec_.config["joint_vel_scale"_] +
        joint_vel_limit * spec_.config["joint_vel_limit_scale"_] +
        total_command * spec_.config["total_command_scale"_] +
        action_rate * spec_.config["action_rate_scale"_];
    reward_gripper_box_ = gripper_box * spec_.config["gripper_box_scale"_];
    reward_box_target_ = box_target * spec_.config["box_target_scale"_];
    reward_box_orientation_ =
        box_orientation * spec_.config["box_orientation_scale"_];
    reward_gripper_collision_side_ =
        gripper_collision_side * spec_.config["gripper_collision_side_scale"_];
    reward_robot_target_qpos_ =
        robot_target_qpos * spec_.config["robot_target_qpos_scale"_];
    reward_joint_vel_ = joint_vel * spec_.config["joint_vel_scale"_];
    reward_joint_vel_limit_ =
        joint_vel_limit * spec_.config["joint_vel_limit_scale"_];
    reward_total_command_ =
        total_command * spec_.config["total_command_scale"_];
    reward_action_rate_ = action_rate * spec_.config["action_rate_scale"_];
    scaled = std::clamp(scaled, static_cast<mjtNum>(-10000.0),
                        static_cast<mjtNum>(10000.0));
    const mjtNum scale_sum = spec_.config["gripper_box_scale"_] +
                             spec_.config["box_target_scale"_] +
                             spec_.config["box_orientation_scale"_] +
                             spec_.config["gripper_collision_side_scale"_] +
                             spec_.config["robot_target_qpos_scale"_] +
                             spec_.config["joint_vel_scale"_] +
                             spec_.config["joint_vel_limit_scale"_] +
                             spec_.config["total_command_scale"_] +
                             spec_.config["action_rate_scale"_];
    return scaled / scale_sum;
  }

  bool SensorContact(
      const std::array<int, kPandaRobotiqContactSensors>& adrs) const {
    return std::any_of(adrs.begin(), adrs.end(), [this](int adr) {
      return data_->sensordata[adr] > 0.0;
    });
  }

  bool Termination() const {
    const mjtNum* box_pos = data_->xpos + 3 * obj_body_id_;
    const mjtNum* gripper_pos = data_->site_xpos + 3 * gripper_site_id_;
    bool done = box_pos[2] < -0.01;
    done |= box_pos[0] > 0.75 || box_pos[0] < 0.3;
    done |= box_pos[1] > 0.7 || box_pos[1] < -0.5;
    done |= gripper_pos[2] > 0.5;
    done |= gripper_pos[0] > 0.75 || gripper_pos[0] < 0.3;
    done |= gripper_pos[1] > 0.7 || gripper_pos[1] < -0.5;
    done |= SensorContact(wall_sensor_adrs_);
    done |= SensorContact(floor_sensor_adrs_);
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    static constexpr mjtNum k_jnt_range[kPandaRobotiqArmJoints][2] = {
        {-2.8973, 2.8973},  {-1.7628, 1.7628}, {-2.8973, 2.8973},
        {-3.0718, -0.0698}, {-2.8973, 2.8973}, {-0.0175, 3.7525},
        {-2.8973, 2.8973}};
    for (int i = 0; i < kPandaRobotiqArmJoints; ++i) {
      done |= data_->qpos[arm_qposadr_[i]] > k_jnt_range[i][1] * 0.9;
      done |= data_->qpos[arm_qposadr_[i]] < k_jnt_range[i][0] * 0.9;
    }
    return done;
  }

  bool HasNaN() const {
    return std::any_of(data_->qpos, data_->qpos + model_->nq,
                       [](mjtNum q) { return std::isnan(q); }) ||
           std::any_of(data_->qvel, data_->qvel + model_->nv,
                       [](mjtNum qvel) { return std::isnan(qvel); });
  }

  bool UpdateSuccess() {
    const mjtNum* target_pos = data_->mocap_pos + 3 * mocap_target_id_;
    const mjtNum* target_quat = data_->mocap_quat + 4 * mocap_target_id_;
    const mjtNum* box_pos = data_->xpos + 3 * obj_body_id_;
    const mjtNum* box_quat = data_->xquat + 4 * obj_body_id_;
    const mjtNum pos_err =
        std::sqrt((target_pos[0] - box_pos[0]) * (target_pos[0] - box_pos[0]) +
                  (target_pos[1] - box_pos[1]) * (target_pos[1] - box_pos[1]) +
                  (target_pos[2] - box_pos[2]) * (target_pos[2] - box_pos[2]));
    const mjtNum ori_err = OrientationError(box_quat, target_quat);
    const bool cond1 = pos_err < 0.03;
    const bool cond2 = ori_err < (10.0 / 180.0 * M_PI);
    const bool cond3 =
        success_step_count_ >= spec_.config["success_step_count"_];
    success_ = cond1 && cond2 && cond3 ? 1.0 : 0.0;
    success_1_ = cond1 ? 1.0 : 0.0;
    success_2_ = cond2 ? 1.0 : 0.0;
    const bool sub_success = cond1 && cond2;
    prev_step_success_ = sub_success ? 1 : 0;
    success_step_count_ = sub_success ? success_step_count_ + 1 : 0;
    if (success_ > 0.0) {
      prev_step_success_ = 0;
      success_step_count_ = 0;
    }
    return sub_success;
  }

  void ResetTargetAfterSuccess() {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    static constexpr mjtNum k_angles[5] = {45.0, 45.0, 90.0, 135.0, 180.0};
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    static constexpr mjtNum k_positions[5] = {0.05, 0.05, 0.1, 0.2, 0.2};
    curriculum_id_ = std::min(curriculum_id_ + 1, 4);
    const auto target_pos = RandomTargetPos(k_positions[curriculum_id_]);
    const auto target_delta = RandomTargetQuat(k_angles[curriculum_id_]);
    const mjtNum* box_quat = data_->xquat + 4 * obj_body_id_;
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum target_quat[4];
    mju_mulQuat(target_quat, box_quat, target_delta.data());
    for (int i = 0; i < 3; ++i) {
      data_->mocap_pos[3 * mocap_target_id_ + i] = target_pos[i];
    }
    for (int i = 0; i < 4; ++i) {
      data_->mocap_quat[4 * mocap_target_id_ + i] = target_quat[i];
    }
  }

  void UpdateSingleObs() {
    int index = 0;
    const mjtNum* target_pos = data_->mocap_pos + 3 * mocap_target_id_;
    for (int i = 0; i < 3; ++i) {
      obs_[index++] = target_pos[i];
    }
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum target_mat[9];
    mju_quat2Mat(target_mat, data_->mocap_quat + 4 * mocap_target_id_);
    for (int i = 3; i < 9; ++i) {
      obs_[index++] = target_mat[i];
    }
    for (mjtNum value : last_action_) {
      obs_[index++] = value;
    }
    for (int i = 0; i < kPandaRobotiqArmJoints; ++i) {
      obs_[index++] = data_->qpos[arm_qposadr_[i]] +
                      Uniform(0.0, spec_.config["noise_robot_qpos"_]);
    }
    for (int i = 0; i < kPandaRobotiqArmJoints; ++i) {
      obs_[index++] =
          data_->qvel[i] + Uniform(0.0, spec_.config["noise_robot_qvel"_]);
    }
    const mjtNum* gripper_pos = data_->site_xpos + 3 * gripper_site_id_;
    for (int i = 0; i < 3; ++i) {
      obs_[index++] =
          gripper_pos[i] + Uniform(0.0, spec_.config["noise_eef_pos"_]);
    }
    const mjtNum* gripper_mat = data_->site_xmat + 9 * gripper_site_id_;
    for (int i = 3; i < 9; ++i) {
      obs_[index++] = gripper_mat[i];
    }
    const mjtNum* obj_quat = data_->xquat + 4 * obj_body_id_;
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum obj_mat[9];
    mju_quat2Mat(obj_mat, obj_quat);
    for (int i = 3; i < 9; ++i) {
      obs_[index++] = obj_mat[i];
    }
    const mjtNum* obj_pos = data_->xpos + 3 * obj_body_id_;
    for (int i = 0; i < 3; ++i) {
      obs_[index++] =
          obj_pos[i] + Uniform(-1.0, 1.0) * spec_.config["noise_obj_pos"_];
    }
  }

  void UpdateDelayedObs() {
    UpdateSingleObs();
    for (int i = static_cast<int>(obs_history_.size()) - 1;
         i >= kPandaRobotiqObsDim; --i) {
      obs_history_[i] = obs_history_[i - kPandaRobotiqObsDim];
    }
    for (int i = 0; i < kPandaRobotiqObsDim; ++i) {
      obs_history_[i] = obs_[i];
    }
    const int delay = SampleDelay(spec_.config["obs_min_delay"_],
                                  spec_.config["obs_max_delay"_],
                                  spec_.config["obs_history_len"_]);
    for (int i = 0; i < kPandaRobotiqObsDim; ++i) {
      obs_[i] = obs_history_[delay * kPandaRobotiqObsDim + i];
    }
  }

  template <std::size_t N>
  static void CopyPadded(const mjtNum* src, int count,
                         std::array<mjtNum, N>* dst) {
    dst->fill(0.0);
    const int n = std::min<int>(count, static_cast<int>(N));
    std::copy(src, src + n, dst->begin());
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
      std::copy(obs_.begin(), obs_.end(), obs);
      CommitObservation("obs:state", &obs_state, reset);
    }
    state["info:terminated"_] = terminated_;
    state["info:out_of_bounds"_] = out_of_bounds_;
    state["info:success"_] = success_;
    state["info:success_1"_] = success_1_;
    state["info:success_2"_] = success_2_;
    state["info:reward_gripper_box"_] = reward_gripper_box_;
    state["info:reward_box_target"_] = reward_box_target_;
    state["info:reward_box_orientation"_] = reward_box_orientation_;
    state["info:reward_gripper_collision_side"_] =
        reward_gripper_collision_side_;
    state["info:reward_robot_target_qpos"_] = reward_robot_target_qpos_;
    state["info:reward_joint_vel"_] = reward_joint_vel_;
    state["info:reward_joint_vel_limit"_] = reward_joint_vel_limit_;
    state["info:reward_total_command"_] = reward_total_command_;
    state["info:reward_action_rate"_] = reward_action_rate_;
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
    std::array<mjtNum, 512> pad512{};
    CopyPadded(data_->xfrc_applied, model_->nbody * 6, &pad512);
    state["info:xfrc_applied"_].Assign(pad512.data(), pad512.size());
    CopyPadded(data_->xpos, model_->nbody * 3, &pad512);
    state["info:xpos"_].Assign(pad512.data(), pad512.size());
    CopyPadded(data_->xquat, model_->nbody * 4, &pad512);
    state["info:xquat"_].Assign(pad512.data(), pad512.size());
    CopyPadded(data_->site_xpos, model_->nsite * 3, &pad512);
    state["info:site_xpos"_].Assign(pad512.data(), pad512.size());
    std::array<mjtNum, 2048> pad2048{};
    CopyPadded(data_->xmat, model_->nbody * 9, &pad2048);
    state["info:xmat"_].Assign(pad2048.data(), pad2048.size());
    CopyPadded(data_->site_xmat, model_->nsite * 9, &pad2048);
    state["info:site_xmat"_].Assign(pad2048.data(), pad2048.size());
    CopyPadded(data_->sensordata, model_->nsensordata, &pad512);
    state["info:sensordata"_].Assign(pad512.data(), pad512.size());
    state["info:last_action"_].Assign(last_action_.data(), last_action_.size());
    std::array<mjtNum, 64> action_history_pad{};
    std::copy(action_history_.begin(),
              action_history_.begin() +
                  std::min<std::size_t>(action_history_.size(),
                                        action_history_pad.size()),
              action_history_pad.begin());
    state["info:action_history"_].Assign(action_history_pad.data(),
                                         action_history_pad.size());
    std::array<mjtNum, 2048> obs_history_pad{};
    std::copy(
        obs_history_.begin(),
        obs_history_.begin() +
            std::min<std::size_t>(obs_history_.size(), obs_history_pad.size()),
        obs_history_pad.begin());
    state["info:obs_history"_].Assign(obs_history_pad.data(),
                                      obs_history_pad.size());
    state["info:success_step_count"_] = success_step_count_;
    state["info:prev_step_success"_] = prev_step_success_;
    state["info:curriculum_id"_] = curriculum_id_;
#endif
  }
};

using PandaRobotiqSpec = PlaygroundPandaRobotiqEnvSpec;
using PandaRobotiqPixelSpec = PlaygroundPandaRobotiqPixelEnvSpec;
template <typename Spec, bool kFromPixels>
using PandaRobotiqBase = PlaygroundPandaRobotiqEnvBase<Spec, kFromPixels>;
using PandaRobotiqEnv = PandaRobotiqBase<PandaRobotiqSpec, false>;
using PandaRobotiqPixelEnv = PandaRobotiqBase<PandaRobotiqPixelSpec, true>;
using PlaygroundPandaRobotiqEnv = PandaRobotiqEnv;
using PlaygroundPandaRobotiqPixelEnv = PandaRobotiqPixelEnv;
using PandaRobotiqEnvPool = PlaygroundEnvPoolT<PlaygroundPandaRobotiqEnv>;
using PandaRobotiqPixelEnvPool = PlaygroundEnvPoolT<PandaRobotiqPixelEnv>;
using PlaygroundPandaRobotiqEnvPool = PandaRobotiqEnvPool;
using PlaygroundPandaRobotiqPixelEnvPool = PandaRobotiqPixelEnvPool;

}  // namespace mujoco_playground

#endif  // ENVPOOL_MUJOCO_PLAYGROUND_PANDA_ROBOTIQ_H_
