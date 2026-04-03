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

#ifndef ENVPOOL_MUJOCO_GYMNASIUM_ROBOTICS_HAND_H_
#define ENVPOOL_MUJOCO_GYMNASIUM_ROBOTICS_HAND_H_

#include <mujoco.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/gymnasium_robotics/mujoco_env.h"
#include "envpool/mujoco/gymnasium_robotics/utils.h"

namespace gymnasium_robotics {

class HandEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(20),
        "xml_file"_.Bind(std::string("hand/reach.xml")),
        "hand_task"_.Bind(std::string("reach")),
        "reward_type"_.Bind(std::string("sparse")), "obs_dim"_.Bind(63),
        "goal_dim"_.Bind(15), "qpos_dim"_.Bind(24), "qvel_dim"_.Bind(24),
        "relative_control"_.Bind(false), "distance_threshold"_.Bind(0.01),
        "rotation_threshold"_.Bind(0.1),
        "target_position"_.Bind(std::string("ignore")),
        "target_rotation"_.Bind(std::string("ignore")),
        "target_position_low0"_.Bind(-0.04),
        "target_position_high0"_.Bind(0.04),
        "target_position_low1"_.Bind(-0.06),
        "target_position_high1"_.Bind(0.02), "target_position_low2"_.Bind(0.0),
        "target_position_high2"_.Bind(0.06),
        "randomize_initial_position"_.Bind(true),
        "randomize_initial_rotation"_.Bind(true),
        "ignore_z_target_rotation"_.Bind(false),
        "touch_visualisation"_.Bind(std::string("off")),
        "touch_get_obs"_.Bind(std::string("off")));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    int obs_dim = conf["obs_dim"_];
    int goal_dim = conf["goal_dim"_];
#ifdef ENVPOOL_TEST
    int qpos_dim = conf["qpos_dim"_];
    int qvel_dim = conf["qvel_dim"_];
#endif
    return MakeDict(
        "obs:observation"_.Bind(Spec<mjtNum>({obs_dim}, {-inf, inf})),
        "obs:achieved_goal"_.Bind(Spec<mjtNum>({goal_dim}, {-inf, inf})),
        "obs:desired_goal"_.Bind(Spec<mjtNum>({goal_dim}, {-inf, inf})),
        "info:is_success"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({qpos_dim})),
        "info:qvel0"_.Bind(Spec<mjtNum>({qvel_dim})),
        "info:goal0"_.Bind(Spec<mjtNum>({goal_dim})),
#endif
        "info:distance"_.Bind(Spec<mjtNum>({-1}, {0.0, inf})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<float>({-1, 20}, {-1.0, 1.0})));
  }
};

using HandEnvSpec = EnvSpec<HandEnvFns>;

class HandEnv : public Env<HandEnvSpec>, public MujocoRobotEnv {
 protected:
  enum class TaskType : std::uint8_t {
    kReach,
    kManipulate,
  };

  enum class TargetPositionType : std::uint8_t {
    kIgnore,
    kFixed,
    kRandom,
  };

  enum class TargetRotationType : std::uint8_t {
    kIgnore,
    kFixed,
    kXyz,
    kZ,
    kParallel,
  };

  enum class TouchVisualizationType : std::uint8_t {
    kOff,
    kAlways,
    kOnTouch,
  };

  enum class TouchObsType : std::uint8_t {
    kOff,
    kSensordata,
    kBoolean,
    kLog,
  };

  static constexpr std::array<const char*, 5> kFingertipSiteNames = {
      "robot0:S_fftip", "robot0:S_mftip", "robot0:S_rftip",
      "robot0:S_lftip", "robot0:S_thtip",
  };

  TaskType task_type_;
  bool sparse_reward_;
  bool relative_control_;
  mjtNum distance_threshold_;
  mjtNum rotation_threshold_;
  TargetPositionType target_position_;
  TargetRotationType target_rotation_;
  std::array<mjtNum, 3> target_position_low_{};
  std::array<mjtNum, 3> target_position_high_{};
  bool randomize_initial_position_;
  bool randomize_initial_rotation_;
  bool ignore_z_target_rotation_;
  TouchVisualizationType touch_visualization_;
  TouchObsType touch_obs_;
  std::vector<mjtNum> goal_;
  std::vector<mjtNum> initial_goal_;
  std::array<mjtNum, 3> palm_xpos_{};
  std::array<int, 5> fingertip_site_ids_{};
  std::array<int, 5> target_site_ids_{};
  std::array<int, 5> finger_site_ids_{};
  int palm_body_id_{-1};
  int object_center_site_id_{-1};
  int object_hidden_geom_id_{-1};
  std::vector<int> actuator_qpos_addr_;
  std::vector<int> actuator_coupled_qpos_addr_;
  std::vector<int> touch_sensor_addrs_;
  std::vector<int> touch_site_ids_;
  std::uniform_int_distribution<int> reach_finger_dist_{0, 3};
  std::uniform_real_distribution<> unit_dist_{0.0, 1.0};
  std::uniform_real_distribution<> rotation_dist_{-3.14159265358979323846,
                                                  3.14159265358979323846};
  std::normal_distribution<> goal_noise_dist_{0.0, 0.005};
  std::normal_distribution<> object_noise_dist_{0.0, 0.005};

 public:
  HandEnv(const Spec& spec, int env_id)
      : Env<HandEnvSpec>(spec, env_id),
        MujocoRobotEnv(spec.config["base_path"_], spec.config["xml_file"_],
                       spec.config["frame_skip"_],
                       spec.config["max_episode_steps"_]),
        task_type_(ParseTaskType(spec.config["hand_task"_])),
        sparse_reward_(spec.config["reward_type"_] == "sparse"),
        relative_control_(spec.config["relative_control"_]),
        distance_threshold_(spec.config["distance_threshold"_]),
        rotation_threshold_(spec.config["rotation_threshold"_]),
        target_position_(ParseTargetPosition(spec.config["target_position"_])),
        target_rotation_(ParseTargetRotation(spec.config["target_rotation"_])),
        target_position_low_({spec.config["target_position_low0"_],
                              spec.config["target_position_low1"_],
                              spec.config["target_position_low2"_]}),
        target_position_high_({spec.config["target_position_high0"_],
                               spec.config["target_position_high1"_],
                               spec.config["target_position_high2"_]}),
        randomize_initial_position_(spec.config["randomize_initial_position"_]),
        randomize_initial_rotation_(spec.config["randomize_initial_rotation"_]),
        ignore_z_target_rotation_(spec.config["ignore_z_target_rotation"_]),
        touch_visualization_(
            ParseTouchVisualization(spec.config["touch_visualisation"_])),
        touch_obs_(ParseTouchObs(spec.config["touch_get_obs"_])),
        goal_(spec.config["goal_dim"_], 0.0),
        initial_goal_(spec.config["goal_dim"_], 0.0) {
    SetupActuatorMaps();
    SetupTaskIds();
    SetupInitialEnvState();
    InitializeRobotEnv();
    SetupTouchSensors();
    if (touch_visualization_ == TouchVisualizationType::kOff) {
      for (int site_id : touch_site_ids_) {
        model_->site_rgba[4 * site_id + 3] = 0.0;
      }
    }
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    while (!ResetSim()) {
    }
    goal_ = SampleGoal();
    CaptureResetState();
    WriteState(0.0F);
  }

  void Step(const Action& action) override {
    SetHandAction(static_cast<const float*>(action["action"_].Data()));
    DoSimulation();
    ++elapsed_step_;
    done_ = elapsed_step_ >= max_episode_steps_;
    WriteState(static_cast<float>(ComputeReward(AchievedGoal(), goal_)));
  }

 protected:
  void EnvSetup() override { SetupInitialEnvState(); }

  void RenderCallback() override {
    if (task_type_ == TaskType::kReach) {
      RenderReachGoal();
    } else {
      RenderManipulateGoal();
    }
  }

 private:
  void SetupInitialEnvState() {
    if (task_type_ == TaskType::kReach) {
      SetDefaultReachInitialQpos();
    }
    mj_forward(model_, data_);
    if (task_type_ == TaskType::kReach) {
      initial_goal_ = AchievedGoal();
      for (int i = 0; i < 3; ++i) {
        palm_xpos_[i] = data_->xpos[3 * palm_body_id_ + i];
      }
    }
  }
  static TaskType ParseTaskType(const std::string& value) {
    if (value == "reach") {
      return TaskType::kReach;
    }
    if (value == "manipulate") {
      return TaskType::kManipulate;
    }
    throw std::runtime_error("Unknown Hand task type: " + value);
  }

  static TargetPositionType ParseTargetPosition(const std::string& value) {
    if (value == "ignore") {
      return TargetPositionType::kIgnore;
    }
    if (value == "fixed") {
      return TargetPositionType::kFixed;
    }
    if (value == "random") {
      return TargetPositionType::kRandom;
    }
    throw std::runtime_error("Unknown Hand target position: " + value);
  }

  static TargetRotationType ParseTargetRotation(const std::string& value) {
    if (value == "ignore") {
      return TargetRotationType::kIgnore;
    }
    if (value == "fixed") {
      return TargetRotationType::kFixed;
    }
    if (value == "xyz") {
      return TargetRotationType::kXyz;
    }
    if (value == "z") {
      return TargetRotationType::kZ;
    }
    if (value == "parallel") {
      return TargetRotationType::kParallel;
    }
    throw std::runtime_error("Unknown Hand target rotation: " + value);
  }

  static TouchVisualizationType ParseTouchVisualization(
      const std::string& value) {
    if (value == "off") {
      return TouchVisualizationType::kOff;
    }
    if (value == "always") {
      return TouchVisualizationType::kAlways;
    }
    if (value == "on_touch") {
      return TouchVisualizationType::kOnTouch;
    }
    throw std::runtime_error("Unknown Hand touch visualization: " + value);
  }

  static TouchObsType ParseTouchObs(const std::string& value) {
    if (value == "off") {
      return TouchObsType::kOff;
    }
    if (value == "sensordata") {
      return TouchObsType::kSensordata;
    }
    if (value == "boolean") {
      return TouchObsType::kBoolean;
    }
    if (value == "log") {
      return TouchObsType::kLog;
    }
    throw std::runtime_error("Unknown Hand touch obs mode: " + value);
  }

  static bool StartsWith(std::string_view value, std::string_view prefix) {
    return value.substr(0, prefix.size()) == prefix;
  }

  static std::string ReplaceActuatorPrefix(const std::string& actuator_name) {
    constexpr std::string_view k_prefix = ":A_";
    std::size_t pos = actuator_name.find(k_prefix);
    if (pos == std::string::npos) {
      throw std::runtime_error("Unexpected Hand actuator name: " +
                               actuator_name);
    }
    std::string joint_name = actuator_name;
    joint_name.replace(pos, k_prefix.size(), ":");
    return joint_name;
  }

  static std::string CoupledJointName(const std::string& joint_name) {
    if (joint_name == "robot0:FFJ1") {
      return "robot0:FFJ0";
    }
    if (joint_name == "robot0:MFJ1") {
      return "robot0:MFJ0";
    }
    if (joint_name == "robot0:RFJ1") {
      return "robot0:RFJ0";
    }
    if (joint_name == "robot0:LFJ1") {
      return "robot0:LFJ0";
    }
    return "";
  }

  void SetupActuatorMaps() {
    actuator_qpos_addr_.resize(model_->nu);
    actuator_coupled_qpos_addr_.assign(model_->nu, -1);
    for (int act_id = 0; act_id < model_->nu; ++act_id) {
      const char* actuator_name = mj_id2name(model_, mjOBJ_ACTUATOR, act_id);
      if (actuator_name == nullptr) {
        throw std::runtime_error("Unnamed Hand actuator.");
      }
      std::string joint_name = ReplaceActuatorPrefix(actuator_name);
      actuator_qpos_addr_[act_id] = JointQposAddress(model_, joint_name);
      std::string coupled_joint_name = CoupledJointName(joint_name);
      if (!coupled_joint_name.empty()) {
        actuator_coupled_qpos_addr_[act_id] =
            JointQposAddress(model_, coupled_joint_name);
      }
    }
  }

  void SetupTaskIds() {
    if (task_type_ == TaskType::kReach) {
      for (int finger_idx = 0; finger_idx < 5; ++finger_idx) {
        fingertip_site_ids_[finger_idx] =
            SiteId(model_, kFingertipSiteNames[finger_idx]);
        target_site_ids_[finger_idx] =
            SiteId(model_, "target" + std::to_string(finger_idx));
        finger_site_ids_[finger_idx] =
            SiteId(model_, "finger" + std::to_string(finger_idx));
      }
      palm_body_id_ = BodyId(model_, "robot0:palm");
    } else {
      object_center_site_id_ = SiteId(model_, "object:center");
      object_hidden_geom_id_ = mj_name2id(model_, mjOBJ_GEOM, "object_hidden");
    }
  }

  void SetupTouchSensors() {
    touch_sensor_addrs_.clear();
    touch_site_ids_.clear();
    if (touch_obs_ == TouchObsType::kOff &&
        touch_visualization_ == TouchVisualizationType::kOff) {
      return;
    }
    for (int sensor_id = 0; sensor_id < model_->nsensor; ++sensor_id) {
      const char* sensor_name = mj_id2name(model_, mjOBJ_SENSOR, sensor_id);
      if (sensor_name == nullptr || !StartsWith(sensor_name, "robot0:TS_")) {
        continue;
      }
      touch_sensor_addrs_.push_back(model_->sensor_adr[sensor_id]);
      std::string site_name = sensor_name;
      site_name.replace(7, 3, "T_");
      touch_site_ids_.push_back(SiteId(model_, site_name));
    }
  }

  void SetDefaultReachInitialQpos() {
    SetJointQpos(model_, data_, "robot0:WRJ1", -0.16514339750464327);
    SetJointQpos(model_, data_, "robot0:WRJ0", -0.31973286565062153);
    SetJointQpos(model_, data_, "robot0:FFJ3", 0.14340512546557435);
    SetJointQpos(model_, data_, "robot0:FFJ2", 0.32028208333591573);
    SetJointQpos(model_, data_, "robot0:FFJ1", 0.7126053607727917);
    SetJointQpos(model_, data_, "robot0:FFJ0", 0.6705281001412586);
    SetJointQpos(model_, data_, "robot0:MFJ3", 0.000246444303701037);
    SetJointQpos(model_, data_, "robot0:MFJ2", 0.3152655251085491);
    SetJointQpos(model_, data_, "robot0:MFJ1", 0.7659800313729842);
    SetJointQpos(model_, data_, "robot0:MFJ0", 0.7323156897425923);
    SetJointQpos(model_, data_, "robot0:RFJ3", 0.00038520700007378114);
    SetJointQpos(model_, data_, "robot0:RFJ2", 0.36743546201985233);
    SetJointQpos(model_, data_, "robot0:RFJ1", 0.7119514095008576);
    SetJointQpos(model_, data_, "robot0:RFJ0", 0.6699446327514138);
    SetJointQpos(model_, data_, "robot0:LFJ4", 0.0525442258033891);
    SetJointQpos(model_, data_, "robot0:LFJ3", -0.13615534724474673);
    SetJointQpos(model_, data_, "robot0:LFJ2", 0.39872030433433003);
    SetJointQpos(model_, data_, "robot0:LFJ1", 0.7415570009679252);
    SetJointQpos(model_, data_, "robot0:LFJ0", 0.704096378652974);
    SetJointQpos(model_, data_, "robot0:THJ4", 0.003673823825070126);
    SetJointQpos(model_, data_, "robot0:THJ3", 0.5506291436028695);
    SetJointQpos(model_, data_, "robot0:THJ2", -0.014515151997119306);
    SetJointQpos(model_, data_, "robot0:THJ1", -0.0015229223564485414);
    SetJointQpos(model_, data_, "robot0:THJ0", -0.7894883021600622);
  }

  void SetHandAction(const float* raw_action) {
    for (int i = 0; i < model_->nu; ++i) {
      mjtNum range_low = model_->actuator_ctrlrange[2 * i];
      mjtNum range_high = model_->actuator_ctrlrange[2 * i + 1];
      mjtNum actuation_range = (range_high - range_low) / 2.0;
      mjtNum actuation_center = relative_control_
                                    ? data_->qpos[actuator_qpos_addr_[i]]
                                    : (range_high + range_low) / 2.0;
      if (relative_control_ && actuator_coupled_qpos_addr_[i] >= 0) {
        actuation_center += data_->qpos[actuator_coupled_qpos_addr_[i]];
      }
      mjtNum action =
          static_cast<mjtNum>(std::clamp(raw_action[i], -1.0F, 1.0F));
      data_->ctrl[i] = std::clamp(actuation_center + action * actuation_range,
                                  range_low, range_high);
    }
  }

  bool ResetSim() {
    ResetToInitialState();
    if (task_type_ == TaskType::kReach) {
      mj_forward(model_, data_);
      return true;
    }
    mj_forward(model_, data_);
    auto object_qpos = GetJointQpos(model_, data_, "object:joint");
    std::array<mjtNum, 3> initial_pos{
        object_qpos[0],
        object_qpos[1],
        object_qpos[2],
    };
    std::array<mjtNum, 4> initial_quat{
        object_qpos[3],
        object_qpos[4],
        object_qpos[5],
        object_qpos[6],
    };
    if (randomize_initial_rotation_) {
      initial_quat = QuatMul(initial_quat, SampleManipulationQuatOffset());
    }
    if (randomize_initial_position_ &&
        target_position_ != TargetPositionType::kFixed) {
      for (int i = 0; i < 3; ++i) {
        initial_pos[i] += static_cast<mjtNum>(object_noise_dist_(gen_));
      }
    }
    NormalizeQuat(&initial_quat);
    SetJointQpos(
        model_, data_, "object:joint",
        {initial_pos[0], initial_pos[1], initial_pos[2], initial_quat[0],
         initial_quat[1], initial_quat[2], initial_quat[3]});
    std::array<float, 20> zero_action{};
    for (int i = 0; i < 10; ++i) {
      SetHandAction(zero_action.data());
      DoSimulation();
    }
    mj_forward(model_, data_);
    return data_->site_xpos[3 * object_center_site_id_ + 2] > 0.04;
  }

  std::vector<mjtNum> SampleGoal() {
    if (task_type_ == TaskType::kReach) {
      return SampleReachGoal();
    }
    return SampleManipulationGoal();
  }

  std::vector<mjtNum> SampleReachGoal() {
    constexpr int k_thumb_idx = 4;
    int finger_idx = reach_finger_dist_(gen_);
    std::array<mjtNum, 3> meeting_pos{
        static_cast<mjtNum>(palm_xpos_[0] + goal_noise_dist_(gen_)),
        static_cast<mjtNum>(palm_xpos_[1] - 0.09 + goal_noise_dist_(gen_)),
        static_cast<mjtNum>(palm_xpos_[2] + 0.05 + goal_noise_dist_(gen_)),
    };
    std::vector<mjtNum> goal = initial_goal_;
    for (int selected_idx : std::array<int, 2>{k_thumb_idx, finger_idx}) {
      std::array<mjtNum, 3> offset_direction{
          meeting_pos[0] - goal[3 * selected_idx],
          meeting_pos[1] - goal[3 * selected_idx + 1],
          meeting_pos[2] - goal[3 * selected_idx + 2],
      };
      mjtNum norm = std::sqrt(offset_direction[0] * offset_direction[0] +
                              offset_direction[1] * offset_direction[1] +
                              offset_direction[2] * offset_direction[2]);
      if (norm <= 0.0) {
        continue;
      }
      for (int i = 0; i < 3; ++i) {
        goal[3 * selected_idx + i] =
            meeting_pos[i] - 0.005 * offset_direction[i] / norm;
      }
    }
    if (unit_dist_(gen_) < 0.1) {
      goal = initial_goal_;
    }
    return goal;
  }

  std::vector<mjtNum> SampleManipulationGoal() {
    auto object_qpos = GetJointQpos(model_, data_, "object:joint");
    std::array<mjtNum, 3> target_pos{
        object_qpos[0],
        object_qpos[1],
        object_qpos[2],
    };
    if (target_position_ == TargetPositionType::kRandom) {
      for (int i = 0; i < 3; ++i) {
        std::uniform_real_distribution<> offset_dist(target_position_low_[i],
                                                     target_position_high_[i]);
        target_pos[i] += static_cast<mjtNum>(offset_dist(gen_));
      }
    }
    std::array<mjtNum, 4> target_quat{
        object_qpos[3],
        object_qpos[4],
        object_qpos[5],
        object_qpos[6],
    };
    if (target_rotation_ == TargetRotationType::kZ ||
        target_rotation_ == TargetRotationType::kParallel ||
        target_rotation_ == TargetRotationType::kXyz) {
      target_quat = SampleManipulationTargetQuat();
    }
    NormalizeQuat(&target_quat);
    return {target_pos[0],  target_pos[1],  target_pos[2], target_quat[0],
            target_quat[1], target_quat[2], target_quat[3]};
  }

  std::array<mjtNum, 4> SampleManipulationQuatOffset() {
    if (target_rotation_ == TargetRotationType::kFixed) {
      return {1.0, 0.0, 0.0, 0.0};
    }
    if (target_rotation_ == TargetRotationType::kZ) {
      return QuatFromAngleAndAxis(static_cast<mjtNum>(rotation_dist_(gen_)),
                                  {0.0, 0.0, 1.0});
    }
    if (target_rotation_ == TargetRotationType::kParallel) {
      auto z_quat = QuatFromAngleAndAxis(
          static_cast<mjtNum>(rotation_dist_(gen_)), {0.0, 0.0, 1.0});
      const auto& parallel_quats = ParallelQuats();
      std::uniform_int_distribution<int> quat_dist(
          0, static_cast<int>(parallel_quats.size()) - 1);
      return QuatMul(z_quat, parallel_quats[quat_dist(gen_)]);
    }
    if (target_rotation_ == TargetRotationType::kXyz ||
        target_rotation_ == TargetRotationType::kIgnore) {
      return QuatFromAngleAndAxis(
          static_cast<mjtNum>(rotation_dist_(gen_)),
          {static_cast<mjtNum>(unit_dist_(gen_) * 2.0 - 1.0),
           static_cast<mjtNum>(unit_dist_(gen_) * 2.0 - 1.0),
           static_cast<mjtNum>(unit_dist_(gen_) * 2.0 - 1.0)});
    }
    return {1.0, 0.0, 0.0, 0.0};
  }

  std::array<mjtNum, 4> SampleManipulationTargetQuat() {
    if (target_rotation_ == TargetRotationType::kZ) {
      return QuatFromAngleAndAxis(static_cast<mjtNum>(rotation_dist_(gen_)),
                                  {0.0, 0.0, 1.0});
    }
    if (target_rotation_ == TargetRotationType::kParallel) {
      auto z_quat = QuatFromAngleAndAxis(
          static_cast<mjtNum>(rotation_dist_(gen_)), {0.0, 0.0, 1.0});
      const auto& parallel_quats = ParallelQuats();
      std::uniform_int_distribution<int> quat_dist(
          0, static_cast<int>(parallel_quats.size()) - 1);
      return QuatMul(z_quat, parallel_quats[quat_dist(gen_)]);
    }
    return QuatFromAngleAndAxis(
        static_cast<mjtNum>(rotation_dist_(gen_)),
        {static_cast<mjtNum>(unit_dist_(gen_) * 2.0 - 1.0),
         static_cast<mjtNum>(unit_dist_(gen_) * 2.0 - 1.0),
         static_cast<mjtNum>(unit_dist_(gen_) * 2.0 - 1.0)});
  }

  std::vector<mjtNum> AchievedGoal() const {
    if (task_type_ == TaskType::kReach) {
      std::vector<mjtNum> achieved_goal;
      achieved_goal.reserve(15);
      for (int site_id : fingertip_site_ids_) {
        auto site_xpos = GetSiteXpos(model_, data_, site_id);
        achieved_goal.insert(achieved_goal.end(), site_xpos.begin(),
                             site_xpos.end());
      }
      return achieved_goal;
    }
    return GetJointQpos(model_, data_, "object:joint");
  }

  std::pair<mjtNum, mjtNum> ManipulationGoalDistance(
      const std::vector<mjtNum>& achieved_goal,
      const std::vector<mjtNum>& desired_goal) const {
    mjtNum d_pos = 0.0;
    if (target_position_ != TargetPositionType::kIgnore) {
      mjtNum dx = achieved_goal[0] - desired_goal[0];
      mjtNum dy = achieved_goal[1] - desired_goal[1];
      mjtNum dz = achieved_goal[2] - desired_goal[2];
      d_pos = std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    mjtNum d_rot = 0.0;
    if (target_rotation_ != TargetRotationType::kIgnore) {
      std::array<mjtNum, 4> quat_a{achieved_goal[3], achieved_goal[4],
                                   achieved_goal[5], achieved_goal[6]};
      std::array<mjtNum, 4> quat_b{desired_goal[3], desired_goal[4],
                                   desired_goal[5], desired_goal[6]};
      if (ignore_z_target_rotation_) {
        auto euler_a = Quat2Euler(quat_a);
        auto euler_b = Quat2Euler(quat_b);
        euler_a[2] = euler_b[2];
        quat_a = Euler2Quat(euler_a);
      }
      auto quat_diff = QuatMul(quat_a, QuatConjugate(quat_b));
      d_rot =
          2.0 * std::acos(std::clamp(quat_diff[0], mjtNum{-1.0}, mjtNum{1.0}));
    }
    return {d_pos, d_rot};
  }

  mjtNum ComputeReward(const std::vector<mjtNum>& achieved_goal,
                       const std::vector<mjtNum>& desired_goal) const {
    if (task_type_ == TaskType::kReach) {
      mjtNum distance = ReachGoalDistance(achieved_goal, desired_goal);
      if (sparse_reward_) {
        return distance > distance_threshold_ ? -1.0 : 0.0;
      }
      return -distance;
    }
    auto [d_pos, d_rot] = ManipulationGoalDistance(achieved_goal, desired_goal);
    if (sparse_reward_) {
      return (d_pos < distance_threshold_ && d_rot < rotation_threshold_)
                 ? 0.0
                 : -1.0;
    }
    return -(10.0 * d_pos + d_rot);
  }

  bool IsSuccess(const std::vector<mjtNum>& achieved_goal,
                 const std::vector<mjtNum>& desired_goal) const {
    if (task_type_ == TaskType::kReach) {
      return ReachGoalDistance(achieved_goal, desired_goal) <
             distance_threshold_;
    }
    auto [d_pos, d_rot] = ManipulationGoalDistance(achieved_goal, desired_goal);
    return d_pos < distance_threshold_ && d_rot < rotation_threshold_;
  }

  mjtNum InfoDistance(const std::vector<mjtNum>& achieved_goal,
                      const std::vector<mjtNum>& desired_goal) const {
    if (task_type_ == TaskType::kReach) {
      return ReachGoalDistance(achieved_goal, desired_goal);
    }
    auto [d_pos, d_rot] = ManipulationGoalDistance(achieved_goal, desired_goal);
    return d_pos + d_rot;
  }

  mjtNum ReachGoalDistance(const std::vector<mjtNum>& lhs,
                           const std::vector<mjtNum>& rhs) const {
    mjtNum dist = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
      mjtNum delta = lhs[i] - rhs[i];
      dist += delta * delta;
    }
    return std::sqrt(dist);
  }

  static void NormalizeQuat(std::array<mjtNum, 4>* quat) {
    mjtNum norm = std::sqrt((*quat)[0] * (*quat)[0] + (*quat)[1] * (*quat)[1] +
                            (*quat)[2] * (*quat)[2] + (*quat)[3] * (*quat)[3]);
    if (norm <= 0.0) {
      *quat = {1.0, 0.0, 0.0, 0.0};
      return;
    }
    for (mjtNum& value : *quat) {
      value /= norm;
    }
  }

  void RenderReachGoal() {
    std::vector<mjtNum> achieved_goal = AchievedGoal();
    for (int finger_idx = 0; finger_idx < 5; ++finger_idx) {
      UpdateSitePosition(target_site_ids_[finger_idx], &goal_[3 * finger_idx]);
      UpdateSitePosition(finger_site_ids_[finger_idx],
                         &achieved_goal[3 * finger_idx]);
    }
    mj_forward(model_, data_);
  }

  void RenderManipulateGoal() {
    std::vector<mjtNum> render_goal = goal_;
    if (target_position_ == TargetPositionType::kIgnore) {
      render_goal[0] += 0.15;
    }
    SetJointQpos(model_, data_, "target:joint", render_goal);
    SetJointQvel(model_, data_, "target:joint",
                 std::vector<mjtNum>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    if (object_hidden_geom_id_ >= 0) {
      model_->geom_rgba[4 * object_hidden_geom_id_ + 3] = 1.0;
    }
    mj_forward(model_, data_);
    if (touch_visualization_ == TouchVisualizationType::kOnTouch) {
      for (std::size_t i = 0; i < touch_sensor_addrs_.size(); ++i) {
        float* site_rgba = model_->site_rgba + 4 * touch_site_ids_[i];
        if (data_->sensordata[touch_sensor_addrs_[i]] != 0.0) {
          site_rgba[0] = 1.0;
          site_rgba[1] = 0.0;
          site_rgba[2] = 0.0;
          site_rgba[3] = 0.5;
        } else {
          site_rgba[0] = 0.0;
          site_rgba[1] = 0.5;
          site_rgba[2] = 0.0;
          site_rgba[3] = 0.2;
        }
      }
    }
  }

  void UpdateSitePosition(int site_id, const mjtNum* goal) {
    for (int i = 0; i < 3; ++i) {
      mjtNum site_offset =
          data_->site_xpos[3 * site_id + i] - model_->site_pos[3 * site_id + i];
      model_->site_pos[3 * site_id + i] = goal[i] - site_offset;
    }
  }

  void WriteState(float reward) {
    auto state = Allocate();
    state["reward"_] = reward;
    mjtNum* obs = static_cast<mjtNum*>(state["obs:observation"_].Data());
    auto [robot_qpos, robot_qvel] = RobotGetObs(model_, data_);
    for (mjtNum value : robot_qpos) {
      *(obs++) = value;
    }
    for (mjtNum value : robot_qvel) {
      *(obs++) = value;
    }
    std::vector<mjtNum> achieved_goal = AchievedGoal();
    if (task_type_ == TaskType::kManipulate) {
      auto object_qvel = GetJointQvel(model_, data_, "object:joint");
      for (mjtNum value : object_qvel) {
        *(obs++) = value;
      }
    }
    for (mjtNum value : achieved_goal) {
      *(obs++) = value;
    }
    if (task_type_ == TaskType::kManipulate) {
      if (touch_obs_ == TouchObsType::kSensordata) {
        for (int sensor_addr : touch_sensor_addrs_) {
          *(obs++) = data_->sensordata[sensor_addr];
        }
      } else if (touch_obs_ == TouchObsType::kBoolean) {
        for (int sensor_addr : touch_sensor_addrs_) {
          *(obs++) = data_->sensordata[sensor_addr] > 0.0 ? 1.0 : 0.0;
        }
      } else if (touch_obs_ == TouchObsType::kLog) {
        for (int sensor_addr : touch_sensor_addrs_) {
          *(obs++) = std::log(data_->sensordata[sensor_addr] + 1.0);
        }
      }
    }

    state["obs:achieved_goal"_].Assign(achieved_goal.data(),
                                       achieved_goal.size());
    state["obs:desired_goal"_].Assign(goal_.data(), goal_.size());
    state["info:is_success"_] = IsSuccess(achieved_goal, goal_) ? 1.0 : 0.0;
    state["info:distance"_] = InfoDistance(achieved_goal, goal_);
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
    state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
    state["info:goal0"_].Assign(goal_.data(), goal_.size());
#endif
  }
};

using HandEnvPool = AsyncEnvPool<HandEnv>;

}  // namespace gymnasium_robotics

#endif  // ENVPOOL_MUJOCO_GYMNASIUM_ROBOTICS_HAND_H_
