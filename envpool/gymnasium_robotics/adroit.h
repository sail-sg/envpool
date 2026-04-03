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

#ifndef ENVPOOL_GYMNASIUM_ROBOTICS_ADROIT_H_
#define ENVPOOL_GYMNASIUM_ROBOTICS_ADROIT_H_

#include <mujoco.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/gymnasium_robotics/mujoco_env.h"
#include "envpool/gymnasium_robotics/utils.h"

namespace gymnasium_robotics {

class AdroitEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(5),
        "xml_file"_.Bind(std::string("adroit_hand/adroit_door.xml")),
        "adroit_task"_.Bind(std::string("door")),
        "reward_type"_.Bind(std::string("dense")), "obs_dim"_.Bind(39),
        "action_dim"_.Bind(28), "qpos_dim"_.Bind(30), "qvel_dim"_.Bind(30),
        "reset_dim"_.Bind(3));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
#ifdef ENVPOOL_TEST
    int qpos_dim = conf["qpos_dim"_];
    int qvel_dim = conf["qvel_dim"_];
    int reset_dim = conf["reset_dim"_];
#endif
    return MakeDict("obs"_.Bind(Spec<mjtNum>({conf["obs_dim"_]}, {-inf, inf})),
                    "info:success"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
                    "info:distance"_.Bind(Spec<mjtNum>({-1}, {0.0, inf}))
#ifdef ENVPOOL_TEST
                        ,
                    "info:qpos0"_.Bind(Spec<mjtNum>({qpos_dim})),
                    "info:qvel0"_.Bind(Spec<mjtNum>({qvel_dim})),
                    "info:extra0"_.Bind(Spec<mjtNum>({reset_dim}))
#endif
    );
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<float>({-1, conf["action_dim"_]}, {-1.0, 1.0})));
  }
};

using AdroitEnvSpec = EnvSpec<AdroitEnvFns>;

class AdroitEnv : public Env<AdroitEnvSpec>, public MujocoRobotEnv {
 protected:
  enum class TaskType : std::uint8_t {
    kDoor,
    kHammer,
    kPen,
    kRelocate,
  };

  struct RewardInfo {
    mjtNum reward;
    mjtNum distance;
    bool success;
  };

  TaskType task_type_;
  bool sparse_reward_;
  int obs_dim_;
  int action_dim_;
  std::vector<mjtNum> act_mean_;
  std::vector<mjtNum> act_rng_;
  int door_hinge_addr_{-1};
  int grasp_site_id_{-1};
  int handle_site_id_{-1};
  int door_body_id_{-1};
  int target_site_id_{-1};
  int object_body_id_{-1};
  int tool_site_id_{-1};
  int goal_site_id_{-1};
  int target_body_id_{-1};
  int nail_sensor_addr_{-1};
  int eps_ball_site_id_{-1};
  int object_top_site_id_{-1};
  int object_bottom_site_id_{-1};
  int target_top_site_id_{-1};
  int target_bottom_site_id_{-1};
  mjtNum pen_length_{1.0};
  mjtNum target_length_{1.0};
  std::uniform_real_distribution<> unit_dist_{0.0, 1.0};
#ifdef ENVPOOL_TEST
  std::vector<mjtNum> extra0_;
#endif

 public:
  AdroitEnv(const Spec& spec, int env_id)
      : Env<AdroitEnvSpec>(spec, env_id),
        MujocoRobotEnv(spec.config["base_path"_], spec.config["xml_file"_],
                       spec.config["frame_skip"_],
                       spec.config["max_episode_steps"_]),
        task_type_(ParseTaskType(spec.config["adroit_task"_])),
        sparse_reward_(spec.config["reward_type"_] == "sparse"),
        obs_dim_(spec.config["obs_dim"_]),
        action_dim_(spec.config["action_dim"_]),
        act_mean_(spec.config["action_dim"_]),
        act_rng_(spec.config["action_dim"_])
#ifdef ENVPOOL_TEST
        ,
        extra0_(spec.config["reset_dim"_])
#endif
  {
    SetupTaskIds();
    SetupActuators();
    SetupActionScale();
    InitializeRobotEnv();
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    ResetTaskState();
    CaptureResetState();
    CaptureTaskState();
    WriteState(0.0);
  }

  void Step(const Action& action) override {
    const float* act = static_cast<const float*>(action["action"_].Data());
    for (int i = 0; i < action_dim_; ++i) {
      mjtNum value =
          std::clamp(static_cast<mjtNum>(act[i]), static_cast<mjtNum>(-1.0),
                     static_cast<mjtNum>(1.0));
      data_->ctrl[i] = act_mean_[i] + value * act_rng_[i];
    }
    DoSimulation();
    ++elapsed_step_;
    done_ = elapsed_step_ >= max_episode_steps_;
    auto reward_info = ComputeRewardInfo();
    WriteState(reward_info.reward, reward_info.distance, reward_info.success);
  }

 protected:
  static TaskType ParseTaskType(const std::string& task) {
    if (task == "door") {
      return TaskType::kDoor;
    }
    if (task == "hammer") {
      return TaskType::kHammer;
    }
    if (task == "pen") {
      return TaskType::kPen;
    }
    if (task == "relocate") {
      return TaskType::kRelocate;
    }
    throw std::runtime_error("Unknown Adroit task: " + task);
  }

  static mjtNum L2(const std::array<mjtNum, 3>& value) {
    return std::sqrt(value[0] * value[0] + value[1] * value[1] +
                     value[2] * value[2]);
  }

  static mjtNum Dot(const std::array<mjtNum, 3>& lhs,
                    const std::array<mjtNum, 3>& rhs) {
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
  }

  static std::array<mjtNum, 3> BodyXpos(const mjData* data, int body_id) {
    return {data->xpos[3 * body_id], data->xpos[3 * body_id + 1],
            data->xpos[3 * body_id + 2]};
  }

  static std::array<mjtNum, 4> BodyQuat(const mjModel* model, int body_id) {
    return {model->body_quat[4 * body_id], model->body_quat[4 * body_id + 1],
            model->body_quat[4 * body_id + 2],
            model->body_quat[4 * body_id + 3]};
  }

  static std::array<mjtNum, 3> Diff(const std::array<mjtNum, 3>& lhs,
                                    const std::array<mjtNum, 3>& rhs) {
    return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
  }

  mjtNum Uniform(mjtNum low, mjtNum high) {
    return low + (high - low) * unit_dist_(gen_);
  }

  std::array<mjtNum, 3> Orientation(int top_site_id, int bottom_site_id,
                                    mjtNum length) const {
    auto top = GetSiteXpos(model_, data_, top_site_id);
    auto bottom = GetSiteXpos(model_, data_, bottom_site_id);
    if (length <= 0.0) {
      return {0.0, 0.0, 0.0};
    }
    return {(top[0] - bottom[0]) / length, (top[1] - bottom[1]) / length,
            (top[2] - bottom[2]) / length};
  }

  void SetupTaskIds() {
    switch (task_type_) {
      case TaskType::kDoor:
        door_hinge_addr_ = JointQposAddress(model_, "door_hinge");
        grasp_site_id_ = SiteId(model_, "S_grasp");
        handle_site_id_ = SiteId(model_, "S_handle");
        door_body_id_ = BodyId(model_, "frame");
        break;
      case TaskType::kHammer:
        target_site_id_ = SiteId(model_, "S_target");
        grasp_site_id_ = SiteId(model_, "S_grasp");
        object_body_id_ = BodyId(model_, "Object");
        tool_site_id_ = SiteId(model_, "tool");
        goal_site_id_ = SiteId(model_, "nail_goal");
        target_body_id_ = BodyId(model_, "nail_board");
        nail_sensor_addr_ = model_->sensor_adr[SensorId(model_, "S_nail")];
        break;
      case TaskType::kPen:
        target_body_id_ = BodyId(model_, "target");
        grasp_site_id_ = SiteId(model_, "S_grasp");
        object_body_id_ = BodyId(model_, "Object");
        eps_ball_site_id_ = SiteId(model_, "eps_ball");
        object_top_site_id_ = SiteId(model_, "object_top");
        object_bottom_site_id_ = SiteId(model_, "object_bottom");
        target_top_site_id_ = SiteId(model_, "target_top");
        target_bottom_site_id_ = SiteId(model_, "target_bottom");
        break;
      case TaskType::kRelocate:
        target_site_id_ = SiteId(model_, "target");
        grasp_site_id_ = SiteId(model_, "S_grasp");
        object_body_id_ = BodyId(model_, "Object");
        break;
    }
  }

  void SetupActuators() {
    int wrist_begin = ActuatorId(model_, "A_WRJ1");
    int wrist_end = ActuatorId(model_, "A_WRJ0");
    int hand_begin = ActuatorId(model_, "A_FFJ3");
    int hand_end = ActuatorId(model_, "A_THJ0");
    for (int actuator_id = wrist_begin; actuator_id <= wrist_end;
         ++actuator_id) {
      model_->actuator_gainprm[mjNGAIN * actuator_id] = 10.0;
      model_->actuator_gainprm[mjNGAIN * actuator_id + 1] = 0.0;
      model_->actuator_gainprm[mjNGAIN * actuator_id + 2] = 0.0;
      model_->actuator_biasprm[mjNBIAS * actuator_id] = 0.0;
      model_->actuator_biasprm[mjNBIAS * actuator_id + 1] = -10.0;
      model_->actuator_biasprm[mjNBIAS * actuator_id + 2] = 0.0;
    }
    for (int actuator_id = hand_begin; actuator_id <= hand_end; ++actuator_id) {
      model_->actuator_gainprm[mjNGAIN * actuator_id] = 1.0;
      model_->actuator_gainprm[mjNGAIN * actuator_id + 1] = 0.0;
      model_->actuator_gainprm[mjNGAIN * actuator_id + 2] = 0.0;
      model_->actuator_biasprm[mjNBIAS * actuator_id] = 0.0;
      model_->actuator_biasprm[mjNBIAS * actuator_id + 1] = -1.0;
      model_->actuator_biasprm[mjNBIAS * actuator_id + 2] = 0.0;
    }
  }

  void SetupActionScale() {
    for (int i = 0; i < action_dim_; ++i) {
      act_mean_[i] = 0.5 * (model_->actuator_ctrlrange[2 * i] +
                            model_->actuator_ctrlrange[2 * i + 1]);
      act_rng_[i] = 0.5 * (model_->actuator_ctrlrange[2 * i + 1] -
                           model_->actuator_ctrlrange[2 * i]);
    }
  }

  void ResetTaskState() {
    ResetToInitialState();
    switch (task_type_) {
      case TaskType::kDoor:
        model_->body_pos[3 * door_body_id_] = Uniform(-0.3, -0.2);
        model_->body_pos[3 * door_body_id_ + 1] = Uniform(0.25, 0.35);
        model_->body_pos[3 * door_body_id_ + 2] = Uniform(0.252, 0.35);
        break;
      case TaskType::kHammer:
        model_->body_pos[3 * target_body_id_ + 2] = Uniform(0.1, 0.25);
        break;
      case TaskType::kPen: {
        auto desired_orien =
            Euler2Quat({Uniform(-1.0, 1.0), Uniform(-1.0, 1.0), 0.0});
        for (int i = 0; i < 4; ++i) {
          model_->body_quat[4 * target_body_id_ + i] = desired_orien[i];
        }
        break;
      }
      case TaskType::kRelocate:
        model_->body_pos[3 * object_body_id_] = Uniform(-0.15, 0.15);
        model_->body_pos[3 * object_body_id_ + 1] = Uniform(-0.15, 0.3);
        model_->site_pos[3 * target_site_id_] = Uniform(-0.2, 0.2);
        model_->site_pos[3 * target_site_id_ + 1] = Uniform(-0.2, 0.2);
        model_->site_pos[3 * target_site_id_ + 2] = Uniform(0.15, 0.35);
        break;
    }
    mj_forward(model_, data_);
    if (task_type_ == TaskType::kPen) {
      pen_length_ =
          L2(Diff(GetSiteXpos(model_, data_, object_top_site_id_),
                  GetSiteXpos(model_, data_, object_bottom_site_id_)));
      target_length_ =
          L2(Diff(GetSiteXpos(model_, data_, target_top_site_id_),
                  GetSiteXpos(model_, data_, target_bottom_site_id_)));
    }
  }

  void CaptureTaskState() {
#ifdef ENVPOOL_TEST
    switch (task_type_) {
      case TaskType::kDoor:
        for (int i = 0; i < 3; ++i) {
          extra0_[i] = model_->body_pos[3 * door_body_id_ + i];
        }
        break;
      case TaskType::kHammer:
        for (int i = 0; i < 3; ++i) {
          extra0_[i] = model_->body_pos[3 * target_body_id_ + i];
        }
        break;
      case TaskType::kPen:
        for (int i = 0; i < 4; ++i) {
          extra0_[i] = model_->body_quat[4 * target_body_id_ + i];
        }
        break;
      case TaskType::kRelocate:
        for (int i = 0; i < 3; ++i) {
          extra0_[i] = model_->body_pos[3 * object_body_id_ + i];
          extra0_[i + 3] = model_->site_pos[3 * target_site_id_ + i];
        }
        break;
    }
#endif
  }

  std::vector<mjtNum> DoorObs() const {
    std::vector<mjtNum> obs;
    obs.reserve(39);
    for (int i = 1; i < model_->nq - 2; ++i) {
      obs.push_back(data_->qpos[i]);
    }
    obs.push_back(data_->qpos[model_->nq - 1]);
    mjtNum door_pos = data_->qpos[door_hinge_addr_];
    obs.push_back(door_pos);
    auto palm_pos = GetSiteXpos(model_, data_, grasp_site_id_);
    auto handle_pos = GetSiteXpos(model_, data_, handle_site_id_);
    for (int i = 0; i < 3; ++i) {
      obs.push_back(palm_pos[i]);
    }
    for (int i = 0; i < 3; ++i) {
      obs.push_back(handle_pos[i]);
    }
    for (int i = 0; i < 3; ++i) {
      obs.push_back(palm_pos[i] - handle_pos[i]);
    }
    obs.push_back(door_pos > 1.0 ? 1.0 : -1.0);
    return obs;
  }

  std::vector<mjtNum> HammerObs() const {
    std::vector<mjtNum> obs;
    obs.reserve(46);
    for (int i = 0; i < model_->nq - 6; ++i) {
      obs.push_back(data_->qpos[i]);
    }
    for (int i = model_->nv - 6; i < model_->nv; ++i) {
      obs.push_back(std::clamp(data_->qvel[i], static_cast<mjtNum>(-1.0),
                               static_cast<mjtNum>(1.0)));
    }
    auto palm_pos = GetSiteXpos(model_, data_, grasp_site_id_);
    auto obj_pos = BodyXpos(data_, object_body_id_);
    auto obj_rot = Quat2Euler({data_->xquat[4 * object_body_id_],
                               data_->xquat[4 * object_body_id_ + 1],
                               data_->xquat[4 * object_body_id_ + 2],
                               data_->xquat[4 * object_body_id_ + 3]});
    auto target_pos = GetSiteXpos(model_, data_, target_site_id_);
    for (int i = 0; i < 3; ++i) {
      obs.push_back(palm_pos[i]);
    }
    for (int i = 0; i < 3; ++i) {
      obs.push_back(obj_pos[i]);
    }
    for (int i = 0; i < 3; ++i) {
      obs.push_back(obj_rot[i]);
    }
    for (int i = 0; i < 3; ++i) {
      obs.push_back(target_pos[i]);
    }
    obs.push_back(std::clamp(data_->sensordata[nail_sensor_addr_],
                             static_cast<mjtNum>(-1.0),
                             static_cast<mjtNum>(1.0)));
    return obs;
  }

  std::vector<mjtNum> PenObs() const {
    std::vector<mjtNum> obs;
    obs.reserve(45);
    for (int i = 0; i < model_->nq - 6; ++i) {
      obs.push_back(data_->qpos[i]);
    }
    auto obj_pos = BodyXpos(data_, object_body_id_);
    auto desired_pos = GetSiteXpos(model_, data_, eps_ball_site_id_);
    auto obj_orien =
        Orientation(object_top_site_id_, object_bottom_site_id_, pen_length_);
    auto desired_orien = Orientation(target_top_site_id_,
                                     target_bottom_site_id_, target_length_);
    for (int i = 0; i < 3; ++i) {
      obs.push_back(obj_pos[i]);
    }
    for (int i = model_->nv - 6; i < model_->nv; ++i) {
      obs.push_back(data_->qvel[i]);
    }
    for (int i = 0; i < 3; ++i) {
      obs.push_back(obj_orien[i]);
    }
    for (int i = 0; i < 3; ++i) {
      obs.push_back(desired_orien[i]);
    }
    for (int i = 0; i < 3; ++i) {
      obs.push_back(obj_pos[i] - desired_pos[i]);
    }
    for (int i = 0; i < 3; ++i) {
      obs.push_back(obj_orien[i] - desired_orien[i]);
    }
    return obs;
  }

  std::vector<mjtNum> RelocateObs() const {
    std::vector<mjtNum> obs;
    obs.reserve(39);
    for (int i = 0; i < model_->nq - 6; ++i) {
      obs.push_back(data_->qpos[i]);
    }
    auto palm_pos = GetSiteXpos(model_, data_, grasp_site_id_);
    auto obj_pos = BodyXpos(data_, object_body_id_);
    auto target_pos = GetSiteXpos(model_, data_, target_site_id_);
    for (int i = 0; i < 3; ++i) {
      obs.push_back(palm_pos[i] - obj_pos[i]);
    }
    for (int i = 0; i < 3; ++i) {
      obs.push_back(palm_pos[i] - target_pos[i]);
    }
    for (int i = 0; i < 3; ++i) {
      obs.push_back(obj_pos[i] - target_pos[i]);
    }
    return obs;
  }

  RewardInfo DoorRewardInfo() const {
    mjtNum distance = data_->qpos[door_hinge_addr_];
    bool success = distance >= 1.35;
    mjtNum reward = success ? 10.0 : -0.1;
    if (!sparse_reward_) {
      auto handle_pos = GetSiteXpos(model_, data_, handle_site_id_);
      auto palm_pos = GetSiteXpos(model_, data_, grasp_site_id_);
      reward = 0.1 * L2(Diff(palm_pos, handle_pos));
      reward += -0.1 * (distance - 1.57) * (distance - 1.57);
      mjtNum qvel_sq_sum = 0.0;
      for (int i = 0; i < model_->nv; ++i) {
        qvel_sq_sum += data_->qvel[i] * data_->qvel[i];
      }
      reward += -1e-5 * qvel_sq_sum;
      if (distance > 0.2) {
        reward += 2.0;
      }
      if (distance > 1.0) {
        reward += 8.0;
      }
      if (success) {
        reward += 10.0;
      }
    }
    return {reward, distance, success};
  }

  RewardInfo HammerRewardInfo() const {
    auto hamm_pos = BodyXpos(data_, object_body_id_);
    auto palm_pos = GetSiteXpos(model_, data_, grasp_site_id_);
    auto head_pos = GetSiteXpos(model_, data_, tool_site_id_);
    auto nail_pos = GetSiteXpos(model_, data_, target_site_id_);
    auto goal_pos = GetSiteXpos(model_, data_, goal_site_id_);
    mjtNum distance = L2(Diff(nail_pos, goal_pos));
    bool success = distance < 0.01;
    mjtNum reward = success ? 10.0 : -0.1;
    if (!sparse_reward_) {
      reward = 0.1 * L2(Diff(palm_pos, hamm_pos));
      reward -= L2(Diff(head_pos, nail_pos));
      reward -= 10.0 * distance;
      mjtNum qvel_sq_sum = 0.0;
      for (int i = 0; i < model_->nv; ++i) {
        qvel_sq_sum += data_->qvel[i] * data_->qvel[i];
      }
      reward -= 1e-2 * std::sqrt(qvel_sq_sum);
      if (hamm_pos[2] > 0.04 && head_pos[2] > 0.04) {
        reward += 2.0;
      }
      if (distance < 0.02) {
        reward += 25.0;
      }
      if (success) {
        reward += 75.0;
      }
    }
    return {reward, distance, success};
  }

  RewardInfo PenRewardInfo() const {
    auto obj_pos = BodyXpos(data_, object_body_id_);
    auto desired_pos = GetSiteXpos(model_, data_, eps_ball_site_id_);
    auto obj_orien =
        Orientation(object_top_site_id_, object_bottom_site_id_, pen_length_);
    auto desired_orien = Orientation(target_top_site_id_,
                                     target_bottom_site_id_, target_length_);
    mjtNum distance = L2(Diff(obj_pos, desired_pos));
    mjtNum similarity = Dot(obj_orien, desired_orien);
    bool success = distance < 0.075 && similarity > 0.95;
    mjtNum reward = success ? 10.0 : -0.1;
    if (!sparse_reward_) {
      reward = -distance + similarity;
      if (distance < 0.075 && similarity > 0.9) {
        reward += 10.0;
      }
      if (success) {
        reward += 50.0;
      }
      if (obj_pos[2] < 0.075) {
        reward -= 5.0;
      }
    }
    return {reward, distance, success};
  }

  RewardInfo RelocateRewardInfo() const {
    auto palm_pos = GetSiteXpos(model_, data_, grasp_site_id_);
    auto obj_pos = BodyXpos(data_, object_body_id_);
    auto target_pos = GetSiteXpos(model_, data_, target_site_id_);
    mjtNum distance = L2(Diff(obj_pos, target_pos));
    bool success = distance < 0.1;
    mjtNum reward = success ? 10.0 : -0.1;
    if (!sparse_reward_) {
      reward = 0.1 * L2(Diff(palm_pos, obj_pos));
      if (obj_pos[2] > 0.04) {
        reward += 1.0;
        reward += -0.5 * L2(Diff(palm_pos, target_pos));
        reward += -0.5 * distance;
      }
      if (distance < 0.1) {
        reward += 10.0;
      }
      if (distance < 0.05) {
        reward += 20.0;
      }
    }
    return {reward, distance, success};
  }

  std::vector<mjtNum> Observation() const {
    switch (task_type_) {
      case TaskType::kDoor:
        return DoorObs();
      case TaskType::kHammer:
        return HammerObs();
      case TaskType::kPen:
        return PenObs();
      case TaskType::kRelocate:
        return RelocateObs();
    }
    throw std::runtime_error("Unknown Adroit task type.");
  }

  RewardInfo ComputeRewardInfo() const {
    switch (task_type_) {
      case TaskType::kDoor:
        return DoorRewardInfo();
      case TaskType::kHammer:
        return HammerRewardInfo();
      case TaskType::kPen:
        return PenRewardInfo();
      case TaskType::kRelocate:
        return RelocateRewardInfo();
    }
    throw std::runtime_error("Unknown Adroit task type.");
  }

  void WriteState(mjtNum reward, mjtNum distance = 0.0, bool success = false) {
    State state = Allocate();
    auto obs = Observation();
    if (static_cast<int>(obs.size()) != obs_dim_) {
      throw std::runtime_error("Unexpected Adroit observation size.");
    }
    state["obs"_].Assign(obs.data(), obs.size());
    state["reward"_] = static_cast<float>(reward);
    state["info:success"_] = success ? 1.0 : 0.0;
    state["info:distance"_] = distance;
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
    state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
    state["info:extra0"_].Assign(extra0_.data(), extra0_.size());
#endif
  }
};

using AdroitEnvPool = AsyncEnvPool<AdroitEnv>;

}  // namespace gymnasium_robotics

#endif  // ENVPOOL_GYMNASIUM_ROBOTICS_ADROIT_H_
