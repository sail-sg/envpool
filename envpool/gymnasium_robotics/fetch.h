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

#ifndef ENVPOOL_GYMNASIUM_ROBOTICS_FETCH_H_
#define ENVPOOL_GYMNASIUM_ROBOTICS_FETCH_H_

#include <mujoco.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/gymnasium_robotics/mujoco_env.h"
#include "envpool/gymnasium_robotics/utils.h"

namespace gymnasium_robotics {

class FetchEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(20),
        "xml_file"_.Bind(std::string("fetch/reach.xml")),
        "reward_type"_.Bind(std::string("sparse")),
        "has_object"_.Bind(false), "block_gripper"_.Bind(true),
        "target_in_the_air"_.Bind(true), "gripper_extra_height"_.Bind(0.2),
        "target_offset_x"_.Bind(0.0), "target_offset_y"_.Bind(0.0),
        "target_offset_z"_.Bind(0.0), "obj_range"_.Bind(0.15),
        "target_range"_.Bind(0.15), "distance_threshold"_.Bind(0.05),
        "initial_slide0"_.Bind(0.4049), "initial_slide1"_.Bind(0.48),
        "initial_slide2"_.Bind(0.0), "initial_object_x"_.Bind(1.25),
        "initial_object_y"_.Bind(0.53), "initial_object_z"_.Bind(0.4));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    int obs_dim = conf["has_object"_] ? 25 : 10;
#ifdef ENVPOOL_TEST
    int qpos_dim = conf["has_object"_] ? 22 : 15;
    int qvel_dim = conf["has_object"_] ? 21 : 15;
#endif
    return MakeDict(
        "obs:observation"_.Bind(Spec<mjtNum>({obs_dim}, {-inf, inf})),
        "obs:achieved_goal"_.Bind(Spec<mjtNum>({3}, {-inf, inf})),
        "obs:desired_goal"_.Bind(Spec<mjtNum>({3}, {-inf, inf})),
        "info:is_success"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({qpos_dim})),
        "info:qvel0"_.Bind(Spec<mjtNum>({qvel_dim})),
        "info:goal0"_.Bind(Spec<mjtNum>({3})),
#endif
        "info:distance"_.Bind(Spec<mjtNum>({-1}, {0.0, inf})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<float>({-1, 4}, {-1.0, 1.0})));
  }
};

using FetchEnvSpec = EnvSpec<FetchEnvFns>;

class FetchEnv : public Env<FetchEnvSpec>, public MujocoRobotEnv {
 protected:
  bool has_object_, block_gripper_, target_in_the_air_, sparse_reward_;
  mjtNum gripper_extra_height_, obj_range_, target_range_, distance_threshold_;
  std::array<mjtNum, 3> target_offset_{};
  std::array<mjtNum, 3> goal_{};
  std::array<mjtNum, 3> initial_gripper_xpos_{};
  mjtNum height_offset_{0.0};
  int grip_site_id_, object_site_id_, target_site_id_;
  std::uniform_real_distribution<> goal_dist_;
  std::uniform_real_distribution<> obj_dist_;
  std::uniform_real_distribution<> air_goal_dist_;
  std::uniform_real_distribution<> coin_dist_;

 public:
  FetchEnv(const Spec& spec, int env_id)
      : Env<FetchEnvSpec>(spec, env_id),
        MujocoRobotEnv(spec.config["base_path"_], spec.config["xml_file"_],
                       spec.config["frame_skip"_],
                       spec.config["max_episode_steps"_]),
        has_object_(spec.config["has_object"_]),
        block_gripper_(spec.config["block_gripper"_]),
        target_in_the_air_(spec.config["target_in_the_air"_]),
        sparse_reward_(spec.config["reward_type"_] == "sparse"),
        gripper_extra_height_(spec.config["gripper_extra_height"_]),
        obj_range_(spec.config["obj_range"_]),
        target_range_(spec.config["target_range"_]),
        distance_threshold_(spec.config["distance_threshold"_]),
        target_offset_({spec.config["target_offset_x"_],
                        spec.config["target_offset_y"_],
                        spec.config["target_offset_z"_]}),
        grip_site_id_(SiteId(model_, "robot0:grip")),
        object_site_id_(has_object_ ? SiteId(model_, "object0") : -1),
        target_site_id_(SiteId(model_, "target0")),
        goal_dist_(-target_range_, target_range_),
        obj_dist_(-obj_range_, obj_range_),
        air_goal_dist_(0.0, 0.45),
        coin_dist_(0.0, 1.0) {
    EnvSetup();
    InitializeRobotEnv();
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    while (!ResetSim()) {
    }
    goal_ = SampleGoal();
    CaptureResetState();
    WriteState(0.0);
  }

  void Step(const Action& action) override {
    std::vector<mjtNum> ctrl_action =
        BuildAction(static_cast<float*>(action["action"_].Data()));
    CtrlSetAction(model_, data_, ctrl_action);
    MocapSetAction(model_, data_, ctrl_action);
    DoSimulation();
    StepCallback();
    ++elapsed_step_;
    done_ = elapsed_step_ >= max_episode_steps_;
    auto achieved_goal = AchievedGoal();
    mjtNum distance = GoalDistance(achieved_goal, goal_);
    mjtNum reward = sparse_reward_ ? (distance > distance_threshold_ ? -1.0 : 0.0)
                                   : -distance;
    WriteState(static_cast<float>(reward));
  }

 protected:
  bool ResetSim() {
    ResetToInitialState();
    if (has_object_) {
      std::array<mjtNum, 2> object_xpos{
          initial_gripper_xpos_[0],
          initial_gripper_xpos_[1],
      };
      while (true) {
        object_xpos[0] = initial_gripper_xpos_[0] + obj_dist_(gen_);
        object_xpos[1] = initial_gripper_xpos_[1] + obj_dist_(gen_);
        mjtNum dx = object_xpos[0] - initial_gripper_xpos_[0];
        mjtNum dy = object_xpos[1] - initial_gripper_xpos_[1];
        if (std::sqrt(dx * dx + dy * dy) >= 0.1) {
          break;
        }
      }
      auto object_qpos = GetJointQpos(model_, data_, "object0:joint");
      object_qpos[0] = object_xpos[0];
      object_qpos[1] = object_xpos[1];
      SetJointQpos(model_, data_, "object0:joint", object_qpos);
    }
    mj_forward(model_, data_);
    return true;
  }

  void EnvSetup() override {
    SetJointQpos(model_, data_, "robot0:slide0", spec_.config["initial_slide0"_]);
    SetJointQpos(model_, data_, "robot0:slide1", spec_.config["initial_slide1"_]);
    SetJointQpos(model_, data_, "robot0:slide2", spec_.config["initial_slide2"_]);
    if (spec_.config["has_object"_]) {
      SetJointQpos(model_, data_, "object0:joint",
                   std::vector<mjtNum>{spec_.config["initial_object_x"_],
                                       spec_.config["initial_object_y"_],
                                       spec_.config["initial_object_z"_],
                                       1.0, 0.0, 0.0, 0.0});
    }
    ResetMocapWelds(model_, data_);
    mj_forward(model_, data_);

    auto grip_pos = GetSiteXpos(model_, data_, SiteId(model_, "robot0:grip"));
    std::vector<mjtNum> gripper_target{
        grip_pos[0] - 0.498,
        grip_pos[1] + 0.005,
        grip_pos[2] - 0.431 + spec_.config["gripper_extra_height"_],
    };
    SetMocapPos("robot0:mocap", gripper_target);
    SetMocapQuat("robot0:mocap", {1.0, 0.0, 1.0, 0.0});
    for (int i = 0; i < 10; ++i) {
      DoSimulation();
    }

    initial_gripper_xpos_ = GetSiteXpos(model_, data_, SiteId(model_, "robot0:grip"));
    if (spec_.config["has_object"_]) {
      height_offset_ = GetSiteXpos(model_, data_, SiteId(model_, "object0"))[2];
    }
  }

  void StepCallback() override {
    if (!block_gripper_) {
      return;
    }
    SetJointQpos(model_, data_, "robot0:l_gripper_finger_joint", 0.0);
    SetJointQpos(model_, data_, "robot0:r_gripper_finger_joint", 0.0);
    mj_forward(model_, data_);
  }

  void RenderCallback() override {
    std::array<mjtNum, 3> target_xpos = GetSiteXpos(model_, data_, target_site_id_);
    for (int i = 0; i < 3; ++i) {
      model_->site_pos[3 * target_site_id_ + i] =
          goal_[i] - (target_xpos[i] - model_->site_pos[3 * target_site_id_ + i]);
    }
    mj_forward(model_, data_);
  }

 private:
  void SetMocapPos(const std::string& body_name,
                   const std::vector<mjtNum>& value) {
    int body_id = BodyId(model_, body_name);
    int mocap_id = model_->body_mocapid[body_id];
    for (int i = 0; i < 3; ++i) {
      data_->mocap_pos[3 * mocap_id + i] = value[i];
    }
  }

  void SetMocapQuat(const std::string& body_name,
                    const std::vector<mjtNum>& value) {
    int body_id = BodyId(model_, body_name);
    int mocap_id = model_->body_mocapid[body_id];
    for (int i = 0; i < 4; ++i) {
      data_->mocap_quat[4 * mocap_id + i] = value[i];
    }
  }

  std::vector<mjtNum> BuildAction(const float* raw_action) const {
    std::vector<mjtNum> action{
        static_cast<mjtNum>(std::clamp(raw_action[0], -1.0F, 1.0F)) *
            mjtNum{0.05},
        static_cast<mjtNum>(std::clamp(raw_action[1], -1.0F, 1.0F)) *
            mjtNum{0.05},
        static_cast<mjtNum>(std::clamp(raw_action[2], -1.0F, 1.0F)) *
            mjtNum{0.05},
        1.0,
        0.0,
        1.0,
        0.0,
        block_gripper_ ? mjtNum{0.0}
                       : static_cast<mjtNum>(
                             std::clamp(raw_action[3], -1.0F, 1.0F)),
        block_gripper_ ? mjtNum{0.0}
                       : static_cast<mjtNum>(
                             std::clamp(raw_action[3], -1.0F, 1.0F)),
    };
    return action;
  }

  std::array<mjtNum, 3> SampleGoal() {
    std::array<mjtNum, 3> goal{
        initial_gripper_xpos_[0] + goal_dist_(gen_) + target_offset_[0],
        initial_gripper_xpos_[1] + goal_dist_(gen_) + target_offset_[1],
        initial_gripper_xpos_[2] + goal_dist_(gen_) + target_offset_[2],
    };
    if (has_object_) {
      goal[2] = height_offset_;
      if (target_in_the_air_ && coin_dist_(gen_) < 0.5) {
        goal[2] += air_goal_dist_(gen_);
      }
    }
    return goal;
  }

  std::array<mjtNum, 3> AchievedGoal() const {
    return has_object_ ? GetSiteXpos(model_, data_, object_site_id_)
                       : GetSiteXpos(model_, data_, grip_site_id_);
  }

  mjtNum GoalDistance(const std::array<mjtNum, 3>& a,
                      const std::array<mjtNum, 3>& b) const {
    mjtNum dx = a[0] - b[0];
    mjtNum dy = a[1] - b[1];
    mjtNum dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  void WriteState(float reward) {
    auto state = Allocate();
    state["reward"_] = reward;

    mjtNum* obs = static_cast<mjtNum*>(state["obs:observation"_].Data());
    auto grip_pos = GetSiteXpos(model_, data_, grip_site_id_);
    auto grip_velp = GetSiteXvelp(model_, data_, grip_site_id_);
    auto [robot_qpos, robot_qvel] = RobotGetObs(model_, data_);
    for (int i = 0; i < 3; ++i) {
      *(obs++) = grip_pos[i];
    }
    if (has_object_) {
      auto object_pos = GetSiteXpos(model_, data_, object_site_id_);
      auto object_rot = Mat2Euler(GetSiteXmat(model_, data_, object_site_id_));
      auto object_velp = GetSiteXvelp(model_, data_, object_site_id_);
      auto object_velr = GetSiteXvelr(model_, data_, object_site_id_);
      for (int i = 0; i < 3; ++i) {
        object_velp[i] = (object_velp[i] - grip_velp[i]) * Dt();
        object_velr[i] *= Dt();
        *(obs++) = object_pos[i];
      }
      for (int i = 0; i < 3; ++i) {
        *(obs++) = object_pos[i] - grip_pos[i];
      }
      for (int i = 0; i < 2; ++i) {
        *(obs++) = robot_qpos[robot_qpos.size() - 2 + i];
      }
      for (int i = 0; i < 3; ++i) {
        *(obs++) = object_rot[i];
      }
      for (int i = 0; i < 3; ++i) {
        *(obs++) = object_velp[i];
      }
      for (int i = 0; i < 3; ++i) {
        *(obs++) = object_velr[i];
      }
    } else {
      for (int i = 0; i < 2; ++i) {
        *(obs++) = robot_qpos[robot_qpos.size() - 2 + i];
      }
    }
    for (int i = 0; i < 3; ++i) {
      *(obs++) = grip_velp[i] * Dt();
    }
    for (int i = 0; i < 2; ++i) {
      *(obs++) = robot_qvel[robot_qvel.size() - 2 + i] * Dt();
    }

    auto achieved_goal = AchievedGoal();
    state["obs:achieved_goal"_].Assign(achieved_goal.data(), 3);
    state["obs:desired_goal"_].Assign(goal_.data(), 3);
    mjtNum distance = GoalDistance(achieved_goal, goal_);
    state["info:is_success"_] = distance < distance_threshold_ ? 1.0 : 0.0;
    state["info:distance"_] = distance;
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
    state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
    state["info:goal0"_].Assign(goal_.data(), goal_.size());
#endif
  }
};

using FetchEnvPool = AsyncEnvPool<FetchEnv>;

}  // namespace gymnasium_robotics

#endif  // ENVPOOL_GYMNASIUM_ROBOTICS_FETCH_H_
