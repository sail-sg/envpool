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
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/manipulator.py

#ifndef ENVPOOL_MUJOCO_DMC_MANIPULATOR_H_
#define ENVPOOL_MUJOCO_DMC_MANIPULATOR_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <regex>
#include <set>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/dmc/mujoco_env.h"
#include "envpool/mujoco/dmc/utils.h"

namespace mujoco_dmc {

std::string GetManipulatorXML(const std::string& base_path,
                              const std::string& task_name) {
  auto content = GetFileContent(base_path, "manipulator.xml");
  if (task_name == "bring_ball") {
    return XMLRemoveByBodyName(content, {"slot", "target_peg", "cup", "peg"});
  }
  if (task_name == "bring_peg") {
    return XMLRemoveByBodyName(content, {"slot", "target_ball", "cup", "ball"});
  }
  if (task_name == "insert_ball") {
    return XMLRemoveByBodyName(content, {"slot", "target_peg", "peg"});
  }
  if (task_name == "insert_peg") {
    return XMLRemoveByBodyName(content, {"target_ball", "cup", "ball"});
  }
  throw std::runtime_error("Unknown task_name " + task_name +
                           " for dmc manipulator.");
}

class ManipulatorEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("frame_skip"_.Bind(10),
                    "task_name"_.Bind(std::string("bring_ball")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs:arm_pos"_.Bind(Spec<mjtNum>({8, 2})),
                    "obs:arm_vel"_.Bind(Spec<mjtNum>({8})),
                    "obs:touch"_.Bind(Spec<mjtNum>({5})),
                    "obs:hand_pos"_.Bind(Spec<mjtNum>({4})),
                    "obs:object_pos"_.Bind(Spec<mjtNum>({4})),
                    "obs:object_vel"_.Bind(Spec<mjtNum>({3})),
                    "obs:target_pos"_.Bind(Spec<mjtNum>({4}))
#ifdef ENVPOOL_TEST
                        ,
                    "info:qpos0"_.Bind(Spec<mjtNum>({11})),
                    "info:random_info"_.Bind(Spec<mjtNum>({8}))
#endif
    );  // NOLINT
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 5}, {-1.0, 1.0})));
  }
};

using ManipulatorEnvSpec = EnvSpec<ManipulatorEnvFns>;

class ManipulatorEnv : public Env<ManipulatorEnvSpec>, public MujocoEnv {
 protected:
  const mjtNum kClose = 0.01;
  const mjtNum kPInHand = 0.1;
  const mjtNum kPInTarget = 0.1;
  const std::array<std::string, 8> kArmJoints = {
      "arm_root", "arm_shoulder", "arm_elbow", "arm_wrist",
      "finger",   "fingertip",    "thumb",     "thumbtip"};
  const std::array<std::string, 6> kAllProps = {"ball", "target_ball", "cup",
                                                "peg",  "target_peg",  "slot"};
  const std::array<std::string, 5> kTouchSensors = {
      "palm_touch", "finger_touch", "thumb_touch", "fingertip_touch",
      "thumbtip_touch"};

  bool use_peg_, insert_;
  std::array<mjtNum, 8> random_info_;
  // target_x, target_z, target_angle, init_type, object_x, object_z,
  // object_angle, qvel_objx

  // ids
  std::array<int, 8> id_arm_joints_, id_arm_qpos_, id_arm_qvel_;
  int id_finger_, id_thumb_, id_body_receptacle_, id_body_target_;
  int id_xbody_hand_, id_xbody_object_, id_xbody_target_, id_object_x_;
  std::array<int, 3> id_qpos_object_joints_, id_qvel_object_joints_;
  std::array<int, 5> id_touch_sensors_;
  int id_site_peg_grasp_, id_site_grasp_, id_site_peg_pinch_, id_site_pinch_;
  int id_site_peg_, id_site_target_peg_, id_site_target_peg_tip_;
  int id_site_peg_tip_, id_site_ball_, id_site_target_ball_;

 public:
  ManipulatorEnv(const Spec& spec, int env_id)
      : Env<ManipulatorEnvSpec>(spec, env_id),
        MujocoEnv(spec.config["base_path"_],
                  GetManipulatorXML(spec.config["base_path"_],
                                    spec.config["task_name"_]),
                  spec.config["frame_skip"_],
                  spec.config["max_episode_steps"_]),
        use_peg_(spec.config["task_name"_] == "bring_peg" ||
                 spec.config["task_name"_] == "insert_peg"),
        insert_(spec.config["task_name"_] == "insert_peg" ||
                spec.config["task_name"_] == "insert_ball"),
        id_finger_(GetQposId(model_, "finger")),
        id_thumb_(GetQposId(model_, "thumb")),
        id_body_receptacle_(
            mj_name2id(model_, mjOBJ_BODY, use_peg_ ? "slot" : "cup")),
        id_body_target_(mj_name2id(model_, mjOBJ_BODY,
                                   use_peg_ ? "target_peg" : "target_ball")),
        id_xbody_hand_(mj_name2id(model_, mjOBJ_XBODY, "hand")),
        id_xbody_object_(
            mj_name2id(model_, mjOBJ_XBODY, use_peg_ ? "peg" : "ball")),
        id_xbody_target_(mj_name2id(model_, mjOBJ_XBODY,
                                    use_peg_ ? "target_peg" : "target_ball")),
        id_object_x_(GetQvelId(model_, use_peg_ ? "peg_x" : "ball_x")),
        id_site_peg_grasp_(mj_name2id(model_, mjOBJ_SITE, "peg_grasp")),
        id_site_grasp_(mj_name2id(model_, mjOBJ_SITE, "grasp")),
        id_site_peg_pinch_(mj_name2id(model_, mjOBJ_SITE, "peg_pinch")),
        id_site_pinch_(mj_name2id(model_, mjOBJ_SITE, "pinch")),
        id_site_peg_(mj_name2id(model_, mjOBJ_SITE, "peg")),
        id_site_target_peg_(mj_name2id(model_, mjOBJ_SITE, "target_peg")),
        id_site_target_peg_tip_(
            mj_name2id(model_, mjOBJ_SITE, "target_peg_tip")),
        id_site_peg_tip_(mj_name2id(model_, mjOBJ_SITE, "peg_tip")),
        id_site_ball_(mj_name2id(model_, mjOBJ_SITE, "ball")),
        id_site_target_ball_(mj_name2id(model_, mjOBJ_SITE, "target_ball")) {
    for (std::size_t i = 0; i < kArmJoints.size(); ++i) {
      id_arm_joints_[i] =
          mj_name2id(model_, mjOBJ_JOINT, kArmJoints[i].c_str());
      id_arm_qpos_[i] = GetQposId(model_, kArmJoints[i]);
      id_arm_qvel_[i] = GetQvelId(model_, kArmJoints[i]);
    }
    std::array<std::string, 3> object_joints;
    if (use_peg_) {
      object_joints = {"peg_x", "peg_z", "peg_y"};
    } else {
      object_joints = {"ball_x", "ball_z", "ball_y"};
    }
    for (std::size_t i = 0; i < object_joints.size(); ++i) {
      id_qpos_object_joints_[i] = GetQposId(model_, object_joints[i]);
      id_qvel_object_joints_[i] = GetQvelId(model_, object_joints[i]);
    }
    for (std::size_t i = 0; i < kTouchSensors.size(); ++i) {
      id_touch_sensors_[i] = GetSensorId(model_, kTouchSensors[i]);
    }
  }

  void TaskInitializeEpisode() override {
    bool penetrating = true;
    while (penetrating) {
      for (std::size_t i = 0; i < kArmJoints.size(); ++i) {
        int id_joint = id_arm_joints_[i];
        bool is_limited = model_->jnt_limited[id_joint] == 1 ? true : false;
        mjtNum lower = is_limited ? model_->jnt_range[id_joint * 2 + 0] : -M_PI;
        mjtNum upper = is_limited ? model_->jnt_range[id_joint * 2 + 1] : M_PI;
        data_->qpos[id_arm_qpos_[i]] = RandUniform(lower, upper)(gen_);
      }
      data_->qpos[id_finger_] = data_->qpos[id_thumb_];
#ifdef ENVPOOL_TEST
      std::memcpy(qpos0_.get(), data_->qpos, sizeof(mjtNum) * model_->nq);
#endif
      mjtNum target_x = random_info_[0] = RandUniform(-0.4, 0.4)(gen_);
      mjtNum target_z = RandUniform(0.1, 0.4)(gen_);
      mjtNum target_angle;
      if (insert_) {
        target_angle = RandUniform(-M_PI / 3, M_PI / 3)(gen_);
        // model.body_pos[self._receptacle, ['x', 'z']]
        model_->body_pos[id_body_receptacle_ * 3 + 0] = target_x;
        model_->body_pos[id_body_receptacle_ * 3 + 2] = target_z;
        // model.body_quat[self._receptacle, ['qw', 'qy']]
        model_->body_quat[id_body_receptacle_ * 4 + 0] =
            std::cos(target_angle / 2);
        model_->body_quat[id_body_receptacle_ * 4 + 2] =
            std::sin(target_angle / 2);
      } else {
        target_angle = RandUniform(-M_PI, M_PI)(gen_);
      }
      random_info_[0] = target_x;
      random_info_[1] = target_z;
      random_info_[2] = target_angle;

      // model.body_pos[self._target, ['x', 'z']] = target_x, target_z
      // model.body_quat[self._target, ['qw', 'qy']] = [
      //     np.cos(target_angle/2), np.sin(target_angle/2)]
      model_->body_pos[id_body_target_ * 3 + 0] = target_x;
      model_->body_pos[id_body_target_ * 3 + 2] = target_z;
      model_->body_quat[id_body_target_ * 4 + 0] = std::cos(target_angle / 2);
      model_->body_quat[id_body_target_ * 4 + 2] = std::sin(target_angle / 2);

      mjtNum choice = RandUniform(0, 1)(gen_);
      mjtNum object_x;
      mjtNum object_z;
      mjtNum object_angle;
      if (choice <= kPInTarget) {
        // in_target
        random_info_[3] = 1;
        object_x = target_x;
        object_z = target_z;
        object_angle = target_angle;
      } else if (choice <= kPInTarget + kPInHand) {
        // in_hand
        random_info_[3] = 2;
        // physics.after_reset()
        // object_x = data.site_xpos['grasp', 'x']
        // object_z = data.site_xpos['grasp', 'z']
        // grasp_direction = data.site_xmat['grasp', ['xx', 'zx']]
        // object_angle = np.pi-np.arctan2(grasp_direction[1],
        // grasp_direction[0])
        PhysicsAfterReset();
        object_x = data_->site_xpos[id_site_grasp_ * 3 + 0];
        object_z = data_->site_xpos[id_site_grasp_ * 3 + 2];
        std::array<mjtNum, 2> grasp_direction = {
            data_->site_xmat[id_site_grasp_ * 9 + 0],
            data_->site_xmat[id_site_grasp_ * 9 + 6]};
        object_angle =
            M_PI - std::atan2(grasp_direction[1], grasp_direction[0]);
      } else {
        // uniform
        random_info_[3] = 3;
        // object_x = uniform(-.5, .5)
        // object_z = uniform(0, .7)
        // object_angle = uniform(0, 2*np.pi)
        // data.qvel[self._object + '_x'] = uniform(-5, 5)
        object_x = RandUniform(-0.5, 0.5)(gen_);
        object_z = RandUniform(0, 0.7)(gen_);
        object_angle = RandUniform(0, M_PI * 2)(gen_);
        data_->qvel[id_object_x_] = random_info_[7] = RandUniform(-5, 5)(gen_);
      }
      data_->qpos[id_qpos_object_joints_[0]] = random_info_[4] = object_x;
      data_->qpos[id_qpos_object_joints_[1]] = random_info_[5] = object_z;
      data_->qpos[id_qpos_object_joints_[2]] = random_info_[6] = object_angle;
      // Check for collisions.
      PhysicsAfterReset();
      penetrating = data_->ncon > 0;
    }
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    ControlReset();
    WriteState();
  }

  void Step(const Action& action) override {
    mjtNum* act = static_cast<mjtNum*>(action["action"_].Data());
    ControlStep(act);
    WriteState();
  }

  float TaskGetReward() override {
    if (use_peg_) {
      return static_cast<float>(PegReward());
    }
    return static_cast<float>(BallReward());
  }

  bool TaskShouldTerminateEpisode() override { return false; }

 private:
  std::array<mjtNum, 16> BoundedJointPos() {
    std::array<mjtNum, 16> bound;
    for (std::size_t i = 0; i < id_arm_qpos_.size(); i++) {
      bound[i * 2 + 0] = std::sin(data_->qpos[id_arm_qpos_[i]]);
      bound[i * 2 + 1] = std::cos(data_->qpos[id_arm_qpos_[i]]);
    }
    return bound;
  }

  // bug :(
  std::array<mjtNum, 8> JointVelArm() {
    std::array<mjtNum, 8> joint;
    for (std::size_t i = 0; i < id_arm_qvel_.size(); i++) {
      joint[i] = data_->qvel[id_arm_qvel_[i]];
    }
    return joint;
  }

  std::array<mjtNum, 3> JointVelObj() {
    std::array<mjtNum, 3> joint;
    for (std::size_t i = 0; i < id_qvel_object_joints_.size(); i++) {
      joint[i] = data_->qvel[id_qvel_object_joints_[i]];
    }
    return joint;
  }

  std::array<mjtNum, 5> Touch() {
    std::array<mjtNum, 5> touch;
    for (std::size_t i = 0; i < id_touch_sensors_.size(); i++) {
      touch[i] = std::log1p(data_->sensordata[id_touch_sensors_[i]]);
    }
    return touch;
  }

  std::array<mjtNum, 4> Body2dPose(int id) {
    // self.named.data.xpos[body_names, ['x', 'z']]
    // self.named.data.xquat[body_names, ['qw', 'qy']]
    return {data_->xpos[id * 3 + 0], data_->xpos[id * 3 + 2],
            data_->xquat[id * 4 + 0], data_->xquat[id * 4 + 2]};
  }

  mjtNum SiteDistance(int id_site1, int id_site2) {
    std::array<mjtNum, 3> diff = {
        data_->site_xpos[id_site1 * 3 + 0] - data_->site_xpos[id_site2 * 3 + 0],
        data_->site_xpos[id_site1 * 3 + 1] - data_->site_xpos[id_site2 * 3 + 1],
        data_->site_xpos[id_site1 * 3 + 2] -
            data_->site_xpos[id_site2 * 3 + 2]};
    return std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
  }

  [[nodiscard]] mjtNum IsClose(mjtNum distance) const {
    // return rewards.tolerance(distance, (0, _CLOSE), _CLOSE * 2)
    return RewardTolerance(distance, 0, kClose, kClose * 2);
  }

  mjtNum PegReward() {
    auto grasping =
        (IsClose(SiteDistance(id_site_peg_grasp_, id_site_grasp_)) +
         IsClose(SiteDistance(id_site_peg_pinch_, id_site_pinch_))) /
        2;
    auto bringing =
        (IsClose(SiteDistance(id_site_peg_, id_site_target_peg_)) +
         IsClose(SiteDistance(id_site_target_peg_tip_, id_site_peg_tip_))) /
        2;
    return std::max(bringing, grasping / 3);
  }
  mjtNum BallReward() {
    // return self._is_close(physics.site_distance('ball', 'target_ball'))
    return IsClose(SiteDistance(id_site_ball_, id_site_target_ball_));
  }

  void WriteState() {
    const auto& bounded_joint_pos = BoundedJointPos();
    const auto& joint_vel_arm = JointVelArm();
    const auto& touch = Touch();
    const auto& hand_pos = Body2dPose(id_xbody_hand_);
    const auto& object_pos = Body2dPose(id_xbody_object_);
    const auto& joint_vel_obj = JointVelObj();
    const auto& target_pos = Body2dPose(id_xbody_target_);

    State state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    // obs
    state["obs:arm_pos"_].Assign(bounded_joint_pos.begin(),
                                 bounded_joint_pos.size());
    state["obs:arm_vel"_].Assign(joint_vel_arm.begin(), joint_vel_arm.size());
    state["obs:touch"_].Assign(touch.begin(), touch.size());
    state["obs:hand_pos"_].Assign(hand_pos.begin(), hand_pos.size());
    state["obs:object_pos"_].Assign(object_pos.begin(), object_pos.size());
    state["obs:object_vel"_].Assign(joint_vel_obj.begin(),
                                    joint_vel_obj.size());
    state["obs:target_pos"_].Assign(target_pos.begin(), target_pos.size());
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
    state["info:random_info"_].Assign(random_info_.begin(), 8);
#endif
  }
};

using ManipulatorEnvPool = AsyncEnvPool<ManipulatorEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_MANIPULATOR_H_
