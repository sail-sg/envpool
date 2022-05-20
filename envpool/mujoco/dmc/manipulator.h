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
    return MakeDict("max_episode_steps"_.Bind(1000), "frame_skip"_.Bind(10),
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
                    "obs:target_pos"_.Bind(Spec<mjtNum>({4})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.Bind(Spec<mjtNum>({14})),
                    "info:qvel"_.Bind(Spec<mjtNum>({14})),
                    "info:body_pos"_.Bind(Spec<mjtNum>({16, 4})),
                    "info:body_quat"_.Bind(Spec<mjtNum>({16, 3})),
#endif
                    "discount"_.Bind(Spec<float>({-1}, {0.0, 1.0})));
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
  std::string target_, object_;
  std::array<std::string, 3> object_joints_;
  std::string receptacle_;
  std::uniform_real_distribution<> dist_uniform_;
#ifdef ENVPOOL_TEST
  std::unique_ptr<mjtNum> qvel_;
  std::unique_ptr<mjtNum> body_pos_;
  std::unique_ptr<mjtNum> body_quat_;
#endif

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
        target_(use_peg_ ? "target_peg" : "target_ball"),
        object_(use_peg_ ? "peg" : "ball"),
        receptacle_(use_peg_ ? "slot" : "cup"),
        dist_uniform_(0, 1) {
    if (use_peg_) {
      object_joints_ = {"peg_x", "peg_z", "peg_y"};
    } else {
      object_joints_ = {"ball_x", "ball_z", "ball_y"};
    }
#ifdef ENVPOOL_TEST
    qvel_.reset(new mjtNum[model_->nv]);
    body_pos_.reset(new mjtNum[model_->nbody * 3]);
    body_quat_.reset(new mjtNum[model_->nbody * 4]);
#endif
  }

  void TaskInitializeEpisode() override {
    bool penetrating = true;
    while (penetrating) {
      for (const auto& arm_joint : kArmJoints) {
        int id_joint = mj_name2id(model_, mjOBJ_JOINT, arm_joint.c_str());
        bool is_limited = model_->jnt_limited[id_joint] == 1 ? true : false;
        mjtNum lower = is_limited ? model_->jnt_range[id_joint * 2 + 0] : -M_PI;
        mjtNum upper = is_limited ? model_->jnt_range[id_joint * 2 + 1] : M_PI;
        data_->qpos[model_->jnt_qposadr[id_joint]] =
            dist_uniform_(gen_) * (upper - lower) + lower;
      }
      data_->qpos[mj_name2id(model_, mjOBJ_JOINT, "finger")] =
          data_->qpos[mj_name2id(model_, mjOBJ_JOINT, "thumb")];
      mjtNum target_x = dist_uniform_(gen_) * 0.4 - 0.4;
      mjtNum target_z = dist_uniform_(gen_) * 0.3 + 0.1;
      mjtNum target_angle;
      if (insert_) {
        target_angle = dist_uniform_(gen_) * (M_PI * 2 / 3) - M_PI / 3;
        int id_body_receptacle =
            mj_name2id(model_, mjOBJ_JOINT, receptacle_.c_str());
        // model.body_pos[self._receptacle, ['x', 'z']]
        model_->body_pos[id_body_receptacle * 3 + 0] = target_x;
        model_->body_pos[id_body_receptacle * 3 + 2] = target_z;
        // model.body_quat[self._receptacle, ['qw', 'qy']]
        model_->body_quat[id_body_receptacle * 4 + 0] =
            std::cos(target_angle / 2);
        model_->body_quat[id_body_receptacle * 4 + 2] =
            std::sin(target_angle / 2);
      } else {
        target_angle = dist_uniform_(gen_) * 2 * M_PI - M_PI;
      }
      int id_body_target = mj_name2id(model_, mjOBJ_JOINT, target_.c_str());
      //   model.body_pos[self._target, ['x', 'z']] = target_x, target_z
      //   model.body_quat[self._target, ['qw', 'qy']] = [
      //       np.cos(target_angle/2), np.sin(target_angle/2)]
      model_->body_pos[id_body_target * 3 + 0] = target_x;
      model_->body_pos[id_body_target * 3 + 2] = target_z;
      model_->body_quat[id_body_target * 4 + 0] = std::cos(target_angle / 2);
      model_->body_quat[id_body_target * 4 + 2] = std::sin(target_angle / 2);

      mjtNum choice = dist_uniform_(gen_);
      mjtNum object_x;
      mjtNum object_z;
      mjtNum object_angle;
      if (choice <= kPInHand) {
        // in_hand
        object_x = target_x;
        object_z = target_z;
        object_angle = target_angle;
      } else if (choice <= kPInHand + kPInTarget) {
        // in_target
        // physics.after_reset()
        // object_x = data.site_xpos['grasp', 'x']
        // object_z = data.site_xpos['grasp', 'z']
        // grasp_direction = data.site_xmat['grasp', ['xx', 'zx']]
        // object_angle = np.pi-np.arctan2(grasp_direction[1],
        // grasp_direction[0])
        PhysicsAfterReset();
        int id_site_grasp = mj_name2id(model_, mjOBJ_SITE, "grasp");
        object_x = data_->site_xpos[id_site_grasp * 3 + 0];
        object_z = data_->site_xpos[id_site_grasp * 3 + 2];
        std::array<mjtNum, 2> grasp_direction = {
            data_->site_xmat[id_site_grasp * 9 + 0],
            data_->site_xmat[id_site_grasp * 9 + 6]};
        object_angle =
            M_PI - std::atan2(grasp_direction[1], grasp_direction[0]);
      } else {
        // uniform
        // object_x = uniform(-.5, .5)
        // object_z = uniform(0, .7)
        // object_angle = uniform(0, 2*np.pi)
        // data.qvel[self._object + '_x'] = uniform(-5, 5)
        object_x = dist_uniform_(gen_) * 1 - 0.5;
        object_z = dist_uniform_(gen_) * 0.7;
        object_angle = dist_uniform_(gen_) * 2 * M_PI;
        data_->qvel[GetQposId(model_, object_ + "_x")] =
            dist_uniform_(gen_) * 10 - 5;
      }
      data_->qpos[mj_name2id(model_, mjOBJ_JOINT, object_joints_[0].c_str())] =
          object_x;
      data_->qpos[mj_name2id(model_, mjOBJ_JOINT, object_joints_[1].c_str())] =
          object_z;
      data_->qpos[mj_name2id(model_, mjOBJ_JOINT, object_joints_[2].c_str())] =
          object_angle;
      // Check for collisions.
      PhysicsAfterReset();
      penetrating = data_->ncon > 0;
    }

#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_.get(), data_->qpos, sizeof(mjtNum) * model_->nq);
    std::memcpy(qvel_.get(), data_->qvel, sizeof(mjtNum) * model_->nv);
    std::memcpy(body_pos_.get(), model_->body_pos,
                sizeof(mjtNum) * model_->nbody * 3);
    std::memcpy(body_quat_.get(), model_->body_quat,
                sizeof(mjtNum) * model_->nbody * 4);
#endif
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
    for (std::size_t i = 0; i < kArmJoints.size(); i++) {
      int id = GetQposId(model_, kArmJoints[i]);
      bound[i * 2 + 0] = std::sin(data_->qpos[id]);
      bound[i * 2 + 1] = std::cos(data_->qpos[id]);
    }
    return bound;
  }

  // bug :(
  std::array<mjtNum, 8> JointVelArm() {
    std::array<mjtNum, 8> joint;
    for (std::size_t i = 0; i < kArmJoints.size(); i++) {
      int id = GetQposId(model_, kArmJoints[i]);
      joint[i] = data_->qvel[id];
    }
    return joint;
  }

  std::array<mjtNum, 3> JointVelObj() {
    std::array<mjtNum, 3> joint;
    for (std::size_t i = 0; i < object_joints_.size(); i++) {
      int id = GetQposId(model_, object_joints_[i]);
      joint[i] = data_->qvel[id];
    }
    return joint;
  }

  std::array<mjtNum, 5> Touch() {
    std::array<mjtNum, 5> touch;
    for (std::size_t i = 0; i < kTouchSensors.size(); i++) {
      int id = GetSensorId(model_, kTouchSensors[i]);
      touch[i] = std::log1p(data_->sensordata[id]);
    }
    return touch;
  }

  std::array<mjtNum, 4> Body2dPose(const std::string& body_names) {
    int id = mj_name2id(model_, mjOBJ_XBODY, body_names.c_str());
    // self.named.data.xpos[body_names, ['x', 'z']]
    // self.named.data.xquat[body_names, ['qw', 'qy']]
    return {data_->xpos[id * 3 + 0], data_->xpos[id * 3 + 2],
            data_->xquat[id * 4 + 0], data_->xquat[id * 4 + 2]};
  }

  mjtNum SiteDistance(const std::string& site1, const std::string& site2) {
    int id_site_1 = mj_name2id(model_, mjOBJ_SITE, site1.c_str());
    int id_site_2 = mj_name2id(model_, mjOBJ_SITE, site2.c_str());
    std::array<mjtNum, 3> diff = {data_->site_xpos[id_site_1 * 3 + 0] -
                                      data_->site_xpos[id_site_2 * 3 + 0],
                                  data_->site_xpos[id_site_1 * 3 + 1] -
                                      data_->site_xpos[id_site_2 * 3 + 1],
                                  data_->site_xpos[id_site_1 * 3 + 2] -
                                      data_->site_xpos[id_site_2 * 3 + 2]};
    return std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
  }

  [[nodiscard]] mjtNum IsClose(mjtNum distance) const {
    // return rewards.tolerance(distance, (0, _CLOSE), _CLOSE * 2)
    return RewardTolerance(distance, 0, kClose, kClose * 2);
  }

  mjtNum PegReward() {
    auto grasping = (IsClose(SiteDistance("peg_grasp", "grasp")) +
                     IsClose(SiteDistance("peg_pinch", "pinch"))) /
                    2;
    auto bringing = (IsClose(SiteDistance("peg", "target_peg")) +
                     IsClose(SiteDistance("target_peg_tip", "peg_tip"))) /
                    2;
    return std::max(bringing, grasping / 3);
  }
  mjtNum BallReward() {
    // return self._is_close(physics.site_distance('ball', 'target_ball'))
    return IsClose(SiteDistance("ball", "target_ball"));
  }

  void WriteState() {
    const auto& bounded_joint_pos = BoundedJointPos();
    const auto& joint_vel_arm = JointVelArm();
    const auto& touch = Touch();
    const auto& hand_pos = Body2dPose("hand");
    const auto& object_pos = Body2dPose(object_);
    const auto& joint_vel_obj = JointVelObj();
    const auto& target_pos = Body2dPose(target_);

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
    state["info:qvel"_].Assign(qvel_.get(), model_->nv);
    state["info:body_pos"_].Assign(body_pos_.get(), model_->nbody * 3);
    state["info:body_quat"_].Assign(body_quat_.get(), model_->nbody * 4);
#endif
  }
};

using ManipulatorEnvPool = AsyncEnvPool<ManipulatorEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_MANIPULATOR_H_
