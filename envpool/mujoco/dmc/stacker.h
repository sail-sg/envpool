/*
 * Copyright 2026 Garena Online Private Limited
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
// https://github.com/deepmind/dm_control/blob/1.0.38/dm_control/suite/stacker.py

#ifndef ENVPOOL_MUJOCO_DMC_STACKER_H_
#define ENVPOOL_MUJOCO_DMC_STACKER_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/dmc/mujoco_env.h"
#include "envpool/mujoco/dmc/utils.h"

namespace mujoco_dmc {

int StackerNumBoxes(const std::string& task_name) {
  if (task_name == "stack_2") {
    return 2;
  }
  if (task_name == "stack_4") {
    return 4;
  }
  throw std::runtime_error("Unknown task_name " + task_name +
                           " for dmc stacker.");
}

std::string GetStackerXML(const std::string& base_path,
                          const std::string& task_name) {
  return XMLMakeStacker(GetFileContent(base_path, "stacker.xml"),
                        StackerNumBoxes(task_name));
}

class StackerEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("frame_skip"_.Bind(10), "frame_stack"_.Bind(1),
                    "task_name"_.Bind(std::string("stack_2")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    int n_boxes = StackerNumBoxes(conf["task_name"_]);
    [[maybe_unused]] int qpos_size = 8 + 3 * n_boxes;
    return MakeDict(
        "obs:arm_pos"_.Bind(
            StackSpec(Spec<mjtNum>({8, 2}), conf["frame_stack"_])),
        "obs:arm_vel"_.Bind(StackSpec(Spec<mjtNum>({8}), conf["frame_stack"_])),
        "obs:touch"_.Bind(StackSpec(Spec<mjtNum>({5}), conf["frame_stack"_])),
        "obs:hand_pos"_.Bind(
            StackSpec(Spec<mjtNum>({4}), conf["frame_stack"_])),
        "obs:box_pos"_.Bind(
            StackSpec(Spec<mjtNum>({n_boxes, 4}), conf["frame_stack"_])),
        "obs:box_vel"_.Bind(
            StackSpec(Spec<mjtNum>({3 * n_boxes}), conf["frame_stack"_])),
        "obs:target_pos"_.Bind(
            StackSpec(Spec<mjtNum>({2}), conf["frame_stack"_]))
#ifdef ENVPOOL_TEST
            ,
        "info:qpos0"_.Bind(Spec<mjtNum>({qpos_size})),
        "info:qvel0"_.Bind(Spec<mjtNum>({qpos_size})),
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({qpos_size})),
        "info:target0"_.Bind(Spec<mjtNum>({2}))
#endif
    );  // NOLINT
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 5}, {-1.0, 1.0})));
  }
};

using StackerEnvSpec = EnvSpec<StackerEnvFns>;
using StackerPixelEnvFns = PixelObservationEnvFns<StackerEnvFns>;
using StackerPixelEnvSpec = EnvSpec<StackerPixelEnvFns>;

template <typename EnvSpecT, bool kFromPixels>
class StackerEnvBase : public Env<EnvSpecT>, public MujocoEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;

  static constexpr mjtNum kClose = 0.01;
  const std::array<std::string, 8> kArmJoints = {
      "arm_root", "arm_shoulder", "arm_elbow", "arm_wrist",
      "finger",   "fingertip",    "thumb",     "thumbtip"};

  int n_boxes_;
  std::array<int, 8> id_arm_joints_{};
  std::array<int, 8> id_arm_qpos_{};
  std::array<int, 8> id_arm_qvel_{};
  int id_finger_;
  int id_thumb_;
  int id_target_body_;
  int id_target_geom_;
  int id_target_site_;
  int id_hand_;
  int id_grasp_site_;
  std::vector<int> id_box_xbody_;
  std::vector<int> id_box_site_;
  std::vector<std::array<int, 3>> id_box_qpos_;
  std::vector<std::array<int, 3>> id_box_qvel_;
#ifdef ENVPOOL_TEST
  std::unique_ptr<mjtNum[]> qvel0_;
  std::array<mjtNum, 2> target0_{};
#endif

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  StackerEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        MujocoEnv(spec.config["base_path"_],
                  GetStackerXML(spec.config["base_path"_],
                                spec.config["task_name"_]),
                  spec.config["frame_skip"_], spec.config["max_episode_steps"_],
                  spec.config["frame_stack"_],
                  RenderWidthOrDefault<kFromPixels>(spec.config),
                  RenderHeightOrDefault<kFromPixels>(spec.config),
                  RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        n_boxes_(StackerNumBoxes(spec.config["task_name"_])),
        id_finger_(GetQposId(model_, "finger")),
        id_thumb_(GetQposId(model_, "thumb")),
        id_target_body_(mj_name2id(model_, mjOBJ_BODY, "target")),
        id_target_geom_(mj_name2id(model_, mjOBJ_GEOM, "target")),
        id_target_site_(mj_name2id(model_, mjOBJ_SITE, "target")),
        id_hand_(mj_name2id(model_, mjOBJ_XBODY, "hand")),
        id_grasp_site_(mj_name2id(model_, mjOBJ_SITE, "grasp")) {
#ifdef ENVPOOL_TEST
    qvel0_.reset(new mjtNum[model_->nv]);
#endif
    for (std::size_t i = 0; i < kArmJoints.size(); ++i) {
      id_arm_joints_[i] =
          mj_name2id(model_, mjOBJ_JOINT, kArmJoints[i].c_str());
      id_arm_qpos_[i] = GetQposId(model_, kArmJoints[i]);
      id_arm_qvel_[i] = GetQvelId(model_, kArmJoints[i]);
    }
    for (int box = 0; box < n_boxes_; ++box) {
      std::string name = "box" + std::to_string(box);
      id_box_xbody_.push_back(mj_name2id(model_, mjOBJ_XBODY, name.c_str()));
      id_box_site_.push_back(mj_name2id(model_, mjOBJ_SITE, name.c_str()));
      id_box_qpos_.push_back(
          {GetQposId(model_, name + "_x"), GetQposId(model_, name + "_z"),
           GetQposId(model_, name + "_y")});
      id_box_qvel_.push_back(
          {GetQvelId(model_, name + "_x"), GetQvelId(model_, name + "_y"),
           GetQvelId(model_, name + "_z")});
    }
  }

  void TaskInitializeEpisode() override {
    bool penetrating = true;
    while (penetrating) {
      for (std::size_t i = 0; i < kArmJoints.size(); ++i) {
        int id_joint = id_arm_joints_[i];
        bool is_limited = model_->jnt_limited[id_joint] == 1;
        mjtNum lower =
            is_limited ? model_->jnt_range[id_joint * 2 + 0] : -M_PI;
        mjtNum upper =
            is_limited ? model_->jnt_range[id_joint * 2 + 1] : M_PI;
        data_->qpos[id_arm_qpos_[i]] = RandUniform(lower, upper)(gen_);
      }
      data_->qpos[id_finger_] = data_->qpos[id_thumb_];

      int target_height = 2 * RandInt(0, n_boxes_ - 1)(gen_) + 1;
      mjtNum box_size = model_->geom_size[id_target_geom_ * 3 + 0];
      model_->body_pos[id_target_body_ * 3 + 2] = box_size * target_height;
      model_->body_pos[id_target_body_ * 3 + 0] =
          RandUniform(-0.37, 0.37)(gen_);

      for (int box = 0; box < n_boxes_; ++box) {
        data_->qpos[id_box_qpos_[box][0]] = RandUniform(0.1, 0.3)(gen_);
        data_->qpos[id_box_qpos_[box][1]] = RandUniform(0.0, 0.7)(gen_);
        data_->qpos[id_box_qpos_[box][2]] = RandUniform(0.0, 2 * M_PI)(gen_);
      }

      PhysicsAfterReset();
      penetrating = data_->ncon > 0;
    }
#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_.get(), data_->qpos, sizeof(mjtNum) * model_->nq);
    std::memcpy(qvel0_.get(), data_->qvel, sizeof(mjtNum) * model_->nv);
    target0_ = {model_->body_pos[id_target_body_ * 3 + 0],
                model_->body_pos[id_target_body_ * 3 + 2]};
#endif
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    ControlReset();
    WriteState(true);
  }

  void Step(const Action& action) override {
    auto* act = static_cast<mjtNum*>(action["action"_].Data());
    ControlStep(act);
    WriteState(false);
  }

  float TaskGetReward() override {
    mjtNum box_size = model_->geom_size[id_target_geom_ * 3 + 0];
    mjtNum min_box_to_target_distance =
        std::numeric_limits<mjtNum>::infinity();
    for (int box = 0; box < n_boxes_; ++box) {
      min_box_to_target_distance = std::min(
          min_box_to_target_distance,
          SiteDistance(id_box_site_[box], id_target_site_));
    }
    mjtNum box_is_close =
        RewardTolerance(min_box_to_target_distance, 0.0, 0.0, 2 * box_size);
    mjtNum hand_to_target_distance =
        SiteDistance(id_grasp_site_, id_target_site_);
    mjtNum hand_is_far =
        RewardTolerance(hand_to_target_distance, 0.1,
                        std::numeric_limits<double>::infinity(), kClose);
    return static_cast<float>(box_is_close * hand_is_far);
  }

  bool TaskShouldTerminateEpisode() override { return false; }

 private:
  mjtNum SiteDistance(int site1, int site2) {
    mjtNum dx = data_->site_xpos[site2 * 3 + 0] -
                data_->site_xpos[site1 * 3 + 0];
    mjtNum dy = data_->site_xpos[site2 * 3 + 1] -
                data_->site_xpos[site1 * 3 + 1];
    mjtNum dz = data_->site_xpos[site2 * 3 + 2] -
                data_->site_xpos[site1 * 3 + 2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  std::array<mjtNum, 16> ArmPos() {
    std::array<mjtNum, 16> result;
    for (std::size_t i = 0; i < kArmJoints.size(); ++i) {
      mjtNum pos = data_->qpos[id_arm_qpos_[i]];
      result[2 * i] = std::sin(pos);
      result[2 * i + 1] = std::cos(pos);
    }
    return result;
  }

  std::array<mjtNum, 8> ArmVel() {
    std::array<mjtNum, 8> result;
    for (std::size_t i = 0; i < kArmJoints.size(); ++i) {
      result[i] = data_->qvel[id_arm_qvel_[i]];
    }
    return result;
  }

  std::array<mjtNum, 5> Touch() {
    return {std::log1p(data_->sensordata[0]),
            std::log1p(data_->sensordata[1]),
            std::log1p(data_->sensordata[2]),
            std::log1p(data_->sensordata[3]),
            std::log1p(data_->sensordata[4])};
  }

  std::array<mjtNum, 4> Body2dPose(int xbody_id) {
    return {data_->xpos[xbody_id * 3 + 0], data_->xpos[xbody_id * 3 + 2],
            data_->xquat[xbody_id * 4 + 0], data_->xquat[xbody_id * 4 + 2]};
  }

  std::vector<mjtNum> BoxPos() {
    std::vector<mjtNum> result(n_boxes_ * 4);
    for (int box = 0; box < n_boxes_; ++box) {
      const auto& pose = Body2dPose(id_box_xbody_[box]);
      for (int i = 0; i < 4; ++i) {
        result[box * 4 + i] = pose[i];
      }
    }
    return result;
  }

  std::vector<mjtNum> BoxVel() {
    std::vector<mjtNum> result(n_boxes_ * 3);
    for (int box = 0; box < n_boxes_; ++box) {
      for (int i = 0; i < 3; ++i) {
        result[box * 3 + i] = data_->qvel[id_box_qvel_[box][i]];
      }
    }
    return result;
  }

  std::array<mjtNum, 2> TargetPos() {
    return {data_->xpos[id_target_body_ * 3 + 0],
            data_->xpos[id_target_body_ * 3 + 2]};
  }

  void WriteState(bool reset) {
    auto state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      const auto& arm_pos = ArmPos();
      const auto& arm_vel = ArmVel();
      const auto& touch = Touch();
      const auto& hand_pos = Body2dPose(id_hand_);
      const auto& box_pos = BoxPos();
      const auto& box_vel = BoxVel();
      const auto& target_pos = TargetPos();

      auto obs_arm_pos = state["obs:arm_pos"_];
      AssignObservation("obs:arm_pos", &obs_arm_pos, arm_pos.data(),
                        arm_pos.size(), reset);
      auto obs_arm_vel = state["obs:arm_vel"_];
      AssignObservation("obs:arm_vel", &obs_arm_vel, arm_vel.data(),
                        arm_vel.size(), reset);
      auto obs_touch = state["obs:touch"_];
      AssignObservation("obs:touch", &obs_touch, touch.data(), touch.size(),
                        reset);
      auto obs_hand_pos = state["obs:hand_pos"_];
      AssignObservation("obs:hand_pos", &obs_hand_pos, hand_pos.data(),
                        hand_pos.size(), reset);
      auto obs_box_pos = state["obs:box_pos"_];
      AssignObservation("obs:box_pos", &obs_box_pos, box_pos.data(),
                        box_pos.size(), reset);
      auto obs_box_vel = state["obs:box_vel"_];
      AssignObservation("obs:box_vel", &obs_box_vel, box_vel.data(),
                        box_vel.size(), reset);
      auto obs_target_pos = state["obs:target_pos"_];
      AssignObservation("obs:target_pos", &obs_target_pos, target_pos.data(),
                        target_pos.size(), reset);
    }
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
    state["info:qvel0"_].Assign(qvel0_.get(), model_->nv);
    state["info:qacc_warmstart0"_].Assign(data_->qacc_warmstart, model_->nv);
    state["info:target0"_].Assign(target0_.data(), target0_.size());
#endif
  }
};

using StackerEnv = StackerEnvBase<StackerEnvSpec, false>;
using StackerPixelEnv = StackerEnvBase<StackerPixelEnvSpec, true>;
using StackerEnvPool = AsyncEnvPool<StackerEnv>;
using StackerPixelEnvPool = AsyncEnvPool<StackerPixelEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_STACKER_H_
