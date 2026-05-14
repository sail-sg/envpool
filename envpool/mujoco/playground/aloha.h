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

#ifndef ENVPOOL_MUJOCO_PLAYGROUND_ALOHA_H_
#define ENVPOOL_MUJOCO_PLAYGROUND_ALOHA_H_

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

namespace mujoco_playground {

using envpool::mujoco::PixelObservationEnvFns;
using envpool::mujoco::RenderCameraIdOrDefault;
using envpool::mujoco::RenderHeightOrDefault;
using envpool::mujoco::RenderWidthOrDefault;
using envpool::mujoco::StackSpec;

constexpr int kAlohaActionDim = 14;
constexpr int kAlohaHandoverStateDim = 83;
constexpr int kAlohaPegStateDim = 82;
constexpr int kAlohaMaxStateDim = 83;
constexpr int kAlohaArmJoints = 12;
constexpr int kAlohaFingerJoints = 4;
constexpr int kAlohaFingerSensors = 8;

class PlaygroundAlohaEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "frame_stack"_.Bind(1), "task_name"_.Bind(std::string("AlohaHandOver")),
        "ctrl_dt"_.Bind(0.02), "sim_dt"_.Bind(0.005), "action_repeat"_.Bind(1),
        "action_scale"_.Bind(0.015), "gripper_box_scale"_.Bind(1.0),
        "box_handover_scale"_.Bind(4.0), "handover_target_scale"_.Bind(8.0),
        "no_table_collision_scale"_.Bind(0.3), "left_reward_scale"_.Bind(1.0),
        "right_reward_scale"_.Bind(1.0), "left_target_qpos_scale"_.Bind(0.3),
        "right_target_qpos_scale"_.Bind(0.3), "socket_z_up_scale"_.Bind(0.5),
        "peg_z_up_scale"_.Bind(0.5), "socket_entrance_reward_scale"_.Bind(4.0),
        "peg_end2_reward_scale"_.Bind(4.0),
        "peg_insertion_reward_scale"_.Bind(8.0));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    const std::string task_name = conf["task_name"_];
    const int state_dim = task_name == "AlohaSinglePegInsertion"
                              ? kAlohaPegStateDim
                              : kAlohaHandoverStateDim;
    return MakeDict(
        "obs:state"_.Bind(StackSpec(Spec<mjtNum>({state_dim}, {-inf, inf}),
                                    conf["frame_stack"_])),
        "info:terminated"_.Bind(Spec<bool>({-1})),
        "info:out_of_bounds"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_gripper_box"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_box_handover"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_handover_target"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_no_table_collision"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_left_reward"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_right_reward"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_left_target_qpos"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_right_target_qpos"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_socket_z_up"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_peg_z_up"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_socket_entrance_reward"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_peg_end2_reward"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_peg_insertion_reward"_.Bind(Spec<mjtNum>({-1})),
        "info:peg_end2_dist_to_line"_.Bind(Spec<mjtNum>({-1}))
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
        "info:site_xpos"_.Bind(Spec<mjtNum>({512})),
        "info:site_xmat"_.Bind(Spec<mjtNum>({2048})),
        "info:sensordata"_.Bind(Spec<mjtNum>({256})),
        "info:target_pos"_.Bind(Spec<mjtNum>({3})),
        "info:prev_potential"_.Bind(Spec<mjtNum>({-1})),
        "info:episode_picked"_.Bind(Spec<mjtNum>({-1})),
        "info:steps"_.Bind(Spec<mjtNum>({-1}))
#endif
    );  // NOLINT
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<mjtNum>({-1, kAlohaActionDim}, {-1.0, 1.0})));
  }
};

using PlaygroundAlohaEnvSpec = EnvSpec<PlaygroundAlohaEnvFns>;
using PlaygroundAlohaPixelEnvFns =
    PixelObservationEnvFns<PlaygroundAlohaEnvFns>;
using PlaygroundAlohaPixelEnvSpec = EnvSpec<PlaygroundAlohaPixelEnvFns>;

template <typename EnvSpecT, bool kFromPixels>
class PlaygroundAlohaEnvBase : public Env<EnvSpecT>,
                               public PlaygroundMujocoEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  int n_substeps_{4};
  int action_repeat_{1};
  bool is_peg_{false};
  int state_dim_{kAlohaHandoverStateDim};
  std::vector<mjtNum> init_qpos_;
  std::vector<mjtNum> init_ctrl_;
  std::array<mjtNum, kAlohaActionDim> lowers_{};
  std::array<mjtNum, kAlohaActionDim> uppers_{};
  std::array<int, kAlohaArmJoints> arm_qposadr_{};
  std::array<int, kAlohaFingerJoints> finger_qposadr_{};
  std::array<int, kAlohaFingerSensors> table_finger_sensor_adrs_{};
  std::array<mjtNum, 3> target_pos_{};
  std::array<mjtNum, kAlohaMaxStateDim> obs_{};
  int left_gripper_site_{-1};
  int right_gripper_site_{-1};
  int table_geom_{-1};
  int mocap_target_{-1};
  int box_body_{-1};
  int box_top_site_{-1};
  int box_bottom_site_{-1};
  int box_qposadr_{-1};
  int box_geom_{-1};
  int socket_entrance_site_{-1};
  int socket_rear_site_{-1};
  int peg_end2_site_{-1};
  int socket_body_{-1};
  int peg_body_{-1};
  int socket_qposadr_{-1};
  int peg_qposadr_{-1};
  mjtNum prev_potential_{0.0};
  bool episode_picked_{false};
  int task_steps_{0};
  mjtNum out_of_bounds_{0.0};
  mjtNum peg_end2_dist_to_line_{0.0};
  mjtNum reward_gripper_box_{0.0};
  mjtNum reward_box_handover_{0.0};
  mjtNum reward_handover_target_{0.0};
  mjtNum reward_no_table_collision_{0.0};
  mjtNum reward_left_reward_{0.0};
  mjtNum reward_right_reward_{0.0};
  mjtNum reward_left_target_qpos_{0.0};
  mjtNum reward_right_target_qpos_{0.0};
  mjtNum reward_socket_z_up_{0.0};
  mjtNum reward_peg_z_up_{0.0};
  mjtNum reward_socket_entrance_reward_{0.0};
  mjtNum reward_peg_end2_reward_{0.0};
  mjtNum reward_peg_insertion_reward_{0.0};
  std::uniform_real_distribution<mjtNum> unit_uniform_{0.0, 1.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  PlaygroundAlohaEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        PlaygroundMujocoEnv(
            AlohaXmlPath(spec.config["base_path"_], spec.config["task_name"_]),
            spec.config["max_episode_steps"_], spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config),
            envpool::mujoco::CameraPolicy::kDmControl) {
    const std::string task_name = spec.config["task_name"_];
    is_peg_ = task_name == "AlohaSinglePegInsertion";
    if (!is_peg_ && task_name != "AlohaHandOver") {
      throw std::runtime_error("Unsupported Aloha task_name " + task_name);
    }
    state_dim_ = is_peg_ ? kAlohaPegStateDim : kAlohaHandoverStateDim;
    if (model_->nu != kAlohaActionDim) {
      throw std::runtime_error("Unexpected Aloha action dimension.");
    }
    n_substeps_ = static_cast<int>(
        std::llround(spec.config["ctrl_dt"_] / spec.config["sim_dt"_]));
    action_repeat_ = static_cast<int>(spec.config["action_repeat"_]);
    model_->opt.timestep = spec.config["sim_dt"_];
    InitModelIds();
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    has_cached_render_ = false;
    done_ = false;
    terminated_ = false;
    elapsed_step_ = 0;
    prev_potential_ = 0.0;
    episode_picked_ = false;
    task_steps_ = 0;
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

    if (is_peg_) {
      ResetPeg();
    } else {
      ResetHandover();
    }
    mj_forward(model_, data_);
    UpdateObs();
    WriteState(0.0, true);
  }

  void Step(const Action& action) override {
    has_cached_render_ = false;
    const auto* act = static_cast<const mjtNum*>(action["action"_].Data());
    for (int i = 0; i < kAlohaActionDim; ++i) {
      data_->ctrl[i] =
          std::clamp(data_->ctrl[i] + act[i] * spec_.config["action_scale"_],
                     lowers_[i], uppers_[i]);
    }
    for (int i = 0; i < n_substeps_; ++i) {
      mj_step(model_, data_);
    }

    const mjtNum reward =
        is_peg_ ? ComputePegReward() : ComputeHandoverReward();
    UpdateObs();
    terminated_ = out_of_bounds_ > 0.0 || HasNaN();
    ++elapsed_step_;
    done_ = terminated_ || elapsed_step_ >= max_episode_steps_;
    WriteState(static_cast<float>(reward), false);
  }

 private:
  static std::string AlohaXmlPath(const std::string& base_path,
                                  const std::string& task_name) {
    const char* xml_name = task_name == "AlohaSinglePegInsertion"
                               ? "mjx_single_peg_insertion.xml"
                               : "mjx_hand_over.xml";
    return base_path +
           "/mujoco/playground/assets/mujoco_playground/_src/"
           "manipulation/aloha/xmls/" +
           xml_name;
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

  mjtNum Uniform(mjtNum low, mjtNum high) {
    return low + (high - low) * unit_uniform_(gen_);
  }

  void InitModelIds() {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const char* arm_joints[kAlohaArmJoints] = {
        "left/waist",         "left/shoulder",     "left/elbow",
        "left/forearm_roll",  "left/wrist_angle",  "left/wrist_rotate",
        "right/waist",        "right/shoulder",    "right/elbow",
        "right/forearm_roll", "right/wrist_angle", "right/wrist_rotate"};
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const char* finger_joints[kAlohaFingerJoints] = {
        "left/left_finger", "left/right_finger", "right/left_finger",
        "right/right_finger"};
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const char* finger_geoms[kAlohaFingerSensors] = {
        "left/left_finger_top",   "left/left_finger_bottom",
        "left/right_finger_top",  "left/right_finger_bottom",
        "right/left_finger_top",  "right/left_finger_bottom",
        "right/right_finger_top", "right/right_finger_bottom"};
    for (int i = 0; i < kAlohaArmJoints; ++i) {
      const int joint_id = RequireId(mjOBJ_JOINT, arm_joints[i]);
      arm_qposadr_[i] = model_->jnt_qposadr[joint_id];
    }
    for (int i = 0; i < kAlohaFingerJoints; ++i) {
      const int joint_id = RequireId(mjOBJ_JOINT, finger_joints[i]);
      finger_qposadr_[i] = model_->jnt_qposadr[joint_id];
    }
    for (int i = 0; i < kAlohaFingerSensors; ++i) {
      table_finger_sensor_adrs_[i] =
          SensorAdr(std::string("table_") + finger_geoms[i] + "_found");
    }
    left_gripper_site_ = RequireId(mjOBJ_SITE, "left/gripper");
    right_gripper_site_ = RequireId(mjOBJ_SITE, "right/gripper");
    table_geom_ = RequireId(mjOBJ_GEOM, "table");
    const int key_id = RequireId(mjOBJ_KEY, "home");
    init_qpos_.assign(model_->key_qpos + key_id * model_->nq,
                      model_->key_qpos + (key_id + 1) * model_->nq);
    init_ctrl_.assign(model_->key_ctrl + key_id * model_->nu,
                      model_->key_ctrl + (key_id + 1) * model_->nu);
    for (int i = 0; i < kAlohaActionDim; ++i) {
      lowers_[i] = model_->actuator_ctrlrange[2 * i];
      uppers_[i] = model_->actuator_ctrlrange[2 * i + 1];
    }

    if (is_peg_) {
      socket_entrance_site_ = RequireId(mjOBJ_SITE, "socket_entrance");
      socket_rear_site_ = RequireId(mjOBJ_SITE, "socket_rear");
      peg_end2_site_ = RequireId(mjOBJ_SITE, "peg_end2");
      socket_body_ = RequireId(mjOBJ_BODY, "socket");
      peg_body_ = RequireId(mjOBJ_BODY, "peg");
      socket_qposadr_ = model_->jnt_qposadr[model_->body_jntadr[socket_body_]];
      peg_qposadr_ = model_->jnt_qposadr[model_->body_jntadr[peg_body_]];
    } else {
      lowers_[6] = 0.01;
      const int mocap_body = RequireId(mjOBJ_BODY, "mocap_target");
      mocap_target_ = model_->body_mocapid[mocap_body];
      box_body_ = RequireId(mjOBJ_BODY, "box");
      box_top_site_ = RequireId(mjOBJ_SITE, "box_top");
      box_bottom_site_ = RequireId(mjOBJ_SITE, "box_bottom");
      box_qposadr_ = model_->jnt_qposadr[model_->body_jntadr[box_body_]];
      box_geom_ = RequireId(mjOBJ_GEOM, "box");
    }
  }

  void ResetHandover() {
    data_->qpos[box_qposadr_ + 0] += Uniform(-0.05, 0.05);
    data_->qpos[box_qposadr_ + 1] += Uniform(-0.1, 0.1);
    target_pos_ = {0.20 + Uniform(-0.15, 0.15), Uniform(-0.15, 0.15),
                   0.25 + Uniform(-0.15, 0.15)};
    target_pos_[0] = std::max<mjtNum>(target_pos_[0], 0.15);
    data_->mocap_pos[3 * mocap_target_ + 0] = target_pos_[0];
    data_->mocap_pos[3 * mocap_target_ + 1] = target_pos_[1];
    data_->mocap_pos[3 * mocap_target_ + 2] = target_pos_[2];
  }

  void ResetPeg() {
    for (int i = 0; i < 2; ++i) {
      data_->qpos[peg_qposadr_ + i] += Uniform(-0.1, 0.1);
      data_->qpos[socket_qposadr_ + i] += Uniform(-0.1, 0.1);
    }
    target_pos_ = {0.0, 0.0, 0.0};
  }

  void ResetRewards() {
    out_of_bounds_ = 0.0;
    peg_end2_dist_to_line_ = 0.0;
    reward_gripper_box_ = 0.0;
    reward_box_handover_ = 0.0;
    reward_handover_target_ = 0.0;
    reward_no_table_collision_ = 0.0;
    reward_left_reward_ = 0.0;
    reward_right_reward_ = 0.0;
    reward_left_target_qpos_ = 0.0;
    reward_right_target_qpos_ = 0.0;
    reward_socket_z_up_ = 0.0;
    reward_peg_z_up_ = 0.0;
    reward_socket_entrance_reward_ = 0.0;
    reward_peg_end2_reward_ = 0.0;
    reward_peg_insertion_reward_ = 0.0;
  }

  static mjtNum Norm3(const mjtNum* values) {
    return std::sqrt(values[0] * values[0] + values[1] * values[1] +
                     values[2] * values[2]);
  }

  static mjtNum Distance3(const mjtNum* lhs, const mjtNum* rhs) {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum delta[3] = {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
    return Norm3(delta);
  }

  static mjtNum Dot3(const mjtNum* lhs, const mjtNum* rhs) {
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
  }

  static mjtNum LinearTolerance(mjtNum x, mjtNum lower, mjtNum upper,
                                mjtNum margin) {
    if (lower <= x && x <= upper) {
      return 1.0;
    }
    const mjtNum d = (x < lower ? lower - x : x - upper) / margin;
    const mjtNum scaled = d * 0.9;
    return std::abs(scaled) < 1.0 ? 1.0 - scaled : 0.0;
  }

  static mjtNum GaussianTolerance(mjtNum x, mjtNum lower, mjtNum upper,
                                  mjtNum margin) {
    if (lower <= x && x <= upper) {
      return 1.0;
    }
    const mjtNum d = (x < lower ? lower - x : x - upper) / margin;
    const mjtNum scale = std::sqrt(-2.0 * std::log(0.1));
    return std::exp(-0.5 * (d * scale) * (d * scale));
  }

  mjtNum HandTableCollision() const {
    for (int adr : table_finger_sensor_adrs_) {
      if (data_->sensordata[adr] > 0.0) {
        return 1.0;
      }
    }
    return 0.0;
  }

  static mjtNum LogisticBarrier(mjtNum x, mjtNum x0 = 0.0, mjtNum k = 100.0,
                                mjtNum direction = 1.0) {
    return 1.0 / (1.0 + std::exp(-k * direction * (x - x0)));
  }

  mjtNum ComputeHandoverReward() {
    const bool newly_reset = task_steps_ == 0;
    if (newly_reset) {
      episode_picked_ = false;
      prev_potential_ = 0.0;
    }

    const mjtNum* box_top = data_->site_xpos + 3 * box_top_site_;
    const mjtNum* box_bottom = data_->site_xpos + 3 * box_bottom_site_;
    const mjtNum* box = data_->xpos + 3 * box_body_;
    const mjtNum* left_gripper = data_->site_xpos + 3 * left_gripper_site_;
    const mjtNum* right_gripper = data_->site_xpos + 3 * right_gripper_site_;
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const mjtNum handover_pos[3] = {0.0, 0.0, 0.24};

    const mjtNum pre = box[0] < -0.1 ? 1.0 : 0.0;
    const mjtNum past = box[0] >= 0.0 ? 1.0 : 0.0;
    const mjtNum btwn = (1.0 - pre) * (1.0 - past);
    const mjtNum r_lg =
        std::exp(-10.0 * Distance3(box_top, left_gripper)) * (pre + btwn);
    const mjtNum r_rg =
        std::exp(-10.0 * Distance3(box_bottom, right_gripper)) * (btwn + past);
    const mjtNum r_rg_bias =
        std::exp(-10.0 * Distance3(box_bottom, right_gripper)) * past;
    reward_gripper_box_ =
        (r_lg + r_rg + r_rg_bias) * spec_.config["gripper_box_scale"_];

    const mjtNum box_handover = std::exp(-10.0 * Distance3(box, handover_pos));
    const mjtNum hand_handover =
        std::exp(-10.0 * Distance3(left_gripper, handover_pos)) * past;
    reward_box_handover_ = std::max(box_handover, hand_handover) *
                           spec_.config["box_handover_scale"_];

    const mjtNum box_target =
        std::exp(-10.0 * Distance3(target_pos_.data(), box)) *
        (r_rg + r_rg_bias) * LogisticBarrier(left_gripper[0], 0.0, 100.0, -1.0);
    reward_handover_target_ =
        box_target * spec_.config["handover_target_scale"_];
    reward_no_table_collision_ = (1.0 - HandTableCollision()) *
                                 spec_.config["no_table_collision_scale"_];

    const mjtNum scale_sum = spec_.config["gripper_box_scale"_] +
                             spec_.config["box_handover_scale"_] +
                             spec_.config["handover_target_scale"_] +
                             spec_.config["no_table_collision_scale"_];
    const mjtNum potential =
        (reward_gripper_box_ + reward_box_handover_ + reward_handover_target_ +
         reward_no_table_collision_) /
        scale_sum;
    mjtNum reward = std::max<mjtNum>(potential - prev_potential_, 0.0);
    const mjtNum condition =
        LogisticBarrier(left_gripper[0], 0.0, 100.0, -1.0) *
        LogisticBarrier(box[0], 0.10, 100.0, 1.0);
    reward += 0.02 * potential * condition;
    prev_potential_ = std::max(prev_potential_, potential);
    if (newly_reset) {
      reward = 0.0;
    }

    const bool picked = box[2] > 0.15;
    episode_picked_ = episode_picked_ || picked;
    const bool dropped = box[2] < 0.05 && episode_picked_;
    if (dropped) {
      reward -= 0.1;
    }
    out_of_bounds_ = (std::abs(box[0]) > 1.0 || std::abs(box[1]) > 1.0 ||
                      std::abs(box[2]) > 1.0 || box[2] < 0.0)
                         ? 1.0
                         : 0.0;
    task_steps_ += action_repeat_;
    if (out_of_bounds_ > 0.0 || dropped || task_steps_ >= max_episode_steps_) {
      task_steps_ = 0;
    }
    terminated_ = out_of_bounds_ > 0.0 || dropped || HasNaN();
    return reward;
  }

  mjtNum ComputePegReward() {
    const mjtNum* socket_entrance_pos =
        data_->site_xpos + 3 * socket_entrance_site_;
    const mjtNum* socket_rear_pos = data_->site_xpos + 3 * socket_rear_site_;
    const mjtNum* peg_end2_pos = data_->site_xpos + 3 * peg_end2_site_;
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum socket_ab[3] = {
        socket_entrance_pos[0] - socket_rear_pos[0],
        socket_entrance_pos[1] - socket_rear_pos[1],
        socket_entrance_pos[2] - socket_rear_pos[2],
    };
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum peg_rear[3] = {peg_end2_pos[0] - socket_rear_pos[0],
                          peg_end2_pos[1] - socket_rear_pos[1],
                          peg_end2_pos[2] - socket_rear_pos[2]};
    const mjtNum socket_t =
        Dot3(peg_rear, socket_ab) / (Dot3(socket_ab, socket_ab) + 1e-6);
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum nearest_pt[3] = {socket_rear_pos[0] + socket_t * socket_ab[0],
                            socket_rear_pos[1] + socket_t * socket_ab[1],
                            socket_rear_pos[2] + socket_t * socket_ab[2]};
    peg_end2_dist_to_line_ = Distance3(peg_end2_pos, nearest_pt);

    const mjtNum* socket_pos = data_->xpos + 3 * socket_body_;
    const mjtNum* peg_pos = data_->xpos + 3 * peg_body_;
    out_of_bounds_ =
        (std::abs(socket_pos[0]) > 1.0 || std::abs(socket_pos[1]) > 1.0 ||
         std::abs(socket_pos[2]) > 1.0 || std::abs(peg_pos[0]) > 1.0 ||
         std::abs(peg_pos[1]) > 1.0 || std::abs(peg_pos[2]) > 1.0)
            ? 1.0
            : 0.0;

    const mjtNum* left_gripper = data_->site_xpos + 3 * left_gripper_site_;
    const mjtNum* right_gripper = data_->site_xpos + 3 * right_gripper_site_;
    const mjtNum left_reward =
        LinearTolerance(Distance3(socket_pos, left_gripper), 0.0, 0.001, 0.3);
    const mjtNum right_reward =
        LinearTolerance(Distance3(peg_pos, right_gripper), 0.0, 0.001, 0.3);

    mjtNum left_pose_delta = 0.0;
    mjtNum right_pose_delta = 0.0;
    for (int i = 0; i < 6; ++i) {
      const mjtNum d =
          data_->qpos[arm_qposadr_[i]] - init_qpos_[arm_qposadr_[i]];
      left_pose_delta += d * d;
    }
    for (int i = 6; i < kAlohaArmJoints; ++i) {
      const mjtNum d =
          data_->qpos[arm_qposadr_[i]] - init_qpos_[arm_qposadr_[i]];
      right_pose_delta += d * d;
    }
    const mjtNum left_pose =
        GaussianTolerance(std::sqrt(left_pose_delta), 0.0, 0.01, 2.0);
    const mjtNum right_pose =
        GaussianTolerance(std::sqrt(right_pose_delta), 0.0, 0.01, 2.0);

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const mjtNum socket_goal[3] = {-0.05, 0.0, 0.15};
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const mjtNum peg_goal[3] = {0.05, 0.0, 0.15};
    const mjtNum socket_lift =
        LinearTolerance(Distance3(socket_goal, socket_pos), 0.0, 0.01, 0.15);
    const mjtNum peg_lift =
        LinearTolerance(Distance3(peg_goal, peg_pos), 0.0, 0.01, 0.15);

    const mjtNum table_collision = HandTableCollision();
    const mjtNum* socket_xmat = data_->xmat + 9 * socket_body_;
    const mjtNum* peg_xmat = data_->xmat + 9 * peg_body_;
    const mjtNum socket_orientation =
        LinearTolerance(socket_xmat[8], 0.99, 1.0, 0.03);
    const mjtNum peg_orientation =
        LinearTolerance(peg_xmat[8], 0.99, 1.0, 0.03);
    const mjtNum peg_insertion_dist = Distance3(peg_end2_pos, socket_rear_pos);
    const mjtNum peg_insertion =
        LinearTolerance(peg_insertion_dist, 0.0, 0.001, 0.1) *
        (peg_end2_dist_to_line_ < 0.005 ? 1.0 : 0.0);

    reward_left_reward_ = left_reward * spec_.config["left_reward_scale"_];
    reward_right_reward_ = right_reward * spec_.config["right_reward_scale"_];
    reward_left_target_qpos_ = left_pose * left_reward * right_reward *
                               spec_.config["left_target_qpos_scale"_];
    reward_right_target_qpos_ = right_pose * left_reward * right_reward *
                                spec_.config["right_target_qpos_scale"_];
    reward_no_table_collision_ =
        (1.0 - table_collision) * spec_.config["no_table_collision_scale"_];
    reward_socket_entrance_reward_ =
        socket_lift * spec_.config["socket_entrance_reward_scale"_];
    reward_peg_end2_reward_ = peg_lift * spec_.config["peg_end2_reward_scale"_];
    reward_socket_z_up_ =
        socket_orientation * socket_lift * spec_.config["socket_z_up_scale"_];
    reward_peg_z_up_ =
        peg_orientation * peg_lift * spec_.config["peg_z_up_scale"_];
    reward_peg_insertion_reward_ =
        peg_insertion * spec_.config["peg_insertion_reward_scale"_];
    const mjtNum scale_sum = spec_.config["left_reward_scale"_] +
                             spec_.config["right_reward_scale"_] +
                             spec_.config["left_target_qpos_scale"_] +
                             spec_.config["right_target_qpos_scale"_] +
                             spec_.config["no_table_collision_scale"_] +
                             spec_.config["socket_z_up_scale"_] +
                             spec_.config["peg_z_up_scale"_] +
                             spec_.config["socket_entrance_reward_scale"_] +
                             spec_.config["peg_end2_reward_scale"_] +
                             spec_.config["peg_insertion_reward_scale"_];
    terminated_ = out_of_bounds_ > 0.0 || HasNaN();
    return (reward_left_reward_ + reward_right_reward_ +
            reward_left_target_qpos_ + reward_right_target_qpos_ +
            reward_no_table_collision_ + reward_socket_z_up_ +
            reward_peg_z_up_ + reward_socket_entrance_reward_ +
            reward_peg_end2_reward_ + reward_peg_insertion_reward_) /
           scale_sum;
  }

  bool HasNaN() const {
    return std::any_of(data_->qpos, data_->qpos + model_->nq,
                       [](mjtNum q) { return std::isnan(q); }) ||
           std::any_of(data_->qvel, data_->qvel + model_->nv,
                       [](mjtNum qvel) { return std::isnan(qvel); });
  }

  void UpdateHandoverObs() {
    int index = 0;
    std::copy(data_->qpos, data_->qpos + model_->nq, obs_.begin() + index);
    index += model_->nq;
    std::copy(data_->qvel, data_->qvel + model_->nv, obs_.begin() + index);
    index += model_->nv;
    for (int adr : finger_qposadr_) {
      obs_[index++] = data_->qpos[adr] - model_->geom_size[3 * box_geom_ + 1];
    }
    const mjtNum* box_top = data_->site_xpos + 3 * box_top_site_;
    std::copy(box_top, box_top + 3, obs_.begin() + index);
    index += 3;
    const mjtNum* box_bottom = data_->site_xpos + 3 * box_bottom_site_;
    std::copy(box_bottom, box_bottom + 3, obs_.begin() + index);
    index += 3;
    const mjtNum* left_gripper = data_->site_xpos + 3 * left_gripper_site_;
    std::copy(left_gripper, left_gripper + 3, obs_.begin() + index);
    index += 3;
    const mjtNum* left_mat = data_->site_xmat + 9 * left_gripper_site_;
    std::copy(left_mat + 3, left_mat + 9, obs_.begin() + index);
    index += 6;
    const mjtNum* right_gripper = data_->site_xpos + 3 * right_gripper_site_;
    std::copy(right_gripper, right_gripper + 3, obs_.begin() + index);
    index += 3;
    const mjtNum* right_mat = data_->site_xmat + 9 * right_gripper_site_;
    std::copy(right_mat + 3, right_mat + 9, obs_.begin() + index);
    index += 6;
    const mjtNum* box_mat = data_->xmat + 9 * box_body_;
    std::copy(box_mat + 3, box_mat + 9, obs_.begin() + index);
    index += 6;
    const mjtNum* box = data_->xpos + 3 * box_body_;
    for (int i = 0; i < 3; ++i) {
      obs_[index++] = box[i] - target_pos_[i];
    }
    const float step_fraction = static_cast<float>(task_steps_) /
                                static_cast<float>(max_episode_steps_);
    obs_[index++] = static_cast<mjtNum>(step_fraction);
  }

  void UpdatePegObs() {
    int index = 0;
    std::copy(data_->qpos, data_->qpos + model_->nq, obs_.begin() + index);
    index += model_->nq;
    std::copy(data_->qvel, data_->qvel + model_->nv, obs_.begin() + index);
    index += model_->nv;
    const mjtNum* left_gripper = data_->site_xpos + 3 * left_gripper_site_;
    std::copy(left_gripper, left_gripper + 3, obs_.begin() + index);
    index += 3;
    const mjtNum* socket_pos = data_->xpos + 3 * socket_body_;
    std::copy(socket_pos, socket_pos + 3, obs_.begin() + index);
    index += 3;
    const mjtNum* right_gripper = data_->site_xpos + 3 * right_gripper_site_;
    std::copy(right_gripper, right_gripper + 3, obs_.begin() + index);
    index += 3;
    const mjtNum* peg_pos = data_->xpos + 3 * peg_body_;
    std::copy(peg_pos, peg_pos + 3, obs_.begin() + index);
    index += 3;
    const mjtNum* socket_entrance =
        data_->site_xpos + 3 * socket_entrance_site_;
    std::copy(socket_entrance, socket_entrance + 3, obs_.begin() + index);
    index += 3;
    const mjtNum* peg_end2 = data_->site_xpos + 3 * peg_end2_site_;
    std::copy(peg_end2, peg_end2 + 3, obs_.begin() + index);
    index += 3;
    const mjtNum* socket_mat = data_->xmat + 9 * socket_body_;
    std::copy(socket_mat + 6, socket_mat + 9, obs_.begin() + index);
    index += 3;
    const mjtNum* peg_mat = data_->xmat + 9 * peg_body_;
    std::copy(peg_mat + 6, peg_mat + 9, obs_.begin() + index);
  }

  void UpdateObs() {
    obs_.fill(0.0);
    if (is_peg_) {
      UpdatePegObs();
    } else {
      UpdateHandoverObs();
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
      std::copy(obs_.begin(), obs_.begin() + state_dim_, obs);
      CommitObservation("obs:state", &obs_state, reset);
    }
    state["info:terminated"_] = terminated_;
    state["info:out_of_bounds"_] = out_of_bounds_;
    state["info:reward_gripper_box"_] = reward_gripper_box_;
    state["info:reward_box_handover"_] = reward_box_handover_;
    state["info:reward_handover_target"_] = reward_handover_target_;
    state["info:reward_no_table_collision"_] = reward_no_table_collision_;
    state["info:reward_left_reward"_] = reward_left_reward_;
    state["info:reward_right_reward"_] = reward_right_reward_;
    state["info:reward_left_target_qpos"_] = reward_left_target_qpos_;
    state["info:reward_right_target_qpos"_] = reward_right_target_qpos_;
    state["info:reward_socket_z_up"_] = reward_socket_z_up_;
    state["info:reward_peg_z_up"_] = reward_peg_z_up_;
    state["info:reward_socket_entrance_reward"_] =
        reward_socket_entrance_reward_;
    state["info:reward_peg_end2_reward"_] = reward_peg_end2_reward_;
    state["info:reward_peg_insertion_reward"_] = reward_peg_insertion_reward_;
    state["info:peg_end2_dist_to_line"_] = peg_end2_dist_to_line_;
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
    CopyPadded(data_->site_xpos, model_->nsite * 3, &pad512);
    state["info:site_xpos"_].Assign(pad512.data(), pad512.size());
    std::array<mjtNum, 2048> pad2048{};
    CopyPadded(data_->xmat, model_->nbody * 9, &pad2048);
    state["info:xmat"_].Assign(pad2048.data(), pad2048.size());
    CopyPadded(data_->site_xmat, model_->nsite * 9, &pad2048);
    state["info:site_xmat"_].Assign(pad2048.data(), pad2048.size());
    std::array<mjtNum, 256> sensor_pad{};
    CopyPadded(data_->sensordata, model_->nsensordata, &sensor_pad);
    state["info:sensordata"_].Assign(sensor_pad.data(), sensor_pad.size());
    state["info:target_pos"_].Assign(target_pos_.data(), target_pos_.size());
    state["info:prev_potential"_] = prev_potential_;
    state["info:episode_picked"_] = episode_picked_ ? 1.0 : 0.0;
    state["info:steps"_] = static_cast<mjtNum>(task_steps_);
#endif
  }
};

using PlaygroundAlohaEnv =
    PlaygroundAlohaEnvBase<PlaygroundAlohaEnvSpec, false>;
using PlaygroundAlohaPixelEnv =
    PlaygroundAlohaEnvBase<PlaygroundAlohaPixelEnvSpec, true>;
using PlaygroundAlohaEnvPool = AsyncEnvPool<PlaygroundAlohaEnv>;
using PlaygroundAlohaPixelEnvPool = AsyncEnvPool<PlaygroundAlohaPixelEnv>;

}  // namespace mujoco_playground

#endif  // ENVPOOL_MUJOCO_PLAYGROUND_ALOHA_H_
