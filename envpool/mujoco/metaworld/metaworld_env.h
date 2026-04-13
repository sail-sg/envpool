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

#ifndef ENVPOOL_MUJOCO_METAWORLD_METAWORLD_ENV_H_
#define ENVPOOL_MUJOCO_METAWORLD_METAWORLD_ENV_H_

#include <mujoco.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/metaworld/tasks.h"
#include "envpool/mujoco/robotics/mujoco_env.h"
#include "envpool/mujoco/robotics/utils.h"

namespace metaworld {

using envpool::mujoco::PixelObservationEnvFns;
using envpool::mujoco::RenderCameraIdOrDefault;
using envpool::mujoco::RenderHeightOrDefault;
using envpool::mujoco::RenderWidthOrDefault;
using envpool::mujoco::StackSpec;
using gymnasium_robotics::BodyId;
using gymnasium_robotics::SiteId;

class MetaWorldEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("reward_threshold"_.Bind(10.0), "frame_skip"_.Bind(5),
                    "frame_stack"_.Bind(1),
                    "task_name"_.Bind(std::string("reach-v3")),
                    "partially_observable"_.Bind(true));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict("obs"_.Bind(StackSpec(Spec<mjtNum>({39}, {-inf, inf}),
                                          conf["frame_stack"_])),
                    "info:success"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
                    "info:near_object"_.Bind(Spec<mjtNum>({-1})),
                    "info:grasp_success"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
                    "info:grasp_reward"_.Bind(Spec<mjtNum>({-1})),
                    "info:in_place_reward"_.Bind(Spec<mjtNum>({-1})),
                    "info:obj_to_target"_.Bind(Spec<mjtNum>({-1})),
                    "info:unscaled_reward"_.Bind(Spec<mjtNum>({-1})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.Bind(Spec<mjtNum>({64})),
                    "info:qvel0"_.Bind(Spec<mjtNum>({64})),
                    "info:qacc0"_.Bind(Spec<mjtNum>({64})),
                    "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({64})),
                    "info:target0"_.Bind(Spec<mjtNum>({3})),
                    "info:rand_vec0"_.Bind(Spec<mjtNum>({6})),
                    "info:init_tcp0"_.Bind(Spec<mjtNum>({3})),
                    "info:init_left_pad0"_.Bind(Spec<mjtNum>({3})),
                    "info:init_right_pad0"_.Bind(Spec<mjtNum>({3})),
                    "info:mocap_pos0"_.Bind(Spec<mjtNum>({3})),
                    "info:mocap_quat0"_.Bind(Spec<mjtNum>({4})),
#endif
                    "info:task_id"_.Bind(Spec<int>({-1})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<float>({-1, 4}, {-1.0, 1.0})));
  }
};

using MetaWorldEnvSpec = EnvSpec<MetaWorldEnvFns>;
using MetaWorldPixelEnvFns = PixelObservationEnvFns<MetaWorldEnvFns>;
using MetaWorldPixelEnvSpec = EnvSpec<MetaWorldPixelEnvFns>;

struct RewardInfo {
  mjtNum reward{0.0};
  mjtNum success{0.0};
  mjtNum near_object{0.0};
  mjtNum grasp_success{0.0};
  mjtNum grasp_reward{0.0};
  mjtNum in_place_reward{0.0};
  mjtNum obj_to_target{0.0};
};

template <typename EnvSpecT, bool kFromPixels>
class MetaWorldEnvBase : public Env<EnvSpecT>,
                         public gymnasium_robotics::MujocoRobotEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  const TaskDef& task_;
  int task_index_;
  bool partially_observable_;
  std::array<mjtNum, 3> target_pos_{};
  std::array<mjtNum, 3> obj_init_pos_{};
  std::array<mjtNum, 3> init_tcp_{};
  std::array<mjtNum, 3> init_left_pad_{};
  std::array<mjtNum, 3> init_right_pad_{};
  std::array<mjtNum, 3> handle_init_pos_{};
  std::array<mjtNum, 3> hammer_init_pos_{};
  std::array<mjtNum, 3> nail_init_pos_{};
  std::array<mjtNum, 3> peg_head_pos_init_{};
  std::array<mjtNum, 3> stick_init_pos_{};
  std::array<mjtNum, 3> window_handle_pos_init_{};
  std::array<mjtNum, 3> lever_pos_init_{};
  std::array<mjtNum, 6> rand_vec_{};
  std::array<mjtNum, 18> prev_obs_{};
  int mocap_body_id_{-1};
  int mocap_id_{-1};
  int goal_site_id_{-1};
  int obj_body_id_{-1};
  int obj_geom_id_{-1};
  mjtNum obj_height_{0.0};
  mjtNum height_target_{0.0};
  mjtNum lift_thresh_{0.04};
  mjtNum max_dist_{0.0};
  mjtNum max_push_dist_{0.0};
  mjtNum max_pull_dist_{0.0};
  mjtNum max_place_dist_{0.0};
  mjtNum max_placing_dist_{0.0};
  mjtNum max_reach_dist_{0.0};
  mjtNum obj_to_target_init_{0.0};
  mjtNum target_to_obj_init_{0.0};
  mjtNum target_reward_{0.0};
  std::array<mjtNum, 64> qpos0_{};
  std::array<mjtNum, 64> qvel0_{};
  std::array<mjtNum, 64> qacc0_{};
  std::array<mjtNum, 64> qacc_warmstart0_{};
  std::array<mjtNum, 3> mocap_pos0_{};
  std::array<mjtNum, 4> mocap_quat0_{};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  MetaWorldEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        gymnasium_robotics::MujocoRobotEnv(
            spec.config["base_path"_],
            ModelPath(
                spec.config["base_path"_],
                GetTaskDef(std::string(spec.config["task_name"_])).xml_file),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_],
            spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        task_(GetTaskDef(std::string(spec.config["task_name"_]))),
        task_index_(TaskIndex(task_.name)),
        partially_observable_(spec.config["partially_observable"_]) {
    target_pos_ = ToMjt(task_.init_goal);
    obj_init_pos_ = ToMjt(task_.init_obj);
    CacheIds();
    ResetMetaWorldMocapWelds();
    InitializeRobotEnv();
    init_left_pad_ = BodyPos("leftpad");
    init_right_pad_ = BodyPos("rightpad");
    prev_obs_ = CurrentObsNoGoal();
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    ResetToInitialState();
    ResetModel();
    CaptureResetState();
    prev_obs_ = CurrentObsNoGoal();
    WriteState(ComputeReward({0.0, 0.0, 0.0, 0.0}), true);
  }

  void Step(const Action& action) override {
    const auto* raw = static_cast<float*>(action["action"_].Data());
    std::array<mjtNum, 4> act{raw[0], raw[1], raw[2], raw[3]};
    SetXYZAction({act[0], act[1], act[2]});
    if (model_->nu >= 2) {
      data_->ctrl[0] = act[3];
      data_->ctrl[1] = -act[3];
    }
    DoSimulation();
    ++elapsed_step_;
    done_ = elapsed_step_ >= max_episode_steps_;
    PositionTargetSites();
    mj_forward(model_, data_);
    WriteState(ComputeReward(act), false);
  }

 protected:
  static std::string ModelPath(const std::string& base_path,
                               const std::string& xml_file) {
    return base_path + "/mujoco/metaworld/assets/" + xml_file;
  }

  static std::array<mjtNum, 3> ToMjt(const std::array<double, 3>& value) {
    return {static_cast<mjtNum>(value[0]), static_cast<mjtNum>(value[1]),
            static_cast<mjtNum>(value[2])};
  }

  static int TaskIndex(std::string_view task_name) {
    for (int i = 0; i < static_cast<int>(kMetaWorldTasks.size()); ++i) {
      if (kMetaWorldTasks[i].name == task_name) {
        return i;
      }
    }
    throw std::runtime_error("Unknown MetaWorld task_name: " +
                             std::string(task_name));
  }

  static mjtNum Norm(const std::array<mjtNum, 3>& value) {
    return std::sqrt(value[0] * value[0] + value[1] * value[1] +
                     value[2] * value[2]);
  }

  static mjtNum Norm4(const std::array<mjtNum, 4>& value) {
    return std::sqrt(value[0] * value[0] + value[1] * value[1] +
                     value[2] * value[2] + value[3] * value[3]);
  }

  static mjtNum Distance(const std::array<mjtNum, 3>& lhs,
                         const std::array<mjtNum, 3>& rhs) {
    return Norm({lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]});
  }

  static mjtNum Distance2(const std::array<mjtNum, 3>& lhs,
                          const std::array<mjtNum, 3>& rhs) {
    mjtNum dx = lhs[0] - rhs[0];
    mjtNum dy = lhs[1] - rhs[1];
    return std::sqrt(dx * dx + dy * dy);
  }

  static mjtNum LongTailTolerance(mjtNum x, mjtNum lower, mjtNum upper,
                                  mjtNum margin) {
    if (lower <= x && x <= upper) {
      return 1.0;
    }
    if (margin <= 0.0) {
      return 0.0;
    }
    mjtNum d = x < lower ? lower - x : x - upper;
    mjtNum scaled = d / margin;
    // MetaWorld uses dm_control's default value_at_margin=0.1. For a
    // long-tail sigmoid that is equivalent to sqrt(1 / 0.1 - 1) == 3.
    scaled *= 3.0;
    return 1.0 / ((scaled * scaled) + 1.0);
  }

  static mjtNum HamacherProduct(mjtNum lhs, mjtNum rhs) {
    mjtNum denom = lhs + rhs - lhs * rhs;
    if (denom <= 0.0) {
      return 0.0;
    }
    return lhs * rhs / denom;
  }

  static mjtNum GaussianTolerance(mjtNum x, mjtNum lower, mjtNum upper,
                                  mjtNum margin) {
    if (lower <= x && x <= upper) {
      return 1.0;
    }
    if (margin <= 0.0) {
      return 0.0;
    }
    mjtNum d = (x < lower ? lower - x : x - upper) / margin;
    mjtNum scale = std::sqrt(-2.0 * std::log(0.1));
    return std::exp(-0.5 * (d * scale) * (d * scale));
  }

  static mjtNum RectPrismTolerance(const std::array<mjtNum, 3>& curr,
                                   const std::array<mjtNum, 3>& zero,
                                   const std::array<mjtNum, 3>& one) {
    auto in_range = [](mjtNum a, mjtNum b, mjtNum c) {
      return c >= b ? (b <= a && a <= c) : (c <= a && a <= b);
    };
    if (!in_range(curr[0], zero[0], one[0]) ||
        !in_range(curr[1], zero[1], one[1]) ||
        !in_range(curr[2], zero[2], one[2])) {
      return 1.0;
    }
    auto diff = Sub(one, zero);
    if (diff[0] == 0.0 || diff[1] == 0.0 || diff[2] == 0.0) {
      return 1.0;
    }
    mjtNum x_scale = (curr[0] - zero[0]) / diff[0];
    mjtNum y_scale = (curr[1] - zero[1]) / diff[1];
    mjtNum z_scale = (curr[2] - zero[2]) / diff[2];
    return x_scale * y_scale * z_scale;
  }

  static std::array<mjtNum, 3> Scale(const std::array<mjtNum, 3>& value,
                                     const std::array<mjtNum, 3>& scale) {
    return {value[0] * scale[0], value[1] * scale[1], value[2] * scale[2]};
  }

  static mjtNum DistanceScaled(const std::array<mjtNum, 3>& lhs,
                               const std::array<mjtNum, 3>& rhs,
                               const std::array<mjtNum, 3>& scale) {
    return Norm(Scale(Sub(lhs, rhs), scale));
  }

  static bool HasName(const mjModel* model, mjtObj type, const char* name) {
    return mj_name2id(model, type, name) >= 0;
  }

  static std::array<mjtNum, 3> Add(const std::array<mjtNum, 3>& lhs,
                                   const std::array<mjtNum, 3>& rhs) {
    return {lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
  }

  static std::array<mjtNum, 3> Sub(const std::array<mjtNum, 3>& lhs,
                                   const std::array<mjtNum, 3>& rhs) {
    return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
  }

  static std::array<mjtNum, 3> WithZ(const std::array<mjtNum, 3>& value,
                                     mjtNum z) {
    return {value[0], value[1], z};
  }

  void CacheIds() {
    mocap_body_id_ = mj_name2id(model_, mjOBJ_BODY, "mocap");
    mocap_id_ = mocap_body_id_ >= 0 ? model_->body_mocapid[mocap_body_id_] : -1;
    goal_site_id_ = mj_name2id(model_, mjOBJ_SITE, "goal");
    obj_body_id_ = mj_name2id(model_, mjOBJ_BODY, "obj");
    obj_geom_id_ = mj_name2id(model_, mjOBJ_GEOM, "objGeom");
  }

  void ResetMetaWorldMocapWelds() {
    if (model_->nmocap <= 0) {
      return;
    }
    constexpr std::array<mjtNum, 11> k_weld_data{0.0,  0.0, 0.0, 0.0, 0.0, 0.0,
                                                 -1.0, 0.0, 0.0, 0.0, 5.0};
    for (int eq_id = 0; eq_id < model_->neq; ++eq_id) {
      if (model_->eq_type[eq_id] != mjEQ_WELD) {
        continue;
      }
      mjtNum* eq_data = model_->eq_data + eq_id * mjNEQDATA;
      std::memcpy(
          eq_data, k_weld_data.data(),
          sizeof(mjtNum) *
              std::min<int>(mjNEQDATA, static_cast<int>(k_weld_data.size())));
    }
    mj_forward(model_, data_);
  }

  std::array<mjtNum, 3> BodyPos(const char* name) const {
    int id = mj_name2id(model_, mjOBJ_BODY, name);
    if (id < 0) {
      return {0.0, 0.0, 0.0};
    }
    return {data_->xpos[3 * id], data_->xpos[3 * id + 1],
            data_->xpos[3 * id + 2]};
  }

  std::array<mjtNum, 3> BodyModelPos(const char* name) const {
    int id = mj_name2id(model_, mjOBJ_BODY, name);
    if (id < 0) {
      return {0.0, 0.0, 0.0};
    }
    return {model_->body_pos[3 * id], model_->body_pos[3 * id + 1],
            model_->body_pos[3 * id + 2]};
  }

  std::array<mjtNum, 3> GeomPos(const char* name) const {
    int id = mj_name2id(model_, mjOBJ_GEOM, name);
    if (id < 0) {
      return {0.0, 0.0, 0.0};
    }
    return {data_->geom_xpos[3 * id], data_->geom_xpos[3 * id + 1],
            data_->geom_xpos[3 * id + 2]};
  }

  std::array<mjtNum, 3> SitePos(const char* name) const {
    int id = mj_name2id(model_, mjOBJ_SITE, name);
    if (id < 0) {
      return {0.0, 0.0, 0.0};
    }
    return {data_->site_xpos[3 * id], data_->site_xpos[3 * id + 1],
            data_->site_xpos[3 * id + 2]};
  }

  std::array<mjtNum, 3> SiteModelPos(const char* name) const {
    int id = mj_name2id(model_, mjOBJ_SITE, name);
    if (id < 0) {
      return {0.0, 0.0, 0.0};
    }
    return {model_->site_pos[3 * id], model_->site_pos[3 * id + 1],
            model_->site_pos[3 * id + 2]};
  }

  std::array<mjtNum, 4> BodyQuat(const char* name) const {
    int id = mj_name2id(model_, mjOBJ_BODY, name);
    if (id < 0) {
      return {0.0, 0.0, 0.0, 0.0};
    }
    const mjtNum* quat = data_->xquat + 4 * id;
    return {quat[0], quat[1], quat[2], quat[3]};
  }

  std::array<mjtNum, 4> BodyQuatXYZW(const char* name) const {
    auto quat = BodyQuat(name);
    return {quat[1], quat[2], quat[3], quat[0]};
  }

  std::array<mjtNum, 4> MatToQuat(const mjtNum* xmat) const {
    std::array<mjtNum, 4> quat{};
    mju_mat2Quat(quat.data(), xmat);
    return {quat[1], quat[2], quat[3], quat[0]};
  }

  std::array<mjtNum, 4> GeomQuat(const char* name) const {
    int id = mj_name2id(model_, mjOBJ_GEOM, name);
    if (id < 0) {
      return {0.0, 0.0, 0.0, 0.0};
    }
    return MatToQuat(data_->geom_xmat + 9 * id);
  }

  std::array<mjtNum, 4> SiteQuat(const char* name) const {
    int id = mj_name2id(model_, mjOBJ_SITE, name);
    if (id < 0) {
      return {0.0, 0.0, 0.0, 0.0};
    }
    return MatToQuat(data_->site_xmat + 9 * id);
  }

  void SetModelBodyPos(const char* name, const std::array<mjtNum, 3>& pos) {
    int id = mj_name2id(model_, mjOBJ_BODY, name);
    if (id < 0) {
      return;
    }
    for (int i = 0; i < 3; ++i) {
      model_->body_pos[3 * id + i] = pos[i];
    }
  }

  void SetModelSitePos(const char* name, const std::array<mjtNum, 3>& pos) {
    int id = mj_name2id(model_, mjOBJ_SITE, name);
    if (id < 0) {
      return;
    }
    for (int i = 0; i < 3; ++i) {
      model_->site_pos[3 * id + i] = pos[i];
    }
  }

  void SetJointQpos(const char* name, mjtNum value) {
    int id = mj_name2id(model_, mjOBJ_JOINT, name);
    if (id < 0) {
      return;
    }
    int adr = model_->jnt_qposadr[id];
    if (0 <= adr && adr < model_->nq) {
      data_->qpos[adr] = value;
    }
  }

  std::array<mjtNum, 3> TcpCenter() const {
    auto right = SitePos("rightEndEffector");
    auto left = SitePos("leftEndEffector");
    return {(right[0] + left[0]) * 0.5, (right[1] + left[1]) * 0.5,
            (right[2] + left[2]) * 0.5};
  }

  void AppendObject(std::array<mjtNum, 14>* padded, int* offset,
                    const std::array<mjtNum, 3>& pos,
                    const std::array<mjtNum, 4>& quat) const {
    if (*offset + 7 > static_cast<int>(padded->size())) {
      return;
    }
    for (int i = 0; i < 3; ++i) {
      (*padded)[(*offset)++] = pos[i];
    }
    for (int i = 0; i < 4; ++i) {
      (*padded)[(*offset)++] = quat[i];
    }
  }

  std::array<mjtNum, 14> ObjectObsPadded() const {
    std::array<mjtNum, 14> padded{};
    int offset = 0;
    auto zeros = std::array<mjtNum, 4>{0.0, 0.0, 0.0, 0.0};
    switch (task_index_) {
      case 0:   // assembly
      case 12:  // disassemble
        AppendObject(&padded, &offset, SitePos("RoundNut-8"),
                     BodyQuat("RoundNut"));
        break;
      case 1:
        AppendObject(&padded, &offset, BodyPos("bsktball"),
                     BodyQuat("bsktball"));
        break;
      case 2:
      case 17:
      case 29:
        AppendObject(&padded, &offset, BodyPos("obj"), BodyQuat("obj"));
        break;
      case 3:
        AppendObject(&padded, &offset, BodyPos("top_link"),
                     BodyQuat("top_link"));
        break;
      case 4:
      case 5:
        AppendObject(&padded, &offset,
                     Add(BodyPos("button"), {0.0, 0.0, 0.193}),
                     BodyQuat("button"));
        break;
      case 6:
      case 7:
        AppendObject(&padded, &offset,
                     Add(BodyPos("button"), {0.0, -0.193, 0.0}),
                     BodyQuat("button"));
        break;
      case 8:
        AppendObject(&padded, &offset, SitePos("buttonStart"),
                     {1.0, 0.0, 0.0, 0.0});
        break;
      case 9:
      case 10:
        AppendObject(&padded, &offset, BodyPos("obj"), GeomQuat("mug"));
        break;
      case 11: {
        auto dial = BodyPos("dial");
        int joint_id = mj_name2id(model_, mjOBJ_JOINT, "knob_Joint_1");
        mjtNum angle = 0.0;
        if (joint_id >= 0) {
          int adr = model_->jnt_qposadr[joint_id];
          if (0 <= adr && adr < model_->nq) {
            angle = data_->qpos[adr];
          }
        }
        AppendObject(
            &padded, &offset,
            Add(dial, {std::sin(angle) * 0.05, -std::cos(angle) * 0.05, 0.0}),
            BodyQuat("dial"));
        break;
      }
      case 13:
      case 15:
        AppendObject(&padded, &offset, GeomPos("handle"), GeomQuat("handle"));
        break;
      case 14:
        AppendObject(&padded, &offset, SitePos("lockStartLock"),
                     BodyQuat("door_link"));
        break;
      case 16:
        AppendObject(&padded, &offset, SitePos("lockStartUnlock"),
                     BodyQuat("door_link"));
        break;
      case 18:
        AppendObject(&padded, &offset,
                     Add(BodyPos("drawer_link"), {0.0, -0.16, 0.05}), zeros);
        break;
      case 19:
        AppendObject(&padded, &offset,
                     Add(BodyPos("drawer_link"), {0.0, -0.16, 0.0}),
                     BodyQuat("drawer_link"));
        break;
      case 20:
        AppendObject(&padded, &offset,
                     Add(SitePos("handleStartOpen"), {0.0, 0.0, -0.01}),
                     BodyQuat("faucetBase"));
        break;
      case 21:
        AppendObject(&padded, &offset,
                     Add(SitePos("handleStartClose"), {0.0, 0.0, -0.01}),
                     BodyQuat("faucetBase"));
        break;
      case 22:
        AppendObject(&padded, &offset, BodyPos("hammer"), BodyQuat("hammer"));
        AppendObject(&padded, &offset, BodyPos("nail_link"),
                     BodyQuat("nail_link"));
        break;
      case 23:
      case 24:
        AppendObject(&padded, &offset, SitePos("handleStart"), zeros);
        break;
      case 25:
        AppendObject(&padded, &offset, SitePos("handleCenter"), zeros);
        break;
      case 26:
        AppendObject(&padded, &offset, SitePos("handleRight"), zeros);
        break;
      case 27:
        AppendObject(&padded, &offset, SitePos("leverStart"),
                     GeomQuat("objGeom"));
        break;
      case 28:
      case 40:
      case 41:
      case 42:
        AppendObject(&padded, &offset, GeomPos("objGeom"), GeomQuat("objGeom"));
        break;
      case 30:
      case 43:
      case 44:
      case 45:
      case 46:
        AppendObject(&padded, &offset, BodyPos("obj"), GeomQuat("objGeom"));
        break;
      case 31:
      case 32:
      case 33:
      case 34:
        AppendObject(&padded, &offset, GeomPos("puck"), GeomQuat("puck"));
        break;
      case 35:
        AppendObject(&padded, &offset, SitePos("pegGrasp"),
                     SiteQuat("pegGrasp"));
        break;
      case 36:
        AppendObject(&padded, &offset, SitePos("pegEnd"), BodyQuat("plug1"));
        break;
      case 37:
        AppendObject(&padded, &offset, BodyPos("soccer_ball"),
                     BodyQuatXYZW("soccer_ball"));
        break;
      case 38:
        AppendObject(&padded, &offset, BodyPos("stick"), BodyQuatXYZW("stick"));
        AppendObject(&padded, &offset,
                     Add(SitePos("insertion"), {0.0, 0.09, 0.0}), zeros);
        break;
      case 39:
        AppendObject(&padded, &offset, BodyPos("stick"), BodyQuatXYZW("stick"));
        AppendObject(&padded, &offset, SitePos("insertion"), zeros);
        break;
      case 47:
        AppendObject(&padded, &offset, BodyPos("obj"), BodyQuat("obj"));
        break;
      case 48:
        AppendObject(&padded, &offset, SitePos("handleOpenStart"), zeros);
        break;
      case 49:
        AppendObject(&padded, &offset, SitePos("handleCloseStart"), zeros);
        break;
      default:
        AppendObject(&padded, &offset, BodyPos("obj"), GeomQuat("objGeom"));
        break;
    }
    return padded;
  }

  std::array<mjtNum, 3> ObjPos() const {
    auto padded = ObjectObsPadded();
    return {padded[0], padded[1], padded[2]};
  }

  std::array<mjtNum, 18> CurrentObsNoGoal() const {
    std::array<mjtNum, 18> obs{};
    auto hand = BodyPos("hand");
    auto right = BodyPos("rightclaw");
    auto left = BodyPos("leftclaw");
    auto obj = ObjectObsPadded();
    mjtNum gripper_distance = Distance(right, left) / 0.1;
    gripper_distance = std::clamp<mjtNum>(gripper_distance, 0.0, 1.0);
    obs[0] = hand[0];
    obs[1] = hand[1];
    obs[2] = hand[2];
    obs[3] = gripper_distance;
    for (int i = 0; i < 14; ++i) {
      obs[4 + i] = obj[i];
    }
    return obs;
  }

  void SetMocapPos(const std::array<mjtNum, 3>& pos) {
    if (mocap_id_ < 0) {
      return;
    }
    for (int i = 0; i < 3; ++i) {
      data_->mocap_pos[3 * mocap_id_ + i] = pos[i];
    }
    data_->mocap_quat[4 * mocap_id_] = 1.0;
    data_->mocap_quat[4 * mocap_id_ + 1] = 0.0;
    data_->mocap_quat[4 * mocap_id_ + 2] = 1.0;
    data_->mocap_quat[4 * mocap_id_ + 3] = 0.0;
  }

  void SetXYZAction(const std::array<mjtNum, 3>& action) {
    if (mocap_id_ < 0) {
      return;
    }
    for (int i = 0; i < 3; ++i) {
      mjtNum delta = std::clamp<mjtNum>(action[i], -1.0, 1.0) * 0.01;
      mjtNum value = data_->mocap_pos[3 * mocap_id_ + i] + delta;
      value = std::clamp<mjtNum>(value, task_.hand_low[i], task_.hand_high[i]);
      data_->mocap_pos[3 * mocap_id_ + i] = value;
    }
    data_->mocap_quat[4 * mocap_id_] = 1.0;
    data_->mocap_quat[4 * mocap_id_ + 1] = 0.0;
    data_->mocap_quat[4 * mocap_id_ + 2] = 1.0;
    data_->mocap_quat[4 * mocap_id_ + 3] = 0.0;
  }

  void SetQposSlice(int start, std::initializer_list<mjtNum> values,
                    bool scalar = false) {
    if (start >= model_->nq || values.size() == 0) {
      return;
    }
    const int end = std::min<int>(model_->nq, start + 3);
    if (scalar) {
      mjtNum value = *values.begin();
      for (int i = start; i < end; ++i) {
        data_->qpos[i] = value;
      }
    } else {
      int i = start;
      for (mjtNum value : values) {
        if (i >= end) {
          break;
        }
        data_->qpos[i++] = value;
      }
    }
    const int qvel_end = std::min<int>(model_->nv, start + 6);
    for (int i = start; i < qvel_end; ++i) {
      data_->qvel[i] = 0.0;
    }
    mj_forward(model_, data_);
  }

  void SetObjXYZ(const std::array<mjtNum, 3>& pos) {
    SetQposSlice(9, {pos[0], pos[1], pos[2]});
  }

  void SetPegUnplugObjXYZ(const std::array<mjtNum, 3>& pos) {
    for (int i = 0; i < 3 && 9 + i < model_->nq; ++i) {
      data_->qpos[9 + i] = pos[i];
    }
    if (model_->nq >= 16) {
      data_->qpos[12] = 1.0;
      data_->qpos[13] = 0.0;
      data_->qpos[14] = 0.0;
      data_->qpos[15] = 0.0;
    }
    for (int i = 9; i < std::min<int>(12, model_->nv); ++i) {
      data_->qvel[i] = 0.0;
    }
    mj_forward(model_, data_);
  }

  void SetCoffeeObjXYZ(const std::array<mjtNum, 3>& pos) {
    for (int i = 0; i < 3 && i < model_->nq; ++i) {
      data_->qpos[i] = pos[i];
    }
    for (int i = 9; i < std::min<int>(15, model_->nv); ++i) {
      data_->qvel[i] = 0.0;
    }
    mj_forward(model_, data_);
  }

  void SetObjScalar(mjtNum value) { SetQposSlice(9, {value}, true); }

  void SetObjQpos2(mjtNum first, mjtNum second) {
    SetQposSlice(9, {first, second});
  }

  void SetStickXYZ(const std::array<mjtNum, 3>& pos) {
    SetQposSlice(9, {pos[0], pos[1], pos[2]});
  }

  void SetStickObjectQpos(mjtNum first, mjtNum second) {
    SetQposSlice(16, {first, second});
  }

  void StepMj(int steps) {
    for (int i = 0; i < steps; ++i) {
      mj_step(model_, data_);
    }
  }

  std::array<mjtNum, 6> SampleRandVec() {
    std::array<mjtNum, 6> rand_vec{};
    for (int i = 0; i < task_.random_dim; ++i) {
      std::uniform_real_distribution<mjtNum> dist(task_.random_low[i],
                                                  task_.random_high[i]);
      rand_vec[i] = dist(gen_);
    }
    rand_vec_ = rand_vec;
    return rand_vec;
  }

  void ResetHand(int steps = 50) {
    for (int i = 0; i < steps; ++i) {
      SetMocapPos(ToMjt(task_.init_hand));
      if (model_->nu >= 2) {
        data_->ctrl[0] = -1.0;
        data_->ctrl[1] = 1.0;
      }
      DoSimulation();
    }
    init_tcp_ = TcpCenter();
  }

  void ResetModel() {
    ResetHand();
    obj_init_pos_ = ToMjt(task_.init_obj);
    target_pos_ = ToMjt(task_.init_goal);
    lift_thresh_ = 0.04;
    max_dist_ = 0.0;
    max_push_dist_ = 0.0;
    max_pull_dist_ = 0.0;
    max_place_dist_ = 0.0;
    max_placing_dist_ = 0.0;
    max_reach_dist_ = 0.0;
    obj_to_target_init_ = 0.0;
    target_to_obj_init_ = 0.0;
    target_reward_ = 0.0;

    auto rand_vec = SampleRandVec();
    auto split_first = [&]() -> std::array<mjtNum, 3> {
      return {rand_vec[0], rand_vec[1], rand_vec[2]};
    };
    auto split_second = [&]() -> std::array<mjtNum, 3> {
      return {rand_vec[3], rand_vec[4], rand_vec[5]};
    };

    switch (task_index_) {
      case 0: {  // assembly-v3
        while (Distance2(split_first(), split_second()) < 0.1) {
          rand_vec = SampleRandVec();
        }
        obj_init_pos_ = split_first();
        target_pos_ = split_second();
        SetObjXYZ(obj_init_pos_);
        SetModelBodyPos("peg", Sub(target_pos_, {0.0, 0.0, 0.05}));
        SetModelSitePos("pegTop", target_pos_);
        break;
      }
      case 1: {  // basketball-v3
        auto basket_pos = split_second();
        while (Distance2(split_first(), basket_pos) < 0.15) {
          rand_vec = SampleRandVec();
          basket_pos = split_second();
        }
        obj_init_pos_ = {rand_vec[0], rand_vec[1], task_.init_obj[2]};
        SetModelBodyPos("basket_goal", basket_pos);
        mj_forward(model_, data_);
        target_pos_ = SitePos("goal");
        SetObjXYZ(obj_init_pos_);
        SetModelSitePos("goal", target_pos_);
        lift_thresh_ = 0.3;
        obj_height_ = GeomPos("objGeom")[2];
        height_target_ = obj_height_ + lift_thresh_;
        max_placing_dist_ =
            Distance({obj_init_pos_[0], obj_init_pos_[1], height_target_},
                     target_pos_) +
            height_target_;
        break;
      }
      case 2: {  // bin-picking-v3
        auto obj_height = BodyPos("obj")[2];
        obj_init_pos_ = {rand_vec[0], rand_vec[1], obj_height};
        SetObjXYZ(obj_init_pos_);
        target_pos_ = BodyPos("bin_goal");
        lift_thresh_ = 0.1;
        obj_height_ = BodyPos("obj")[2];
        height_target_ = obj_height_ + lift_thresh_;
        max_placing_dist_ =
            Distance2(obj_init_pos_, target_pos_) + height_target_;
        break;
      }
      case 3: {  // box-close-v3
        while (Distance2(split_first(), split_second()) < 0.25) {
          rand_vec = SampleRandVec();
        }
        obj_init_pos_ = {rand_vec[0], rand_vec[1], task_.init_obj[2]};
        target_pos_ = split_second();
        SetModelBodyPos(
            "boxbody", {target_pos_[0], target_pos_[1], BodyPos("boxbody")[2]});
        StepMj(spec_.config["frame_skip"_]);
        SetObjXYZ(obj_init_pos_);
        SetModelSitePos("goal", target_pos_);
        lift_thresh_ = 0.12;
        obj_height_ = GeomPos("BoxHandleGeom")[2];
        height_target_ = obj_height_ + lift_thresh_;
        max_placing_dist_ =
            Distance({obj_init_pos_[0], obj_init_pos_[1], height_target_},
                     target_pos_) +
            height_target_;
        break;
      }
      case 4:    // button-press-topdown-v3
      case 5: {  // button-press-topdown-wall-v3
        obj_init_pos_ = split_first();
        SetModelBodyPos("box", obj_init_pos_);
        mj_forward(model_, data_);
        target_pos_ = SitePos("hole");
        obj_to_target_init_ =
            std::abs(target_pos_[2] - SitePos("buttonStart")[2]);
        max_dist_ = obj_to_target_init_;
        break;
      }
      case 6:    // button-press-v3
      case 7: {  // button-press-wall-v3
        obj_init_pos_ = split_first();
        SetModelBodyPos("box", obj_init_pos_);
        SetObjScalar(0.0);
        target_pos_ = SitePos("hole");
        obj_to_target_init_ =
            std::abs(target_pos_[1] - SitePos("buttonStart")[1]);
        max_dist_ = task_index_ == 7
                        ? std::abs(SitePos("buttonStart")[2] - target_pos_[2])
                        : obj_to_target_init_;
        break;
      }
      case 8: {  // coffee-button-v3
        obj_init_pos_ = split_first();
        SetModelBodyPos("coffee_machine", obj_init_pos_);
        SetCoffeeObjXYZ(Add(obj_init_pos_, {0.0, -0.22, 0.0}));
        target_pos_ = Add(obj_init_pos_, {0.0, -0.22 + 0.03, 0.3});
        max_dist_ = std::abs(SitePos("buttonStart")[1] - target_pos_[1]);
        break;
      }
      case 9:     // coffee-pull-v3
      case 10: {  // coffee-push-v3
        auto pos_mug_init = split_first();
        auto pos_mug_goal = split_second();
        while (Distance2(pos_mug_init, pos_mug_goal) < 0.15) {
          rand_vec = SampleRandVec();
          pos_mug_init = split_first();
          pos_mug_goal = split_second();
        }
        obj_init_pos_ = pos_mug_init;
        SetCoffeeObjXYZ(obj_init_pos_);
        SetModelBodyPos("coffee_machine",
                        Add(task_index_ == 9 ? pos_mug_init : pos_mug_goal,
                            {0.0, 0.22, 0.0}));
        target_pos_ = pos_mug_goal;
        SetModelSitePos("mug_goal", target_pos_);
        max_pull_dist_ = Distance2(obj_init_pos_, target_pos_);
        max_push_dist_ = max_pull_dist_;
        break;
      }
      case 11: {  // dial-turn-v3
        obj_init_pos_ = split_first();
        target_pos_ = Add(obj_init_pos_, {0.0, 0.03, 0.03});
        SetModelBodyPos("dial", obj_init_pos_);
        mj_forward(model_, data_);
        handle_init_pos_ = Add(ObjPos(), {0.05, 0.02, 0.09});
        SetModelSitePos("goal", target_pos_);
        max_pull_dist_ = std::abs(target_pos_[1] - obj_init_pos_[1]);
        break;
      }
      case 12: {  // disassemble-v3
        while (Distance2(split_first(), split_second()) < 0.1) {
          rand_vec = SampleRandVec();
        }
        obj_init_pos_ = split_first();
        target_pos_ = Add(obj_init_pos_, {0.0, 0.0, 0.15});
        SetModelBodyPos("peg", Add(obj_init_pos_, {0.0, 0.0, 0.03}));
        SetModelSitePos("pegTop", Add(obj_init_pos_, {0.0, 0.0, 0.08}));
        mj_forward(model_, data_);
        SetObjXYZ(obj_init_pos_);
        lift_thresh_ = 0.05;
        obj_height_ = BodyPos("RoundNut")[2];
        height_target_ = obj_height_ + lift_thresh_;
        max_placing_dist_ =
            Distance({obj_init_pos_[0], obj_init_pos_[1], height_target_},
                     target_pos_) +
            height_target_;
        break;
      }
      case 13: {  // door-close-v3
        obj_init_pos_ = split_first();
        target_pos_ = Add(obj_init_pos_, {0.2, -0.2, 0.0});
        SetModelBodyPos("door", obj_init_pos_);
        SetModelSitePos("goal", target_pos_);
        SetObjScalar(-1.5708);
        max_pull_dist_ = Distance2(GeomPos("handle"), target_pos_);
        break;
      }
      case 14: {  // door-lock-v3
        SetModelBodyPos("door", split_first());
        StepMj(spec_.config["frame_skip"_]);
        obj_init_pos_ = BodyPos("lock_link");
        target_pos_ = Add(obj_init_pos_, {0.0, -0.04, -0.1});
        max_pull_dist_ = Distance(target_pos_, obj_init_pos_);
        break;
      }
      case 15: {  // door-open-v3
        obj_init_pos_ = split_first();
        target_pos_ = Add(obj_init_pos_, {-0.3, -0.45, 0.0});
        SetModelBodyPos("door", obj_init_pos_);
        SetModelSitePos("goal", target_pos_);
        SetObjScalar(0.0);
        max_pull_dist_ = Distance2(GeomPos("handle"), target_pos_);
        target_reward_ = 1000.0 * max_pull_dist_ + 2000.0;
        break;
      }
      case 16: {  // door-unlock-v3
        SetModelBodyPos("door", split_first());
        SetObjScalar(1.5708);
        obj_init_pos_ = BodyPos("lock_link");
        target_pos_ = Add(obj_init_pos_, {0.1, -0.04, 0.0});
        max_pull_dist_ = Distance(target_pos_, obj_init_pos_);
        break;
      }
      case 17: {  // hand-insert-v3
        while (Distance2(split_first(), split_second()) < 0.15) {
          rand_vec = SampleRandVec();
        }
        obj_init_pos_ = {rand_vec[0], rand_vec[1], task_.init_obj[2]};
        target_pos_ = split_second();
        SetObjXYZ(obj_init_pos_);
        SetModelSitePos("goal", target_pos_);
        max_reach_dist_ = std::abs(task_.init_hand[2] - target_pos_[2]);
        break;
      }
      case 18: {  // drawer-close-v3
        max_dist_ = 0.15;
        SetModelBodyPos("drawer", split_first());
        target_pos_ = Add(split_first(), {0.0, -0.16, 0.09});
        SetObjScalar(-max_dist_);
        obj_init_pos_ = ObjPos();
        SetModelSitePos("goal", target_pos_);
        target_reward_ = 1000.0 * max_dist_ + 2000.0;
        break;
      }
      case 19: {  // drawer-open-v3
        max_dist_ = 0.2;
        obj_init_pos_ = split_first();
        SetModelBodyPos("drawer", obj_init_pos_);
        target_pos_ = Add(obj_init_pos_, {0.0, -0.16 - max_dist_, 0.09});
        SetModelSitePos("goal", target_pos_);
        target_reward_ = 1000.0 * max_dist_ + 2000.0;
        break;
      }
      case 20: {  // faucet-open-v3
        obj_init_pos_ = split_first();
        SetModelBodyPos("faucetBase", obj_init_pos_);
        target_pos_ = Add(obj_init_pos_, {0.175, 0.0, 0.125});
        SetModelSitePos("goal_open", target_pos_);
        max_pull_dist_ = Distance(target_pos_, obj_init_pos_);
        break;
      }
      case 21: {  // faucet-close-v3
        obj_init_pos_ = split_first();
        SetModelBodyPos("faucetBase", obj_init_pos_);
        target_pos_ = Add(obj_init_pos_, {-0.175, 0.0, 0.125});
        mj_forward(model_, data_);
        SetModelSitePos("goal_close", target_pos_);
        max_pull_dist_ = Distance(target_pos_, obj_init_pos_);
        break;
      }
      case 22: {  // hammer-v3
        SetModelBodyPos("box", {0.24, 0.85, 0.0});
        mj_forward(model_, data_);
        target_pos_ = SitePos("goal");
        hammer_init_pos_ = split_first();
        nail_init_pos_ = SitePos("nailHead");
        obj_init_pos_ = hammer_init_pos_;
        SetObjXYZ(hammer_init_pos_);
        lift_thresh_ = 0.09;
        obj_height_ = BodyPos("hammer")[2];
        height_target_ = obj_height_ + lift_thresh_;
        max_placing_dist_ =
            Distance({hammer_init_pos_[0], hammer_init_pos_[1], height_target_},
                     obj_init_pos_) +
            height_target_ + std::abs(obj_init_pos_[1] - target_pos_[1]);
        break;
      }
      case 23:    // handle-press-side-v3
      case 24: {  // handle-press-v3
        obj_init_pos_ = split_first();
        SetModelBodyPos("box", obj_init_pos_);
        SetObjScalar(-0.001);
        target_pos_ = SitePos("goalPress");
        handle_init_pos_ = ObjPos();
        max_dist_ = std::abs(SitePos("handleStart")[2] - target_pos_[2]);
        target_reward_ = 1000.0 * max_dist_ + 2000.0;
        break;
      }
      case 25:    // handle-pull-side-v3
      case 26: {  // handle-pull-v3
        obj_init_pos_ = split_first();
        SetModelBodyPos("box", obj_init_pos_);
        SetObjScalar(-0.1);
        target_pos_ = SitePos("goalPull");
        max_dist_ = std::abs((task_index_ == 26 ? SiteModelPos("handleStart")
                                                : SitePos("handleStart"))[2] -
                             target_pos_[2]);
        target_reward_ = 1000.0 * max_dist_ + 2000.0;
        if (task_index_ == 25) {
          obj_init_pos_ = ObjPos();
        }
        break;
      }
      case 27: {  // lever-pull-v3
        obj_init_pos_ = split_first();
        SetModelBodyPos("lever", obj_init_pos_);
        lever_pos_init_ = Add(obj_init_pos_, {0.12, -0.2, 0.25});
        target_pos_ = Add(obj_init_pos_, {0.12, 0.0, 0.45});
        SetModelSitePos("goal", target_pos_);
        max_pull_dist_ = Distance(target_pos_, obj_init_pos_);
        break;
      }
      case 28:    // pick-place-wall-v3
      case 30:    // pick-place-v3
      case 40:    // push-v3
      case 41:    // push-wall-v3
      case 42:    // push-back-v3
      case 43:    // reach-v3
      case 44: {  // reach-wall-v3
        mjtNum base_z =
            (task_index_ == 41 || task_index_ == 42 || task_index_ == 28)
                ? GeomPos("objGeom")[2]
                : BodyPos("obj")[2];
        while (Distance2(split_first(), split_second()) < 0.15) {
          rand_vec = SampleRandVec();
        }
        if (task_index_ == 40 || task_index_ == 41 || task_index_ == 42) {
          target_pos_ = {rand_vec[3], rand_vec[4], base_z};
          obj_init_pos_ = {rand_vec[0], rand_vec[1], base_z};
        } else {
          target_pos_ = split_second();
          obj_init_pos_ = {rand_vec[0], rand_vec[1], rand_vec[2]};
        }
        SetObjXYZ(obj_init_pos_);
        SetModelSitePos("goal", target_pos_);
        lift_thresh_ = 0.04;
        obj_height_ = GeomPos("objGeom")[2];
        height_target_ = obj_height_ + lift_thresh_;
        max_reach_dist_ = Distance(init_tcp_, target_pos_);
        max_push_dist_ = Distance2(obj_init_pos_, target_pos_);
        max_placing_dist_ =
            Distance({obj_init_pos_[0], obj_init_pos_[1], height_target_},
                     target_pos_) +
            height_target_;
        break;
      }
      case 29: {  // pick-out-of-hole-v3
        while (Distance2(split_first(), split_second()) < 0.15) {
          rand_vec = SampleRandVec();
        }
        obj_init_pos_ = split_first();
        target_pos_ = split_second();
        SetObjXYZ(obj_init_pos_);
        SetModelSitePos("goal", target_pos_);
        lift_thresh_ = 0.11;
        obj_height_ = GeomPos("objGeom")[2];
        height_target_ = obj_height_ + lift_thresh_;
        max_placing_dist_ =
            Distance({obj_init_pos_[0], obj_init_pos_[1], height_target_},
                     target_pos_) +
            height_target_;
        break;
      }
      case 31: {  // plate-slide-v3
        obj_init_pos_ = split_first();
        target_pos_ = split_second();
        SetModelBodyPos("puck_goal", target_pos_);
        SetObjQpos2(0.0, 0.0);
        SetModelSitePos("goal", target_pos_);
        max_dist_ = Distance2(obj_init_pos_, target_pos_);
        break;
      }
      case 32: {  // plate-slide-side-v3
        obj_init_pos_ = split_first();
        target_pos_ = split_second();
        SetObjQpos2(0.0, 0.0);
        SetModelSitePos("goal", target_pos_);
        max_dist_ = Distance2(obj_init_pos_, target_pos_);
        break;
      }
      case 33: {  // plate-slide-back-v3
        obj_init_pos_ = split_first();
        target_pos_ = split_second();
        SetObjQpos2(0.0, 0.15);
        SetModelSitePos("goal", target_pos_);
        max_dist_ = Distance2(GeomPos("puck"), target_pos_);
        break;
      }
      case 34: {  // plate-slide-back-side-v3
        obj_init_pos_ = split_first();
        target_pos_ = split_second();
        SetModelBodyPos("puck_goal", obj_init_pos_);
        SetObjQpos2(-0.15, 0.0);
        SetModelSitePos("goal", target_pos_);
        max_dist_ = Distance2(GeomPos("puck"), target_pos_);
        break;
      }
      case 35: {  // peg-insert-side-v3
        while (Distance2(split_first(), split_second()) < 0.1) {
          rand_vec = SampleRandVec();
        }
        obj_init_pos_ = split_first();
        auto pos_box = split_second();
        peg_head_pos_init_ = SitePos("pegHead");
        SetObjXYZ(obj_init_pos_);
        SetModelBodyPos("box", pos_box);
        target_pos_ = Add(pos_box, {0.03, 0.0, 0.13});
        SetModelSitePos("goal", target_pos_);
        lift_thresh_ = 0.11;
        obj_height_ = BodyPos("peg")[2];
        height_target_ = obj_height_ + lift_thresh_;
        max_placing_dist_ =
            Distance({obj_init_pos_[0], obj_init_pos_[1], height_target_},
                     target_pos_) +
            height_target_;
        break;
      }
      case 36: {  // peg-unplug-side-v3
        auto pos_box = split_first();
        SetModelBodyPos("box", pos_box);
        auto pos_plug = Add(pos_box, {0.044, 0.0, 0.131});
        SetPegUnplugObjXYZ(pos_plug);
        obj_init_pos_ = SitePos("pegEnd");
        target_pos_ = Add(pos_plug, {0.15, 0.0, 0.0});
        SetModelSitePos("goal", target_pos_);
        max_placing_dist_ = Distance(target_pos_, obj_init_pos_);
        break;
      }
      case 37: {  // soccer-v3
        while (Distance2(split_first(), split_second()) < 0.15) {
          rand_vec = SampleRandVec();
        }
        target_pos_ = split_second();
        obj_init_pos_ = {rand_vec[0], rand_vec[1], task_.init_obj[2]};
        SetModelBodyPos("goal_whole", target_pos_);
        SetObjXYZ(obj_init_pos_);
        max_push_dist_ = Distance2(obj_init_pos_, target_pos_);
        SetModelSitePos("goal", target_pos_);
        break;
      }
      case 38: {  // stick-push-v3
        stick_init_pos_ = {-0.1, 0.6, 0.02};
        while (Distance2(split_first(), split_second()) < 0.1) {
          rand_vec = SampleRandVec();
        }
        stick_init_pos_ = {rand_vec[0], rand_vec[1], stick_init_pos_[2]};
        target_pos_ = {rand_vec[3], rand_vec[4], SitePos("insertion")[2]};
        SetStickXYZ(stick_init_pos_);
        SetStickObjectQpos(0.0, 0.0);
        obj_init_pos_ = BodyPos("object");
        SetModelSitePos("goal", target_pos_);
        lift_thresh_ = 0.04;
        obj_height_ = BodyPos("stick")[2];
        height_target_ = obj_height_ + lift_thresh_;
        max_place_dist_ =
            Distance({obj_init_pos_[0], obj_init_pos_[1], height_target_},
                     stick_init_pos_) +
            height_target_;
        max_push_dist_ = Distance2(obj_init_pos_, target_pos_);
        break;
      }
      case 39: {  // stick-pull-v3
        stick_init_pos_ = {0.0, 0.6, 0.02};
        while (Distance2(split_first(), split_second()) < 0.1) {
          rand_vec = SampleRandVec();
        }
        stick_init_pos_ = {rand_vec[0], rand_vec[1], stick_init_pos_[2]};
        target_pos_ = {rand_vec[3], rand_vec[4], stick_init_pos_[2]};
        SetStickXYZ(stick_init_pos_);
        SetStickObjectQpos(0.0, 0.09);
        obj_init_pos_ = BodyPos("object");
        SetModelSitePos("goal", target_pos_);
        lift_thresh_ = 0.04;
        obj_height_ = BodyPos("stick")[2];
        height_target_ = obj_height_ + lift_thresh_;
        max_pull_dist_ = Distance2(obj_init_pos_, target_pos_);
        max_place_dist_ =
            Distance({obj_init_pos_[0], obj_init_pos_[1], height_target_},
                     stick_init_pos_) +
            height_target_;
        break;
      }
      case 45: {  // shelf-place-v3
        while (Distance2(split_first(), split_second()) < 0.1) {
          rand_vec = SampleRandVec();
        }
        auto base_shelf_pos =
            std::array<mjtNum, 3>{rand_vec[3], rand_vec[4], rand_vec[5] - 0.3};
        obj_init_pos_ = {rand_vec[0], rand_vec[1], BodyPos("obj")[2]};
        SetModelBodyPos("shelf", base_shelf_pos);
        mj_forward(model_, data_);
        target_pos_ = Add(SiteModelPos("goal"), BodyModelPos("shelf"));
        SetObjXYZ(obj_init_pos_);
        SetModelSitePos("goal", target_pos_);
        lift_thresh_ = 0.04;
        obj_height_ = GeomPos("objGeom")[2];
        height_target_ = obj_height_ + lift_thresh_;
        max_placing_dist_ =
            Distance({obj_init_pos_[0], obj_init_pos_[1], height_target_},
                     target_pos_) +
            height_target_;
        break;
      }
      case 46: {  // sweep-into-v3
        target_pos_ = ToMjt(task_.init_goal);
        while (Distance2(split_first(), target_pos_) < 0.15) {
          rand_vec = SampleRandVec();
        }
        obj_init_pos_ = {rand_vec[0], rand_vec[1], BodyPos("obj")[2]};
        SetObjXYZ(obj_init_pos_);
        SetModelSitePos("goal", target_pos_);
        max_push_dist_ = Distance2(obj_init_pos_, target_pos_);
        break;
      }
      case 47: {  // sweep-v3
        obj_init_pos_ = {rand_vec[0], rand_vec[1], task_.init_obj[2]};
        target_pos_ = ToMjt(task_.init_goal);
        target_pos_[1] = rand_vec[1];
        SetObjXYZ(obj_init_pos_);
        SetModelSitePos("goal", target_pos_);
        lift_thresh_ = 0.04;
        obj_height_ = GeomPos("objGeom")[2];
        height_target_ = obj_height_ + lift_thresh_;
        max_reach_dist_ = Distance(init_tcp_, target_pos_);
        max_push_dist_ = Distance2(obj_init_pos_, target_pos_);
        max_placing_dist_ =
            Distance({obj_init_pos_[0], obj_init_pos_[1], height_target_},
                     target_pos_) +
            height_target_;
        break;
      }
      case 48: {  // window-open-v3
        obj_init_pos_ = split_first();
        target_pos_ = Add(obj_init_pos_, {0.2, 0.0, 0.0});
        SetModelBodyPos("window", obj_init_pos_);
        mj_forward(model_, data_);
        window_handle_pos_init_ = ObjPos();
        SetJointQpos("window_slide", 0.0);
        SetModelSitePos("goal", target_pos_);
        break;
      }
      case 49: {  // window-close-v3
        obj_init_pos_ = split_first();
        target_pos_ = obj_init_pos_;
        SetModelBodyPos("window", obj_init_pos_);
        mj_forward(model_, data_);
        window_handle_pos_init_ = Add(ObjPos(), {0.2, 0.0, 0.0});
        SetJointQpos("window_slide", 0.2);
        SetModelSitePos("goal", target_pos_);
        lift_thresh_ = 0.02;
        break;
      }
      default:
        if (task_.random_dim >= 3) {
          obj_init_pos_ = split_first();
        }
        if (task_.random_dim >= 6) {
          target_pos_ = split_second();
        }
        SetObjXYZ(obj_init_pos_);
        SetModelSitePos("goal", target_pos_);
        break;
    }
    PositionTargetSites();
    mj_forward(model_, data_);
  }

  void PositionTargetSites() {
    switch (task_index_) {
      case 0:
      case 12:
        SetModelSitePos("pegTop", target_pos_);
        break;
      case 8:
        SetModelSitePos("coffee_goal", target_pos_);
        break;
      case 9:
      case 10:
        SetModelSitePos("mug_goal", target_pos_);
        break;
      case 20:
        SetModelSitePos("goal_open", target_pos_);
        break;
      case 21:
        SetModelSitePos("goal_close", target_pos_);
        break;
      default:
        SetModelSitePos("goal", target_pos_);
        break;
    }
  }

  mjtNum JointQpos(const char* name) const {
    int id = mj_name2id(model_, mjOBJ_JOINT, name);
    if (id < 0) {
      return 0.0;
    }
    int adr = model_->jnt_qposadr[id];
    return 0 <= adr && adr < model_->nq ? data_->qpos[adr] : 0.0;
  }

  int MainObjectGeomId() const {
    switch (task_index_) {
      case 0:
      case 12:
        return mj_name2id(model_, mjOBJ_GEOM, "WrenchHandle");
      case 3:
        return mj_name2id(model_, mjOBJ_GEOM, "BoxHandleGeom");
      case 4:
      case 5:
      case 6:
      case 7:
        return mj_name2id(model_, mjOBJ_GEOM, "btnGeom");
      case 9:
      case 10:
        return mj_name2id(model_, mjOBJ_GEOM, "mug");
      case 22:
        return mj_name2id(model_, mjOBJ_GEOM, "HammerHandle");
      default:
        return mj_name2id(model_, mjOBJ_GEOM, "objGeom");
    }
  }

  bool TouchingMainObject() const {
    int object_geom_id = MainObjectGeomId();
    int leftpad_geom_id = mj_name2id(model_, mjOBJ_GEOM, "leftpad_geom");
    int rightpad_geom_id = mj_name2id(model_, mjOBJ_GEOM, "rightpad_geom");
    if (object_geom_id < 0 || leftpad_geom_id < 0 || rightpad_geom_id < 0) {
      return false;
    }
    mjtNum left_force = 0.0;
    mjtNum right_force = 0.0;
    for (int i = 0; i < data_->ncon; ++i) {
      const mjContact& contact = data_->contact[i];
      if (contact.efc_address < 0) {
        continue;
      }
      bool object_contact =
          contact.geom1 == object_geom_id || contact.geom2 == object_geom_id;
      if (!object_contact) {
        continue;
      }
      mjtNum force = data_->efc_force[contact.efc_address];
      if (contact.geom1 == leftpad_geom_id ||
          contact.geom2 == leftpad_geom_id) {
        left_force += force;
      }
      if (contact.geom1 == rightpad_geom_id ||
          contact.geom2 == rightpad_geom_id) {
        right_force += force;
      }
    }
    return left_force > 0.0 && right_force > 0.0;
  }

  RewardInfo MakeInfo(mjtNum reward, mjtNum success, mjtNum near_object,
                      mjtNum grasp_success, mjtNum grasp_reward,
                      mjtNum in_place_reward, mjtNum obj_to_target) const {
    RewardInfo info;
    info.reward = reward;
    info.success = success;
    info.near_object = near_object;
    info.grasp_success = grasp_success;
    info.grasp_reward = grasp_reward;
    info.in_place_reward = in_place_reward;
    info.obj_to_target = obj_to_target;
    return info;
  }

  mjtNum GripperCagingReward(
      const std::array<mjtNum, 4>& action, const std::array<mjtNum, 3>& obj_pos,
      mjtNum obj_radius, mjtNum pad_success_thresh, mjtNum object_reach_radius,
      mjtNum xz_thresh, mjtNum desired_gripper_effort = 1.0,
      bool high_density = false, bool medium_density = false,
      const std::array<mjtNum, 3>* init_obj = nullptr, bool signed_y = false,
      mjtNum grip_success_extra = -1.0, mjtNum caging_threshold = 0.97) const {
    const auto& initial_obj = init_obj == nullptr ? obj_init_pos_ : *init_obj;
    auto left_pad = BodyPos("leftpad");
    auto right_pad = BodyPos("rightpad");

    mjtNum left_delta = signed_y ? left_pad[1] - obj_pos[1]
                                 : std::abs(left_pad[1] - obj_pos[1]);
    mjtNum right_delta = signed_y ? obj_pos[1] - right_pad[1]
                                  : std::abs(right_pad[1] - obj_pos[1]);
    mjtNum left_init = std::abs(left_pad[1] - initial_obj[1]);
    mjtNum right_init = std::abs(right_pad[1] - initial_obj[1]);
    mjtNum left_margin = std::abs(left_init - pad_success_thresh);
    mjtNum right_margin = std::abs(right_init - pad_success_thresh);

    mjtNum left_caging = LongTailTolerance(left_delta, obj_radius,
                                           pad_success_thresh, left_margin);
    mjtNum right_caging = LongTailTolerance(right_delta, obj_radius,
                                            pad_success_thresh, right_margin);
    mjtNum y_caging = HamacherProduct(left_caging, right_caging);

    auto tcp = TcpCenter();
    auto tcp_xz = std::array<mjtNum, 3>{tcp[0], 0.0, tcp[2]};
    auto obj_xz = std::array<mjtNum, 3>{obj_pos[0], 0.0, obj_pos[2]};
    auto init_obj_xz =
        std::array<mjtNum, 3>{initial_obj[0], 0.0, initial_obj[2]};
    auto init_tcp_xz = std::array<mjtNum, 3>{init_tcp_[0], 0.0, init_tcp_[2]};
    mjtNum xz_margin = Distance(init_obj_xz, init_tcp_xz) - xz_thresh;
    mjtNum xz_caging =
        LongTailTolerance(Distance(tcp_xz, obj_xz), 0.0, xz_thresh, xz_margin);

    mjtNum caging = HamacherProduct(y_caging, xz_caging);
    mjtNum gripper_closed =
        std::clamp<mjtNum>(action[3], 0.0, desired_gripper_effort) /
        desired_gripper_effort;

    mjtNum gripping = 0.0;
    if (grip_success_extra >= 0.0) {
      mjtNum grip_success_margin = obj_radius + grip_success_extra;
      mjtNum left_gripping = LongTailTolerance(
          left_delta, obj_radius, grip_success_margin, left_margin);
      mjtNum right_gripping = LongTailTolerance(
          right_delta, obj_radius, grip_success_margin, right_margin);
      mjtNum y_gripping = HamacherProduct(left_gripping, right_gripping);
      gripping = caging > caging_threshold ? y_gripping : 0.0;
    } else {
      gripping = caging > caging_threshold ? gripper_closed : 0.0;
    }

    mjtNum caging_and_gripping = grip_success_extra >= 0.0
                                     ? (caging + gripping) / 2.0
                                     : HamacherProduct(caging, gripping);
    if (high_density) {
      caging_and_gripping = (caging_and_gripping + caging) / 2.0;
    }
    if (medium_density) {
      mjtNum tcp_to_obj = Distance(obj_pos, tcp);
      mjtNum tcp_to_obj_init = Distance(initial_obj, init_tcp_);
      mjtNum reach_margin = std::abs(tcp_to_obj_init - object_reach_radius);
      mjtNum reach =
          LongTailTolerance(tcp_to_obj, 0.0, object_reach_radius, reach_margin);
      caging_and_gripping = (caging_and_gripping + reach) / 2.0;
    }
    return caging_and_gripping;
  }

  mjtNum PickPlaceCagingReward(const std::array<mjtNum, 4>& action,
                               const std::array<mjtNum, 3>& obj_pos) const {
    mjtNum pad_success_margin = 0.05;
    mjtNum xz_success_margin = 0.005;
    mjtNum obj_radius = 0.015;
    auto tcp = TcpCenter();
    auto left_pad = BodyPos("leftpad");
    auto right_pad = BodyPos("rightpad");
    mjtNum left_delta = left_pad[1] - obj_pos[1];
    mjtNum right_delta = obj_pos[1] - right_pad[1];
    mjtNum right_margin = std::abs(std::abs(obj_pos[1] - init_right_pad_[1]) -
                                   pad_success_margin);
    mjtNum left_margin =
        std::abs(std::abs(obj_pos[1] - init_left_pad_[1]) - pad_success_margin);
    mjtNum right_caging = LongTailTolerance(right_delta, obj_radius,
                                            pad_success_margin, right_margin);
    mjtNum left_caging = LongTailTolerance(left_delta, obj_radius,
                                           pad_success_margin, left_margin);
    mjtNum y_caging = HamacherProduct(left_caging, right_caging);
    auto tcp_xz = std::array<mjtNum, 3>{tcp[0], 0.0, tcp[2]};
    auto obj_xz = std::array<mjtNum, 3>{obj_pos[0], 0.0, obj_pos[2]};
    auto init_obj_xz =
        std::array<mjtNum, 3>{obj_init_pos_[0], 0.0, obj_init_pos_[2]};
    auto init_tcp_xz = std::array<mjtNum, 3>{init_tcp_[0], 0.0, init_tcp_[2]};
    mjtNum xz_margin = Distance(init_obj_xz, init_tcp_xz) - xz_success_margin;
    mjtNum xz_caging = LongTailTolerance(Distance(tcp_xz, obj_xz), 0.0,
                                         xz_success_margin, xz_margin);
    mjtNum caging = HamacherProduct(y_caging, xz_caging);
    mjtNum gripper_closed = std::clamp<mjtNum>(action[3], 0.0, 1.0);
    mjtNum gripping = caging > 0.97 ? gripper_closed : 0.0;
    mjtNum caging_and_gripping = HamacherProduct(caging, gripping);
    return (caging_and_gripping + caging) / 2.0;
  }

  mjtNum SweepStyleCagingReward(const std::array<mjtNum, 4>& action,
                                const std::array<mjtNum, 3>& obj_pos,
                                mjtNum obj_radius, mjtNum grip_success_extra,
                                mjtNum xz_thresh) const {
    mjtNum pad_success_margin = 0.05;
    mjtNum grip_success_margin = obj_radius + grip_success_extra;
    auto tcp = TcpCenter();
    auto left_pad = BodyPos("leftpad");
    auto right_pad = BodyPos("rightpad");
    mjtNum left_delta = left_pad[1] - obj_pos[1];
    mjtNum right_delta = obj_pos[1] - right_pad[1];
    mjtNum right_margin = std::abs(std::abs(obj_pos[1] - init_right_pad_[1]) -
                                   pad_success_margin);
    mjtNum left_margin =
        std::abs(std::abs(obj_pos[1] - init_left_pad_[1]) - pad_success_margin);
    mjtNum right_caging = LongTailTolerance(right_delta, obj_radius,
                                            pad_success_margin, right_margin);
    mjtNum left_caging = LongTailTolerance(left_delta, obj_radius,
                                           pad_success_margin, left_margin);
    mjtNum right_gripping = LongTailTolerance(
        right_delta, obj_radius, grip_success_margin, right_margin);
    mjtNum left_gripping = LongTailTolerance(left_delta, obj_radius,
                                             grip_success_margin, left_margin);
    mjtNum y_caging = HamacherProduct(right_caging, left_caging);
    mjtNum y_gripping = HamacherProduct(right_gripping, left_gripping);
    auto tcp_xz = std::array<mjtNum, 3>{tcp[0], 0.0, tcp[2]};
    auto obj_xz = std::array<mjtNum, 3>{obj_pos[0], 0.0, obj_pos[2]};
    auto init_obj_xz =
        std::array<mjtNum, 3>{obj_init_pos_[0], 0.0, obj_init_pos_[2]};
    auto init_tcp_xz = std::array<mjtNum, 3>{init_tcp_[0], 0.0, init_tcp_[2]};
    mjtNum xz_margin = Distance(init_obj_xz, init_tcp_xz) - xz_thresh;
    mjtNum xz_caging =
        LongTailTolerance(Distance(tcp_xz, obj_xz), 0.0, xz_thresh, xz_margin);
    mjtNum caging = HamacherProduct(y_caging, xz_caging);
    mjtNum gripping = caging > 0.95 ? y_gripping : 0.0;
    return (caging + gripping) / 2.0;
  }

  mjtNum StickCagingReward(const std::array<mjtNum, 4>& action,
                           const std::array<mjtNum, 3>& obj_pos,
                           mjtNum obj_radius, mjtNum pad_success_thresh,
                           mjtNum object_reach_radius, mjtNum xz_thresh,
                           bool high_density = false,
                           bool medium_density = false,
                           bool use_stick_init = false) const {
    const std::array<mjtNum, 3>* initial_obj =
        use_stick_init ? &stick_init_pos_ : nullptr;
    return GripperCagingReward(action, obj_pos, obj_radius, pad_success_thresh,
                               object_reach_radius, xz_thresh, 1.0,
                               high_density, medium_density, initial_obj);
  }

  RewardInfo ComputeReward(const std::array<mjtNum, 4>& action) {
    auto obs = CurrentObsNoGoal();
    auto hand = std::array<mjtNum, 3>{obs[0], obs[1], obs[2]};
    auto tcp = TcpCenter();
    auto obj = std::array<mjtNum, 3>{obs[4], obs[5], obs[6]};
    mjtNum tcp_open = obs[3];

    auto default_six_info = [&](mjtNum reward, mjtNum tcp_to_obj,
                                mjtNum tcp_opened, mjtNum obj_to_target,
                                mjtNum grasp_reward, mjtNum in_place_reward,
                                mjtNum success_threshold,
                                mjtNum near_threshold = 0.03,
                                bool require_lift_touch = false,
                                mjtNum lift_delta = 0.02) {
      mjtNum grasp_success = tcp_opened > 0.0 ? 1.0 : 0.0;
      if (require_lift_touch) {
        grasp_success = TouchingMainObject() && tcp_opened > 0.0 &&
                                obj[2] - lift_delta > obj_init_pos_[2]
                            ? 1.0
                            : 0.0;
      }
      return MakeInfo(reward, obj_to_target <= success_threshold ? 1.0 : 0.0,
                      tcp_to_obj <= near_threshold ? 1.0 : 0.0, grasp_success,
                      grasp_reward, in_place_reward, obj_to_target);
    };

    auto nut_reward_pos = [&](const std::array<mjtNum, 3>& wrench_center,
                              bool* success) {
      auto pos_error = Sub(target_pos_, wrench_center);
      mjtNum radius =
          std::sqrt(pos_error[0] * pos_error[0] + pos_error[1] * pos_error[1]);
      bool aligned = radius < 0.02;
      bool hooked = pos_error[2] > 0.0;
      *success = aligned && hooked;
      mjtNum threshold = *success ? 0.02 : 0.01;
      mjtNum target_height = 0.0;
      if (radius > threshold) {
        target_height = 0.02 * std::log(radius - threshold) + 0.2;
      }
      pos_error[2] = target_height - wrench_center[2];
      bool lifted = wrench_center[2] > 0.02 || radius < threshold;
      return 0.1 * static_cast<mjtNum>(lifted) +
             0.9 * LongTailTolerance(Norm(Scale(pos_error, {1.0, 1.0, 3.0})),
                                     0.0, 0.02, 0.4);
    };

    switch (task_index_) {
      case 0: {  // assembly-v3
        auto wrench = obj;
        auto wrench_center = SitePos("RoundNut");
        if (std::abs(wrench[0] - hand[0]) < 0.01) {
          wrench[0] = hand[0];
        }
        mjtNum quat_error =
            Norm4({obs[7] - 0.707, obs[8], obs[9], obs[10] - 0.707});
        mjtNum reward_quat = std::max<mjtNum>(1.0 - quat_error / 0.4, 0.0);
        mjtNum reward_grab = GripperCagingReward(action, wrench, 0.015, 0.02,
                                                 0.01, 0.01, 1.0, false, true);
        bool success = false;
        mjtNum in_place = nut_reward_pos(wrench_center, &success);
        mjtNum reward = (2.0 * reward_grab + 6.0 * in_place) * reward_quat;
        if (success) {
          reward = 10.0;
        }
        return MakeInfo(reward, success ? 1.0 : 0.0, reward_quat,
                        reward_grab >= 0.5 ? 1.0 : 0.0, reward_grab, in_place,
                        0.0);
      }
      case 1: {  // basketball-v3
        auto target = target_pos_;
        target[2] = 0.3;
        mjtNum target_to_obj = DistanceScaled(obj, target, {1.0, 1.0, 2.0});
        mjtNum target_to_obj_init =
            DistanceScaled(obj_init_pos_, target, {1.0, 1.0, 2.0});
        mjtNum in_place = LongTailTolerance(
            target_to_obj, 0.0, task_.target_radius, target_to_obj_init);
        mjtNum tcp_to_obj = Distance(obj, tcp);
        mjtNum object_grasped = GripperCagingReward(
            action, obj, 0.025, 0.06, 0.01, 0.005, 1.0, true, false);
        if (tcp_to_obj < 0.035 && tcp_open > 0.0 &&
            obj[2] - 0.01 > obj_init_pos_[2]) {
          object_grasped = 1.0;
        }
        mjtNum reward = HamacherProduct(object_grasped, in_place);
        if (tcp_to_obj < 0.035 && tcp_open > 0.0 &&
            obj[2] - 0.01 > obj_init_pos_[2]) {
          reward += 1.0 + 5.0 * in_place;
        }
        if (target_to_obj < task_.target_radius) {
          reward = 10.0;
        }
        return MakeInfo(
            reward, target_to_obj <= task_.target_radius ? 1.0 : 0.0,
            tcp_to_obj <= 0.05 ? 1.0 : 0.0,
            (tcp_open > 0.0 && obj[2] - 0.03 > obj_init_pos_[2]) ? 1.0 : 0.0,
            object_grasped, in_place, target_to_obj);
      }
      case 2: {  // bin-picking-v3
        mjtNum obj_to_target = Distance(obj, target_pos_);
        mjtNum init = Distance(obj_init_pos_, target_pos_);
        mjtNum in_place =
            LongTailTolerance(obj_to_target, 0.0, task_.target_radius, init);
        mjtNum threshold = 0.03;
        mjtNum radius0 = Distance2({hand[0], hand[1], 0.0},
                                   {obj_init_pos_[0], obj_init_pos_[1], 0.0});
        mjtNum radius1 = Distance2({hand[0], hand[1], 0.0},
                                   {target_pos_[0], target_pos_[1], 0.0});
        auto floor_for_radius = [&](mjtNum radius) {
          return radius > threshold ? 0.02 * std::log(radius - threshold) + 0.2
                                    : 0.0;
        };
        mjtNum floor =
            std::min(floor_for_radius(radius0), floor_for_radius(radius1));
        mjtNum above_floor =
            hand[2] >= floor
                ? 1.0
                : LongTailTolerance(std::max<mjtNum>(floor - hand[2], 0.0), 0.0,
                                    0.01, 0.05);
        mjtNum object_grasped = GripperCagingReward(
            action, obj, 0.015, 0.05, 0.01, 0.01, 0.7, true, false);
        mjtNum reward = HamacherProduct(object_grasped, in_place);
        bool near_object = Distance(obj, hand) < 0.04;
        bool pinched_without_obj = tcp_open < 0.43;
        bool lifted = obj[2] - 0.02 > obj_init_pos_[2];
        bool grasp_success = near_object && lifted && !pinched_without_obj;
        if (grasp_success) {
          reward += 1.0 + 5.0 * HamacherProduct(above_floor, in_place);
        }
        if (obj_to_target < task_.target_radius) {
          reward = 10.0;
        }
        return MakeInfo(reward, obj_to_target <= 0.05 ? 1.0 : 0.0,
                        near_object ? 1.0 : 0.0, grasp_success ? 1.0 : 0.0,
                        object_grasped, in_place, obj_to_target);
      }
      case 3: {  // box-close-v3
        mjtNum reward_grab = std::clamp<mjtNum>(
            (std::clamp<mjtNum>(action[3], -1.0, 1.0) + 1.0) / 2.0, 0.0, 1.0);
        mjtNum quat_error =
            Norm4({obs[7] - 0.707, obs[8], obs[9], obs[10] - 0.707});
        mjtNum reward_quat = std::max<mjtNum>(1.0 - quat_error / 0.2, 0.0);
        auto lid = Add(obj, {0.0, 0.0, 0.02});
        mjtNum radius = std::sqrt((hand[0] - lid[0]) * (hand[0] - lid[0]) +
                                  (hand[1] - lid[1]) * (hand[1] - lid[1]));
        mjtNum floor =
            radius <= 0.02 ? 0.0 : 0.04 * std::log(radius - 0.02) + 0.4;
        mjtNum above_floor =
            hand[2] >= floor
                ? 1.0
                : LongTailTolerance(floor - hand[2], 0.0, 0.01, floor / 2.0);
        mjtNum in_place =
            LongTailTolerance(Distance(hand, lid), 0.0, 0.02, 0.5);
        mjtNum ready_to_lift = HamacherProduct(above_floor, in_place);
        mjtNum pos_error = Norm(Scale(Sub(target_pos_, lid), {1.0, 1.0, 3.0}));
        mjtNum lifted = 0.2 * static_cast<mjtNum>(lid[2] > 0.04) +
                        0.8 * LongTailTolerance(pos_error, 0.0, 0.05, 0.25);
        mjtNum reward =
            2.0 * HamacherProduct(reward_grab, ready_to_lift) + 8.0 * lifted;
        bool success = Distance(obj, target_pos_) < 0.08;
        if (success) {
          reward = 10.0;
        }
        reward *= reward_quat;
        return MakeInfo(reward, success ? 1.0 : 0.0, ready_to_lift,
                        reward_grab >= 0.5 ? 1.0 : 0.0, reward_grab, lifted,
                        0.0);
      }
      case 4:
      case 5: {  // topdown buttons
        mjtNum tcp_to_obj = Distance(obj, tcp);
        mjtNum tcp_to_obj_init = Distance(obj, init_tcp_);
        mjtNum obj_to_target = std::abs(target_pos_[2] - obj[2]);
        mjtNum tcp_closed = 1.0 - tcp_open;
        mjtNum near_button =
            LongTailTolerance(tcp_to_obj, 0.0, 0.01, tcp_to_obj_init);
        mjtNum pressed =
            LongTailTolerance(obj_to_target, 0.0, 0.005, obj_to_target_init_);
        mjtNum reward = 5.0 * HamacherProduct(tcp_closed, near_button);
        if (tcp_to_obj <= 0.03) {
          reward += 5.0 * pressed;
        }
        return MakeInfo(reward, obj_to_target <= 0.024 ? 1.0 : 0.0,
                        tcp_to_obj <= 0.05 ? 1.0 : 0.0,
                        tcp_open > 0.0 ? 1.0 : 0.0, near_button, pressed,
                        obj_to_target);
      }
      case 6: {  // button-press-v3
        mjtNum tcp_to_obj = Distance(obj, tcp);
        mjtNum near_button =
            LongTailTolerance(tcp_to_obj, 0.0, 0.05, Distance(obj, init_tcp_));
        mjtNum obj_to_target = std::abs(target_pos_[1] - obj[1]);
        mjtNum pressed =
            LongTailTolerance(obj_to_target, 0.0, 0.005, obj_to_target_init_);
        mjtNum tcp_closed = std::max<mjtNum>(tcp_open, 0.0);
        mjtNum reward = 2.0 * HamacherProduct(tcp_closed, near_button);
        if (tcp_to_obj <= 0.05) {
          reward += 8.0 * pressed;
        }
        return MakeInfo(reward, obj_to_target <= 0.02 ? 1.0 : 0.0,
                        tcp_to_obj <= 0.05 ? 1.0 : 0.0,
                        tcp_open > 0.0 ? 1.0 : 0.0, near_button, pressed,
                        obj_to_target);
      }
      case 7: {  // button-press-wall-v3
        mjtNum tcp_to_obj = Distance(obj, tcp);
        mjtNum near_button =
            LongTailTolerance(tcp_to_obj, 0.0, 0.01, Distance(obj, init_tcp_));
        mjtNum obj_to_target = std::abs(target_pos_[1] - obj[1]);
        mjtNum pressed =
            LongTailTolerance(obj_to_target, 0.0, 0.005, obj_to_target_init_);
        mjtNum reward = 0.0;
        if (tcp_to_obj > 0.07) {
          reward = 2.0 * HamacherProduct((1.0 - tcp_open) / 2.0, near_button);
        } else {
          reward = 2.0 + 2.0 * (1.0 + tcp_open) + 4.0 * pressed * pressed;
        }
        return MakeInfo(reward, obj_to_target <= 0.03 ? 1.0 : 0.0,
                        tcp_to_obj <= 0.05 ? 1.0 : 0.0,
                        tcp_open > 0.0 ? 1.0 : 0.0, near_button, pressed,
                        obj_to_target);
      }
      case 8: {  // coffee-button-v3
        mjtNum tcp_to_obj = Distance(obj, tcp);
        mjtNum near_button =
            LongTailTolerance(tcp_to_obj, 0.0, 0.05, Distance(obj, init_tcp_));
        mjtNum obj_to_target = std::abs(target_pos_[1] - obj[1]);
        mjtNum pressed = LongTailTolerance(obj_to_target, 0.0, 0.005, 0.03);
        mjtNum reward =
            2.0 * HamacherProduct(std::max<mjtNum>(tcp_open, 0.0), near_button);
        if (tcp_to_obj <= 0.05) {
          reward += 8.0 * pressed;
        }
        return MakeInfo(reward, obj_to_target <= 0.02 ? 1.0 : 0.0,
                        tcp_to_obj <= 0.05 ? 1.0 : 0.0,
                        tcp_open > 0.0 ? 1.0 : 0.0, near_button, pressed,
                        obj_to_target);
      }
      case 9:
      case 10: {  // coffee pull/push
        mjtNum scaled_to_target =
            DistanceScaled(obj, target_pos_, {2.0, 2.0, 1.0});
        mjtNum scaled_init =
            DistanceScaled(obj_init_pos_, target_pos_, {2.0, 2.0, 1.0});
        mjtNum in_place =
            LongTailTolerance(scaled_to_target, 0.0, 0.05, scaled_init);
        mjtNum tcp_to_obj = Distance(obj, tcp);
        mjtNum object_grasped = GripperCagingReward(
            action, obj, 0.02, 0.05, 0.04, 0.05, 0.7, false, true);
        mjtNum reward = HamacherProduct(object_grasped, in_place);
        if (tcp_to_obj < 0.04 && tcp_open > 0.0) {
          reward += 1.0 + 5.0 * in_place;
        }
        if (scaled_to_target < 0.05) {
          reward = 10.0;
        }
        mjtNum obj_to_target = Distance(obj, target_pos_);
        return MakeInfo(reward, obj_to_target <= 0.07 ? 1.0 : 0.0,
                        tcp_to_obj <= 0.03 ? 1.0 : 0.0,
                        TouchingMainObject() && tcp_open > 0.0 ? 1.0 : 0.0,
                        object_grasped, in_place, obj_to_target);
      }
      case 11: {  // dial-turn-v3
        auto dial_obj = ObjPos();
        auto dial_push = Add(dial_obj, {0.05, 0.02, 0.09});
        mjtNum obj_to_target = Distance(dial_obj, target_pos_);
        mjtNum init_margin = std::abs(Distance(handle_init_pos_, target_pos_) -
                                      task_.target_radius);
        mjtNum in_place = LongTailTolerance(obj_to_target, 0.0,
                                            task_.target_radius, init_margin);
        mjtNum tcp_to_obj = Distance(dial_push, tcp);
        mjtNum reach = GaussianTolerance(
            tcp_to_obj, 0.0, 0.005,
            std::abs(Distance(handle_init_pos_, init_tcp_) - 0.005));
        reach = HamacherProduct(reach, std::clamp<mjtNum>(action[3], 0.0, 1.0));
        mjtNum reward = 10.0 * HamacherProduct(reach, in_place);
        return MakeInfo(reward,
                        obj_to_target <= task_.target_radius ? 1.0 : 0.0,
                        tcp_to_obj <= 0.01 ? 1.0 : 0.0, 1.0, reach, in_place,
                        obj_to_target);
      }
      case 12: {  // disassemble-v3
        auto wrench = obj;
        auto wrench_center = SitePos("RoundNut");
        if (std::abs(wrench[0] - hand[0]) < 0.01) {
          wrench[0] = hand[0];
        }
        mjtNum quat_error =
            Norm4({obs[7] - 0.707, obs[8], obs[9], obs[10] - 0.707});
        mjtNum reward_quat = std::max<mjtNum>(1.0 - quat_error / 0.4, 0.0);
        mjtNum reward_grab = GripperCagingReward(action, wrench, 0.015, 0.02,
                                                 0.01, 0.01, 1.0, true, false);
        auto pos_error = Sub(Add(target_pos_, {0.0, 0.0, 0.1}), wrench_center);
        mjtNum in_place =
            0.1 * static_cast<mjtNum>(wrench_center[2] > 0.02) +
            0.9 * LongTailTolerance(Norm(pos_error), 0.0, 0.02, 0.2);
        mjtNum reward = (2.0 * reward_grab + 6.0 * in_place) * reward_quat;
        bool success = obj[2] > target_pos_[2];
        if (success) {
          reward = 10.0;
        }
        return MakeInfo(reward, success ? 1.0 : 0.0, reward_quat,
                        reward_grab >= 0.5 ? 1.0 : 0.0, reward_grab, in_place,
                        0.0);
      }
      case 13: {  // door-close-v3
        mjtNum obj_to_target = Distance(obj, target_pos_);
        mjtNum in_place = GaussianTolerance(
            obj_to_target, 0.0, 0.05, Distance(obj_init_pos_, target_pos_));
        mjtNum tcp_to_target = Distance(tcp, target_pos_);
        mjtNum hand_margin = Distance(ToMjt(task_.init_hand), obj) + 0.1;
        mjtNum hand_in_place =
            GaussianTolerance(tcp_to_target, 0.0, 0.25 * 0.05, hand_margin);
        mjtNum reward = 3.0 * hand_in_place + 6.0 * in_place;
        if (obj_to_target < 0.05) {
          reward = 10.0;
        }
        return MakeInfo(reward, obj_to_target <= 0.08 ? 1.0 : 0.0, 0.0, 1.0,
                        1.0, hand_in_place, obj_to_target);
      }
      case 14: {  // door-lock-v3
        auto leftpad = BodyPos("leftpad");
        mjtNum tcp_to_obj = DistanceScaled(obj, leftpad, {0.25, 1.0, 0.5});
        mjtNum tcp_to_obj_init =
            DistanceScaled(obj, init_left_pad_, {0.25, 1.0, 0.5});
        mjtNum obj_to_target = std::abs(target_pos_[2] - obj[2]);
        mjtNum near_lock =
            LongTailTolerance(tcp_to_obj, 0.0, 0.01, tcp_to_obj_init);
        mjtNum pressed = LongTailTolerance(obj_to_target, 0.0, 0.005, 0.1);
        mjtNum reward =
            2.0 * HamacherProduct(std::max<mjtNum>(tcp_open, 0.0), near_lock) +
            8.0 * pressed;
        return MakeInfo(reward, obj_to_target <= 0.02 ? 1.0 : 0.0,
                        tcp_to_obj <= 0.05 ? 1.0 : 0.0,
                        tcp_open > 0.0 ? 1.0 : 0.0, near_lock, pressed,
                        obj_to_target);
      }
      case 15: {  // door-open-v3
        mjtNum theta = JointQpos("doorjoint");
        mjtNum reward_grab =
            (std::clamp<mjtNum>(action[3], -1.0, 1.0) + 1.0) / 2.0;
        auto door = Add(obj, {-0.05, 0.0, 0.0});
        mjtNum radius = Distance2(hand, door);
        mjtNum floor =
            radius <= 0.12 ? 0.0 : 0.04 * std::log(radius - 0.12) + 0.4;
        mjtNum above_floor =
            hand[2] >= floor
                ? 1.0
                : LongTailTolerance(floor - hand[2], 0.0, 0.01, floor / 2.0);
        mjtNum in_place = LongTailTolerance(
            Norm(Sub(Sub(hand, door), {0.05, 0.03, -0.01})), 0.0, 0.06, 0.5);
        mjtNum ready_to_open = HamacherProduct(above_floor, in_place);
        mjtNum door_angle = -theta;
        constexpr mjtNum k_pi = 3.14159265358979323846;
        mjtNum opened =
            0.2 * static_cast<mjtNum>(theta < -k_pi / 90.0) +
            0.8 * LongTailTolerance(k_pi / 2.0 + k_pi / 6.0 - door_angle, 0.0,
                                    0.5, k_pi / 3.0);
        mjtNum reward =
            2.0 * HamacherProduct(ready_to_open, reward_grab) + 8.0 * opened;
        bool success = std::abs(obj[0] - target_pos_[0]) <= 0.08;
        if (success) {
          reward = 10.0;
        }
        return MakeInfo(reward, success ? 1.0 : 0.0, ready_to_open,
                        reward_grab >= 0.5 ? 1.0 : 0.0, reward_grab, opened,
                        0.0);
      }
      case 16: {  // door-unlock-v3
        auto shoulder = Add(hand, {0.0, 0.055, 0.07});
        auto init_shoulder = Add(init_tcp_, {0.0, 0.055, 0.07});
        mjtNum shoulder_to_lock =
            DistanceScaled(shoulder, obj, {0.25, 1.0, 0.5});
        mjtNum shoulder_init =
            DistanceScaled(init_shoulder, obj_init_pos_, {0.25, 1.0, 0.5});
        mjtNum ready =
            LongTailTolerance(shoulder_to_lock, 0.0, 0.02, shoulder_init);
        mjtNum obj_to_target = std::abs(target_pos_[0] - obj[0]);
        mjtNum pushed = LongTailTolerance(obj_to_target, 0.0, 0.005, 0.1);
        mjtNum reward = 2.0 * ready + 8.0 * pushed;
        return MakeInfo(reward, obj_to_target <= 0.02 ? 1.0 : 0.0,
                        shoulder_to_lock <= 0.05 ? 1.0 : 0.0,
                        tcp_open > 0.0 ? 1.0 : 0.0, ready, pushed,
                        obj_to_target);
      }
      case 17: {  // hand-insert-v3
        mjtNum obj_to_target = Distance(obj, target_pos_);
        mjtNum in_place =
            LongTailTolerance(obj_to_target, 0.0, task_.target_radius,
                              Distance(obj_init_pos_, target_pos_));
        mjtNum object_grasped = GripperCagingReward(
            action, obj, 0.015, 0.05, 0.01, 0.005, 1.0, true, false);
        mjtNum reward = HamacherProduct(object_grasped, in_place);
        mjtNum tcp_to_obj = Distance(obj, tcp);
        if (tcp_to_obj < 0.02 && tcp_open > 0.0) {
          reward += 1.0 + 7.0 * in_place;
        }
        if (obj_to_target < task_.target_radius) {
          reward = 10.0;
        }
        return MakeInfo(reward, obj_to_target <= 0.05 ? 1.0 : 0.0,
                        tcp_to_obj <= 0.03 ? 1.0 : 0.0,
                        TouchingMainObject() && tcp_open > 0.0 &&
                                obj[2] - 0.02 > obj_init_pos_[2]
                            ? 1.0
                            : 0.0,
                        object_grasped, in_place, obj_to_target);
      }
      case 18: {  // drawer-close-v3
        mjtNum obj_to_target = Distance(obj, target_pos_);
        mjtNum in_place =
            LongTailTolerance(obj_to_target, 0.0, task_.target_radius,
                              std::abs(Distance(obj_init_pos_, target_pos_) -
                                       task_.target_radius));
        mjtNum tcp_to_obj = Distance(obj, tcp);
        mjtNum reach = GaussianTolerance(
            tcp_to_obj, 0.0, 0.005,
            std::abs(Distance(obj_init_pos_, init_tcp_) - 0.005));
        reach = HamacherProduct(reach, std::clamp<mjtNum>(action[3], 0.0, 1.0));
        mjtNum reward = HamacherProduct(reach, in_place);
        if (obj_to_target <= task_.target_radius + 0.015) {
          reward = 1.0;
        }
        reward *= 10.0;
        return MakeInfo(
            reward, obj_to_target <= task_.target_radius + 0.015 ? 1.0 : 0.0,
            tcp_to_obj <= 0.01 ? 1.0 : 0.0, 1.0, reach, in_place,
            obj_to_target);
      }
      case 19: {  // drawer-open-v3
        mjtNum handle_error = Distance(obj, target_pos_);
        mjtNum opening = LongTailTolerance(handle_error, 0.0, 0.02, max_dist_);
        auto handle_init = Add(target_pos_, {0.0, max_dist_, 0.0});
        mjtNum gripper_error = DistanceScaled(obj, hand, {3.0, 3.0, 1.0});
        mjtNum gripper_error_init =
            DistanceScaled(handle_init, init_tcp_, {3.0, 3.0, 1.0});
        mjtNum caging =
            LongTailTolerance(gripper_error, 0.0, 0.01, gripper_error_init);
        mjtNum reward = 5.0 * (caging + opening);
        return MakeInfo(reward, handle_error <= 0.03 ? 1.0 : 0.0,
                        gripper_error <= 0.03 ? 1.0 : 0.0,
                        tcp_open > 0.0 ? 1.0 : 0.0, caging, opening,
                        handle_error);
      }
      case 20:
      case 21: {  // faucet open/close
        auto adjusted_obj = obj;
        if (task_index_ == 20) {
          adjusted_obj = Add(adjusted_obj, {-0.04, 0.0, 0.03});
        }
        mjtNum obj_to_target = Distance(adjusted_obj, target_pos_);
        mjtNum init_margin =
            std::abs(Distance(obj_init_pos_, target_pos_) - 0.07);
        mjtNum in_place =
            LongTailTolerance(obj_to_target, 0.0, 0.07, init_margin);
        mjtNum tcp_to_obj = Distance(adjusted_obj, tcp);
        mjtNum reach = GaussianTolerance(
            tcp_to_obj, 0.0, 0.01,
            std::abs(Distance(obj_init_pos_, init_tcp_) - 0.01));
        mjtNum reward = (2.0 * reach + 3.0 * in_place) * 2.0;
        if (obj_to_target <= 0.07) {
          reward = 10.0;
        }
        return MakeInfo(reward, obj_to_target <= 0.07 ? 1.0 : 0.0,
                        tcp_to_obj <= 0.01 ? 1.0 : 0.0, 1.0, reach, in_place,
                        obj_to_target);
      }
      case 22: {  // hammer-v3
        auto hammer = obj;
        auto hammer_head = Add(hammer, {0.16, 0.06, 0.0});
        auto hammer_threshed = hammer;
        if (std::abs(hammer[0] - hand[0]) < 0.07) {
          hammer_threshed[0] = hand[0];
        }
        mjtNum quat_error = Norm4({obs[7] - 1.0, obs[8], obs[9], obs[10]});
        mjtNum reward_quat = std::max<mjtNum>(1.0 - quat_error / 0.4, 0.0);
        mjtNum reward_grab = GripperCagingReward(
            action, hammer_threshed, 0.015, 0.02, 0.01, 0.01, 1.0, true, false);
        mjtNum in_place =
            0.1 * static_cast<mjtNum>(hammer_head[2] > 0.02) +
            0.9 * LongTailTolerance(Distance(hammer_head, target_pos_), 0.0,
                                    0.02, 0.2);
        mjtNum reward = (2.0 * reward_grab + 6.0 * in_place) * reward_quat;
        bool success = JointQpos("NailSlideJoint") > 0.09;
        if (success && reward > 5.0) {
          reward = 10.0;
        }
        return MakeInfo(reward, success ? 1.0 : 0.0, reward_quat,
                        reward_grab >= 0.5 ? 1.0 : 0.0, reward_grab, in_place,
                        0.0);
      }
      case 23:
      case 24: {  // handle press
        auto handle = SitePos("handleStart");
        mjtNum obj_to_target = std::abs(handle[2] - target_pos_[2]);
        mjtNum in_place = LongTailTolerance(
            obj_to_target, 0.0, task_.target_radius,
            std::abs(std::abs(handle_init_pos_[2] - target_pos_[2]) -
                     task_.target_radius));
        mjtNum tcp_to_obj = Distance(handle, tcp);
        mjtNum reach = LongTailTolerance(
            tcp_to_obj, 0.0, 0.02,
            std::abs(Distance(handle_init_pos_, init_tcp_) - 0.02));
        mjtNum reward = HamacherProduct(reach, in_place);
        if (obj_to_target <= task_.target_radius) {
          reward = 1.0;
        }
        reward *= 10.0;
        return MakeInfo(reward,
                        obj_to_target <= task_.target_radius ? 1.0 : 0.0,
                        tcp_to_obj <= 0.05 ? 1.0 : 0.0, 1.0, reach, in_place,
                        obj_to_target);
      }
      case 25:
      case 26: {  // handle pull
        mjtNum obj_to_target = task_index_ == 26
                                   ? std::abs(target_pos_[2] - obj[2])
                                   : Distance(obj, target_pos_);
        mjtNum init_distance = task_index_ == 26
                                   ? std::abs(target_pos_[2] - obj_init_pos_[2])
                                   : Distance(obj_init_pos_, target_pos_);
        mjtNum in_place = LongTailTolerance(obj_to_target, 0.0,
                                            task_.target_radius, init_distance);
        mjtNum object_grasped = GripperCagingReward(
            action, obj, task_index_ == 26 ? 0.022 : 0.032,
            task_index_ == 26 ? 0.05 : 0.06, 0.01, 0.01, 1.0, true, false);
        mjtNum reward = HamacherProduct(object_grasped, in_place);
        mjtNum tcp_to_obj = Distance(obj, tcp);
        if (tcp_to_obj < 0.035 && tcp_open > 0.0 &&
            obj[2] - 0.01 > obj_init_pos_[2]) {
          reward += 1.0 + 5.0 * in_place;
        }
        if (obj_to_target < task_.target_radius) {
          reward = 10.0;
        }
        return MakeInfo(
            reward,
            obj_to_target <= (task_index_ == 25 ? 0.08 : task_.target_radius)
                ? 1.0
                : 0.0,
            tcp_to_obj <= 0.05 ? 1.0 : 0.0,
            tcp_open > 0.0 && obj[2] - 0.03 > obj_init_pos_[2] ? 1.0 : 0.0,
            object_grasped, in_place, obj_to_target);
      }
      case 27: {  // lever-pull-v3
        auto shoulder = Add(hand, {0.0, 0.055, 0.07});
        auto init_shoulder = Add(init_tcp_, {0.0, 0.055, 0.07});
        mjtNum shoulder_to_lever =
            DistanceScaled(shoulder, obj, {4.0, 1.0, 4.0});
        mjtNum shoulder_init =
            DistanceScaled(init_shoulder, lever_pos_init_, {4.0, 1.0, 4.0});
        mjtNum ready =
            LongTailTolerance(shoulder_to_lever, 0.0, 0.02, shoulder_init);
        constexpr mjtNum k_pi = 3.14159265358979323846;
        mjtNum lever_angle = -JointQpos("LeverAxis");
        mjtNum lever_error = std::abs(lever_angle - k_pi / 2.0);
        mjtNum engagement = LongTailTolerance(lever_error, 0.0, k_pi / 48.0,
                                              k_pi / 2.0 - k_pi / 12.0);
        mjtNum in_place =
            LongTailTolerance(Distance(obj, target_pos_), 0.0, 0.04,
                              Distance(lever_pos_init_, target_pos_));
        mjtNum reward = 10.0 * HamacherProduct(ready, in_place);
        return MakeInfo(reward, lever_error <= k_pi / 24.0 ? 1.0 : 0.0,
                        shoulder_to_lever < 0.03 ? 1.0 : 0.0,
                        ready > 0.9 ? 1.0 : 0.0, ready, engagement,
                        shoulder_to_lever);
      }
      case 28: {  // pick-place-wall-v3
        auto midpoint = std::array<mjtNum, 3>{target_pos_[0], 0.77, 0.25};
        mjtNum tcp_to_obj = Distance(obj, tcp);
        mjtNum obj_to_mid = DistanceScaled(obj, midpoint, {1.0, 1.0, 3.0});
        mjtNum obj_to_mid_init =
            DistanceScaled(obj_init_pos_, midpoint, {1.0, 1.0, 3.0});
        mjtNum part1 =
            LongTailTolerance(obj_to_mid, 0.0, 0.05, obj_to_mid_init);
        mjtNum obj_to_target = Distance(obj, target_pos_);
        mjtNum part2 = LongTailTolerance(obj_to_target, 0.0, 0.05,
                                         Distance(obj_init_pos_, target_pos_));
        mjtNum object_grasped = GripperCagingReward(
            action, obj, 0.015, 0.05, 0.01, 0.005, 1.0, false, false);
        mjtNum grasped_place = HamacherProduct(object_grasped, part1);
        mjtNum reward = grasped_place;
        if (tcp_to_obj < 0.02 && tcp_open > 0.0 &&
            obj[2] - 0.015 > obj_init_pos_[2]) {
          reward = grasped_place + 1.0 + 4.0 * part1;
          if (obj[1] > 0.75) {
            reward = grasped_place + 1.0 + 4.0 + 3.0 * part2;
          }
        }
        if (obj_to_target < 0.05) {
          reward = 10.0;
        }
        return MakeInfo(reward, obj_to_target <= 0.07 ? 1.0 : 0.0,
                        tcp_to_obj <= 0.03 ? 1.0 : 0.0,
                        TouchingMainObject() && tcp_open > 0.0 &&
                                obj[2] - 0.02 > obj_init_pos_[2]
                            ? 1.0
                            : 0.0,
                        object_grasped, part2, obj_to_target);
      }
      case 29: {  // pick-out-of-hole-v3
        mjtNum obj_to_target = Distance(obj, target_pos_);
        mjtNum tcp_to_obj = Distance(obj, tcp);
        mjtNum radius = Distance2(tcp, obj_init_pos_);
        mjtNum floor =
            radius <= 0.03 ? 0.0 : 0.015 * std::log(radius - 0.03) + 0.15;
        mjtNum above_floor =
            tcp[2] >= floor
                ? 1.0
                : LongTailTolerance(std::max<mjtNum>(floor - tcp[2], 0.0), 0.0,
                                    0.01, 0.02);
        mjtNum object_grasped = GripperCagingReward(
            action, obj, 0.015, 0.02, 0.01, 0.03, 0.1, true, false);
        mjtNum in_place = LongTailTolerance(
            obj_to_target, 0.0, 0.02, Distance(obj_init_pos_, target_pos_));
        mjtNum reward = HamacherProduct(object_grasped, in_place);
        bool grasp_success = tcp_to_obj < 0.04 &&
                             obj[2] - 0.02 > obj_init_pos_[2] &&
                             !(tcp_open < 0.33);
        if (grasp_success) {
          reward += 1.0 + 5.0 * HamacherProduct(in_place, above_floor);
        }
        if (obj_to_target < task_.target_radius) {
          reward = 10.0;
        }
        return MakeInfo(reward, obj_to_target <= 0.07 ? 1.0 : 0.0,
                        tcp_to_obj <= 0.03 ? 1.0 : 0.0,
                        grasp_success ? 1.0 : 0.0, object_grasped, in_place,
                        obj_to_target);
      }
      case 30: {  // pick-place-v3
        mjtNum obj_to_target = Distance(obj, target_pos_);
        mjtNum tcp_to_obj = Distance(obj, tcp);
        mjtNum in_place = LongTailTolerance(
            obj_to_target, 0.0, 0.05, Distance(obj_init_pos_, target_pos_));
        mjtNum object_grasped = PickPlaceCagingReward(action, obj);
        mjtNum reward = HamacherProduct(object_grasped, in_place);
        if (tcp_to_obj < 0.02 && tcp_open > 0.0 &&
            obj[2] - 0.01 > obj_init_pos_[2]) {
          reward += 1.0 + 5.0 * in_place;
        }
        if (obj_to_target < 0.05) {
          reward = 10.0;
        }
        return MakeInfo(reward, obj_to_target <= 0.07 ? 1.0 : 0.0,
                        tcp_to_obj <= 0.03 ? 1.0 : 0.0,
                        TouchingMainObject() && tcp_open > 0.0 &&
                                obj[2] - 0.02 > obj_init_pos_[2]
                            ? 1.0
                            : 0.0,
                        object_grasped, in_place, obj_to_target);
      }
      case 31:
      case 32:
      case 33:
      case 34: {  // plate-slide variants
        mjtNum obj_to_target = Distance(obj, target_pos_);
        mjtNum in_margin = Distance(obj_init_pos_, target_pos_);
        mjtNum grasp_margin = Distance(init_tcp_, obj_init_pos_);
        if (task_index_ != 31) {
          in_margin -= 0.05;
          grasp_margin -= 0.05;
        }
        mjtNum in_place =
            LongTailTolerance(obj_to_target, 0.0, 0.05, in_margin);
        mjtNum tcp_to_obj = Distance(tcp, obj);
        mjtNum object_grasped =
            LongTailTolerance(tcp_to_obj, 0.0, 0.05, grasp_margin);
        mjtNum reward = 0.0;
        if (task_index_ == 31) {
          reward = 8.0 * HamacherProduct(object_grasped, in_place);
        } else {
          reward = 1.5 * object_grasped;
          if (tcp[2] <= 0.03 && tcp_to_obj < 0.07) {
            reward = 2.0 + 7.0 * in_place;
          }
        }
        if (obj_to_target < 0.05) {
          reward = 10.0;
        }
        return MakeInfo(reward, obj_to_target <= 0.07 ? 1.0 : 0.0,
                        tcp_to_obj <= 0.03 ? 1.0 : 0.0, 0.0, object_grasped,
                        in_place, obj_to_target);
      }
      case 35: {  // peg-insert-side-v3
        auto peg_head = SitePos("pegHead");
        mjtNum tcp_to_obj = Distance(obj, tcp);
        mjtNum obj_to_target =
            DistanceScaled(peg_head, target_pos_, {1.0, 2.0, 2.0});
        mjtNum in_margin =
            DistanceScaled(peg_head_pos_init_, target_pos_, {1.0, 2.0, 2.0});
        mjtNum in_place = LongTailTolerance(obj_to_target, 0.0,
                                            task_.target_radius, in_margin);
        mjtNum ip_orig = in_place;
        mjtNum col1 = RectPrismTolerance(
            peg_head, SitePos("bottom_right_corner_collision_box_1"),
            SitePos("top_left_corner_collision_box_1"));
        mjtNum col2 = RectPrismTolerance(
            peg_head, SitePos("bottom_right_corner_collision_box_2"),
            SitePos("top_left_corner_collision_box_2"));
        mjtNum collision_boxes = HamacherProduct(col2, col1);
        (void)ip_orig;
        in_place = HamacherProduct(in_place, collision_boxes);
        mjtNum object_grasped = GripperCagingReward(
            action, obj, 0.0075, 0.03, 0.01, 0.005, 1.0, true, false);
        bool lifted = tcp_to_obj < 0.08 && tcp_open > 0.0 &&
                      obj[2] - 0.01 > obj_init_pos_[2];
        if (lifted) {
          object_grasped = 1.0;
        }
        mjtNum reward = HamacherProduct(object_grasped, in_place);
        if (lifted) {
          reward += 1.0 + 5.0 * in_place;
        }
        if (obj_to_target <= 0.07) {
          reward = 10.0;
        }
        return MakeInfo(reward, obj_to_target <= 0.07 ? 1.0 : 0.0,
                        tcp_to_obj <= 0.03 ? 1.0 : 0.0,
                        tcp_to_obj < 0.02 && tcp_open > 0.0 &&
                                obj[2] - 0.01 > obj_init_pos_[2]
                            ? 1.0
                            : 0.0,
                        object_grasped, in_place, obj_to_target);
      }
      case 36: {  // peg-unplug-side-v3
        mjtNum tcp_to_obj = Distance(obj, tcp);
        mjtNum obj_to_target = Distance(obj, target_pos_);
        mjtNum object_grasped = GripperCagingReward(
            action, obj, 0.025, 0.05, 0.01, 0.005, 0.8, true, false);
        mjtNum in_place = LongTailTolerance(
            obj_to_target, 0.0, 0.05, Distance(obj_init_pos_, target_pos_));
        bool grasp_success =
            tcp_open > 0.5 && obj[0] - obj_init_pos_[0] > 0.015;
        mjtNum reward = 2.0 * object_grasped;
        if (grasp_success && tcp_to_obj < 0.035) {
          reward = 1.0 + 2.0 * object_grasped + 5.0 * in_place;
        }
        if (obj_to_target <= 0.05) {
          reward = 10.0;
        }
        return MakeInfo(reward, obj_to_target <= 0.07 ? 1.0 : 0.0,
                        tcp_to_obj <= 0.03 ? 1.0 : 0.0,
                        grasp_success ? 1.0 : 0.0, object_grasped, in_place,
                        obj_to_target);
      }
      case 37: {  // soccer-v3
        mjtNum tcp_to_obj = Distance(obj, tcp);
        mjtNum scaled_to_target =
            DistanceScaled(obj, target_pos_, {3.0, 1.0, 1.0});
        mjtNum scaled_init =
            DistanceScaled(obj, obj_init_pos_, {3.0, 1.0, 1.0});
        mjtNum in_place = LongTailTolerance(scaled_to_target, 0.0,
                                            task_.target_radius, scaled_init);
        mjtNum goal_line = target_pos_[1] - 0.1;
        if (obj[1] > goal_line && std::abs(obj[0] - target_pos_[0]) > 0.10) {
          in_place = std::clamp<mjtNum>(
              in_place - 2.0 * ((obj[1] - goal_line) / (1.0 - goal_line)), 0.0,
              1.0);
        }
        mjtNum object_grasped =
            SweepStyleCagingReward(action, obj, 0.013, 0.01, 0.005);
        mjtNum reward = 3.0 * object_grasped + 6.5 * in_place;
        if (scaled_to_target < task_.target_radius) {
          reward = 10.0;
        }
        mjtNum obj_to_target = Distance(obj, target_pos_);
        return MakeInfo(reward, obj_to_target <= 0.07 ? 1.0 : 0.0,
                        tcp_to_obj <= 0.03 ? 1.0 : 0.0,
                        TouchingMainObject() && tcp_open > 0.0 &&
                                obj[2] - 0.02 > obj_init_pos_[2]
                            ? 1.0
                            : 0.0,
                        object_grasped, in_place, obj_to_target);
      }
      case 38: {  // stick-push-v3
        auto stick = Add(obj, {0.015, 0.0, 0.0});
        auto container = std::array<mjtNum, 3>{obs[11], obs[12], obs[13]};
        mjtNum tcp_to_stick = Distance(stick, tcp);
        mjtNum stick_to_target = Distance(stick, target_pos_);
        mjtNum stick_in_place =
            LongTailTolerance(stick_to_target, 0.0, 0.12,
                              Distance(stick_init_pos_, target_pos_) - 0.12);
        mjtNum container_to_target = Distance(container, target_pos_);
        mjtNum container_in_place =
            LongTailTolerance(container_to_target, 0.0, 0.12,
                              Distance(obj_init_pos_, target_pos_) - 0.12);
        mjtNum object_grasped = StickCagingReward(
            action, stick, 0.04, 0.05, 0.01, 0.01, true, false, true);
        mjtNum reward = object_grasped;
        bool grasp_success = tcp_to_stick < 0.02 && tcp_open > 0.0 &&
                             stick[2] - 0.01 > stick_init_pos_[2];
        if (grasp_success) {
          object_grasped = 1.0;
          reward = 2.0 + 5.0 * stick_in_place + 3.0 * container_in_place;
          if (container_to_target <= 0.12) {
            reward = 10.0;
          }
        }
        bool success = Distance(container, target_pos_) <= 0.12;
        return MakeInfo(reward, grasp_success && success ? 1.0 : 0.0,
                        tcp_to_stick <= 0.03 ? 1.0 : 0.0,
                        TouchingMainObject() && tcp_open > 0.0 &&
                                stick[2] - 0.01 > stick_init_pos_[2]
                            ? 1.0
                            : 0.0,
                        object_grasped, stick_in_place, container_to_target);
      }
      case 39: {  // stick-pull-v3
        auto stick = obj;
        auto handle = std::array<mjtNum, 3>{obs[11], obs[12], obs[13]};
        auto end_of_stick = SitePos("stick_end");
        auto container = Add(handle, {0.05, 0.0, 0.0});
        auto container_init = Add(obj_init_pos_, {0.05, 0.0, 0.0});
        mjtNum tcp_to_stick = Distance(stick, tcp);
        mjtNum handle_to_target = Distance(handle, target_pos_);
        mjtNum stick_to_container =
            DistanceScaled(stick, container, {1.0, 1.0, 2.0});
        mjtNum stick_in_place = LongTailTolerance(
            stick_to_container, 0.0, 0.05,
            DistanceScaled(stick_init_pos_, container_init, {1.0, 1.0, 2.0}));
        mjtNum stick_to_target = Distance(stick, target_pos_);
        mjtNum stick_in_place2 = LongTailTolerance(
            stick_to_target, 0.0, 0.05, Distance(stick_init_pos_, target_pos_));
        mjtNum container_to_target = Distance(container, target_pos_);
        mjtNum container_in_place =
            LongTailTolerance(container_to_target, 0.0, 0.05,
                              Distance(obj_init_pos_, target_pos_));
        mjtNum object_grasped =
            StickCagingReward(action, stick, 0.014, 0.05, 0.01, 0.01, true);
        bool grasp_success = tcp_to_stick < 0.02 && tcp_open > 0.0 &&
                             stick[2] - 0.01 > stick_init_pos_[2];
        if (grasp_success) {
          object_grasped = 1.0;
        }
        mjtNum grasped_place = HamacherProduct(object_grasped, stick_in_place);
        mjtNum reward = grasped_place;
        bool inserted = end_of_stick[0] >= handle[0] &&
                        std::abs(end_of_stick[1] - handle[1]) <= 0.040 &&
                        std::abs(end_of_stick[2] - handle[2]) <= 0.060;
        if (grasp_success) {
          reward = 1.0 + grasped_place + 5.0 * stick_in_place;
          if (inserted) {
            reward = 1.0 + grasped_place + 5.0 + 2.0 * stick_in_place2 +
                     container_in_place;
            if (handle_to_target <= 0.12) {
              reward = 10.0;
            }
          }
        }
        bool success = Distance(handle, target_pos_) <= 0.12 && inserted;
        return MakeInfo(reward, success ? 1.0 : 0.0,
                        tcp_to_stick <= 0.03 ? 1.0 : 0.0,
                        TouchingMainObject() && tcp_open > 0.0 &&
                                stick[2] - 0.02 > obj_init_pos_[2]
                            ? 1.0
                            : 0.0,
                        object_grasped, stick_in_place, handle_to_target);
      }
      case 40: {  // push-v3
        mjtNum tcp_to_obj = Distance(obj, tcp);
        mjtNum target_to_obj = Distance(obj, target_pos_);
        mjtNum in_place =
            LongTailTolerance(target_to_obj, 0.0, task_.target_radius,
                              Distance(obj_init_pos_, target_pos_));
        mjtNum object_grasped = GripperCagingReward(
            action, obj, 0.015, 0.05, 0.01, 0.005, 1.0, true, false);
        mjtNum reward = 2.0 * object_grasped;
        if (tcp_to_obj < 0.02 && tcp_open > 0.0) {
          reward += 1.0 + reward + 5.0 * in_place;
        }
        if (target_to_obj < task_.target_radius) {
          reward = 10.0;
        }
        return default_six_info(reward, tcp_to_obj, tcp_open, target_to_obj,
                                object_grasped, in_place, task_.target_radius,
                                0.03, true);
      }
      case 41: {  // push-wall-v3
        auto midpoint = std::array<mjtNum, 3>{-0.05, 0.77, obj[2]};
        mjtNum tcp_to_obj = Distance(obj, tcp);
        mjtNum mid = DistanceScaled(obj, midpoint, {3.0, 1.0, 1.0});
        mjtNum mid_init =
            DistanceScaled(obj_init_pos_, midpoint, {3.0, 1.0, 1.0});
        mjtNum part1 = LongTailTolerance(mid, 0.0, 0.05, mid_init);
        mjtNum target_to_obj = Distance(obj, target_pos_);
        mjtNum part2 = LongTailTolerance(target_to_obj, 0.0, 0.05,
                                         Distance(obj_init_pos_, target_pos_));
        mjtNum object_grasped = GripperCagingReward(
            action, obj, 0.015, 0.05, 0.01, 0.005, 1.0, true, false);
        mjtNum reward = 2.0 * object_grasped;
        if (tcp_to_obj < 0.02 && tcp_open > 0.0) {
          reward = 2.0 * object_grasped + 1.0 + 4.0 * part1;
          if (obj[1] > 0.75) {
            reward = 2.0 * object_grasped + 1.0 + 4.0 + 3.0 * part2;
          }
        }
        if (target_to_obj < 0.05) {
          reward = 10.0;
        }
        return default_six_info(reward, tcp_to_obj, tcp_open, target_to_obj,
                                object_grasped, part2, 0.07, 0.03, true);
      }
      case 42: {  // push-back-v3
        mjtNum tcp_to_obj = Distance(obj, tcp);
        mjtNum target_to_obj = Distance(obj, target_pos_);
        mjtNum init = Distance(obj_init_pos_, target_pos_);
        mjtNum in_place =
            LongTailTolerance(target_to_obj, 0.0, task_.target_radius, init);
        mjtNum object_grasped =
            SweepStyleCagingReward(action, obj, 0.007, 0.003, 0.01);
        mjtNum reward = HamacherProduct(object_grasped, in_place);
        if (tcp_to_obj < 0.01 && 0.0 < tcp_open && tcp_open < 0.55 &&
            init - target_to_obj > 0.01) {
          reward += 1.0 + 5.0 * in_place;
        }
        if (target_to_obj < task_.target_radius) {
          reward = 10.0;
        }
        return default_six_info(reward, tcp_to_obj, tcp_open, target_to_obj,
                                object_grasped, in_place, 0.07, 0.03, true);
      }
      case 43:
      case 44: {  // reach variants
        mjtNum tcp_to_target = Distance(tcp, target_pos_);
        mjtNum in_place =
            LongTailTolerance(tcp_to_target, 0.0, 0.05,
                              Distance(ToMjt(task_.init_hand), target_pos_));
        mjtNum reward = 10.0 * in_place;
        if (task_index_ == 43) {
          return MakeInfo(reward, tcp_to_target <= 0.05 ? 1.0 : 0.0,
                          tcp_to_target, 1.0, tcp_to_target, in_place,
                          tcp_to_target);
        }
        return MakeInfo(reward, tcp_to_target <= 0.05 ? 1.0 : 0.0, 0.0, 0.0,
                        0.0, in_place, tcp_to_target);
      }
      case 45: {  // shelf-place-v3
        mjtNum obj_to_target = Distance(obj, target_pos_);
        mjtNum tcp_to_obj = Distance(obj, tcp);
        mjtNum in_place = LongTailTolerance(
            obj_to_target, 0.0, 0.05, Distance(obj_init_pos_, target_pos_));
        mjtNum object_grasped = GripperCagingReward(
            action, obj, 0.02, 0.05, 0.01, 0.01, 1.0, false, false);
        mjtNum reward = HamacherProduct(object_grasped, in_place);
        if (0.0 < obj[2] && obj[2] < 0.24 && target_pos_[0] - 0.15 < obj[0] &&
            obj[0] < target_pos_[0] + 0.15 && target_pos_[1] - 0.15 < obj[1] &&
            obj[1] < target_pos_[1]) {
          mjtNum z_scaling = (0.24 - obj[2]) / 0.24;
          mjtNum y_scaling = (obj[1] - (target_pos_[1] - 0.15)) / 0.15;
          mjtNum bound_loss = HamacherProduct(y_scaling, z_scaling);
          in_place = std::clamp<mjtNum>(in_place - bound_loss, 0.0, 1.0);
        }
        if (0.0 < obj[2] && obj[2] < 0.24 && target_pos_[0] - 0.15 < obj[0] &&
            obj[0] < target_pos_[0] + 0.15 && obj[1] > target_pos_[1]) {
          in_place = 0.0;
        }
        if (tcp_to_obj < 0.025 && tcp_open > 0.0 &&
            obj[2] - 0.01 > obj_init_pos_[2]) {
          reward += 1.0 + 5.0 * in_place;
        }
        if (obj_to_target < 0.05) {
          reward = 10.0;
        }
        return default_six_info(reward, tcp_to_obj, tcp_open, obj_to_target,
                                object_grasped, in_place, 0.07, 0.03, true);
      }
      case 46:
      case 47: {  // sweep variants
        auto target = target_pos_;
        if (task_index_ == 46) {
          target[2] = obj[2];
        }
        mjtNum obj_to_target = Distance(obj, target);
        mjtNum tcp_to_obj = Distance(obj, tcp);
        mjtNum in_place = LongTailTolerance(obj_to_target, 0.0, 0.05,
                                            Distance(obj_init_pos_, target));
        mjtNum grip_success_extra = task_index_ == 46 ? 0.005 : 0.01;
        mjtNum xz_thresh = task_index_ == 46 ? 0.01 : 0.005;
        mjtNum object_grasped = SweepStyleCagingReward(
            action, obj, 0.02, grip_success_extra, xz_thresh);
        mjtNum reward = 2.0 * object_grasped +
                        6.0 * HamacherProduct(object_grasped, in_place);
        if (obj_to_target < 0.05) {
          reward = 10.0;
        }
        return MakeInfo(reward, obj_to_target <= 0.05 ? 1.0 : 0.0,
                        tcp_to_obj <= 0.03 ? 1.0 : 0.0,
                        TouchingMainObject() && tcp_open > 0.0 ? 1.0 : 0.0,
                        object_grasped, in_place, obj_to_target);
      }
      case 48:
      case 49: {  // window open/close
        auto handle = task_index_ == 48 ? SitePos("handleOpenStart")
                                        : SitePos("handleCloseStart");
        mjtNum target_to_obj = std::abs(handle[0] - target_pos_[0]);
        mjtNum target_to_obj_init =
            std::abs((task_index_ == 48 ? obj_init_pos_[0]
                                        : window_handle_pos_init_[0]) -
                     target_pos_[0]);
        mjtNum in_place = LongTailTolerance(
            target_to_obj, 0.0, task_.target_radius,
            std::abs(target_to_obj_init - task_.target_radius));
        mjtNum tcp_to_obj = Distance(handle, tcp);
        mjtNum reach =
            (task_index_ == 48 ? LongTailTolerance : GaussianTolerance)(
                tcp_to_obj, 0.0, 0.02,
                std::abs(Distance(window_handle_pos_init_, init_tcp_) - 0.02));
        mjtNum reward = 10.0 * HamacherProduct(reach, in_place);
        return MakeInfo(reward,
                        target_to_obj <= task_.target_radius ? 1.0 : 0.0,
                        tcp_to_obj <= 0.05 ? 1.0 : 0.0, 1.0, reach, in_place,
                        target_to_obj);
      }
      default:
        break;
    }
    mjtNum obj_to_target = Distance(obj, target_pos_);
    mjtNum in_place = LongTailTolerance(
        obj_to_target, 0.0, task_.target_radius,
        std::max<mjtNum>(Distance(obj_init_pos_, target_pos_), 1e-6));
    mjtNum reward = 10.0 * in_place;
    return MakeInfo(reward, obj_to_target <= task_.target_radius ? 1.0 : 0.0,
                    Distance(obj, tcp) <= 0.03 ? 1.0 : 0.0, 0.0, in_place,
                    in_place, obj_to_target);
  }

  void CaptureResetState() {
    qpos0_.fill(0.0);
    qvel0_.fill(0.0);
    qacc0_.fill(0.0);
    qacc_warmstart0_.fill(0.0);
    mocap_pos0_.fill(0.0);
    mocap_quat0_.fill(0.0);
    std::memcpy(
        qpos0_.data(), data_->qpos,
        sizeof(mjtNum) * std::min<int>(static_cast<int>(model_->nq),
                                       static_cast<int>(qpos0_.size())));
    std::memcpy(
        qvel0_.data(), data_->qvel,
        sizeof(mjtNum) * std::min<int>(static_cast<int>(model_->nv),
                                       static_cast<int>(qvel0_.size())));
    std::memcpy(
        qacc0_.data(), data_->qacc,
        sizeof(mjtNum) * std::min<int>(static_cast<int>(model_->nv),
                                       static_cast<int>(qacc0_.size())));
    std::memcpy(qacc_warmstart0_.data(), data_->qacc_warmstart,
                sizeof(mjtNum) *
                    std::min<int>(static_cast<int>(model_->nv),
                                  static_cast<int>(qacc_warmstart0_.size())));
    if (mocap_id_ >= 0 && model_->nmocap > mocap_id_) {
      std::memcpy(mocap_pos0_.data(), data_->mocap_pos + 3 * mocap_id_,
                  sizeof(mjtNum) * mocap_pos0_.size());
      std::memcpy(mocap_quat0_.data(), data_->mocap_quat + 4 * mocap_id_,
                  sizeof(mjtNum) * mocap_quat0_.size());
    }
  }

  void WriteState(const RewardInfo& reward_info, bool reset) {
    auto state = Allocate();
    state["reward"_] = static_cast<float>(reward_info.reward);
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      auto obs_state = state["obs"_];
      mjtNum* obs = PrepareObservation("obs", &obs_state);
      auto curr = CurrentObsNoGoal();
      for (mjtNum value : curr) {
        *(obs++) = value;
      }
      for (mjtNum value : prev_obs_) {
        *(obs++) = value;
      }
      std::array<mjtNum, 3> goal = partially_observable_
                                       ? std::array<mjtNum, 3>{0.0, 0.0, 0.0}
                                       : target_pos_;
      for (mjtNum value : goal) {
        *(obs++) = value;
      }
      CommitObservation("obs", &obs_state, reset);
      prev_obs_ = curr;
    }
    state["info:success"_] = reward_info.success;
    state["info:near_object"_] = reward_info.near_object;
    state["info:grasp_success"_] = reward_info.grasp_success;
    state["info:grasp_reward"_] = reward_info.grasp_reward;
    state["info:in_place_reward"_] = reward_info.in_place_reward;
    state["info:obj_to_target"_] = reward_info.obj_to_target;
    state["info:unscaled_reward"_] = reward_info.reward;
    state["info:task_id"_] = task_index_;
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
    state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
    state["info:qacc0"_].Assign(qacc0_.data(), qacc0_.size());
    state["info:qacc_warmstart0"_].Assign(qacc_warmstart0_.data(),
                                          qacc_warmstart0_.size());
    state["info:target0"_].Assign(target_pos_.data(), target_pos_.size());
    state["info:rand_vec0"_].Assign(rand_vec_.data(), rand_vec_.size());
    state["info:init_tcp0"_].Assign(init_tcp_.data(), init_tcp_.size());
    state["info:init_left_pad0"_].Assign(init_left_pad_.data(),
                                         init_left_pad_.size());
    state["info:init_right_pad0"_].Assign(init_right_pad_.data(),
                                          init_right_pad_.size());
    state["info:mocap_pos0"_].Assign(mocap_pos0_.data(), mocap_pos0_.size());
    state["info:mocap_quat0"_].Assign(mocap_quat0_.data(), mocap_quat0_.size());
#endif
  }
};

using MetaWorldEnv = MetaWorldEnvBase<MetaWorldEnvSpec, false>;
using MetaWorldPixelEnv = MetaWorldEnvBase<MetaWorldPixelEnvSpec, true>;
using MetaWorldEnvPool = AsyncEnvPool<MetaWorldEnv>;
using MetaWorldPixelEnvPool = AsyncEnvPool<MetaWorldPixelEnv>;

}  // namespace metaworld

#endif  // ENVPOOL_MUJOCO_METAWORLD_METAWORLD_ENV_H_
