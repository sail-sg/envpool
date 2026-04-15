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

#ifndef ENVPOOL_MUJOCO_MYOSUITE_MYOBASE_H_
#define ENVPOOL_MUJOCO_MYOSUITE_MYOBASE_H_

#include <mujoco.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/myosuite/paths.h"
#include "envpool/mujoco/robotics/mujoco_env.h"

namespace myosuite_envpool {

using envpool::mujoco::PixelObservationEnvFns;
using envpool::mujoco::RenderCameraIdOrDefault;
using envpool::mujoco::RenderHeightOrDefault;
using envpool::mujoco::RenderWidthOrDefault;
using envpool::mujoco::StackSpec;

namespace detail {

constexpr mjtNum kPoseFarThreshold = static_cast<mjtNum>(6.283185307179586);

inline std::vector<mjtNum> ToMjtVector(const std::vector<double>& input) {
  return {input.begin(), input.end()};
}

inline mjtNum ClampNormalized(mjtNum value) {
  return std::clamp(value, static_cast<mjtNum>(-1.0), static_cast<mjtNum>(1.0));
}

inline mjtNum MuscleActivation(mjtNum value) {
  return static_cast<mjtNum>(
      1.0 / (1.0 + std::exp(-5.0 * (static_cast<double>(value) - 0.5))));
}

inline mjtNum VectorNorm(const std::vector<mjtNum>& value) {
  mjtNum total = 0.0;
  for (mjtNum item : value) {
    total += item * item;
  }
  return std::sqrt(total);
}

inline mjtNum VectorDot(const std::vector<mjtNum>& lhs,
                        const std::vector<mjtNum>& rhs) {
  mjtNum total = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    total += lhs[i] * rhs[i];
  }
  return total;
}

inline mjtNum CosineSimilarity(const std::vector<mjtNum>& lhs,
                               const std::vector<mjtNum>& rhs) {
  mjtNum lhs_norm = VectorNorm(lhs);
  mjtNum rhs_norm = VectorNorm(rhs);
  if (lhs_norm == 0.0 || rhs_norm == 0.0) {
    return 0.0;
  }
  return VectorDot(lhs, rhs) / (lhs_norm * rhs_norm);
}

inline std::vector<mjtNum> CurrentAct(const mjModel* model,
                                      const mjData* data) {
  if (model->na == 0) {
    return {};
  }
  return {data->act, data->act + model->na};
}

inline mjtNum ActReg(const mjModel* model, const mjData* data) {
  if (model->na == 0) {
    return 0.0;
  }
  return VectorNorm(CurrentAct(model, data)) / static_cast<mjtNum>(model->na);
}

inline std::vector<mjtNum> CopyQpos(const mjModel* model, const mjData* data) {
  return {data->qpos, data->qpos + model->nq};
}

inline std::vector<mjtNum> CopyQvel(const mjModel* model, const mjData* data,
                                    mjtNum dt) {
  std::vector<mjtNum> qvel(model->nv);
  for (int i = 0; i < model->nv; ++i) {
    qvel[i] = data->qvel[i] * dt;
  }
  return qvel;
}

inline void RestoreVector(const std::vector<mjtNum>& src, mjtNum* dst) {
  if (!src.empty()) {
    std::memcpy(dst, src.data(), sizeof(mjtNum) * src.size());
  }
}

inline void CopySitePos(const mjModel* model, const mjData* data, int src_site,
                        mjtNum* dst_site_pos) {
  for (int axis = 0; axis < 3; ++axis) {
    dst_site_pos[axis] = data->site_xpos[src_site * 3 + axis];
  }
}

inline void CopyModelBodyPos(const mjModel* model, int body_id,
                             std::vector<mjtNum>* out) {
  out->assign(model->body_pos + body_id * 3, model->body_pos + body_id * 3 + 3);
}

inline void RestoreModelBodyPos(mjModel* model, int body_id,
                                const std::vector<mjtNum>& value) {
  if (value.empty()) {
    return;
  }
  std::memcpy(model->body_pos + body_id * 3, value.data(), sizeof(mjtNum) * 3);
}

inline void CopyModelBodyQuat(const mjModel* model, int body_id,
                              std::vector<mjtNum>* out) {
  out->assign(model->body_quat + body_id * 4,
              model->body_quat + body_id * 4 + 4);
}

inline void RestoreModelBodyQuat(mjModel* model, int body_id,
                                 const std::vector<mjtNum>& value) {
  if (value.empty()) {
    return;
  }
  std::memcpy(model->body_quat + body_id * 4, value.data(), sizeof(mjtNum) * 4);
}

inline void CopyModelSitePos(const mjModel* model, int site_id,
                             std::vector<mjtNum>* out) {
  out->assign(model->site_pos + site_id * 3, model->site_pos + site_id * 3 + 3);
}

inline void RestoreModelSitePos(mjModel* model, int site_id,
                                const std::vector<mjtNum>& value) {
  if (value.empty()) {
    return;
  }
  std::memcpy(model->site_pos + site_id * 3, value.data(), sizeof(mjtNum) * 3);
}

inline void CopyModelSiteSize(const mjModel* model, int site_id,
                              std::vector<mjtNum>* out) {
  out->assign(model->site_size + site_id * 3,
              model->site_size + site_id * 3 + 3);
}

inline void RestoreModelSiteSize(mjModel* model, int site_id,
                                 const std::vector<mjtNum>& value) {
  if (value.empty()) {
    return;
  }
  std::memcpy(model->site_size + site_id * 3, value.data(), sizeof(mjtNum) * 3);
}

inline void CopyModelGeomSize(const mjModel* model, int geom_id,
                              std::vector<mjtNum>* out) {
  out->assign(model->geom_size + geom_id * 3,
              model->geom_size + geom_id * 3 + 3);
}

inline void RestoreModelGeomSize(mjModel* model, int geom_id,
                                 const std::vector<mjtNum>& value) {
  if (value.empty()) {
    return;
  }
  std::memcpy(model->geom_size + geom_id * 3, value.data(), sizeof(mjtNum) * 3);
}

inline void CopyModelGeomPos(const mjModel* model, int geom_id,
                             std::vector<mjtNum>* out) {
  out->assign(model->geom_pos + geom_id * 3, model->geom_pos + geom_id * 3 + 3);
}

inline void RestoreModelGeomPos(mjModel* model, int geom_id,
                                const std::vector<mjtNum>& value) {
  if (value.empty()) {
    return;
  }
  std::memcpy(model->geom_pos + geom_id * 3, value.data(), sizeof(mjtNum) * 3);
}

inline void CopyModelGeomQuat(const mjModel* model, int geom_id,
                              std::vector<mjtNum>* out) {
  out->assign(model->geom_quat + geom_id * 4,
              model->geom_quat + geom_id * 4 + 4);
}

inline void RestoreModelGeomQuat(mjModel* model, int geom_id,
                                 const std::vector<mjtNum>& value) {
  if (value.empty()) {
    return;
  }
  std::memcpy(model->geom_quat + geom_id * 4, value.data(), sizeof(mjtNum) * 4);
}

inline void CopyModelGeomRgba(const mjModel* model, int geom_id,
                              std::vector<mjtNum>* out) {
  out->assign(model->geom_rgba + geom_id * 4,
              model->geom_rgba + geom_id * 4 + 4);
}

inline void RestoreModelGeomRgba(mjModel* model, int geom_id,
                                 const std::vector<mjtNum>& value) {
  if (value.empty()) {
    return;
  }
  std::memcpy(model->geom_rgba + geom_id * 4, value.data(), sizeof(mjtNum) * 4);
}

inline void CopyModelSiteRgba(const mjModel* model, int site_id,
                              std::vector<mjtNum>* out) {
  out->assign(model->site_rgba + site_id * 4,
              model->site_rgba + site_id * 4 + 4);
}

inline void RestoreModelSiteRgba(mjModel* model, int site_id,
                                 const std::vector<mjtNum>& value) {
  if (value.empty()) {
    return;
  }
  std::memcpy(model->site_rgba + site_id * 4, value.data(), sizeof(mjtNum) * 4);
}

inline void CopyModelSitePosAndSize(const mjModel* model, int site_id,
                                    std::vector<mjtNum>* out_pos,
                                    std::vector<mjtNum>* out_size) {
  CopyModelSitePos(model, site_id, out_pos);
  CopyModelSiteSize(model, site_id, out_size);
}

inline void CopyModelBodyMass(const mjModel* model, int body_id, mjtNum* out) {
  *out = model->body_mass[body_id];
}

inline void RestoreModelBodyMass(mjModel* model, int body_id, mjtNum value) {
  model->body_mass[body_id] = value;
}

inline void CopyModelGeomType(const mjModel* model, int geom_id, int* out) {
  *out = model->geom_type[geom_id];
}

inline void RestoreModelGeomType(mjModel* model, int geom_id, int value) {
  if (value < 0) {
    return;
  }
  model->geom_type[geom_id] = value;
}

inline void CopyModelGeomCondim(const mjModel* model, int geom_id, int* out) {
  *out = model->geom_condim[geom_id];
}

inline void RestoreModelGeomCondim(mjModel* model, int geom_id, int value) {
  if (value < 0) {
    return;
  }
  model->geom_condim[geom_id] = value;
}

inline void CopyModelGeomContype(const mjModel* model, int geom_id, int* out) {
  *out = model->geom_contype[geom_id];
}

inline void RestoreModelGeomContype(mjModel* model, int geom_id, int value) {
  if (value < 0) {
    return;
  }
  model->geom_contype[geom_id] = value;
}

inline void CopyModelGeomConaffinity(const mjModel* model, int geom_id,
                                     int* out) {
  *out = model->geom_conaffinity[geom_id];
}

inline void RestoreModelGeomConaffinity(mjModel* model, int geom_id,
                                        int value) {
  if (value < 0) {
    return;
  }
  model->geom_conaffinity[geom_id] = value;
}

inline void CopyModelGeomFriction(const mjModel* model, int geom_id,
                                  std::vector<mjtNum>* out) {
  out->assign(model->geom_friction + geom_id * 3,
              model->geom_friction + geom_id * 3 + 3);
}

inline void RestoreModelGeomFriction(mjModel* model, int geom_id,
                                     const std::vector<mjtNum>& value) {
  if (value.empty()) {
    return;
  }
  std::memcpy(model->geom_friction + geom_id * 3, value.data(),
              sizeof(mjtNum) * 3);
}

inline void CopyModelHfieldData(const mjModel* model, int hfield_id,
                                std::vector<mjtNum>* out) {
  int adr = model->hfield_adr[hfield_id];
  int rows = model->hfield_nrow[hfield_id];
  int cols = model->hfield_ncol[hfield_id];
  out->assign(model->hfield_data + adr, model->hfield_data + adr + rows * cols);
}

inline void RestoreModelHfieldData(mjModel* model, int hfield_id,
                                   const std::vector<mjtNum>& value) {
  if (value.empty()) {
    return;
  }
  int adr = model->hfield_adr[hfield_id];
  int rows = model->hfield_nrow[hfield_id];
  int cols = model->hfield_ncol[hfield_id];
  std::size_t expected = static_cast<std::size_t>(rows) * cols;
  std::memcpy(model->hfield_data + adr, value.data(),
              sizeof(mjtNum) * expected);
}

inline void BuildMuscleMask(const mjModel* model,
                            std::vector<bool>* muscle_actuator) {
  muscle_actuator->assign(model->nu, false);
  for (int i = 0; i < model->nu; ++i) {
    (*muscle_actuator)[i] = model->actuator_dyntype[i] == mjDYN_MUSCLE;
  }
}

inline void AdjustInitialQposForNormalizedActions(const mjModel* model,
                                                  mjData* data,
                                                  bool normalize_act) {
  if (!normalize_act) {
    return;
  }
  std::vector<bool> updated(model->njnt, false);
  for (int actuator_id = 0; actuator_id < model->nu; ++actuator_id) {
    if (model->actuator_trntype[actuator_id] != mjTRN_JOINT) {
      continue;
    }
    int joint_id = model->actuator_trnid[actuator_id * 2];
    if (joint_id < 0 || updated[joint_id]) {
      continue;
    }
    int joint_type = model->jnt_type[joint_id];
    if (joint_type != mjJNT_HINGE && joint_type != mjJNT_SLIDE) {
      continue;
    }
    int qpos_addr = model->jnt_qposadr[joint_id];
    mjtNum low = model->jnt_range[joint_id * 2];
    mjtNum high = model->jnt_range[joint_id * 2 + 1];
    data->qpos[qpos_addr] = (low + high) * static_cast<mjtNum>(0.5);
    updated[joint_id] = true;
  }
  mj_forward(model, data);
}

inline void ApplyMyoSuiteAction(const mjModel* model, mjData* data,
                                const std::vector<bool>& muscle_actuator,
                                bool normalize_act, const float* raw) {
  for (int i = 0; i < model->nu; ++i) {
    mjtNum value = ClampNormalized(static_cast<mjtNum>(raw[i]));
    if (normalize_act && muscle_actuator[i] && model->na != 0) {
      value = MuscleActivation(value);
    } else if (normalize_act && model->na == 0) {
      mjtNum low = model->actuator_ctrlrange[i * 2];
      mjtNum high = model->actuator_ctrlrange[i * 2 + 1];
      value = (low + high) * static_cast<mjtNum>(0.5) +
              value * (high - low) * static_cast<mjtNum>(0.5);
    }
    data->ctrl[i] = value;
  }
}

}  // namespace detail

class MyoSuitePoseEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(10),
        "frame_stack"_.Bind(1), "model_path"_.Bind(std::string()),
        "normalize_act"_.Bind(true), "obs_dim"_.Bind(0), "qpos_dim"_.Bind(0),
        "qvel_dim"_.Bind(0), "act_dim"_.Bind(0), "action_dim"_.Bind(0),
        "pose_thd"_.Bind(0.35), "reward_pose_w"_.Bind(1.0),
        "reward_bonus_w"_.Bind(4.0), "reward_act_reg_w"_.Bind(1.0),
        "reward_penalty_w"_.Bind(50.0), "reset_type"_.Bind(std::string("init")),
        "target_type"_.Bind(std::string("generate")),
        "target_qpos_min"_.Bind(std::vector<double>{}),
        "target_qpos_max"_.Bind(std::vector<double>{}),
        "target_qpos_value"_.Bind(std::vector<double>{}),
        "viz_site_targets"_.Bind(std::vector<std::string>{}),
        "weight_bodyname"_.Bind(std::string()),
        "weight_range"_.Bind(std::vector<double>{}),
        "test_reset_qpos"_.Bind(std::vector<double>{}),
        "test_reset_qvel"_.Bind(std::vector<double>{}),
        "test_reset_act"_.Bind(std::vector<double>{}),
        "test_reset_qacc_warmstart"_.Bind(std::vector<double>{}),
        "test_target_qpos"_.Bind(std::vector<double>{}),
        "test_body_mass"_.Bind(std::vector<double>{}),
        "test_geom_size0"_.Bind(std::vector<double>{}));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs"_.Bind(StackSpec(Spec<mjtNum>({conf["obs_dim"_]}, {-inf, inf}),
                              conf["frame_stack"_])),
        "info:pose_dist"_.Bind(Spec<mjtNum>({-1}, {0.0, inf})),
        "info:act_reg"_.Bind(Spec<mjtNum>({-1}, {0.0, inf})),
        "info:success"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
        "info:target_qpos"_.Bind(Spec<mjtNum>({conf["qpos_dim"_]})),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({conf["qpos_dim"_]})),
        "info:qvel0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
#endif
        "info:weight_mass"_.Bind(Spec<mjtNum>({-1}, {0.0, inf})),
        "info:weight_geom_size0"_.Bind(Spec<mjtNum>({-1}, {0.0, inf})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<float>({-1, conf["action_dim"_]}, {-1.0, 1.0})));
  }
};

using MyoSuitePoseEnvSpec = EnvSpec<MyoSuitePoseEnvFns>;
using MyoSuitePosePixelEnvFns = PixelObservationEnvFns<MyoSuitePoseEnvFns>;
using MyoSuitePosePixelEnvSpec = EnvSpec<MyoSuitePosePixelEnvFns>;

class MyoSuiteReachEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(10),
        "frame_stack"_.Bind(1), "model_path"_.Bind(std::string()),
        "normalize_act"_.Bind(true), "obs_dim"_.Bind(0), "qpos_dim"_.Bind(0),
        "qvel_dim"_.Bind(0), "act_dim"_.Bind(0), "action_dim"_.Bind(0),
        "target_site_count"_.Bind(0), "far_th"_.Bind(0.35),
        "reward_reach_w"_.Bind(1.0), "reward_bonus_w"_.Bind(4.0),
        "reward_act_reg_w"_.Bind(0.0), "reward_penalty_w"_.Bind(50.0),
        "target_site_names"_.Bind(std::vector<std::string>{}),
        "target_pos_min"_.Bind(std::vector<double>{}),
        "target_pos_max"_.Bind(std::vector<double>{}),
        "test_reset_qpos"_.Bind(std::vector<double>{}),
        "test_reset_qvel"_.Bind(std::vector<double>{}),
        "test_reset_act"_.Bind(std::vector<double>{}),
        "test_reset_qacc_warmstart"_.Bind(std::vector<double>{}),
        "test_target_pos"_.Bind(std::vector<double>{}));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs"_.Bind(StackSpec(Spec<mjtNum>({conf["obs_dim"_]}, {-inf, inf}),
                              conf["frame_stack"_])),
        "info:reach_dist"_.Bind(Spec<mjtNum>({-1}, {0.0, inf})),
        "info:act_reg"_.Bind(Spec<mjtNum>({-1}, {0.0, inf})),
        "info:success"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
        "info:target_pos"_.Bind(Spec<mjtNum>({conf["target_site_count"_] * 3})),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({conf["qpos_dim"_]})),
        "info:qvel0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
#endif
        "info:time"_.Bind(Spec<mjtNum>({-1}, {0.0, inf})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<float>({-1, conf["action_dim"_]}, {-1.0, 1.0})));
  }
};

using MyoSuiteReachEnvSpec = EnvSpec<MyoSuiteReachEnvFns>;
using MyoSuiteReachPixelEnvFns = PixelObservationEnvFns<MyoSuiteReachEnvFns>;
using MyoSuiteReachPixelEnvSpec = EnvSpec<MyoSuiteReachPixelEnvFns>;

class MyoSuiteKeyTurnEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(10),
        "frame_stack"_.Bind(1), "model_path"_.Bind(std::string()),
        "normalize_act"_.Bind(true), "obs_dim"_.Bind(0), "qpos_dim"_.Bind(0),
        "qvel_dim"_.Bind(0), "act_dim"_.Bind(0), "action_dim"_.Bind(0),
        "goal_th"_.Bind(3.14), "reward_key_turn_w"_.Bind(1.0),
        "reward_iftip_approach_w"_.Bind(10.0),
        "reward_thtip_approach_w"_.Bind(10.0), "reward_act_reg_w"_.Bind(1.0),
        "reward_bonus_w"_.Bind(4.0), "reward_penalty_w"_.Bind(25.0),
        "key_init_range"_.Bind(std::vector<double>{0.0, 0.0}),
        "test_reset_qpos"_.Bind(std::vector<double>{}),
        "test_reset_qvel"_.Bind(std::vector<double>{}),
        "test_reset_act"_.Bind(std::vector<double>{}),
        "test_reset_qacc_warmstart"_.Bind(std::vector<double>{}),
        "test_key_body_pos"_.Bind(std::vector<double>{}));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs"_.Bind(StackSpec(Spec<mjtNum>({conf["obs_dim"_]}, {-inf, inf}),
                              conf["frame_stack"_])),
        "info:key_turn"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:iftip_approach_dist"_.Bind(Spec<mjtNum>({-1}, {0.0, inf})),
        "info:thtip_approach_dist"_.Bind(Spec<mjtNum>({-1}, {0.0, inf})),
        "info:success"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({conf["qpos_dim"_]})),
        "info:qvel0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
#endif
        "info:key_body_pos"_.Bind(Spec<mjtNum>({3})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<float>({-1, conf["action_dim"_]}, {-1.0, 1.0})));
  }
};

using MyoSuiteKeyTurnEnvSpec = EnvSpec<MyoSuiteKeyTurnEnvFns>;
using MyoSuiteKeyTurnPixelEnvFns =
    // NOLINTNEXTLINE(whitespace/indent_namespace)
    PixelObservationEnvFns<MyoSuiteKeyTurnEnvFns>;
using MyoSuiteKeyTurnPixelEnvSpec = EnvSpec<MyoSuiteKeyTurnPixelEnvFns>;

class MyoSuiteObjHoldEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(10),
        "frame_stack"_.Bind(1), "model_path"_.Bind(std::string()),
        "normalize_act"_.Bind(true), "obs_dim"_.Bind(0), "qpos_dim"_.Bind(0),
        "qvel_dim"_.Bind(0), "act_dim"_.Bind(0), "action_dim"_.Bind(0),
        "randomize_on_reset"_.Bind(false), "reward_goal_dist_w"_.Bind(100.0),
        "reward_bonus_w"_.Bind(4.0), "reward_penalty_w"_.Bind(10.0),
        "test_reset_qpos"_.Bind(std::vector<double>{}),
        "test_reset_qvel"_.Bind(std::vector<double>{}),
        "test_reset_act"_.Bind(std::vector<double>{}),
        "test_reset_qacc_warmstart"_.Bind(std::vector<double>{}),
        "test_goal_pos"_.Bind(std::vector<double>{}),
        "test_object_geom_size"_.Bind(std::vector<double>{}));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs"_.Bind(StackSpec(Spec<mjtNum>({conf["obs_dim"_]}, {-inf, inf}),
                              conf["frame_stack"_])),
        "info:goal_dist"_.Bind(Spec<mjtNum>({-1}, {0.0, inf})),
        "info:success"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({conf["qpos_dim"_]})),
        "info:qvel0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
#endif
        "info:goal_pos"_.Bind(Spec<mjtNum>({3})),
        "info:object_geom_size"_.Bind(Spec<mjtNum>({3})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<float>({-1, conf["action_dim"_]}, {-1.0, 1.0})));
  }
};

using MyoSuiteObjHoldEnvSpec = EnvSpec<MyoSuiteObjHoldEnvFns>;
using MyoSuiteObjHoldPixelEnvFns =
    // NOLINTNEXTLINE(whitespace/indent_namespace)
    PixelObservationEnvFns<MyoSuiteObjHoldEnvFns>;
using MyoSuiteObjHoldPixelEnvSpec = EnvSpec<MyoSuiteObjHoldPixelEnvFns>;

class MyoSuiteTorsoEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(5),
        "frame_stack"_.Bind(1), "model_path"_.Bind(std::string()),
        "normalize_act"_.Bind(true), "obs_dim"_.Bind(0), "qpos_dim"_.Bind(0),
        "qvel_dim"_.Bind(0), "act_dim"_.Bind(0), "action_dim"_.Bind(0),
        "pose_dim"_.Bind(0), "pose_thd"_.Bind(0.25), "reward_pose_w"_.Bind(1.0),
        "reward_bonus_w"_.Bind(4.0), "reward_act_reg_w"_.Bind(1.0),
        "reward_penalty_w"_.Bind(50.0),
        "target_qpos_value"_.Bind(std::vector<double>{}),
        "test_reset_qpos"_.Bind(std::vector<double>{}),
        "test_reset_qvel"_.Bind(std::vector<double>{}),
        "test_reset_act"_.Bind(std::vector<double>{}),
        "test_reset_qacc_warmstart"_.Bind(std::vector<double>{}));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs"_.Bind(StackSpec(Spec<mjtNum>({conf["obs_dim"_]}, {-inf, inf}),
                              conf["frame_stack"_])),
        "info:pose_dist"_.Bind(Spec<mjtNum>({-1}, {0.0, inf})),
        "info:success"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({conf["qpos_dim"_]})),
        "info:qvel0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
#endif
        "info:target_qpos"_.Bind(Spec<mjtNum>({conf["pose_dim"_]})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<float>({-1, conf["action_dim"_]}, {-1.0, 1.0})));
  }
};

using MyoSuiteTorsoEnvSpec = EnvSpec<MyoSuiteTorsoEnvFns>;
using MyoSuiteTorsoPixelEnvFns = PixelObservationEnvFns<MyoSuiteTorsoEnvFns>;
using MyoSuiteTorsoPixelEnvSpec = EnvSpec<MyoSuiteTorsoPixelEnvFns>;

class MyoSuitePenTwirlEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(5),
        "frame_stack"_.Bind(1), "model_path"_.Bind(std::string()),
        "normalize_act"_.Bind(true), "obs_dim"_.Bind(0), "qpos_dim"_.Bind(0),
        "qvel_dim"_.Bind(0), "act_dim"_.Bind(0), "action_dim"_.Bind(0),
        "randomize_target"_.Bind(false), "reward_pos_align_w"_.Bind(1.0),
        "reward_rot_align_w"_.Bind(1.0), "reward_act_reg_w"_.Bind(5.0),
        "reward_drop_w"_.Bind(5.0), "reward_bonus_w"_.Bind(10.0),
        "test_reset_qpos"_.Bind(std::vector<double>{}),
        "test_reset_qvel"_.Bind(std::vector<double>{}),
        "test_reset_act"_.Bind(std::vector<double>{}),
        "test_reset_qacc_warmstart"_.Bind(std::vector<double>{}),
        "test_target_body_quat"_.Bind(std::vector<double>{}));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs"_.Bind(StackSpec(Spec<mjtNum>({conf["obs_dim"_]}, {-inf, inf}),
                              conf["frame_stack"_])),
        "info:pos_align"_.Bind(Spec<mjtNum>({-1}, {0.0, inf})),
        "info:rot_align"_.Bind(Spec<mjtNum>({-1}, {-1.0, 1.0})),
        "info:success"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({conf["qpos_dim"_]})),
        "info:qvel0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
#endif
        "info:target_body_quat"_.Bind(Spec<mjtNum>({4})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<float>({-1, conf["action_dim"_]}, {-1.0, 1.0})));
  }
};

using MyoSuitePenTwirlEnvSpec = EnvSpec<MyoSuitePenTwirlEnvFns>;
using MyoSuitePenTwirlPixelEnvFns =
    // NOLINTNEXTLINE(whitespace/indent_namespace)
    PixelObservationEnvFns<MyoSuitePenTwirlEnvFns>;
using MyoSuitePenTwirlPixelEnvSpec = EnvSpec<MyoSuitePenTwirlPixelEnvFns>;

template <typename EnvSpecT, bool kFromPixels>
class MyoSuitePoseEnvBase : public Env<EnvSpecT>,
                            public gymnasium_robotics::MujocoRobotEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  struct RewardInfo {
    mjtNum dense_reward{0.0};
    mjtNum pose_dist{0.0};
    mjtNum act_reg{0.0};
    bool success{false};
    bool done{false};
  };

  bool normalize_act_;
  mjtNum pose_thd_;
  mjtNum reward_pose_w_;
  mjtNum reward_bonus_w_;
  mjtNum reward_act_reg_w_;
  mjtNum reward_penalty_w_;
  std::string reset_type_;
  std::string target_type_;
  std::vector<mjtNum> target_qpos_min_;
  std::vector<mjtNum> target_qpos_max_;
  std::vector<mjtNum> default_target_qpos_;
  std::vector<mjtNum> current_target_qpos_;
  std::vector<int> tip_site_ids_;
  std::vector<int> target_site_ids_;
  std::vector<mjtNum> initial_target_site_pos_;
  std::vector<bool> muscle_actuator_;
  int weight_body_id_{-1};
  int weight_geom_id_{-1};
  mjtNum initial_weight_body_mass_{0.0};
  mjtNum initial_weight_geom_size0_{0.0};
  std::vector<mjtNum> weight_range_;
  std::vector<mjtNum> test_reset_qpos_;
  std::vector<mjtNum> test_reset_qvel_;
  std::vector<mjtNum> test_reset_act_;
  std::vector<mjtNum> test_reset_qacc_warmstart_;
  std::vector<mjtNum> test_target_qpos_;
  std::vector<mjtNum> test_body_mass_;
  std::vector<mjtNum> test_geom_size0_;
  std::uniform_real_distribution<double> unit_dist_{0.0, 1.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  MyoSuitePoseEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        gymnasium_robotics::MujocoRobotEnv(
            spec.config["base_path"_],
            myosuite::MyoSuiteModelPath(spec.config["base_path"_],
                                        spec.config["model_path"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_],
            spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        normalize_act_(spec.config["normalize_act"_]),
        pose_thd_(spec.config["pose_thd"_]),
        reward_pose_w_(spec.config["reward_pose_w"_]),
        reward_bonus_w_(spec.config["reward_bonus_w"_]),
        reward_act_reg_w_(spec.config["reward_act_reg_w"_]),
        reward_penalty_w_(spec.config["reward_penalty_w"_]),
        reset_type_(spec.config["reset_type"_]),
        target_type_(spec.config["target_type"_]),
        target_qpos_min_(detail::ToMjtVector(spec.config["target_qpos_min"_])),
        target_qpos_max_(detail::ToMjtVector(spec.config["target_qpos_max"_])),
        default_target_qpos_(
            detail::ToMjtVector(spec.config["target_qpos_value"_])),
        current_target_qpos_(default_target_qpos_),
        muscle_actuator_(model_->nu, false),
        weight_range_(detail::ToMjtVector(spec.config["weight_range"_])),
        test_reset_qpos_(detail::ToMjtVector(spec.config["test_reset_qpos"_])),
        test_reset_qvel_(detail::ToMjtVector(spec.config["test_reset_qvel"_])),
        test_reset_act_(detail::ToMjtVector(spec.config["test_reset_act"_])),
        test_reset_qacc_warmstart_(
            detail::ToMjtVector(spec.config["test_reset_qacc_warmstart"_])),
        test_target_qpos_(
            detail::ToMjtVector(spec.config["test_target_qpos"_])),
        test_body_mass_(detail::ToMjtVector(spec.config["test_body_mass"_])),
        test_geom_size0_(detail::ToMjtVector(spec.config["test_geom_size0"_])) {
    ValidateConfig();
    BuildMuscleMask();
    AdjustInitialQposForNormalizedActions();
    CacheTargetSites(spec.config["viz_site_targets"_]);
    CacheWeightRandomization(spec.config["weight_bodyname"_]);
    InitializeRobotEnv();
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    ResetToInitialState();
    RestoreTargetSites();
    RestoreWeightRandomization();
    UpdateTargetQpos();
    ApplyWeightRandomization();
    ApplyResetState();
    CaptureResetState();
    RewardInfo reward = ComputeRewardInfo();
    WriteState(reward, true, 0.0);
  }

  void Step(const Action& action) override {
    const auto* raw = static_cast<const float*>(action["action"_].Data());
    ApplyAction(raw);
    DoSimulation();
    ++elapsed_step_;
    RewardInfo reward = ComputeRewardInfo();
    done_ = reward.done || elapsed_step_ >= max_episode_steps_;
    WriteState(reward, false, reward.dense_reward);
  }

 private:
  void ValidateConfig() {
    if (model_->nq != spec_.config["qpos_dim"_]) {
      throw std::runtime_error("Pose config qpos_dim does not match model.");
    }
    if (model_->nv != spec_.config["qvel_dim"_]) {
      throw std::runtime_error("Pose config qvel_dim does not match model.");
    }
    if (model_->nu != spec_.config["action_dim"_]) {
      throw std::runtime_error("Pose config action_dim does not match model.");
    }
    if (model_->na != spec_.config["act_dim"_]) {
      throw std::runtime_error("Pose config act_dim does not match model.");
    }
    int expected_obs = model_->nq + model_->nv + model_->nq + model_->na;
    if (expected_obs != spec_.config["obs_dim"_]) {
      throw std::runtime_error("Pose config obs_dim does not match model.");
    }
    if (!default_target_qpos_.empty() &&
        static_cast<int>(default_target_qpos_.size()) != model_->nq) {
      throw std::runtime_error("Pose target_qpos_value has wrong length.");
    }
    if (!target_qpos_min_.empty() &&
        static_cast<int>(target_qpos_min_.size()) != model_->nq) {
      throw std::runtime_error("Pose target_qpos_min has wrong length.");
    }
    if (!target_qpos_max_.empty() &&
        static_cast<int>(target_qpos_max_.size()) != model_->nq) {
      throw std::runtime_error("Pose target_qpos_max has wrong length.");
    }
    if (!test_reset_qpos_.empty() &&
        static_cast<int>(test_reset_qpos_.size()) != model_->nq) {
      throw std::runtime_error("Pose test_reset_qpos has wrong length.");
    }
    if (!test_reset_qvel_.empty() &&
        static_cast<int>(test_reset_qvel_.size()) != model_->nv) {
      throw std::runtime_error("Pose test_reset_qvel has wrong length.");
    }
    if (!test_reset_act_.empty() &&
        static_cast<int>(test_reset_act_.size()) != model_->na) {
      throw std::runtime_error("Pose test_reset_act has wrong length.");
    }
    if (!test_reset_qacc_warmstart_.empty() &&
        static_cast<int>(test_reset_qacc_warmstart_.size()) != model_->nv) {
      throw std::runtime_error(
          "Pose test_reset_qacc_warmstart has wrong length.");
    }
    if (!test_target_qpos_.empty() &&
        static_cast<int>(test_target_qpos_.size()) != model_->nq) {
      throw std::runtime_error("Pose test_target_qpos has wrong length.");
    }
  }

  void BuildMuscleMask() {
    for (int i = 0; i < model_->nu; ++i) {
      muscle_actuator_[i] = model_->actuator_dyntype[i] == mjDYN_MUSCLE;
    }
  }

  void AdjustInitialQposForNormalizedActions() {
    if (!normalize_act_) {
      return;
    }
    std::vector<bool> updated(model_->njnt, false);
    for (int actuator_id = 0; actuator_id < model_->nu; ++actuator_id) {
      if (model_->actuator_trntype[actuator_id] != mjTRN_JOINT) {
        continue;
      }
      int joint_id = model_->actuator_trnid[actuator_id * 2];
      if (joint_id < 0 || updated[joint_id]) {
        continue;
      }
      int joint_type = model_->jnt_type[joint_id];
      if (joint_type != mjJNT_HINGE && joint_type != mjJNT_SLIDE) {
        continue;
      }
      int qpos_addr = model_->jnt_qposadr[joint_id];
      mjtNum low = model_->jnt_range[joint_id * 2];
      mjtNum high = model_->jnt_range[joint_id * 2 + 1];
      data_->qpos[qpos_addr] = (low + high) * static_cast<mjtNum>(0.5);
      updated[joint_id] = true;
    }
    mj_forward(model_, data_);
  }

  void CacheTargetSites(const std::vector<std::string>& viz_site_targets) {
    tip_site_ids_.reserve(viz_site_targets.size());
    target_site_ids_.reserve(viz_site_targets.size());
    initial_target_site_pos_.reserve(viz_site_targets.size() * 3);
    for (const auto& site_name : viz_site_targets) {
      int tip_site = mj_name2id(model_, mjOBJ_SITE, site_name.c_str());
      int target_site =
          mj_name2id(model_, mjOBJ_SITE, (site_name + "_target").c_str());
      if (tip_site == -1 || target_site == -1) {
        throw std::runtime_error("Pose target visualization site missing.");
      }
      tip_site_ids_.push_back(tip_site);
      target_site_ids_.push_back(target_site);
      initial_target_site_pos_.insert(initial_target_site_pos_.end(),
                                      model_->site_pos + target_site * 3,
                                      model_->site_pos + target_site * 3 + 3);
    }
  }

  void CacheWeightRandomization(const std::string& weight_bodyname) {
    if (weight_bodyname.empty()) {
      return;
    }
    weight_body_id_ = mj_name2id(model_, mjOBJ_BODY, weight_bodyname.c_str());
    if (weight_body_id_ == -1) {
      throw std::runtime_error("Pose weight body missing.");
    }
    weight_geom_id_ = model_->body_geomadr[weight_body_id_];
    initial_weight_body_mass_ = model_->body_mass[weight_body_id_];
    if (weight_geom_id_ >= 0) {
      initial_weight_geom_size0_ = model_->geom_size[weight_geom_id_ * 3];
    }
  }

  void RestoreTargetSites() {
    for (std::size_t i = 0; i < target_site_ids_.size(); ++i) {
      std::memcpy(model_->site_pos + target_site_ids_[i] * 3,
                  initial_target_site_pos_.data() + i * 3, sizeof(mjtNum) * 3);
    }
  }

  void RestoreWeightRandomization() {
    if (weight_body_id_ == -1) {
      return;
    }
    model_->body_mass[weight_body_id_] = initial_weight_body_mass_;
    if (weight_geom_id_ >= 0) {
      model_->geom_size[weight_geom_id_ * 3] = initial_weight_geom_size0_;
    }
  }

  std::vector<mjtNum> SampleTargetQpos() {
    std::vector<mjtNum> target(model_->nq);
    for (int i = 0; i < model_->nq; ++i) {
      double alpha = unit_dist_(gen_);
      target[i] =
          target_qpos_min_[i] + static_cast<mjtNum>(alpha) *
                                    (target_qpos_max_[i] - target_qpos_min_[i]);
    }
    return target;
  }

  void ApplyTargetVisualization() {
    if (target_site_ids_.empty()) {
      return;
    }
    std::vector<mjtNum> saved_qpos = detail::CopyQpos(model_, data_);
    std::vector<mjtNum> saved_qvel(data_->qvel, data_->qvel + model_->nv);
    detail::RestoreVector(current_target_qpos_, data_->qpos);
    mju_zero(data_->qvel, model_->nv);
    mj_forward(model_, data_);
    for (std::size_t i = 0; i < tip_site_ids_.size(); ++i) {
      detail::CopySitePos(model_, data_, tip_site_ids_[i],
                          model_->site_pos + target_site_ids_[i] * 3);
    }
    detail::RestoreVector(saved_qpos, data_->qpos);
    detail::RestoreVector(saved_qvel, data_->qvel);
    mj_forward(model_, data_);
  }

  void UpdateTargetQpos() {
    if (!test_target_qpos_.empty()) {
      current_target_qpos_ = test_target_qpos_;
    } else if (target_type_ == "generate") {
      current_target_qpos_ = SampleTargetQpos();
    } else if (target_type_ == "fixed") {
      current_target_qpos_ = default_target_qpos_;
    } else {
      throw std::runtime_error("Unsupported Pose target_type.");
    }
    ApplyTargetVisualization();
  }

  void ApplyWeightRandomization() {
    if (weight_body_id_ == -1) {
      return;
    }
    mjtNum weight = initial_weight_body_mass_;
    if (!test_body_mass_.empty()) {
      weight = test_body_mass_[0];
    } else if (weight_range_.size() == 2) {
      double alpha = unit_dist_(gen_);
      weight = weight_range_[0] + static_cast<mjtNum>(alpha) *
                                      (weight_range_[1] - weight_range_[0]);
    }
    model_->body_mass[weight_body_id_] = weight;
    if (weight_geom_id_ >= 0) {
      mjtNum geom_size0 =
          !test_geom_size0_.empty()
              ? test_geom_size0_[0]
              : static_cast<mjtNum>(0.01 +
                                    2.5 * static_cast<double>(weight) / 100.0);
      model_->geom_size[weight_geom_id_ * 3] = geom_size0;
    }
  }

  void ApplyResetState() {
    if (!test_reset_qpos_.empty()) {
      detail::RestoreVector(test_reset_qpos_, data_->qpos);
      detail::RestoreVector(test_reset_qvel_, data_->qvel);
    } else if (reset_type_ == "random") {
      for (int joint_id = 0; joint_id < model_->njnt; ++joint_id) {
        int joint_type = model_->jnt_type[joint_id];
        if (joint_type != mjJNT_HINGE && joint_type != mjJNT_SLIDE) {
          throw std::runtime_error(
              "Pose random reset only supports 1-DoF joints.");
        }
        int qpos_addr = model_->jnt_qposadr[joint_id];
        double alpha = unit_dist_(gen_);
        mjtNum low = model_->jnt_range[joint_id * 2];
        mjtNum high = model_->jnt_range[joint_id * 2 + 1];
        data_->qpos[qpos_addr] =
            low + static_cast<mjtNum>(alpha) * (high - low);
      }
      detail::RestoreVector(initial_qvel_, data_->qvel);
    } else if (reset_type_ != "init") {
      throw std::runtime_error("Unsupported Pose reset_type.");
    }
    mj_forward(model_, data_);
    if (!test_reset_act_.empty()) {
      detail::RestoreVector(test_reset_act_, data_->act);
    }
    if (!test_reset_qacc_warmstart_.empty()) {
      detail::RestoreVector(test_reset_qacc_warmstart_, data_->qacc_warmstart);
    }
  }

  void ApplyAction(const float* raw) {
    for (int i = 0; i < model_->nu; ++i) {
      mjtNum value = detail::ClampNormalized(static_cast<mjtNum>(raw[i]));
      if (normalize_act_ && muscle_actuator_[i] && model_->na != 0) {
        value = detail::MuscleActivation(value);
      } else if (normalize_act_ && model_->na == 0) {
        mjtNum low = model_->actuator_ctrlrange[i * 2];
        mjtNum high = model_->actuator_ctrlrange[i * 2 + 1];
        value = (low + high) * static_cast<mjtNum>(0.5) +
                value * (high - low) * static_cast<mjtNum>(0.5);
      }
      data_->ctrl[i] = value;
    }
  }

  RewardInfo ComputeRewardInfo() const {
    RewardInfo reward;
    std::vector<mjtNum> pose_err(model_->nq);
    for (int i = 0; i < model_->nq; ++i) {
      pose_err[i] = current_target_qpos_[i] - data_->qpos[i];
    }
    reward.pose_dist = detail::VectorNorm(pose_err);
    reward.act_reg = detail::ActReg(model_, data_);
    reward.success = reward.pose_dist < pose_thd_;
    reward.done = reward.pose_dist > detail::kPoseFarThreshold;
    mjtNum bonus = static_cast<mjtNum>(reward.pose_dist < pose_thd_) +
                   static_cast<mjtNum>(reward.pose_dist <
                                       static_cast<mjtNum>(1.5) * pose_thd_);
    mjtNum penalty =
        -static_cast<mjtNum>(reward.pose_dist > detail::kPoseFarThreshold);
    reward.dense_reward =
        -reward_pose_w_ * reward.pose_dist + reward_bonus_w_ * bonus -
        reward_act_reg_w_ * reward.act_reg + reward_penalty_w_ * penalty;
    return reward;
  }

  void WriteState(const RewardInfo& reward, bool reset, mjtNum reward_value) {
    auto state = Allocate();
    state["reward"_] = reward_value;
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      auto obs = state["obs"_];
      mjtNum* buffer = PrepareObservation("obs", &obs);
      for (int i = 0; i < model_->nq; ++i) {
        *(buffer++) = data_->qpos[i];
      }
      for (int i = 0; i < model_->nv; ++i) {
        *(buffer++) = data_->qvel[i] * Dt();
      }
      for (int i = 0; i < model_->nq; ++i) {
        *(buffer++) = current_target_qpos_[i] - data_->qpos[i];
      }
      for (int i = 0; i < model_->na; ++i) {
        *(buffer++) = data_->act[i];
      }
      CommitObservation("obs", &obs, reset);
    }
    state["info:pose_dist"_] = reward.pose_dist;
    state["info:act_reg"_] = reward.act_reg;
    state["info:success"_] = reward.success;
    state["info:target_qpos"_].Assign(current_target_qpos_.data(),
                                      current_target_qpos_.size());
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
    state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
    state["info:qacc0"_].Assign(qacc0_.data(), qacc0_.size());
    state["info:qacc_warmstart0"_].Assign(qacc_warmstart0_.data(),
                                          qacc_warmstart0_.size());
#endif
    mjtNum weight_mass =
        weight_body_id_ == -1 ? 0.0 : model_->body_mass[weight_body_id_];
    mjtNum geom_size0 =
        weight_geom_id_ == -1 ? 0.0 : model_->geom_size[weight_geom_id_ * 3];
    state["info:weight_mass"_] = weight_mass;
    state["info:weight_geom_size0"_] = geom_size0;
  }
};

template <typename EnvSpecT, bool kFromPixels>
class MyoSuiteReachEnvBase : public Env<EnvSpecT>,
                             public gymnasium_robotics::MujocoRobotEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  struct RewardInfo {
    mjtNum dense_reward{0.0};
    mjtNum reach_dist{0.0};
    mjtNum act_reg{0.0};
    bool success{false};
    bool done{false};
  };

  bool normalize_act_;
  mjtNum far_th_;
  mjtNum reward_reach_w_;
  mjtNum reward_bonus_w_;
  mjtNum reward_act_reg_w_;
  mjtNum reward_penalty_w_;
  std::vector<int> tip_site_ids_;
  std::vector<int> target_site_ids_;
  std::vector<mjtNum> initial_target_site_pos_;
  std::vector<mjtNum> target_pos_min_;
  std::vector<mjtNum> target_pos_max_;
  std::vector<mjtNum> current_target_pos_;
  std::vector<bool> muscle_actuator_;
  std::vector<mjtNum> test_reset_qpos_;
  std::vector<mjtNum> test_reset_qvel_;
  std::vector<mjtNum> test_reset_act_;
  std::vector<mjtNum> test_reset_qacc_warmstart_;
  std::vector<mjtNum> test_target_pos_;
  std::uniform_real_distribution<double> unit_dist_{0.0, 1.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  MyoSuiteReachEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        gymnasium_robotics::MujocoRobotEnv(
            spec.config["base_path"_],
            myosuite::MyoSuiteModelPath(spec.config["base_path"_],
                                        spec.config["model_path"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_],
            spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        normalize_act_(spec.config["normalize_act"_]),
        far_th_(spec.config["far_th"_]),
        reward_reach_w_(spec.config["reward_reach_w"_]),
        reward_bonus_w_(spec.config["reward_bonus_w"_]),
        reward_act_reg_w_(spec.config["reward_act_reg_w"_]),
        reward_penalty_w_(spec.config["reward_penalty_w"_]),
        target_pos_min_(detail::ToMjtVector(spec.config["target_pos_min"_])),
        target_pos_max_(detail::ToMjtVector(spec.config["target_pos_max"_])),
        current_target_pos_(target_pos_min_),
        muscle_actuator_(model_->nu, false),
        test_reset_qpos_(detail::ToMjtVector(spec.config["test_reset_qpos"_])),
        test_reset_qvel_(detail::ToMjtVector(spec.config["test_reset_qvel"_])),
        test_reset_act_(detail::ToMjtVector(spec.config["test_reset_act"_])),
        test_reset_qacc_warmstart_(
            detail::ToMjtVector(spec.config["test_reset_qacc_warmstart"_])),
        test_target_pos_(detail::ToMjtVector(spec.config["test_target_pos"_])) {
    ValidateConfig();
    BuildMuscleMask();
    AdjustInitialQposForNormalizedActions();
    CacheTargetSites(spec.config["target_site_names"_]);
    InitializeRobotEnv();
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    ResetToInitialState();
    RestoreTargetSites();
    UpdateTargetSites();
    ApplyResetState();
    CaptureResetState();
    RewardInfo reward = ComputeRewardInfo();
    WriteState(reward, true, 0.0);
  }

  void Step(const Action& action) override {
    const auto* raw = static_cast<const float*>(action["action"_].Data());
    ApplyAction(raw);
    DoSimulation();
    ++elapsed_step_;
    RewardInfo reward = ComputeRewardInfo();
    done_ = reward.done || elapsed_step_ >= max_episode_steps_;
    WriteState(reward, false, reward.dense_reward);
  }

 private:
  void ValidateConfig() {
    if (model_->nq != spec_.config["qpos_dim"_]) {
      throw std::runtime_error("Reach config qpos_dim does not match model.");
    }
    if (model_->nv != spec_.config["qvel_dim"_]) {
      throw std::runtime_error("Reach config qvel_dim does not match model.");
    }
    if (model_->nu != spec_.config["action_dim"_]) {
      throw std::runtime_error("Reach config action_dim does not match model.");
    }
    if (model_->na != spec_.config["act_dim"_]) {
      throw std::runtime_error("Reach config act_dim does not match model.");
    }
    int site_count = spec_.config["target_site_count"_];
    if (static_cast<int>(target_pos_min_.size()) != site_count * 3 ||
        static_cast<int>(target_pos_max_.size()) != site_count * 3) {
      throw std::runtime_error("Reach target position config has wrong size.");
    }
    if (!test_target_pos_.empty() &&
        static_cast<int>(test_target_pos_.size()) != site_count * 3) {
      throw std::runtime_error("Reach test_target_pos has wrong length.");
    }
    if (!test_reset_act_.empty() &&
        static_cast<int>(test_reset_act_.size()) != model_->na) {
      throw std::runtime_error("Reach test_reset_act has wrong length.");
    }
    if (!test_reset_qacc_warmstart_.empty() &&
        static_cast<int>(test_reset_qacc_warmstart_.size()) != model_->nv) {
      throw std::runtime_error(
          "Reach test_reset_qacc_warmstart has wrong length.");
    }
    int expected_obs =
        model_->nq + model_->nv + site_count * 3 + site_count * 3 + model_->na;
    if (expected_obs != spec_.config["obs_dim"_]) {
      throw std::runtime_error("Reach config obs_dim does not match model.");
    }
  }

  void BuildMuscleMask() {
    for (int i = 0; i < model_->nu; ++i) {
      muscle_actuator_[i] = model_->actuator_dyntype[i] == mjDYN_MUSCLE;
    }
  }

  void AdjustInitialQposForNormalizedActions() {
    if (!normalize_act_) {
      return;
    }
    std::vector<bool> updated(model_->njnt, false);
    for (int actuator_id = 0; actuator_id < model_->nu; ++actuator_id) {
      if (model_->actuator_trntype[actuator_id] != mjTRN_JOINT) {
        continue;
      }
      int joint_id = model_->actuator_trnid[actuator_id * 2];
      if (joint_id < 0 || updated[joint_id]) {
        continue;
      }
      int joint_type = model_->jnt_type[joint_id];
      if (joint_type != mjJNT_HINGE && joint_type != mjJNT_SLIDE) {
        continue;
      }
      int qpos_addr = model_->jnt_qposadr[joint_id];
      mjtNum low = model_->jnt_range[joint_id * 2];
      mjtNum high = model_->jnt_range[joint_id * 2 + 1];
      data_->qpos[qpos_addr] = (low + high) * static_cast<mjtNum>(0.5);
      updated[joint_id] = true;
    }
    mj_forward(model_, data_);
  }

  void CacheTargetSites(const std::vector<std::string>& site_names) {
    tip_site_ids_.reserve(site_names.size());
    target_site_ids_.reserve(site_names.size());
    initial_target_site_pos_.reserve(site_names.size() * 3);
    for (const auto& site_name : site_names) {
      int tip_site = mj_name2id(model_, mjOBJ_SITE, site_name.c_str());
      int target_site =
          mj_name2id(model_, mjOBJ_SITE, (site_name + "_target").c_str());
      if (tip_site == -1 || target_site == -1) {
        throw std::runtime_error("Reach target site missing.");
      }
      tip_site_ids_.push_back(tip_site);
      target_site_ids_.push_back(target_site);
      initial_target_site_pos_.insert(initial_target_site_pos_.end(),
                                      model_->site_pos + target_site * 3,
                                      model_->site_pos + target_site * 3 + 3);
    }
  }

  void RestoreTargetSites() {
    for (std::size_t i = 0; i < target_site_ids_.size(); ++i) {
      std::memcpy(model_->site_pos + target_site_ids_[i] * 3,
                  initial_target_site_pos_.data() + i * 3, sizeof(mjtNum) * 3);
    }
  }

  void UpdateTargetSites() {
    if (!test_target_pos_.empty()) {
      current_target_pos_ = test_target_pos_;
    } else {
      current_target_pos_.resize(target_pos_min_.size());
      for (std::size_t i = 0; i < target_pos_min_.size(); ++i) {
        double alpha = unit_dist_(gen_);
        current_target_pos_[i] =
            target_pos_min_[i] + static_cast<mjtNum>(alpha) *
                                     (target_pos_max_[i] - target_pos_min_[i]);
      }
    }
    for (std::size_t i = 0; i < target_site_ids_.size(); ++i) {
      std::memcpy(model_->site_pos + target_site_ids_[i] * 3,
                  current_target_pos_.data() + i * 3, sizeof(mjtNum) * 3);
    }
  }

  void ApplyResetState() {
    if (!test_reset_qpos_.empty()) {
      detail::RestoreVector(test_reset_qpos_, data_->qpos);
      detail::RestoreVector(test_reset_qvel_, data_->qvel);
    }
    mj_forward(model_, data_);
    if (!test_reset_act_.empty()) {
      detail::RestoreVector(test_reset_act_, data_->act);
    }
    if (!test_reset_qacc_warmstart_.empty()) {
      detail::RestoreVector(test_reset_qacc_warmstart_, data_->qacc_warmstart);
    }
  }

  void ApplyAction(const float* raw) {
    for (int i = 0; i < model_->nu; ++i) {
      mjtNum value = detail::ClampNormalized(static_cast<mjtNum>(raw[i]));
      if (normalize_act_ && muscle_actuator_[i] && model_->na != 0) {
        value = detail::MuscleActivation(value);
      } else if (normalize_act_ && model_->na == 0) {
        mjtNum low = model_->actuator_ctrlrange[i * 2];
        mjtNum high = model_->actuator_ctrlrange[i * 2 + 1];
        value = (low + high) * static_cast<mjtNum>(0.5) +
                value * (high - low) * static_cast<mjtNum>(0.5);
      }
      data_->ctrl[i] = value;
    }
  }

  RewardInfo ComputeRewardInfo() const {
    RewardInfo reward;
    std::vector<mjtNum> reach_err;
    reach_err.reserve(tip_site_ids_.size() * 3);
    for (std::size_t i = 0; i < tip_site_ids_.size(); ++i) {
      for (int axis = 0; axis < 3; ++axis) {
        reach_err.push_back(data_->site_xpos[target_site_ids_[i] * 3 + axis] -
                            data_->site_xpos[tip_site_ids_[i] * 3 + axis]);
      }
    }
    reward.reach_dist = detail::VectorNorm(reach_err);
    reward.act_reg = detail::ActReg(model_, data_);
    auto site_count = static_cast<mjtNum>(tip_site_ids_.size());
    mjtNum near_th = site_count * static_cast<mjtNum>(0.0125);
    mjtNum far_th = data_->time > 2.0 * Dt()
                        ? far_th_ * site_count
                        : std::numeric_limits<mjtNum>::infinity();
    reward.success = reward.reach_dist < near_th;
    reward.done = reward.reach_dist > far_th;
    mjtNum bonus = static_cast<mjtNum>(reward.reach_dist < 2.0 * near_th) +
                   static_cast<mjtNum>(reward.reach_dist < near_th);
    mjtNum penalty = -static_cast<mjtNum>(reward.reach_dist > far_th);
    reward.dense_reward =
        -reward_reach_w_ * reward.reach_dist + reward_bonus_w_ * bonus -
        reward_act_reg_w_ * reward.act_reg + reward_penalty_w_ * penalty;
    return reward;
  }

  void WriteState(const RewardInfo& reward, bool reset, mjtNum reward_value) {
    auto state = Allocate();
    state["reward"_] = reward_value;
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      auto obs = state["obs"_];
      mjtNum* buffer = PrepareObservation("obs", &obs);
      for (int i = 0; i < model_->nq; ++i) {
        *(buffer++) = data_->qpos[i];
      }
      for (int i = 0; i < model_->nv; ++i) {
        *(buffer++) = data_->qvel[i] * Dt();
      }
      for (int site_id : tip_site_ids_) {
        for (int axis = 0; axis < 3; ++axis) {
          *(buffer++) = data_->site_xpos[site_id * 3 + axis];
        }
      }
      for (std::size_t i = 0; i < tip_site_ids_.size(); ++i) {
        for (int axis = 0; axis < 3; ++axis) {
          *(buffer++) = data_->site_xpos[target_site_ids_[i] * 3 + axis] -
                        data_->site_xpos[tip_site_ids_[i] * 3 + axis];
        }
      }
      for (int i = 0; i < model_->na; ++i) {
        *(buffer++) = data_->act[i];
      }
      CommitObservation("obs", &obs, reset);
    }
    state["info:reach_dist"_] = reward.reach_dist;
    state["info:act_reg"_] = reward.act_reg;
    state["info:success"_] = reward.success;
    state["info:target_pos"_].Assign(current_target_pos_.data(),
                                     current_target_pos_.size());
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
    state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
    state["info:qacc0"_].Assign(qacc0_.data(), qacc0_.size());
    state["info:qacc_warmstart0"_].Assign(qacc_warmstart0_.data(),
                                          qacc_warmstart0_.size());
#endif
    state["info:time"_] = data_->time;
  }
};

template <typename EnvSpecT, bool kFromPixels>
class MyoSuiteKeyTurnEnvBase : public Env<EnvSpecT>,
                               public gymnasium_robotics::MujocoRobotEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  struct RewardInfo {
    mjtNum dense_reward{0.0};
    mjtNum key_turn{0.0};
    mjtNum iftip_approach_dist{0.0};
    mjtNum thtip_approach_dist{0.0};
    mjtNum act_reg{0.0};
    bool success{false};
    bool done{false};
  };

  bool normalize_act_;
  mjtNum goal_th_;
  mjtNum reward_key_turn_w_;
  mjtNum reward_iftip_approach_w_;
  mjtNum reward_thtip_approach_w_;
  mjtNum reward_act_reg_w_;
  mjtNum reward_bonus_w_;
  mjtNum reward_penalty_w_;
  std::vector<mjtNum> key_init_range_;
  int keyhead_sid_{-1};
  int if_sid_{-1};
  int th_sid_{-1};
  int key_body_id_{-1};
  std::vector<mjtNum> initial_key_body_pos_;
  std::vector<bool> muscle_actuator_;
  std::vector<mjtNum> test_reset_qpos_;
  std::vector<mjtNum> test_reset_qvel_;
  std::vector<mjtNum> test_reset_act_;
  std::vector<mjtNum> test_reset_qacc_warmstart_;
  std::vector<mjtNum> test_key_body_pos_;
  std::uniform_real_distribution<double> unit_dist_{0.0, 1.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  MyoSuiteKeyTurnEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        gymnasium_robotics::MujocoRobotEnv(
            spec.config["base_path"_],
            myosuite::MyoSuiteModelPath(spec.config["base_path"_],
                                        spec.config["model_path"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_],
            spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        normalize_act_(spec.config["normalize_act"_]),
        goal_th_(spec.config["goal_th"_]),
        reward_key_turn_w_(spec.config["reward_key_turn_w"_]),
        reward_iftip_approach_w_(spec.config["reward_iftip_approach_w"_]),
        reward_thtip_approach_w_(spec.config["reward_thtip_approach_w"_]),
        reward_act_reg_w_(spec.config["reward_act_reg_w"_]),
        reward_bonus_w_(spec.config["reward_bonus_w"_]),
        reward_penalty_w_(spec.config["reward_penalty_w"_]),
        key_init_range_(detail::ToMjtVector(spec.config["key_init_range"_])),
        test_reset_qpos_(detail::ToMjtVector(spec.config["test_reset_qpos"_])),
        test_reset_qvel_(detail::ToMjtVector(spec.config["test_reset_qvel"_])),
        test_reset_act_(detail::ToMjtVector(spec.config["test_reset_act"_])),
        test_reset_qacc_warmstart_(
            detail::ToMjtVector(spec.config["test_reset_qacc_warmstart"_])),
        test_key_body_pos_(
            detail::ToMjtVector(spec.config["test_key_body_pos"_])) {
    ValidateConfig();
    CacheSites();
    detail::BuildMuscleMask(model_, &muscle_actuator_);
    detail::AdjustInitialQposForNormalizedActions(model_, data_,
                                                  normalize_act_);
    InitializeRobotEnv();
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    ResetToInitialState();
    detail::RestoreModelBodyPos(model_, key_body_id_, initial_key_body_pos_);
    ApplyResetState();
    CaptureResetState();
    RewardInfo reward = ComputeRewardInfo();
    WriteState(reward, true, 0.0);
  }

  void Step(const Action& action) override {
    const auto* raw = static_cast<const float*>(action["action"_].Data());
    detail::ApplyMyoSuiteAction(model_, data_, muscle_actuator_, normalize_act_,
                                raw);
    DoSimulation();
    ++elapsed_step_;
    RewardInfo reward = ComputeRewardInfo();
    done_ = reward.done || elapsed_step_ >= max_episode_steps_;
    WriteState(reward, false, reward.dense_reward);
  }

 private:
  void ValidateConfig() {
    if (model_->nq != spec_.config["qpos_dim"_]) {
      throw std::runtime_error("KeyTurn config qpos_dim does not match model.");
    }
    if (model_->nv != spec_.config["qvel_dim"_]) {
      throw std::runtime_error("KeyTurn config qvel_dim does not match model.");
    }
    if (model_->nu != spec_.config["action_dim"_]) {
      throw std::runtime_error(
          "KeyTurn config action_dim does not match model.");
    }
    if (model_->na != spec_.config["act_dim"_]) {
      throw std::runtime_error("KeyTurn config act_dim does not match model.");
    }
    int expected_obs = model_->nq + model_->nv + 6 + model_->na;
    if (expected_obs != spec_.config["obs_dim"_]) {
      throw std::runtime_error("KeyTurn config obs_dim does not match model.");
    }
    if (key_init_range_.size() != 2) {
      throw std::runtime_error("KeyTurn key_init_range must have length 2.");
    }
    if (!test_reset_qpos_.empty() &&
        static_cast<int>(test_reset_qpos_.size()) != model_->nq) {
      throw std::runtime_error("KeyTurn test_reset_qpos has wrong length.");
    }
    if (!test_reset_qvel_.empty() &&
        static_cast<int>(test_reset_qvel_.size()) != model_->nv) {
      throw std::runtime_error("KeyTurn test_reset_qvel has wrong length.");
    }
    if (!test_reset_act_.empty() &&
        static_cast<int>(test_reset_act_.size()) != model_->na) {
      throw std::runtime_error("KeyTurn test_reset_act has wrong length.");
    }
    if (!test_reset_qacc_warmstart_.empty() &&
        static_cast<int>(test_reset_qacc_warmstart_.size()) != model_->nv) {
      throw std::runtime_error(
          "KeyTurn test_reset_qacc_warmstart has wrong length.");
    }
    if (!test_key_body_pos_.empty() &&
        static_cast<int>(test_key_body_pos_.size()) != 3) {
      throw std::runtime_error("KeyTurn test_key_body_pos has wrong length.");
    }
  }

  void CacheSites() {
    keyhead_sid_ = mj_name2id(model_, mjOBJ_SITE, "keyhead");
    if_sid_ = mj_name2id(model_, mjOBJ_SITE, "IFtip");
    th_sid_ = mj_name2id(model_, mjOBJ_SITE, "THtip");
    if (keyhead_sid_ == -1 || if_sid_ == -1 || th_sid_ == -1) {
      throw std::runtime_error("KeyTurn site missing.");
    }
    key_body_id_ = model_->site_bodyid[keyhead_sid_];
    detail::CopyModelBodyPos(model_, key_body_id_, &initial_key_body_pos_);
  }

  void ApplyResetState() {
    if (!test_reset_qpos_.empty()) {
      detail::RestoreVector(test_reset_qpos_, data_->qpos);
      detail::RestoreVector(test_reset_qvel_, data_->qvel);
    }
    if (!test_key_body_pos_.empty()) {
      detail::RestoreModelBodyPos(model_, key_body_id_, test_key_body_pos_);
    } else if (key_init_range_[0] != key_init_range_[1]) {
      std::vector<mjtNum> body_pos = initial_key_body_pos_;
      for (int axis = 0; axis < 3; ++axis) {
        body_pos[axis] += static_cast<mjtNum>(unit_dist_(gen_) * 0.02 - 0.01);
      }
      detail::RestoreModelBodyPos(model_, key_body_id_, body_pos);
    }
    if (test_reset_qpos_.empty()) {
      data_->qpos[model_->nq - 1] =
          key_init_range_[0] + static_cast<mjtNum>(unit_dist_(gen_)) *
                                   (key_init_range_[1] - key_init_range_[0]);
    }
    mj_forward(model_, data_);
    if (!test_reset_act_.empty()) {
      detail::RestoreVector(test_reset_act_, data_->act);
    }
    if (!test_reset_qacc_warmstart_.empty()) {
      detail::RestoreVector(test_reset_qacc_warmstart_, data_->qacc_warmstart);
    }
  }

  RewardInfo ComputeRewardInfo() const {
    RewardInfo reward;
    const mjtNum* key_pos3 = data_->site_xpos + keyhead_sid_ * 3;
    const mjtNum* if_pos3 = data_->site_xpos + if_sid_ * 3;
    const mjtNum* th_pos3 = data_->site_xpos + th_sid_ * 3;
    mjtNum if_sq = 0.0;
    mjtNum th_sq = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      mjtNum if_delta = key_pos3[axis] - if_pos3[axis];
      mjtNum th_delta = key_pos3[axis] - th_pos3[axis];
      if_sq += if_delta * if_delta;
      th_sq += th_delta * th_delta;
    }
    reward.key_turn = data_->qpos[model_->nq - 1];
    reward.iftip_approach_dist =
        std::abs(std::sqrt(if_sq) - static_cast<mjtNum>(0.03));
    reward.thtip_approach_dist =
        std::abs(std::sqrt(th_sq) - static_cast<mjtNum>(0.03));
    reward.act_reg = detail::ActReg(model_, data_);
    reward.success = reward.key_turn > goal_th_;
    reward.done = reward.iftip_approach_dist > static_cast<mjtNum>(0.1) ||
                  reward.thtip_approach_dist > static_cast<mjtNum>(0.1);
    mjtNum bonus = static_cast<mjtNum>(reward.key_turn > mjPI / 2.0) +
                   static_cast<mjtNum>(reward.key_turn > mjPI);
    mjtNum penalty = -static_cast<mjtNum>(reward.iftip_approach_dist > 0.05) -
                     static_cast<mjtNum>(reward.thtip_approach_dist > 0.05);
    reward.dense_reward =
        reward_key_turn_w_ * reward.key_turn -
        reward_iftip_approach_w_ * reward.iftip_approach_dist -
        reward_thtip_approach_w_ * reward.thtip_approach_dist -
        reward_act_reg_w_ * reward.act_reg + reward_bonus_w_ * bonus +
        reward_penalty_w_ * penalty;
    return reward;
  }

  void WriteState(const RewardInfo& reward, bool reset, mjtNum reward_value) {
    auto state = Allocate();
    state["reward"_] = reward_value;
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      auto obs = state["obs"_];
      mjtNum* buffer = PrepareObservation("obs", &obs);
      for (int i = 0; i < model_->nq - 1; ++i) {
        *(buffer++) = data_->qpos[i];
      }
      for (int i = 0; i < model_->nv - 1; ++i) {
        *(buffer++) = data_->qvel[i] * Dt();
      }
      *(buffer++) = data_->qpos[model_->nq - 1];
      *(buffer++) = data_->qvel[model_->nv - 1] * Dt();
      for (int axis = 0; axis < 3; ++axis) {
        *(buffer++) = data_->site_xpos[keyhead_sid_ * 3 + axis] -
                      data_->site_xpos[if_sid_ * 3 + axis];
      }
      for (int axis = 0; axis < 3; ++axis) {
        *(buffer++) = data_->site_xpos[keyhead_sid_ * 3 + axis] -
                      data_->site_xpos[th_sid_ * 3 + axis];
      }
      for (int i = 0; i < model_->na; ++i) {
        *(buffer++) = data_->act[i];
      }
      CommitObservation("obs", &obs, reset);
    }
    state["info:key_turn"_] = reward.key_turn;
    state["info:iftip_approach_dist"_] = reward.iftip_approach_dist;
    state["info:thtip_approach_dist"_] = reward.thtip_approach_dist;
    state["info:success"_] = reward.success;
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
    state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
    state["info:qacc0"_].Assign(qacc0_.data(), qacc0_.size());
    state["info:qacc_warmstart0"_].Assign(qacc_warmstart0_.data(),
                                          qacc_warmstart0_.size());
#endif
    state["info:key_body_pos"_].Assign(model_->body_pos + key_body_id_ * 3, 3);
  }
};

template <typename EnvSpecT, bool kFromPixels>
class MyoSuiteObjHoldEnvBase : public Env<EnvSpecT>,
                               public gymnasium_robotics::MujocoRobotEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  struct RewardInfo {
    mjtNum dense_reward{0.0};
    mjtNum goal_dist{0.0};
    bool success{false};
    bool done{false};
  };

  bool normalize_act_;
  bool randomize_on_reset_;
  mjtNum reward_goal_dist_w_;
  mjtNum reward_bonus_w_;
  mjtNum reward_penalty_w_;
  int object_sid_{-1};
  int goal_sid_{-1};
  int object_geom_id_{-1};
  std::vector<mjtNum> initial_object_pos_;
  std::vector<mjtNum> initial_goal_pos_;
  std::vector<mjtNum> initial_goal_size_;
  std::vector<mjtNum> initial_object_geom_size_;
  std::vector<bool> muscle_actuator_;
  std::vector<mjtNum> test_reset_qpos_;
  std::vector<mjtNum> test_reset_qvel_;
  std::vector<mjtNum> test_reset_act_;
  std::vector<mjtNum> test_reset_qacc_warmstart_;
  std::vector<mjtNum> test_goal_pos_;
  std::vector<mjtNum> test_object_geom_size_;
  std::uniform_real_distribution<double> unit_dist_{0.0, 1.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  MyoSuiteObjHoldEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        gymnasium_robotics::MujocoRobotEnv(
            spec.config["base_path"_],
            myosuite::MyoSuiteModelPath(spec.config["base_path"_],
                                        spec.config["model_path"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_],
            spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        normalize_act_(spec.config["normalize_act"_]),
        randomize_on_reset_(spec.config["randomize_on_reset"_]),
        reward_goal_dist_w_(spec.config["reward_goal_dist_w"_]),
        reward_bonus_w_(spec.config["reward_bonus_w"_]),
        reward_penalty_w_(spec.config["reward_penalty_w"_]),
        test_reset_qpos_(detail::ToMjtVector(spec.config["test_reset_qpos"_])),
        test_reset_qvel_(detail::ToMjtVector(spec.config["test_reset_qvel"_])),
        test_reset_act_(detail::ToMjtVector(spec.config["test_reset_act"_])),
        test_reset_qacc_warmstart_(
            detail::ToMjtVector(spec.config["test_reset_qacc_warmstart"_])),
        test_goal_pos_(detail::ToMjtVector(spec.config["test_goal_pos"_])),
        test_object_geom_size_(
            detail::ToMjtVector(spec.config["test_object_geom_size"_])) {
    ValidateConfig();
    CacheObjects();
    detail::BuildMuscleMask(model_, &muscle_actuator_);
    detail::AdjustInitialQposForNormalizedActions(model_, data_,
                                                  normalize_act_);
    InitializeRobotEnv();
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    ResetToInitialState();
    RestoreModelState();
    ApplyResetState();
    CaptureResetState();
    RewardInfo reward = ComputeRewardInfo();
    WriteState(reward, true, 0.0);
  }

  void Step(const Action& action) override {
    const auto* raw = static_cast<const float*>(action["action"_].Data());
    detail::ApplyMyoSuiteAction(model_, data_, muscle_actuator_, normalize_act_,
                                raw);
    DoSimulation();
    ++elapsed_step_;
    RewardInfo reward = ComputeRewardInfo();
    done_ = reward.done || elapsed_step_ >= max_episode_steps_;
    WriteState(reward, false, reward.dense_reward);
  }

 private:
  void ValidateConfig() {
    if (model_->nq != spec_.config["qpos_dim"_] ||
        model_->nv != spec_.config["qvel_dim"_] ||
        model_->nu != spec_.config["action_dim"_] ||
        model_->na != spec_.config["act_dim"_]) {
      throw std::runtime_error("ObjHold config dims do not match model.");
    }
    int expected_obs = (model_->nq - 7) + (model_->nv - 6) + 6 + model_->na;
    if (expected_obs != spec_.config["obs_dim"_]) {
      throw std::runtime_error("ObjHold config obs_dim does not match model.");
    }
    if (!test_goal_pos_.empty() &&
        static_cast<int>(test_goal_pos_.size()) != 3) {
      throw std::runtime_error("ObjHold test_goal_pos has wrong length.");
    }
    if (!test_object_geom_size_.empty() &&
        static_cast<int>(test_object_geom_size_.size()) != 3) {
      throw std::runtime_error(
          "ObjHold test_object_geom_size has wrong length.");
    }
  }

  void CacheObjects() {
    object_sid_ = mj_name2id(model_, mjOBJ_SITE, "object");
    goal_sid_ = mj_name2id(model_, mjOBJ_SITE, "goal");
    object_geom_id_ = mj_name2id(model_, mjOBJ_GEOM, "object");
    if (object_sid_ == -1 || goal_sid_ == -1 || object_geom_id_ == -1) {
      throw std::runtime_error("ObjHold object/goal ids missing.");
    }
    initial_object_pos_.assign(data_->site_xpos + object_sid_ * 3,
                               data_->site_xpos + object_sid_ * 3 + 3);
    detail::CopyModelSitePos(model_, goal_sid_, &initial_goal_pos_);
    detail::CopyModelSiteSize(model_, goal_sid_, &initial_goal_size_);
    detail::CopyModelGeomSize(model_, object_geom_id_,
                              &initial_object_geom_size_);
  }

  void RestoreModelState() {
    detail::RestoreModelSitePos(model_, goal_sid_, initial_goal_pos_);
    detail::RestoreModelSiteSize(model_, goal_sid_, initial_goal_size_);
    detail::RestoreModelGeomSize(model_, object_geom_id_,
                                 initial_object_geom_size_);
  }

  void ApplyResetState() {
    if (!test_reset_qpos_.empty()) {
      detail::RestoreVector(test_reset_qpos_, data_->qpos);
      detail::RestoreVector(test_reset_qvel_, data_->qvel);
    }
    if (!test_goal_pos_.empty()) {
      detail::RestoreModelSitePos(model_, goal_sid_, test_goal_pos_);
    } else if (randomize_on_reset_) {
      std::vector<mjtNum> goal_pos = initial_object_pos_;
      for (int axis = 0; axis < 3; ++axis) {
        goal_pos[axis] += static_cast<mjtNum>(unit_dist_(gen_) * 0.06 - 0.03);
      }
      detail::RestoreModelSitePos(model_, goal_sid_, goal_pos);
    }
    if (!test_object_geom_size_.empty()) {
      detail::RestoreModelGeomSize(model_, object_geom_id_,
                                   test_object_geom_size_);
      detail::RestoreModelSiteSize(model_, goal_sid_, test_object_geom_size_);
    } else if (randomize_on_reset_) {
      std::vector<mjtNum> size(3);
      for (int axis = 0; axis < 3; ++axis) {
        size[axis] = static_cast<mjtNum>(0.02 + unit_dist_(gen_) * 0.01);
      }
      detail::RestoreModelGeomSize(model_, object_geom_id_, size);
      detail::RestoreModelSiteSize(model_, goal_sid_, size);
    }
    mj_forward(model_, data_);
    if (!test_reset_act_.empty()) {
      detail::RestoreVector(test_reset_act_, data_->act);
    }
    if (!test_reset_qacc_warmstart_.empty()) {
      detail::RestoreVector(test_reset_qacc_warmstart_, data_->qacc_warmstart);
    }
  }

  RewardInfo ComputeRewardInfo() const {
    RewardInfo reward;
    mjtNum goal_sq = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      mjtNum delta = data_->site_xpos[goal_sid_ * 3 + axis] -
                     data_->site_xpos[object_sid_ * 3 + axis];
      goal_sq += delta * delta;
    }
    reward.goal_dist = std::sqrt(goal_sq);
    reward.success = reward.goal_dist < static_cast<mjtNum>(0.01);
    reward.done = reward.goal_dist > static_cast<mjtNum>(0.3);
    mjtNum bonus = static_cast<mjtNum>(reward.goal_dist < 0.02) +
                   static_cast<mjtNum>(reward.goal_dist < 0.01);
    mjtNum penalty = -static_cast<mjtNum>(reward.done);
    reward.dense_reward = -reward_goal_dist_w_ * reward.goal_dist +
                          reward_bonus_w_ * bonus + reward_penalty_w_ * penalty;
    return reward;
  }

  void WriteState(const RewardInfo& reward, bool reset, mjtNum reward_value) {
    auto state = Allocate();
    state["reward"_] = reward_value;
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      auto obs = state["obs"_];
      mjtNum* buffer = PrepareObservation("obs", &obs);
      for (int i = 0; i < model_->nq - 7; ++i) {
        *(buffer++) = data_->qpos[i];
      }
      for (int i = 0; i < model_->nv - 6; ++i) {
        *(buffer++) = data_->qvel[i] * Dt();
      }
      for (int axis = 0; axis < 3; ++axis) {
        *(buffer++) = data_->site_xpos[object_sid_ * 3 + axis];
      }
      for (int axis = 0; axis < 3; ++axis) {
        *(buffer++) = data_->site_xpos[goal_sid_ * 3 + axis] -
                      data_->site_xpos[object_sid_ * 3 + axis];
      }
      for (int i = 0; i < model_->na; ++i) {
        *(buffer++) = data_->act[i];
      }
      CommitObservation("obs", &obs, reset);
    }
    state["info:goal_dist"_] = reward.goal_dist;
    state["info:success"_] = reward.success;
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
    state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
    state["info:qacc0"_].Assign(qacc0_.data(), qacc0_.size());
    state["info:qacc_warmstart0"_].Assign(qacc_warmstart0_.data(),
                                          qacc_warmstart0_.size());
#endif
    state["info:goal_pos"_].Assign(model_->site_pos + goal_sid_ * 3, 3);
    state["info:object_geom_size"_].Assign(
        model_->geom_size + object_geom_id_ * 3, 3);
  }
};

template <typename EnvSpecT, bool kFromPixels>
class MyoSuiteTorsoEnvBase : public Env<EnvSpecT>,
                             public gymnasium_robotics::MujocoRobotEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::spec_;

  struct RewardInfo {
    mjtNum dense_reward{0.0};
    mjtNum pose_dist{0.0};
    mjtNum act_reg{0.0};
    bool success{false};
    bool done{false};
  };

  bool normalize_act_;
  int pose_dim_;
  mjtNum pose_thd_;
  mjtNum reward_pose_w_;
  mjtNum reward_bonus_w_;
  mjtNum reward_act_reg_w_;
  mjtNum reward_penalty_w_;
  std::vector<mjtNum> target_qpos_value_;
  std::vector<bool> muscle_actuator_;
  std::vector<mjtNum> test_reset_qpos_;
  std::vector<mjtNum> test_reset_qvel_;
  std::vector<mjtNum> test_reset_act_;
  std::vector<mjtNum> test_reset_qacc_warmstart_;

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  MyoSuiteTorsoEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        gymnasium_robotics::MujocoRobotEnv(
            spec.config["base_path"_],
            myosuite::MyoSuiteModelPath(spec.config["base_path"_],
                                        spec.config["model_path"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_],
            spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        normalize_act_(spec.config["normalize_act"_]),
        pose_dim_(spec.config["pose_dim"_]),
        pose_thd_(spec.config["pose_thd"_]),
        reward_pose_w_(spec.config["reward_pose_w"_]),
        reward_bonus_w_(spec.config["reward_bonus_w"_]),
        reward_act_reg_w_(spec.config["reward_act_reg_w"_]),
        reward_penalty_w_(spec.config["reward_penalty_w"_]),
        target_qpos_value_(
            detail::ToMjtVector(spec.config["target_qpos_value"_])),
        test_reset_qpos_(detail::ToMjtVector(spec.config["test_reset_qpos"_])),
        test_reset_qvel_(detail::ToMjtVector(spec.config["test_reset_qvel"_])),
        test_reset_act_(detail::ToMjtVector(spec.config["test_reset_act"_])),
        test_reset_qacc_warmstart_(
            detail::ToMjtVector(spec.config["test_reset_qacc_warmstart"_])) {
    ValidateConfig();
    detail::BuildMuscleMask(model_, &muscle_actuator_);
    detail::AdjustInitialQposForNormalizedActions(model_, data_,
                                                  normalize_act_);
    InitializeRobotEnv();
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    ResetToInitialState();
    if (!test_reset_qpos_.empty()) {
      detail::RestoreVector(test_reset_qpos_, data_->qpos);
      detail::RestoreVector(test_reset_qvel_, data_->qvel);
      mj_forward(model_, data_);
      if (!test_reset_act_.empty()) {
        detail::RestoreVector(test_reset_act_, data_->act);
      }
      if (!test_reset_qacc_warmstart_.empty()) {
        detail::RestoreVector(test_reset_qacc_warmstart_,
                              data_->qacc_warmstart);
      }
    }
    CaptureResetState();
    RewardInfo reward = ComputeRewardInfo();
    WriteState(reward, true, 0.0);
  }

  void Step(const Action& action) override {
    const auto* raw = static_cast<const float*>(action["action"_].Data());
    detail::ApplyMyoSuiteAction(model_, data_, muscle_actuator_, normalize_act_,
                                raw);
    DoSimulation();
    ++elapsed_step_;
    RewardInfo reward = ComputeRewardInfo();
    done_ = reward.done || elapsed_step_ >= max_episode_steps_;
    WriteState(reward, false, reward.dense_reward);
  }

 private:
  void ValidateConfig() {
    if (model_->nq != spec_.config["qpos_dim"_] ||
        model_->nv != spec_.config["qvel_dim"_] ||
        model_->nu != spec_.config["action_dim"_] ||
        model_->na != spec_.config["act_dim"_]) {
      throw std::runtime_error("Torso config dims do not match model.");
    }
    if (static_cast<int>(target_qpos_value_.size()) != pose_dim_) {
      throw std::runtime_error("Torso target_qpos_value has wrong length.");
    }
    int expected_obs = model_->nq + model_->nv + pose_dim_ + model_->na;
    if (expected_obs != spec_.config["obs_dim"_]) {
      throw std::runtime_error("Torso config obs_dim does not match model.");
    }
  }

  RewardInfo ComputeRewardInfo() const {
    RewardInfo reward;
    std::vector<mjtNum> pose_err(pose_dim_);
    for (int i = 0; i < pose_dim_; ++i) {
      pose_err[i] = target_qpos_value_[i] - data_->qpos[i];
    }
    reward.pose_dist = detail::VectorNorm(pose_err);
    reward.act_reg = detail::ActReg(model_, data_);
    reward.success = reward.pose_dist < pose_thd_;
    reward.done = reward.pose_dist > mjPI;
    mjtNum bonus = static_cast<mjtNum>(reward.pose_dist < pose_thd_) +
                   static_cast<mjtNum>(reward.pose_dist < 1.5 * pose_thd_);
    mjtNum penalty = -static_cast<mjtNum>(reward.done);
    reward.dense_reward = -reward_pose_w_ * reward.pose_dist -
                          reward_act_reg_w_ * reward.act_reg +
                          reward_bonus_w_ * bonus + reward_penalty_w_ * penalty;
    return reward;
  }

  void WriteState(const RewardInfo& reward, bool reset, mjtNum reward_value) {
    auto state = Allocate();
    state["reward"_] = reward_value;
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      auto obs = state["obs"_];
      mjtNum* buffer = PrepareObservation("obs", &obs);
      for (int i = 0; i < model_->nq; ++i) {
        *(buffer++) = data_->qpos[i];
      }
      for (int i = 0; i < model_->nv; ++i) {
        *(buffer++) = data_->qvel[i] * Dt();
      }
      for (int i = 0; i < pose_dim_; ++i) {
        *(buffer++) = target_qpos_value_[i] - data_->qpos[i];
      }
      for (int i = 0; i < model_->na; ++i) {
        *(buffer++) = data_->act[i];
      }
      CommitObservation("obs", &obs, reset);
    }
    state["info:pose_dist"_] = reward.pose_dist;
    state["info:success"_] = reward.success;
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
    state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
    state["info:qacc0"_].Assign(qacc0_.data(), qacc0_.size());
    state["info:qacc_warmstart0"_].Assign(qacc_warmstart0_.data(),
                                          qacc_warmstart0_.size());
#endif
    state["info:target_qpos"_].Assign(target_qpos_value_.data(), pose_dim_);
  }
};

template <typename EnvSpecT, bool kFromPixels>
class MyoSuitePenTwirlEnvBase : public Env<EnvSpecT>,
                                public gymnasium_robotics::MujocoRobotEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  struct RewardInfo {
    mjtNum dense_reward{0.0};
    mjtNum pos_align{0.0};
    mjtNum rot_align{0.0};
    mjtNum act_reg{0.0};
    bool success{false};
    bool done{false};
  };

  bool normalize_act_;
  bool randomize_target_;
  mjtNum reward_pos_align_w_;
  mjtNum reward_rot_align_w_;
  mjtNum reward_act_reg_w_;
  mjtNum reward_drop_w_;
  mjtNum reward_bonus_w_;
  int target_body_id_{-1};
  int obj_body_id_{-1};
  int eps_ball_sid_{-1};
  int obj_t_sid_{-1};
  int obj_b_sid_{-1};
  int tar_t_sid_{-1};
  int tar_b_sid_{-1};
  mjtNum pen_length_{0.0};
  mjtNum tar_length_{0.0};
  std::vector<mjtNum> initial_target_body_quat_;
  std::vector<bool> muscle_actuator_;
  std::vector<mjtNum> test_reset_qpos_;
  std::vector<mjtNum> test_reset_qvel_;
  std::vector<mjtNum> test_reset_act_;
  std::vector<mjtNum> test_reset_qacc_warmstart_;
  std::vector<mjtNum> test_target_body_quat_;
  std::uniform_real_distribution<double> unit_dist_{0.0, 1.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  MyoSuitePenTwirlEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        gymnasium_robotics::MujocoRobotEnv(
            spec.config["base_path"_],
            myosuite::MyoSuiteModelPath(spec.config["base_path"_],
                                        spec.config["model_path"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_],
            spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        normalize_act_(spec.config["normalize_act"_]),
        randomize_target_(spec.config["randomize_target"_]),
        reward_pos_align_w_(spec.config["reward_pos_align_w"_]),
        reward_rot_align_w_(spec.config["reward_rot_align_w"_]),
        reward_act_reg_w_(spec.config["reward_act_reg_w"_]),
        reward_drop_w_(spec.config["reward_drop_w"_]),
        reward_bonus_w_(spec.config["reward_bonus_w"_]),
        test_reset_qpos_(detail::ToMjtVector(spec.config["test_reset_qpos"_])),
        test_reset_qvel_(detail::ToMjtVector(spec.config["test_reset_qvel"_])),
        test_reset_act_(detail::ToMjtVector(spec.config["test_reset_act"_])),
        test_reset_qacc_warmstart_(
            detail::ToMjtVector(spec.config["test_reset_qacc_warmstart"_])),
        test_target_body_quat_(
            detail::ToMjtVector(spec.config["test_target_body_quat"_])) {
    ValidateConfig();
    CacheObjects();
    detail::BuildMuscleMask(model_, &muscle_actuator_);
    detail::AdjustInitialQposForNormalizedActions(model_, data_,
                                                  normalize_act_);
    InitializeRobotEnv();
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    ResetToInitialState();
    detail::RestoreModelBodyQuat(model_, target_body_id_,
                                 initial_target_body_quat_);
    ApplyResetState();
    CaptureResetState();
    RewardInfo reward = ComputeRewardInfo();
    WriteState(reward, true, 0.0);
  }

  void Step(const Action& action) override {
    const auto* raw = static_cast<const float*>(action["action"_].Data());
    detail::ApplyMyoSuiteAction(model_, data_, muscle_actuator_, normalize_act_,
                                raw);
    DoSimulation();
    ++elapsed_step_;
    RewardInfo reward = ComputeRewardInfo();
    done_ = reward.done || elapsed_step_ >= max_episode_steps_;
    WriteState(reward, false, reward.dense_reward);
  }

 private:
  void ValidateConfig() {
    if (model_->nq != spec_.config["qpos_dim"_] ||
        model_->nv != spec_.config["qvel_dim"_] ||
        model_->nu != spec_.config["action_dim"_] ||
        model_->na != spec_.config["act_dim"_]) {
      throw std::runtime_error("PenTwirl config dims do not match model.");
    }
    int expected_obs = (model_->nq - 6) + 21 + model_->na;
    if (expected_obs != spec_.config["obs_dim"_]) {
      throw std::runtime_error("PenTwirl config obs_dim does not match model.");
    }
    if (!test_target_body_quat_.empty() &&
        static_cast<int>(test_target_body_quat_.size()) != 4) {
      throw std::runtime_error(
          "PenTwirl test_target_body_quat has wrong length.");
    }
  }

  void CacheObjects() {
    target_body_id_ = mj_name2id(model_, mjOBJ_BODY, "target");
    obj_body_id_ = mj_name2id(model_, mjOBJ_BODY, "Object");
    eps_ball_sid_ = mj_name2id(model_, mjOBJ_SITE, "eps_ball");
    obj_t_sid_ = mj_name2id(model_, mjOBJ_SITE, "object_top");
    obj_b_sid_ = mj_name2id(model_, mjOBJ_SITE, "object_bottom");
    tar_t_sid_ = mj_name2id(model_, mjOBJ_SITE, "target_top");
    tar_b_sid_ = mj_name2id(model_, mjOBJ_SITE, "target_bottom");
    if (target_body_id_ == -1 || obj_body_id_ == -1 || eps_ball_sid_ == -1 ||
        obj_t_sid_ == -1 || obj_b_sid_ == -1 || tar_t_sid_ == -1 ||
        tar_b_sid_ == -1) {
      throw std::runtime_error("PenTwirl ids missing.");
    }
    std::vector<mjtNum> obj_top(model_->site_pos + obj_t_sid_ * 3,
                                model_->site_pos + obj_t_sid_ * 3 + 3);
    std::vector<mjtNum> obj_bottom(model_->site_pos + obj_b_sid_ * 3,
                                   model_->site_pos + obj_b_sid_ * 3 + 3);
    std::vector<mjtNum> tar_top(model_->site_pos + tar_t_sid_ * 3,
                                model_->site_pos + tar_t_sid_ * 3 + 3);
    std::vector<mjtNum> tar_bottom(model_->site_pos + tar_b_sid_ * 3,
                                   model_->site_pos + tar_b_sid_ * 3 + 3);
    for (int axis = 0; axis < 3; ++axis) {
      obj_top[axis] -= obj_bottom[axis];
      tar_top[axis] -= tar_bottom[axis];
    }
    pen_length_ = detail::VectorNorm(obj_top);
    tar_length_ = detail::VectorNorm(tar_top);
    detail::CopyModelBodyQuat(model_, target_body_id_,
                              &initial_target_body_quat_);
  }

  void ApplyResetState() {
    if (!test_reset_qpos_.empty()) {
      detail::RestoreVector(test_reset_qpos_, data_->qpos);
      detail::RestoreVector(test_reset_qvel_, data_->qvel);
    }
    if (!test_target_body_quat_.empty()) {
      detail::RestoreModelBodyQuat(model_, target_body_id_,
                                   test_target_body_quat_);
    } else if (randomize_target_) {
      double x = unit_dist_(gen_) * 2.0 - 1.0;
      double y = unit_dist_(gen_) * 2.0 - 1.0;
      double cx = std::cos(x * 0.5);
      double sx = std::sin(x * 0.5);
      double cy = std::cos(y * 0.5);
      double sy = std::sin(y * 0.5);
      std::vector<mjtNum> quat = {
          static_cast<mjtNum>(cy * cx), static_cast<mjtNum>(cy * sx),
          static_cast<mjtNum>(sy * cx), static_cast<mjtNum>(-sy * sx)};
      detail::RestoreModelBodyQuat(model_, target_body_id_, quat);
    }
    mj_forward(model_, data_);
    if (!test_reset_act_.empty()) {
      detail::RestoreVector(test_reset_act_, data_->act);
    }
    if (!test_reset_qacc_warmstart_.empty()) {
      detail::RestoreVector(test_reset_qacc_warmstart_, data_->qacc_warmstart);
    }
  }

  RewardInfo ComputeRewardInfo() const {
    RewardInfo reward;
    std::vector<mjtNum> obj_pos(3);
    std::vector<mjtNum> obj_des_pos(3);
    std::vector<mjtNum> obj_rot(3);
    std::vector<mjtNum> obj_des_rot(3);
    for (int axis = 0; axis < 3; ++axis) {
      obj_pos[axis] = data_->xpos[obj_body_id_ * 3 + axis];
      obj_des_pos[axis] = data_->site_xpos[eps_ball_sid_ * 3 + axis];
      obj_rot[axis] = (data_->site_xpos[obj_t_sid_ * 3 + axis] -
                       data_->site_xpos[obj_b_sid_ * 3 + axis]) /
                      pen_length_;
      obj_des_rot[axis] = (data_->site_xpos[tar_t_sid_ * 3 + axis] -
                           data_->site_xpos[tar_b_sid_ * 3 + axis]) /
                          tar_length_;
    }
    for (int axis = 0; axis < 3; ++axis) {
      obj_pos[axis] -= obj_des_pos[axis];
    }
    reward.pos_align = detail::VectorNorm(obj_pos);
    reward.rot_align = detail::CosineSimilarity(obj_rot, obj_des_rot);
    reward.act_reg = detail::ActReg(model_, data_);
    reward.done = reward.pos_align > static_cast<mjtNum>(0.075);
    reward.success =
        reward.rot_align > static_cast<mjtNum>(0.95) && !reward.done;
    mjtNum bonus = static_cast<mjtNum>(reward.rot_align > 0.9 &&
                                       reward.pos_align < 0.075) +
                   static_cast<mjtNum>(5.0 * (reward.rot_align > 0.95 &&
                                              reward.pos_align < 0.075));
    mjtNum drop = -static_cast<mjtNum>(reward.done);
    reward.dense_reward = -reward_pos_align_w_ * reward.pos_align +
                          reward_rot_align_w_ * reward.rot_align -
                          reward_act_reg_w_ * reward.act_reg +
                          reward_drop_w_ * drop + reward_bonus_w_ * bonus;
    return reward;
  }

  void WriteState(const RewardInfo& reward, bool reset, mjtNum reward_value) {
    auto state = Allocate();
    state["reward"_] = reward_value;
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      auto obs = state["obs"_];
      mjtNum* buffer = PrepareObservation("obs", &obs);
      for (int i = 0; i < model_->nq - 6; ++i) {
        *(buffer++) = data_->qpos[i];
      }
      for (int axis = 0; axis < 3; ++axis) {
        *(buffer++) = data_->xpos[obj_body_id_ * 3 + axis];
      }
      for (int i = model_->nv - 6; i < model_->nv; ++i) {
        *(buffer++) = data_->qvel[i] * Dt();
      }
      for (int axis = 0; axis < 3; ++axis) {
        *(buffer++) = (data_->site_xpos[obj_t_sid_ * 3 + axis] -
                       data_->site_xpos[obj_b_sid_ * 3 + axis]) /
                      pen_length_;
      }
      for (int axis = 0; axis < 3; ++axis) {
        *(buffer++) = (data_->site_xpos[tar_t_sid_ * 3 + axis] -
                       data_->site_xpos[tar_b_sid_ * 3 + axis]) /
                      tar_length_;
      }
      for (int axis = 0; axis < 3; ++axis) {
        *(buffer++) = data_->xpos[obj_body_id_ * 3 + axis] -
                      data_->site_xpos[eps_ball_sid_ * 3 + axis];
      }
      for (int axis = 0; axis < 3; ++axis) {
        *(buffer++) = (data_->site_xpos[obj_t_sid_ * 3 + axis] -
                       data_->site_xpos[obj_b_sid_ * 3 + axis]) /
                          pen_length_ -
                      (data_->site_xpos[tar_t_sid_ * 3 + axis] -
                       data_->site_xpos[tar_b_sid_ * 3 + axis]) /
                          tar_length_;
      }
      for (int i = 0; i < model_->na; ++i) {
        *(buffer++) = data_->act[i];
      }
      CommitObservation("obs", &obs, reset);
    }
    state["info:pos_align"_] = reward.pos_align;
    state["info:rot_align"_] = reward.rot_align;
    state["info:success"_] = reward.success;
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
    state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
    state["info:qacc0"_].Assign(qacc0_.data(), qacc0_.size());
    state["info:qacc_warmstart0"_].Assign(qacc_warmstart0_.data(),
                                          qacc_warmstart0_.size());
#endif
    state["info:target_body_quat"_].Assign(
        model_->body_quat + target_body_id_ * 4, 4);
  }
};

template <typename Spec>
using PosePixelEnvBase = MyoSuitePoseEnvBase<Spec, true>;

template <typename Spec>
using ReachPixelEnvBase = MyoSuiteReachEnvBase<Spec, true>;

template <typename Spec>
using KeyTurnPixelEnvBase = MyoSuiteKeyTurnEnvBase<Spec, true>;

template <typename Spec>
using ObjHoldPixelEnvBase = MyoSuiteObjHoldEnvBase<Spec, true>;

template <typename Spec>
using TorsoPixelEnvBase = MyoSuiteTorsoEnvBase<Spec, true>;

template <typename Spec>
using PenTwirlPixelEnvBase = MyoSuitePenTwirlEnvBase<Spec, true>;

using MyoSuitePoseEnv = MyoSuitePoseEnvBase<MyoSuitePoseEnvSpec, false>;
using MyoSuitePosePixelEnv = PosePixelEnvBase<MyoSuitePosePixelEnvSpec>;
using MyoSuitePoseEnvPool = AsyncEnvPool<MyoSuitePoseEnv>;
using MyoSuitePosePixelEnvPool = AsyncEnvPool<MyoSuitePosePixelEnv>;

using MyoSuiteReachEnv = MyoSuiteReachEnvBase<MyoSuiteReachEnvSpec, false>;
using MyoSuiteReachPixelEnv = ReachPixelEnvBase<MyoSuiteReachPixelEnvSpec>;
using MyoSuiteReachEnvPool = AsyncEnvPool<MyoSuiteReachEnv>;
using MyoSuiteReachPixelEnvPool = AsyncEnvPool<MyoSuiteReachPixelEnv>;

using MyoSuiteKeyTurnEnv =
    // NOLINTNEXTLINE(whitespace/indent_namespace)
    MyoSuiteKeyTurnEnvBase<MyoSuiteKeyTurnEnvSpec, false>;
using MyoSuiteKeyTurnPixelEnv =
    // NOLINTNEXTLINE(whitespace/indent_namespace)
    KeyTurnPixelEnvBase<MyoSuiteKeyTurnPixelEnvSpec>;
using MyoSuiteKeyTurnEnvPool = AsyncEnvPool<MyoSuiteKeyTurnEnv>;
using MyoSuiteKeyTurnPixelEnvPool = AsyncEnvPool<MyoSuiteKeyTurnPixelEnv>;

using MyoSuiteObjHoldEnv =
    // NOLINTNEXTLINE(whitespace/indent_namespace)
    MyoSuiteObjHoldEnvBase<MyoSuiteObjHoldEnvSpec, false>;
using MyoSuiteObjHoldPixelEnv =
    // NOLINTNEXTLINE(whitespace/indent_namespace)
    ObjHoldPixelEnvBase<MyoSuiteObjHoldPixelEnvSpec>;
using MyoSuiteObjHoldEnvPool = AsyncEnvPool<MyoSuiteObjHoldEnv>;
using MyoSuiteObjHoldPixelEnvPool = AsyncEnvPool<MyoSuiteObjHoldPixelEnv>;

using MyoSuiteTorsoEnv = MyoSuiteTorsoEnvBase<MyoSuiteTorsoEnvSpec, false>;
using MyoSuiteTorsoPixelEnv = TorsoPixelEnvBase<MyoSuiteTorsoPixelEnvSpec>;
using MyoSuiteTorsoEnvPool = AsyncEnvPool<MyoSuiteTorsoEnv>;
using MyoSuiteTorsoPixelEnvPool = AsyncEnvPool<MyoSuiteTorsoPixelEnv>;

using MyoSuitePenTwirlEnv =
    // NOLINTNEXTLINE(whitespace/indent_namespace)
    MyoSuitePenTwirlEnvBase<MyoSuitePenTwirlEnvSpec, false>;
using MyoSuitePenTwirlPixelEnv =
    // NOLINTNEXTLINE(whitespace/indent_namespace)
    PenTwirlPixelEnvBase<MyoSuitePenTwirlPixelEnvSpec>;
using MyoSuitePenTwirlEnvPool = AsyncEnvPool<MyoSuitePenTwirlEnv>;
using MyoSuitePenTwirlPixelEnvPool = AsyncEnvPool<MyoSuitePenTwirlPixelEnv>;

}  // namespace myosuite_envpool

#endif  // ENVPOOL_MUJOCO_MYOSUITE_MYOBASE_H_
