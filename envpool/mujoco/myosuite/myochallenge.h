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

#ifndef ENVPOOL_MUJOCO_MYOSUITE_MYOCHALLENGE_H_
#define ENVPOOL_MUJOCO_MYOSUITE_MYOCHALLENGE_H_

#include <mujoco.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "envpool/mujoco/myosuite/myobase.h"

namespace myosuite_envpool {

namespace challenge_detail {

inline std::array<mjtNum, 4> EulerXYZToQuat(
    const std::array<mjtNum, 3>& euler) {
  mjtNum ai = euler[2] / 2.0;
  mjtNum aj = -euler[1] / 2.0;
  mjtNum ak = euler[0] / 2.0;
  mjtNum si = std::sin(ai);
  mjtNum sj = std::sin(aj);
  mjtNum sk = std::sin(ak);
  mjtNum ci = std::cos(ai);
  mjtNum cj = std::cos(aj);
  mjtNum ck = std::cos(ak);
  mjtNum cc = ci * ck;
  mjtNum cs = ci * sk;
  mjtNum sc = si * ck;
  mjtNum ss = si * sk;
  return {cj * cc + sj * ss, cj * cs - sj * sc, -(cj * ss + sj * cc),
          cj * sc - sj * cs};
}

inline std::array<mjtNum, 3> Mat9ToEuler(const mjtNum* mat) {
  mjtNum cy = std::sqrt(mat[8] * mat[8] + mat[5] * mat[5]);
  bool regular = cy > std::numeric_limits<mjtNum>::epsilon() * 4.0;
  return {
      regular ? -std::atan2(mat[5], mat[8]) : 0.0,
      -std::atan2(-mat[2], cy),
      regular ? -std::atan2(mat[1], mat[0]) : -std::atan2(-mat[3], mat[4]),
  };
}

inline std::array<mjtNum, 3> RotateByQuat(const mjtNum* quat,
                                          const mjtNum* vec) {
  std::array<mjtNum, 9> mat{};
  mju_quat2Mat(mat.data(), quat);
  return {
      mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2],
      mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2],
      mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2],
  };
}

inline std::array<float, 3> RotateByQuat(const mjtNum* quat, const float* vec) {
  std::array<mjtNum, 9> mat{};
  mju_quat2Mat(mat.data(), quat);
  return {
      static_cast<float>(mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2]),
      static_cast<float>(mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2]),
      static_cast<float>(mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2]),
  };
}

inline mjtNum Norm3(const std::array<mjtNum, 3>& value) {
  return std::sqrt(value[0] * value[0] + value[1] * value[1] +
                   value[2] * value[2]);
}

inline std::array<mjtNum, 3> Sub3(const std::array<mjtNum, 3>& lhs,
                                  const std::array<mjtNum, 3>& rhs) {
  return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
}

inline std::array<mjtNum, 3> UniformVec3(std::mt19937* gen, mjtNum low,
                                         mjtNum high) {
  std::uniform_real_distribution<double> dist(low, high);
  return {static_cast<mjtNum>(dist(*gen)), static_cast<mjtNum>(dist(*gen)),
          static_cast<mjtNum>(dist(*gen))};
}

inline std::array<mjtNum, 3> UniformVec3(std::mt19937* gen,
                                         const std::array<mjtNum, 3>& low,
                                         const std::array<mjtNum, 3>& high) {
  std::array<mjtNum, 3> value{};
  for (int axis = 0; axis < 3; ++axis) {
    std::uniform_real_distribution<double> dist(low[axis], high[axis]);
    value[axis] = static_cast<mjtNum>(dist(*gen));
  }
  return value;
}

inline mjtNum UniformScalar(std::mt19937* gen, mjtNum low, mjtNum high) {
  std::uniform_real_distribution<double> dist(low, high);
  return static_cast<mjtNum>(dist(*gen));
}

inline void CopyBodyGeomSlice3(const mjModel* model, const mjtNum* src,
                               int body_id, std::vector<mjtNum>* out) {
  int start_geom = model->body_geomadr[body_id];
  int geom_count = model->body_geomnum[body_id];
  out->assign(src + start_geom * 3, src + (start_geom + geom_count) * 3);
}

inline void RestoreBodyGeomSlice3(mjModel* model, mjtNum* target, int body_id,
                                  const std::vector<mjtNum>& value) {
  if (value.empty()) {
    return;
  }
  int start_geom = model->body_geomadr[body_id];
  int geom_count = model->body_geomnum[body_id];
  std::memcpy(target + start_geom * 3, value.data(),
              sizeof(mjtNum) * 3 * geom_count);
}

inline void CopyBodyGeomSlice4(const mjModel* model, const mjtNum* src,
                               int body_id, std::vector<mjtNum>* out) {
  int start_geom = model->body_geomadr[body_id];
  int geom_count = model->body_geomnum[body_id];
  out->assign(src + start_geom * 4, src + (start_geom + geom_count) * 4);
}

inline void RestoreBodyGeomSlice4(mjModel* model, mjtNum* target, int body_id,
                                  const std::vector<mjtNum>& value) {
  if (value.empty()) {
    return;
  }
  int start_geom = model->body_geomadr[body_id];
  int geom_count = model->body_geomnum[body_id];
  std::memcpy(target + start_geom * 4, value.data(),
              sizeof(mjtNum) * 4 * geom_count);
}

inline void CopyBodyGeomSlice4f(const mjModel* model, const float* src,
                                int body_id, std::vector<float>* out) {
  int start_geom = model->body_geomadr[body_id];
  int geom_count = model->body_geomnum[body_id];
  out->assign(src + start_geom * 4, src + (start_geom + geom_count) * 4);
}

inline void RestoreBodyGeomSlice4f(mjModel* model, float* target, int body_id,
                                   const std::vector<float>& value) {
  if (value.empty()) {
    return;
  }
  int start_geom = model->body_geomadr[body_id];
  int geom_count = model->body_geomnum[body_id];
  std::memcpy(target + start_geom * 4, value.data(),
              sizeof(float) * 4 * geom_count);
}

inline void CopyBodyGeomTypeSlice(const mjModel* model, int body_id,
                                  std::vector<int>* out) {
  int start_geom = model->body_geomadr[body_id];
  int geom_count = model->body_geomnum[body_id];
  out->assign(model->geom_type + start_geom,
              model->geom_type + start_geom + geom_count);
}

inline void RestoreBodyGeomTypeSlice(mjModel* model, int body_id,
                                     const std::vector<int>& value) {
  if (value.empty()) {
    return;
  }
  int start_geom = model->body_geomadr[body_id];
  int geom_count = model->body_geomnum[body_id];
  std::memcpy(model->geom_type + start_geom, value.data(),
              sizeof(int) * geom_count);
}

inline std::vector<int> ToIntVector(const std::vector<double>& input) {
  std::vector<int> out(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    out[i] = static_cast<int>(input[i]);
  }
  return out;
}

inline std::vector<float> ToFloatVector(const std::vector<double>& input) {
  return {input.begin(), input.end()};
}

inline void SetSiteSuccessColor(mjModel* model, int site_id, bool success) {
  if (site_id < 0) {
    return;
  }
  model->site_rgba[site_id * 4 + 0] = success ? 0.0 : 2.0;
  model->site_rgba[site_id * 4 + 1] = success ? 2.0 : 0.0;
}

}  // namespace challenge_detail

class MyoChallengeReorientEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(5),
        "frame_stack"_.Bind(1), "model_path"_.Bind(std::string()),
        "normalize_act"_.Bind(true), "muscle_condition"_.Bind(std::string()),
        "fatigue_reset_vec"_.Bind(std::vector<double>{}),
        "fatigue_reset_random"_.Bind(false), "obs_dim"_.Bind(0),
        "qpos_dim"_.Bind(0), "qvel_dim"_.Bind(0), "act_dim"_.Bind(0),
        "action_dim"_.Bind(0), "goal_pos_low"_.Bind(0.0),
        "goal_pos_high"_.Bind(0.0), "goal_rot_low"_.Bind(0.0),
        "goal_rot_high"_.Bind(0.0), "obj_size_change"_.Bind(0.0),
        "obj_mass_low"_.Bind(0.108), "obj_mass_high"_.Bind(0.108),
        "obj_friction_change"_.Bind(std::vector<double>{0.0, 0.0, 0.0}),
        "pos_th"_.Bind(0.025), "rot_th"_.Bind(0.262), "drop_th"_.Bind(0.2),
        "reward_pos_dist_w"_.Bind(100.0), "reward_rot_dist_w"_.Bind(1.0),
        "reward_bonus_w"_.Bind(0.0), "reward_act_reg_w"_.Bind(0.0),
        "reward_penalty_w"_.Bind(0.0),
        "test_reset_qpos"_.Bind(std::vector<double>{}),
        "test_reset_qvel"_.Bind(std::vector<double>{}),
        "test_reset_act"_.Bind(std::vector<double>{}),
        "test_reset_qacc_warmstart"_.Bind(std::vector<double>{}),
        "test_goal_body_pos"_.Bind(std::vector<double>{}),
        "test_goal_body_quat"_.Bind(std::vector<double>{}),
        "test_target_geom_size"_.Bind(std::vector<double>{}),
        "test_object_geom_size"_.Bind(std::vector<double>{}),
        "test_object_geom_pos"_.Bind(std::vector<double>{}),
        "test_object_geom_friction"_.Bind(std::vector<double>{}),
        "test_object_body_mass"_.Bind(std::vector<double>{}));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs"_.Bind(StackSpec(Spec<mjtNum>({conf["obs_dim"_]}, {-inf, inf}),
                              conf["frame_stack"_])),
        "info:pos_dist"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:rot_dist"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:bonus"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:act_reg"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:penalty"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:sparse"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:solved"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({conf["qpos_dim"_]})),
        "info:qvel0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
#endif
        "info:goal_pos"_.Bind(Spec<mjtNum>({3})),
        "info:goal_rot"_.Bind(Spec<mjtNum>({3})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<float>({-1, conf["action_dim"_]}, {-1.0, 1.0})));
  }
};

using ReorientPixelFns = PixelObservationEnvFns<MyoChallengeReorientEnvFns>;
using MyoChallengeReorientEnvSpec = EnvSpec<MyoChallengeReorientEnvFns>;
using MyoChallengeReorientPixelEnvSpec = EnvSpec<ReorientPixelFns>;

class MyoChallengeRelocateEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(5),
        "frame_stack"_.Bind(1), "model_path"_.Bind(std::string()),
        "normalize_act"_.Bind(true), "muscle_condition"_.Bind(std::string()),
        "fatigue_reset_vec"_.Bind(std::vector<double>{}),
        "fatigue_reset_random"_.Bind(false), "obs_dim"_.Bind(0),
        "qpos_dim"_.Bind(0), "qvel_dim"_.Bind(0), "act_dim"_.Bind(0),
        "action_dim"_.Bind(0),
        "target_xyz_low"_.Bind(std::vector<double>{0.0, -0.35, 0.9}),
        "target_xyz_high"_.Bind(std::vector<double>{0.2, -0.1, 0.9}),
        "target_rxryrz_low"_.Bind(std::vector<double>{0.0, 0.0, 0.0}),
        "target_rxryrz_high"_.Bind(std::vector<double>{0.0, 0.0, 0.0}),
        "obj_xyz_low"_.Bind(std::vector<double>{}),
        "obj_xyz_high"_.Bind(std::vector<double>{}),
        "obj_geom_low"_.Bind(std::vector<double>{}),
        "obj_geom_high"_.Bind(std::vector<double>{}), "obj_mass_low"_.Bind(0.0),
        "obj_mass_high"_.Bind(0.0),
        "obj_friction_low"_.Bind(std::vector<double>{}),
        "obj_friction_high"_.Bind(std::vector<double>{}),
        "qpos_noise_range"_.Bind(0.0), "pos_th"_.Bind(0.025),
        "rot_th"_.Bind(0.262), "drop_th"_.Bind(0.5),
        "reward_pos_dist_w"_.Bind(100.0), "reward_rot_dist_w"_.Bind(1.0),
        "reward_act_reg_w"_.Bind(0.0),
        "test_reset_qpos"_.Bind(std::vector<double>{}),
        "test_reset_qvel"_.Bind(std::vector<double>{}),
        "test_reset_act"_.Bind(std::vector<double>{}),
        "test_reset_qacc_warmstart"_.Bind(std::vector<double>{}),
        "test_goal_body_pos"_.Bind(std::vector<double>{}),
        "test_goal_body_quat"_.Bind(std::vector<double>{}),
        "test_object_body_pos"_.Bind(std::vector<double>{}),
        "test_object_body_mass"_.Bind(std::vector<double>{}),
        "test_object_geom_type"_.Bind(std::vector<double>{}),
        "test_object_geom_size"_.Bind(std::vector<double>{}),
        "test_object_geom_pos"_.Bind(std::vector<double>{}),
        "test_object_geom_quat"_.Bind(std::vector<double>{}),
        "test_object_geom_rgba"_.Bind(std::vector<double>{}),
        "test_object_geom_friction"_.Bind(std::vector<double>{}));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs"_.Bind(StackSpec(Spec<mjtNum>({conf["obs_dim"_]}, {-inf, inf}),
                              conf["frame_stack"_])),
        "info:reach_dist"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:pos_dist"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:rot_dist"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:act_reg"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:sparse"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:solved"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({conf["qpos_dim"_]})),
        "info:qvel0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
#endif
        "info:goal_pos"_.Bind(Spec<mjtNum>({3})),
        "info:goal_rot"_.Bind(Spec<mjtNum>({3})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<float>({-1, conf["action_dim"_]}, {-1.0, 1.0})));
  }
};

using RelocatePixelFns = PixelObservationEnvFns<MyoChallengeRelocateEnvFns>;
using MyoChallengeRelocateEnvSpec = EnvSpec<MyoChallengeRelocateEnvFns>;
using MyoChallengeRelocatePixelEnvSpec = EnvSpec<RelocatePixelFns>;

class MyoChallengeBaodingEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(10),
        "frame_stack"_.Bind(1), "model_path"_.Bind(std::string()),
        "normalize_act"_.Bind(true), "muscle_condition"_.Bind(std::string()),
        "fatigue_reset_vec"_.Bind(std::vector<double>{}),
        "fatigue_reset_random"_.Bind(false), "obs_dim"_.Bind(0),
        "qpos_dim"_.Bind(0), "qvel_dim"_.Bind(0), "act_dim"_.Bind(0),
        "action_dim"_.Bind(0), "drop_th"_.Bind(1.25),
        "proximity_th"_.Bind(0.015), "goal_time_period_low"_.Bind(5.0),
        "goal_time_period_high"_.Bind(5.0), "goal_xrange_low"_.Bind(0.025),
        "goal_xrange_high"_.Bind(0.025), "goal_yrange_low"_.Bind(0.028),
        "goal_yrange_high"_.Bind(0.028),
        "task_choice"_.Bind(std::string("fixed")), "fixed_task"_.Bind(2),
        "reward_pos_dist_1_w"_.Bind(5.0), "reward_pos_dist_2_w"_.Bind(5.0),
        "obj_size_low"_.Bind(0.0), "obj_size_high"_.Bind(0.0),
        "obj_mass_low"_.Bind(0.0), "obj_mass_high"_.Bind(0.0),
        "obj_friction_low"_.Bind(std::vector<double>{}),
        "obj_friction_high"_.Bind(std::vector<double>{}),
        "test_reset_qpos"_.Bind(std::vector<double>{}),
        "test_reset_qvel"_.Bind(std::vector<double>{}),
        "test_reset_act"_.Bind(std::vector<double>{}),
        "test_reset_qacc_warmstart"_.Bind(std::vector<double>{}),
        "test_task"_.Bind(-1),
        "test_ball1_starting_angle"_.Bind(
            std::numeric_limits<double>::quiet_NaN()),
        "test_ball2_starting_angle"_.Bind(
            std::numeric_limits<double>::quiet_NaN()),
        "test_x_radius"_.Bind(std::numeric_limits<double>::quiet_NaN()),
        "test_y_radius"_.Bind(std::numeric_limits<double>::quiet_NaN()),
        "test_goal_trajectory"_.Bind(std::vector<double>{}),
        "test_object1_body_mass"_.Bind(std::vector<double>{}),
        "test_object2_body_mass"_.Bind(std::vector<double>{}),
        "test_object1_geom_size"_.Bind(std::vector<double>{}),
        "test_object2_geom_size"_.Bind(std::vector<double>{}),
        "test_object1_geom_friction"_.Bind(std::vector<double>{}),
        "test_object2_geom_friction"_.Bind(std::vector<double>{}));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs"_.Bind(StackSpec(Spec<mjtNum>({conf["obs_dim"_]}, {-inf, inf}),
                              conf["frame_stack"_])),
        "info:pos_dist_1"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:pos_dist_2"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:act_reg"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:sparse"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:solved"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({conf["qpos_dim"_]})),
        "info:qvel0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
#endif
        "info:target1_pos"_.Bind(Spec<mjtNum>({3})),
        "info:target2_pos"_.Bind(Spec<mjtNum>({3})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<float>({-1, conf["action_dim"_]}, {-1.0, 1.0})));
  }
};

using BaodingPixelFns = PixelObservationEnvFns<MyoChallengeBaodingEnvFns>;
using MyoChallengeBaodingEnvSpec = EnvSpec<MyoChallengeBaodingEnvFns>;
using MyoChallengeBaodingPixelEnvSpec = EnvSpec<BaodingPixelFns>;

class MyoChallengeBimanualEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(5),
        "frame_stack"_.Bind(1), "model_path"_.Bind(std::string()),
        "normalize_act"_.Bind(true), "muscle_condition"_.Bind(std::string()),
        "fatigue_reset_vec"_.Bind(std::vector<double>{}),
        "fatigue_reset_random"_.Bind(false), "obs_dim"_.Bind(0),
        "qpos_dim"_.Bind(0), "qvel_dim"_.Bind(0), "act_dim"_.Bind(0),
        "action_dim"_.Bind(0), "proximity_th"_.Bind(0.17),
        "start_center"_.Bind(std::vector<double>{}),
        "goal_center"_.Bind(std::vector<double>{}),
        "start_shifts"_.Bind(std::vector<double>{}),
        "goal_shifts"_.Bind(std::vector<double>{}),
        "reward_reach_dist_w"_.Bind(-0.1), "reward_act_w"_.Bind(0.0),
        "reward_fin_dis_w"_.Bind(-0.5), "reward_pass_err_w"_.Bind(-1.0),
        "obj_scale_change"_.Bind(std::vector<double>{}),
        "obj_mass_low"_.Bind(0.0), "obj_mass_high"_.Bind(0.0),
        "obj_friction_low"_.Bind(std::vector<double>{}),
        "obj_friction_high"_.Bind(std::vector<double>{}),
        "test_reset_qpos"_.Bind(std::vector<double>{}),
        "test_reset_qvel"_.Bind(std::vector<double>{}),
        "test_reset_act"_.Bind(std::vector<double>{}),
        "test_reset_qacc_warmstart"_.Bind(std::vector<double>{}),
        "test_start_pos"_.Bind(std::vector<double>{}),
        "test_goal_pos"_.Bind(std::vector<double>{}),
        "test_object_body_mass"_.Bind(std::vector<double>{}),
        "test_object_geom_size"_.Bind(std::vector<double>{}),
        "test_object_geom_friction"_.Bind(std::vector<double>{}));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs"_.Bind(StackSpec(Spec<mjtNum>({conf["obs_dim"_]}, {-inf, inf}),
                              conf["frame_stack"_])),
        "info:reach_dist"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:act"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:fin_dis"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:pass_err"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:goal_dist"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:solved"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({conf["qpos_dim"_]})),
        "info:qvel0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
#endif
        "info:start_pos"_.Bind(Spec<mjtNum>({3})),
        "info:goal_pos"_.Bind(Spec<mjtNum>({3})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<float>({-1, conf["action_dim"_]}, {-1.0, 1.0})));
  }
};

using BimanualPixelFns = PixelObservationEnvFns<MyoChallengeBimanualEnvFns>;
using MyoChallengeBimanualEnvSpec = EnvSpec<MyoChallengeBimanualEnvFns>;
using MyoChallengeBimanualPixelEnvSpec = EnvSpec<BimanualPixelFns>;

template <typename EnvSpecT, bool kFromPixels>
class MyoChallengeReorientEnvBase : public Env<EnvSpecT>,
                                    public gymnasium_robotics::MujocoRobotEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  struct RewardInfo {
    mjtNum dense_reward{0.0};
    mjtNum pos_dist_term{0.0};
    mjtNum rot_dist_term{0.0};
    mjtNum bonus{0.0};
    mjtNum act_reg_term{0.0};
    mjtNum penalty{0.0};
    mjtNum sparse{0.0};
    bool solved{false};
    bool done{false};
  };

  bool normalize_act_;
  mjtNum goal_pos_low_;
  mjtNum goal_pos_high_;
  mjtNum goal_rot_low_;
  mjtNum goal_rot_high_;
  mjtNum obj_size_change_;
  mjtNum obj_mass_low_;
  mjtNum obj_mass_high_;
  std::array<mjtNum, 3> obj_friction_change_;
  mjtNum pos_th_;
  mjtNum rot_th_;
  mjtNum drop_th_;
  mjtNum reward_pos_dist_w_;
  mjtNum reward_rot_dist_w_;
  mjtNum reward_bonus_w_;
  mjtNum reward_act_reg_w_;
  mjtNum reward_penalty_w_;
  int object_sid_{-1};
  int goal_sid_{-1};
  int success_indicator_sid_{-1};
  int goal_bid_{-1};
  int target_gid_{-1};
  int object_bid_{-1};
  int object_gid0_{-1};
  int object_gidn_{-1};
  std::vector<mjtNum> default_goal_body_pos_;
  std::vector<mjtNum> default_goal_body_quat_;
  std::vector<mjtNum> goal_init_pos_;
  std::vector<mjtNum> goal_obj_offset_;
  std::vector<mjtNum> target_default_size_;
  std::vector<mjtNum> object_default_size_;
  std::vector<mjtNum> object_default_pos_;
  std::vector<mjtNum> object_default_friction_;
  mjtNum default_object_body_mass_{0.0};
  std::vector<mjtNum> test_reset_qpos_;
  std::vector<mjtNum> test_reset_qvel_;
  std::vector<mjtNum> test_reset_act_;
  std::vector<mjtNum> test_reset_qacc_warmstart_;
  std::vector<mjtNum> test_goal_body_pos_;
  std::vector<mjtNum> test_goal_body_quat_;
  std::vector<mjtNum> test_target_geom_size_;
  std::vector<mjtNum> test_object_geom_size_;
  std::vector<mjtNum> test_object_geom_pos_;
  std::vector<mjtNum> test_object_geom_friction_;
  std::vector<mjtNum> test_object_body_mass_;
  std::vector<bool> muscle_actuator_;
  detail::MyoConditionState muscle_condition_state_;

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  MyoChallengeReorientEnvBase(const Spec& spec, int env_id)
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
        goal_pos_low_(spec.config["goal_pos_low"_]),
        goal_pos_high_(spec.config["goal_pos_high"_]),
        goal_rot_low_(spec.config["goal_rot_low"_]),
        goal_rot_high_(spec.config["goal_rot_high"_]),
        obj_size_change_(spec.config["obj_size_change"_]),
        obj_mass_low_(spec.config["obj_mass_low"_]),
        obj_mass_high_(spec.config["obj_mass_high"_]),
        obj_friction_change_{
            static_cast<mjtNum>(spec.config["obj_friction_change"_][0]),
            static_cast<mjtNum>(spec.config["obj_friction_change"_][1]),
            static_cast<mjtNum>(spec.config["obj_friction_change"_][2])},
        pos_th_(spec.config["pos_th"_]),
        rot_th_(spec.config["rot_th"_]),
        drop_th_(spec.config["drop_th"_]),
        reward_pos_dist_w_(spec.config["reward_pos_dist_w"_]),
        reward_rot_dist_w_(spec.config["reward_rot_dist_w"_]),
        reward_bonus_w_(spec.config["reward_bonus_w"_]),
        reward_act_reg_w_(spec.config["reward_act_reg_w"_]),
        reward_penalty_w_(spec.config["reward_penalty_w"_]),
        test_reset_qpos_(detail::ToMjtVector(spec.config["test_reset_qpos"_])),
        test_reset_qvel_(detail::ToMjtVector(spec.config["test_reset_qvel"_])),
        test_reset_act_(detail::ToMjtVector(spec.config["test_reset_act"_])),
        test_reset_qacc_warmstart_(
            detail::ToMjtVector(spec.config["test_reset_qacc_warmstart"_])),
        test_goal_body_pos_(
            detail::ToMjtVector(spec.config["test_goal_body_pos"_])),
        test_goal_body_quat_(
            detail::ToMjtVector(spec.config["test_goal_body_quat"_])),
        test_target_geom_size_(
            detail::ToMjtVector(spec.config["test_target_geom_size"_])),
        test_object_geom_size_(
            detail::ToMjtVector(spec.config["test_object_geom_size"_])),
        test_object_geom_pos_(
            detail::ToMjtVector(spec.config["test_object_geom_pos"_])),
        test_object_geom_friction_(
            detail::ToMjtVector(spec.config["test_object_geom_friction"_])),
        test_object_body_mass_(
            detail::ToMjtVector(spec.config["test_object_body_mass"_])) {
    ValidateConfig();
    CacheIds();
    detail::BuildMuscleMask(model_, &muscle_actuator_);
    detail::InitializeMyoConditionState(
        model_, spec.config["muscle_condition"_],
        spec.config["fatigue_reset_vec"_], spec.config["fatigue_reset_random"_],
        spec.config["frame_skip"_], this->seed_, &muscle_condition_state_);
    for (int i = 0; i < model_->nq - 7; ++i) {
      data_->qpos[i] = 0.0;
    }
    data_->qpos[0] = static_cast<mjtNum>(-1.5);
    mj_forward(model_, data_);
    CacheDefaultState();
    InitializeRobotEnv();
  }

  envpool::mujoco::CameraPolicy RenderCameraPolicy() const override {
    return detail::MyoSuiteRenderCameraPolicy();
  }

  void ConfigureRenderOption(mjvOption* option) const override {
    detail::ConfigureMyoSuiteRenderOptions(option);
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    detail::ResetMyoConditionState(&muscle_condition_state_);
    ResetToInitialState();
    RestoreModelState();
    ApplyModelRandomization();
    ApplyResetState();
    CaptureResetState();
    RewardInfo reward = ComputeRewardInfo();
    WriteState(reward, true, 0.0);
  }

  void Step(const Action& action) override {
    const auto* raw = static_cast<const float*>(action["action"_].Data());
    detail::ApplyMyoSuiteAction(model_, data_, muscle_actuator_, normalize_act_,
                                raw);
    detail::ApplyMyoConditionAdjustments(model_, data_, muscle_actuator_,
                                         &muscle_condition_state_);
    InvalidateRenderCache();
    detail::DoMyoSuiteSimulation(model_, data_, frame_skip_);
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
      throw std::runtime_error(
          "MyoChallenge Reorient dims do not match model.");
    }
    int expected_obs = (model_->nq - 7) + (model_->nv - 6) + 18 + model_->na;
    if (expected_obs != spec_.config["obs_dim"_]) {
      throw std::runtime_error(
          "MyoChallenge Reorient obs_dim does not match model.");
    }
  }

  void CacheIds() {
    object_sid_ = mj_name2id(model_, mjOBJ_SITE, "object_o");
    goal_sid_ = mj_name2id(model_, mjOBJ_SITE, "target_o");
    success_indicator_sid_ = mj_name2id(model_, mjOBJ_SITE, "target_ball");
    goal_bid_ = mj_name2id(model_, mjOBJ_BODY, "target");
    target_gid_ = mj_name2id(model_, mjOBJ_GEOM, "target_dice");
    object_bid_ = mj_name2id(model_, mjOBJ_BODY, "Object");
    if (object_sid_ == -1 || goal_sid_ == -1 || goal_bid_ == -1 ||
        target_gid_ == -1 || object_bid_ == -1) {
      throw std::runtime_error("MyoChallenge Reorient ids missing.");
    }
    object_gid0_ = model_->body_geomadr[object_bid_];
    object_gidn_ = object_gid0_ + model_->body_geomnum[object_bid_];
  }

  void CacheDefaultState() {
    detail::CopyModelBodyPos(model_, goal_bid_, &default_goal_body_pos_);
    detail::CopyModelBodyQuat(model_, goal_bid_, &default_goal_body_quat_);
    goal_init_pos_.resize(3);
    goal_obj_offset_.resize(3);
    detail::CopySitePos(model_, data_, goal_sid_, goal_init_pos_.data());
    std::array<mjtNum, 3> goal{};
    std::array<mjtNum, 3> obj{};
    detail::CopySitePos(model_, data_, goal_sid_, goal.data());
    detail::CopySitePos(model_, data_, object_sid_, obj.data());
    for (int axis = 0; axis < 3; ++axis) {
      goal_obj_offset_[axis] = goal[axis] - obj[axis];
    }
    detail::CopyModelGeomSize(model_, target_gid_, &target_default_size_);
    challenge_detail::CopyBodyGeomSlice3(model_, model_->geom_size, object_bid_,
                                         &object_default_size_);
    challenge_detail::CopyBodyGeomSlice3(model_, model_->geom_pos, object_bid_,
                                         &object_default_pos_);
    challenge_detail::CopyBodyGeomSlice3(
        model_, model_->geom_friction, object_bid_, &object_default_friction_);
    detail::CopyModelBodyMass(model_, object_bid_, &default_object_body_mass_);
  }

  void RestoreModelState() {
    detail::RestoreModelBodyPos(model_, goal_bid_, default_goal_body_pos_);
    detail::RestoreModelBodyQuat(model_, goal_bid_, default_goal_body_quat_);
    challenge_detail::RestoreBodyGeomSlice3(model_, model_->geom_size,
                                            object_bid_, object_default_size_);
    challenge_detail::RestoreBodyGeomSlice3(model_, model_->geom_pos,
                                            object_bid_, object_default_pos_);
    challenge_detail::RestoreBodyGeomSlice3(
        model_, model_->geom_friction, object_bid_, object_default_friction_);
    detail::RestoreModelGeomSize(model_, target_gid_, target_default_size_);
    detail::RestoreModelBodyMass(model_, object_bid_,
                                 default_object_body_mass_);
  }

  void ApplyGoalBodyRandomization() {
    if (!test_goal_body_pos_.empty()) {
      detail::RestoreModelBodyPos(model_, goal_bid_, test_goal_body_pos_);
    } else {
      auto pos =
          challenge_detail::UniformVec3(&gen_, goal_pos_low_, goal_pos_high_);
      for (int axis = 0; axis < 3; ++axis) {
        model_->body_pos[goal_bid_ * 3 + axis] =
            goal_init_pos_[axis] + pos[axis];
      }
    }
    if (!test_goal_body_quat_.empty()) {
      detail::RestoreModelBodyQuat(model_, goal_bid_, test_goal_body_quat_);
    } else {
      auto goal_rot =
          challenge_detail::UniformVec3(&gen_, goal_rot_low_, goal_rot_high_);
      auto quat = challenge_detail::EulerXYZToQuat(goal_rot);
      std::memcpy(model_->body_quat + goal_bid_ * 4, quat.data(),
                  sizeof(mjtNum) * 4);
    }
  }

  void ApplyObjectRandomization() {
    if (!test_target_geom_size_.empty()) {
      detail::RestoreModelGeomSize(model_, target_gid_, test_target_geom_size_);
    }
    if (!test_object_geom_size_.empty()) {
      challenge_detail::RestoreBodyGeomSlice3(
          model_, model_->geom_size, object_bid_, test_object_geom_size_);
    }
    if (!test_object_geom_pos_.empty()) {
      challenge_detail::RestoreBodyGeomSlice3(
          model_, model_->geom_pos, object_bid_, test_object_geom_pos_);
    }
    if (!test_object_geom_friction_.empty()) {
      challenge_detail::RestoreBodyGeomSlice3(model_, model_->geom_friction,
                                              object_bid_,
                                              test_object_geom_friction_);
    }
    if (!test_object_body_mass_.empty()) {
      detail::RestoreModelBodyMass(model_, object_bid_,
                                   test_object_body_mass_[0]);
    }
    if (!test_target_geom_size_.empty() || !test_object_geom_size_.empty() ||
        !test_object_geom_pos_.empty() || !test_object_geom_friction_.empty() ||
        !test_object_body_mass_.empty()) {
      return;
    }

    model_->body_mass[object_bid_] =
        challenge_detail::UniformScalar(&gen_, obj_mass_low_, obj_mass_high_);
    std::array<mjtNum, 3> low = {
        object_default_friction_[0] - obj_friction_change_[0],
        object_default_friction_[1] - obj_friction_change_[1],
        object_default_friction_[2] - obj_friction_change_[2],
    };
    std::array<mjtNum, 3> high = {
        object_default_friction_[0] + obj_friction_change_[0],
        object_default_friction_[1] + obj_friction_change_[1],
        object_default_friction_[2] + obj_friction_change_[2],
    };
    for (int gid = object_gid0_; gid < object_gidn_; ++gid) {
      auto friction = challenge_detail::UniformVec3(&gen_, low, high);
      std::memcpy(model_->geom_friction + gid * 3, friction.data(),
                  sizeof(mjtNum) * 3);
    }

    mjtNum del_size = challenge_detail::UniformScalar(&gen_, -obj_size_change_,
                                                      obj_size_change_);
    for (int axis = 0; axis < 3; ++axis) {
      model_->geom_size[target_gid_ * 3 + axis] =
          target_default_size_[axis] + del_size;
    }
    for (int gid = object_gid0_; gid < object_gidn_ - 3; ++gid) {
      int offset = gid * 3;
      model_->geom_size[offset + 0] =
          object_default_size_[(gid - object_gid0_) * 3];
      model_->geom_size[offset + 1] =
          object_default_size_[(gid - object_gid0_) * 3 + 1] + del_size;
      model_->geom_size[offset + 2] =
          object_default_size_[(gid - object_gid0_) * 3 + 2];
    }
    for (int gid = object_gidn_ - 3; gid < object_gidn_; ++gid) {
      int rel = (gid - object_gid0_) * 3;
      for (int axis = 0; axis < 3; ++axis) {
        model_->geom_size[gid * 3 + axis] =
            object_default_size_[rel + axis] + del_size;
      }
    }
    for (int gid = object_gid0_; gid < object_gidn_; ++gid) {
      for (int axis = 0; axis < 3; ++axis) {
        int index = (gid - object_gid0_) * 3 + axis;
        mjtNum value = object_default_pos_[index];
        mjtNum sign = value / std::abs(value + static_cast<mjtNum>(1e-16));
        model_->geom_pos[gid * 3 + axis] =
            sign * (std::abs(object_default_pos_[index]) + del_size);
      }
    }
  }

  void ApplyModelRandomization() {
    ApplyGoalBodyRandomization();
    ApplyObjectRandomization();
  }

  void ApplyResetState() {
    if (!test_reset_qpos_.empty()) {
      detail::RestoreVector(test_reset_qpos_, data_->qpos);
    }
    if (!test_reset_qvel_.empty()) {
      detail::RestoreVector(test_reset_qvel_, data_->qvel);
    }
    mj_forward(model_, data_);
    bool rerun_forward = false;
    if (!test_reset_act_.empty()) {
      detail::RestoreVector(test_reset_act_, data_->act);
      rerun_forward = true;
    }
    if (!test_reset_qacc_warmstart_.empty()) {
      detail::RestoreVector(test_reset_qacc_warmstart_, data_->qacc_warmstart);
    }
    if (rerun_forward) {
      mj_forward(model_, data_);
    }
  }

  std::vector<mjtNum> Observation() const {
    std::vector<mjtNum> obs;
    obs.reserve((model_->nq - 7) + (model_->nv - 6) + 18 + model_->na);
    obs.insert(obs.end(), data_->qpos, data_->qpos + model_->nq - 7);
    mjtNum dt = Dt();
    for (int i = 0; i < model_->nv - 6; ++i) {
      obs.push_back(data_->qvel[i] * dt);
    }
    std::array<mjtNum, 3> obj_pos{};
    std::array<mjtNum, 3> goal_pos{};
    detail::CopySitePos(model_, data_, object_sid_, obj_pos.data());
    detail::CopySitePos(model_, data_, goal_sid_, goal_pos.data());
    auto pos_err = challenge_detail::Sub3(goal_pos, obj_pos);
    for (int axis = 0; axis < 3; ++axis) {
      pos_err[axis] -= goal_obj_offset_[axis];
    }
    auto obj_rot =
        challenge_detail::Mat9ToEuler(data_->site_xmat + object_sid_ * 9);
    auto goal_rot =
        challenge_detail::Mat9ToEuler(data_->site_xmat + goal_sid_ * 9);
    auto rot_err = challenge_detail::Sub3(goal_rot, obj_rot);
    obs.insert(obs.end(), obj_pos.begin(), obj_pos.end());
    obs.insert(obs.end(), goal_pos.begin(), goal_pos.end());
    obs.insert(obs.end(), pos_err.begin(), pos_err.end());
    obs.insert(obs.end(), obj_rot.begin(), obj_rot.end());
    obs.insert(obs.end(), goal_rot.begin(), goal_rot.end());
    obs.insert(obs.end(), rot_err.begin(), rot_err.end());
    if (model_->na > 0) {
      obs.insert(obs.end(), data_->act, data_->act + model_->na);
    }
    return obs;
  }

  RewardInfo ComputeRewardInfo() {
    std::array<mjtNum, 3> obj_pos{};
    std::array<mjtNum, 3> goal_pos{};
    detail::CopySitePos(model_, data_, object_sid_, obj_pos.data());
    detail::CopySitePos(model_, data_, goal_sid_, goal_pos.data());
    auto pos_err = challenge_detail::Sub3(goal_pos, obj_pos);
    for (int axis = 0; axis < 3; ++axis) {
      pos_err[axis] -= goal_obj_offset_[axis];
    }
    auto obj_rot =
        challenge_detail::Mat9ToEuler(data_->site_xmat + object_sid_ * 9);
    auto goal_rot =
        challenge_detail::Mat9ToEuler(data_->site_xmat + goal_sid_ * 9);
    auto rot_err = challenge_detail::Sub3(goal_rot, obj_rot);

    mjtNum pos_dist = challenge_detail::Norm3(pos_err);
    mjtNum rot_dist = challenge_detail::Norm3(rot_err);
    mjtNum act_reg = detail::ActReg(model_, data_);
    bool drop = pos_dist > drop_th_;
    RewardInfo info;
    info.pos_dist_term = -pos_dist;
    info.rot_dist_term = -rot_dist;
    info.bonus = static_cast<mjtNum>(pos_dist < 2 * pos_th_) +
                 static_cast<mjtNum>(pos_dist < pos_th_);
    info.act_reg_term = -act_reg;
    info.penalty = -static_cast<mjtNum>(drop);
    info.sparse = -rot_dist - static_cast<mjtNum>(10.0) * pos_dist;
    info.solved = pos_dist < pos_th_ && rot_dist < rot_th_ && !drop;
    info.done = drop;
    info.dense_reward = reward_pos_dist_w_ * info.pos_dist_term +
                        reward_rot_dist_w_ * info.rot_dist_term +
                        reward_bonus_w_ * info.bonus +
                        reward_act_reg_w_ * info.act_reg_term +
                        reward_penalty_w_ * info.penalty;
    challenge_detail::SetSiteSuccessColor(model_, success_indicator_sid_,
                                          info.solved);
    return info;
  }

  void WriteState(const RewardInfo& reward, bool reset, float reward_value) {
    auto obs = Observation();
    auto state = Allocate();
    if constexpr (!kFromPixels) {
      AssignObservation("obs", &state["obs"_], obs.data(), obs.size(), reset);
    }
    state["reward"_] = reward_value;
    state["discount"_] = 1.0f;
    state["done"_] = done_;
    state["trunc"_] = elapsed_step_ >= max_episode_steps_;
    state["elapsed_step"_] = elapsed_step_;
    state["info:pos_dist"_] = reward.pos_dist_term;
    state["info:rot_dist"_] = reward.rot_dist_term;
    state["info:bonus"_] = reward.bonus;
    state["info:act_reg"_] = reward.act_reg_term;
    state["info:penalty"_] = reward.penalty;
    state["info:sparse"_] = reward.sparse;
    state["info:solved"_] = static_cast<mjtNum>(reward.solved);
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
    state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
    state["info:qacc0"_].Assign(qacc0_.data(), qacc0_.size());
    state["info:qacc_warmstart0"_].Assign(qacc_warmstart0_.data(),
                                          qacc_warmstart0_.size());
#endif
    std::array<mjtNum, 3> goal_pos{};
    detail::CopySitePos(model_, data_, goal_sid_, goal_pos.data());
    auto goal_rot =
        challenge_detail::Mat9ToEuler(data_->site_xmat + goal_sid_ * 9);
    state["info:goal_pos"_].Assign(goal_pos.data(), 3);
    state["info:goal_rot"_].Assign(goal_rot.data(), 3);
    if constexpr (kFromPixels) {
      AssignPixelObservation("obs:pixels", &state["obs:pixels"_], reset);
    }
  }
};

template <typename EnvSpecT, bool kFromPixels>
class MyoChallengeRelocateEnvBase : public Env<EnvSpecT>,
                                    public gymnasium_robotics::MujocoRobotEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  struct RewardInfo {
    mjtNum dense_reward{0.0};
    mjtNum reach_dist_term{0.0};
    mjtNum pos_dist_term{0.0};
    mjtNum rot_dist_term{0.0};
    mjtNum act_reg_term{0.0};
    mjtNum sparse{0.0};
    bool solved{false};
    bool done{false};
  };

  bool normalize_act_;
  std::array<mjtNum, 3> target_xyz_low_{};
  std::array<mjtNum, 3> target_xyz_high_{};
  std::array<mjtNum, 3> target_rxryrz_low_{};
  std::array<mjtNum, 3> target_rxryrz_high_{};
  std::array<mjtNum, 3> obj_xyz_low_{};
  std::array<mjtNum, 3> obj_xyz_high_{};
  std::array<mjtNum, 3> obj_geom_low_{};
  std::array<mjtNum, 3> obj_geom_high_{};
  std::array<mjtNum, 3> obj_friction_low_{};
  std::array<mjtNum, 3> obj_friction_high_{};
  bool has_obj_xyz_range_{false};
  bool has_obj_geom_range_{false};
  bool has_obj_mass_range_{false};
  bool has_obj_friction_range_{false};
  mjtNum obj_mass_low_{0.0};
  mjtNum obj_mass_high_{0.0};
  mjtNum qpos_noise_range_{0.0};
  mjtNum pos_th_;
  mjtNum rot_th_;
  mjtNum drop_th_;
  mjtNum reward_pos_dist_w_;
  mjtNum reward_rot_dist_w_;
  mjtNum reward_act_reg_w_;
  int palm_sid_{-1};
  int object_sid_{-1};
  int object_bid_{-1};
  int goal_sid_{-1};
  int success_indicator_sid_{-1};
  int goal_bid_{-1};
  int goal_mocap_id_{-1};
  std::vector<mjtNum> initial_qpos_override_;
  std::vector<mjtNum> default_goal_body_pos_;
  std::vector<mjtNum> default_goal_body_quat_;
  std::vector<mjtNum> default_object_body_pos_;
  mjtNum default_object_body_mass_{0.0};
  std::vector<mjtNum> default_object_geom_size_;
  std::vector<mjtNum> default_object_geom_pos_;
  std::vector<mjtNum> default_object_geom_quat_;
  std::vector<float> default_object_geom_rgba_;
  std::vector<mjtNum> default_object_geom_friction_;
  std::vector<int> default_object_geom_type_;
  std::vector<mjtNum> test_reset_qpos_;
  std::vector<mjtNum> test_reset_qvel_;
  std::vector<mjtNum> test_reset_act_;
  std::vector<mjtNum> test_reset_qacc_warmstart_;
  std::vector<mjtNum> test_goal_body_pos_;
  std::vector<mjtNum> test_goal_body_quat_;
  std::vector<mjtNum> test_object_body_pos_;
  std::vector<mjtNum> test_object_body_mass_;
  std::vector<int> test_object_geom_type_;
  std::vector<mjtNum> test_object_geom_size_;
  std::vector<mjtNum> test_object_geom_pos_;
  std::vector<mjtNum> test_object_geom_quat_;
  std::vector<float> test_object_geom_rgba_;
  std::vector<mjtNum> test_object_geom_friction_;
  std::vector<bool> muscle_actuator_;
  detail::MyoConditionState muscle_condition_state_;

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  MyoChallengeRelocateEnvBase(const Spec& spec, int env_id)
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
        target_xyz_low_(ToArray3(spec.config["target_xyz_low"_])),
        target_xyz_high_(ToArray3(spec.config["target_xyz_high"_])),
        target_rxryrz_low_(ToArray3(spec.config["target_rxryrz_low"_])),
        target_rxryrz_high_(ToArray3(spec.config["target_rxryrz_high"_])),
        pos_th_(spec.config["pos_th"_]),
        rot_th_(spec.config["rot_th"_]),
        drop_th_(spec.config["drop_th"_]),
        reward_pos_dist_w_(spec.config["reward_pos_dist_w"_]),
        reward_rot_dist_w_(spec.config["reward_rot_dist_w"_]),
        reward_act_reg_w_(spec.config["reward_act_reg_w"_]),
        test_reset_qpos_(detail::ToMjtVector(spec.config["test_reset_qpos"_])),
        test_reset_qvel_(detail::ToMjtVector(spec.config["test_reset_qvel"_])),
        test_reset_act_(detail::ToMjtVector(spec.config["test_reset_act"_])),
        test_reset_qacc_warmstart_(
            detail::ToMjtVector(spec.config["test_reset_qacc_warmstart"_])),
        test_goal_body_pos_(
            detail::ToMjtVector(spec.config["test_goal_body_pos"_])),
        test_goal_body_quat_(
            detail::ToMjtVector(spec.config["test_goal_body_quat"_])),
        test_object_body_pos_(
            detail::ToMjtVector(spec.config["test_object_body_pos"_])),
        test_object_body_mass_(
            detail::ToMjtVector(spec.config["test_object_body_mass"_])),
        test_object_geom_type_(challenge_detail::ToIntVector(
            spec.config["test_object_geom_type"_])),
        test_object_geom_size_(
            detail::ToMjtVector(spec.config["test_object_geom_size"_])),
        test_object_geom_pos_(
            detail::ToMjtVector(spec.config["test_object_geom_pos"_])),
        test_object_geom_quat_(
            detail::ToMjtVector(spec.config["test_object_geom_quat"_])),
        test_object_geom_rgba_(challenge_detail::ToFloatVector(
            spec.config["test_object_geom_rgba"_])),
        test_object_geom_friction_(
            detail::ToMjtVector(spec.config["test_object_geom_friction"_])) {
    ParseOptionalRanges(spec);
    ValidateConfig();
    CacheIds();
    detail::BuildMuscleMask(model_, &muscle_actuator_);
    detail::InitializeMyoConditionState(
        model_, spec.config["muscle_condition"_],
        spec.config["fatigue_reset_vec"_], spec.config["fatigue_reset_random"_],
        spec.config["frame_skip"_], this->seed_, &muscle_condition_state_);
    int key_frame_id = has_obj_xyz_range_ ? 1 : 0;
    initial_qpos_override_.assign(
        model_->key_qpos + key_frame_id * model_->nq,
        model_->key_qpos + (key_frame_id + 1) * model_->nq);
    detail::RestoreVector(initial_qpos_override_, data_->qpos);
    mju_zero(data_->qvel, model_->nv);
    mj_forward(model_, data_);
    CacheDefaultState();
    InitializeRobotEnv();
  }

  envpool::mujoco::CameraPolicy RenderCameraPolicy() const override {
    return detail::MyoSuiteRenderCameraPolicy();
  }

  void ConfigureRenderOption(mjvOption* option) const override {
    detail::ConfigureMyoSuiteRenderOptions(option);
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    detail::ResetMyoConditionState(&muscle_condition_state_);
    for (int attempt = 0; attempt < 16; ++attempt) {
      ResetToInitialState();
      RestoreModelState();
      ApplyModelRandomization();
      ApplyResetState();
      if (data_->ncon == 0 || !test_reset_qpos_.empty()) {
        break;
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
    detail::ApplyMyoConditionAdjustments(model_, data_, muscle_actuator_,
                                         &muscle_condition_state_);
    InvalidateRenderCache();
    detail::DoMyoSuiteSimulation(model_, data_, frame_skip_);
    ++elapsed_step_;
    RewardInfo reward = ComputeRewardInfo();
    done_ = reward.done || elapsed_step_ >= max_episode_steps_;
    WriteState(reward, false, reward.dense_reward);
  }

 private:
  static std::array<mjtNum, 3> ToArray3(const std::vector<double>& value) {
    if (value.size() != 3) {
      throw std::runtime_error("Expected a length-3 vector.");
    }
    return {static_cast<mjtNum>(value[0]), static_cast<mjtNum>(value[1]),
            static_cast<mjtNum>(value[2])};
  }

  void ParseOptionalRanges(const Spec& spec) {
    auto load_range = [](const std::vector<double>& low_vec,
                         const std::vector<double>& high_vec, bool* has_range,
                         std::array<mjtNum, 3>* low_out,
                         std::array<mjtNum, 3>* high_out) {
      *has_range = !low_vec.empty();
      if (*has_range) {
        *low_out = ToArray3(low_vec);
        *high_out = ToArray3(high_vec);
      }
    };
    load_range(spec.config["obj_xyz_low"_], spec.config["obj_xyz_high"_],
               &has_obj_xyz_range_, &obj_xyz_low_, &obj_xyz_high_);
    load_range(spec.config["obj_geom_low"_], spec.config["obj_geom_high"_],
               &has_obj_geom_range_, &obj_geom_low_, &obj_geom_high_);
    load_range(spec.config["obj_friction_low"_],
               spec.config["obj_friction_high"_], &has_obj_friction_range_,
               &obj_friction_low_, &obj_friction_high_);
    has_obj_mass_range_ =
        spec.config["obj_mass_high"_] > spec.config["obj_mass_low"_];
    obj_mass_low_ = spec.config["obj_mass_low"_];
    obj_mass_high_ = spec.config["obj_mass_high"_];
    qpos_noise_range_ = spec.config["qpos_noise_range"_];
  }

  void ValidateConfig() {
    if (model_->nq != spec_.config["qpos_dim"_] ||
        model_->nv != spec_.config["qvel_dim"_] ||
        model_->nu != spec_.config["action_dim"_] ||
        model_->na != spec_.config["act_dim"_]) {
      throw std::runtime_error(
          "MyoChallenge Relocate dims do not match model.");
    }
    int expected_obs = (model_->nq - 7) + (model_->nv - 6) + 18 + model_->na;
    if (expected_obs != spec_.config["obs_dim"_]) {
      throw std::runtime_error(
          "MyoChallenge Relocate obs_dim does not match model.");
    }
  }

  void CacheIds() {
    palm_sid_ = mj_name2id(model_, mjOBJ_SITE, "S_grasp");
    object_sid_ = mj_name2id(model_, mjOBJ_SITE, "object_o");
    object_bid_ = mj_name2id(model_, mjOBJ_BODY, "Object");
    goal_sid_ = mj_name2id(model_, mjOBJ_SITE, "target_o");
    success_indicator_sid_ = mj_name2id(model_, mjOBJ_SITE, "target_ball");
    goal_bid_ = mj_name2id(model_, mjOBJ_BODY, "target");
    goal_mocap_id_ = goal_bid_ >= 0 ? model_->body_mocapid[goal_bid_] : -1;
    if (palm_sid_ == -1 || object_sid_ == -1 || object_bid_ == -1 ||
        goal_sid_ == -1 || goal_bid_ == -1) {
      throw std::runtime_error("MyoChallenge Relocate ids missing.");
    }
  }

  void SetGoalPose(const std::vector<mjtNum>& body_pos,
                   const std::vector<mjtNum>& body_quat) {
    detail::RestoreModelBodyPos(model_, goal_bid_, body_pos);
    detail::RestoreModelBodyQuat(model_, goal_bid_, body_quat);
    if (goal_mocap_id_ >= 0) {
      std::memcpy(data_->mocap_pos + goal_mocap_id_ * 3, body_pos.data(),
                  sizeof(mjtNum) * 3);
      std::memcpy(data_->mocap_quat + goal_mocap_id_ * 4, body_quat.data(),
                  sizeof(mjtNum) * 4);
    }
  }

  void CacheDefaultState() {
    detail::CopyModelBodyPos(model_, goal_bid_, &default_goal_body_pos_);
    detail::CopyModelBodyQuat(model_, goal_bid_, &default_goal_body_quat_);
    detail::CopyModelBodyPos(model_, object_bid_, &default_object_body_pos_);
    detail::CopyModelBodyMass(model_, object_bid_, &default_object_body_mass_);
    challenge_detail::CopyBodyGeomSlice3(model_, model_->geom_size, object_bid_,
                                         &default_object_geom_size_);
    challenge_detail::CopyBodyGeomSlice3(model_, model_->geom_pos, object_bid_,
                                         &default_object_geom_pos_);
    challenge_detail::CopyBodyGeomSlice4(model_, model_->geom_quat, object_bid_,
                                         &default_object_geom_quat_);
    challenge_detail::CopyBodyGeomSlice4f(
        model_, model_->geom_rgba, object_bid_, &default_object_geom_rgba_);
    challenge_detail::CopyBodyGeomSlice3(model_, model_->geom_friction,
                                         object_bid_,
                                         &default_object_geom_friction_);
    challenge_detail::CopyBodyGeomTypeSlice(model_, object_bid_,
                                            &default_object_geom_type_);
  }

  void RestoreModelState() {
    SetGoalPose(default_goal_body_pos_, default_goal_body_quat_);
    detail::RestoreModelBodyPos(model_, object_bid_, default_object_body_pos_);
    detail::RestoreModelBodyMass(model_, object_bid_,
                                 default_object_body_mass_);
    challenge_detail::RestoreBodyGeomSlice3(
        model_, model_->geom_size, object_bid_, default_object_geom_size_);
    challenge_detail::RestoreBodyGeomSlice3(
        model_, model_->geom_pos, object_bid_, default_object_geom_pos_);
    challenge_detail::RestoreBodyGeomSlice4(
        model_, model_->geom_quat, object_bid_, default_object_geom_quat_);
    challenge_detail::RestoreBodyGeomSlice4f(
        model_, model_->geom_rgba, object_bid_, default_object_geom_rgba_);
    challenge_detail::RestoreBodyGeomSlice3(model_, model_->geom_friction,
                                            object_bid_,
                                            default_object_geom_friction_);
    challenge_detail::RestoreBodyGeomTypeSlice(model_, object_bid_,
                                               default_object_geom_type_);
  }

  void ApplyModelRandomization() {
    std::vector<mjtNum> goal_body_pos = default_goal_body_pos_;
    std::vector<mjtNum> goal_body_quat = default_goal_body_quat_;
    if (!test_goal_body_pos_.empty()) {
      goal_body_pos = test_goal_body_pos_;
    } else {
      auto goal_pos = challenge_detail::UniformVec3(&gen_, target_xyz_low_,
                                                    target_xyz_high_);
      goal_body_pos.assign(goal_pos.begin(), goal_pos.end());
    }
    if (!test_goal_body_quat_.empty()) {
      goal_body_quat = test_goal_body_quat_;
    } else {
      auto goal_rot = challenge_detail::UniformVec3(&gen_, target_rxryrz_low_,
                                                    target_rxryrz_high_);
      auto quat = challenge_detail::EulerXYZToQuat(goal_rot);
      goal_body_quat.assign(quat.begin(), quat.end());
    }
    SetGoalPose(goal_body_pos, goal_body_quat);

    if (!test_object_body_pos_.empty()) {
      detail::RestoreModelBodyPos(model_, object_bid_, test_object_body_pos_);
    } else if (has_obj_xyz_range_) {
      auto body_pos =
          challenge_detail::UniformVec3(&gen_, obj_xyz_low_, obj_xyz_high_);
      std::memcpy(model_->body_pos + object_bid_ * 3, body_pos.data(),
                  sizeof(mjtNum) * 3);
    }

    if (!test_object_geom_type_.empty()) {
      challenge_detail::RestoreBodyGeomTypeSlice(model_, object_bid_,
                                                 test_object_geom_type_);
    }
    if (!test_object_geom_size_.empty()) {
      challenge_detail::RestoreBodyGeomSlice3(
          model_, model_->geom_size, object_bid_, test_object_geom_size_);
    }
    if (!test_object_geom_pos_.empty()) {
      challenge_detail::RestoreBodyGeomSlice3(
          model_, model_->geom_pos, object_bid_, test_object_geom_pos_);
    }
    if (!test_object_geom_quat_.empty()) {
      challenge_detail::RestoreBodyGeomSlice4(
          model_, model_->geom_quat, object_bid_, test_object_geom_quat_);
    }
    if (!test_object_geom_rgba_.empty()) {
      challenge_detail::RestoreBodyGeomSlice4f(
          model_, model_->geom_rgba, object_bid_, test_object_geom_rgba_);
    }
    if (!test_object_geom_friction_.empty()) {
      challenge_detail::RestoreBodyGeomSlice3(model_, model_->geom_friction,
                                              object_bid_,
                                              test_object_geom_friction_);
    }
    if (!test_object_body_mass_.empty()) {
      detail::RestoreModelBodyMass(model_, object_bid_,
                                   test_object_body_mass_[0]);
    }

    bool using_test_override =
        !test_object_geom_type_.empty() || !test_object_geom_size_.empty() ||
        !test_object_geom_pos_.empty() || !test_object_geom_quat_.empty() ||
        !test_object_geom_rgba_.empty() || !test_object_geom_friction_.empty();
    if (!using_test_override && has_obj_geom_range_) {
      int start_geom = model_->body_geomadr[object_bid_];
      int geom_count = model_->body_geomnum[object_bid_];
      constexpr std::array<int, 5> geom_types = {2, 3, 4, 5, 6};
      const auto half_pi_value = static_cast<mjtNum>(1.5707963267948966);
      std::uniform_int_distribution<int> type_dist(
          0, static_cast<int>(geom_types.size() - 1));
      std::array<mjtNum, 3> half_pi = {
          -half_pi_value,
          -half_pi_value,
          -half_pi_value,
      };
      std::array<mjtNum, 3> pos_half_pi = {
          half_pi_value,
          half_pi_value,
          half_pi_value,
      };
      mjtNum max_size =
          std::max({obj_geom_high_[0], obj_geom_high_[1], obj_geom_high_[2]});
      for (int i = 0; i < geom_count; ++i) {
        int gid = start_geom + i;
        model_->geom_type[gid] = geom_types[type_dist(gen_)];
        auto geom_size =
            challenge_detail::UniformVec3(&gen_, obj_geom_low_, obj_geom_high_);
        std::memcpy(model_->geom_size + gid * 3, geom_size.data(),
                    sizeof(mjtNum) * 3);
        for (int axis = 0; axis < 3; ++axis) {
          model_->geom_aabb[gid * 6 + 3 + axis] = obj_geom_high_[axis];
        }
        model_->geom_rbound[gid] = static_cast<mjtNum>(2.0) * max_size;
        std::array<mjtNum, 3> pos_low = {-geom_size[0], -geom_size[1],
                                         -geom_size[2]};
        auto geom_pos =
            challenge_detail::UniformVec3(&gen_, pos_low, geom_size);
        std::memcpy(model_->geom_pos + gid * 3, geom_pos.data(),
                    sizeof(mjtNum) * 3);
        auto geom_euler =
            challenge_detail::UniformVec3(&gen_, half_pi, pos_half_pi);
        auto geom_quat = challenge_detail::EulerXYZToQuat(geom_euler);
        std::memcpy(model_->geom_quat + gid * 4, geom_quat.data(),
                    sizeof(mjtNum) * 4);
        std::array<mjtNum, 4> rgba_low = {0.2, 0.2, 0.2, 1.0};
        std::array<mjtNum, 4> rgba_high = {0.9, 0.9, 0.9, 1.0};
        for (int axis = 0; axis < 4; ++axis) {
          std::uniform_real_distribution<double> dist(rgba_low[axis],
                                                      rgba_high[axis]);
          model_->geom_rgba[gid * 4 + axis] = static_cast<mjtNum>(dist(gen_));
        }
        if (has_obj_friction_range_) {
          auto friction = challenge_detail::UniformVec3(
              &gen_, obj_friction_low_, obj_friction_high_);
          std::memcpy(model_->geom_friction + gid * 3, friction.data(),
                      sizeof(mjtNum) * 3);
        }
      }
    }
    if (!test_object_body_mass_.empty()) {
      detail::RestoreModelBodyMass(model_, object_bid_,
                                   test_object_body_mass_[0]);
    } else if (has_obj_mass_range_) {
      model_->body_mass[object_bid_] =
          challenge_detail::UniformScalar(&gen_, obj_mass_low_, obj_mass_high_);
    }
  }

  void ApplyResetState() {
    bool used_override_qpos = false;
    if (!test_reset_qpos_.empty()) {
      detail::RestoreVector(test_reset_qpos_, data_->qpos);
      used_override_qpos = true;
    } else if (qpos_noise_range_ != 0.0) {
      std::vector<mjtNum> qpos = initial_qpos_override_;
      for (int joint_id = 0; joint_id < model_->njnt; ++joint_id) {
        int qposadr = model_->jnt_qposadr[joint_id];
        if (qposadr < model_->nq - 6) {
          qpos[qposadr] +=
              qpos_noise_range_ * (model_->jnt_range[joint_id * 2 + 1] -
                                   model_->jnt_range[joint_id * 2]);
        }
      }
      for (int i = model_->nq - 6; i < model_->nq; ++i) {
        qpos[i] = initial_qpos_override_[i];
      }
      detail::RestoreVector(qpos, data_->qpos);
    }
    if (!test_reset_qvel_.empty()) {
      detail::RestoreVector(test_reset_qvel_, data_->qvel);
    }
    if (!used_override_qpos && test_reset_qvel_.empty()) {
      mju_zero(data_->qvel, model_->nv);
    }
    mj_forward(model_, data_);
    bool rerun_forward = false;
    if (!test_reset_act_.empty()) {
      detail::RestoreVector(test_reset_act_, data_->act);
      rerun_forward = true;
    }
    if (!test_reset_qacc_warmstart_.empty()) {
      detail::RestoreVector(test_reset_qacc_warmstart_, data_->qacc_warmstart);
    }
    if (rerun_forward) {
      mj_forward(model_, data_);
    }
  }

  std::vector<mjtNum> Observation() const {
    std::vector<mjtNum> obs;
    obs.reserve((model_->nq - 7) + (model_->nv - 6) + 18 + model_->na);
    obs.insert(obs.end(), data_->qpos, data_->qpos + model_->nq - 7);
    mjtNum dt = Dt();
    for (int i = 0; i < model_->nv - 6; ++i) {
      obs.push_back(data_->qvel[i] * dt);
    }
    std::array<mjtNum, 3> obj_pos{};
    std::array<mjtNum, 3> goal_pos{};
    detail::CopySitePos(model_, data_, object_sid_, obj_pos.data());
    detail::CopySitePos(model_, data_, goal_sid_, goal_pos.data());
    auto pos_err = challenge_detail::Sub3(goal_pos, obj_pos);
    auto obj_rot =
        challenge_detail::Mat9ToEuler(data_->site_xmat + object_sid_ * 9);
    auto goal_rot =
        challenge_detail::Mat9ToEuler(data_->site_xmat + goal_sid_ * 9);
    auto rot_err = challenge_detail::Sub3(goal_rot, obj_rot);
    obs.insert(obs.end(), obj_pos.begin(), obj_pos.end());
    obs.insert(obs.end(), goal_pos.begin(), goal_pos.end());
    obs.insert(obs.end(), pos_err.begin(), pos_err.end());
    obs.insert(obs.end(), obj_rot.begin(), obj_rot.end());
    obs.insert(obs.end(), goal_rot.begin(), goal_rot.end());
    obs.insert(obs.end(), rot_err.begin(), rot_err.end());
    if (model_->na > 0) {
      obs.insert(obs.end(), data_->act, data_->act + model_->na);
    }
    return obs;
  }

  RewardInfo ComputeRewardInfo() {
    std::array<mjtNum, 3> palm_pos{};
    std::array<mjtNum, 3> obj_pos{};
    std::array<mjtNum, 3> goal_pos{};
    detail::CopySitePos(model_, data_, palm_sid_, palm_pos.data());
    detail::CopySitePos(model_, data_, object_sid_, obj_pos.data());
    detail::CopySitePos(model_, data_, goal_sid_, goal_pos.data());
    auto reach_err = challenge_detail::Sub3(palm_pos, obj_pos);
    auto pos_err = challenge_detail::Sub3(goal_pos, obj_pos);
    auto obj_rot =
        challenge_detail::Mat9ToEuler(data_->site_xmat + object_sid_ * 9);
    auto goal_rot =
        challenge_detail::Mat9ToEuler(data_->site_xmat + goal_sid_ * 9);
    auto rot_err = challenge_detail::Sub3(goal_rot, obj_rot);

    mjtNum reach_dist = challenge_detail::Norm3(reach_err);
    mjtNum pos_dist = challenge_detail::Norm3(pos_err);
    mjtNum rot_dist = challenge_detail::Norm3(rot_err);
    mjtNum act_reg = detail::ActReg(model_, data_);
    bool drop = reach_dist > drop_th_;
    RewardInfo info;
    info.reach_dist_term = -reach_dist;
    info.pos_dist_term = -pos_dist;
    info.rot_dist_term = -rot_dist;
    info.act_reg_term = -act_reg;
    info.sparse = -rot_dist - static_cast<mjtNum>(10.0) * pos_dist;
    info.solved = pos_dist < pos_th_ && rot_dist < rot_th_ && !drop;
    info.done = drop;
    info.dense_reward = reward_pos_dist_w_ * info.pos_dist_term +
                        reward_rot_dist_w_ * info.rot_dist_term +
                        reward_act_reg_w_ * info.act_reg_term;
    challenge_detail::SetSiteSuccessColor(model_, success_indicator_sid_,
                                          info.solved);
    if (success_indicator_sid_ >= 0) {
      mjtNum size =
          info.solved ? static_cast<mjtNum>(0.25) : static_cast<mjtNum>(0.1);
      model_->site_size[success_indicator_sid_ * 3 + 0] = size;
      model_->site_size[success_indicator_sid_ * 3 + 1] = size;
      model_->site_size[success_indicator_sid_ * 3 + 2] = size;
    }
    return info;
  }

  void WriteState(const RewardInfo& reward, bool reset, float reward_value) {
    auto obs = Observation();
    auto state = Allocate();
    if constexpr (!kFromPixels) {
      AssignObservation("obs", &state["obs"_], obs.data(), obs.size(), reset);
    }
    state["reward"_] = reward_value;
    state["discount"_] = 1.0f;
    state["done"_] = done_;
    state["trunc"_] = elapsed_step_ >= max_episode_steps_;
    state["elapsed_step"_] = elapsed_step_;
    state["info:reach_dist"_] = reward.reach_dist_term;
    state["info:pos_dist"_] = reward.pos_dist_term;
    state["info:rot_dist"_] = reward.rot_dist_term;
    state["info:act_reg"_] = reward.act_reg_term;
    state["info:sparse"_] = reward.sparse;
    state["info:solved"_] = static_cast<mjtNum>(reward.solved);
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
    state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
    state["info:qacc0"_].Assign(qacc0_.data(), qacc0_.size());
    state["info:qacc_warmstart0"_].Assign(qacc_warmstart0_.data(),
                                          qacc_warmstart0_.size());
#endif
    std::array<mjtNum, 3> goal_pos{};
    detail::CopySitePos(model_, data_, goal_sid_, goal_pos.data());
    auto goal_rot =
        challenge_detail::Mat9ToEuler(data_->site_xmat + goal_sid_ * 9);
    state["info:goal_pos"_].Assign(goal_pos.data(), 3);
    state["info:goal_rot"_].Assign(goal_rot.data(), 3);
    if constexpr (kFromPixels) {
      AssignPixelObservation("obs:pixels", &state["obs:pixels"_], reset);
    }
  }
};

template <typename EnvSpecT, bool kFromPixels>
class MyoChallengeBaodingEnvBase : public Env<EnvSpecT>,
                                   public gymnasium_robotics::MujocoRobotEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  struct RewardInfo {
    mjtNum dense_reward{0.0};
    mjtNum pos_dist_1_term{0.0};
    mjtNum pos_dist_2_term{0.0};
    mjtNum act_reg_term{0.0};
    mjtNum sparse{0.0};
    bool solved{false};
    bool done{false};
  };

  static constexpr mjtNum kCenterX = static_cast<mjtNum>(-0.0125);
  static constexpr mjtNum kCenterY = static_cast<mjtNum>(-0.07);

  bool normalize_act_;
  mjtNum drop_th_;
  mjtNum proximity_th_;
  mjtNum goal_time_period_low_;
  mjtNum goal_time_period_high_;
  mjtNum goal_xrange_low_;
  mjtNum goal_xrange_high_;
  mjtNum goal_yrange_low_;
  mjtNum goal_yrange_high_;
  std::string task_choice_;
  int fixed_task_;
  mjtNum reward_pos_dist_1_w_;
  mjtNum reward_pos_dist_2_w_;
  mjtNum obj_size_low_;
  mjtNum obj_size_high_;
  mjtNum obj_mass_low_;
  mjtNum obj_mass_high_;
  std::array<mjtNum, 3> obj_friction_low_{0.0, 0.0, 0.0};
  std::array<mjtNum, 3> obj_friction_high_{0.0, 0.0, 0.0};
  bool has_obj_size_range_{false};
  bool has_obj_mass_range_{false};
  bool has_obj_friction_range_{false};
  int object1_bid_{-1};
  int object2_bid_{-1};
  int object1_sid_{-1};
  int object2_sid_{-1};
  int object1_gid_{-1};
  int object2_gid_{-1};
  int target1_sid_{-1};
  int target2_sid_{-1};
  std::vector<mjtNum> default_target1_site_pos_;
  std::vector<mjtNum> default_target2_site_pos_;
  std::vector<mjtNum> default_object1_geom_size_;
  std::vector<mjtNum> default_object2_geom_size_;
  std::vector<mjtNum> default_object1_geom_friction_;
  std::vector<mjtNum> default_object2_geom_friction_;
  std::vector<mjtNum> default_object1_geom_rgba_;
  std::vector<mjtNum> default_object2_geom_rgba_;
  mjtNum default_object1_body_mass_{0.0};
  mjtNum default_object2_body_mass_{0.0};
  std::vector<bool> muscle_actuator_;
  int counter_{0};
  int current_task_{2};
  mjtNum x_radius_{0.025};
  mjtNum y_radius_{0.028};
  mjtNum goal_time_period_{5.0};
  mjtNum ball1_starting_angle_{detail::kPi / static_cast<mjtNum>(4.0)};
  mjtNum ball2_starting_angle_{-detail::kPi * static_cast<mjtNum>(3.0 / 4.0)};
  std::vector<mjtNum> test_reset_qpos_;
  std::vector<mjtNum> test_reset_qvel_;
  std::vector<mjtNum> test_reset_act_;
  std::vector<mjtNum> test_reset_qacc_warmstart_;
  int test_task_;
  mjtNum test_ball1_starting_angle_;
  mjtNum test_ball2_starting_angle_;
  mjtNum test_x_radius_;
  mjtNum test_y_radius_;
  std::vector<mjtNum> test_goal_trajectory_;
  std::vector<mjtNum> test_object1_body_mass_;
  std::vector<mjtNum> test_object2_body_mass_;
  std::vector<mjtNum> test_object1_geom_size_;
  std::vector<mjtNum> test_object2_geom_size_;
  std::vector<mjtNum> test_object1_geom_friction_;
  std::vector<mjtNum> test_object2_geom_friction_;
  detail::MyoConditionState muscle_condition_state_;

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  MyoChallengeBaodingEnvBase(const Spec& spec, int env_id)
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
        drop_th_(spec.config["drop_th"_]),
        proximity_th_(spec.config["proximity_th"_]),
        goal_time_period_low_(spec.config["goal_time_period_low"_]),
        goal_time_period_high_(spec.config["goal_time_period_high"_]),
        goal_xrange_low_(spec.config["goal_xrange_low"_]),
        goal_xrange_high_(spec.config["goal_xrange_high"_]),
        goal_yrange_low_(spec.config["goal_yrange_low"_]),
        goal_yrange_high_(spec.config["goal_yrange_high"_]),
        task_choice_(spec.config["task_choice"_]),
        fixed_task_(spec.config["fixed_task"_]),
        reward_pos_dist_1_w_(spec.config["reward_pos_dist_1_w"_]),
        reward_pos_dist_2_w_(spec.config["reward_pos_dist_2_w"_]),
        obj_size_low_(spec.config["obj_size_low"_]),
        obj_size_high_(spec.config["obj_size_high"_]),
        obj_mass_low_(spec.config["obj_mass_low"_]),
        obj_mass_high_(spec.config["obj_mass_high"_]),
        has_obj_size_range_(spec.config["obj_size_low"_] != 0.0 ||
                            spec.config["obj_size_high"_] != 0.0),
        has_obj_mass_range_(spec.config["obj_mass_low"_] != 0.0 ||
                            spec.config["obj_mass_high"_] != 0.0),
        has_obj_friction_range_(!spec.config["obj_friction_low"_].empty() &&
                                !spec.config["obj_friction_high"_].empty()),
        test_reset_qpos_(detail::ToMjtVector(spec.config["test_reset_qpos"_])),
        test_reset_qvel_(detail::ToMjtVector(spec.config["test_reset_qvel"_])),
        test_reset_act_(detail::ToMjtVector(spec.config["test_reset_act"_])),
        test_reset_qacc_warmstart_(
            detail::ToMjtVector(spec.config["test_reset_qacc_warmstart"_])),
        test_task_(spec.config["test_task"_]),
        test_ball1_starting_angle_(spec.config["test_ball1_starting_angle"_]),
        test_ball2_starting_angle_(spec.config["test_ball2_starting_angle"_]),
        test_x_radius_(spec.config["test_x_radius"_]),
        test_y_radius_(spec.config["test_y_radius"_]),
        test_goal_trajectory_(
            detail::ToMjtVector(spec.config["test_goal_trajectory"_])),
        test_object1_body_mass_(
            detail::ToMjtVector(spec.config["test_object1_body_mass"_])),
        test_object2_body_mass_(
            detail::ToMjtVector(spec.config["test_object2_body_mass"_])),
        test_object1_geom_size_(
            detail::ToMjtVector(spec.config["test_object1_geom_size"_])),
        test_object2_geom_size_(
            detail::ToMjtVector(spec.config["test_object2_geom_size"_])),
        test_object1_geom_friction_(
            detail::ToMjtVector(spec.config["test_object1_geom_friction"_])),
        test_object2_geom_friction_(
            detail::ToMjtVector(spec.config["test_object2_geom_friction"_])) {
    auto friction_low = spec.config["obj_friction_low"_];
    auto friction_high = spec.config["obj_friction_high"_];
    if (friction_low.size() == 3 && friction_high.size() == 3) {
      for (int axis = 0; axis < 3; ++axis) {
        obj_friction_low_[axis] = static_cast<mjtNum>(friction_low[axis]);
        obj_friction_high_[axis] = static_cast<mjtNum>(friction_high[axis]);
      }
    }
    ValidateConfig();
    CacheIds();
    detail::BuildMuscleMask(model_, &muscle_actuator_);
    detail::InitializeMyoConditionState(
        model_, spec.config["muscle_condition"_],
        spec.config["fatigue_reset_vec"_], spec.config["fatigue_reset_random"_],
        spec.config["frame_skip"_], this->seed_, &muscle_condition_state_);
    for (int i = 0; i < model_->nq - 14; ++i) {
      data_->qpos[i] = 0.0;
    }
    data_->qpos[0] = static_cast<mjtNum>(-1.57);
    for (int i = 0; i < model_->nv; ++i) {
      data_->qvel[i] = 0.0;
    }
    mj_forward(model_, data_);
    CacheDefaultState();
    InitializeRobotEnv();
  }

  envpool::mujoco::CameraPolicy RenderCameraPolicy() const override {
    return detail::MyoSuiteRenderCameraPolicy();
  }

  void ConfigureRenderOption(mjvOption* option) const override {
    detail::ConfigureMyoSuiteRenderOptions(option);
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    counter_ = 0;
    detail::ResetMyoConditionState(&muscle_condition_state_);
    ResetToInitialState();
    RestoreModelState();
    SampleEpisodeParameters();
    ApplyObjectRandomization();
    UpdateTargetsForCurrentCounter();
    ApplyResetState();
    CaptureResetState();
    RewardInfo reward = ComputeRewardInfo();
    WriteState(reward, true, 0.0);
  }

  void Step(const Action& action) override {
    UpdateTargetsForCurrentCounter();
    const auto* raw = static_cast<const float*>(action["action"_].Data());
    detail::ApplyMyoSuiteAction(model_, data_, muscle_actuator_, normalize_act_,
                                raw);
    detail::ApplyMyoConditionAdjustments(model_, data_, muscle_actuator_,
                                         &muscle_condition_state_);
    InvalidateRenderCache();
    detail::DoMyoSuiteSimulation(model_, data_, frame_skip_);
    ++counter_;
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
      throw std::runtime_error("MyoChallenge Baoding dims do not match model.");
    }
    int expected_obs = (model_->nq - 14) + 24 + model_->na;
    if (expected_obs != spec_.config["obs_dim"_]) {
      throw std::runtime_error(
          "MyoChallenge Baoding obs_dim does not match model.");
    }
  }

  void CacheIds() {
    object1_bid_ = mj_name2id(model_, mjOBJ_BODY, "ball1");
    object2_bid_ = mj_name2id(model_, mjOBJ_BODY, "ball2");
    object1_sid_ = mj_name2id(model_, mjOBJ_SITE, "ball1_site");
    object2_sid_ = mj_name2id(model_, mjOBJ_SITE, "ball2_site");
    object1_gid_ = mj_name2id(model_, mjOBJ_GEOM, "ball1");
    object2_gid_ = mj_name2id(model_, mjOBJ_GEOM, "ball2");
    target1_sid_ = mj_name2id(model_, mjOBJ_SITE, "target1_site");
    target2_sid_ = mj_name2id(model_, mjOBJ_SITE, "target2_site");
    if (object1_bid_ == -1 || object2_bid_ == -1 || object1_sid_ == -1 ||
        object2_sid_ == -1 || object1_gid_ == -1 || object2_gid_ == -1 ||
        target1_sid_ == -1 || target2_sid_ == -1) {
      throw std::runtime_error("MyoChallenge Baoding ids missing.");
    }
    model_->site_group[target1_sid_] = 2;
    model_->site_group[target2_sid_] = 2;
  }

  void CacheDefaultState() {
    detail::CopyModelSitePos(model_, target1_sid_, &default_target1_site_pos_);
    detail::CopyModelSitePos(model_, target2_sid_, &default_target2_site_pos_);
    detail::CopyModelGeomSize(model_, object1_gid_,
                              &default_object1_geom_size_);
    detail::CopyModelGeomSize(model_, object2_gid_,
                              &default_object2_geom_size_);
    detail::CopyModelGeomFriction(model_, object1_gid_,
                                  &default_object1_geom_friction_);
    detail::CopyModelGeomFriction(model_, object2_gid_,
                                  &default_object2_geom_friction_);
    detail::CopyModelGeomRgba(model_, object1_gid_,
                              &default_object1_geom_rgba_);
    detail::CopyModelGeomRgba(model_, object2_gid_,
                              &default_object2_geom_rgba_);
    detail::CopyModelBodyMass(model_, object1_bid_,
                              &default_object1_body_mass_);
    detail::CopyModelBodyMass(model_, object2_bid_,
                              &default_object2_body_mass_);
  }

  void RestoreModelState() {
    detail::RestoreModelSitePos(model_, target1_sid_,
                                default_target1_site_pos_);
    detail::RestoreModelSitePos(model_, target2_sid_,
                                default_target2_site_pos_);
    detail::RestoreModelGeomSize(model_, object1_gid_,
                                 default_object1_geom_size_);
    detail::RestoreModelGeomSize(model_, object2_gid_,
                                 default_object2_geom_size_);
    detail::RestoreModelGeomFriction(model_, object1_gid_,
                                     default_object1_geom_friction_);
    detail::RestoreModelGeomFriction(model_, object2_gid_,
                                     default_object2_geom_friction_);
    detail::RestoreModelGeomRgba(model_, object1_gid_,
                                 default_object1_geom_rgba_);
    detail::RestoreModelGeomRgba(model_, object2_gid_,
                                 default_object2_geom_rgba_);
    detail::RestoreModelBodyMass(model_, object1_bid_,
                                 default_object1_body_mass_);
    detail::RestoreModelBodyMass(model_, object2_bid_,
                                 default_object2_body_mass_);
  }

  void SampleEpisodeParameters() {
    if (test_task_ >= 0) {
      current_task_ = test_task_;
    } else if (task_choice_ == "random") {
      std::uniform_int_distribution<int> task_dist(0, 2);
      current_task_ = task_dist(gen_);
    } else {
      current_task_ = fixed_task_;
    }

    if (!std::isnan(test_ball1_starting_angle_)) {
      ball1_starting_angle_ = test_ball1_starting_angle_;
      ball2_starting_angle_ = std::isnan(test_ball2_starting_angle_)
                                  ? ball1_starting_angle_ - detail::kPi
                                  : test_ball2_starting_angle_;
    } else if (task_choice_ == "random") {
      ball1_starting_angle_ =
          challenge_detail::UniformScalar(&gen_, 0.0, 2.0 * detail::kPi);
      ball2_starting_angle_ = ball1_starting_angle_ - detail::kPi;
    } else {
      ball1_starting_angle_ = detail::kPi / static_cast<mjtNum>(4.0);
      ball2_starting_angle_ =
          ball1_starting_angle_ - static_cast<mjtNum>(detail::kPi);
    }

    x_radius_ = !std::isnan(test_x_radius_)
                    ? test_x_radius_
                    : challenge_detail::UniformScalar(&gen_, goal_xrange_low_,
                                                      goal_xrange_high_);
    y_radius_ = !std::isnan(test_y_radius_)
                    ? test_y_radius_
                    : challenge_detail::UniformScalar(&gen_, goal_yrange_low_,
                                                      goal_yrange_high_);
    goal_time_period_ = challenge_detail::UniformScalar(
        &gen_, goal_time_period_low_, goal_time_period_high_);
  }

  void ApplyObjectRandomization() {
    if (!test_object1_body_mass_.empty()) {
      detail::RestoreModelBodyMass(model_, object1_bid_,
                                   test_object1_body_mass_[0]);
    } else if (has_obj_mass_range_) {
      detail::RestoreModelBodyMass(model_, object1_bid_,
                                   challenge_detail::UniformScalar(
                                       &gen_, obj_mass_low_, obj_mass_high_));
    }
    if (!test_object2_body_mass_.empty()) {
      detail::RestoreModelBodyMass(model_, object2_bid_,
                                   test_object2_body_mass_[0]);
    } else if (has_obj_mass_range_) {
      detail::RestoreModelBodyMass(model_, object2_bid_,
                                   challenge_detail::UniformScalar(
                                       &gen_, obj_mass_low_, obj_mass_high_));
    }

    if (!test_object1_geom_friction_.empty()) {
      detail::RestoreModelGeomFriction(model_, object1_gid_,
                                       test_object1_geom_friction_);
    } else if (has_obj_friction_range_) {
      auto friction = challenge_detail::UniformVec3(&gen_, obj_friction_low_,
                                                    obj_friction_high_);
      detail::RestoreModelGeomFriction(
          model_, object1_gid_,
          std::vector<mjtNum>(friction.begin(), friction.end()));
    }
    if (!test_object2_geom_friction_.empty()) {
      detail::RestoreModelGeomFriction(model_, object2_gid_,
                                       test_object2_geom_friction_);
    } else if (has_obj_friction_range_) {
      auto friction = challenge_detail::UniformVec3(&gen_, obj_friction_low_,
                                                    obj_friction_high_);
      detail::RestoreModelGeomFriction(
          model_, object2_gid_,
          std::vector<mjtNum>(friction.begin(), friction.end()));
    }

    if (!test_object1_geom_size_.empty()) {
      detail::RestoreModelGeomSize(model_, object1_gid_,
                                   test_object1_geom_size_);
    } else if (has_obj_size_range_) {
      mjtNum size =
          challenge_detail::UniformScalar(&gen_, obj_size_low_, obj_size_high_);
      detail::RestoreModelGeomSize(model_, object1_gid_, {size, size, size});
    }
    if (!test_object2_geom_size_.empty()) {
      detail::RestoreModelGeomSize(model_, object2_gid_,
                                   test_object2_geom_size_);
    } else if (has_obj_size_range_) {
      mjtNum size =
          challenge_detail::UniformScalar(&gen_, obj_size_low_, obj_size_high_);
      detail::RestoreModelGeomSize(model_, object2_gid_, {size, size, size});
    }
  }

  std::array<mjtNum, 2> GoalAnglesAt(int counter) const {
    if (!test_goal_trajectory_.empty()) {
      int length = static_cast<int>(test_goal_trajectory_.size() / 2);
      int index = std::min(counter, std::max(length - 1, 0));
      return {test_goal_trajectory_[index * 2 + 0],
              test_goal_trajectory_[index * 2 + 1]};
    }
    mjtNum sign = 0.0;
    if (current_task_ == 1) {
      sign = -1.0;
    } else if (current_task_ == 2) {
      sign = 1.0;
    }
    mjtNum angle = sign * static_cast<mjtNum>(2.0) * detail::kPi *
                   (static_cast<mjtNum>(counter) * Dt() / goal_time_period_);
    return {angle, angle};
  }

  void UpdateTargetsForCurrentCounter() {
    auto angles = GoalAnglesAt(counter_);
    angles[0] += ball1_starting_angle_;
    angles[1] += ball2_starting_angle_;
    std::vector<mjtNum> target1 = default_target1_site_pos_;
    std::vector<mjtNum> target2 = default_target2_site_pos_;
    target1[0] = x_radius_ * std::cos(angles[0]) + kCenterX;
    target1[1] = y_radius_ * std::sin(angles[0]) + kCenterY;
    target2[0] = x_radius_ * std::cos(angles[1]) + kCenterX;
    target2[1] = y_radius_ * std::sin(angles[1]) + kCenterY;
    detail::RestoreModelSitePos(model_, target1_sid_, target1);
    detail::RestoreModelSitePos(model_, target2_sid_, target2);
  }

  void ApplyResetState() {
    if (!test_reset_qpos_.empty()) {
      detail::RestoreVector(test_reset_qpos_, data_->qpos);
    }
    if (!test_reset_qvel_.empty()) {
      detail::RestoreVector(test_reset_qvel_, data_->qvel);
    }
    mj_forward(model_, data_);
    bool rerun_forward = false;
    if (!test_reset_act_.empty()) {
      detail::RestoreVector(test_reset_act_, data_->act);
      rerun_forward = true;
    }
    if (!test_reset_qacc_warmstart_.empty()) {
      detail::RestoreVector(test_reset_qacc_warmstart_, data_->qacc_warmstart);
    }
    if (rerun_forward) {
      mj_forward(model_, data_);
    }
  }

  std::vector<mjtNum> Observation() const {
    std::vector<mjtNum> obs;
    obs.reserve((model_->nq - 14) + 24 + model_->na);
    obs.insert(obs.end(), data_->qpos, data_->qpos + model_->nq - 14);
    std::array<mjtNum, 3> object1_pos{};
    std::array<mjtNum, 3> object2_pos{};
    std::array<mjtNum, 3> target1_pos{};
    std::array<mjtNum, 3> target2_pos{};
    detail::CopySitePos(model_, data_, object1_sid_, object1_pos.data());
    detail::CopySitePos(model_, data_, object2_sid_, object2_pos.data());
    detail::CopySitePos(model_, data_, target1_sid_, target1_pos.data());
    detail::CopySitePos(model_, data_, target2_sid_, target2_pos.data());
    obs.insert(obs.end(), object1_pos.begin(), object1_pos.end());
    for (int i = model_->nv - 12; i < model_->nv - 9; ++i) {
      obs.push_back(data_->qvel[i] * Dt());
    }
    obs.insert(obs.end(), object2_pos.begin(), object2_pos.end());
    for (int i = model_->nv - 6; i < model_->nv - 3; ++i) {
      obs.push_back(data_->qvel[i] * Dt());
    }
    obs.insert(obs.end(), target1_pos.begin(), target1_pos.end());
    obs.insert(obs.end(), target2_pos.begin(), target2_pos.end());
    auto target1_err = challenge_detail::Sub3(target1_pos, object1_pos);
    auto target2_err = challenge_detail::Sub3(target2_pos, object2_pos);
    obs.insert(obs.end(), target1_err.begin(), target1_err.end());
    obs.insert(obs.end(), target2_err.begin(), target2_err.end());
    if (model_->na > 0) {
      obs.insert(obs.end(), data_->act, data_->act + model_->na);
    }
    return obs;
  }

  RewardInfo ComputeRewardInfo() {
    std::array<mjtNum, 3> object1_pos{};
    std::array<mjtNum, 3> object2_pos{};
    std::array<mjtNum, 3> target1_pos{};
    std::array<mjtNum, 3> target2_pos{};
    detail::CopySitePos(model_, data_, object1_sid_, object1_pos.data());
    detail::CopySitePos(model_, data_, object2_sid_, object2_pos.data());
    detail::CopySitePos(model_, data_, target1_sid_, target1_pos.data());
    detail::CopySitePos(model_, data_, target2_sid_, target2_pos.data());
    auto target1_err = challenge_detail::Sub3(target1_pos, object1_pos);
    auto target2_err = challenge_detail::Sub3(target2_pos, object2_pos);
    mjtNum target1_dist = challenge_detail::Norm3(target1_err);
    mjtNum target2_dist = challenge_detail::Norm3(target2_err);
    mjtNum act_reg = detail::ActReg(model_, data_);
    bool is_fall = object1_pos[2] < drop_th_ || object2_pos[2] < drop_th_;

    RewardInfo info;
    info.pos_dist_1_term = -target1_dist;
    info.pos_dist_2_term = -target2_dist;
    info.act_reg_term = -act_reg;
    info.sparse = -(target1_dist + target2_dist);
    info.solved = target1_dist < proximity_th_ &&
                  target2_dist < proximity_th_ && !is_fall;
    info.done = is_fall;
    info.dense_reward = reward_pos_dist_1_w_ * info.pos_dist_1_term +
                        reward_pos_dist_2_w_ * info.pos_dist_2_term;

    std::vector<mjtNum> rgba1 = default_object1_geom_rgba_;
    std::vector<mjtNum> rgba2 = default_object2_geom_rgba_;
    rgba1[0] = target1_dist < proximity_th_ ? 1.0 : 0.5;
    rgba1[1] = target1_dist < proximity_th_ ? 1.0 : 0.5;
    rgba2[0] = target1_dist < proximity_th_ ? 0.9 : 0.5;
    rgba2[1] = target1_dist < proximity_th_ ? 0.7 : 0.5;
    detail::RestoreModelGeomRgba(model_, object1_gid_, rgba1);
    detail::RestoreModelGeomRgba(model_, object2_gid_, rgba2);
    return info;
  }

  void WriteState(const RewardInfo& reward, bool reset, float reward_value) {
    auto obs = Observation();
    auto state = Allocate();
    if constexpr (!kFromPixels) {
      AssignObservation("obs", &state["obs"_], obs.data(), obs.size(), reset);
    }
    state["reward"_] = reward_value;
    state["discount"_] = 1.0f;
    state["done"_] = done_;
    state["trunc"_] = elapsed_step_ >= max_episode_steps_;
    state["elapsed_step"_] = elapsed_step_;
    state["info:pos_dist_1"_] = reward.pos_dist_1_term;
    state["info:pos_dist_2"_] = reward.pos_dist_2_term;
    state["info:act_reg"_] = reward.act_reg_term;
    state["info:sparse"_] = reward.sparse;
    state["info:solved"_] = static_cast<mjtNum>(reward.solved);
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
    state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
    state["info:qacc0"_].Assign(qacc0_.data(), qacc0_.size());
    state["info:qacc_warmstart0"_].Assign(qacc_warmstart0_.data(),
                                          qacc_warmstart0_.size());
#endif
    std::array<mjtNum, 3> target1_pos{};
    std::array<mjtNum, 3> target2_pos{};
    detail::CopySitePos(model_, data_, target1_sid_, target1_pos.data());
    detail::CopySitePos(model_, data_, target2_sid_, target2_pos.data());
    state["info:target1_pos"_].Assign(target1_pos.data(), 3);
    state["info:target2_pos"_].Assign(target2_pos.data(), 3);
    if constexpr (kFromPixels) {
      AssignPixelObservation("obs:pixels", &state["obs:pixels"_], reset);
    }
  }
};

template <typename EnvSpecT, bool kFromPixels>
class MyoChallengeBimanualEnvBase : public Env<EnvSpecT>,
                                    public gymnasium_robotics::MujocoRobotEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  struct RewardInfo {
    mjtNum dense_reward{0.0};
    mjtNum reach_dist{0.0};
    mjtNum act{0.0};
    mjtNum fin_dis{0.0};
    mjtNum pass_err{0.0};
    mjtNum goal_dist{0.0};
    bool solved{false};
    bool done{false};
  };

  static constexpr mjtNum kPillarHeight = static_cast<mjtNum>(1.09);
  static constexpr mjtNum kMaxTime = static_cast<mjtNum>(10.0);
  static constexpr int kTargetGoalTouch = 10;

  bool normalize_act_;
  mjtNum proximity_th_;
  std::array<mjtNum, 3> start_center_;
  std::array<mjtNum, 3> goal_center_;
  std::array<mjtNum, 3> start_shifts_;
  std::array<mjtNum, 3> goal_shifts_;
  mjtNum reward_reach_dist_w_;
  mjtNum reward_act_w_;
  mjtNum reward_fin_dis_w_;
  mjtNum reward_pass_err_w_;
  std::array<mjtNum, 3> obj_scale_change_{0.0, 0.0, 0.0};
  bool has_obj_scale_change_{false};
  bool has_obj_mass_range_{false};
  bool has_obj_friction_range_{false};
  mjtNum obj_mass_low_{0.0};
  mjtNum obj_mass_high_{0.0};
  std::array<mjtNum, 3> obj_friction_low_{0.0, 0.0, 0.0};
  std::array<mjtNum, 3> obj_friction_high_{0.0, 0.0, 0.0};
  int start_bid_{-1};
  int goal_bid_{-1};
  int obj_bid_{-1};
  int obj_sid_{-1};
  int obj_gid_{-1};
  int palm_sid_{-1};
  int rpalm1_sid_{-1};
  int rpalm2_sid_{-1};
  std::array<int, 5> fin_sid_{-1, -1, -1, -1, -1};
  int max_force_sensor_adr_{0};
  int manip_qpos_start_{-1};
  int manip_dof_start_{-1};
  int myo_body_min_{-1};
  int myo_body_max_{-1};
  int prosth_body_min_{-1};
  int prosth_body_max_{-1};
  std::vector<int> myo_joint_qpos_indices_;
  std::vector<int> myo_dof_indices_;
  std::vector<int> prosth_joint_qpos_indices_;
  std::vector<int> prosth_dof_indices_;
  std::vector<mjtNum> default_start_body_pos_;
  std::vector<mjtNum> default_goal_body_pos_;
  std::vector<mjtNum> default_object_geom_size_;
  std::vector<mjtNum> default_object_geom_friction_;
  mjtNum default_object_body_mass_{0.0};
  std::vector<bool> muscle_actuator_;
  std::vector<mjtNum> grasp_key_qpos_;
  std::array<mjtNum, 3> start_pos_{0.0, 0.0, 0.0};
  std::array<mjtNum, 3> goal_pos_{0.0, 0.0, 0.0};
  mjtNum max_force_seen_{0.0};
  int goal_touch_{0};
  mjtNum init_obj_z_{0.0};
  mjtNum init_palm_z_{0.0};
  std::vector<mjtNum> test_reset_qpos_;
  std::vector<mjtNum> test_reset_qvel_;
  std::vector<mjtNum> test_reset_act_;
  std::vector<mjtNum> test_reset_qacc_warmstart_;
  std::vector<mjtNum> test_start_pos_;
  std::vector<mjtNum> test_goal_pos_;
  std::vector<mjtNum> test_object_body_mass_;
  std::vector<mjtNum> test_object_geom_size_;
  std::vector<mjtNum> test_object_geom_friction_;
  detail::MyoConditionState muscle_condition_state_;

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  MyoChallengeBimanualEnvBase(const Spec& spec, int env_id)
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
        proximity_th_(spec.config["proximity_th"_]),
        reward_reach_dist_w_(spec.config["reward_reach_dist_w"_]),
        reward_act_w_(spec.config["reward_act_w"_]),
        reward_fin_dis_w_(spec.config["reward_fin_dis_w"_]),
        reward_pass_err_w_(spec.config["reward_pass_err_w"_]),
        has_obj_scale_change_(!spec.config["obj_scale_change"_].empty()),
        has_obj_mass_range_(spec.config["obj_mass_low"_] != 0.0 ||
                            spec.config["obj_mass_high"_] != 0.0),
        has_obj_friction_range_(!spec.config["obj_friction_low"_].empty() &&
                                !spec.config["obj_friction_high"_].empty()),
        obj_mass_low_(spec.config["obj_mass_low"_]),
        obj_mass_high_(spec.config["obj_mass_high"_]),
        test_reset_qpos_(detail::ToMjtVector(spec.config["test_reset_qpos"_])),
        test_reset_qvel_(detail::ToMjtVector(spec.config["test_reset_qvel"_])),
        test_reset_act_(detail::ToMjtVector(spec.config["test_reset_act"_])),
        test_reset_qacc_warmstart_(
            detail::ToMjtVector(spec.config["test_reset_qacc_warmstart"_])),
        test_start_pos_(detail::ToMjtVector(spec.config["test_start_pos"_])),
        test_goal_pos_(detail::ToMjtVector(spec.config["test_goal_pos"_])),
        test_object_body_mass_(
            detail::ToMjtVector(spec.config["test_object_body_mass"_])),
        test_object_geom_size_(
            detail::ToMjtVector(spec.config["test_object_geom_size"_])),
        test_object_geom_friction_(
            detail::ToMjtVector(spec.config["test_object_geom_friction"_])) {
    auto start_center = spec.config["start_center"_];
    auto goal_center = spec.config["goal_center"_];
    auto start_shifts = spec.config["start_shifts"_];
    auto goal_shifts = spec.config["goal_shifts"_];
    auto obj_scale_change = spec.config["obj_scale_change"_];
    auto obj_friction_low = spec.config["obj_friction_low"_];
    auto obj_friction_high = spec.config["obj_friction_high"_];
    for (int axis = 0; axis < 3; ++axis) {
      start_center_[axis] = static_cast<mjtNum>(start_center[axis]);
      goal_center_[axis] = static_cast<mjtNum>(goal_center[axis]);
      start_shifts_[axis] = static_cast<mjtNum>(start_shifts[axis]);
      goal_shifts_[axis] = static_cast<mjtNum>(goal_shifts[axis]);
      if (has_obj_scale_change_) {
        obj_scale_change_[axis] = static_cast<mjtNum>(obj_scale_change[axis]);
      }
      if (has_obj_friction_range_) {
        obj_friction_low_[axis] = static_cast<mjtNum>(obj_friction_low[axis]);
        obj_friction_high_[axis] = static_cast<mjtNum>(obj_friction_high[axis]);
      }
    }
    ValidateConfig();
    CacheIds();
    detail::BuildMuscleMask(model_, &muscle_actuator_);
    detail::InitializeMyoConditionState(
        model_, spec.config["muscle_condition"_],
        spec.config["fatigue_reset_vec"_], spec.config["fatigue_reset_random"_],
        spec.config["frame_skip"_], this->seed_, &muscle_condition_state_);
    grasp_key_qpos_.assign(model_->key_qpos + 2 * model_->nq,
                           model_->key_qpos + 3 * model_->nq);
    InitializeRobotEnv();
  }

  envpool::mujoco::CameraPolicy RenderCameraPolicy() const override {
    return detail::MyoSuiteRenderCameraPolicy();
  }

  void ConfigureRenderOption(mjvOption* option) const override {
    detail::ConfigureMyoSuiteRenderOptions(option);
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    max_force_seen_ = 0.0;
    goal_touch_ = 0;
    detail::ResetMyoConditionState(&muscle_condition_state_);
    ResetToInitialState();
    RestoreModelState();
    SampleStartAndGoal();
    ApplyObjectRandomization();
    ApplyResetState();
    SetObjectAtStartPose();
    CaptureResetState();
    RewardInfo reward = ComputeRewardInfo();
    WriteState(reward, true, 0.0);
  }

  void Step(const Action& action) override {
    const auto* raw = static_cast<const float*>(action["action"_].Data());
    for (int actuator = 0; actuator < model_->nu; ++actuator) {
      auto value = static_cast<mjtNum>(raw[actuator]);
      if (normalize_act_ && muscle_actuator_[actuator] && model_->na != 0) {
        value = detail::MuscleActivation(detail::ClampNormalized(value));
      } else if (normalize_act_ && !muscle_actuator_[actuator]) {
        mjtNum low = model_->actuator_ctrlrange[2 * actuator + 0];
        mjtNum high = model_->actuator_ctrlrange[2 * actuator + 1];
        value = 0.5 * (low + high) + value * 0.5 * (high - low);
      }
      data_->ctrl[actuator] = value;
    }
    detail::ApplyMyoConditionAdjustments(model_, data_, muscle_actuator_,
                                         &muscle_condition_state_);
    InvalidateRenderCache();
    detail::DoMyoSuiteSimulation(model_, data_, frame_skip_);
    ++elapsed_step_;
    RewardInfo reward = ComputeRewardInfo();
    done_ = reward.done || elapsed_step_ >= max_episode_steps_;
    WriteState(reward, false, reward.dense_reward);
  }

 private:
  static bool StartsWith(std::string_view value, std::string_view prefix) {
    return value.substr(0, prefix.size()) == prefix;
  }

  void ValidateConfig() {
    if (model_->nq != spec_.config["qpos_dim"_] ||
        model_->nv != spec_.config["qvel_dim"_] ||
        model_->nu != spec_.config["action_dim"_] ||
        model_->na != spec_.config["act_dim"_]) {
      throw std::runtime_error(
          "MyoChallenge Bimanual dims do not match model.");
    }
  }

  void CacheIds() {
    obj_bid_ = mj_name2id(model_, mjOBJ_BODY, "manip_object");
    obj_sid_ = mj_name2id(model_, mjOBJ_SITE, "touch_site");
    obj_gid_ = model_->body_geomadr[obj_bid_] + 1;
    start_bid_ = mj_name2id(model_, mjOBJ_BODY, "start");
    goal_bid_ = mj_name2id(model_, mjOBJ_BODY, "goal");
    palm_sid_ = mj_name2id(model_, mjOBJ_SITE, "S_grasp");
    fin_sid_[0] = mj_name2id(model_, mjOBJ_SITE, "THtip");
    fin_sid_[1] = mj_name2id(model_, mjOBJ_SITE, "IFtip");
    fin_sid_[2] = mj_name2id(model_, mjOBJ_SITE, "MFtip");
    fin_sid_[3] = mj_name2id(model_, mjOBJ_SITE, "RFtip");
    fin_sid_[4] = mj_name2id(model_, mjOBJ_SITE, "LFtip");
    rpalm1_sid_ = mj_name2id(model_, mjOBJ_SITE, "prosthesis/palm_thumb");
    rpalm2_sid_ = mj_name2id(model_, mjOBJ_SITE, "prosthesis/palm_pinky");
    if (obj_bid_ == -1 || obj_sid_ == -1 || start_bid_ == -1 ||
        goal_bid_ == -1 || palm_sid_ == -1 || rpalm1_sid_ == -1 ||
        rpalm2_sid_ == -1) {
      throw std::runtime_error("MyoChallenge Bimanual ids missing.");
    }

    CenterBoxMeshIfNeeded();

    myo_body_min_ = model_->nbody;
    myo_body_max_ = -1;
    prosth_body_min_ = model_->nbody;
    prosth_body_max_ = -1;
    for (int body = 0; body < model_->nbody; ++body) {
      const char* raw_name = mj_id2name(model_, mjOBJ_BODY, body);
      std::string_view name = raw_name != nullptr ? raw_name : "";
      if (StartsWith(name, "prosthesis/")) {
        prosth_body_min_ = std::min(prosth_body_min_, body);
        prosth_body_max_ = std::max(prosth_body_max_, body);
      } else if (name != "start" && name != "goal" && name != "manip_object") {
        myo_body_min_ = std::min(myo_body_min_, body);
        myo_body_max_ = std::max(myo_body_max_, body);
      }
    }

    for (int joint = 0; joint < model_->njnt; ++joint) {
      const char* raw_name = mj_id2name(model_, mjOBJ_JOINT, joint);
      std::string_view name = raw_name != nullptr ? raw_name : "";
      if (name == "manip_object/freejoint") {
        manip_qpos_start_ = model_->jnt_qposadr[joint];
        manip_dof_start_ = model_->jnt_dofadr[joint];
        continue;
      }
      if (StartsWith(name, "prosthesis")) {
        prosth_joint_qpos_indices_.push_back(model_->jnt_qposadr[joint]);
        prosth_dof_indices_.push_back(model_->jnt_dofadr[joint]);
      } else {
        myo_joint_qpos_indices_.push_back(model_->jnt_qposadr[joint]);
        myo_dof_indices_.push_back(model_->jnt_dofadr[joint]);
      }
    }

    detail::CopyModelBodyPos(model_, start_bid_, &default_start_body_pos_);
    detail::CopyModelBodyPos(model_, goal_bid_, &default_goal_body_pos_);
    detail::CopyModelGeomSize(model_, obj_gid_, &default_object_geom_size_);
    detail::CopyModelGeomFriction(model_, obj_gid_,
                                  &default_object_geom_friction_);
    detail::CopyModelBodyMass(model_, obj_bid_, &default_object_body_mass_);
    max_force_sensor_adr_ = model_->sensor_adr[0];

    int expected_obs = 1 + static_cast<int>(myo_joint_qpos_indices_.size()) +
                       static_cast<int>(myo_dof_indices_.size()) +
                       static_cast<int>(prosth_joint_qpos_indices_.size()) +
                       static_cast<int>(prosth_dof_indices_.size()) + 7 + 6 +
                       5 + model_->na;
    if (expected_obs != spec_.config["obs_dim"_]) {
      throw std::runtime_error(
          "MyoChallenge Bimanual obs_dim does not match model.");
    }
  }

  void CenterBoxMeshIfNeeded() {
    if (!has_obj_scale_change_ || obj_gid_ <= 0) {
      return;
    }
    int box_mesh = -1;
    for (int mesh = 0; mesh < model_->nmesh; ++mesh) {
      const char* raw_name = mj_id2name(model_, mjOBJ_MESH, mesh);
      std::string_view name = raw_name != nullptr ? raw_name : "";
      if (name.find("box") != std::string_view::npos) {
        box_mesh = mesh;
        break;
      }
    }
    if (box_mesh < 0) {
      return;
    }
    int vert_adr = model_->mesh_vertadr[box_mesh];
    int vert_num = model_->mesh_vertnum[box_mesh];
    const mjtNum* mesh_quat = model_->geom_quat + (obj_gid_ - 1) * 4;
    std::array<mjtNum, 3> mesh_pos{};
    std::array<mjtNum, 3> box_pos{};
    std::memcpy(mesh_pos.data(), model_->geom_pos + (obj_gid_ - 1) * 3,
                sizeof(mjtNum) * 3);
    std::memcpy(box_pos.data(), model_->geom_pos + obj_gid_ * 3,
                sizeof(mjtNum) * 3);
    for (int vertex = 0; vertex < vert_num; ++vertex) {
      float* vert = model_->mesh_vert + (vert_adr + vertex) * 3;
      auto rotated_vert = challenge_detail::RotateByQuat(mesh_quat, vert);
      auto rotated_normal = challenge_detail::RotateByQuat(
          mesh_quat, model_->mesh_normal + (vert_adr + vertex) * 3);
      for (int axis = 0; axis < 3; ++axis) {
        vert[axis] = static_cast<float>(rotated_vert[axis] + mesh_pos[axis] -
                                        box_pos[axis]);
        model_->mesh_normal[(vert_adr + vertex) * 3 + axis] =
            rotated_normal[axis];
      }
    }
    model_->geom_quat[(obj_gid_ - 1) * 4 + 0] = 1.0;
    model_->geom_quat[(obj_gid_ - 1) * 4 + 1] = 0.0;
    model_->geom_quat[(obj_gid_ - 1) * 4 + 2] = 0.0;
    model_->geom_quat[(obj_gid_ - 1) * 4 + 3] = 0.0;
    std::memcpy(model_->geom_pos + (obj_gid_ - 1) * 3, box_pos.data(),
                sizeof(mjtNum) * 3);
  }

  void RestoreModelState() {
    detail::RestoreModelBodyPos(model_, start_bid_, default_start_body_pos_);
    detail::RestoreModelBodyPos(model_, goal_bid_, default_goal_body_pos_);
    detail::RestoreModelGeomSize(model_, obj_gid_, default_object_geom_size_);
    detail::RestoreModelGeomFriction(model_, obj_gid_,
                                     default_object_geom_friction_);
    detail::RestoreModelBodyMass(model_, obj_bid_, default_object_body_mass_);
  }

  void SampleStartAndGoal() {
    if (!test_start_pos_.empty()) {
      std::copy_n(test_start_pos_.begin(), 3, start_pos_.begin());
    } else {
      for (int axis = 0; axis < 3; ++axis) {
        start_pos_[axis] =
            start_center_[axis] +
            start_shifts_[axis] *
                challenge_detail::UniformScalar(&gen_, -1.0, 1.0);
      }
    }
    if (!test_goal_pos_.empty()) {
      std::copy_n(test_goal_pos_.begin(), 3, goal_pos_.begin());
    } else {
      for (int axis = 0; axis < 3; ++axis) {
        goal_pos_[axis] = goal_center_[axis] +
                          goal_shifts_[axis] *
                              challenge_detail::UniformScalar(&gen_, -1.0, 1.0);
      }
    }
    detail::RestoreModelBodyPos(model_, start_bid_,
                                {start_pos_[0], start_pos_[1], start_pos_[2]});
    detail::RestoreModelBodyPos(model_, goal_bid_,
                                {goal_pos_[0], goal_pos_[1], goal_pos_[2]});
  }

  void ApplyObjectRandomization() {
    if (!test_object_body_mass_.empty()) {
      detail::RestoreModelBodyMass(model_, obj_bid_, test_object_body_mass_[0]);
    } else if (has_obj_mass_range_) {
      detail::RestoreModelBodyMass(model_, obj_bid_,
                                   challenge_detail::UniformScalar(
                                       &gen_, obj_mass_low_, obj_mass_high_));
    }
    if (!test_object_geom_friction_.empty()) {
      detail::RestoreModelGeomFriction(model_, obj_gid_,
                                       test_object_geom_friction_);
    } else if (has_obj_friction_range_) {
      auto friction = challenge_detail::UniformVec3(&gen_, obj_friction_low_,
                                                    obj_friction_high_);
      detail::RestoreModelGeomFriction(
          model_, obj_gid_,
          std::vector<mjtNum>(friction.begin(), friction.end()));
    }
    if (!test_object_geom_size_.empty()) {
      detail::RestoreModelGeomSize(model_, obj_gid_, test_object_geom_size_);
    } else if (has_obj_scale_change_) {
      std::vector<mjtNum> scaled = default_object_geom_size_;
      for (int axis = 0; axis < 3; ++axis) {
        mjtNum delta = challenge_detail::UniformScalar(
            &gen_, -obj_scale_change_[axis], obj_scale_change_[axis]);
        scaled[axis] = default_object_geom_size_[axis] * (1.0 + delta);
      }
      detail::RestoreModelGeomSize(model_, obj_gid_, scaled);
    }
  }

  void ApplyResetState() {
    if (!test_reset_qpos_.empty()) {
      detail::RestoreVector(test_reset_qpos_, data_->qpos);
    } else {
      detail::RestoreVector(grasp_key_qpos_, data_->qpos);
    }
    if (!test_reset_qvel_.empty()) {
      detail::RestoreVector(test_reset_qvel_, data_->qvel);
    } else {
      std::fill(data_->qvel, data_->qvel + model_->nv, 0.0);
    }
    mj_forward(model_, data_);
    bool rerun_forward = false;
    if (!test_reset_act_.empty()) {
      detail::RestoreVector(test_reset_act_, data_->act);
      rerun_forward = true;
    }
    if (!test_reset_qacc_warmstart_.empty()) {
      detail::RestoreVector(test_reset_qacc_warmstart_, data_->qacc_warmstart);
    }
    if (rerun_forward) {
      mj_forward(model_, data_);
    }
  }

  void SetObjectAtStartPose() {
    data_->qpos[manip_qpos_start_ + 0] = start_pos_[0];
    data_->qpos[manip_qpos_start_ + 1] = start_pos_[1];
    data_->qpos[manip_qpos_start_ + 2] =
        start_pos_[2] + static_cast<mjtNum>(0.1);
    init_obj_z_ = data_->site_xpos[obj_sid_ * 3 + 2];
    init_palm_z_ = data_->site_xpos[palm_sid_ * 3 + 2];
  }

  enum class ObjLabel : std::uint8_t {
    MYO = 0,
    PROSTH = 1,
    START = 2,
    GOAL = 3,
    ENV = 4,
  };

  ObjLabel ClassifyBody(int body_id) const {
    if (myo_body_min_ <= body_id && body_id <= myo_body_max_) {
      return ObjLabel::MYO;
    }
    if (prosth_body_min_ <= body_id && body_id <= prosth_body_max_) {
      return ObjLabel::PROSTH;
    }
    if (body_id == start_bid_) {
      return ObjLabel::START;
    }
    if (body_id == goal_bid_) {
      return ObjLabel::GOAL;
    }
    return ObjLabel::ENV;
  }

  std::array<mjtNum, 5> TouchVector() {
    std::array<mjtNum, 5> touch = {0.0, 0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < data_->ncon; ++i) {
      const mjContact& contact = data_->contact[i];
      int body1 = model_->geom_bodyid[contact.geom1];
      int body2 = model_->geom_bodyid[contact.geom2];
      if (body1 == obj_bid_) {
        touch[static_cast<int>(ClassifyBody(body2))] = 1.0;
      } else if (body2 == obj_bid_) {
        touch[static_cast<int>(ClassifyBody(body1))] = 1.0;
      }
    }
    return touch;
  }

  std::array<mjtNum, 3> SitePos(int site_id) const {
    std::array<mjtNum, 3> out{};
    detail::CopySitePos(model_, data_, site_id, out.data());
    return out;
  }

  std::vector<mjtNum> Observation() const {
    std::vector<mjtNum> obs;
    obs.reserve(spec_.config["obs_dim"_]);
    obs.push_back(data_->time);
    for (int idx : myo_joint_qpos_indices_) {
      obs.push_back(data_->qpos[idx]);
    }
    for (int idx : myo_dof_indices_) {
      obs.push_back(data_->qvel[idx]);
    }
    for (int idx : prosth_joint_qpos_indices_) {
      obs.push_back(data_->qpos[idx]);
    }
    for (int idx : prosth_dof_indices_) {
      obs.push_back(data_->qvel[idx]);
    }
    obs.insert(obs.end(), data_->qpos + manip_qpos_start_,
               data_->qpos + manip_qpos_start_ + 7);
    obs.insert(obs.end(), data_->qvel + manip_dof_start_,
               data_->qvel + manip_dof_start_ + 6);
    auto touch = const_cast<MyoChallengeBimanualEnvBase*>(this)->TouchVector();
    obs.insert(obs.end(), touch.begin(), touch.end());
    if (model_->na > 0) {
      obs.insert(obs.end(), data_->act, data_->act + model_->na);
    }
    return obs;
  }

  RewardInfo ComputeRewardInfo() {
    auto touch = TouchVector();
    if (touch[static_cast<int>(ObjLabel::GOAL)] > 0.0) {
      ++goal_touch_;
    }
    max_force_seen_ = std::max(
        max_force_seen_, std::abs(data_->sensordata[max_force_sensor_adr_]));
    auto palm_pos = SitePos(palm_sid_);
    auto obj_pos = SitePos(obj_sid_);
    std::array<mjtNum, 3> rpalm_pos{};
    for (int axis = 0; axis < 3; ++axis) {
      rpalm_pos[axis] = (data_->site_xpos[rpalm1_sid_ * 3 + axis] +
                         data_->site_xpos[rpalm2_sid_ * 3 + axis]) *
                        static_cast<mjtNum>(0.5);
    }

    auto reach_err = challenge_detail::Sub3(palm_pos, obj_pos);
    auto pass_err = challenge_detail::Sub3(rpalm_pos, obj_pos);
    mjtNum reach_dist = challenge_detail::Norm3(reach_err);
    mjtNum pass_dist = challenge_detail::Norm3(pass_err);
    mjtNum act = detail::ActReg(model_, data_);
    mjtNum fin_dis = 0.0;
    for (int site_id : fin_sid_) {
      fin_dis += challenge_detail::Norm3(
          challenge_detail::Sub3(SitePos(site_id), obj_pos));
    }
    std::array<mjtNum, 3> goal = {goal_pos_[0], goal_pos_[1], kPillarHeight};
    mjtNum goal_dist =
        challenge_detail::Norm3(challenge_detail::Sub3(obj_pos, goal));

    RewardInfo reward;
    reward.reach_dist = reach_dist + std::log(reach_dist + 1e-6);
    reward.act = act;
    reward.fin_dis = fin_dis + std::log(fin_dis + 1e-6);
    reward.pass_err = pass_dist + std::log(pass_dist + 1e-3);
    reward.goal_dist = goal_dist;
    reward.solved =
        goal_dist < proximity_th_ && goal_touch_ >= kTargetGoalTouch;
    reward.done = data_->time > kMaxTime ||
                  obj_pos[2] < static_cast<mjtNum>(0.3) || reward.solved;
    reward.dense_reward = reward_reach_dist_w_ * reward.reach_dist +
                          reward_act_w_ * reward.act +
                          reward_fin_dis_w_ * reward.fin_dis +
                          reward_pass_err_w_ * reward.pass_err;
    return reward;
  }

  void WriteState(const RewardInfo& reward, bool reset, float reward_value) {
    auto obs = Observation();
    auto state = Allocate();
    if constexpr (!kFromPixels) {
      AssignObservation("obs", &state["obs"_], obs.data(), obs.size(), reset);
    }
    state["reward"_] = reward_value;
    state["discount"_] = 1.0f;
    state["done"_] = done_;
    state["trunc"_] = elapsed_step_ >= max_episode_steps_;
    state["elapsed_step"_] = elapsed_step_;
    state["info:reach_dist"_] = reward.reach_dist;
    state["info:act"_] = reward.act;
    state["info:fin_dis"_] = reward.fin_dis;
    state["info:pass_err"_] = reward.pass_err;
    state["info:goal_dist"_] = reward.goal_dist;
    state["info:solved"_] = static_cast<mjtNum>(reward.solved);
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
    state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
    state["info:qacc0"_].Assign(qacc0_.data(), qacc0_.size());
    state["info:qacc_warmstart0"_].Assign(qacc_warmstart0_.data(),
                                          qacc_warmstart0_.size());
#endif
    state["info:start_pos"_].Assign(start_pos_.data(), 3);
    state["info:goal_pos"_].Assign(goal_pos_.data(), 3);
    if constexpr (kFromPixels) {
      AssignPixelObservation("obs:pixels", &state["obs:pixels"_], reset);
    }
  }
};

template <typename Spec>
using ReorientEnvAlias = MyoChallengeReorientEnvBase<Spec, false>;

template <typename Spec>
using ReorientPixelEnvAlias = MyoChallengeReorientEnvBase<Spec, true>;

template <typename Spec>
using RelocateEnvAlias = MyoChallengeRelocateEnvBase<Spec, false>;

template <typename Spec>
using RelocatePixelEnvAlias = MyoChallengeRelocateEnvBase<Spec, true>;

template <typename Spec>
using BaodingEnvAlias = MyoChallengeBaodingEnvBase<Spec, false>;

template <typename Spec>
using BaodingPixelEnvAlias = MyoChallengeBaodingEnvBase<Spec, true>;

template <typename Spec>
using BimanualEnvAlias = MyoChallengeBimanualEnvBase<Spec, false>;

template <typename Spec>
using BimanualPixelEnvAlias = MyoChallengeBimanualEnvBase<Spec, true>;

using ReorientSpec = MyoChallengeReorientEnvSpec;
using ReorientPixelSpec = MyoChallengeReorientPixelEnvSpec;
using RelocateSpec = MyoChallengeRelocateEnvSpec;
using RelocatePixelSpec = MyoChallengeRelocatePixelEnvSpec;
using BaodingSpec = MyoChallengeBaodingEnvSpec;
using BaodingPixelSpec = MyoChallengeBaodingPixelEnvSpec;
using BimanualSpec = MyoChallengeBimanualEnvSpec;
using BimanualPixelSpec = MyoChallengeBimanualPixelEnvSpec;

using ReorientEnv = ReorientEnvAlias<ReorientSpec>;
using ReorientPixelEnv = ReorientPixelEnvAlias<ReorientPixelSpec>;
using MyoChallengeReorientEnv = ReorientEnv;
using MyoChallengeReorientPixelEnv = ReorientPixelEnv;
using MyoChallengeReorientEnvPool = AsyncEnvPool<ReorientEnv>;
using MyoChallengeReorientPixelEnvPool = AsyncEnvPool<ReorientPixelEnv>;

using RelocateEnv = RelocateEnvAlias<RelocateSpec>;
using RelocatePixelEnv = RelocatePixelEnvAlias<RelocatePixelSpec>;
using MyoChallengeRelocateEnv = RelocateEnv;
using MyoChallengeRelocatePixelEnv = RelocatePixelEnv;
using MyoChallengeRelocateEnvPool = AsyncEnvPool<RelocateEnv>;
using MyoChallengeRelocatePixelEnvPool = AsyncEnvPool<RelocatePixelEnv>;

using BaodingEnv = BaodingEnvAlias<BaodingSpec>;
using BaodingPixelEnv = BaodingPixelEnvAlias<BaodingPixelSpec>;
using MyoChallengeBaodingEnv = BaodingEnv;
using MyoChallengeBaodingPixelEnv = BaodingPixelEnv;
using MyoChallengeBaodingEnvPool = AsyncEnvPool<BaodingEnv>;
using MyoChallengeBaodingPixelEnvPool = AsyncEnvPool<BaodingPixelEnv>;

using BimanualEnv = BimanualEnvAlias<BimanualSpec>;
using BimanualPixelEnv = BimanualPixelEnvAlias<BimanualPixelSpec>;
using MyoChallengeBimanualEnv = BimanualEnv;
using MyoChallengeBimanualPixelEnv = BimanualPixelEnv;
using MyoChallengeBimanualEnvPool = AsyncEnvPool<BimanualEnv>;
using MyoChallengeBimanualPixelEnvPool = AsyncEnvPool<BimanualPixelEnv>;

}  // namespace myosuite_envpool

#endif  // ENVPOOL_MUJOCO_MYOSUITE_MYOCHALLENGE_H_
