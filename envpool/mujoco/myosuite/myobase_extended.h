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

#ifndef ENVPOOL_MUJOCO_MYOSUITE_MYOBASE_EXTENDED_H_
#define ENVPOOL_MUJOCO_MYOSUITE_MYOBASE_EXTENDED_H_

#include <mujoco.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "envpool/mujoco/myosuite/myobase.h"

namespace myosuite_envpool {

namespace detail {

inline std::array<mjtNum, 4> EulerXYZToQuat(mjtNum rx, mjtNum ry, mjtNum rz) {
  mjtNum cx = std::cos(rx * static_cast<mjtNum>(0.5));
  mjtNum sx = std::sin(rx * static_cast<mjtNum>(0.5));
  mjtNum cy = std::cos(ry * static_cast<mjtNum>(0.5));
  mjtNum sy = std::sin(ry * static_cast<mjtNum>(0.5));
  mjtNum cz = std::cos(rz * static_cast<mjtNum>(0.5));
  mjtNum sz = std::sin(rz * static_cast<mjtNum>(0.5));
  return {
      cx * cy * cz + sx * sy * sz,
      sx * cy * cz - cx * sy * sz,
      cx * sy * cz + sx * cy * sz,
      cx * cy * sz - sx * sy * cz,
  };
}

inline std::vector<mjtNum> RandomRgba(std::mt19937* gen) {
  std::uniform_real_distribution<double> dist(0.1, 0.9);
  return {
      static_cast<mjtNum>(dist(*gen)),
      static_cast<mjtNum>(dist(*gen)),
      static_cast<mjtNum>(dist(*gen)),
      static_cast<mjtNum>(1.0),
  };
}

inline void NormalizeRange(const std::vector<mjtNum>& input,
                           std::vector<mjtNum>* output) {
  mjtNum low = *std::min_element(input.begin(), input.end());
  mjtNum high = *std::max_element(input.begin(), input.end());
  output->resize(input.size());
  mjtNum denom = std::max(high - low, static_cast<mjtNum>(1e-12));
  for (std::size_t i = 0; i < input.size(); ++i) {
    (*output)[i] = (input[i] - low) / denom;
  }
}

inline std::vector<mjtNum> FlipGrid(const std::vector<mjtNum>& grid, int rows,
                                    int cols) {
  std::vector<mjtNum> flipped(grid.size());
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      int src = row * cols + col;
      int dst = (rows - 1 - row) * cols + (cols - 1 - col);
      flipped[dst] = grid[src];
    }
  }
  return flipped;
}

inline mjtNum ClipValue(mjtNum value, mjtNum lo, mjtNum hi) {
  return std::min(std::max(value, lo), hi);
}

}  // namespace detail

class MyoSuiteReorientEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(5),
        "frame_stack"_.Bind(1), "model_path"_.Bind(std::string()),
        "normalize_act"_.Bind(true), "muscle_condition"_.Bind(std::string()),
        "fatigue_reset_vec"_.Bind(std::vector<double>{}),
        "fatigue_reset_random"_.Bind(false), "obs_dim"_.Bind(0),
        "qpos_dim"_.Bind(0),
        "qvel_dim"_.Bind(0), "act_dim"_.Bind(0), "action_dim"_.Bind(0),
        "randomization_mode"_.Bind(std::string("100")),
        "reward_pos_align_w"_.Bind(1.0), "reward_rot_align_w"_.Bind(1.0),
        "reward_act_reg_w"_.Bind(5.0), "reward_drop_w"_.Bind(5.0),
        "reward_bonus_w"_.Bind(10.0),
        "test_reset_qpos"_.Bind(std::vector<double>{}),
        "test_reset_qvel"_.Bind(std::vector<double>{}),
        "test_reset_act"_.Bind(std::vector<double>{}),
        "test_reset_qacc_warmstart"_.Bind(std::vector<double>{}),
        "test_target_body_quat"_.Bind(std::vector<double>{}),
        "test_object_geom_size"_.Bind(std::vector<double>{}),
        "test_object_geom_rgba"_.Bind(std::vector<double>{}),
        "test_object_geom_top_pos"_.Bind(std::vector<double>{}),
        "test_object_geom_bottom_pos"_.Bind(std::vector<double>{}),
        "test_target_geom_size"_.Bind(std::vector<double>{}),
        "test_target_geom_rgba"_.Bind(std::vector<double>{}),
        "test_target_geom_top_pos"_.Bind(std::vector<double>{}),
        "test_target_geom_bottom_pos"_.Bind(std::vector<double>{}),
        "test_object_body_mass"_.Bind(std::vector<double>{}),
        "test_success_site_rgba"_.Bind(std::vector<double>{}),
        "test_object_geom_type"_.Bind(-1), "test_target_geom_type"_.Bind(-1),
        "test_object_geom_condim"_.Bind(-1));
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

using ReorientPixelFns = PixelObservationEnvFns<MyoSuiteReorientEnvFns>;
using MyoSuiteReorientEnvSpec = EnvSpec<MyoSuiteReorientEnvFns>;
using MyoSuiteReorientPixelEnvSpec = EnvSpec<ReorientPixelFns>;

class MyoSuiteWalkEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(10),
        "frame_stack"_.Bind(1), "model_path"_.Bind(std::string()),
        "normalize_act"_.Bind(true), "muscle_condition"_.Bind(std::string()),
        "fatigue_reset_vec"_.Bind(std::vector<double>{}),
        "fatigue_reset_random"_.Bind(false), "obs_dim"_.Bind(0),
        "qpos_dim"_.Bind(0),
        "qvel_dim"_.Bind(0), "act_dim"_.Bind(0), "action_dim"_.Bind(0),
        "min_height"_.Bind(0.8), "max_rot"_.Bind(0.8), "hip_period"_.Bind(100),
        "reset_type"_.Bind(std::string("init")), "target_x_vel"_.Bind(0.0),
        "target_y_vel"_.Bind(1.2), "target_rot"_.Bind(std::vector<double>{}),
        "terrain"_.Bind(std::string()), "terrain_variant"_.Bind(std::string()),
        "use_knee_condition"_.Bind(false), "reward_vel_w"_.Bind(5.0),
        "reward_done_w"_.Bind(-100.0), "reward_cyclic_hip_w"_.Bind(-10.0),
        "reward_ref_rot_w"_.Bind(10.0), "reward_joint_angle_w"_.Bind(5.0),
        "test_reset_qpos"_.Bind(std::vector<double>{}),
        "test_reset_qvel"_.Bind(std::vector<double>{}),
        "test_reset_act"_.Bind(std::vector<double>{}),
        "test_reset_qacc_warmstart"_.Bind(std::vector<double>{}),
        "test_hfield_data"_.Bind(std::vector<double>{}),
        "test_terrain_geom_rgba"_.Bind(std::vector<double>{}),
        "test_terrain_geom_pos"_.Bind(std::vector<double>{}),
        "test_terrain_geom_contype"_.Bind(-1),
        "test_terrain_geom_conaffinity"_.Bind(-1));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs"_.Bind(StackSpec(Spec<mjtNum>({conf["obs_dim"_]}, {-inf, inf}),
                              conf["frame_stack"_])),
        "info:vel_reward"_.Bind(Spec<mjtNum>({-1}, {-inf, inf})),
        "info:cyclic_hip"_.Bind(Spec<mjtNum>({-1}, {0.0, inf})),
        "info:ref_rot"_.Bind(Spec<mjtNum>({-1}, {0.0, inf})),
        "info:joint_angle_rew"_.Bind(Spec<mjtNum>({-1}, {0.0, inf})),
        "info:success"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({conf["qpos_dim"_]})),
        "info:qvel0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({conf["qvel_dim"_]})),
#endif
        "info:phase_var"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<float>({-1, conf["action_dim"_]}, {-1.0, 1.0})));
  }
};

using MyoSuiteWalkEnvSpec = EnvSpec<MyoSuiteWalkEnvFns>;
using MyoSuiteWalkPixelEnvFns = PixelObservationEnvFns<MyoSuiteWalkEnvFns>;
using MyoSuiteWalkPixelEnvSpec = EnvSpec<MyoSuiteWalkPixelEnvFns>;

class MyoSuiteTerrainEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MyoSuiteWalkEnvFns::DefaultConfig();
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MyoSuiteWalkEnvFns::StateSpec(conf);
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MyoSuiteWalkEnvFns::ActionSpec(conf);
  }
};

using TerrainPixelFns = PixelObservationEnvFns<MyoSuiteTerrainEnvFns>;
using MyoSuiteTerrainEnvSpec = EnvSpec<MyoSuiteTerrainEnvFns>;
using MyoSuiteTerrainPixelEnvSpec = EnvSpec<TerrainPixelFns>;

template <typename EnvSpecT, bool kFromPixels>
class MyoSuiteReorientEnvBase : public Env<EnvSpecT>,
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
  std::string randomization_mode_;
  mjtNum reward_pos_align_w_;
  mjtNum reward_rot_align_w_;
  mjtNum reward_act_reg_w_;
  mjtNum reward_drop_w_;
  mjtNum reward_bonus_w_;
  int target_body_id_{-1};
  int object_body_id_{-1};
  int eps_ball_sid_{-1};
  int success_sid_{-1};
  int obj_geom_id_{-1};
  int target_geom_id_{-1};
  int obj_top_geom_id_{-1};
  int obj_bottom_geom_id_{-1};
  int target_top_geom_id_{-1};
  int target_bottom_geom_id_{-1};
  mjtNum pen_length_{0.0};
  mjtNum target_length_{0.0};
  mjtNum initial_object_body_mass_{0.0};
  int initial_object_geom_type_{-1};
  int initial_object_geom_condim_{-1};
  int initial_target_geom_type_{-1};
  std::vector<mjtNum> initial_target_body_quat_;
  std::vector<mjtNum> initial_object_geom_size_;
  std::vector<mjtNum> initial_target_geom_size_;
  std::vector<mjtNum> initial_object_geom_rgba_;
  std::vector<mjtNum> initial_target_geom_rgba_;
  std::vector<mjtNum> initial_object_top_pos_;
  std::vector<mjtNum> initial_object_bottom_pos_;
  std::vector<mjtNum> initial_target_top_pos_;
  std::vector<mjtNum> initial_target_bottom_pos_;
  std::vector<mjtNum> initial_success_rgba_;
  std::vector<bool> muscle_actuator_;
  detail::MyoConditionState muscle_condition_state_;
  std::vector<mjtNum> test_reset_qpos_;
  std::vector<mjtNum> test_reset_qvel_;
  std::vector<mjtNum> test_reset_act_;
  std::vector<mjtNum> test_reset_qacc_warmstart_;
  std::vector<mjtNum> test_target_body_quat_;
  std::vector<mjtNum> test_object_geom_size_;
  std::vector<mjtNum> test_object_geom_rgba_;
  std::vector<mjtNum> test_object_geom_top_pos_;
  std::vector<mjtNum> test_object_geom_bottom_pos_;
  std::vector<mjtNum> test_target_geom_size_;
  std::vector<mjtNum> test_target_geom_rgba_;
  std::vector<mjtNum> test_target_geom_top_pos_;
  std::vector<mjtNum> test_target_geom_bottom_pos_;
  std::vector<mjtNum> test_object_body_mass_;
  std::vector<mjtNum> test_success_site_rgba_;
  int test_object_geom_type_;
  int test_target_geom_type_;
  int test_object_geom_condim_;
  std::uniform_real_distribution<double> unit_dist_{0.0, 1.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  MyoSuiteReorientEnvBase(const Spec& spec, int env_id)
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
        randomization_mode_(spec.config["randomization_mode"_]),
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
            detail::ToMjtVector(spec.config["test_target_body_quat"_])),
        test_object_geom_size_(
            detail::ToMjtVector(spec.config["test_object_geom_size"_])),
        test_object_geom_rgba_(
            detail::ToMjtVector(spec.config["test_object_geom_rgba"_])),
        test_object_geom_top_pos_(
            detail::ToMjtVector(spec.config["test_object_geom_top_pos"_])),
        test_object_geom_bottom_pos_(
            detail::ToMjtVector(spec.config["test_object_geom_bottom_pos"_])),
        test_target_geom_size_(
            detail::ToMjtVector(spec.config["test_target_geom_size"_])),
        test_target_geom_rgba_(
            detail::ToMjtVector(spec.config["test_target_geom_rgba"_])),
        test_target_geom_top_pos_(
            detail::ToMjtVector(spec.config["test_target_geom_top_pos"_])),
        test_target_geom_bottom_pos_(
            detail::ToMjtVector(spec.config["test_target_geom_bottom_pos"_])),
        test_object_body_mass_(
            detail::ToMjtVector(spec.config["test_object_body_mass"_])),
        test_success_site_rgba_(
            detail::ToMjtVector(spec.config["test_success_site_rgba"_])),
        test_object_geom_type_(spec.config["test_object_geom_type"_]),
        test_target_geom_type_(spec.config["test_target_geom_type"_]),
        test_object_geom_condim_(spec.config["test_object_geom_condim"_]) {
    ValidateConfig();
    CacheObjects();
    detail::BuildMuscleMask(model_, &muscle_actuator_);
    detail::InitializeMyoConditionState(
        model_, spec.config["muscle_condition"_],
        spec.config["fatigue_reset_vec"_],
        spec.config["fatigue_reset_random"_], spec.config["frame_skip"_],
        this->seed_, &muscle_condition_state_);
    detail::AdjustInitialQposForNormalizedActions(model_, data_,
                                                  normalize_act_);
    for (int i = 0; i < model_->nq - 6; ++i) {
      data_->qpos[i] = 0.0;
    }
    data_->qpos[0] = static_cast<mjtNum>(-1.5);
    mj_forward(model_, data_);
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
      throw std::runtime_error("Reorient config dims do not match model.");
    }
    int expected_obs = (model_->nq - 6) + 21 + 3 * model_->nu + model_->na;
    if (expected_obs != spec_.config["obs_dim"_]) {
      throw std::runtime_error("Reorient config obs_dim does not match model.");
    }
  }

  void CacheObjects() {
    target_body_id_ = mj_name2id(model_, mjOBJ_BODY, "target");
    object_body_id_ = mj_name2id(model_, mjOBJ_BODY, "Object");
    eps_ball_sid_ = mj_name2id(model_, mjOBJ_SITE, "eps_ball");
    success_sid_ = mj_name2id(model_, mjOBJ_SITE, "success");
    obj_geom_id_ = mj_name2id(model_, mjOBJ_GEOM, "obj");
    target_geom_id_ = mj_name2id(model_, mjOBJ_GEOM, "target");
    obj_top_geom_id_ = mj_name2id(model_, mjOBJ_GEOM, "top");
    obj_bottom_geom_id_ = mj_name2id(model_, mjOBJ_GEOM, "bot");
    target_top_geom_id_ = mj_name2id(model_, mjOBJ_GEOM, "t_top");
    target_bottom_geom_id_ = mj_name2id(model_, mjOBJ_GEOM, "t_bot");
    if (target_body_id_ == -1 || object_body_id_ == -1 || eps_ball_sid_ == -1 ||
        success_sid_ == -1 || obj_geom_id_ == -1 || target_geom_id_ == -1 ||
        obj_top_geom_id_ == -1 || obj_bottom_geom_id_ == -1 ||
        target_top_geom_id_ == -1 || target_bottom_geom_id_ == -1) {
      throw std::runtime_error("Reorient ids missing.");
    }
    detail::CopyModelBodyQuat(model_, target_body_id_,
                              &initial_target_body_quat_);
    detail::CopyModelGeomSize(model_, obj_geom_id_, &initial_object_geom_size_);
    detail::CopyModelGeomSize(model_, target_geom_id_,
                              &initial_target_geom_size_);
    detail::CopyModelGeomRgba(model_, obj_geom_id_, &initial_object_geom_rgba_);
    detail::CopyModelGeomRgba(model_, target_geom_id_,
                              &initial_target_geom_rgba_);
    detail::CopyModelGeomPos(model_, obj_top_geom_id_,
                             &initial_object_top_pos_);
    detail::CopyModelGeomPos(model_, obj_bottom_geom_id_,
                             &initial_object_bottom_pos_);
    detail::CopyModelGeomPos(model_, target_top_geom_id_,
                             &initial_target_top_pos_);
    detail::CopyModelGeomPos(model_, target_bottom_geom_id_,
                             &initial_target_bottom_pos_);
    detail::CopyModelSiteRgba(model_, success_sid_, &initial_success_rgba_);
    detail::CopyModelGeomType(model_, obj_geom_id_, &initial_object_geom_type_);
    detail::CopyModelGeomType(model_, target_geom_id_,
                              &initial_target_geom_type_);
    detail::CopyModelGeomCondim(model_, obj_geom_id_,
                                &initial_object_geom_condim_);
    detail::CopyModelBodyMass(model_, object_body_id_,
                              &initial_object_body_mass_);
    UpdateLengths();
  }

  void RestoreModelState() {
    detail::RestoreModelBodyQuat(model_, target_body_id_,
                                 initial_target_body_quat_);
    detail::RestoreModelGeomSize(model_, obj_geom_id_,
                                 initial_object_geom_size_);
    detail::RestoreModelGeomSize(model_, target_geom_id_,
                                 initial_target_geom_size_);
    detail::RestoreModelGeomRgba(model_, obj_geom_id_,
                                 initial_object_geom_rgba_);
    detail::RestoreModelGeomRgba(model_, target_geom_id_,
                                 initial_target_geom_rgba_);
    detail::RestoreModelGeomPos(model_, obj_top_geom_id_,
                                initial_object_top_pos_);
    detail::RestoreModelGeomPos(model_, obj_bottom_geom_id_,
                                initial_object_bottom_pos_);
    detail::RestoreModelGeomPos(model_, target_top_geom_id_,
                                initial_target_top_pos_);
    detail::RestoreModelGeomPos(model_, target_bottom_geom_id_,
                                initial_target_bottom_pos_);
    detail::RestoreModelSiteRgba(model_, success_sid_, initial_success_rgba_);
    detail::RestoreModelGeomType(model_, obj_geom_id_,
                                 initial_object_geom_type_);
    detail::RestoreModelGeomType(model_, target_geom_id_,
                                 initial_target_geom_type_);
    detail::RestoreModelGeomCondim(model_, obj_geom_id_,
                                   initial_object_geom_condim_);
    detail::RestoreModelBodyMass(model_, object_body_id_,
                                 initial_object_body_mass_);
  }

  void UpdateLengths() {
    std::vector<mjtNum> top(3);
    std::vector<mjtNum> bottom(3);
    for (int axis = 0; axis < 3; ++axis) {
      top[axis] = model_->geom_pos[obj_top_geom_id_ * 3 + axis];
      bottom[axis] = model_->geom_pos[obj_bottom_geom_id_ * 3 + axis];
    }
    for (int axis = 0; axis < 3; ++axis) {
      top[axis] -= bottom[axis];
    }
    pen_length_ = detail::VectorNorm(top);
    for (int axis = 0; axis < 3; ++axis) {
      top[axis] = model_->geom_pos[target_top_geom_id_ * 3 + axis];
      bottom[axis] = model_->geom_pos[target_bottom_geom_id_ * 3 + axis];
    }
    for (int axis = 0; axis < 3; ++axis) {
      top[axis] -= bottom[axis];
    }
    target_length_ = detail::VectorNorm(top);
  }

  std::vector<mjtNum> RandomSizeForType(int geom_type) {
    if (randomization_mode_ == "8") {
      static const std::array<std::array<mjtNum, 3>, 8> kSizes = {{
          {0.013, 0.025, 0.025},
          {0.019, 0.040, 0.040},
          {0.017, 0.017, 0.017},
          {0.023, 0.023, 0.023},
          {0.013, 0.025, 0.025},
          {0.019, 0.040, 0.040},
          {0.013, 0.025, 0.025},
          {0.019, 0.040, 0.040},
      }};
      std::array<int, 2> ids = geom_type == 3   ? std::array<int, 2>{4, 5}
                               : geom_type == 4 ? std::array<int, 2>{0, 1}
                               : geom_type == 5 ? std::array<int, 2>{6, 7}
                                                : std::array<int, 2>{2, 3};
      int pick =
          ids[static_cast<int>(unit_dist_(gen_) * ids.size()) % ids.size()];
      return {kSizes[pick][0], kSizes[pick][1], kSizes[pick][2]};
    }
    std::uniform_real_distribution<double> xdist(0.008, 0.028);
    std::uniform_real_distribution<double> ydist(0.020, 0.050);
    std::uniform_real_distribution<double> zdist(0.020, 0.050);
    if (randomization_mode_ == "ood") {
      xdist = std::uniform_real_distribution<double>(0.015, 0.035);
      ydist = std::uniform_real_distribution<double>(0.015, 0.055);
      zdist = std::uniform_real_distribution<double>(0.015, 0.055);
    }
    return {
        static_cast<mjtNum>(xdist(gen_)),
        static_cast<mjtNum>(ydist(gen_)),
        static_cast<mjtNum>(zdist(gen_)),
    };
  }

  void RandomizeGeometry() {
    static const std::array<int, 4> kGeomTypes = {3, 4, 5, 6};
    int geom_type =
        kGeomTypes[static_cast<int>(unit_dist_(gen_) * kGeomTypes.size()) %
                   kGeomTypes.size()];
    std::vector<mjtNum> size = RandomSizeForType(geom_type);
    std::vector<mjtNum> color = detail::RandomRgba(&gen_);
    std::vector<mjtNum> top_pos(3, 0.0);
    std::vector<mjtNum> bottom_pos(3, 0.0);
    if (geom_type == 3) {
      top_pos[2] = static_cast<mjtNum>(1.3) * size[1];
      bottom_pos[2] = -static_cast<mjtNum>(1.3) * size[1];
    } else if (geom_type == 4 || geom_type == 6) {
      top_pos[2] = size[2];
      bottom_pos[2] = -size[2];
    } else {
      top_pos[2] = size[1];
      bottom_pos[2] = -size[1];
    }
    detail::RestoreModelGeomSize(model_, obj_geom_id_, size);
    detail::RestoreModelGeomType(model_, obj_geom_id_, geom_type);
    detail::RestoreModelGeomRgba(model_, obj_geom_id_, color);
    detail::RestoreModelGeomPos(model_, obj_top_geom_id_, top_pos);
    detail::RestoreModelGeomPos(model_, obj_bottom_geom_id_, bottom_pos);
    detail::RestoreModelBodyMass(model_, object_body_id_,
                                 static_cast<mjtNum>(1.2));
    detail::RestoreModelGeomSize(model_, target_geom_id_, size);
    detail::RestoreModelGeomType(model_, target_geom_id_, geom_type);
    detail::RestoreModelGeomRgba(model_, target_geom_id_, color);
    detail::RestoreModelGeomPos(model_, target_top_geom_id_, top_pos);
    detail::RestoreModelGeomPos(model_, target_bottom_geom_id_, bottom_pos);
    detail::RestoreModelGeomCondim(model_, obj_geom_id_, 3);
    auto quat = detail::EulerXYZToQuat(
        static_cast<mjtNum>(unit_dist_(gen_) * 2.0 - 1.0),
        static_cast<mjtNum>(unit_dist_(gen_) * 2.0 - 0.8), 0.0);
    std::vector<mjtNum> quat_vec = {quat[0], quat[1], quat[2], quat[3]};
    detail::RestoreModelBodyQuat(model_, target_body_id_, quat_vec);
    std::vector<mjtNum> success_rgba = initial_success_rgba_;
    success_rgba[0] = 2.0;
    success_rgba[1] = 0.0;
    detail::RestoreModelSiteRgba(model_, success_sid_, success_rgba);
  }

  void ApplyResetState() {
    if (test_object_geom_type_ >= 0) {
      detail::RestoreModelGeomType(model_, obj_geom_id_,
                                   test_object_geom_type_);
      detail::RestoreModelGeomType(model_, target_geom_id_,
                                   test_target_geom_type_);
      detail::RestoreModelGeomCondim(model_, obj_geom_id_,
                                     test_object_geom_condim_);
      detail::RestoreModelGeomSize(model_, obj_geom_id_,
                                   test_object_geom_size_);
      detail::RestoreModelGeomSize(model_, target_geom_id_,
                                   test_target_geom_size_);
      detail::RestoreModelGeomRgba(model_, obj_geom_id_,
                                   test_object_geom_rgba_);
      detail::RestoreModelGeomRgba(model_, target_geom_id_,
                                   test_target_geom_rgba_);
      detail::RestoreModelGeomPos(model_, obj_top_geom_id_,
                                  test_object_geom_top_pos_);
      detail::RestoreModelGeomPos(model_, obj_bottom_geom_id_,
                                  test_object_geom_bottom_pos_);
      detail::RestoreModelGeomPos(model_, target_top_geom_id_,
                                  test_target_geom_top_pos_);
      detail::RestoreModelGeomPos(model_, target_bottom_geom_id_,
                                  test_target_geom_bottom_pos_);
      if (!test_object_body_mass_.empty()) {
        detail::RestoreModelBodyMass(model_, object_body_id_,
                                     test_object_body_mass_[0]);
      }
      detail::RestoreModelBodyQuat(model_, target_body_id_,
                                   test_target_body_quat_);
      if (!test_success_site_rgba_.empty()) {
        detail::RestoreModelSiteRgba(model_, success_sid_,
                                     test_success_site_rgba_);
      } else {
        std::vector<mjtNum> success_rgba = initial_success_rgba_;
        success_rgba[0] = 2.0;
        success_rgba[1] = 0.0;
        detail::RestoreModelSiteRgba(model_, success_sid_, success_rgba);
      }
    } else {
      RandomizeGeometry();
    }
    if (!test_reset_qpos_.empty()) {
      detail::RestoreVector(test_reset_qpos_, data_->qpos);
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

  RewardInfo ComputeRewardInfo() const {
    RewardInfo reward;
    std::vector<mjtNum> obj_rot(3);
    std::vector<mjtNum> target_rot(3);
    std::vector<mjtNum> obj_err_pos(3);
    for (int axis = 0; axis < 3; ++axis) {
      obj_err_pos[axis] = data_->xpos[object_body_id_ * 3 + axis] -
                          data_->site_xpos[eps_ball_sid_ * 3 + axis];
      obj_rot[axis] = (data_->geom_xpos[obj_top_geom_id_ * 3 + axis] -
                       data_->geom_xpos[obj_bottom_geom_id_ * 3 + axis]) /
                      pen_length_;
      target_rot[axis] = (data_->geom_xpos[target_top_geom_id_ * 3 + axis] -
                          data_->geom_xpos[target_bottom_geom_id_ * 3 + axis]) /
                         target_length_;
    }
    reward.pos_align = detail::VectorNorm(obj_err_pos);
    reward.rot_align = detail::CosineSimilarity(obj_rot, target_rot);
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
        *(buffer++) = data_->xpos[object_body_id_ * 3 + axis];
      }
      for (int i = model_->nv - 6; i < model_->nv; ++i) {
        *(buffer++) = data_->qvel[i] * Dt();
      }
      for (int axis = 0; axis < 3; ++axis) {
        *(buffer++) = (data_->geom_xpos[obj_top_geom_id_ * 3 + axis] -
                       data_->geom_xpos[obj_bottom_geom_id_ * 3 + axis]) /
                      pen_length_;
      }
      for (int axis = 0; axis < 3; ++axis) {
        *(buffer++) = (data_->geom_xpos[target_top_geom_id_ * 3 + axis] -
                       data_->geom_xpos[target_bottom_geom_id_ * 3 + axis]) /
                      target_length_;
      }
      for (int axis = 0; axis < 3; ++axis) {
        *(buffer++) = data_->xpos[object_body_id_ * 3 + axis] -
                      data_->site_xpos[eps_ball_sid_ * 3 + axis];
      }
      for (int axis = 0; axis < 3; ++axis) {
        *(buffer++) = (data_->geom_xpos[obj_top_geom_id_ * 3 + axis] -
                       data_->geom_xpos[obj_bottom_geom_id_ * 3 + axis]) /
                          pen_length_ -
                      (data_->geom_xpos[target_top_geom_id_ * 3 + axis] -
                       data_->geom_xpos[target_bottom_geom_id_ * 3 + axis]) /
                          target_length_;
      }
      for (int i = 0; i < model_->nu; ++i) {
        *(buffer++) = data_->actuator_length[i];
      }
      for (int i = 0; i < model_->nu; ++i) {
        *(buffer++) = data_->actuator_velocity[i];
      }
      for (int i = 0; i < model_->nu; ++i) {
        *(buffer++) = data_->actuator_force[i];
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

template <typename EnvSpecT, bool kFromPixels>
class MyoSuiteWalkLikeEnvBase : public Env<EnvSpecT>,
                                public gymnasium_robotics::MujocoRobotEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  struct RewardInfo {
    mjtNum dense_reward{0.0};
    mjtNum vel_reward{0.0};
    mjtNum cyclic_hip{0.0};
    mjtNum ref_rot{0.0};
    mjtNum joint_angle_rew{0.0};
    bool success{false};
    bool done{false};
  };

  bool normalize_act_;
  mjtNum min_height_;
  mjtNum max_rot_;
  int hip_period_;
  std::string reset_type_;
  mjtNum target_x_vel_;
  mjtNum target_y_vel_;
  std::vector<mjtNum> target_rot_;
  std::string terrain_;
  std::string terrain_variant_;
  bool use_knee_condition_;
  mjtNum reward_vel_w_;
  mjtNum reward_done_w_;
  mjtNum reward_cyclic_hip_w_;
  mjtNum reward_ref_rot_w_;
  mjtNum reward_joint_angle_w_;
  int terrain_geom_id_{-1};
  int terrain_hfield_id_{-1};
  int torso_body_id_{-1};
  int pelvis_body_id_{-1};
  int foot_l_body_id_{-1};
  int foot_r_body_id_{-1};
  std::array<int, 2> hip_flex_qposadr_{-1, -1};
  std::array<int, 4> reward_joint_qposadr_{-1, -1, -1, -1};
  std::vector<mjtNum> initial_hfield_data_;
  std::vector<mjtNum> initial_terrain_rgba_;
  std::vector<mjtNum> initial_terrain_pos_;
  int initial_terrain_contype_{-1};
  int initial_terrain_conaffinity_{-1};
  std::vector<bool> muscle_actuator_;
  detail::MyoConditionState muscle_condition_state_;
  int gait_steps_{0};
  std::normal_distribution<double> init_noise_dist_{0.0, 0.02};
  std::vector<mjtNum> test_reset_qpos_;
  std::vector<mjtNum> test_reset_qvel_;
  std::vector<mjtNum> test_reset_act_;
  std::vector<mjtNum> test_reset_qacc_warmstart_;
  std::vector<mjtNum> test_hfield_data_;
  std::vector<mjtNum> test_terrain_geom_rgba_;
  std::vector<mjtNum> test_terrain_geom_pos_;
  int test_terrain_geom_contype_{-1};
  int test_terrain_geom_conaffinity_{-1};
  detail::NumpyPcg64 terrain_rng_;
  std::uniform_real_distribution<double> unit_dist_{0.0, 1.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  MyoSuiteWalkLikeEnvBase(const Spec& spec, int env_id)
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
        min_height_(spec.config["min_height"_]),
        max_rot_(spec.config["max_rot"_]),
        hip_period_(spec.config["hip_period"_]),
        reset_type_(spec.config["reset_type"_]),
        target_x_vel_(spec.config["target_x_vel"_]),
        target_y_vel_(spec.config["target_y_vel"_]),
        target_rot_(detail::ToMjtVector(spec.config["target_rot"_])),
        terrain_(spec.config["terrain"_]),
        terrain_variant_(spec.config["terrain_variant"_]),
        use_knee_condition_(spec.config["use_knee_condition"_]),
        reward_vel_w_(spec.config["reward_vel_w"_]),
        reward_done_w_(spec.config["reward_done_w"_]),
        reward_cyclic_hip_w_(spec.config["reward_cyclic_hip_w"_]),
        reward_ref_rot_w_(spec.config["reward_ref_rot_w"_]),
        reward_joint_angle_w_(spec.config["reward_joint_angle_w"_]),
        test_reset_qpos_(detail::ToMjtVector(spec.config["test_reset_qpos"_])),
        test_reset_qvel_(detail::ToMjtVector(spec.config["test_reset_qvel"_])),
        test_reset_act_(detail::ToMjtVector(spec.config["test_reset_act"_])),
        test_reset_qacc_warmstart_(
            detail::ToMjtVector(spec.config["test_reset_qacc_warmstart"_])),
        test_hfield_data_(
            detail::ToMjtVector(spec.config["test_hfield_data"_])),
        test_terrain_geom_rgba_(
            detail::ToMjtVector(spec.config["test_terrain_geom_rgba"_])),
        test_terrain_geom_pos_(
            detail::ToMjtVector(spec.config["test_terrain_geom_pos"_])),
        test_terrain_geom_contype_(spec.config["test_terrain_geom_contype"_]),
        test_terrain_geom_conaffinity_(
            spec.config["test_terrain_geom_conaffinity"_]),
        terrain_rng_(static_cast<std::uint64_t>(this->seed_)) {
    ValidateConfig();
    CacheIds();
    detail::BuildMuscleMask(model_, &muscle_actuator_);
    detail::InitializeMyoConditionState(
        model_, spec.config["muscle_condition"_],
        spec.config["fatigue_reset_vec"_],
        spec.config["fatigue_reset_random"_], spec.config["frame_skip"_],
        this->seed_, &muscle_condition_state_);
    detail::AdjustInitialQposForNormalizedActions(model_, data_,
                                                  normalize_act_);
    if (target_rot_.empty()) {
      target_rot_.assign(model_->key_qpos + 3, model_->key_qpos + 7);
    }
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
    gait_steps_ = 0;
    detail::ResetMyoConditionState(&muscle_condition_state_);
    ResetToInitialState();
    RestoreTerrainState();
    ApplyTerrainReset();
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
    ++gait_steps_;
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
      throw std::runtime_error("Walk config dims do not match model.");
    }
    int expected_obs = (model_->nq - 2) + model_->nv + 2 + 4 + 2 + 1 + 6 + 1 +
                       3 * model_->nu + model_->na;
    if (expected_obs != spec_.config["obs_dim"_]) {
      throw std::runtime_error("Walk config obs_dim does not match model.");
    }
  }

  void CacheIds() {
    terrain_geom_id_ = mj_name2id(model_, mjOBJ_GEOM, "terrain");
    torso_body_id_ = mj_name2id(model_, mjOBJ_BODY, "torso");
    pelvis_body_id_ = mj_name2id(model_, mjOBJ_BODY, "pelvis");
    foot_l_body_id_ = mj_name2id(model_, mjOBJ_BODY, "talus_l");
    foot_r_body_id_ = mj_name2id(model_, mjOBJ_BODY, "talus_r");
    hip_flex_qposadr_[0] =
        model_->jnt_qposadr[mj_name2id(model_, mjOBJ_JOINT, "hip_flexion_l")];
    hip_flex_qposadr_[1] =
        model_->jnt_qposadr[mj_name2id(model_, mjOBJ_JOINT, "hip_flexion_r")];
    reward_joint_qposadr_[0] =
        model_->jnt_qposadr[mj_name2id(model_, mjOBJ_JOINT, "hip_adduction_l")];
    reward_joint_qposadr_[1] =
        model_->jnt_qposadr[mj_name2id(model_, mjOBJ_JOINT, "hip_adduction_r")];
    reward_joint_qposadr_[2] =
        model_->jnt_qposadr[mj_name2id(model_, mjOBJ_JOINT, "hip_rotation_l")];
    reward_joint_qposadr_[3] =
        model_->jnt_qposadr[mj_name2id(model_, mjOBJ_JOINT, "hip_rotation_r")];
    if (terrain_geom_id_ == -1 || torso_body_id_ == -1 ||
        pelvis_body_id_ == -1 || foot_l_body_id_ == -1 ||
        foot_r_body_id_ == -1) {
      throw std::runtime_error("Walk ids missing.");
    }
    terrain_hfield_id_ = model_->geom_dataid[terrain_geom_id_];
    if (terrain_hfield_id_ >= 0) {
      detail::CopyModelHfieldData(model_, terrain_hfield_id_,
                                  &initial_hfield_data_);
    }
    detail::CopyModelGeomRgba(model_, terrain_geom_id_, &initial_terrain_rgba_);
    detail::CopyModelGeomPos(model_, terrain_geom_id_, &initial_terrain_pos_);
    detail::CopyModelGeomContype(model_, terrain_geom_id_,
                                 &initial_terrain_contype_);
    detail::CopyModelGeomConaffinity(model_, terrain_geom_id_,
                                     &initial_terrain_conaffinity_);
  }

  void RestoreTerrainState() {
    if (terrain_hfield_id_ >= 0) {
      detail::RestoreModelHfieldData(model_, terrain_hfield_id_,
                                     initial_hfield_data_);
    }
    detail::RestoreModelGeomRgba(model_, terrain_geom_id_,
                                 initial_terrain_rgba_);
    detail::RestoreModelGeomPos(model_, terrain_geom_id_, initial_terrain_pos_);
    detail::RestoreModelGeomContype(model_, terrain_geom_id_,
                                    initial_terrain_contype_);
    detail::RestoreModelGeomConaffinity(model_, terrain_geom_id_,
                                        initial_terrain_conaffinity_);
  }

  void ApplyTerrainVisibility(bool visible) {
    std::vector<mjtNum> rgba = initial_terrain_rgba_;
    std::vector<mjtNum> pos = initial_terrain_pos_;
    rgba[3] = visible ? static_cast<mjtNum>(1.0) : static_cast<mjtNum>(0.0);
    pos[0] = 0.0;
    pos[1] = 0.0;
    pos[2] = visible ? static_cast<mjtNum>(0.0) : static_cast<mjtNum>(-10.0);
    detail::RestoreModelGeomRgba(model_, terrain_geom_id_, rgba);
    detail::RestoreModelGeomPos(model_, terrain_geom_id_, pos);
    if (visible) {
      detail::RestoreModelGeomContype(model_, terrain_geom_id_, 1);
      detail::RestoreModelGeomConaffinity(model_, terrain_geom_id_, 1);
    } else {
      detail::RestoreModelGeomContype(model_, terrain_geom_id_,
                                      initial_terrain_contype_);
      detail::RestoreModelGeomConaffinity(model_, terrain_geom_id_,
                                          initial_terrain_conaffinity_);
    }
  }

  std::vector<mjtNum> RandomizeRoughTerrain() {
    int rows = model_->hfield_nrow[terrain_hfield_id_];
    int cols = model_->hfield_ncol[terrain_hfield_id_];
    std::vector<mjtNum> rough(rows * cols);
    for (mjtNum& value : rough) {
      value = terrain_rng_.UniformMjt(static_cast<mjtNum>(-0.5),
                                      static_cast<mjtNum>(0.5));
    }
    std::vector<mjtNum> normalized;
    detail::NormalizeRange(rough, &normalized);
    for (mjtNum& value : normalized) {
      value = value * static_cast<mjtNum>(0.08) - static_cast<mjtNum>(0.02);
    }
    return normalized;
  }

  std::vector<mjtNum> RandomizeHillyTerrain() {
    int rows = model_->hfield_nrow[terrain_hfield_id_];
    int cols = model_->hfield_ncol[terrain_hfield_id_];
    int total = rows * cols;
    int flat_length = 3000;
    int frequency = 3;
    mjtNum scalar = terrain_variant_ == "fixed"
                        ? static_cast<mjtNum>(0.63)
                        : terrain_rng_.UniformMjt(static_cast<mjtNum>(0.53),
                                                  static_cast<mjtNum>(0.73));
    std::vector<mjtNum> combined(total, static_cast<mjtNum>(-2.0));
    for (int i = flat_length; i < total; ++i) {
      double phase = static_cast<double>(i - flat_length) /
                         std::max(total - flat_length - 1, 1) * frequency *
                         detail::kPi +
                     detail::kPi / 2.0;
      combined[i] = static_cast<mjtNum>(-2.0 + 0.5 * (std::sin(phase) - 1.0));
    }
    std::vector<mjtNum> normalized;
    detail::NormalizeRange(combined, &normalized);
    for (mjtNum& value : normalized) {
      value *= scalar;
    }
    return detail::FlipGrid(normalized, rows, cols);
  }

  std::vector<mjtNum> RandomizeStairsTerrain() {
    int rows = model_->hfield_nrow[terrain_hfield_id_];
    int cols = model_->hfield_ncol[terrain_hfield_id_];
    int total = rows * cols;
    int num_stairs = 12;
    mjtNum stair_height = static_cast<mjtNum>(0.1);
    int flat = 5200 - (total - 5200) % num_stairs;
    int stairs_width = (total - flat) / num_stairs;
    mjtNum scalar = terrain_variant_ == "fixed"
                        ? static_cast<mjtNum>(2.5)
                        : terrain_rng_.UniformMjt(static_cast<mjtNum>(1.5),
                                                  static_cast<mjtNum>(3.5));
    std::vector<mjtNum> data(total, static_cast<mjtNum>(-2.0));
    for (int stair = 0; stair < num_stairs; ++stair) {
      int start = flat + stair * stairs_width;
      int end = std::min(start + stairs_width, total);
      for (int idx = start; idx < end; ++idx) {
        data[idx] = static_cast<mjtNum>(-2.0) + stair_height * stair;
      }
    }
    for (mjtNum& value : data) {
      value = (value + static_cast<mjtNum>(2.0)) /
              (static_cast<mjtNum>(2.0) + stair_height * num_stairs) * scalar;
    }
    return detail::FlipGrid(data, rows, cols);
  }

  void ApplyTerrainReset() {
    if (terrain_.empty()) {
      if constexpr (kFromPixels) {
        std::vector<mjtNum> rgba = initial_terrain_rgba_;
        std::vector<mjtNum> pos = initial_terrain_pos_;
        rgba[3] = static_cast<mjtNum>(0.0);
        pos[0] = 0.0;
        pos[1] = 0.0;
        pos[2] = 0.0;
        detail::RestoreModelGeomRgba(model_, terrain_geom_id_, rgba);
        detail::RestoreModelGeomPos(model_, terrain_geom_id_, pos);
        detail::RestoreModelGeomContype(model_, terrain_geom_id_, 0);
        detail::RestoreModelGeomConaffinity(model_, terrain_geom_id_, 0);
      } else {
        ApplyTerrainVisibility(false);
      }
      if (!test_terrain_geom_rgba_.empty()) {
        detail::RestoreModelGeomRgba(model_, terrain_geom_id_,
                                     test_terrain_geom_rgba_);
      }
      if (!test_terrain_geom_pos_.empty()) {
        detail::RestoreModelGeomPos(model_, terrain_geom_id_,
                                    test_terrain_geom_pos_);
      }
      if (test_terrain_geom_contype_ >= 0) {
        detail::RestoreModelGeomContype(model_, terrain_geom_id_,
                                        test_terrain_geom_contype_);
      }
      if (test_terrain_geom_conaffinity_ >= 0) {
        detail::RestoreModelGeomConaffinity(model_, terrain_geom_id_,
                                            test_terrain_geom_conaffinity_);
      }
      return;
    }
    ApplyTerrainVisibility(true);
    if (!test_terrain_geom_rgba_.empty()) {
      detail::RestoreModelGeomRgba(model_, terrain_geom_id_,
                                   test_terrain_geom_rgba_);
    }
    if (!test_terrain_geom_pos_.empty()) {
      detail::RestoreModelGeomPos(model_, terrain_geom_id_,
                                  test_terrain_geom_pos_);
    }
    if (test_terrain_geom_contype_ >= 0) {
      detail::RestoreModelGeomContype(model_, terrain_geom_id_,
                                      test_terrain_geom_contype_);
    }
    if (test_terrain_geom_conaffinity_ >= 0) {
      detail::RestoreModelGeomConaffinity(model_, terrain_geom_id_,
                                          test_terrain_geom_conaffinity_);
    }
    if (!test_hfield_data_.empty()) {
      detail::RestoreModelHfieldData(model_, terrain_hfield_id_,
                                     test_hfield_data_);
      return;
    }
    std::vector<mjtNum> hfield;
    if (terrain_ == "rough") {
      hfield = RandomizeRoughTerrain();
    } else if (terrain_ == "hilly") {
      hfield = RandomizeHillyTerrain();
    } else if (terrain_ == "stairs") {
      hfield = RandomizeStairsTerrain();
    }
    detail::RestoreModelHfieldData(model_, terrain_hfield_id_, hfield);
  }

  std::pair<std::vector<mjtNum>, std::vector<mjtNum>> GetResetState() {
    int key_id = reset_type_ == "random" ? (unit_dist_(gen_) < 0.5 ? 2 : 3)
                                         : (reset_type_ == "init" ? 2 : 0);
    std::vector<mjtNum> qpos(model_->key_qpos + key_id * model_->nq,
                             model_->key_qpos + (key_id + 1) * model_->nq);
    std::vector<mjtNum> qvel(model_->key_qvel + key_id * model_->nv,
                             model_->key_qvel + (key_id + 1) * model_->nv);
    if (reset_type_ == "random") {
      std::vector<mjtNum> root_quat(qpos.begin() + 3, qpos.begin() + 7);
      mjtNum height = qpos[2];
      for (mjtNum& value : qpos) {
        value += static_cast<mjtNum>(init_noise_dist_(gen_));
      }
      std::copy(root_quat.begin(), root_quat.end(), qpos.begin() + 3);
      qpos[2] = height;
    }
    return {qpos, qvel};
  }

  void ApplyResetState() {
    if (!test_reset_qpos_.empty()) {
      detail::RestoreVector(test_reset_qpos_, data_->qpos);
      detail::RestoreVector(test_reset_qvel_, data_->qvel);
    } else {
      auto [qpos, qvel] = GetResetState();
      detail::RestoreVector(qpos, data_->qpos);
      detail::RestoreVector(qvel, data_->qvel);
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

  std::array<mjtNum, 3> GetCom() const {
    mjtNum total_mass = 0.0;
    std::array<mjtNum, 3> com = {0.0, 0.0, 0.0};
    for (int body = 0; body < model_->nbody; ++body) {
      mjtNum mass = model_->body_mass[body];
      total_mass += mass;
      for (int axis = 0; axis < 3; ++axis) {
        com[axis] += mass * data_->xipos[body * 3 + axis];
      }
    }
    for (mjtNum& value : com) {
      value /= total_mass;
    }
    return com;
  }

  std::array<mjtNum, 2> GetComVelocity() const {
    mjtNum total_mass = 0.0;
    std::array<mjtNum, 2> velocity = {0.0, 0.0};
    for (int body = 0; body < model_->nbody; ++body) {
      mjtNum mass = model_->body_mass[body];
      total_mass += mass;
      velocity[0] += mass * (-data_->cvel[body * 6 + 3]);
      velocity[1] += mass * (-data_->cvel[body * 6 + 4]);
    }
    velocity[0] /= total_mass;
    velocity[1] /= total_mass;
    return velocity;
  }

  std::array<mjtNum, 2> GetFeetHeights() const {
    return {data_->xpos[foot_l_body_id_ * 3 + 2],
            data_->xpos[foot_r_body_id_ * 3 + 2]};
  }

  std::array<mjtNum, 6> GetFeetRelativePositions() const {
    std::array<mjtNum, 6> out{};
    for (int axis = 0; axis < 3; ++axis) {
      out[axis] = data_->xpos[foot_l_body_id_ * 3 + axis] -
                  data_->xpos[pelvis_body_id_ * 3 + axis];
      out[3 + axis] = data_->xpos[foot_r_body_id_ * 3 + axis] -
                      data_->xpos[pelvis_body_id_ * 3 + axis];
    }
    return out;
  }

  mjtNum GetJointAngleReward() const {
    mjtNum mean_abs = 0.0;
    for (int adr : reward_joint_qposadr_) {
      mean_abs += std::abs(data_->qpos[adr]);
    }
    mean_abs /= reward_joint_qposadr_.size();
    return std::exp(static_cast<mjtNum>(-5.0) * mean_abs);
  }

  mjtNum GetCyclicReward() const {
    mjtNum phase = static_cast<mjtNum>(
        std::fmod(static_cast<double>(gait_steps_) / hip_period_, 1.0));
    mjtNum desired_l = static_cast<mjtNum>(
        0.8 *
        std::cos(phase * static_cast<mjtNum>(2.0) * detail::kPi + detail::kPi));
    mjtNum desired_r = static_cast<mjtNum>(
        0.8 * std::cos(phase * static_cast<mjtNum>(2.0) * detail::kPi));
    mjtNum diff_l = desired_l - data_->qpos[hip_flex_qposadr_[0]];
    mjtNum diff_r = desired_r - data_->qpos[hip_flex_qposadr_[1]];
    return std::sqrt(diff_l * diff_l + diff_r * diff_r);
  }

  mjtNum GetRefRotReward() const {
    mjtNum diff_norm = 0.0;
    for (int i = 0; i < 4; ++i) {
      mjtNum diff =
          static_cast<mjtNum>(5.0) * (data_->qpos[3 + i] - target_rot_[i]);
      diff_norm += diff * diff;
    }
    return std::exp(-std::sqrt(diff_norm));
  }

  bool GetRotCondition() const {
    mjtNum mat[9];
    mju_quat2Mat(mat, data_->qpos + 3);
    return std::abs(mat[0]) > max_rot_;
  }

  bool GetKneeCondition() const {
    auto feet = GetFeetHeights();
    mjtNum mean_feet = (feet[0] + feet[1]) * static_cast<mjtNum>(0.5);
    return (GetCom()[2] - mean_feet) < static_cast<mjtNum>(0.61);
  }

  bool GetDoneCondition() const {
    if (GetCom()[2] < min_height_) {
      return true;
    }
    if (GetRotCondition()) {
      return true;
    }
    return use_knee_condition_ && GetKneeCondition();
  }

  RewardInfo ComputeRewardInfo() const {
    RewardInfo reward;
    auto com_velocity = GetComVelocity();
    reward.vel_reward =
        std::exp(-std::pow(target_y_vel_ - com_velocity[1], 2.0)) +
        std::exp(-std::pow(target_x_vel_ - com_velocity[0], 2.0));
    reward.cyclic_hip = GetCyclicReward();
    reward.ref_rot = GetRefRotReward();
    reward.joint_angle_rew = GetJointAngleReward();
    reward.success = reward.vel_reward >= static_cast<mjtNum>(1.0);
    reward.done = GetDoneCondition();
    reward.dense_reward = reward_vel_w_ * reward.vel_reward +
                          reward_done_w_ * static_cast<mjtNum>(reward.done) +
                          reward_cyclic_hip_w_ * reward.cyclic_hip +
                          reward_ref_rot_w_ * reward.ref_rot +
                          reward_joint_angle_w_ * reward.joint_angle_rew;
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
      for (int i = 2; i < model_->nq; ++i) {
        *(buffer++) = data_->qpos[i];
      }
      for (int i = 0; i < model_->nv; ++i) {
        *(buffer++) = data_->qvel[i] * Dt();
      }
      auto com_velocity = GetComVelocity();
      *(buffer++) = com_velocity[0];
      *(buffer++) = com_velocity[1];
      for (int i = 0; i < 4; ++i) {
        *(buffer++) = data_->xquat[torso_body_id_ * 4 + i];
      }
      auto feet = GetFeetHeights();
      *(buffer++) = feet[0];
      *(buffer++) = feet[1];
      *(buffer++) = GetCom()[2];
      auto feet_rel = GetFeetRelativePositions();
      for (mjtNum value : feet_rel) {
        *(buffer++) = value;
      }
      *(buffer++) = static_cast<mjtNum>(
          std::fmod(static_cast<double>(gait_steps_) / hip_period_, 1.0));
      for (int i = 0; i < model_->nu; ++i) {
        *(buffer++) = data_->actuator_length[i];
      }
      for (int i = 0; i < model_->nu; ++i) {
        *(buffer++) =
            detail::ClipValue(data_->actuator_velocity[i], -100.0, 100.0);
      }
      for (int i = 0; i < model_->nu; ++i) {
        *(buffer++) =
            detail::ClipValue(data_->actuator_force[i] / 1000.0, -100.0, 100.0);
      }
      for (int i = 0; i < model_->na; ++i) {
        *(buffer++) = data_->act[i];
      }
      CommitObservation("obs", &obs, reset);
    }
    state["info:vel_reward"_] = reward.vel_reward;
    state["info:cyclic_hip"_] = reward.cyclic_hip;
    state["info:ref_rot"_] = reward.ref_rot;
    state["info:joint_angle_rew"_] = reward.joint_angle_rew;
    state["info:success"_] = reward.success;
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
    state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
    state["info:qacc0"_].Assign(qacc0_.data(), qacc0_.size());
    state["info:qacc_warmstart0"_].Assign(qacc_warmstart0_.data(),
                                          qacc_warmstart0_.size());
#endif
    state["info:phase_var"_] = static_cast<mjtNum>(
        std::fmod(static_cast<double>(gait_steps_) / hip_period_, 1.0));
  }
};

template <typename Spec>
using ReorientEnvBase = MyoSuiteReorientEnvBase<Spec, false>;

template <typename Spec>
using ReorientPixelEnvBase = MyoSuiteReorientEnvBase<Spec, true>;

template <typename Spec>
using WalkEnvBase = MyoSuiteWalkLikeEnvBase<Spec, false>;

template <typename Spec>
using WalkPixelEnvBase = MyoSuiteWalkLikeEnvBase<Spec, true>;

using ReorientEnv = ReorientEnvBase<MyoSuiteReorientEnvSpec>;
using ReorientPixelEnv = ReorientPixelEnvBase<MyoSuiteReorientPixelEnvSpec>;
using MyoSuiteReorientEnv = ReorientEnv;
using MyoSuiteReorientPixelEnv = ReorientPixelEnv;
using MyoSuiteReorientEnvPool = AsyncEnvPool<ReorientEnv>;
using MyoSuiteReorientPixelEnvPool = AsyncEnvPool<ReorientPixelEnv>;

using MyoSuiteWalkEnv = WalkEnvBase<MyoSuiteWalkEnvSpec>;
using MyoSuiteWalkPixelEnv = WalkPixelEnvBase<MyoSuiteWalkPixelEnvSpec>;
using MyoSuiteWalkEnvPool = AsyncEnvPool<MyoSuiteWalkEnv>;
using MyoSuiteWalkPixelEnvPool = AsyncEnvPool<MyoSuiteWalkPixelEnv>;

using MyoSuiteTerrainEnv = WalkEnvBase<MyoSuiteTerrainEnvSpec>;
using MyoSuiteTerrainPixelEnv = WalkPixelEnvBase<MyoSuiteTerrainPixelEnvSpec>;
using MyoSuiteTerrainEnvPool = AsyncEnvPool<MyoSuiteTerrainEnv>;
using MyoSuiteTerrainPixelEnvPool = AsyncEnvPool<MyoSuiteTerrainPixelEnv>;

}  // namespace myosuite_envpool

#endif  // ENVPOOL_MUJOCO_MYOSUITE_MYOBASE_EXTENDED_H_
