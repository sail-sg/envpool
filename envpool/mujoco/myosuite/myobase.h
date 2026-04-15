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

template <typename Spec>
using PosePixelEnvBase = MyoSuitePoseEnvBase<Spec, true>;

template <typename Spec>
using ReachPixelEnvBase = MyoSuiteReachEnvBase<Spec, true>;

using MyoSuitePoseEnv = MyoSuitePoseEnvBase<MyoSuitePoseEnvSpec, false>;
using MyoSuitePosePixelEnv = PosePixelEnvBase<MyoSuitePosePixelEnvSpec>;
using MyoSuitePoseEnvPool = AsyncEnvPool<MyoSuitePoseEnv>;
using MyoSuitePosePixelEnvPool = AsyncEnvPool<MyoSuitePosePixelEnv>;

using MyoSuiteReachEnv = MyoSuiteReachEnvBase<MyoSuiteReachEnvSpec, false>;
using MyoSuiteReachPixelEnv = ReachPixelEnvBase<MyoSuiteReachPixelEnvSpec>;
using MyoSuiteReachEnvPool = AsyncEnvPool<MyoSuiteReachEnv>;
using MyoSuiteReachPixelEnvPool = AsyncEnvPool<MyoSuiteReachPixelEnv>;

}  // namespace myosuite_envpool

#endif  // ENVPOOL_MUJOCO_MYOSUITE_MYOBASE_H_
