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

#ifndef ENVPOOL_MUJOCO_MYOSUITE_MYOSUITE_ENV_H_
#define ENVPOOL_MUJOCO_MYOSUITE_MYOSUITE_ENV_H_

#include <mujoco.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/robotics/mujoco_env.h"
#include "third_party/myosuite/myosuite_reference_data.h"
#include "third_party/myosuite/myosuite_task_metadata.h"
#include "third_party/myosuite/myosuite_tasks.h"

namespace myosuite {

constexpr int kMyoSuiteTestStatePad = 65536;

using envpool::mujoco::PixelObservationEnvFns;
using envpool::mujoco::RenderCameraIdOrDefault;
using envpool::mujoco::RenderHeightOrDefault;
using envpool::mujoco::RenderWidthOrDefault;
using envpool::mujoco::StackSpec;
using third_party::myosuite::GetMyoSuiteReferenceData;
using third_party::myosuite::GetMyoSuiteTask;
using third_party::myosuite::GetMyoSuiteTaskMetadata;
using third_party::myosuite::MyoSuiteMuscleCondition;
using third_party::myosuite::MyoSuiteReferenceData;
using third_party::myosuite::MyoSuiteReferenceType;
using third_party::myosuite::MyoSuiteTaskDef;
using third_party::myosuite::MyoSuiteTaskKind;
using third_party::myosuite::MyoSuiteTaskMetadata;

class MyoSuiteEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("frame_stack"_.Bind(1),
                    "task_name"_.Bind(std::string("myoFingerReachFixed-v0")));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const auto& task = GetMyoSuiteTask(std::string(conf["task_name"_]));
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs"_.Bind(StackSpec(Spec<mjtNum>({task.obs_dim}, {-inf, inf}),
                              conf["frame_stack"_])),
        "info:task_id"_.Bind(Spec<int>({-1})),
        "info:sparse"_.Bind(Spec<mjtNum>({-1})),
        "info:solved"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
        "info:oracle_numpy2_broken"_.Bind(Spec<bool>({})),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({2048})),
        "info:qvel0"_.Bind(Spec<mjtNum>({2048})),
        "info:act0"_.Bind(Spec<mjtNum>({2048})),
        "info:qacc0"_.Bind(Spec<mjtNum>({2048})),
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({2048})),
        "info:qpos"_.Bind(Spec<mjtNum>({2048})),
        "info:qvel"_.Bind(Spec<mjtNum>({2048})),
        "info:act"_.Bind(Spec<mjtNum>({2048})),
        "info:ctrl"_.Bind(Spec<mjtNum>({2048})),
        "info:qacc"_.Bind(Spec<mjtNum>({2048})),
        "info:qacc_warmstart"_.Bind(Spec<mjtNum>({2048})),
        "info:actuator_length"_.Bind(Spec<mjtNum>({2048})),
        "info:actuator_velocity"_.Bind(Spec<mjtNum>({2048})),
        "info:actuator_force"_.Bind(Spec<mjtNum>({2048})),
        "info:fatigue_ma"_.Bind(Spec<mjtNum>({2048})),
        "info:fatigue_mr"_.Bind(Spec<mjtNum>({2048})),
        "info:fatigue_mf"_.Bind(Spec<mjtNum>({2048})),
        "info:fatigue_tl"_.Bind(Spec<mjtNum>({2048})),
        "info:fatigue_tauact"_.Bind(Spec<mjtNum>({2048})),
        "info:fatigue_taudeact"_.Bind(Spec<mjtNum>({2048})),
        "info:fatigue_dt"_.Bind(Spec<mjtNum>({})),
        "info:site_pos"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:site_quat"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:site_xpos"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:site_size"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:site_rgba"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:body_pos"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:body_quat"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:body_mass"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:light_xpos"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:light_xdir"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:geom_pos"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:geom_quat"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:geom_size"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:geom_xpos"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:geom_xmat"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:geom_rgba"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:geom_friction"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:geom_aabb"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:geom_rbound"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:geom_contype"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:geom_conaffinity"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:geom_type"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:geom_condim"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:hfield_data"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:mocap_pos"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:mocap_quat"_.Bind(Spec<mjtNum>({kMyoSuiteTestStatePad})),
        "info:time"_.Bind(Spec<mjtNum>({})),
        "info:model_timestep"_.Bind(Spec<mjtNum>({})),
        "info:frame_skip"_.Bind(Spec<int>({})),
#endif
        "info:model_nq"_.Bind(Spec<int>({})),
        "info:model_nv"_.Bind(Spec<int>({})),
        "info:model_na"_.Bind(Spec<int>({})),
        "info:model_nu"_.Bind(Spec<int>({})),
        "info:model_nsite"_.Bind(Spec<int>({})),
        "info:model_nbody"_.Bind(Spec<int>({})),
        "info:model_ngeom"_.Bind(Spec<int>({})),
        "info:model_nhfielddata"_.Bind(Spec<int>({})),
        "info:model_nmocap"_.Bind(Spec<int>({})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    const auto& task = GetMyoSuiteTask(std::string(conf["task_name"_]));
    return MakeDict(
        "action"_.Bind(Spec<float>({-1, task.action_dim}, {-1.0, 1.0})));
  }
};

using MyoSuiteEnvSpec = EnvSpec<MyoSuiteEnvFns>;
using MyoSuitePixelEnvFns = PixelObservationEnvFns<MyoSuiteEnvFns>;
using MyoSuitePixelEnvSpec = EnvSpec<MyoSuitePixelEnvFns>;

template <typename EnvSpecT, bool kFromPixels>
class MyoSuiteEnvBase : public Env<EnvSpecT>,
                        public gymnasium_robotics::MujocoRobotEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  enum class OslPhase : std::uint8_t {
    kEStance,
    kLStance,
    kESwing,
    kLSwing,
  };

  struct OslStateParams {
    mjtNum knee_stiffness;
    mjtNum knee_damping;
    mjtNum knee_target_angle;
    mjtNum ankle_stiffness;
    mjtNum ankle_damping;
    mjtNum ankle_target_angle;
  };

  const MyoSuiteTaskDef& task_;
  const MyoSuiteTaskMetadata& metadata_;
  const MyoSuiteReferenceData& reference_;
  int task_index_;
  std::vector<std::string> obs_keys_;
  std::vector<std::pair<std::string, mjtNum>> reward_weights_;
  std::vector<mjtNum> metadata_init_qpos_;
  std::vector<mjtNum> metadata_init_qvel_;
  std::vector<mjtNum> reset_qacc_warmstart_;
  std::vector<mjtNum> target_jnt_value_;
  std::vector<std::string> tip_sites_;
  std::vector<std::string> target_sites_;
  std::vector<std::vector<mjtNum>> target_reach_low_;
  std::vector<std::vector<mjtNum>> target_reach_high_;
  std::vector<mjtNum> last_ctrl_;
  std::vector<int> muscle_actuator_ids_;
  std::vector<mjtNum> fatigue_tauact_;
  std::vector<mjtNum> fatigue_taudeact_;
  std::vector<mjtNum> fatigue_ma_;
  std::vector<mjtNum> fatigue_mr_;
  std::vector<mjtNum> fatigue_mf_;
  std::vector<mjtNum> fatigue_tl_;
  int task_step_{0};
  int myodm_reference_index_{0};
  mjtNum myodm_lift_z_{0.0};
  OslPhase osl_phase_{OslPhase::kEStance};
  mjtNum osl_body_weight_{0.0};
  int osl_knee_actuator_id_{-1};
  int osl_ankle_actuator_id_{-1};
  int osl_knee_joint_id_{-1};
  int osl_ankle_joint_id_{-1};
  int osl_load_sensor_id_{-1};
  std::vector<mjtNum> tabletennis_init_paddle_quat_;
  std::array<mjtNum, 3> challenge_reorient_goal_obj_offset_{};
  int bimanual_goal_touch_{0};
  mjtNum bimanual_init_obj_z_{0.0};
  mjtNum bimanual_init_palm_z_{0.0};
  mjtNum sparse_{0.0};
  mjtNum solved_{0.0};
#ifdef ENVPOOL_TEST
  std::vector<mjtNum> qpos0_pad_;
  std::vector<mjtNum> qvel0_pad_;
  std::vector<mjtNum> act0_pad_;
  std::vector<mjtNum> qacc0_pad_;
  std::vector<mjtNum> qacc_warmstart0_pad_;
  std::vector<mjtNum> qpos_pad_;
  std::vector<mjtNum> qvel_pad_;
  std::vector<mjtNum> act_pad_;
  std::vector<mjtNum> ctrl_pad_;
  std::vector<mjtNum> qacc_pad_;
  std::vector<mjtNum> qacc_warmstart_pad_;
  std::vector<mjtNum> actuator_length_pad_;
  std::vector<mjtNum> actuator_velocity_pad_;
  std::vector<mjtNum> actuator_force_pad_;
  std::vector<mjtNum> fatigue_ma_pad_;
  std::vector<mjtNum> fatigue_mr_pad_;
  std::vector<mjtNum> fatigue_mf_pad_;
  std::vector<mjtNum> fatigue_tl_pad_;
  std::vector<mjtNum> fatigue_tauact_pad_;
  std::vector<mjtNum> fatigue_taudeact_pad_;
  std::vector<mjtNum> site_pos_pad_;
  std::vector<mjtNum> site_quat_pad_;
  std::vector<mjtNum> site_xpos_pad_;
  std::vector<mjtNum> site_size_pad_;
  std::vector<mjtNum> site_rgba_pad_;
  std::vector<mjtNum> body_pos_pad_;
  std::vector<mjtNum> body_quat_pad_;
  std::vector<mjtNum> body_mass_pad_;
  std::vector<mjtNum> light_xpos_pad_;
  std::vector<mjtNum> light_xdir_pad_;
  std::vector<mjtNum> geom_pos_pad_;
  std::vector<mjtNum> geom_quat_pad_;
  std::vector<mjtNum> geom_size_pad_;
  std::vector<mjtNum> geom_xpos_pad_;
  std::vector<mjtNum> geom_xmat_pad_;
  std::vector<mjtNum> geom_rgba_pad_;
  std::vector<mjtNum> geom_friction_pad_;
  std::vector<mjtNum> geom_aabb_pad_;
  std::vector<mjtNum> geom_rbound_pad_;
  std::vector<mjtNum> geom_contype_pad_;
  std::vector<mjtNum> geom_conaffinity_pad_;
  std::vector<mjtNum> geom_type_pad_;
  std::vector<mjtNum> geom_condim_pad_;
  std::vector<mjtNum> hfield_data_pad_;
  std::vector<mjtNum> mocap_pos_pad_;
  std::vector<mjtNum> mocap_quat_pad_;
#endif

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  MyoSuiteEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        gymnasium_robotics::MujocoRobotEnv(
            spec.config["base_path"_],
            AssetPath(spec.config["base_path"_],
                      GetMyoSuiteTask(std::string(spec.config["task_name"_]))
                          .model_path),
            GetMyoSuiteTask(std::string(spec.config["task_name"_])).frame_skip,
            spec.config["max_episode_steps"_], spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        task_(GetMyoSuiteTask(std::string(spec.config["task_name"_]))),
        metadata_(
            GetMyoSuiteTaskMetadata(std::string(spec.config["task_name"_]))),
        reference_(
            GetMyoSuiteReferenceData(std::string(spec.config["task_name"_]))),
        task_index_(TaskIndex(task_.id)),
        obs_keys_(SplitList(metadata_.obs_keys)),
        reward_weights_(ParseWeights(metadata_.rwd_keys_wt)),
        metadata_init_qpos_(ParseNumbers(metadata_.init_qpos)),
        metadata_init_qvel_(ParseNumbers(metadata_.init_qvel)),
        reset_qacc_warmstart_(ParseNumbers(metadata_.reset_qacc_warmstart)),
        target_jnt_value_(ParseNumbers(metadata_.target_jnt_value)),
        tip_sites_(SplitList(metadata_.tip_sites)),
        target_sites_(SplitList(metadata_.target_sites)),
        target_reach_low_(ParseNumberGroups(metadata_.target_reach_low)),
        target_reach_high_(ParseNumberGroups(metadata_.target_reach_high)),
        last_ctrl_(model_->nu, 0.0)
#ifdef ENVPOOL_TEST
        ,
        qpos0_pad_(2048, 0.0),
        qvel0_pad_(2048, 0.0),
        act0_pad_(2048, 0.0),
        qacc0_pad_(2048, 0.0),
        qacc_warmstart0_pad_(2048, 0.0),
        qpos_pad_(2048, 0.0),
        qvel_pad_(2048, 0.0),
        act_pad_(2048, 0.0),
        ctrl_pad_(2048, 0.0),
        qacc_pad_(2048, 0.0),
        qacc_warmstart_pad_(2048, 0.0),
        actuator_length_pad_(2048, 0.0),
        actuator_velocity_pad_(2048, 0.0),
        actuator_force_pad_(2048, 0.0),
        fatigue_ma_pad_(2048, 0.0),
        fatigue_mr_pad_(2048, 0.0),
        fatigue_mf_pad_(2048, 0.0),
        fatigue_tl_pad_(2048, 0.0),
        fatigue_tauact_pad_(2048, 0.0),
        fatigue_taudeact_pad_(2048, 0.0),
        site_pos_pad_(kMyoSuiteTestStatePad, 0.0),
        site_quat_pad_(kMyoSuiteTestStatePad, 0.0),
        site_xpos_pad_(kMyoSuiteTestStatePad, 0.0),
        site_size_pad_(kMyoSuiteTestStatePad, 0.0),
        site_rgba_pad_(kMyoSuiteTestStatePad, 0.0),
        body_pos_pad_(kMyoSuiteTestStatePad, 0.0),
        body_quat_pad_(kMyoSuiteTestStatePad, 0.0),
        body_mass_pad_(kMyoSuiteTestStatePad, 0.0),
        light_xpos_pad_(kMyoSuiteTestStatePad, 0.0),
        light_xdir_pad_(kMyoSuiteTestStatePad, 0.0),
        geom_pos_pad_(kMyoSuiteTestStatePad, 0.0),
        geom_quat_pad_(kMyoSuiteTestStatePad, 0.0),
        geom_size_pad_(kMyoSuiteTestStatePad, 0.0),
        geom_xpos_pad_(kMyoSuiteTestStatePad, 0.0),
        geom_xmat_pad_(kMyoSuiteTestStatePad, 0.0),
        geom_rgba_pad_(kMyoSuiteTestStatePad, 0.0),
        geom_friction_pad_(kMyoSuiteTestStatePad, 0.0),
        geom_aabb_pad_(kMyoSuiteTestStatePad, 0.0),
        geom_rbound_pad_(kMyoSuiteTestStatePad, 0.0),
        geom_contype_pad_(kMyoSuiteTestStatePad, 0.0),
        geom_conaffinity_pad_(kMyoSuiteTestStatePad, 0.0),
        geom_type_pad_(kMyoSuiteTestStatePad, 0.0),
        geom_condim_pad_(kMyoSuiteTestStatePad, 0.0),
        hfield_data_pad_(kMyoSuiteTestStatePad, 0.0),
        mocap_pos_pad_(kMyoSuiteTestStatePad, 0.0),
        mocap_quat_pad_(kMyoSuiteTestStatePad, 0.0)
#endif
  {
    ApplyMuscleCondition();
    InitializeFatigue();
    ApplyMyoDmModelEdits();
    ApplyBaodingModelEdits();
    ApplyMetadataInitialState();
    SetDefaultInitialQpos();
    ApplyBimanualInitialState();
    InitializeRobotEnv();
    InitializeTaskCaches();
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    task_step_ = 0;
    myodm_reference_index_ = 0;
    bimanual_goal_touch_ = 0;
    sparse_ = 0.0;
    solved_ = 0.0;
    ResetFatigue();
    ResetToInitialState();
    ResetOslController();
    ApplyResetTargets();
    WarmstartFromCurrentAcceleration();
    std::fill(last_ctrl_.begin(), last_ctrl_.end(), 0.0);
    CapturePaddedResetState();
    WriteState(0.0, true);
  }

  void Step(const Action& action) override {
    const auto* raw = static_cast<float*>(action["action"_].Data());
    std::vector<mjtNum> ctrl(model_->nu);
    const int action_dim =
        std::min(static_cast<int>(model_->nu), task_.action_dim);
    for (int i = 0; i < action_dim; ++i) {
      mjtNum value = std::max<mjtNum>(-1.0, std::min<mjtNum>(1.0, raw[i]));
      if (model_->na > 0 && task_.normalize_act &&
          model_->actuator_dyntype[i] == mjDYN_MUSCLE) {
        const auto action_value = static_cast<float>(value);
        value = static_cast<mjtNum>(
            1.0F / (1.0F + std::exp(-5.0F * (action_value - 0.5F))));
      }
      if (task_.muscle_condition == MyoSuiteMuscleCondition::kReafferentation) {
        // Applied below after EIP has been copied to EPL.
      }
      ctrl[i] = value;
    }
    ApplyFatigue(&ctrl);
    ApplyReafferentation(&ctrl);
    ApplyOslControls(&ctrl);
    PreStepTaskUpdate();
    const bool robot_step_normalizes_ctrl =
        task_.normalize_act &&
        (model_->na == 0 ||
         task_.kind == MyoSuiteTaskKind::kChallengeTableTennis);
    for (int i = 0; i < model_->nu; ++i) {
      if (robot_step_normalizes_ctrl &&
          (model_->na == 0 || model_->actuator_dyntype[i] != mjDYN_MUSCLE)) {
        const mjtNum low = model_->actuator_ctrlrange[2 * i];
        const mjtNum high = model_->actuator_ctrlrange[2 * i + 1];
        ctrl[i] = (low + high) * 0.5 + ctrl[i] * (high - low) * 0.5;
      }
      data_->ctrl[i] = ctrl[i];
    }
    last_ctrl_ = ctrl;
    DoSimulation();
    // MyoSuite observes through Robot.sensor2sim(), which calls sim.forward()
    // after copying the final qpos/qvel/act back into the observed sim.
    mj_forward(model_, data_);
    ++elapsed_step_;
    const auto obs_dict = BuildObsDict();
    const RewardResult reward = ComputeReward(obs_dict);
    sparse_ = reward.sparse;
    solved_ = reward.solved;
    done_ = reward.terminated || elapsed_step_ >= max_episode_steps_;
    WriteState(reward.dense, false, obs_dict);
    PostStepTaskUpdate();
  }

  bool RenderCamera(mjvCamera* camera) override {
    mjv_defaultCamera(camera);
    camera->type = mjCAMERA_FREE;
    camera->fixedcamid = -1;
    mjv_defaultFreeCamera(model_, camera);
    return true;
  }

  bool RenderOption(mjvOption* option) override {
    mjv_defaultOption(option);
    option->flags[mjVIS_ACTUATOR] = 1;
    option->flags[mjVIS_ACTIVATION] = 1;
    if (task_.kind == MyoSuiteTaskKind::kChallengeRunTrack ||
        task_.kind == MyoSuiteTaskKind::kChallengeChaseTag ||
        task_.kind == MyoSuiteTaskKind::kChallengeSoccer) {
      option->flags[mjVIS_TENDON] = 1;
    }
    return true;
  }

  void RenderCallback() override { mj_forward(model_, data_); }

  bool DisableAuxiliaryRenderVisuals() const override { return false; }

  bool ShareRenderContext() const override { return true; }

  bool PreferOfflineRenderContext() const override { return true; }

 protected:
  struct RewardResult {
    mjtNum dense{0.0};
    mjtNum sparse{0.0};
    mjtNum solved{0.0};
    bool terminated{false};
  };

  struct MyoDmReferenceFrame {
    std::vector<mjtNum> robot;
    std::vector<mjtNum> robot_vel;
    std::vector<mjtNum> object;
  };

  using ObsDict = std::unordered_map<std::string, std::vector<mjtNum>>;

  static std::vector<std::string> SplitList(std::string_view text,
                                            char delimiter = ',') {
    std::vector<std::string> result;
    std::string item;
    std::stringstream stream{std::string(text)};
    while (std::getline(stream, item, delimiter)) {
      if (!item.empty()) {
        result.push_back(item);
      }
    }
    return result;
  }

  static std::vector<mjtNum> ParseNumbers(std::string_view text) {
    std::vector<mjtNum> result;
    for (const auto& item : SplitList(text)) {
      result.push_back(static_cast<mjtNum>(std::stod(item)));
    }
    return result;
  }

  static std::vector<std::vector<mjtNum>> ParseNumberGroups(
      std::string_view text) {
    std::vector<std::vector<mjtNum>> result;
    for (const auto& group : SplitList(text, ';')) {
      result.push_back(ParseNumbers(group));
    }
    return result;
  }

  static std::vector<std::pair<std::string, mjtNum>> ParseWeights(
      std::string_view text) {
    std::vector<std::pair<std::string, mjtNum>> result;
    for (const auto& item : SplitList(text)) {
      const std::size_t sep = item.find(':');
      if (sep == std::string::npos) {
        continue;
      }
      result.emplace_back(item.substr(0, sep),
                          static_cast<mjtNum>(std::stod(item.substr(sep + 1))));
    }
    return result;
  }

  static mjtNum Norm(const std::vector<mjtNum>& values) {
    mjtNum sum = 0.0;
    for (mjtNum value : values) {
      sum += value * value;
    }
    return std::sqrt(sum);
  }

  static mjtNum SquaredNorm(const std::vector<mjtNum>& values) {
    mjtNum sum = 0.0;
    for (mjtNum value : values) {
      sum += value * value;
    }
    return sum;
  }

  static mjtNum MeanSquare(const std::vector<mjtNum>& values) {
    if (values.empty()) {
      return 0.0;
    }
    return SquaredNorm(values) / static_cast<mjtNum>(values.size());
  }

  static mjtNum Dot(const std::vector<mjtNum>& lhs,
                    const std::vector<mjtNum>& rhs) {
    mjtNum sum = 0.0;
    const int size =
        std::min(static_cast<int>(lhs.size()), static_cast<int>(rhs.size()));
    for (int i = 0; i < size; ++i) {
      sum += lhs[i] * rhs[i];
    }
    return sum;
  }

  static mjtNum Cosine(const std::vector<mjtNum>& lhs,
                       const std::vector<mjtNum>& rhs) {
    const mjtNum denom = Norm(lhs) * Norm(rhs);
    if (denom <= 0.0) {
      return 0.0;
    }
    return Dot(lhs, rhs) / denom;
  }

  static mjtNum QuaternionDistance(const std::vector<mjtNum>& current,
                                   const std::vector<mjtNum>& target) {
    if (current.size() < 4 || target.size() < 4) {
      return 0.0;
    }
    const mjtNum cw = current[0];
    const mjtNum cx = current[1];
    const mjtNum cy = current[2];
    const mjtNum cz = current[3];
    const mjtNum tw = target[0];
    const mjtNum tx = -target[1];
    const mjtNum ty = -target[2];
    const mjtNum tz = -target[3];
    const mjtNum dw = cw * tw - cx * tx - cy * ty - cz * tz;
    const mjtNum dx = cw * tx + cx * tw + cy * tz - cz * ty;
    const mjtNum dy = cw * ty - cx * tz + cy * tw + cz * tx;
    const mjtNum dz = cw * tz + cx * ty - cy * tx + cz * tw;
    const mjtNum axis_norm = std::sqrt(dx * dx + dy * dy + dz * dz);
    return std::abs(2.0 * std::atan2(axis_norm, dw));
  }

  static mjtNum YawFromQuat(const mjtNum* quat) {
    const mjtNum w = quat[0];
    const mjtNum x = quat[1];
    const mjtNum y = quat[2];
    const mjtNum z = quat[3];
    return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
  }

  static bool StartsWith(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() &&
           value.substr(0, prefix.size()) == prefix;
  }

  static int JointQposWidth(int joint_type) {
    if (joint_type == mjJNT_FREE) {
      return 7;
    }
    if (joint_type == mjJNT_BALL) {
      return 4;
    }
    return 1;
  }

  static int JointDofWidth(int joint_type) {
    if (joint_type == mjJNT_FREE) {
      return 6;
    }
    if (joint_type == mjJNT_BALL) {
      return 3;
    }
    return 1;
  }

  static std::vector<mjtNum> Subtract(const std::vector<mjtNum>& lhs,
                                      const std::vector<mjtNum>& rhs) {
    const int size =
        std::min(static_cast<int>(lhs.size()), static_cast<int>(rhs.size()));
    std::vector<mjtNum> result(size);
    for (int i = 0; i < size; ++i) {
      result[i] = lhs[i] - rhs[i];
    }
    return result;
  }

  static std::vector<mjtNum> Add(const std::vector<mjtNum>& lhs,
                                 const std::vector<mjtNum>& rhs) {
    const int size =
        std::min(static_cast<int>(lhs.size()), static_cast<int>(rhs.size()));
    std::vector<mjtNum> result(size);
    for (int i = 0; i < size; ++i) {
      result[i] = lhs[i] + rhs[i];
    }
    return result;
  }

  static std::vector<mjtNum> MatToEuler(const mjtNum* mat) {
    const mjtNum eps4 = std::numeric_limits<mjtNum>::epsilon() * 4.0;
    const mjtNum cy = std::sqrt(mat[8] * mat[8] + mat[5] * mat[5]);
    std::vector<mjtNum> euler(3, 0.0);
    if (cy > eps4) {
      euler[2] = -std::atan2(mat[1], mat[0]);
      euler[1] = -std::atan2(-mat[2], cy);
      euler[0] = -std::atan2(mat[5], mat[8]);
    } else {
      euler[2] = -std::atan2(-mat[3], mat[4]);
      euler[1] = -std::atan2(-mat[2], cy);
      euler[0] = 0.0;
    }
    return euler;
  }

  int SiteId(const char* name) const {
    return mj_name2id(model_, mjOBJ_SITE, name);
  }

  int BodyId(const char* name) const {
    return mj_name2id(model_, mjOBJ_BODY, name);
  }

  int GeomId(const char* name) const {
    return mj_name2id(model_, mjOBJ_GEOM, name);
  }

  int JointId(const char* name) const {
    return mj_name2id(model_, mjOBJ_JOINT, name);
  }

  int ActuatorId(const char* name) const {
    return mj_name2id(model_, mjOBJ_ACTUATOR, name);
  }

  int SensorId(const char* name) const {
    return mj_name2id(model_, mjOBJ_SENSOR, name);
  }

  std::vector<mjtNum> SiteXpos(int site_id) const {
    if (site_id < 0) {
      return {0.0, 0.0, 0.0};
    }
    return {data_->site_xpos[3 * site_id], data_->site_xpos[3 * site_id + 1],
            data_->site_xpos[3 * site_id + 2]};
  }

  std::vector<mjtNum> BodyXpos(int body_id) const {
    if (body_id < 0) {
      return {0.0, 0.0, 0.0};
    }
    return {data_->xpos[3 * body_id], data_->xpos[3 * body_id + 1],
            data_->xpos[3 * body_id + 2]};
  }

  std::vector<mjtNum> BodyPos(int body_id) const {
    if (body_id < 0) {
      return {0.0, 0.0, 0.0};
    }
    return {model_->body_pos[3 * body_id], model_->body_pos[3 * body_id + 1],
            model_->body_pos[3 * body_id + 2]};
  }

  std::vector<mjtNum> BodyXquat(int body_id) const {
    if (body_id < 0) {
      return {1.0, 0.0, 0.0, 0.0};
    }
    return {data_->xquat[4 * body_id], data_->xquat[4 * body_id + 1],
            data_->xquat[4 * body_id + 2], data_->xquat[4 * body_id + 3]};
  }

  std::vector<mjtNum> BodyXmat(int body_id) const {
    std::vector<mjtNum> result(9, 0.0);
    if (body_id >= 0) {
      std::memcpy(result.data(), data_->ximat + 9 * body_id,
                  sizeof(mjtNum) * result.size());
    }
    return result;
  }

  std::vector<mjtNum> GeomXpos(int geom_id) const {
    if (geom_id < 0) {
      return {0.0, 0.0, 0.0};
    }
    return {data_->geom_xpos[3 * geom_id], data_->geom_xpos[3 * geom_id + 1],
            data_->geom_xpos[3 * geom_id + 2]};
  }

  std::vector<mjtNum> SensorData(const char* name) const {
    int sensor_id = SensorId(name);
    if (sensor_id < 0) {
      return {};
    }
    const int start = model_->sensor_adr[sensor_id];
    const int dim = model_->sensor_dim[sensor_id];
    return {data_->sensordata + start, data_->sensordata + start + dim};
  }

  static std::string AssetPath(const std::string& base_path,
                               const std::string& model_path) {
    return base_path + "/mujoco/myosuite/assets/" + model_path;
  }

  static int TaskIndex(std::string_view task_id) {
    const auto& tasks = third_party::myosuite::kMyoSuiteTasks;
    for (int i = 0; i < static_cast<int>(tasks.size()); ++i) {
      if (tasks[i].id == task_id) {
        return i;
      }
    }
    throw std::runtime_error("Unknown MyoSuite task index.");
  }

  void ApplyMuscleCondition() {
    if (task_.muscle_condition == MyoSuiteMuscleCondition::kSarcopenia) {
      for (int i = 0; i < model_->nu; ++i) {
        model_->actuator_gainprm[i * mjNGAIN + 2] *= 0.5;
      }
    }
  }

  void InitializeFatigue() {
    if (task_.muscle_condition != MyoSuiteMuscleCondition::kFatigue) {
      return;
    }
    for (int i = 0; i < model_->nu; ++i) {
      if (model_->actuator_dyntype[i] == mjDYN_MUSCLE) {
        muscle_actuator_ids_.push_back(i);
        fatigue_tauact_.push_back(model_->actuator_dynprm[i * mjNDYN]);
        fatigue_taudeact_.push_back(model_->actuator_dynprm[i * mjNDYN + 1]);
      }
    }
    fatigue_ma_.resize(muscle_actuator_ids_.size());
    fatigue_mr_.resize(muscle_actuator_ids_.size());
    fatigue_mf_.resize(muscle_actuator_ids_.size());
    fatigue_tl_.resize(muscle_actuator_ids_.size());
    ResetFatigue();
  }

  void ResetFatigue() {
    if (task_.muscle_condition != MyoSuiteMuscleCondition::kFatigue) {
      return;
    }
    std::fill(fatigue_ma_.begin(), fatigue_ma_.end(), 0.0);
    std::fill(fatigue_mr_.begin(), fatigue_mr_.end(), 1.0);
    std::fill(fatigue_mf_.begin(), fatigue_mf_.end(), 0.0);
    std::fill(fatigue_tl_.begin(), fatigue_tl_.end(), 0.0);
  }

  void InitializeOslController() {
    if (task_.kind != MyoSuiteTaskKind::kChallengeRunTrack) {
      return;
    }
    osl_knee_actuator_id_ = ActuatorId("osl_knee_torque_actuator");
    osl_ankle_actuator_id_ = ActuatorId("osl_ankle_torque_actuator");
    osl_knee_joint_id_ = JointId("osl_knee_angle_r");
    osl_ankle_joint_id_ = JointId("osl_ankle_angle_r");
    osl_load_sensor_id_ = SensorId("r_osl_load");
    mjtNum body_mass = 0.0;
    for (int i = 0; i < model_->nbody; ++i) {
      body_mass += model_->body_mass[i];
    }
    osl_body_weight_ = body_mass * static_cast<mjtNum>(9.81);
    ResetOslController();
  }

  void ResetOslController() {
    if (task_.kind != MyoSuiteTaskKind::kChallengeRunTrack) {
      return;
    }
    osl_phase_ = OslPhase::kEStance;
    if (model_->nkey < 3) {
      return;
    }
    int closest_key = 0;
    mjtNum closest_distance = std::numeric_limits<mjtNum>::infinity();
    for (int key = 0; key < 3; ++key) {
      const mjtNum* key_qpos = model_->key_qpos + key * model_->nq;
      mjtNum distance = 0.0;
      for (int i = std::min<int>(7, model_->nq); i < model_->nq; ++i) {
        const mjtNum diff = data_->qpos[i] - key_qpos[i];
        distance += diff * diff;
      }
      if (distance < closest_distance) {
        closest_distance = distance;
        closest_key = key;
      }
    }
    osl_phase_ = closest_key == 1 ? OslPhase::kESwing : OslPhase::kEStance;
  }

  static mjtNum DegreesToRadians(mjtNum degrees) {
    return degrees * std::acos(static_cast<mjtNum>(-1.0)) /
           static_cast<mjtNum>(180.0);
  }

  OslStateParams CurrentOslStateParams() const {
    switch (osl_phase_) {
      case OslPhase::kLStance:
        return {99.372, 1.272, DegreesToRadians(8.0),
                79.498, 0.063, DegreesToRadians(-20.0)};
      case OslPhase::kESwing:
        return {39.749, 0.063, DegreesToRadians(60.0),
                7.949,  0.0,   DegreesToRadians(25.0)};
      case OslPhase::kLSwing:
        return {15.899, 3.816, DegreesToRadians(5.0),
                7.949,  0.0,   DegreesToRadians(15.0)};
      case OslPhase::kEStance:
      default:
        return {99.372, 3.180, DegreesToRadians(5.0),
                19.874, 0.0,   DegreesToRadians(-2.0)};
    }
  }

  void UpdateOslPhase(mjtNum knee_angle, mjtNum knee_vel, mjtNum ankle_angle,
                      mjtNum load) {
    switch (osl_phase_) {
      case OslPhase::kEStance:
        if (load > 0.25 * osl_body_weight_ ||
            ankle_angle > DegreesToRadians(6.0)) {
          osl_phase_ = OslPhase::kLStance;
        }
        break;
      case OslPhase::kLStance:
        if (load < 0.15 * osl_body_weight_) {
          osl_phase_ = OslPhase::kESwing;
        }
        break;
      case OslPhase::kESwing:
        if (knee_angle > DegreesToRadians(50.0) ||
            knee_vel < DegreesToRadians(3.0)) {
          osl_phase_ = OslPhase::kLSwing;
        }
        break;
      case OslPhase::kLSwing:
        if (load > 0.4 * osl_body_weight_ ||
            knee_angle < DegreesToRadians(30.0)) {
          osl_phase_ = OslPhase::kEStance;
        }
        break;
    }
  }

  mjtNum JointQpos(int joint_id) const {
    if (joint_id < 0) {
      return 0.0;
    }
    return data_->qpos[model_->jnt_qposadr[joint_id]];
  }

  mjtNum JointQvel(int joint_id) const {
    if (joint_id < 0) {
      return 0.0;
    }
    return data_->qvel[model_->jnt_dofadr[joint_id]];
  }

  mjtNum OslLoad() const {
    if (osl_load_sensor_id_ < 0 ||
        model_->sensor_dim[osl_load_sensor_id_] <= 1) {
      return 0.0;
    }
    const int adr = model_->sensor_adr[osl_load_sensor_id_];
    return -data_->sensordata[adr + 1];
  }

  mjtNum OslActuatorControl(int actuator_id, mjtNum torque) const {
    if (actuator_id < 0) {
      return 0.0;
    }
    const mjtNum gear = model_->actuator_gear[6 * actuator_id];
    mjtNum ctrl = gear != 0.0 ? torque / gear : 0.0;
    const mjtNum low = model_->actuator_ctrlrange[2 * actuator_id];
    const mjtNum high = model_->actuator_ctrlrange[2 * actuator_id + 1];
    ctrl = std::max(low, std::min(high, ctrl));
    if (task_.normalize_act) {
      const mjtNum mean = (low + high) * 0.5;
      const mjtNum range = (high - low) * 0.5;
      ctrl = range != 0.0 ? (ctrl - mean) / range : 0.0;
    }
    return ctrl;
  }

  void ApplyOslControls(std::vector<mjtNum>* ctrl) {
    if (task_.kind != MyoSuiteTaskKind::kChallengeRunTrack ||
        osl_knee_actuator_id_ < 0 || osl_ankle_actuator_id_ < 0) {
      return;
    }
    const mjtNum knee_angle = JointQpos(osl_knee_joint_id_);
    const mjtNum knee_vel = JointQvel(osl_knee_joint_id_);
    const mjtNum ankle_angle = JointQpos(osl_ankle_joint_id_);
    const mjtNum ankle_vel = JointQvel(osl_ankle_joint_id_);
    UpdateOslPhase(knee_angle, knee_vel, ankle_angle, OslLoad());
    const auto params = CurrentOslStateParams();
    const mjtNum knee_torque = std::max<mjtNum>(
        -142.272, std::min<mjtNum>(
                      142.272, params.knee_stiffness *
                                       (params.knee_target_angle - knee_angle) -
                                   params.knee_damping * knee_vel));
    const mjtNum ankle_torque = std::max<mjtNum>(
        -168.192,
        std::min<mjtNum>(
            168.192,
            params.ankle_stiffness * (params.ankle_target_angle - ankle_angle) -
                params.ankle_damping * ankle_vel));
    (*ctrl)[osl_knee_actuator_id_] =
        OslActuatorControl(osl_knee_actuator_id_, knee_torque);
    (*ctrl)[osl_ankle_actuator_id_] =
        OslActuatorControl(osl_ankle_actuator_id_, ankle_torque);
  }

  void ApplyFatigue(std::vector<mjtNum>* ctrl) {
    if (task_.muscle_condition != MyoSuiteMuscleCondition::kFatigue) {
      return;
    }
    constexpr mjtNum k_recovery_multiplier = 10.0 * 15.0;
    constexpr mjtNum k_fatigue_coefficient = 0.00912;
    constexpr mjtNum k_recovery_coefficient = 0.1 * 0.00094;
    const auto dt = static_cast<mjtNum>(Dt());
    for (std::size_t i = 0; i < muscle_actuator_ids_.size(); ++i) {
      const int actuator_id = muscle_actuator_ids_[i];
      fatigue_tl_[i] = (*ctrl)[actuator_id];

      const mjtNum ma = fatigue_ma_[i];
      const mjtNum mr = fatigue_mr_[i];
      const mjtNum mf = fatigue_mf_[i];
      const mjtNum tl = fatigue_tl_[i];
      const mjtNum ld = (1.0 / fatigue_tauact_[i]) * (0.5 + 1.5 * ma);
      const mjtNum lr = (0.5 + 1.5 * ma) / fatigue_taudeact_[i];

      mjtNum transfer = 0.0;
      if (ma < tl && mr > tl - ma) {
        transfer = ld * (tl - ma);
      } else if (ma < tl) {
        transfer = ld * mr;
      } else {
        transfer = lr * (tl - ma);
      }

      const mjtNum recovery =
          ma >= tl ? k_recovery_multiplier * k_recovery_coefficient
                   : k_recovery_coefficient;
      const mjtNum lower = std::max(-ma / dt + k_fatigue_coefficient * ma,
                                    (mr - 1.0) / dt + recovery * mf);
      const mjtNum upper =
          std::min((1.0 - ma) / dt + k_fatigue_coefficient * ma,
                   mr / dt + recovery * mf);
      transfer = std::max(lower, std::min(upper, transfer));

      fatigue_ma_[i] += (transfer - k_fatigue_coefficient * ma) * dt;
      fatigue_mr_[i] += (-transfer + recovery * mf) * dt;
      fatigue_mf_[i] += (k_fatigue_coefficient * ma - recovery * mf) * dt;
      (*ctrl)[actuator_id] =
          static_cast<mjtNum>(static_cast<float>(fatigue_ma_[i]));
    }
  }

  void ApplyReafferentation(std::vector<mjtNum>* ctrl) const {
    if (task_.muscle_condition != MyoSuiteMuscleCondition::kReafferentation) {
      return;
    }
    int epl = mj_name2id(model_, mjOBJ_ACTUATOR, "EPL");
    int eip = mj_name2id(model_, mjOBJ_ACTUATOR, "EIP");
    if (epl >= 0 && eip >= 0) {
      (*ctrl)[epl] = (*ctrl)[eip];
      (*ctrl)[eip] = 0.0;
    }
  }

  void ApplyMyoDmModelEdits() {
    if (task_.kind != MyoSuiteTaskKind::kMyoDmTrack) {
      return;
    }
    const int body_geom = GeomId("body");
    if (body_geom >= 0) {
      model_->geom_rgba[4 * body_geom + 3] = 0.0;
    }
  }

  void ApplyMetadataInitialState() {
    if (static_cast<int>(metadata_init_qpos_.size()) == model_->nq) {
      std::memcpy(data_->qpos, metadata_init_qpos_.data(),
                  sizeof(mjtNum) * model_->nq);
    }
    if (static_cast<int>(metadata_init_qvel_.size()) == model_->nv) {
      std::memcpy(data_->qvel, metadata_init_qvel_.data(),
                  sizeof(mjtNum) * model_->nv);
    }
    if (model_->na > 0) {
      mju_zero(data_->act, model_->na);
    }
    mj_forward(model_, data_);
  }

  void ApplyBaodingModelEdits() {
    if (task_.kind != MyoSuiteTaskKind::kChallengeBaoding) {
      return;
    }
    const int target1 = SiteId("target1_site");
    const int target2 = SiteId("target2_site");
    if (target1 >= 0) {
      model_->site_group[target1] = 2;
    }
    if (target2 >= 0) {
      model_->site_group[target2] = 2;
    }
  }

  void SetDefaultInitialQpos() {
    if (!task_.normalize_act) {
      return;
    }
    for (int actuator_id = 0; actuator_id < model_->nu; ++actuator_id) {
      if (model_->actuator_trntype[actuator_id] != mjTRN_JOINT) {
        continue;
      }
      const int joint_id = model_->actuator_trnid[2 * actuator_id];
      if (joint_id < 0) {
        continue;
      }
      const int joint_type = model_->jnt_type[joint_id];
      if (joint_type != mjJNT_HINGE && joint_type != mjJNT_SLIDE) {
        continue;
      }
      const int qpos_id = model_->jnt_qposadr[joint_id];
      if (metadata_init_qpos_.empty()) {
        data_->qpos[qpos_id] = (model_->jnt_range[2 * joint_id] +
                                model_->jnt_range[2 * joint_id + 1]) *
                               0.5;
      }
    }
    mj_forward(model_, data_);
  }

  void ApplyBimanualInitialState() {
    if (task_.kind != MyoSuiteTaskKind::kChallengeBimanual ||
        model_->nkey <= 2) {
      return;
    }
    std::memcpy(data_->qpos, model_->key_qpos + 2 * model_->nq,
                sizeof(mjtNum) * model_->nq);
    std::fill(data_->qvel, data_->qvel + model_->nv, 0.0);
    if (model_->na > 0) {
      mju_zero(data_->act, model_->na);
    }
    mj_forward(model_, data_);
  }

  void ApplyResetTargets() {
    const int count = std::min({static_cast<int>(target_sites_.size()),
                                static_cast<int>(target_reach_low_.size()),
                                static_cast<int>(target_reach_high_.size())});
    for (int i = 0; i < count; ++i) {
      const int site_id = SiteId(target_sites_[i].c_str());
      if (site_id < 0 || target_reach_low_[i].size() < 3 ||
          target_reach_high_[i].size() < 3) {
        continue;
      }
      for (int axis = 0; axis < 3; ++axis) {
        // Use the midpoint as the native deterministic reset target. Oracle
        // alignment tests sync randomized upstream state immediately after
        // reset when the official task randomizes this site.
        model_->site_pos[3 * site_id + axis] =
            (target_reach_low_[i][axis] + target_reach_high_[i][axis]) * 0.5;
      }
    }
    if (task_.kind == MyoSuiteTaskKind::kChallengeBimanual) {
      const int start = BodyId("start");
      const int goal = BodyId("goal");
      if (start >= 0) {
        model_->body_pos[3 * start] = -0.4;
        model_->body_pos[3 * start + 1] = -0.25;
        model_->body_pos[3 * start + 2] = 1.05;
      }
      if (goal >= 0) {
        model_->body_pos[3 * goal] = 0.4;
        model_->body_pos[3 * goal + 1] = -0.25;
        model_->body_pos[3 * goal + 2] = 1.05;
      }
      const int object_joint = JointId("manip_object/freejoint");
      if (object_joint >= 0) {
        const int qpos_id = model_->jnt_qposadr[object_joint];
        data_->qpos[qpos_id] = -0.4;
        data_->qpos[qpos_id + 1] = -0.25;
        data_->qpos[qpos_id + 2] = 1.15;
      }
    }
    ApplyChallengeRunTrackTerrainReset();
    mj_forward(model_, data_);
    if (task_.kind == MyoSuiteTaskKind::kChallengeBimanual) {
      bimanual_init_obj_z_ = SiteXpos(SiteId("touch_site"))[2];
      bimanual_init_palm_z_ = SiteXpos(SiteId("S_grasp"))[2];
    }
  }

  void ApplyChallengeRunTrackTerrainReset() {
    if (task_.kind != MyoSuiteTaskKind::kChallengeRunTrack) {
      return;
    }
    const int terrain = GeomId("terrain");
    if (terrain < 0) {
      return;
    }
    model_->geom_pos[3 * terrain] = 0.0;
    model_->geom_pos[3 * terrain + 1] = 0.0;
    model_->geom_pos[3 * terrain + 2] = 0.005;
    model_->geom_rgba[4 * terrain + 3] = 1.0;
  }

  void WarmstartFromCurrentAcceleration() {
    if (static_cast<int>(reset_qacc_warmstart_.size()) == model_->nv) {
      std::memcpy(data_->qacc_warmstart, reset_qacc_warmstart_.data(),
                  sizeof(mjtNum) * model_->nv);
    } else {
      std::memcpy(data_->qacc_warmstart, data_->qacc,
                  sizeof(mjtNum) * model_->nv);
    }
  }

  void InitializeTaskCaches() {
    if (task_.kind == MyoSuiteTaskKind::kMyoDmTrack) {
      const int object_bid = task_.object_name[0] != '\0'
                                 ? BodyId(task_.object_name)
                                 : BodyId("Object");
      myodm_lift_z_ = BodyXpos(object_bid)[2] + 0.02;
    }
    InitializeOslController();
    if (task_.kind == MyoSuiteTaskKind::kChallengeTableTennis) {
      tabletennis_init_paddle_quat_ = BodyXquat(BodyId("paddle"));
    }
    if (task_.kind == MyoSuiteTaskKind::kChallengeReorient) {
      const auto target = SiteXpos(SiteId("target_o"));
      const auto object = SiteXpos(SiteId("object_o"));
      for (int axis = 0; axis < 3; ++axis) {
        challenge_reorient_goal_obj_offset_[axis] = target[axis] - object[axis];
      }
    }
    if (task_.kind == MyoSuiteTaskKind::kChallengeBimanual) {
      bimanual_init_obj_z_ = SiteXpos(SiteId("touch_site"))[2];
      bimanual_init_palm_z_ = SiteXpos(SiteId("S_grasp"))[2];
    }
  }

  static std::vector<mjtNum> ReferenceRow(const double* values, int rows,
                                          int cols, int row) {
    if (rows <= 0 || cols <= 0) {
      return {};
    }
    row = std::max(0, std::min(rows - 1, row));
    std::vector<mjtNum> result(cols);
    for (int i = 0; i < cols; ++i) {
      result[i] = static_cast<mjtNum>(values[row * cols + i]);
    }
    return result;
  }

  static std::vector<mjtNum> ReferenceBlend(const double* values, int rows,
                                            int cols, int row, int next,
                                            mjtNum blend) {
    if (rows <= 0 || cols <= 0) {
      return {};
    }
    row = std::max(0, std::min(rows - 1, row));
    next = std::max(0, std::min(rows - 1, next));
    if (row == next) {
      return ReferenceRow(values, rows, cols, row);
    }
    std::vector<mjtNum> result(cols);
    for (int i = 0; i < cols; ++i) {
      const auto a = static_cast<mjtNum>(values[row * cols + i]);
      const auto b = static_cast<mjtNum>(values[next * cols + i]);
      result[i] = (1.0 - blend) * a + blend * b;
    }
    return result;
  }

  std::pair<int, int> MyoDmReferenceRows(mjtNum time) {
    if (reference_.time_size <= 1) {
      return {0, 0};
    }
    const mjtNum rounded = std::round(time * 10000.0) / 10000.0;
    const int last = reference_.time_size - 1;
    if (rounded >= static_cast<mjtNum>(reference_.time[last])) {
      myodm_reference_index_ = last;
      return {last, last};
    }
    if (myodm_reference_index_ < last &&
        rounded ==
            static_cast<mjtNum>(reference_.time[myodm_reference_index_ + 1])) {
      ++myodm_reference_index_;
      return {myodm_reference_index_, myodm_reference_index_};
    }
    if (rounded ==
        static_cast<mjtNum>(reference_.time[myodm_reference_index_])) {
      return {myodm_reference_index_, myodm_reference_index_};
    }
    const double* begin = reference_.time;
    const double* end = reference_.time + reference_.time_size;
    const auto* const upper = std::upper_bound(begin, end, rounded);
    int next = static_cast<int>(upper - begin);
    next = std::max(1, std::min(last, next));
    myodm_reference_index_ = next - 1;
    if (rounded == static_cast<mjtNum>(reference_.time[next])) {
      myodm_reference_index_ = next;
      return {next, next};
    }
    return {myodm_reference_index_, next};
  }

  MyoDmReferenceFrame MyoDmReferenceAt(mjtNum time) {
    MyoDmReferenceFrame frame;
    if (reference_.type == MyoSuiteReferenceType::kNone) {
      return frame;
    }
    if (reference_.type == MyoSuiteReferenceType::kRandom) {
      auto sample = [this](const double* values, int rows, int cols) {
        if (rows < 2 || cols <= 0) {
          return ReferenceRow(values, rows, cols, 0);
        }
        std::vector<mjtNum> result(cols);
        for (int i = 0; i < cols; ++i) {
          std::uniform_real_distribution<mjtNum> dist(
              static_cast<mjtNum>(values[i]),
              static_cast<mjtNum>(values[cols + i]));
          result[i] = dist(gen_);
        }
        return result;
      };
      frame.robot = sample(reference_.robot, reference_.robot_rows,
                           reference_.robot_cols);
      frame.robot_vel = sample(reference_.robot_vel, reference_.robot_vel_rows,
                               reference_.robot_vel_cols);
      frame.object = sample(reference_.object, reference_.object_rows,
                            reference_.object_cols);
      return frame;
    }

    const auto [row, next] = MyoDmReferenceRows(time);
    mjtNum blend = 0.0;
    if (row != next) {
      const auto t0 = static_cast<mjtNum>(reference_.time[row]);
      const auto t1 = static_cast<mjtNum>(reference_.time[next]);
      if (t1 > t0) {
        blend = (time - t0) / (t1 - t0);
      }
    }
    frame.robot = ReferenceBlend(reference_.robot, reference_.robot_rows,
                                 reference_.robot_cols, row, next, blend);
    frame.robot_vel =
        ReferenceBlend(reference_.robot_vel, reference_.robot_vel_rows,
                       reference_.robot_vel_cols, row, next, blend);
    frame.object = ReferenceBlend(reference_.object, reference_.object_rows,
                                  reference_.object_cols, row, next, blend);
    return frame;
  }

  void ApplyMyoDmReferenceSite(const MyoDmReferenceFrame& reference) {
    if (task_.kind != MyoSuiteTaskKind::kMyoDmTrack ||
        reference.object.size() < 3) {
      return;
    }
    const int target_sid = SiteId("target");
    if (target_sid < 0) {
      return;
    }
    for (int axis = 0; axis < 3; ++axis) {
      model_->site_pos[3 * target_sid + axis] = reference.object[axis];
    }
    mj_forward(model_, data_);
  }

  void PreStepTaskUpdate() {
    if (task_.kind != MyoSuiteTaskKind::kChallengeBaoding) {
      return;
    }
    const int target1 = SiteId("target1_site");
    const int target2 = SiteId("target2_site");
    if (target1 < 0 || target2 < 0) {
      return;
    }
    const mjtNum x_radius = 0.025;
    const mjtNum y_radius = 0.028;
    const mjtNum center_x = -0.0125;
    const mjtNum center_y = -0.07;
    const mjtNum phase =
        (static_cast<mjtNum>(task_step_) * Dt()) / static_cast<mjtNum>(6.0);
    const mjtNum angle1 = 2.0 * M_PI * phase + M_PI / 4.0;
    const mjtNum angle2 = angle1 - M_PI;
    model_->site_pos[3 * target1] = x_radius * std::cos(angle1) + center_x;
    model_->site_pos[3 * target1 + 1] = y_radius * std::sin(angle1) + center_y;
    model_->site_pos[3 * target2] = x_radius * std::cos(angle2) + center_x;
    model_->site_pos[3 * target2 + 1] = y_radius * std::sin(angle2) + center_y;
    mj_forward(model_, data_);
  }

  void PostStepTaskUpdate() {
    if (task_.kind == MyoSuiteTaskKind::kWalk ||
        task_.kind == MyoSuiteTaskKind::kTerrain ||
        task_.kind == MyoSuiteTaskKind::kWalkReach ||
        task_.kind == MyoSuiteTaskKind::kChallengeBaoding) {
      ++task_step_;
    }
  }

  std::vector<mjtNum> Observation(const ObsDict& obs_dict) const {
    std::vector<mjtNum> obs(task_.obs_dim, 0.0);
    std::size_t pos = 0;
    auto append = [&](mjtNum value) {
      if (pos < obs.size()) {
        obs[pos++] = value;
      }
    };
    for (const std::string& key : obs_keys_) {
      auto it = obs_dict.find(key);
      if (it == obs_dict.end()) {
        continue;
      }
      for (mjtNum value : it->second) {
        append(value);
      }
    }
    return obs;
  }

  std::vector<mjtNum> QposSlice(int begin, int end) const {
    begin = std::max(0, begin);
    end = std::min(static_cast<int>(model_->nq), end);
    if (end <= begin) {
      return {};
    }
    return {data_->qpos + begin, data_->qpos + end};
  }

  std::vector<mjtNum> QvelSlice(int begin, int end, bool scale_dt) const {
    begin = std::max(0, begin);
    end = std::min(static_cast<int>(model_->nv), end);
    std::vector<mjtNum> result;
    const mjtNum scale = scale_dt ? static_cast<mjtNum>(Dt()) : 1.0;
    for (int i = begin; i < end; ++i) {
      result.push_back(data_->qvel[i] * scale);
    }
    return result;
  }

  std::vector<mjtNum> JointQposValues(int joint_id) const {
    if (joint_id < 0) {
      return {};
    }
    const int begin = model_->jnt_qposadr[joint_id];
    return QposSlice(begin, begin + JointQposWidth(model_->jnt_type[joint_id]));
  }

  std::vector<mjtNum> JointQvelValues(int joint_id, bool scale_dt) const {
    if (joint_id < 0) {
      return {};
    }
    const int begin = model_->jnt_dofadr[joint_id];
    return QvelSlice(begin, begin + JointDofWidth(model_->jnt_type[joint_id]),
                     scale_dt);
  }

  bool IsBimanualProsthesisJoint(int joint_id) const {
    const char* name = mj_id2name(model_, mjOBJ_JOINT, joint_id);
    return name != nullptr && StartsWith(name, "prosthesis");
  }

  bool IsBimanualManipObjectJoint(int joint_id) const {
    const char* name = mj_id2name(model_, mjOBJ_JOINT, joint_id);
    return name != nullptr &&
           std::string_view(name) == "manip_object/freejoint";
  }

  std::vector<mjtNum> BimanualJointQpos(bool prosthesis) const {
    std::vector<mjtNum> result;
    for (int joint_id = 0; joint_id < model_->njnt; ++joint_id) {
      if (IsBimanualManipObjectJoint(joint_id) ||
          IsBimanualProsthesisJoint(joint_id) != prosthesis) {
        continue;
      }
      const auto values = JointQposValues(joint_id);
      result.insert(result.end(), values.begin(), values.end());
    }
    return result;
  }

  std::vector<mjtNum> BimanualJointQvel(bool prosthesis) const {
    std::vector<mjtNum> result;
    for (int joint_id = 0; joint_id < model_->njnt; ++joint_id) {
      if (IsBimanualManipObjectJoint(joint_id) ||
          IsBimanualProsthesisJoint(joint_id) != prosthesis) {
        continue;
      }
      const auto values = JointQvelValues(joint_id, false);
      result.insert(result.end(), values.begin(), values.end());
    }
    return result;
  }

  std::vector<mjtNum> AverageSites(const char* lhs, const char* rhs) const {
    const auto lhs_pos = SiteXpos(SiteId(lhs));
    const auto rhs_pos = SiteXpos(SiteId(rhs));
    return {(lhs_pos[0] + rhs_pos[0]) * 0.5, (lhs_pos[1] + rhs_pos[1]) * 0.5,
            (lhs_pos[2] + rhs_pos[2]) * 0.5};
  }

  int BimanualBodyLabel(int body_id) const {
    const int start_id = BodyId("start");
    const int goal_id = BodyId("goal");
    const int object_id = BodyId("manip_object");
    int myo_min = model_->nbody;
    int myo_max = -1;
    int prosth_min = model_->nbody;
    int prosth_max = -1;
    for (int id = 0; id < model_->nbody; ++id) {
      const char* raw_name = mj_id2name(model_, mjOBJ_BODY, id);
      const std::string_view name = raw_name == nullptr ? "" : raw_name;
      if (StartsWith(name, "prosthesis/")) {
        prosth_min = std::min(prosth_min, id);
        prosth_max = std::max(prosth_max, id);
      } else if (id != start_id && id != goal_id && id != object_id) {
        myo_min = std::min(myo_min, id);
        myo_max = std::max(myo_max, id);
      }
    }
    if (myo_min <= body_id && body_id <= myo_max) {
      return 0;
    }
    if (prosth_min <= body_id && body_id <= prosth_max) {
      return 1;
    }
    if (body_id == start_id) {
      return 2;
    }
    if (body_id == goal_id) {
      return 3;
    }
    return 4;
  }

  std::vector<mjtNum> BimanualTouchingBody() const {
    std::vector<mjtNum> result(5, 0.0);
    const int object_id = BodyId("manip_object");
    if (object_id < 0) {
      return result;
    }
    for (int i = 0; i < data_->ncon; ++i) {
      const mjContact& contact = data_->contact[i];
      const int body1 = model_->geom_bodyid[contact.geom1];
      const int body2 = model_->geom_bodyid[contact.geom2];
      if (body1 == object_id) {
        result[BimanualBodyLabel(body2)] += 1.0;
      } else if (body2 == object_id) {
        result[BimanualBodyLabel(body1)] += 1.0;
      }
    }
    return result;
  }

  std::vector<mjtNum> TableTennisTouchingInfo() const {
    std::vector<mjtNum> result(6, 0.0);
    const int ball_id = BodyId("pingpong");
    if (ball_id < 0) {
      return result;
    }
    const int paddle = GeomId("pad");
    const int own = GeomId("coll_own_half");
    const int opponent = GeomId("coll_opponent_half");
    const int net = GeomId("coll_net");
    const int ground = GeomId("ground");
    auto label = [&](int geom_id) {
      if (geom_id == paddle) {
        return 0;
      }
      if (geom_id == own) {
        return 1;
      }
      if (geom_id == opponent) {
        return 2;
      }
      if (geom_id == net) {
        return 3;
      }
      if (geom_id == ground) {
        return 4;
      }
      return 5;
    };
    for (int i = 0; i < data_->ncon; ++i) {
      const mjContact& contact = data_->contact[i];
      const int body1 = model_->geom_bodyid[contact.geom1];
      const int body2 = model_->geom_bodyid[contact.geom2];
      if (body1 == ball_id) {
        result[label(contact.geom2)] += 1.0;
      } else if (body2 == ball_id) {
        result[label(contact.geom1)] += 1.0;
      }
    }
    return result;
  }

  std::vector<mjtNum> Act() const {
    if (model_->na <= 0) {
      return {};
    }
    return {data_->act, data_->act + model_->na};
  }

  std::vector<mjtNum> ActuatorLength() const {
    return {data_->actuator_length, data_->actuator_length + model_->nu};
  }

  std::vector<mjtNum> ActuatorVelocity() const {
    std::vector<mjtNum> result(model_->nu);
    for (int i = 0; i < model_->nu; ++i) {
      result[i] = std::max<mjtNum>(
          -100.0, std::min<mjtNum>(100.0, data_->actuator_velocity[i]));
    }
    return result;
  }

  std::vector<mjtNum> ActuatorForce(bool scaled) const {
    std::vector<mjtNum> result(model_->nu);
    for (int i = 0; i < model_->nu; ++i) {
      mjtNum value = data_->actuator_force[i];
      if (scaled) {
        value =
            std::max<mjtNum>(-100.0, std::min<mjtNum>(100.0, value / 1000.0));
      }
      result[i] = value;
    }
    return result;
  }

  std::vector<mjtNum> ReachPositions(bool target) const {
    const auto& sites = target ? target_sites_ : tip_sites_;
    std::vector<mjtNum> result;
    for (const std::string& site_name : sites) {
      const auto xyz = SiteXpos(SiteId(site_name.c_str()));
      result.insert(result.end(), xyz.begin(), xyz.end());
    }
    return result;
  }

  std::vector<mjtNum> PenRotation(bool target) const {
    const bool sar = task_.kind == MyoSuiteTaskKind::kReorientSar;
    const int top = sar ? GeomId(target ? "t_top" : "top")
                        : SiteId(target ? "target_top" : "object_top");
    const int bottom = sar ? GeomId(target ? "t_bot" : "bot")
                           : SiteId(target ? "target_bottom" : "object_bottom");
    const auto top_pos = sar ? GeomXpos(top) : SiteXpos(top);
    const auto bottom_pos = sar ? GeomXpos(bottom) : SiteXpos(bottom);
    auto rot = Subtract(top_pos, bottom_pos);
    const mjtNum length = Norm(rot);
    if (length > 0.0) {
      for (mjtNum& value : rot) {
        value /= length;
      }
    }
    return rot;
  }

  std::vector<mjtNum> CenterOfMassVelocity() const {
    mjtNum mass_sum = 0.0;
    std::array<mjtNum, 6> cvel{};
    for (int body = 0; body < model_->nbody; ++body) {
      const mjtNum mass = model_->body_mass[body];
      mass_sum += mass;
      for (int axis = 0; axis < 6; ++axis) {
        cvel[axis] += mass * (-data_->cvel[6 * body + axis]);
      }
    }
    if (mass_sum > 0.0) {
      cvel[3] /= mass_sum;
      cvel[4] /= mass_sum;
    }
    return {cvel[3], cvel[4]};
  }

  std::vector<mjtNum> CenterOfMass() const {
    mjtNum mass_sum = 0.0;
    std::array<mjtNum, 3> com{};
    for (int body = 0; body < model_->nbody; ++body) {
      const mjtNum mass = model_->body_mass[body];
      mass_sum += mass;
      for (int axis = 0; axis < 3; ++axis) {
        com[axis] += mass * data_->xipos[3 * body + axis];
      }
    }
    if (mass_sum > 0.0) {
      for (mjtNum& value : com) {
        value /= mass_sum;
      }
    }
    return {com[0], com[1], com[2]};
  }

  std::vector<mjtNum> FeetRelativePositions() const {
    const auto left = BodyXpos(BodyId("talus_l"));
    const auto right = BodyXpos(BodyId("talus_r"));
    const auto pelvis = BodyXpos(BodyId("pelvis"));
    auto left_rel = Subtract(left, pelvis);
    auto right_rel = Subtract(right, pelvis);
    left_rel.insert(left_rel.end(), right_rel.begin(), right_rel.end());
    return left_rel;
  }

  std::vector<mjtNum> JointAngles(
      const std::vector<const char*>& joint_names) const {
    std::vector<mjtNum> result;
    for (const char* name : joint_names) {
      const int joint_id = JointId(name);
      if (joint_id >= 0) {
        result.push_back(data_->qpos[model_->jnt_qposadr[joint_id]]);
      }
    }
    return result;
  }

  ObsDict BuildObsDict() {
    ObsDict obs;
    MyoDmReferenceFrame myodm_reference;
    if (task_.kind == MyoSuiteTaskKind::kMyoDmTrack) {
      myodm_reference = MyoDmReferenceAt(data_->time);
      ApplyMyoDmReferenceSite(myodm_reference);
    }
    const auto dt = static_cast<mjtNum>(Dt());
    obs["time"] = {data_->time};
    obs["t"] = {data_->time};
    obs["qpos"] = QposSlice(0, model_->nq);
    obs["qp"] = obs["qpos"];
    obs["qvel"] = QvelSlice(0, model_->nv, true);
    obs["qv"] = QvelSlice(0, model_->nv, false);
    obs["act"] = Act();

    const int pose_size = !target_jnt_value_.empty()
                              ? static_cast<int>(target_jnt_value_.size())
                              : model_->nq;
    obs["pose_err"] = std::vector<mjtNum>(pose_size, 0.0);
    for (int i = 0; i < pose_size && i < model_->nq; ++i) {
      const mjtNum target = i < static_cast<int>(target_jnt_value_.size())
                                ? target_jnt_value_[i]
                                : 0.0;
      obs["pose_err"][i] = target - data_->qpos[i];
    }

    obs["tip_pos"] = ReachPositions(false);
    obs["target_pos"] = ReachPositions(true);
    obs["reach_err"] = Subtract(obs["target_pos"], obs["tip_pos"]);

    const bool key_turn = task_.kind == MyoSuiteTaskKind::kKeyTurn;
    const int hand_qpos_end = key_turn ? model_->nq - 1 : model_->nq - 7;
    const int hand_qvel_end = key_turn ? model_->nv - 1 : model_->nv - 6;
    obs["hand_qpos"] = QposSlice(0, hand_qpos_end);
    obs["hand_qpos_noMD5"] = QposSlice(0, model_->nq - 7);
    obs["hand_qpos_corrected"] = QposSlice(0, model_->nq - 6);
    obs["hand_qvel"] = QvelSlice(0, hand_qvel_end, true);
    obs["hand_jnt"] = QposSlice(0, model_->nq - 6);
    obs["hand_pos"] = QposSlice(0, model_->nq - 14);
    obs["key_qpos"] = QposSlice(model_->nq - 1, model_->nq);
    obs["key_qvel"] = QvelSlice(model_->nv - 1, model_->nv, true);
    obs["IFtip_approach"] =
        Subtract(SiteXpos(SiteId("keyhead")), SiteXpos(SiteId("IFtip")));
    obs["THtip_approach"] =
        Subtract(SiteXpos(SiteId("keyhead")), SiteXpos(SiteId("THtip")));

    if (task_.kind == MyoSuiteTaskKind::kObjHoldFixed ||
        task_.kind == MyoSuiteTaskKind::kObjHoldRandom) {
      obs["obj_pos"] = SiteXpos(SiteId("object"));
      obs["obj_err"] =
          Subtract(SiteXpos(SiteId("goal")), SiteXpos(SiteId("object")));
    } else if (task_.kind == MyoSuiteTaskKind::kPenTwirlFixed ||
               task_.kind == MyoSuiteTaskKind::kPenTwirlRandom ||
               task_.kind == MyoSuiteTaskKind::kReorientSar) {
      obs["obj_pos"] = BodyXpos(BodyId("Object"));
      obs["obj_des_pos"] = SiteXpos(SiteId("eps_ball"));
      obs["obj_vel"] = QvelSlice(model_->nv - 6, model_->nv, true);
      obs["obj_rot"] = PenRotation(false);
      obs["obj_des_rot"] = PenRotation(true);
      obs["obj_err_pos"] = Subtract(obs["obj_pos"], obs["obj_des_pos"]);
      obs["obj_err_rot"] = Subtract(obs["obj_rot"], obs["obj_des_rot"]);
      obs["mlen"] = ActuatorLength();
      obs["mvel"] = std::vector<mjtNum>(data_->actuator_velocity,
                                        data_->actuator_velocity + model_->nu);
      obs["mforce"] = std::vector<mjtNum>(data_->actuator_force,
                                          data_->actuator_force + model_->nu);
    } else if (task_.kind == MyoSuiteTaskKind::kChallengeRelocate ||
               task_.kind == MyoSuiteTaskKind::kChallengeReorient) {
      obs["obj_pos"] = SiteXpos(SiteId("object_o"));
      obs["goal_pos"] = SiteXpos(SiteId("target_o"));
      obs["palm_pos"] = SiteXpos(SiteId("S_grasp"));
      obs["pos_err"] = Subtract(obs["goal_pos"], obs["obj_pos"]);
      if (task_.kind == MyoSuiteTaskKind::kChallengeReorient) {
        for (int axis = 0; axis < 3; ++axis) {
          obs["pos_err"][axis] -= challenge_reorient_goal_obj_offset_[axis];
        }
      }
      obs["reach_err"] = Subtract(obs["palm_pos"], obs["obj_pos"]);
      obs["obj_rot"] = MatToEuler(data_->site_xmat + 9 * SiteId("object_o"));
      obs["goal_rot"] = MatToEuler(data_->site_xmat + 9 * SiteId("target_o"));
      obs["rot_err"] = Subtract(obs["goal_rot"], obs["obj_rot"]);
    }

    obs["qpos_without_xy"] = QposSlice(2, model_->nq);
    obs["com_vel"] = CenterOfMassVelocity();
    obs["torso_angle"] = BodyXquat(BodyId("torso"));
    obs["feet_heights"] = {BodyXpos(BodyId("talus_l"))[2],
                           BodyXpos(BodyId("talus_r"))[2]};
    obs["height"] = {CenterOfMass()[2]};
    obs["feet_rel_positions"] = FeetRelativePositions();
    const int hip_period =
        metadata_.hip_period > 0 ? metadata_.hip_period : 100;
    obs["phase_var"] = {std::fmod(
        static_cast<mjtNum>(task_step_) / static_cast<mjtNum>(hip_period),
        1.0)};
    obs["muscle_length"] = ActuatorLength();
    obs["muscle_velocity"] = ActuatorVelocity();
    obs["muscle_force"] = ActuatorForce(true);

    obs["object1_pos"] = SiteXpos(SiteId("ball1_site"));
    obs["object2_pos"] = SiteXpos(SiteId("ball2_site"));
    obs["object1_velp"] = QvelSlice(model_->nv - 12, model_->nv - 9, true);
    obs["object2_velp"] = QvelSlice(model_->nv - 6, model_->nv - 3, true);
    obs["target1_pos"] = SiteXpos(SiteId("target1_site"));
    obs["target2_pos"] = SiteXpos(SiteId("target2_site"));
    obs["target1_err"] = Subtract(obs["target1_pos"], obs["object1_pos"]);
    obs["target2_err"] = Subtract(obs["target2_pos"], obs["object2_pos"]);

    obs["internal_qpos"] = QposSlice(0, model_->nq);
    obs["internal_qvel"] = QvelSlice(0, model_->nv, true);
    obs["grf"] = SensorData("r_foot");
    auto lfoot = SensorData("l_foot");
    obs["grf"].insert(obs["grf"].end(), lfoot.begin(), lfoot.end());
    obs["model_root_pos"] =
        QposSlice(0, std::min(7, static_cast<int>(model_->nq)));
    obs["model_root_vel"] =
        QvelSlice(0, std::min(6, static_cast<int>(model_->nv)), false);
    if (model_->nmocap > 0) {
      obs["opponent_pose"] = {data_->mocap_pos[0], data_->mocap_pos[1],
                              YawFromQuat(data_->mocap_quat)};
    } else {
      const auto opponent = BodyXpos(BodyId("opponent"));
      obs["opponent_pose"] = {opponent[0], opponent[1], 0.0};
    }
    obs["opponent_vel"] = {0.0, 0.0};

    obs["pelvis_pos"] = SiteXpos(SiteId("pelvis"));
    obs["body_qpos"] =
        QposSlice(0, std::max(0, static_cast<int>(model_->nq) - 7));
    obs["body_qvel"] =
        QvelSlice(0, std::max(0, static_cast<int>(model_->nv) - 6), true);
    if (task_.kind == MyoSuiteTaskKind::kChallengeSoccer) {
      obs["ball_pos"] = BodyXpos(BodyId("soccer_ball"));
    }
    if (task_.kind == MyoSuiteTaskKind::kChallengeTableTennis) {
      obs["ball_pos"] = SiteXpos(SiteId("pingpong"));
      obs["ball_vel"] = SensorData("pingpong_vel_sensor");
      obs["paddle_pos"] = SiteXpos(SiteId("paddle"));
      obs["paddle_vel"] = SensorData("paddle_vel_sensor");
      obs["paddle_ori"] = BodyXquat(BodyId("paddle"));
      obs["padde_ori_err"] =
          Subtract(obs["paddle_ori"], tabletennis_init_paddle_quat_);
      obs["reach_err"] = Subtract(obs["paddle_pos"], obs["ball_pos"]);
      obs["palm_pos"] = SiteXpos(SiteId("S_grasp"));
      obs["palm_err"] = Subtract(obs["palm_pos"], obs["paddle_pos"]);
      obs["touching_info"] = TableTennisTouchingInfo();
    }

    if (task_.kind == MyoSuiteTaskKind::kChallengeBimanual) {
      obs["myohand_qpos"] = BimanualJointQpos(false);
      obs["myohand_qvel"] = BimanualJointQvel(false);
      obs["pros_hand_qpos"] = BimanualJointQpos(true);
      obs["pros_hand_qvel"] = BimanualJointQvel(true);
      const int object_joint = JointId("manip_object/freejoint");
      obs["object_qpos"] = JointQposValues(object_joint);
      obs["object_qvel"] = JointQvelValues(object_joint, false);
      obs["touching_body"] = BimanualTouchingBody();
      obs["palm_pos"] = SiteXpos(SiteId("S_grasp"));
      obs["fin0"] = SiteXpos(SiteId("THtip"));
      obs["fin1"] = SiteXpos(SiteId("IFtip"));
      obs["fin2"] = SiteXpos(SiteId("MFtip"));
      obs["fin3"] = SiteXpos(SiteId("RFtip"));
      obs["fin4"] = SiteXpos(SiteId("LFtip"));
      obs["Rpalm_pos"] =
          AverageSites("prosthesis/palm_thumb", "prosthesis/palm_pinky");
      obs["obj_pos"] = SiteXpos(SiteId("touch_site"));
      obs["start_pos"] = BodyPos(BodyId("start"));
      obs["goal_pos"] = BodyPos(BodyId("goal"));
      obs["reach_err"] = Subtract(obs["palm_pos"], obs["obj_pos"]);
      obs["pass_err"] = Subtract(obs["Rpalm_pos"], obs["obj_pos"]);
      const int elbow = JointId("elbow_flexion");
      obs["elbow_fle"] = JointQposValues(elbow);
    }

    if (task_.kind == MyoSuiteTaskKind::kMyoDmTrack) {
      obs["curr_hand_qpos"] =
          QposSlice(0, std::max(0, static_cast<int>(model_->nq) - 6));
      obs["curr_hand_qvel"] =
          QvelSlice(0, std::max(0, static_cast<int>(model_->nv) - 6), false);
      obs["targ_hand_qpos"] =
          myodm_reference.robot.empty()
              ? std::vector<mjtNum>(obs["curr_hand_qpos"].size(), 0.0)
              : myodm_reference.robot;
      obs["targ_hand_qvel"] = myodm_reference.robot_vel.empty()
                                  ? std::vector<mjtNum>{0.0}
                                  : myodm_reference.robot_vel;
      obs["hand_qpos_err"] =
          Subtract(obs["curr_hand_qpos"], obs["targ_hand_qpos"]);
      obs["hand_qvel_err"] =
          myodm_reference.robot_vel.empty()
              ? std::vector<mjtNum>{0.0}
              : Subtract(obs["curr_hand_qvel"], obs["targ_hand_qvel"]);
      const int object_bid = task_.object_name[0] != '\0'
                                 ? BodyId(task_.object_name)
                                 : BodyId("Object");
      obs["curr_obj_com"] = BodyXpos(object_bid);
      obs["curr_obj_rot"] = BodyXquat(object_bid);
      if (myodm_reference.object.size() >= 7) {
        obs["targ_obj_com"] = {myodm_reference.object[0],
                               myodm_reference.object[1],
                               myodm_reference.object[2]};
        obs["targ_obj_rot"] = {
            myodm_reference.object[3], myodm_reference.object[4],
            myodm_reference.object[5], myodm_reference.object[6]};
      } else {
        obs["targ_obj_com"] = {0.2, 0.2, 0.1};
        obs["targ_obj_rot"] = {1.0, 0.0, 0.0, 0.0};
      }
      obs["obj_com_err"] = Subtract(obs["curr_obj_com"], obs["targ_obj_com"]);
      obs["wrist_err"] = BodyXpos(BodyId("lunate"));
      obs["base_error"] = Subtract(obs["curr_obj_com"], obs["wrist_err"]);
    }
    (void)dt;
    return obs;
  }

  const std::vector<mjtNum>& ObsValue(const ObsDict& obs,
                                      const std::string& key) const {
    static const std::vector<mjtNum> k_empty;
    auto it = obs.find(key);
    return it == obs.end() ? k_empty : it->second;
  }

  mjtNum ActMagnitude(const ObsDict& obs) const {
    const auto& act = ObsValue(obs, "act");
    if (act.empty() || model_->na == 0) {
      return 0.0;
    }
    return Norm(act) / static_cast<mjtNum>(model_->na);
  }

  mjtNum ComponentValue(
      const std::string& key,
      const std::unordered_map<std::string, mjtNum>& values) const {
    auto it = values.find(key);
    return it == values.end() ? 0.0 : it->second;
  }

  bool WalkDone(const ObsDict& obs) const {
    const auto& height = ObsValue(obs, "height");
    const mjtNum min_height =
        metadata_.min_height > 0.0 ? metadata_.min_height : 0.8;
    if (!height.empty() && height[0] < min_height) {
      return true;
    }
    const mjtNum max_rot = metadata_.max_rot > 0.0 ? metadata_.max_rot : 0.8;
    const auto quat = QposSlice(3, 7);
    if (quat.size() == 4) {
      std::array<mjtNum, 9> mat{};
      mju_quat2Mat(mat.data(), quat.data());
      if (std::abs(mat[0]) > max_rot) {
        return true;
      }
    }
    if (task_.kind == MyoSuiteTaskKind::kTerrain) {
      const auto& feet = ObsValue(obs, "feet_heights");
      if (!height.empty() && feet.size() >= 2 &&
          height[0] - (feet[0] + feet[1]) * 0.5 < 0.61) {
        return true;
      }
    }
    return false;
  }

  RewardResult ComputeReward(const ObsDict& obs) {
    std::unordered_map<std::string, mjtNum> values;
    bool terminated = false;
    const mjtNum pi = std::acos(static_cast<mjtNum>(-1.0));

    if (task_.kind == MyoSuiteTaskKind::kMyoDmTrack) {
      const mjtNum obj_com_err = Norm(ObsValue(obs, "obj_com_err"));
      const mjtNum obj_rot_err =
          QuaternionDistance(ObsValue(obs, "curr_obj_rot"),
                             ObsValue(obs, "targ_obj_rot")) /
          pi;
      const mjtNum obj_reward =
          std::exp(-50.0 * (obj_com_err + 0.1 * obj_rot_err));
      const auto& targ_obj_com = ObsValue(obs, "targ_obj_com");
      const auto& curr_obj_com = ObsValue(obs, "curr_obj_com");
      const bool lift_bonus =
          targ_obj_com.size() > 2 && curr_obj_com.size() > 2 &&
          targ_obj_com[2] >= myodm_lift_z_ && curr_obj_com[2] >= myodm_lift_z_;
      const mjtNum qpos_reward =
          std::exp(-5.0 * SquaredNorm(ObsValue(obs, "hand_qpos_err")));
      const mjtNum qvel_reward =
          std::exp(-0.1 * SquaredNorm(ObsValue(obs, "hand_qvel_err")));
      const mjtNum pose_reward = 0.35 * qpos_reward + 0.05 * qvel_reward;
      const mjtNum base_error = Norm(ObsValue(obs, "base_error"));
      const mjtNum base_reward = std::exp(-40.0 * base_error);
      const bool myodm_done =
          SquaredNorm(ObsValue(obs, "obj_com_err")) >= 0.25 * 0.25 ||
          SquaredNorm(ObsValue(obs, "base_error")) >= 0.25 * 0.25;
      values["pose"] = pose_reward;
      values["object"] = obj_reward + base_reward;
      values["bonus"] = lift_bonus ? 1.0 : 0.0;
      values["penalty"] = myodm_done ? 1.0 : 0.0;
      values["sparse"] = 0.0;
      values["solved"] = 0.0;
      values["done"] = myodm_done ? 1.0 : 0.0;
      terminated = myodm_done;
    } else if (task_.kind == MyoSuiteTaskKind::kPose ||
               task_.kind == MyoSuiteTaskKind::kTorsoPose) {
      const mjtNum pose_dist = Norm(ObsValue(obs, "pose_err"));
      const mjtNum pose_thd =
          metadata_.pose_thd > 0.0 ? metadata_.pose_thd : 0.35;
      const mjtNum far_th =
          task_.kind == MyoSuiteTaskKind::kTorsoPose ? pi : 2.0 * pi;
      values["pose"] = -pose_dist;
      values["bonus"] = (pose_dist < pose_thd ? 1.0 : 0.0) +
                        (pose_dist < 1.5 * pose_thd ? 1.0 : 0.0);
      values["penalty"] = pose_dist > far_th ? -1.0 : 0.0;
      values["act_reg"] = -ActMagnitude(obs);
      values["sparse"] = -pose_dist;
      values["solved"] = pose_dist < pose_thd ? 1.0 : 0.0;
      terminated = pose_dist > far_th;
    } else if (task_.kind == MyoSuiteTaskKind::kReach ||
               task_.kind == MyoSuiteTaskKind::kWalkReach) {
      const mjtNum reach_dist = Norm(ObsValue(obs, "reach_err"));
      const mjtNum vel_dist = Norm(ObsValue(obs, "qvel"));
      const mjtNum nsites =
          std::max<mjtNum>(1.0, static_cast<mjtNum>(tip_sites_.size()));
      const mjtNum far_base = metadata_.far_th > 0.0 ? metadata_.far_th : 0.35;
      const mjtNum far_th = data_->time > 2.0 * Dt()
                                ? far_base * nsites
                                : std::numeric_limits<mjtNum>::infinity();
      const mjtNum near_th =
          (task_.kind == MyoSuiteTaskKind::kWalkReach ? 0.050 : 0.0125) *
          nsites;
      values["reach"] = task_.kind == MyoSuiteTaskKind::kWalkReach
                            ? 10.0 - reach_dist - 10.0 * vel_dist
                            : -reach_dist;
      values["bonus"] = (reach_dist < 2.0 * near_th ? 1.0 : 0.0) +
                        (reach_dist < near_th ? 1.0 : 0.0);
      values["act_reg"] = task_.kind == MyoSuiteTaskKind::kWalkReach
                              ? -100.0 * ActMagnitude(obs)
                              : -ActMagnitude(obs);
      values["penalty"] = reach_dist > far_th ? -1.0 : 0.0;
      values["sparse"] = -reach_dist;
      values["solved"] = reach_dist < near_th ? 1.0 : 0.0;
      terminated = reach_dist > far_th;
    } else if (task_.kind == MyoSuiteTaskKind::kKeyTurn) {
      const mjtNum if_dist =
          std::abs(Norm(ObsValue(obs, "IFtip_approach")) - 0.030);
      const mjtNum th_dist =
          std::abs(Norm(ObsValue(obs, "THtip_approach")) - 0.030);
      const auto& key_qpos = ObsValue(obs, "key_qpos");
      const mjtNum key_pos = key_qpos.empty() ? 0.0 : key_qpos[0];
      const mjtNum far_th = 0.1;
      const mjtNum goal_th = metadata_.goal_th > 0.0 ? metadata_.goal_th : 3.14;
      values["key_turn"] = key_pos;
      values["IFtip_approach"] = -if_dist;
      values["THtip_approach"] = -th_dist;
      values["act_reg"] = -ActMagnitude(obs);
      values["bonus"] =
          (key_pos > pi / 2.0 ? 1.0 : 0.0) + (key_pos > pi ? 1.0 : 0.0);
      values["penalty"] = (if_dist > far_th / 2.0 ? -1.0 : 0.0) +
                          (th_dist > far_th / 2.0 ? -1.0 : 0.0);
      values["sparse"] = key_pos;
      values["solved"] = key_pos > goal_th ? 1.0 : 0.0;
      terminated = if_dist > far_th || th_dist > far_th;
    } else if (task_.kind == MyoSuiteTaskKind::kObjHoldFixed ||
               task_.kind == MyoSuiteTaskKind::kObjHoldRandom) {
      const mjtNum goal_dist = Norm(ObsValue(obs, "obj_err"));
      const mjtNum goal_th = 0.010;
      const bool drop = goal_dist > 0.300;
      values["goal_dist"] = -goal_dist;
      values["bonus"] = (goal_dist < 2.0 * goal_th ? 1.0 : 0.0) +
                        (goal_dist < goal_th ? 1.0 : 0.0);
      values["act_reg"] = -ActMagnitude(obs);
      values["penalty"] = drop ? -1.0 : 0.0;
      values["sparse"] = -goal_dist;
      values["solved"] = goal_dist < goal_th ? 1.0 : 0.0;
      terminated = drop;
    } else if (task_.kind == MyoSuiteTaskKind::kPenTwirlFixed ||
               task_.kind == MyoSuiteTaskKind::kPenTwirlRandom ||
               task_.kind == MyoSuiteTaskKind::kReorientSar) {
      const mjtNum pos_align = Norm(ObsValue(obs, "obj_err_pos"));
      const mjtNum rot_align =
          Cosine(ObsValue(obs, "obj_rot"), ObsValue(obs, "obj_des_rot"));
      const bool dropped = pos_align > 0.075;
      values["pos_align"] = -pos_align;
      values["rot_align"] = rot_align;
      values["act_reg"] = -ActMagnitude(obs);
      values["drop"] = dropped ? -1.0 : 0.0;
      values["bonus"] = (rot_align > 0.9 && pos_align < 0.075 ? 1.0 : 0.0) +
                        (rot_align > 0.95 && pos_align < 0.075 ? 5.0 : 0.0);
      values["sparse"] = -pos_align + rot_align;
      const bool solved = rot_align > 0.95 && !dropped;
      values["solved"] = solved ? 1.0 : 0.0;
      if (task_.kind == MyoSuiteTaskKind::kReorientSar) {
        const int indicator = SiteId("success");
        if (indicator >= 0 && (model_->site_rgba[4 * indicator] != 0.0 ||
                               model_->site_rgba[4 * indicator + 1] != 2.0)) {
          model_->site_rgba[4 * indicator] = solved ? 0.0 : 2.0;
          model_->site_rgba[4 * indicator + 1] = solved ? 2.0 : 0.0;
        }
      }
      terminated = dropped;
    } else if (task_.kind == MyoSuiteTaskKind::kWalk ||
               task_.kind == MyoSuiteTaskKind::kTerrain) {
      const mjtNum target_x = metadata_.target_x_vel;
      const mjtNum target_y =
          metadata_.target_y_vel != 0.0 ? metadata_.target_y_vel : 1.2;
      const auto& com_vel = ObsValue(obs, "com_vel");
      const mjtNum vx = com_vel.size() > 0 ? com_vel[0] : 0.0;
      const mjtNum vy = com_vel.size() > 1 ? com_vel[1] : 0.0;
      const mjtNum vel_reward = std::exp(-std::pow(target_y - vy, 2)) +
                                std::exp(-std::pow(target_x - vx, 2));
      const int hip_period =
          metadata_.hip_period > 0 ? metadata_.hip_period : 100;
      const mjtNum phase =
          std::fmod(static_cast<mjtNum>(task_step_) / hip_period, 1.0);
      const std::vector<mjtNum> des_angles = {
          0.8 * std::cos(phase * 2.0 * pi + pi),
          0.8 * std::cos(phase * 2.0 * pi)};
      const mjtNum cyclic_hip = Norm(Subtract(
          des_angles, JointAngles({"hip_flexion_l", "hip_flexion_r"})));
      const mjtNum ref_rot = 1.0;
      const auto joint_angles =
          JointAngles({"hip_adduction_l", "hip_adduction_r", "hip_rotation_l",
                       "hip_rotation_r"});
      mjtNum joint_mag = 0.0;
      for (mjtNum angle : joint_angles) {
        joint_mag += std::abs(angle);
      }
      if (!joint_angles.empty()) {
        joint_mag /= static_cast<mjtNum>(joint_angles.size());
      }
      const mjtNum joint_angle_rew = std::exp(-5.0 * joint_mag);
      const bool done = WalkDone(obs);
      values["vel_reward"] = vel_reward;
      values["cyclic_hip"] = cyclic_hip;
      values["ref_rot"] = ref_rot;
      values["joint_angle_rew"] = joint_angle_rew;
      values["act_mag"] = ActMagnitude(obs);
      values["sparse"] = vel_reward;
      values["solved"] = vel_reward >= 1.0 ? 1.0 : 0.0;
      values["done"] = done ? 1.0 : 0.0;
      terminated = done;
    } else if (task_.kind == MyoSuiteTaskKind::kChallengeBaoding) {
      const mjtNum d1 = Norm(ObsValue(obs, "target1_err"));
      const mjtNum d2 = Norm(ObsValue(obs, "target2_err"));
      const auto& object1 = ObsValue(obs, "object1_pos");
      const auto& object2 = ObsValue(obs, "object2_pos");
      const bool fall = (object1.size() > 2 && object1[2] < 1.25) ||
                        (object2.size() > 2 && object2[2] < 1.25);
      values["pos_dist_1"] = -d1;
      values["pos_dist_2"] = -d2;
      values["act_reg"] = -ActMagnitude(obs);
      values["sparse"] = -(d1 + d2);
      values["solved"] = d1 < 0.015 && d2 < 0.015 && !fall ? 1.0 : 0.0;
      const int object1_gid = GeomId("ball1");
      const int object2_gid = GeomId("ball2");
      if (object1_gid >= 0) {
        model_->geom_rgba[4 * object1_gid] =
            d1 < 0.015 ? static_cast<mjtNum>(1.0) : static_cast<mjtNum>(0.5);
        model_->geom_rgba[4 * object1_gid + 1] =
            d1 < 0.015 ? static_cast<mjtNum>(1.0) : static_cast<mjtNum>(0.5);
      }
      if (object2_gid >= 0) {
        model_->geom_rgba[4 * object2_gid] =
            d1 < 0.015 ? static_cast<mjtNum>(0.9) : static_cast<mjtNum>(0.5);
        model_->geom_rgba[4 * object2_gid + 1] =
            d1 < 0.015 ? static_cast<mjtNum>(0.7) : static_cast<mjtNum>(0.5);
      }
      terminated = fall;
    } else if (task_.kind == MyoSuiteTaskKind::kChallengeRelocate ||
               task_.kind == MyoSuiteTaskKind::kChallengeReorient) {
      const mjtNum pos_dist = Norm(ObsValue(obs, "pos_err"));
      const mjtNum rot_dist = Norm(ObsValue(obs, "rot_err"));
      const bool drop = task_.kind == MyoSuiteTaskKind::kChallengeRelocate
                            ? Norm(ObsValue(obs, "reach_err")) > 0.50
                            : pos_dist > 0.200;
      values["pos_dist"] = -pos_dist;
      values["rot_dist"] = -rot_dist;
      values["bonus"] =
          (pos_dist < 0.05 ? 1.0 : 0.0) + (pos_dist < 0.025 ? 1.0 : 0.0);
      values["act_reg"] = -ActMagnitude(obs);
      values["penalty"] = drop ? -1.0 : 0.0;
      values["sparse"] = -rot_dist - 10.0 * pos_dist;
      const bool solved = pos_dist < 0.025 && rot_dist < 0.262 && !drop;
      values["solved"] = solved ? 1.0 : 0.0;
      const int indicator = SiteId("target_ball");
      if (indicator >= 0) {
        model_->site_rgba[4 * indicator] = solved ? 0.0 : 2.0;
        model_->site_rgba[4 * indicator + 1] = solved ? 2.0 : 0.0;
        if (task_.kind == MyoSuiteTaskKind::kChallengeRelocate) {
          for (int axis = 0; axis < 3; ++axis) {
            model_->site_size[3 * indicator + axis] = solved ? 0.25 : 0.1;
          }
        }
      }
      terminated = drop;
    } else if (task_.kind == MyoSuiteTaskKind::kChallengeRunTrack) {
      const auto& root_pos = ObsValue(obs, "model_root_pos");
      const auto& root_vel = ObsValue(obs, "model_root_vel");
      const mjtNum x = root_pos.size() > 0 ? root_pos[0] : 0.0;
      const mjtNum y = root_pos.size() > 1 ? root_pos[1] : 0.0;
      const mjtNum y_vel = root_vel.size() > 1 ? root_vel[1] : 0.0;
      const bool random_track =
          std::string_view(task_.id).find("Random") != std::string_view::npos;
      const mjtNum start_pos = random_track ? 58.0 : 14.0;
      const mjtNum end_pos = random_track ? -45.0 : -15.0;
      const bool fallen = WalkDone(obs);
      const bool win = y < end_pos;
      const bool lose = x > 1.0 || x < -1.0 || y > start_pos + 2.0 || fallen;
      values["act_reg"] = MeanSquare(ObsValue(obs, "act"));
      values["pain"] = 0.0;
      values["sparse"] = -y_vel;
      values["solved"] = win ? 1.0 : 0.0;
      values["done"] = (win || lose) ? 1.0 : 0.0;
      terminated = win || lose;
    } else if (task_.kind == MyoSuiteTaskKind::kChallengeChaseTag) {
      const auto& root_pos = ObsValue(obs, "model_root_pos");
      const auto& opponent_pose = ObsValue(obs, "opponent_pose");
      const mjtNum dx = (root_pos.size() > 0 ? root_pos[0] : 0.0) -
                        (opponent_pose.size() > 0 ? opponent_pose[0] : 0.0);
      const mjtNum dy = (root_pos.size() > 1 ? root_pos[1] : 0.0) -
                        (opponent_pose.size() > 1 ? opponent_pose[1] : 0.0);
      const mjtNum distance = std::sqrt(dx * dx + dy * dy);
      const bool tagged = distance <= 0.5;
      const auto pelvis = BodyXpos(BodyId("pelvis"));
      const bool out_of_bounds =
          std::abs(pelvis[0]) > 6.5 || std::abs(pelvis[1]) > 6.5;
      const bool fallen = pelvis[2] < 0.5;
      const bool win = tagged;
      const bool lose = data_->time >= 20.0 || out_of_bounds || fallen;
      values["act_reg"] = ActMagnitude(obs);
      values["distance"] = distance;
      values["lose"] = lose ? 1.0 : 0.0;
      values["sparse"] =
          win ? 1.0 - std::round(data_->time * 100.0) / 100.0 / 20.0 : 0.0;
      values["solved"] = win ? 1.0 : 0.0;
      values["done"] = (win || lose) ? 1.0 : 0.0;
      const int indicator = SiteId("opponent_indicator");
      if (indicator >= 0) {
        model_->site_rgba[4 * indicator] = win ? 0.0 : 2.0;
        model_->site_rgba[4 * indicator + 1] = win ? 2.0 : 0.0;
        model_->site_rgba[4 * indicator + 2] = 0.0;
        model_->site_rgba[4 * indicator + 3] = win ? 0.2 : 0.0;
      }
      terminated = win || lose;
    } else if (task_.kind == MyoSuiteTaskKind::kChallengeTableTennis) {
      const mjtNum reach_dist = Norm(ObsValue(obs, "reach_err"));
      const mjtNum palm_dist = Norm(ObsValue(obs, "palm_err"));
      const mjtNum paddle_quat_err = Norm(ObsValue(obs, "padde_ori_err"));
      const int torso_joint = JointId("flex_extension");
      const mjtNum torso_err =
          torso_joint >= 0
              ? std::abs(data_->qpos[model_->jnt_qposadr[torso_joint]])
              : 0.0;
      const auto& ball_pos = ObsValue(obs, "ball_pos");
      const auto& touching = ObsValue(obs, "touching_info");
      const bool paddle_touch = !touching.empty() && touching[0] == 1.0;
      const bool ball_done =
          data_->time > 20.0 || (ball_pos.size() > 2 && ball_pos[2] < 0.3);
      values["reach_dist"] = std::exp(-reach_dist);
      values["palm_dist"] = std::exp(-5.0 * palm_dist);
      values["paddle_quat"] = std::exp(-5.0 * paddle_quat_err);
      values["torso_up"] = std::exp(-5.0 * torso_err);
      values["act_reg"] = -ActMagnitude(obs);
      values["sparse"] = paddle_touch ? 1.0 : 0.0;
      values["solved"] = 0.0;
      values["done"] = ball_done ? 1.0 : 0.0;
      terminated = ball_done;
    } else if (task_.kind == MyoSuiteTaskKind::kChallengeBimanual) {
      const mjtNum reach_dist = Norm(ObsValue(obs, "reach_err"));
      const mjtNum pass_dist = Norm(ObsValue(obs, "pass_err"));
      const auto& obj_pos = ObsValue(obs, "obj_pos");
      const auto& palm_pos = ObsValue(obs, "palm_pos");
      auto goal_pos = ObsValue(obs, "goal_pos");
      if (goal_pos.size() >= 3) {
        goal_pos[2] = 1.09;
      }
      mjtNum lift_height = 0.0;
      if (obj_pos.size() >= 3 && palm_pos.size() >= 3) {
        const mjtNum obj_lift = obj_pos[2] - bimanual_init_obj_z_;
        const mjtNum palm_lift = palm_pos[2] - bimanual_init_palm_z_;
        lift_height =
            5.0 * std::exp(-10.0 * ((obj_lift - 0.2) * (obj_lift - 0.2) +
                                    (palm_lift - 0.2) * (palm_lift - 0.2))) -
            5.0;
      }
      mjtNum fin_open = 0.0;
      mjtNum fin_dis = 0.0;
      for (const char* key : {"fin0", "fin1", "fin2", "fin3", "fin4"}) {
        fin_open += Norm(Subtract(ObsValue(obs, key), palm_pos));
        fin_dis += Norm(Subtract(ObsValue(obs, key), obj_pos));
      }
      const auto& elbow = ObsValue(obs, "elbow_fle");
      const mjtNum elbow_value = elbow.empty() ? 0.0 : elbow[0];
      const mjtNum elbow_err =
          5.0 * std::exp(-10.0 * (elbow_value - 1.0) * (elbow_value - 1.0)) -
          5.0;
      const mjtNum goal_dist = Norm(Subtract(obj_pos, goal_pos));
      const auto& touching = ObsValue(obs, "touching_body");
      if (touching.size() > 3 && touching[3] == 1.0) {
        ++bimanual_goal_touch_;
      }
      const bool solved = goal_dist < 0.17 && bimanual_goal_touch_ >= 10;
      const bool done = data_->time > 10.0 ||
                        (obj_pos.size() > 2 && obj_pos[2] < 0.3) || solved;
      values["reach_dist"] = reach_dist + std::log(reach_dist + 1e-6);
      values["pass_err"] = pass_dist + std::log(pass_dist + 1e-3);
      values["act"] = ActMagnitude(obs);
      values["fin_open"] = std::exp(-5.0 * fin_open);
      values["fin_dis"] = fin_dis + std::log(fin_dis + 1e-6);
      values["lift_bonus"] = elbow_err;
      values["lift_height"] = lift_height;
      values["goal_dist"] = goal_dist;
      values["sparse"] = 0.0;
      values["solved"] = solved ? 1.0 : 0.0;
      values["done"] = done ? 1.0 : 0.0;
      terminated = done;
    } else if (task_.kind == MyoSuiteTaskKind::kChallengeSoccer) {
      const auto& root_pos = ObsValue(obs, "model_root_pos");
      const auto& ball_pos = ObsValue(obs, "ball_pos");
      std::vector<mjtNum> root_xyz(3, 0.0);
      for (std::size_t i = 0; i < root_xyz.size() && i < root_pos.size(); ++i) {
        root_xyz[i] = root_pos[i];
      }
      const mjtNum distance = Norm(Subtract(root_xyz, ball_pos));
      const bool goal_scored = ball_pos.size() >= 3 && ball_pos[0] >= 50.0 &&
                               ball_pos[1] >= -3.3 && ball_pos[1] <= 3.3 &&
                               ball_pos[2] >= 0.0 && ball_pos[2] <= 2.2;
      const auto pelvis = BodyXpos(BodyId("pelvis"));
      const bool fallen = pelvis[2] < 0.2;
      const bool done = goal_scored || data_->time >= 10.0 || fallen;
      values["goal_scored"] = goal_scored ? 1.0 : 0.0;
      values["time_cost"] = data_->time;
      values["act_reg"] = ActMagnitude(obs);
      values["pain"] = 0.0;
      values["distance"] = distance;
      values["sparse"] = done ? 1.0 : 0.0;
      values["solved"] = goal_scored ? 1.0 : 0.0;
      values["done"] = done ? 1.0 : 0.0;
      terminated = done;
    } else {
      throw std::runtime_error("Unhandled MyoSuite reward task kind: " +
                               std::string(task_.id));
    }

    RewardResult result;
    for (const auto& [key, weight] : reward_weights_) {
      result.dense += weight * ComponentValue(key, values);
    }
    result.sparse = ComponentValue("sparse", values);
    result.solved = ComponentValue("solved", values);
    result.terminated = terminated || ComponentValue("done", values) != 0.0;
    return result;
  }

  void CapturePaddedResetState() {
    CaptureResetState();
#ifdef ENVPOOL_TEST
    std::fill(qpos0_pad_.begin(), qpos0_pad_.end(), 0.0);
    std::fill(qvel0_pad_.begin(), qvel0_pad_.end(), 0.0);
    std::fill(act0_pad_.begin(), act0_pad_.end(), 0.0);
    std::fill(qacc0_pad_.begin(), qacc0_pad_.end(), 0.0);
    std::fill(qacc_warmstart0_pad_.begin(), qacc_warmstart0_pad_.end(), 0.0);
    for (int i = 0; i < model_->nq && i < 2048; ++i) {
      qpos0_pad_[i] = data_->qpos[i];
    }
    for (int i = 0; i < model_->nv && i < 2048; ++i) {
      qvel0_pad_[i] = data_->qvel[i];
    }
    for (int i = 0; i < model_->na && i < 2048; ++i) {
      act0_pad_[i] = data_->act[i];
    }
    for (int i = 0; i < model_->nv && i < 2048; ++i) {
      qacc0_pad_[i] = data_->qacc[i];
      qacc_warmstart0_pad_[i] = data_->qacc_warmstart[i];
    }
#endif
  }

  void CapturePaddedCurrentState() {
#ifdef ENVPOOL_TEST
    std::fill(qpos_pad_.begin(), qpos_pad_.end(), 0.0);
    std::fill(qvel_pad_.begin(), qvel_pad_.end(), 0.0);
    std::fill(act_pad_.begin(), act_pad_.end(), 0.0);
    std::fill(ctrl_pad_.begin(), ctrl_pad_.end(), 0.0);
    std::fill(qacc_pad_.begin(), qacc_pad_.end(), 0.0);
    std::fill(qacc_warmstart_pad_.begin(), qacc_warmstart_pad_.end(), 0.0);
    std::fill(actuator_length_pad_.begin(), actuator_length_pad_.end(), 0.0);
    std::fill(actuator_velocity_pad_.begin(), actuator_velocity_pad_.end(),
              0.0);
    std::fill(actuator_force_pad_.begin(), actuator_force_pad_.end(), 0.0);
    std::fill(fatigue_ma_pad_.begin(), fatigue_ma_pad_.end(), 0.0);
    std::fill(fatigue_mr_pad_.begin(), fatigue_mr_pad_.end(), 0.0);
    std::fill(fatigue_mf_pad_.begin(), fatigue_mf_pad_.end(), 0.0);
    std::fill(fatigue_tl_pad_.begin(), fatigue_tl_pad_.end(), 0.0);
    std::fill(fatigue_tauact_pad_.begin(), fatigue_tauact_pad_.end(), 0.0);
    std::fill(fatigue_taudeact_pad_.begin(), fatigue_taudeact_pad_.end(), 0.0);
    std::fill(site_pos_pad_.begin(), site_pos_pad_.end(), 0.0);
    std::fill(site_quat_pad_.begin(), site_quat_pad_.end(), 0.0);
    std::fill(site_xpos_pad_.begin(), site_xpos_pad_.end(), 0.0);
    std::fill(site_size_pad_.begin(), site_size_pad_.end(), 0.0);
    std::fill(site_rgba_pad_.begin(), site_rgba_pad_.end(), 0.0);
    std::fill(body_pos_pad_.begin(), body_pos_pad_.end(), 0.0);
    std::fill(body_quat_pad_.begin(), body_quat_pad_.end(), 0.0);
    std::fill(body_mass_pad_.begin(), body_mass_pad_.end(), 0.0);
    std::fill(light_xpos_pad_.begin(), light_xpos_pad_.end(), 0.0);
    std::fill(light_xdir_pad_.begin(), light_xdir_pad_.end(), 0.0);
    std::fill(geom_pos_pad_.begin(), geom_pos_pad_.end(), 0.0);
    std::fill(geom_quat_pad_.begin(), geom_quat_pad_.end(), 0.0);
    std::fill(geom_size_pad_.begin(), geom_size_pad_.end(), 0.0);
    std::fill(geom_xpos_pad_.begin(), geom_xpos_pad_.end(), 0.0);
    std::fill(geom_xmat_pad_.begin(), geom_xmat_pad_.end(), 0.0);
    std::fill(geom_rgba_pad_.begin(), geom_rgba_pad_.end(), 0.0);
    std::fill(geom_friction_pad_.begin(), geom_friction_pad_.end(), 0.0);
    std::fill(geom_aabb_pad_.begin(), geom_aabb_pad_.end(), 0.0);
    std::fill(geom_rbound_pad_.begin(), geom_rbound_pad_.end(), 0.0);
    std::fill(geom_contype_pad_.begin(), geom_contype_pad_.end(), 0.0);
    std::fill(geom_conaffinity_pad_.begin(), geom_conaffinity_pad_.end(), 0.0);
    std::fill(geom_type_pad_.begin(), geom_type_pad_.end(), 0.0);
    std::fill(geom_condim_pad_.begin(), geom_condim_pad_.end(), 0.0);
    std::fill(hfield_data_pad_.begin(), hfield_data_pad_.end(), 0.0);
    std::fill(mocap_pos_pad_.begin(), mocap_pos_pad_.end(), 0.0);
    std::fill(mocap_quat_pad_.begin(), mocap_quat_pad_.end(), 0.0);
    for (int i = 0; i < model_->nq && i < 2048; ++i) {
      qpos_pad_[i] = data_->qpos[i];
    }
    for (int i = 0; i < model_->nv && i < 2048; ++i) {
      qvel_pad_[i] = data_->qvel[i];
      qacc_pad_[i] = data_->qacc[i];
      qacc_warmstart_pad_[i] = data_->qacc_warmstart[i];
    }
    for (int i = 0; i < model_->na && i < 2048; ++i) {
      act_pad_[i] = data_->act[i];
    }
    for (int i = 0; i < model_->nu && i < 2048; ++i) {
      ctrl_pad_[i] = data_->ctrl[i];
      actuator_length_pad_[i] = data_->actuator_length[i];
      actuator_velocity_pad_[i] = data_->actuator_velocity[i];
      actuator_force_pad_[i] = data_->actuator_force[i];
    }
    for (int i = 0; i < static_cast<int>(fatigue_ma_.size()) && i < 2048; ++i) {
      fatigue_ma_pad_[i] = fatigue_ma_[i];
      fatigue_mr_pad_[i] = fatigue_mr_[i];
      fatigue_mf_pad_[i] = fatigue_mf_[i];
      fatigue_tl_pad_[i] = fatigue_tl_[i];
      fatigue_tauact_pad_[i] = fatigue_tauact_[i];
      fatigue_taudeact_pad_[i] = fatigue_taudeact_[i];
    }
    for (int i = 0; i < model_->nsite * 3 && i < kMyoSuiteTestStatePad; ++i) {
      site_pos_pad_[i] = model_->site_pos[i];
      site_xpos_pad_[i] = data_->site_xpos[i];
    }
    for (int i = 0; i < model_->nsite * 4 && i < kMyoSuiteTestStatePad; ++i) {
      site_quat_pad_[i] = model_->site_quat[i];
    }
    for (int i = 0; i < model_->nsite * 3 && i < kMyoSuiteTestStatePad; ++i) {
      site_size_pad_[i] = model_->site_size[i];
    }
    for (int i = 0; i < model_->nsite * 4 && i < kMyoSuiteTestStatePad; ++i) {
      site_rgba_pad_[i] = model_->site_rgba[i];
    }
    for (int i = 0; i < model_->nbody * 3 && i < kMyoSuiteTestStatePad; ++i) {
      body_pos_pad_[i] = model_->body_pos[i];
    }
    for (int i = 0; i < model_->nbody * 4 && i < kMyoSuiteTestStatePad; ++i) {
      body_quat_pad_[i] = model_->body_quat[i];
    }
    for (int i = 0; i < model_->nbody && i < kMyoSuiteTestStatePad; ++i) {
      body_mass_pad_[i] = model_->body_mass[i];
    }
    for (int i = 0; i < model_->nlight * 3 && i < kMyoSuiteTestStatePad; ++i) {
      light_xpos_pad_[i] = data_->light_xpos[i];
      light_xdir_pad_[i] = data_->light_xdir[i];
    }
    for (int i = 0; i < model_->ngeom * 3 && i < kMyoSuiteTestStatePad; ++i) {
      geom_pos_pad_[i] = model_->geom_pos[i];
      geom_size_pad_[i] = model_->geom_size[i];
      geom_friction_pad_[i] = model_->geom_friction[i];
      geom_xpos_pad_[i] = data_->geom_xpos[i];
    }
    for (int i = 0; i < model_->ngeom * 4 && i < kMyoSuiteTestStatePad; ++i) {
      geom_quat_pad_[i] = model_->geom_quat[i];
      geom_rgba_pad_[i] = model_->geom_rgba[i];
    }
    for (int i = 0; i < model_->ngeom * 9 && i < kMyoSuiteTestStatePad; ++i) {
      geom_xmat_pad_[i] = data_->geom_xmat[i];
    }
    for (int i = 0; i < model_->ngeom && i < kMyoSuiteTestStatePad; ++i) {
      geom_rbound_pad_[i] = model_->geom_rbound[i];
      geom_contype_pad_[i] = model_->geom_contype[i];
      geom_conaffinity_pad_[i] = model_->geom_conaffinity[i];
      geom_type_pad_[i] = model_->geom_type[i];
      geom_condim_pad_[i] = model_->geom_condim[i];
    }
    for (int i = 0; i < model_->ngeom * 6 && i < kMyoSuiteTestStatePad; ++i) {
      geom_aabb_pad_[i] = model_->geom_aabb[i];
    }
    for (int i = 0; i < model_->nhfielddata && i < kMyoSuiteTestStatePad; ++i) {
      hfield_data_pad_[i] = model_->hfield_data[i];
    }
    for (int i = 0; i < model_->nmocap * 3 && i < kMyoSuiteTestStatePad; ++i) {
      mocap_pos_pad_[i] = data_->mocap_pos[i];
    }
    for (int i = 0; i < model_->nmocap * 4 && i < kMyoSuiteTestStatePad; ++i) {
      mocap_quat_pad_[i] = data_->mocap_quat[i];
    }
#endif
  }

  void WriteState(mjtNum reward, bool reset) {
    WriteState(reward, reset, BuildObsDict());
  }

  void WriteState(mjtNum reward, bool reset, const ObsDict& obs_dict) {
    auto state = Allocate();
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      std::vector<mjtNum> obs = Observation(obs_dict);
      auto obs_state = state["obs"_];
      AssignObservation("obs", &obs_state, obs.data(), obs.size(), reset);
    }
    state["reward"_] = static_cast<float>(reward);
    state["trunc"_] = elapsed_step_ >= max_episode_steps_;
    state["info:task_id"_] = task_index_;
    state["info:sparse"_] = sparse_;
    state["info:solved"_] = solved_;
    state["info:oracle_numpy2_broken"_] = task_.oracle_numpy2_broken;
    state["info:model_nq"_] = model_->nq;
    state["info:model_nv"_] = model_->nv;
    state["info:model_na"_] = model_->na;
    state["info:model_nu"_] = model_->nu;
    state["info:model_nsite"_] = model_->nsite;
    state["info:model_nbody"_] = model_->nbody;
    state["info:model_ngeom"_] = model_->ngeom;
    state["info:model_nhfielddata"_] = model_->nhfielddata;
    state["info:model_nmocap"_] = model_->nmocap;
#ifdef ENVPOOL_TEST
    CapturePaddedCurrentState();
    state["info:qpos0"_].Assign(qpos0_pad_.data(), qpos0_pad_.size());
    state["info:qvel0"_].Assign(qvel0_pad_.data(), qvel0_pad_.size());
    state["info:act0"_].Assign(act0_pad_.data(), act0_pad_.size());
    state["info:qacc0"_].Assign(qacc0_pad_.data(), qacc0_pad_.size());
    state["info:qacc_warmstart0"_].Assign(qacc_warmstart0_pad_.data(),
                                          qacc_warmstart0_pad_.size());
    state["info:qpos"_].Assign(qpos_pad_.data(), qpos_pad_.size());
    state["info:qvel"_].Assign(qvel_pad_.data(), qvel_pad_.size());
    state["info:act"_].Assign(act_pad_.data(), act_pad_.size());
    state["info:ctrl"_].Assign(ctrl_pad_.data(), ctrl_pad_.size());
    state["info:qacc"_].Assign(qacc_pad_.data(), qacc_pad_.size());
    state["info:qacc_warmstart"_].Assign(qacc_warmstart_pad_.data(),
                                         qacc_warmstart_pad_.size());
    state["info:actuator_length"_].Assign(actuator_length_pad_.data(),
                                          actuator_length_pad_.size());
    state["info:actuator_velocity"_].Assign(actuator_velocity_pad_.data(),
                                            actuator_velocity_pad_.size());
    state["info:actuator_force"_].Assign(actuator_force_pad_.data(),
                                         actuator_force_pad_.size());
    state["info:fatigue_ma"_].Assign(fatigue_ma_pad_.data(),
                                     fatigue_ma_pad_.size());
    state["info:fatigue_mr"_].Assign(fatigue_mr_pad_.data(),
                                     fatigue_mr_pad_.size());
    state["info:fatigue_mf"_].Assign(fatigue_mf_pad_.data(),
                                     fatigue_mf_pad_.size());
    state["info:fatigue_tl"_].Assign(fatigue_tl_pad_.data(),
                                     fatigue_tl_pad_.size());
    state["info:fatigue_tauact"_].Assign(fatigue_tauact_pad_.data(),
                                         fatigue_tauact_pad_.size());
    state["info:fatigue_taudeact"_].Assign(fatigue_taudeact_pad_.data(),
                                           fatigue_taudeact_pad_.size());
    state["info:fatigue_dt"_] =
        task_.muscle_condition == MyoSuiteMuscleCondition::kFatigue
            ? static_cast<mjtNum>(Dt())
            : 0.0;
    state["info:site_pos"_].Assign(site_pos_pad_.data(), site_pos_pad_.size());
    state["info:site_quat"_].Assign(site_quat_pad_.data(),
                                    site_quat_pad_.size());
    state["info:site_xpos"_].Assign(site_xpos_pad_.data(),
                                    site_xpos_pad_.size());
    state["info:site_size"_].Assign(site_size_pad_.data(),
                                    site_size_pad_.size());
    state["info:site_rgba"_].Assign(site_rgba_pad_.data(),
                                    site_rgba_pad_.size());
    state["info:body_pos"_].Assign(body_pos_pad_.data(), body_pos_pad_.size());
    state["info:body_quat"_].Assign(body_quat_pad_.data(),
                                    body_quat_pad_.size());
    state["info:body_mass"_].Assign(body_mass_pad_.data(),
                                    body_mass_pad_.size());
    state["info:light_xpos"_].Assign(light_xpos_pad_.data(),
                                     light_xpos_pad_.size());
    state["info:light_xdir"_].Assign(light_xdir_pad_.data(),
                                     light_xdir_pad_.size());
    state["info:geom_pos"_].Assign(geom_pos_pad_.data(), geom_pos_pad_.size());
    state["info:geom_quat"_].Assign(geom_quat_pad_.data(),
                                    geom_quat_pad_.size());
    state["info:geom_size"_].Assign(geom_size_pad_.data(),
                                    geom_size_pad_.size());
    state["info:geom_xpos"_].Assign(geom_xpos_pad_.data(),
                                    geom_xpos_pad_.size());
    state["info:geom_xmat"_].Assign(geom_xmat_pad_.data(),
                                    geom_xmat_pad_.size());
    state["info:geom_rgba"_].Assign(geom_rgba_pad_.data(),
                                    geom_rgba_pad_.size());
    state["info:geom_friction"_].Assign(geom_friction_pad_.data(),
                                        geom_friction_pad_.size());
    state["info:geom_aabb"_].Assign(geom_aabb_pad_.data(),
                                    geom_aabb_pad_.size());
    state["info:geom_rbound"_].Assign(geom_rbound_pad_.data(),
                                      geom_rbound_pad_.size());
    state["info:geom_contype"_].Assign(geom_contype_pad_.data(),
                                       geom_contype_pad_.size());
    state["info:geom_conaffinity"_].Assign(geom_conaffinity_pad_.data(),
                                           geom_conaffinity_pad_.size());
    state["info:geom_type"_].Assign(geom_type_pad_.data(),
                                    geom_type_pad_.size());
    state["info:geom_condim"_].Assign(geom_condim_pad_.data(),
                                      geom_condim_pad_.size());
    state["info:hfield_data"_].Assign(hfield_data_pad_.data(),
                                      hfield_data_pad_.size());
    state["info:mocap_pos"_].Assign(mocap_pos_pad_.data(),
                                    mocap_pos_pad_.size());
    state["info:mocap_quat"_].Assign(mocap_quat_pad_.data(),
                                     mocap_quat_pad_.size());
    state["info:time"_] = data_->time;
    state["info:model_timestep"_] = model_->opt.timestep;
    state["info:frame_skip"_] = frame_skip_;
#endif
  }
};

using MyoSuiteEnv = MyoSuiteEnvBase<MyoSuiteEnvSpec, false>;
using MyoSuitePixelEnv = MyoSuiteEnvBase<MyoSuitePixelEnvSpec, true>;
using MyoSuiteEnvPool = AsyncEnvPool<MyoSuiteEnv>;
using MyoSuitePixelEnvPool = AsyncEnvPool<MyoSuitePixelEnv>;

}  // namespace myosuite

#endif  // ENVPOOL_MUJOCO_MYOSUITE_MYOSUITE_ENV_H_
