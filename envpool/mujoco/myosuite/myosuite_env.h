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
#include <cmath>
#include <cstring>
#include <limits>
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
#include "third_party/myosuite/myosuite_task_metadata.h"
#include "third_party/myosuite/myosuite_tasks.h"

namespace myosuite {

using envpool::mujoco::PixelObservationEnvFns;
using envpool::mujoco::RenderCameraIdOrDefault;
using envpool::mujoco::RenderHeightOrDefault;
using envpool::mujoco::RenderWidthOrDefault;
using envpool::mujoco::StackSpec;
using third_party::myosuite::GetMyoSuiteTask;
using third_party::myosuite::GetMyoSuiteTaskMetadata;
using third_party::myosuite::MyoSuiteMuscleCondition;
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
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({2048})),
        "info:qpos"_.Bind(Spec<mjtNum>({2048})),
        "info:qvel"_.Bind(Spec<mjtNum>({2048})),
        "info:act"_.Bind(Spec<mjtNum>({2048})),
        "info:ctrl"_.Bind(Spec<mjtNum>({2048})),
        "info:qacc_warmstart"_.Bind(Spec<mjtNum>({2048})),
#endif
        "info:model_nq"_.Bind(Spec<int>({})),
        "info:model_nv"_.Bind(Spec<int>({})),
        "info:model_na"_.Bind(Spec<int>({})));
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

  const MyoSuiteTaskDef& task_;
  const MyoSuiteTaskMetadata& metadata_;
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
  mjtNum sparse_{0.0};
  mjtNum solved_{0.0};
#ifdef ENVPOOL_TEST
  std::vector<mjtNum> qpos0_pad_;
  std::vector<mjtNum> qvel0_pad_;
  std::vector<mjtNum> act0_pad_;
  std::vector<mjtNum> qacc_warmstart0_pad_;
  std::vector<mjtNum> qpos_pad_;
  std::vector<mjtNum> qvel_pad_;
  std::vector<mjtNum> act_pad_;
  std::vector<mjtNum> ctrl_pad_;
  std::vector<mjtNum> qacc_warmstart_pad_;
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
        qacc_warmstart0_pad_(2048, 0.0),
        qpos_pad_(2048, 0.0),
        qvel_pad_(2048, 0.0),
        act_pad_(2048, 0.0),
        ctrl_pad_(2048, 0.0),
        qacc_warmstart_pad_(2048, 0.0)
#endif
  {
    ApplyMuscleCondition();
    InitializeFatigue();
    ApplyMetadataInitialState();
    SetDefaultInitialQpos();
    InitializeRobotEnv();
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    task_step_ = 0;
    sparse_ = 0.0;
    solved_ = 0.0;
    ResetFatigue();
    ResetToInitialState();
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
        const float action_value = static_cast<float>(value);
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
    PreStepTaskUpdate();
    const bool robot_step_normalizes_ctrl =
        task_.normalize_act && model_->na == 0;
    for (int i = 0; i < model_->nu; ++i) {
      if (robot_step_normalizes_ctrl) {
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
    mjv_defaultFreeCamera(model_, camera);
    return true;
  }

 protected:
  struct RewardResult {
    mjtNum dense{0.0};
    mjtNum sparse{0.0};
    mjtNum solved{0.0};
    bool terminated{false};
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
    return std::vector<mjtNum>(data_->sensordata + start,
                               data_->sensordata + start + dim);
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

  void ApplyFatigue(std::vector<mjtNum>* ctrl) {
    if (task_.muscle_condition != MyoSuiteMuscleCondition::kFatigue) {
      return;
    }
    constexpr mjtNum kRecoveryMultiplier = 10.0 * 15.0;
    constexpr mjtNum kFatigueCoefficient = 0.00912;
    constexpr mjtNum kRecoveryCoefficient = 0.1 * 0.00094;
    const mjtNum dt = static_cast<mjtNum>(Dt());
    for (std::size_t i = 0; i < muscle_actuator_ids_.size(); ++i) {
      const int actuator_id = muscle_actuator_ids_[i];
      fatigue_tl_[i] = (*ctrl)[actuator_id];

      const mjtNum ma = fatigue_ma_[i];
      const mjtNum mr = fatigue_mr_[i];
      const mjtNum mf = fatigue_mf_[i];
      const mjtNum tl = fatigue_tl_[i];
      const mjtNum ld = (0.5 + 1.5 * ma) / fatigue_tauact_[i];
      const mjtNum lr = (0.5 + 1.5 * ma) / fatigue_taudeact_[i];

      mjtNum transfer = 0.0;
      if (ma < tl && mr > tl - ma) {
        transfer = ld * (tl - ma);
      } else if (ma < tl) {
        transfer = ld * mr;
      } else {
        transfer = lr * (tl - ma);
      }

      const mjtNum recovery = ma >= tl
                                  ? kRecoveryMultiplier * kRecoveryCoefficient
                                  : kRecoveryCoefficient;
      const mjtNum lower = std::max(-ma / dt + kFatigueCoefficient * ma,
                                    (mr - 1.0) / dt + recovery * mf);
      const mjtNum upper = std::min((1.0 - ma) / dt + kFatigueCoefficient * ma,
                                    mr / dt + recovery * mf);
      transfer = std::max(lower, std::min(upper, transfer));

      fatigue_ma_[i] += (transfer - kFatigueCoefficient * ma) * dt;
      fatigue_mr_[i] += (-transfer + recovery * mf) * dt;
      fatigue_mf_[i] += (kFatigueCoefficient * ma - recovery * mf) * dt;
      (*ctrl)[actuator_id] = fatigue_ma_[i];
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
    mj_forward(model_, data_);
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
    return std::vector<mjtNum>(data_->qpos + begin, data_->qpos + end);
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

  std::vector<mjtNum> Act() const {
    if (model_->na <= 0) {
      return {};
    }
    return std::vector<mjtNum>(data_->act, data_->act + model_->na);
  }

  std::vector<mjtNum> ActuatorLength() const {
    return std::vector<mjtNum>(data_->actuator_length,
                               data_->actuator_length + model_->nu);
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
    mjtNum cvel[6] = {};
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
    mjtNum com[3] = {};
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

  ObsDict BuildObsDict() const {
    ObsDict obs;
    const mjtNum dt = static_cast<mjtNum>(Dt());
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
    obs["opponent_pose"] = {};
    obs["opponent_vel"] = {};

    obs["pelvis_pos"] = SiteXpos(SiteId("pelvis"));
    obs["body_qpos"] =
        QposSlice(0, std::max(0, static_cast<int>(model_->nq) - 7));
    obs["body_qvel"] =
        QvelSlice(0, std::max(0, static_cast<int>(model_->nv) - 6), true);
    obs["ball_pos"] = SiteXpos(SiteId("pingpong"));
    obs["ball_vel"] = SensorData("pingpong_vel_sensor");
    obs["paddle_pos"] = SiteXpos(SiteId("paddle"));
    obs["paddle_vel"] = SensorData("paddle_vel_sensor");
    obs["paddle_ori"] = BodyXquat(BodyId("paddle"));
    obs["touching_info"] = std::vector<mjtNum>(6, 0.0);

    // MyoDM tracking metadata is loaded natively as model state; reference
    // tracks are added separately. Until then, fixed-reference IDs use the
    // reference implied by the official initial qpos.
    obs["curr_hand_qpos"] =
        QposSlice(0, std::max(0, static_cast<int>(model_->nq) - 6));
    obs["curr_hand_qvel"] =
        QvelSlice(0, std::max(0, static_cast<int>(model_->nv) - 6), false);
    obs["targ_hand_qpos"] =
        std::vector<mjtNum>(obs["curr_hand_qpos"].size(), 0.0);
    obs["targ_hand_qvel"] =
        std::vector<mjtNum>(obs["curr_hand_qvel"].size(), 0.0);
    obs["hand_qpos_err"] =
        Subtract(obs["curr_hand_qpos"], obs["targ_hand_qpos"]);
    obs["hand_qvel_err"] =
        Subtract(obs["curr_hand_qvel"], obs["targ_hand_qvel"]);
    const int object_bid = task_.object_name[0] != '\0'
                               ? BodyId(task_.object_name)
                               : BodyId("Object");
    obs["curr_obj_com"] = BodyXpos(object_bid);
    obs["targ_obj_com"] = {0.2, 0.2, 0.1};
    obs["obj_com_err"] = Subtract(obs["curr_obj_com"], obs["targ_obj_com"]);
    (void)dt;
    return obs;
  }

  const std::vector<mjtNum>& ObsValue(const ObsDict& obs,
                                      const std::string& key) const {
    static const std::vector<mjtNum> kEmpty;
    auto it = obs.find(key);
    return it == obs.end() ? kEmpty : it->second;
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
      mjtNum mat[9];
      mju_quat2Mat(mat, quat.data());
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

  RewardResult ComputeReward(const ObsDict& obs) const {
    std::unordered_map<std::string, mjtNum> values;
    bool terminated = false;
    const mjtNum pi = std::acos(static_cast<mjtNum>(-1.0));

    if (task_.kind == MyoSuiteTaskKind::kPose ||
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
      values["solved"] = rot_align > 0.95 && !dropped ? 1.0 : 0.0;
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
      values["solved"] =
          pos_dist < 0.025 && rot_dist < 0.262 && !drop ? 1.0 : 0.0;
      terminated = drop;
    } else {
      values["sparse"] = 0.0;
      values["solved"] = 0.0;
      values["done"] = 0.0;
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
    std::fill(qacc_warmstart_pad_.begin(), qacc_warmstart_pad_.end(), 0.0);
    for (int i = 0; i < model_->nq && i < 2048; ++i) {
      qpos_pad_[i] = data_->qpos[i];
    }
    for (int i = 0; i < model_->nv && i < 2048; ++i) {
      qvel_pad_[i] = data_->qvel[i];
      qacc_warmstart_pad_[i] = data_->qacc_warmstart[i];
    }
    for (int i = 0; i < model_->na && i < 2048; ++i) {
      act_pad_[i] = data_->act[i];
    }
    for (int i = 0; i < model_->nu && i < 2048; ++i) {
      ctrl_pad_[i] = data_->ctrl[i];
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
#ifdef ENVPOOL_TEST
    CapturePaddedCurrentState();
    state["info:qpos0"_].Assign(qpos0_pad_.data(), qpos0_pad_.size());
    state["info:qvel0"_].Assign(qvel0_pad_.data(), qvel0_pad_.size());
    state["info:act0"_].Assign(act0_pad_.data(), act0_pad_.size());
    state["info:qacc_warmstart0"_].Assign(qacc_warmstart0_pad_.data(),
                                          qacc_warmstart0_pad_.size());
    state["info:qpos"_].Assign(qpos_pad_.data(), qpos_pad_.size());
    state["info:qvel"_].Assign(qvel_pad_.data(), qvel_pad_.size());
    state["info:act"_].Assign(act_pad_.data(), act_pad_.size());
    state["info:ctrl"_].Assign(ctrl_pad_.data(), ctrl_pad_.size());
    state["info:qacc_warmstart"_].Assign(qacc_warmstart_pad_.data(),
                                         qacc_warmstart_pad_.size());
#endif
  }
};

using MyoSuiteEnv = MyoSuiteEnvBase<MyoSuiteEnvSpec, false>;
using MyoSuitePixelEnv = MyoSuiteEnvBase<MyoSuitePixelEnvSpec, true>;
using MyoSuiteEnvPool = AsyncEnvPool<MyoSuiteEnv>;
using MyoSuitePixelEnvPool = AsyncEnvPool<MyoSuitePixelEnv>;

}  // namespace myosuite

#endif  // ENVPOOL_MUJOCO_MYOSUITE_MYOSUITE_ENV_H_
