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

#ifndef ENVPOOL_MUJOCO_ROBOTICS_KITCHEN_H_
#define ENVPOOL_MUJOCO_ROBOTICS_KITCHEN_H_

#include <mujoco.h>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/robotics/mujoco_env.h"
#include "envpool/mujoco/robotics/utils.h"

namespace gymnasium_robotics {
namespace kitchen_internal {

constexpr int kTaskCount = 7;
constexpr int kRobotDim = 9;
constexpr int kObjectQposDim = 21;
constexpr int kObjectQvelDim = 20;

struct TaskSpec {
  const char* name;
  int dim;
  std::array<int, 7> qpos_indices;
  std::array<mjtNum, 7> goal;
};

inline constexpr std::array<TaskSpec, kTaskCount> kTaskSpecs = {{
    {"bottom burner",
     2,
     {11, 12, 0, 0, 0, 0, 0},
     {-0.88, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {"top burner",
     2,
     {15, 16, 0, 0, 0, 0, 0},
     {-0.92, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {"light switch",
     2,
     {17, 18, 0, 0, 0, 0, 0},
     {-0.69, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {"slide cabinet",
     1,
     {19, 0, 0, 0, 0, 0, 0},
     {0.37, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {"hinge cabinet",
     2,
     {20, 21, 0, 0, 0, 0, 0},
     {0.0, 1.45, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {"microwave",
     1,
     {22, 0, 0, 0, 0, 0, 0},
     {-0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {"kettle",
     7,
     {23, 24, 25, 26, 27, 28, 29},
     {-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06}},
}};

inline constexpr std::array<mjtNum, 30> kInitQpos = {
    1.48388023e-01,  -1.76848573e00,  1.84390296e00,   -2.47685760e00,
    2.60252026e-01,  7.12533105e-01,  1.59515394e00,   4.79267505e-02,
    3.71350919e-02,  -2.66279850e-04, -5.18043486e-05, 3.12877220e-05,
    -4.51199853e-05, -3.90842156e-06, -4.22629655e-05, 6.28065475e-05,
    4.04984708e-05,  4.62730939e-04,  -2.26906415e-04, -4.65501369e-04,
    -6.44129196e-03, -1.77048263e-03, 1.08009684e-03,  -2.69397440e-01,
    3.50383255e-01,  1.61944683e00,   1.00618764e00,   4.06395120e-03,
    -6.62095997e-03, -2.68278933e-04,
};

inline constexpr std::array<std::array<mjtNum, 2>, 29> kRobotPosBound = {{
    {-2.9, 2.9},     {-1.8, 1.8},     {-2.9, 2.9},     {-3.1, 0.0},
    {-2.9, 2.9},     {0.0, 3.8},      {-2.9, 2.9},     {0.0, 0.04},
    {0.0, 0.04},     {-0.5, 0.0},     {-0.5, 0.0},     {-0.005, 0.0},
    {-0.005, 0.0},   {-0.005, 0.0},   {-0.005, 0.0},   {-0.005, 0.0},
    {-0.005, 0.0},   {-1.5, 1.5},     {-1.5, 1.5},     {-1.5, 1.5},
    {-10.57, 10.57}, {-10.57, 10.57}, {-10.57, 10.57}, {-1.5, 1.5},
    {-1.5, 1.5},     {-1.5, 1.5},     {-10.57, 10.57}, {-10.57, 10.57},
    {-10.57, 10.57},
}};

inline constexpr std::array<std::array<mjtNum, 2>, 29> kRobotVelBound = {{
    {-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0},
    {-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}, {-5.0, 5.0},
    {-5.0, 5.0},   {-5.0, 5.0},   {-5.0, 5.0},   {-5.0, 5.0},   {-5.0, 5.0},
    {-5.0, 5.0},   {-5.0, 5.0},   {-5.0, 5.0},   {-5.0, 5.0},   {-5.0, 5.0},
    {-0.5, 0.5},   {-0.5, 0.5},   {-0.5, 0.5},   {-5.0, 5.0},   {-5.0, 5.0},
    {-5.0, 5.0},   {-0.5, 0.5},   {-0.5, 0.5},   {-0.5, 0.5},
}};

inline constexpr std::array<mjtNum, 29> kRobotPosNoiseAmp = {
    0.1,   0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1,   0.1,   0.005,
    0.005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.005, 0.005, 0.005,
    0.1,   0.1,    0.1,    0.005,  0.005,  0.005,  0.1,    0.1,   0.1,
};

inline constexpr std::array<mjtNum, 29> kRobotVelNoiseAmp = {
    0.1,   0.1,   0.1,   0.1,   0.1,   0.1,   0.1,   0.1,   0.1,   0.005,
    0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
    0.1,   0.1,   0.1,   0.005, 0.005, 0.005, 0.1,   0.1,   0.1,
};

}  // namespace kitchen_internal

class KitchenEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(40),
        "frame_stack"_.Bind(1),
        "xml_file"_.Bind(
            std::string("kitchen_franka/kitchen_assets/kitchen_env_model.xml")),
        "tasks_to_complete"_.Bind(std::vector<std::string>{
            "bottom burner",
            "top burner",
            "light switch",
            "slide cabinet",
            "hinge cabinet",
            "microwave",
            "kettle",
        }),
        "terminate_on_tasks_completed"_.Bind(true),
        "remove_task_when_completed"_.Bind(true),
        "robot_noise_ratio"_.Bind(0.01), "object_noise_ratio"_.Bind(0.0005));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
#ifdef ENVPOOL_TEST
    return MakeDict(
        "obs:observation"_.Bind(
            StackSpec(Spec<mjtNum>({59}, {-inf, inf}), conf["frame_stack"_])),
        "obs:desired_goal:bottom burner"_.Bind(
            StackSpec(Spec<mjtNum>({2}, {-inf, inf}), conf["frame_stack"_])),
        "obs:desired_goal:top burner"_.Bind(
            StackSpec(Spec<mjtNum>({2}, {-inf, inf}), conf["frame_stack"_])),
        "obs:desired_goal:light switch"_.Bind(
            StackSpec(Spec<mjtNum>({2}, {-inf, inf}), conf["frame_stack"_])),
        "obs:desired_goal:slide cabinet"_.Bind(
            StackSpec(Spec<mjtNum>({1}, {-inf, inf}), conf["frame_stack"_])),
        "obs:desired_goal:hinge cabinet"_.Bind(
            StackSpec(Spec<mjtNum>({2}, {-inf, inf}), conf["frame_stack"_])),
        "obs:desired_goal:microwave"_.Bind(
            StackSpec(Spec<mjtNum>({1}, {-inf, inf}), conf["frame_stack"_])),
        "obs:desired_goal:kettle"_.Bind(
            StackSpec(Spec<mjtNum>({7}, {-inf, inf}), conf["frame_stack"_])),
        "obs:achieved_goal:bottom burner"_.Bind(
            StackSpec(Spec<mjtNum>({2}, {-inf, inf}), conf["frame_stack"_])),
        "obs:achieved_goal:top burner"_.Bind(
            StackSpec(Spec<mjtNum>({2}, {-inf, inf}), conf["frame_stack"_])),
        "obs:achieved_goal:light switch"_.Bind(
            StackSpec(Spec<mjtNum>({2}, {-inf, inf}), conf["frame_stack"_])),
        "obs:achieved_goal:slide cabinet"_.Bind(
            StackSpec(Spec<mjtNum>({1}, {-inf, inf}), conf["frame_stack"_])),
        "obs:achieved_goal:hinge cabinet"_.Bind(
            StackSpec(Spec<mjtNum>({2}, {-inf, inf}), conf["frame_stack"_])),
        "obs:achieved_goal:microwave"_.Bind(
            StackSpec(Spec<mjtNum>({1}, {-inf, inf}), conf["frame_stack"_])),
        "obs:achieved_goal:kettle"_.Bind(
            StackSpec(Spec<mjtNum>({7}, {-inf, inf}), conf["frame_stack"_])),
        "info:tasks_to_complete"_.Bind(
            Spec<int>({kitchen_internal::kTaskCount}, {0, 1})),
        "info:step_task_completions"_.Bind(
            Spec<int>({kitchen_internal::kTaskCount}, {0, 1})),
        "info:episode_task_completions"_.Bind(
            Spec<int>({kitchen_internal::kTaskCount}, {0, 1})),
        "info:qpos0"_.Bind(Spec<mjtNum>({30}, {-inf, inf})),
        "info:qvel0"_.Bind(Spec<mjtNum>({29}, {-inf, inf})),
        "info:qacc0"_.Bind(Spec<mjtNum>({29}, {-inf, inf})),
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({29}, {-inf, inf})));
#else
    return MakeDict("obs:observation"_.Bind(StackSpec(
                        Spec<mjtNum>({59}, {-inf, inf}), conf["frame_stack"_])),
                    "obs:desired_goal:bottom burner"_.Bind(StackSpec(
                        Spec<mjtNum>({2}, {-inf, inf}), conf["frame_stack"_])),
                    "obs:desired_goal:top burner"_.Bind(StackSpec(
                        Spec<mjtNum>({2}, {-inf, inf}), conf["frame_stack"_])),
                    "obs:desired_goal:light switch"_.Bind(StackSpec(
                        Spec<mjtNum>({2}, {-inf, inf}), conf["frame_stack"_])),
                    "obs:desired_goal:slide cabinet"_.Bind(StackSpec(
                        Spec<mjtNum>({1}, {-inf, inf}), conf["frame_stack"_])),
                    "obs:desired_goal:hinge cabinet"_.Bind(StackSpec(
                        Spec<mjtNum>({2}, {-inf, inf}), conf["frame_stack"_])),
                    "obs:desired_goal:microwave"_.Bind(StackSpec(
                        Spec<mjtNum>({1}, {-inf, inf}), conf["frame_stack"_])),
                    "obs:desired_goal:kettle"_.Bind(StackSpec(
                        Spec<mjtNum>({7}, {-inf, inf}), conf["frame_stack"_])),
                    "obs:achieved_goal:bottom burner"_.Bind(StackSpec(
                        Spec<mjtNum>({2}, {-inf, inf}), conf["frame_stack"_])),
                    "obs:achieved_goal:top burner"_.Bind(StackSpec(
                        Spec<mjtNum>({2}, {-inf, inf}), conf["frame_stack"_])),
                    "obs:achieved_goal:light switch"_.Bind(StackSpec(
                        Spec<mjtNum>({2}, {-inf, inf}), conf["frame_stack"_])),
                    "obs:achieved_goal:slide cabinet"_.Bind(StackSpec(
                        Spec<mjtNum>({1}, {-inf, inf}), conf["frame_stack"_])),
                    "obs:achieved_goal:hinge cabinet"_.Bind(StackSpec(
                        Spec<mjtNum>({2}, {-inf, inf}), conf["frame_stack"_])),
                    "obs:achieved_goal:microwave"_.Bind(StackSpec(
                        Spec<mjtNum>({1}, {-inf, inf}), conf["frame_stack"_])),
                    "obs:achieved_goal:kettle"_.Bind(StackSpec(
                        Spec<mjtNum>({7}, {-inf, inf}), conf["frame_stack"_])),
                    "info:tasks_to_complete"_.Bind(
                        Spec<int>({kitchen_internal::kTaskCount}, {0, 1})),
                    "info:step_task_completions"_.Bind(
                        Spec<int>({kitchen_internal::kTaskCount}, {0, 1})),
                    "info:episode_task_completions"_.Bind(
                        Spec<int>({kitchen_internal::kTaskCount}, {0, 1})));
#endif
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 9}, {-1.0, 1.0})));
  }
};

using KitchenEnvSpec = EnvSpec<KitchenEnvFns>;
using KitchenPixelEnvFns = PixelObservationEnvFns<KitchenEnvFns>;
using KitchenPixelEnvSpec = EnvSpec<KitchenPixelEnvFns>;

template <typename EnvSpecT, bool kFromPixels>
class KitchenEnvBase : public Env<EnvSpecT>, public MujocoRobotEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  bool terminate_on_tasks_completed_;
  bool remove_task_when_completed_;
  mjtNum robot_noise_ratio_;
  mjtNum object_noise_ratio_;
  std::array<mjtNum, kitchen_internal::kRobotDim> last_robot_qpos_{};
  std::array<int, kitchen_internal::kTaskCount> initial_tasks_{};
  std::array<int, kitchen_internal::kTaskCount> tasks_to_complete_{};
  std::array<int, kitchen_internal::kTaskCount> step_task_completions_{};
  std::array<int, kitchen_internal::kTaskCount> episode_task_completions_{};
  std::uniform_real_distribution<> noise_dist_{-1.0, 1.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;
  using State = typename Base::State;

  KitchenEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        MujocoRobotEnv(spec.config["base_path"_], spec.config["xml_file"_],
                       spec.config["frame_skip"_],
                       spec.config["max_episode_steps"_],
                       spec.config["frame_stack"_],
                       RenderWidthOrDefault<kFromPixels>(spec.config),
                       RenderHeightOrDefault<kFromPixels>(spec.config),
                       RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        terminate_on_tasks_completed_(
            spec.config["terminate_on_tasks_completed"_]),
        remove_task_when_completed_(spec.config["remove_task_when_completed"_]),
        robot_noise_ratio_(spec.config["robot_noise_ratio"_]),
        object_noise_ratio_(spec.config["object_noise_ratio"_]) {
    SetupTaskMask(spec.config["tasks_to_complete"_]);
    SetupInitialState();
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    ResetToInitialState();
    tasks_to_complete_ = initial_tasks_;
    step_task_completions_.fill(0);
    episode_task_completions_.fill(0);
    CaptureResetState();
    WriteState(0.0, true);
  }

  void Step(const Action& action) override {
    const auto* act = static_cast<const mjtNum*>(action["action"_].Data());
    for (int i = 0; i < kitchen_internal::kRobotDim; ++i) {
      mjtNum value = std::clamp(act[i], static_cast<mjtNum>(-1.0),
                                static_cast<mjtNum>(1.0)) *
                     static_cast<mjtNum>(2.0);
      value = std::clamp(value, kitchen_internal::kRobotVelBound[i][0],
                         kitchen_internal::kRobotVelBound[i][1]);
      value = last_robot_qpos_[i] + value * Dt();
      data_->ctrl[i] = std::clamp(value, kitchen_internal::kRobotPosBound[i][0],
                                  kitchen_internal::kRobotPosBound[i][1]);
    }
    DoSimulation();
    mj_rnePostConstraint(model_, data_);
    ++elapsed_step_;
    mjtNum reward = UpdateTaskCompletions();
    done_ = elapsed_step_ >= max_episode_steps_ ||
            (terminate_on_tasks_completed_ && AllTasksCompleted());
    WriteState(reward, false);
  }

 protected:
  bool RenderCamera(mjvCamera* camera) override {
    camera->distance = 2.2;
    camera->azimuth = 70.0;
    camera->elevation = -35.0;
    camera->lookat[0] = -0.2;
    camera->lookat[1] = 0.5;
    camera->lookat[2] = 2.0;
    return true;
  }

  static int TaskId(const std::string& task_name) {
    for (int i = 0; i < kitchen_internal::kTaskCount; ++i) {
      if (task_name == kitchen_internal::kTaskSpecs[i].name) {
        return i;
      }
    }
    throw std::runtime_error("Unknown FrankaKitchen task: " + task_name);
  }

  void SetupTaskMask(const std::vector<std::string>& task_names) {
    initial_tasks_.fill(0);
    for (const auto& task_name : task_names) {
      initial_tasks_[TaskId(task_name)] = 1;
    }
  }

  void SetupInitialState() {
    initial_time_ = data_->time;
    for (int i = 0; i < 30; ++i) {
      initial_qpos_[i] = kitchen_internal::kInitQpos[i];
    }
    std::fill(initial_qvel_.begin(), initial_qvel_.end(), 0.0);
    ResetToInitialState();
    auto robot_obs = NoisyRobotObs();
    for (int i = 0; i < kitchen_internal::kRobotDim; ++i) {
      last_robot_qpos_[i] = robot_obs.first[i];
    }
  }

  std::pair<std::array<mjtNum, kitchen_internal::kRobotDim>,
            std::array<mjtNum, kitchen_internal::kRobotDim>>
  NoisyRobotObs() {
    auto [robot_qpos_vec, robot_qvel_vec] = RobotGetObs(model_, data_);
    if (static_cast<int>(robot_qpos_vec.size()) !=
            kitchen_internal::kRobotDim ||
        static_cast<int>(robot_qvel_vec.size()) !=
            kitchen_internal::kRobotDim) {
      throw std::runtime_error("Unexpected Franka robot observation size.");
    }
    std::array<mjtNum, kitchen_internal::kRobotDim> robot_qpos{};
    std::array<mjtNum, kitchen_internal::kRobotDim> robot_qvel{};
    std::array<mjtNum, kitchen_internal::kRobotDim> robot_qpos_noise{};
    std::array<mjtNum, kitchen_internal::kRobotDim> robot_qvel_noise{};
    for (int i = 0; i < kitchen_internal::kRobotDim; ++i) {
      robot_qpos_noise[i] = noise_dist_(gen_);
    }
    for (int i = 0; i < kitchen_internal::kRobotDim; ++i) {
      robot_qvel_noise[i] = noise_dist_(gen_);
    }
    for (int i = 0; i < kitchen_internal::kRobotDim; ++i) {
      robot_qpos[i] =
          robot_qpos_vec[i] + robot_noise_ratio_ *
                                  kitchen_internal::kRobotPosNoiseAmp[i] *
                                  robot_qpos_noise[i];
      robot_qvel[i] =
          robot_qvel_vec[i] + robot_noise_ratio_ *
                                  kitchen_internal::kRobotVelNoiseAmp[i] *
                                  robot_qvel_noise[i];
    }
    for (int i = 0; i < kitchen_internal::kRobotDim; ++i) {
      last_robot_qpos_[i] = robot_qpos[i];
    }
    return {robot_qpos, robot_qvel};
  }

  mjtNum UpdateTaskCompletions() {
    step_task_completions_.fill(0);
    mjtNum reward = 0.0;
    for (int task_id = 0; task_id < kitchen_internal::kTaskCount; ++task_id) {
      if (tasks_to_complete_[task_id] == 0) {
        continue;
      }
      if (TaskDistance(task_id) < 0.3) {
        step_task_completions_[task_id] = 1;
        episode_task_completions_[task_id] = 1;
        reward += 1.0;
        if (remove_task_when_completed_) {
          tasks_to_complete_[task_id] = 0;
        }
      }
    }
    return reward;
  }

  bool AllTasksCompleted() const {
    for (int i = 0; i < kitchen_internal::kTaskCount; ++i) {
      if (initial_tasks_[i] != 0 && episode_task_completions_[i] == 0) {
        return false;
      }
    }
    return true;
  }

  mjtNum TaskDistance(int task_id) const {
    const auto& task = kitchen_internal::kTaskSpecs[task_id];
    mjtNum distance_sq = 0.0;
    for (int i = 0; i < task.dim; ++i) {
      mjtNum diff = data_->qpos[task.qpos_indices[i]] - task.goal[i];
      distance_sq += diff * diff;
    }
    return std::sqrt(distance_sq);
  }

  void WriteGoalState(State* state, const char* prefix, int task_id,
                      bool reset) {
    const auto& task = kitchen_internal::kTaskSpecs[task_id];
    std::vector<mjtNum> value(task.dim);
    for (int i = 0; i < task.dim; ++i) {
      value[i] =
          prefix[4] == 'd' ? task.goal[i] : data_->qpos[task.qpos_indices[i]];
    }
    if (task_id == 0 && prefix[4] == 'd') {
      auto obs = (*state)["obs:desired_goal:bottom burner"_];
      AssignObservation("obs:desired_goal:bottom burner", &obs, value.data(),
                        value.size(), reset);
    } else if (task_id == 1 && prefix[4] == 'd') {
      auto obs = (*state)["obs:desired_goal:top burner"_];
      AssignObservation("obs:desired_goal:top burner", &obs, value.data(),
                        value.size(), reset);
    } else if (task_id == 2 && prefix[4] == 'd') {
      auto obs = (*state)["obs:desired_goal:light switch"_];
      AssignObservation("obs:desired_goal:light switch", &obs, value.data(),
                        value.size(), reset);
    } else if (task_id == 3 && prefix[4] == 'd') {
      auto obs = (*state)["obs:desired_goal:slide cabinet"_];
      AssignObservation("obs:desired_goal:slide cabinet", &obs, value.data(),
                        value.size(), reset);
    } else if (task_id == 4 && prefix[4] == 'd') {
      auto obs = (*state)["obs:desired_goal:hinge cabinet"_];
      AssignObservation("obs:desired_goal:hinge cabinet", &obs, value.data(),
                        value.size(), reset);
    } else if (task_id == 5 && prefix[4] == 'd') {
      auto obs = (*state)["obs:desired_goal:microwave"_];
      AssignObservation("obs:desired_goal:microwave", &obs, value.data(),
                        value.size(), reset);
    } else if (task_id == 6 && prefix[4] == 'd') {
      auto obs = (*state)["obs:desired_goal:kettle"_];
      AssignObservation("obs:desired_goal:kettle", &obs, value.data(),
                        value.size(), reset);
    } else if (task_id == 0) {
      auto obs = (*state)["obs:achieved_goal:bottom burner"_];
      AssignObservation("obs:achieved_goal:bottom burner", &obs, value.data(),
                        value.size(), reset);
    } else if (task_id == 1) {
      auto obs = (*state)["obs:achieved_goal:top burner"_];
      AssignObservation("obs:achieved_goal:top burner", &obs, value.data(),
                        value.size(), reset);
    } else if (task_id == 2) {
      auto obs = (*state)["obs:achieved_goal:light switch"_];
      AssignObservation("obs:achieved_goal:light switch", &obs, value.data(),
                        value.size(), reset);
    } else if (task_id == 3) {
      auto obs = (*state)["obs:achieved_goal:slide cabinet"_];
      AssignObservation("obs:achieved_goal:slide cabinet", &obs, value.data(),
                        value.size(), reset);
    } else if (task_id == 4) {
      auto obs = (*state)["obs:achieved_goal:hinge cabinet"_];
      AssignObservation("obs:achieved_goal:hinge cabinet", &obs, value.data(),
                        value.size(), reset);
    } else if (task_id == 5) {
      auto obs = (*state)["obs:achieved_goal:microwave"_];
      AssignObservation("obs:achieved_goal:microwave", &obs, value.data(),
                        value.size(), reset);
    } else {
      auto obs = (*state)["obs:achieved_goal:kettle"_];
      AssignObservation("obs:achieved_goal:kettle", &obs, value.data(),
                        value.size(), reset);
    }
  }

  void WriteState(mjtNum reward, bool reset) {
    State state = Allocate();
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      auto [robot_qpos, robot_qvel] = NoisyRobotObs();
      auto obs_observation = state["obs:observation"_];
      auto* obs = PrepareObservation("obs:observation", &obs_observation);
      int obs_index = 0;
      for (int i = 0; i < kitchen_internal::kRobotDim; ++i) {
        obs[obs_index++] = robot_qpos[i];
      }
      for (int i = 0; i < kitchen_internal::kRobotDim; ++i) {
        obs[obs_index++] = robot_qvel[i];
      }
      for (int i = 0; i < kitchen_internal::kObjectQposDim; ++i) {
        obs[obs_index++] = data_->qpos[9 + i] +
                           object_noise_ratio_ *
                               kitchen_internal::kRobotPosNoiseAmp[8 + i] *
                               noise_dist_(gen_);
      }
      for (int i = 0; i < kitchen_internal::kObjectQvelDim; ++i) {
        obs[obs_index++] = data_->qvel[9 + i] +
                           object_noise_ratio_ *
                               kitchen_internal::kRobotVelNoiseAmp[9 + i] *
                               noise_dist_(gen_);
      }
      CommitObservation("obs:observation", &obs_observation, reset);
      for (int task_id = 0; task_id < kitchen_internal::kTaskCount; ++task_id) {
        WriteGoalState(&state, "obs:desired_goal:", task_id, reset);
        WriteGoalState(&state, "obs:achieved_goal:", task_id, reset);
      }
      state["info:tasks_to_complete"_].Assign(tasks_to_complete_.data(),
                                              tasks_to_complete_.size());
      state["info:step_task_completions"_].Assign(
          step_task_completions_.data(), step_task_completions_.size());
      state["info:episode_task_completions"_].Assign(
          episode_task_completions_.data(), episode_task_completions_.size());
    }
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
    state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
    state["info:qacc0"_].Assign(qacc0_.data(), qacc0_.size());
    state["info:qacc_warmstart0"_].Assign(qacc_warmstart0_.data(),
                                          qacc_warmstart0_.size());
#endif
    state["reward"_] = static_cast<float>(reward);
  }
};

using KitchenEnv = KitchenEnvBase<KitchenEnvSpec, false>;
using KitchenPixelEnv = KitchenEnvBase<KitchenPixelEnvSpec, true>;
using KitchenEnvPool = AsyncEnvPool<KitchenEnv>;
using KitchenPixelEnvPool = AsyncEnvPool<KitchenPixelEnv>;

}  // namespace gymnasium_robotics

#endif  // ENVPOOL_MUJOCO_ROBOTICS_KITCHEN_H_
