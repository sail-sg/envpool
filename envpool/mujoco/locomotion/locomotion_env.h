// Copyright 2026 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ENVPOOL_MUJOCO_LOCOMOTION_LOCOMOTION_ENV_H_
#define ENVPOOL_MUJOCO_LOCOMOTION_LOCOMOTION_ENV_H_

#include <algorithm>
#include <cmath>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/locomotion/simulation.h"

namespace mujoco_locomotion {

struct LocomotionEnvFns {
  static auto DefaultConfig() {
    return MakeDict(
        "task_name"_.Bind(std::string("cmu_humanoid_run_walls")),
        "team_size"_.Bind(2), "time_limit"_.Bind(-1.0),
        "disable_walker_contacts"_.Bind(false), "enable_field_box"_.Bind(false),
        "keep_aspect_ratio"_.Bind(false), "terminate_on_goal"_.Bind(true));
  }

  template <typename Config>
  static auto StateSpec(const Config& config) {
    const auto layout =
        ObservationLayout(config["task_name"_], config["team_size"_]);
    // Composer exposes task-dependent dictionaries, including Soccer's
    // team-sized sensor sets. Transport typed, contiguous buffers through the
    // common queue; the Python API only creates named views of these buffers.
    return MakeDict("obs:continuous"_.Bind(Spec<double>(
                        {-1, std::max(1, StorageSize(layout, 0))})),
                    "obs:discrete"_.Bind(Spec<int64_t>(
                        {-1, std::max(1, StorageSize(layout, 1))})),
                    "obs:pixels"_.Bind(Spec<uint8_t>(
                        {-1, std::max(1, StorageSize(layout, 2))}, {0, 255})),
                    "reward64"_.Bind(Spec<double>({-1})),
                    "terminated"_.Bind(Spec<bool>({})));
  }

  template <typename Config>
  static auto ActionSpec(const Config& config) {
    const auto task = GetTaskConfig(config["task_name"_]);
    return MakeDict(
        "action"_.Bind(Spec<double>({-1, ActionSize(task.walker)}, {-1., 1.})));
  }
};

class LocomotionEnvSpec : public EnvSpec<LocomotionEnvFns> {
  using Base = EnvSpec<LocomotionEnvFns>;

 public:
  explicit LocomotionEnvSpec(const ConfigValues& values)
      : Base(Normalize(values)) {}
  LocomotionEnvSpec() : LocomotionEnvSpec(kDefaultConfig.AllValues()) {}

 private:
  static ConfigValues Normalize(const ConfigValues& values) {
    Config config(values);
    const auto task = GetTaskConfig(config["task_name"_]);
    const int team = config["team_size"_];
    if (team < 1 || team > 11) {
      throw std::invalid_argument("team_size must be between 1 and 11");
    }
    config["max_num_players"_] = task.task == Task::kSoccer ? 2 * team : 1;
    double& limit = config["time_limit"_];
    if (limit == -1) {
      limit = task.time_limit;
    }
    if (!std::isfinite(limit) || limit <= 0 ||
        config["max_episode_steps"_] <= 0) {
      throw std::invalid_argument(
          "time_limit and max_episode_steps must be positive");
    }
    return config.AllValues();
  }
};

class LocomotionEnv : public Env<LocomotionEnvSpec>, public RenderableEnv {
  using Base = Env<LocomotionEnvSpec>;

 public:
  LocomotionEnv(const Spec& spec, int env_id)
      : Base(spec, env_id), simulation_(MakeOptions(spec, seed_)) {}

  bool IsDone() override { return simulation_.Done(); }

  void Reset() override {
    std::scoped_lock lock(mutex_);
    simulation_.Reset();
    WriteState();
  }

  void Step(const Action& action) override {
    std::scoped_lock lock(mutex_);
    simulation_.Step(static_cast<const double*>(action["action"_].Data()));
    WriteState();
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 640, height > 0 ? height : 480};
  }

  void Render(int width, int height, int camera,
              unsigned char* output) override {
    std::scoped_lock lock(mutex_);
    if (simulation_.Model() == nullptr) {
      throw std::runtime_error("reset before rendering");
    }
    simulation_.Render(width, height, camera, output);
  }

#ifdef ENVPOOL_TEST
  // Test-only inspection uses the same lock as stepping/rendering. It never
  // changes the simulator or synchronizes either side of an oracle rollout.
  template <typename Function>
  auto Inspect(Function function) {
    std::scoped_lock lock(mutex_);
    if (simulation_.Model() == nullptr) {
      throw std::runtime_error("reset before inspecting physics");
    }
    return function(simulation_);
  }
#endif

 private:
  static Options MakeOptions(const Spec& spec, int seed) {
    const auto& config = spec.config;
    const std::string base = config["base_path"_] + "/mujoco/locomotion/";
    return {config["task_name"_],
            base + "assets_dm_control",
            base + "assets_labmaze",
            base + "assets_dm_control",
            seed,
            config["team_size"_],
            config["max_episode_steps"_],
            config["time_limit"_],
            config["disable_walker_contacts"_],
            config["enable_field_box"_],
            config["keep_aspect_ratio"_],
            config["terminate_on_goal"_]};
  }

  void WriteState() {
    State state = Allocate(simulation_.Players());
    state["obs:continuous"_] = 0;
    state["obs:discrete"_] = 0;
    state["obs:pixels"_] = 0;
    std::copy(simulation_.continuous.begin(), simulation_.continuous.end(),
              static_cast<double*>(state["obs:continuous"_].Data()));
    std::copy(simulation_.discrete.begin(), simulation_.discrete.end(),
              static_cast<int64_t*>(state["obs:discrete"_].Data()));
    std::copy(simulation_.pixels.begin(), simulation_.pixels.end(),
              static_cast<uint8_t*>(state["obs:pixels"_].Data()));
    for (int i = 0; i < simulation_.Players(); ++i) {
      state["reward"_][i] = simulation_.rewards[i];
      state["reward64"_][i] = simulation_.rewards[i];
      state["discount"_][i] = simulation_.discount;
    }
    state["terminated"_] = simulation_.Terminated();
    state["trunc"_] = simulation_.Truncated();
  }

  Simulation simulation_;
  std::mutex mutex_;
};

class LocomotionEnvPool : public AsyncEnvPool<LocomotionEnv> {
  using Base = AsyncEnvPool<LocomotionEnv>;

 public:
  using Base::Base;
  using Base::Send;

  void Send(const std::vector<Array>& actions) override {
    ValidateActions(actions);
    Base::Send(actions);
  }

  void Send(std::vector<Array>&& actions) override {
    ValidateActions(actions);
    Base::Send(std::move(actions));
  }
#ifdef ENVPOOL_TEST
  template <typename Function>
  auto Inspect(int env_id, Function function) {
    if (env_id < 0 || env_id >= static_cast<int>(envs_.size())) {
      throw std::out_of_range("invalid env_id");
    }
    return envs_[env_id]->Inspect(function);
  }
#endif

 private:
  void ValidateActions(const std::vector<Array>& actions) const {
    const int players = spec.config["max_num_players"_];
    const int width =
        ActionSize(GetTaskConfig(spec.config["task_name"_]).walker);
    // Validate on the caller before dispatching to workers: reading a missing
    // Soccer player's controls would otherwise go past the action buffer.
    if (actions.size() != 3 || actions[0].ndim != 1 || actions[1].ndim != 1 ||
        actions[2].ndim != 2 || actions[0].element_size != sizeof(int) ||
        actions[1].element_size != sizeof(int) ||
        actions[2].element_size != sizeof(double) ||
        actions[2].Shape(1) != static_cast<std::size_t>(width) ||
        actions[2].Shape(0) != actions[1].size ||
        actions[1].size != actions[0].size * players) {
      throw std::invalid_argument("expected one action for every player");
    }
    const auto* env_ids = static_cast<const int*>(actions[0].Data());
    const auto* player_ids = static_cast<const int*>(actions[1].Data());
    std::vector<int> counts(envs_.size(), 0);
    std::set<int> selected;
    for (std::size_t i = 0; i < actions[0].size; ++i) {
      const int id = env_ids[i];
      if (id < 0 || id >= static_cast<int>(envs_.size()) ||
          !selected.insert(id).second) {
        throw std::invalid_argument("invalid or duplicate env_id");
      }
    }
    for (std::size_t i = 0; i < actions[1].size; ++i) {
      const int id = player_ids[i];
      if ((selected.count(id) == 0u) || ++counts[id] > players) {
        throw std::invalid_argument("expected one action for every player");
      }
    }
  }
};

}  // namespace mujoco_locomotion

#endif  // ENVPOOL_MUJOCO_LOCOMOTION_LOCOMOTION_ENV_H_
