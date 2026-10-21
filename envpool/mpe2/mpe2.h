// Copyright 2026 Garena Online Private Limited
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ENVPOOL_MPE2_MPE2_H_
#define ENVPOOL_MPE2_MPE2_H_

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mpe2/model.h"

namespace mpe2 {

template <typename Config>
Options ReadOptions(const Config& c) {
  const TaskInfo* task = nullptr;
  for (const auto& t : kTasks) {
    if (c["task"_] == t.name) {
      task = &t;
    }
  }
  if (task == nullptr) {
    throw std::invalid_argument("Unknown MPE2 task: " + c["task"_]);
  }
  auto resolve = [task](int value, int field) {
    return value == -1 ? task->defaults[field] : value;
  };
  Options o{task->task,
            resolve(c["N"_], 0),
            resolve(c["num_good"_], 1),
            resolve(c["num_adversaries"_], 2),
            resolve(c["num_obstacles"_], 3),
            resolve(c["num_food"_], 4),
            resolve(c["num_forests"_], 5),
            resolve(c["num_collectors"_], 6),
            resolve(c["num_deposits"_], 7),
            resolve(c["num_treasures"_], 8)};
  o.agent_neighbors = c["num_agent_neighbors"_];
  o.landmark_neighbors = c["num_landmark_neighbors"_];
  o.local_ratio = c["local_ratio"_];
  o.continuous = c["continuous_actions"_];
  o.rescaling = c["dynamic_rescaling"_];
  o.curriculum = c["curriculum"_];
  o.terminate_on_success = c["terminate_on_success"_];
  o.benchmark = c["benchmark_data"_];
  return o;
}

template <bool Continuous>
class MPE2EnvFns {
 public:
  static auto DefaultConfig() {
    return MakeDict(
        "task"_.Bind(std::string("simple")), "N"_.Bind(-1),
        "num_good"_.Bind(-1), "num_adversaries"_.Bind(-1),
        "num_obstacles"_.Bind(-1), "num_food"_.Bind(-1),
        "num_forests"_.Bind(-1), "num_collectors"_.Bind(-1),
        "num_deposits"_.Bind(-1), "num_treasures"_.Bind(-1),
        "num_agent_neighbors"_.Bind(-1), "num_landmark_neighbors"_.Bind(-1),
        "local_ratio"_.Bind(0.5), "continuous_actions"_.Bind(Continuous),
        "dynamic_rescaling"_.Bind(true), "benchmark_data"_.Bind(false),
        "curriculum"_.Bind(false), "terminate_on_success"_.Bind(false));
  }
  template <typename Config>
  static auto StateSpec(const Config& c) {
    // TODO(core): represent heterogeneous per-player spaces directly. Until
    // then, padding and explicit role lengths preserve the upstream contract.
    auto o = ReadOptions(c);
    int full_size = 0;
    for (int p = 0; p < o.Players(); ++p) {
      full_size += o.ObservationSize(p);
    }
    auto common = MakeDict(
        "obs"_.Bind(Spec<float>({-1, o.MaxObservationSize()})),
        "info:state"_.Bind(Spec<float>({full_size})),
        "info:players.id"_.Bind(Spec<int>({-1})),
        "info:players.obs_size"_.Bind(Spec<int>({-1})),
        "info:players.action_size"_.Bind(Spec<int>({-1})),
        "info:players.action_mask"_.Bind(Spec<bool>({-1, o.MaxActionSize()})),
        "info:players.reward"_.Bind(Spec<double>({-1})),
        "info:players.benchmark"_.Bind(
            Spec<double>({-1, std::max(1, o.BenchmarkSize())})),
        "info:players.benchmark_size"_.Bind(Spec<int>({-1})));
#ifdef ENVPOOL_TEST
    return ConcatDict(
        common, MakeDict("info:snapshot"_.Bind(Spec<double>(
                    {(o.Players() + o.Landmarks()) * (10 + o.Communication()) +
                     3 * o.Players() + 2}))));
#else
    return common;
#endif
  }
  template <typename Config>
  static auto ActionSpec(const Config& c) {
    auto o = ReadOptions(c);
    if constexpr (Continuous) {
      return MakeDict(
          "players.action"_.Bind(Spec<float>({-1, o.MaxActionSize()}, {0, 1})));
    } else {
      return MakeDict(
          "players.action"_.Bind(Spec<int>({-1}, {0, o.MaxActionSize() - 1})));
    }
  }
};

template <bool Continuous>
class MPE2EnvSpec : public EnvSpec<MPE2EnvFns<Continuous>> {
  using Base = EnvSpec<MPE2EnvFns<Continuous>>;
  static typename Base::ConfigValues Normalize(
      const typename Base::ConfigValues& values) {
    typename Base::Config c(values);
    Options o = ReadOptions(c);
    o.Validate();
    if (o.continuous != Continuous || c["max_episode_steps"_] < 1) {
      throw std::invalid_argument("Invalid MPE2 action mode or episode length");
    }
    c["N"_] = o.n;
    c["num_good"_] = o.good;
    c["num_adversaries"_] = o.adversaries;
    c["num_obstacles"_] = o.obstacles;
    c["num_food"_] = o.food;
    c["num_forests"_] = o.forests;
    c["num_collectors"_] = o.collectors;
    c["num_deposits"_] = o.deposits;
    c["num_treasures"_] = o.treasures;
    if (c["max_num_players"_] == 1) {
      c["max_num_players"_] = o.Players();
    }
    if (c["max_num_players"_] < o.Players()) {
      throw std::invalid_argument(
          "max_num_players is smaller than the MPE2 population");
    }
    return c.AllValues();
  }

 public:
  const Options kOptions;
  explicit MPE2EnvSpec(const typename Base::ConfigValues& values)
      : Base(Normalize(values)), kOptions(ReadOptions(this->config)) {}
};

template <bool Continuous>
class MPE2Env : public Env<MPE2EnvSpec<Continuous>>, public RenderableEnv {
  using Base = Env<MPE2EnvSpec<Continuous>>;

 public:
  using typename Base::Action;
  using typename Base::Spec;
  MPE2Env(const Spec& spec, int env_id)
      : Base(spec, env_id),
        model_(spec.kOptions, this->seed_),
        kLimit(spec.config["max_episode_steps"_]) {}
  bool IsDone() override { return done_; }
  void Reset() override {
    model_.Reset();
    steps_ = 0;
    done_ = false;
    WriteState();
  }
  void Step(const Action& action) override {
    int p = model_.kOptions.Players();
    CHECK_EQ(action["players.action"_].Shape(0), p);
    std::vector<std::vector<double>> actions(p);
    for (int i = 0; i < p; ++i) {
      if constexpr (Continuous) {
        for (int j = 0; j < model_.kOptions.ActionSize(i); ++j) {
          actions[i].push_back(std::clamp(
              static_cast<double>(action["players.action"_](i, j)), 0.0, 1.0));
        }
      } else {
        int a = action["players.action"_][i];
        CHECK_GE(a, 0);
        CHECK_LT(a, model_.kOptions.ActionSize(i));
        actions[i].push_back(a);
      }
    }
    model_.Step(actions);
    ++steps_;
    done_ = steps_ >= kLimit || model_.Terminal();
    WriteState();
  }
  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 700, height > 0 ? height : 700};
  }
  void Render(int width, int height, int, unsigned char* rgb) override {
    auto size = RenderSize(width, height);
    mpe2::Render(model_, size.first, size.second, rgb);
  }

 private:
  Model model_;
  const int kLimit;
  int steps_{0};
  bool done_{true};
  void WriteState() {
    auto state = this->Allocate(model_.kOptions.Players());
    state["obs"_] = 0;
    state["info:players.benchmark"_] = 0;
    state["info:players.action_mask"_] = false;
    int offset = 0;
    for (int i = 0; i < model_.kOptions.Players(); ++i) {
      auto obs = model_.Observation(i);
      auto benchmark = steps_ > 0 ? model_.Benchmark(i) : std::vector<double>{};
      state["info:players.id"_][i] = i;
      state["info:players.obs_size"_][i] = obs.size();
      state["info:players.action_size"_][i] = model_.kOptions.ActionSize(i);
      for (std::size_t j = 0; j < obs.size(); ++j) {
        state["obs"_](i, j) = static_cast<float>(obs[j]);
        state["info:state"_][offset++] = static_cast<float>(obs[j]);
      }
      for (int j = 0; j < model_.kOptions.ActionSize(i); ++j) {
        state["info:players.action_mask"_](i, j) = true;
      }
      state["reward"_][i] = static_cast<float>(model_.rewards[i]);
      state["info:players.reward"_][i] = model_.rewards[i];
      state["info:players.benchmark_size"_][i] = benchmark.size();
      for (std::size_t j = 0; j < benchmark.size(); ++j) {
        state["info:players.benchmark"_](i, j) = benchmark[j];
      }
    }
#ifdef ENVPOOL_TEST
    auto snapshot = model_.Snapshot();
    std::copy(snapshot.begin(), snapshot.end(),
              static_cast<double*>(state["info:snapshot"_].Data()));
#endif
  }
};

}  // namespace mpe2
#endif  // ENVPOOL_MPE2_MPE2_H_
