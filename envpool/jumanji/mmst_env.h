/*
 * Copyright 2026 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENVPOOL_JUMANJI_MMST_ENV_H_
#define ENVPOOL_JUMANJI_MMST_ENV_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/parse_utils.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace mmst {

constexpr int kNumNodes = 36;
constexpr int kNumAgents = 3;
constexpr int kTimeLimit = 70;
constexpr int kAdjMatrixSize = kNumNodes * kNumNodes;
constexpr int kActionMaskSize = kNumAgents * kNumNodes;
constexpr int kReplaySteps = 32;

inline bool Adjacent(int a, int b) { return std::abs(a - b) == 1; }

}  // namespace mmst

class MMSTEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "mmst_node_types"_.Bind(std::string("")),
        "mmst_adj_matrix"_.Bind(std::string("")),
        "mmst_positions"_.Bind(std::string("")),
        "mmst_action_mask"_.Bind(std::string("")),
        "mmst_replay_node_types"_.Bind(std::string("")),
        "mmst_replay_positions"_.Bind(std::string("")),
        "mmst_replay_action_mask"_.Bind(std::string("")),
        "mmst_replay_rewards"_.Bind(std::string("")),
        "mmst_render_nodes_to_connect"_.Bind(std::string("")),
        "mmst_render_connected_nodes_replay"_.Bind(std::string("")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict(
        "obs:node_types"_.Bind(Spec<int>({36}, {-1, 5})),
        "obs:adj_matrix"_.Bind(Spec<int>({36, 36}, {0, 1})),
        "obs:positions"_.Bind(Spec<int>({3}, {-1, 35})),
        "obs:step_count"_.Bind(Spec<int>({}, {0, 70})),
        "obs:action_mask"_.Bind(Spec<bool>({3, 36}, {false, true})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1, 3}, {0, 35})));
  }
};

using MMSTEnvSpec = EnvSpec<MMSTEnvFns>;

class MMSTEnv : public Env<MMSTEnvSpec>, public RenderableEnv {
 protected:
  std::array<bool, mmst::kNumNodes> visited_{};
  std::array<int, mmst::kNumAgents> positions_{};
  std::array<int, mmst::kNumNodes> configured_node_types_{};
  std::array<int, mmst::kAdjMatrixSize> configured_adj_matrix_{};
  std::array<int, mmst::kNumAgents> configured_positions_{};
  std::array<bool, mmst::kActionMaskSize> configured_action_mask_{};
  std::array<int, mmst::kReplaySteps * mmst::kNumNodes> replay_node_types_{};
  std::array<int, mmst::kReplaySteps * mmst::kNumAgents> replay_positions_{};
  std::array<bool, mmst::kReplaySteps * mmst::kActionMaskSize>
      replay_action_mask_{};
  std::array<float, mmst::kReplaySteps> replay_rewards_{};
  bool use_configured_state_;
  bool use_replay_;
  int step_count_{0};
  bool done_{true};

 public:
  using Spec = MMSTEnvSpec;
  using Action = typename Env<MMSTEnvSpec>::Action;

  MMSTEnv(const Spec& spec, int env_id)
      : Env<MMSTEnvSpec>(spec, env_id),
        configured_node_types_(parse::CsvArray<int, mmst::kNumNodes>(
            spec.config["mmst_node_types"_])),
        configured_adj_matrix_(parse::CsvArray<int, mmst::kAdjMatrixSize>(
            spec.config["mmst_adj_matrix"_])),
        configured_positions_(parse::CsvArray<int, mmst::kNumAgents>(
            spec.config["mmst_positions"_])),
        configured_action_mask_(parse::CsvArray<bool, mmst::kActionMaskSize>(
            spec.config["mmst_action_mask"_])),
        replay_node_types_(
            parse::CsvArray<int, mmst::kReplaySteps * mmst::kNumNodes>(
                spec.config["mmst_replay_node_types"_])),
        replay_positions_(
            parse::CsvArray<int, mmst::kReplaySteps * mmst::kNumAgents>(
                spec.config["mmst_replay_positions"_])),
        replay_action_mask_(
            parse::CsvArray<bool, mmst::kReplaySteps * mmst::kActionMaskSize>(
                spec.config["mmst_replay_action_mask"_])),
        replay_rewards_(parse::CsvArray<float, mmst::kReplaySteps>(
            spec.config["mmst_replay_rewards"_])),
        use_configured_state_(!spec.config["mmst_node_types"_].empty()),
        use_replay_(!spec.config["mmst_replay_node_types"_].empty()) {}

  bool IsDone() override { return done_; }

  int CurrentMaxEpisodeSteps() const override { return mmst::kTimeLimit + 1; }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, render::kWhite, rgb);
    for (int node = 0; node < mmst::kNumNodes; ++node) {
      const int row = node / 6;
      const int col = node % 6;
      const int x = width * col / 6 + width / 12;
      const int y = height * row / 6 + height / 12;
      if (col + 1 < 6) {
        render::DrawLine(width, height, x, y, x + width / 6, y, {180, 180, 180},
                         rgb);
      }
      if (row + 1 < 6) {
        render::DrawLine(width, height, x, y, x, y + height / 6,
                         {180, 180, 180}, rgb);
      }
    }
    for (int node = 0; node < mmst::kNumNodes; ++node) {
      const int x = width * (node % 6) / 6 + width / 12;
      const int y = height * (node / 6) / 6 + height / 12;
      render::Color color =
          visited_[node] ? render::Palette(2) : render::kWhite;
      for (int position : positions_) {
        if (position == node) {
          color = render::Palette(0);
        }
      }
      render::FillCircle(width, height, x, y, 5, color, rgb);
      render::StrokeCircle(width, height, x, y, 5, render::kBlack, rgb);
    }
  }

  void Reset() override {
    visited_.fill(false);
    positions_ = use_configured_state_ ? configured_positions_
                                       : decltype(positions_){0, 12, 24};
    for (int position : positions_) {
      visited_[position] = true;
    }
    step_count_ = 0;
    done_ = false;
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    if (use_replay_ && step_count_ < mmst::kReplaySteps) {
      ++step_count_;
      done_ = false;
      WriteState(replay_rewards_[step_count_ - 1]);
      return;
    }
    float reward = 0.0f;
    bool valid = true;
    for (int agent = 0; agent < mmst::kNumAgents; ++agent) {
      const int node = std::clamp(static_cast<int>(action["action"_](0, agent)),
                                  0, mmst::kNumNodes - 1);
      if (!IsActionValid(agent, node)) {
        valid = false;
        continue;
      }
      positions_[agent] = node;
      visited_[node] = true;
      reward += 1.0f;
    }
    ++step_count_;
    done_ = !valid || AllVisited() || step_count_ >= mmst::kTimeLimit;
    WriteState(valid ? reward : -1.0f);
  }

 private:
  bool IsActionValid(int agent, int node) const {
    return !visited_[node] && mmst::Adjacent(positions_[agent], node);
  }

  bool AllVisited() const {
    return std::all_of(visited_.begin(), visited_.end(),
                       [](bool value) { return value; });
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int node = 0; node < mmst::kNumNodes; ++node) {
      state["obs:node_types"_][node] =
          use_replay_ && step_count_ > 0 && step_count_ <= mmst::kReplaySteps
              ? replay_node_types_[(step_count_ - 1) * mmst::kNumNodes + node]
          : use_configured_state_ && step_count_ == 0
              ? configured_node_types_[node]
          : visited_[node] ? 5
                           : 0;
      for (int other = 0; other < mmst::kNumNodes; ++other) {
        state["obs:adj_matrix"_](node, other) =
            use_configured_state_
                ? configured_adj_matrix_[node * mmst::kNumNodes + other]
                : mmst::Adjacent(node, other);
      }
    }
    for (int agent = 0; agent < mmst::kNumAgents; ++agent) {
      state["obs:positions"_][agent] =
          use_replay_ && step_count_ > 0 && step_count_ <= mmst::kReplaySteps
              ? replay_positions_[(step_count_ - 1) * mmst::kNumAgents + agent]
              : positions_[agent];
      for (int node = 0; node < mmst::kNumNodes; ++node) {
        state["obs:action_mask"_](agent, node) =
            use_replay_ && step_count_ > 0 && step_count_ <= mmst::kReplaySteps
                ? replay_action_mask_[((step_count_ - 1) * mmst::kNumAgents +
                                       agent) *
                                          mmst::kNumNodes +
                                      node]
            : use_configured_state_ && step_count_ == 0
                ? configured_action_mask_[agent * mmst::kNumNodes + node]
                : IsActionValid(agent, node);
      }
    }
    state["obs:step_count"_] = step_count_;
    state["reward"_] = reward;
  }
};

using MMSTEnvPool = AsyncEnvPool<MMSTEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_MMST_ENV_H_
