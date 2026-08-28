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
#include <numeric>
#include <random>
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
  std::array<int, mmst::kNumNodes> node_types_{};
  std::array<int, mmst::kAdjMatrixSize> adjacency_{};
  std::array<std::array<bool, mmst::kNumNodes>, mmst::kNumAgents> visited_{};
  std::array<std::array<bool, mmst::kNumNodes>, mmst::kNumAgents> blocked_{};
  std::array<bool, mmst::kNumAgents> finished_{};
  std::array<bool, mmst::kNumAgents> mask_finished_{};
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
      const int x = width * (node % 6) / 6 + width / 12;
      const int y = height * (node / 6) / 6 + height / 12;
      for (int other = node + 1; other < mmst::kNumNodes; ++other) {
        if (adjacency_[node * mmst::kNumNodes + other]) {
          render::DrawLine(
              width, height, x, y, width * (other % 6) / 6 + width / 12,
              height * (other / 6) / 6 + height / 12, {180, 180, 180}, rgb);
        }
      }
    }
    for (int node = 0; node < mmst::kNumNodes; ++node) {
      const int x = width * (node % 6) / 6 + width / 12;
      const int y = height * (node / 6) / 6 + height / 12;
      render::Color color = node_types_[node] < 0
                                ? render::kWhite
                                : render::Palette(node_types_[node]);
      for (int agent = 0; agent < mmst::kNumAgents; ++agent) {
        if (visited_[agent][node]) {
          color = render::Palette(agent);
        }
      }
      render::FillCircle(width, height, x, y, 5, color, rgb);
      render::StrokeCircle(width, height, x, y, 5, render::kBlack, rgb);
    }
  }

  void Reset() override {
    for (auto& visits : visited_) {
      visits.fill(false);
    }
    for (auto& blocks : blocked_) {
      blocks.fill(false);
    }
    finished_.fill(false);
    mask_finished_.fill(false);
    if (use_configured_state_) {
      adjacency_ = configured_adj_matrix_;
      positions_ = configured_positions_;
      for (int node = 0; node < mmst::kNumNodes; ++node) {
        const int value = configured_node_types_[node];
        node_types_[node] = value < 0 ? -1 : value / 2;
      }
    } else {
      GenerateGraph();
      node_types_.fill(-1);
      for (int agent = 0; agent < mmst::kNumAgents; ++agent) {
        std::array<int, 12> nodes{};
        std::iota(nodes.begin(), nodes.end(), agent * 12);
        std::shuffle(nodes.begin(), nodes.end(), gen_);
        positions_[agent] = nodes[0];
        for (int i = 0; i < 4; ++i) {
          node_types_[nodes[i]] = agent;
        }
      }
    }
    for (int agent = 0; agent < mmst::kNumAgents; ++agent) {
      visited_[agent][positions_[agent]] = true;
    }
    UpdateBlockedNodes();
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
    std::array<int, mmst::kNumAgents> nodes{};
    std::array<int, mmst::kNumAgents> choices{};
    std::array<int, mmst::kNumAgents> order{};
    std::array<bool, mmst::kNumNodes> selected{};
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), gen_);
    for (int agent : order) {
      const int node = std::clamp(static_cast<int>(action["action"_](0, agent)),
                                  0, mmst::kNumNodes - 1);
      nodes[agent] = HasEdge(agent, node) ? node : -1;
      choices[agent] = nodes[agent] < 0 ? -1 : selected[node] ? -2 : node;
      if (choices[agent] >= 0) {
        selected[node] = true;
      }
      // The pinned implementation marks revisits after resolving ties.
      if (visited_[agent][nodes[agent] < 0 ? mmst::kNumNodes - 1 : node]) {
        choices[agent] = -3;
      }
      if (finished_[agent]) {
        choices[agent] = -1;
      }
    }
    float reward = 0.0f;
    for (int agent = 0; agent < mmst::kNumAgents; ++agent) {
      const int choice = choices[agent];
      if (choice != -1 && choice != -2 && nodes[agent] >= 0) {
        positions_[agent] = nodes[agent];
        visited_[agent][nodes[agent]] = true;
      }
      if (!finished_[agent]) {
        reward += choice == -2 ? 0.0f
                  : choice >= 0 && node_types_[positions_[agent]] == agent
                      ? 10.0f
                  : choice == -1 ? -2.0f
                                 : -1.0f;
      }
    }
    UpdateBlockedNodes();
    mask_finished_ = finished_;
    for (int agent = 0; agent < mmst::kNumAgents; ++agent) {
      finished_[agent] = true;
      for (int node = 0; node < mmst::kNumNodes; ++node) {
        if (node_types_[node] == agent && !visited_[agent][node]) {
          finished_[agent] = false;
        }
      }
    }
    ++step_count_;
    done_ = std::all_of(finished_.begin(), finished_.end(),
                        [](bool value) { return value; }) ||
            step_count_ >= mmst::kTimeLimit;
    WriteState(reward);
  }

 private:
  void GenerateGraph() {
    adjacency_.fill(0);
    std::array<bool, mmst::kAdjMatrixSize> directed_edges{};
    std::array<int, mmst::kNumNodes> degree{};
    int edges = 0;
    auto add_edge = [&](int a, int b) {
      const int index = a * mmst::kNumNodes + b;
      // Jumanji 1.1.2 counts ordered edges and permits degree max_degree + 1.
      if (a == b || directed_edges[index] || degree[a] > 5 || degree[b] > 5) {
        return false;
      }
      directed_edges[index] = true;
      adjacency_[index] = adjacency_[b * mmst::kNumNodes + a] = 1;
      ++degree[a];
      ++degree[b];
      ++edges;
      return true;
    };
    auto random_node = [&](int first, int last) {
      return std::uniform_int_distribution<int>(first, last - 1)(gen_);
    };
    auto fill_edges = [&](int first, int last, int total) {
      while (edges < total) {
        const int a = random_node(first, last);
        int b = random_node(first, last - 1);
        b += b >= a ? 1 : 0;
        add_edge(a, b);
      }
    };
    for (int group = 0; group < mmst::kNumAgents; ++group) {
      const int offset = group * 12;
      std::array<bool, 12> reached{};
      int current = random_node(offset, offset + 11);
      reached[current - offset] = true;
      int count = 1;
      while (count < 12) {
        const int next = random_node(offset, offset + 12);
        if (!reached[next - offset] &&
            add_edge(std::min(current, next), std::max(current, next))) {
          reached[next - offset] = true;
          ++count;
        }
        current = next;
      }
      fill_edges(offset, offset + 12, (group + 1) * 12);
    }
    // Each subgraph has 12 edges; merge with 12, then 24 additional edges.
    const int first = random_node(0, 12);
    const int second = random_node(12, 24);
    add_edge(first, second);
    fill_edges(0, 24, 48);
    const int merged = random_node(0, 24);
    const int third = random_node(24, 36);
    add_edge(merged, third);
    fill_edges(0, 36, 72);
  }

  bool HasEdge(int agent, int node) const {
    return adjacency_[positions_[agent] * mmst::kNumNodes + node] &&
           !blocked_[agent][node];
  }

  bool IsActionValid(int agent, int node) const {
    return HasEdge(agent, node) && !mask_finished_[agent];
  }

  void UpdateBlockedNodes() {
    for (int agent = 0; agent < mmst::kNumAgents; ++agent) {
      const int node = positions_[agent];
      if (node_types_[node] < 0) {
        for (int other = 0; other < mmst::kNumAgents; ++other) {
          if (other != agent) {
            blocked_[other][node] = true;
          }
        }
      }
    }
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int node = 0; node < mmst::kNumNodes; ++node) {
      if (use_replay_ && step_count_ > 0 && step_count_ <= mmst::kReplaySteps) {
        state["obs:node_types"_][node] =
            replay_node_types_[(step_count_ - 1) * mmst::kNumNodes + node];
      } else if (use_configured_state_ && step_count_ == 0) {
        state["obs:node_types"_][node] = configured_node_types_[node];
      } else {
        int type = node_types_[node] < 0 ? -1 : 2 * node_types_[node] + 1;
        for (int agent = 0; agent < mmst::kNumAgents; ++agent) {
          if (visited_[agent][node]) {
            type = 2 * agent;
          }
        }
        state["obs:node_types"_][node] = type;
      }
      for (int other = 0; other < mmst::kNumNodes; ++other) {
        state["obs:adj_matrix"_](node, other) =
            adjacency_[node * mmst::kNumNodes + other];
      }
    }
    for (int agent = 0; agent < mmst::kNumAgents; ++agent) {
      state["obs:positions"_][agent] =
          use_replay_ && step_count_ > 0 && step_count_ <= mmst::kReplaySteps
              ? replay_positions_[(step_count_ - 1) * mmst::kNumAgents + agent]
              : positions_[agent];
      for (int node = 0; node < mmst::kNumNodes; ++node) {
        if (use_replay_ && step_count_ > 0 &&
            step_count_ <= mmst::kReplaySteps) {
          state["obs:action_mask"_](agent, node) =
              replay_action_mask_[((step_count_ - 1) * mmst::kNumAgents +
                                   agent) *
                                      mmst::kNumNodes +
                                  node];
        } else if (use_configured_state_ && step_count_ == 0 &&
                   !spec_.config["mmst_action_mask"_].empty()) {
          state["obs:action_mask"_](agent, node) =
              configured_action_mask_[agent * mmst::kNumNodes + node];
        } else {
          state["obs:action_mask"_](agent, node) = IsActionValid(agent, node);
        }
      }
    }
    state["obs:step_count"_] = step_count_;
    state["reward"_] = reward;
  }
};

using MMSTEnvPool = AsyncEnvPool<MMSTEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_MMST_ENV_H_
