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

#ifndef ENVPOOL_JUMANJI_GRAPH_COLORING_ENV_H_
#define ENVPOOL_JUMANJI_GRAPH_COLORING_ENV_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <sstream>
#include <string>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace graph_coloring {

constexpr int kNumNodes = 20;
constexpr int kAdjSize = kNumNodes * kNumNodes;

using AdjMatrix = std::array<bool, kAdjSize>;
using Colors = std::array<int, kNumNodes>;
using ActionMask = std::array<bool, kNumNodes>;

inline int Offset(int row, int col) { return row * kNumNodes + col; }

inline AdjMatrix ParseEdges(const std::string& text) {
  AdjMatrix adj{};
  if (text.empty()) {
    return adj;
  }
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    const std::size_t sep = token.find('-');
    if (sep == std::string::npos) {
      continue;
    }
    const int a = std::stoi(token.substr(0, sep));
    const int b = std::stoi(token.substr(sep + 1));
    if (0 <= a && a < kNumNodes && 0 <= b && b < kNumNodes && a != b) {
      adj[Offset(a, b)] = true;
      adj[Offset(b, a)] = true;
    }
  }
  return adj;
}

inline ActionMask ValidActions(const AdjMatrix& adj, const Colors& colors,
                               int node) {
  ActionMask mask{};
  mask.fill(true);
  for (int other = 0; other < kNumNodes; ++other) {
    if (adj[Offset(node, other)] && colors[other] >= 0) {
      mask[colors[other]] = false;
    }
  }
  return mask;
}

inline int NumUniqueColors(const Colors& colors) {
  std::array<bool, kNumNodes> used{};
  for (int color : colors) {
    if (0 <= color && color < kNumNodes) {
      used[color] = true;
    }
  }
  return static_cast<int>(std::count(used.begin(), used.end(), true));
}

}  // namespace graph_coloring

class GraphColoringEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("graph_coloring_edges"_.Bind(std::string("")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("obs:adj_matrix"_.Bind(Spec<bool>({20, 20}, {false, true})),
                    "obs:action_mask"_.Bind(Spec<bool>({20}, {false, true})),
                    "obs:colors"_.Bind(Spec<int>({20}, {-1, 19})),
                    "obs:current_node_index"_.Bind(Spec<int>({}, {0, 19})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 19})));
  }
};

using GraphColoringEnvSpec = EnvSpec<GraphColoringEnvFns>;

class GraphColoringEnv : public Env<GraphColoringEnvSpec>,
                         public RenderableEnv {
 protected:
  graph_coloring::AdjMatrix adj_{};
  graph_coloring::AdjMatrix configured_adj_{};
  graph_coloring::Colors colors_{};
  graph_coloring::ActionMask action_mask_{};
  bool use_configured_adj_;
  int current_node_{0};
  bool done_{true};

 public:
  using Spec = GraphColoringEnvSpec;
  using Action = typename Env<GraphColoringEnvSpec>::Action;

  GraphColoringEnv(const Spec& spec, int env_id)
      : Env<GraphColoringEnvSpec>(spec, env_id),
        configured_adj_(
            graph_coloring::ParseEdges(spec.config["graph_coloring_edges"_])),
        use_configured_adj_(!spec.config["graph_coloring_edges"_].empty()) {}

  bool IsDone() override { return done_; }

  int CurrentMaxEpisodeSteps() const override {
    return graph_coloring::kNumNodes + 1;
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, render::kWhite, rgb);
    std::array<int, graph_coloring::kNumNodes> xs{};
    std::array<int, graph_coloring::kNumNodes> ys{};
    const int radius = std::min(width, height) * 38 / 100;
    const int cx = width / 2;
    const int cy = height / 2;
    for (int node = 0; node < graph_coloring::kNumNodes; ++node) {
      const float angle = 6.283185307179586f * node / graph_coloring::kNumNodes;
      xs[node] = cx + static_cast<int>(std::cos(angle) * radius);
      ys[node] = cy + static_cast<int>(std::sin(angle) * radius);
    }
    for (int from = 0; from < graph_coloring::kNumNodes; ++from) {
      for (int to = 0; to < from; ++to) {
        if (adj_[graph_coloring::Offset(from, to)]) {
          render::DrawLine(width, height, xs[from], ys[from], xs[to], ys[to],
                           {205, 205, 205}, rgb);
        }
      }
    }
    for (int node = 0; node < graph_coloring::kNumNodes; ++node) {
      const render::Color fill =
          colors_[node] >= 0 ? render::Palette(colors_[node]) : render::kWhite;
      render::FillCircle(width, height, xs[node], ys[node], 5, fill, rgb);
      render::StrokeCircle(
          width, height, xs[node], ys[node], node == current_node_ ? 7 : 5,
          node == current_node_ ? render::Palette(3) : render::kBlack, rgb);
    }
  }

  void Reset() override {
    colors_.fill(-1);
    current_node_ = 0;
    done_ = false;
    if (use_configured_adj_) {
      adj_ = configured_adj_;
    } else {
      GenerateRandomGraph();
    }
    action_mask_ = graph_coloring::ValidActions(adj_, colors_, current_node_);
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    const int color = std::clamp(static_cast<int>(action["action"_]), 0,
                                 graph_coloring::kNumNodes - 1);
    const bool invalid = !action_mask_[color];
    colors_[current_node_] = color;
    const bool all_colored =
        std::all_of(colors_.begin(), colors_.end(),
                    [](int assigned) { return assigned >= 0; });
    float reward = 0.0f;
    if (all_colored) {
      reward = -static_cast<float>(graph_coloring::NumUniqueColors(colors_));
    }
    if (invalid) {
      reward = -static_cast<float>(graph_coloring::kNumNodes);
    }
    done_ = invalid || all_colored;
    current_node_ = (current_node_ + 1) % graph_coloring::kNumNodes;
    action_mask_ = graph_coloring::ValidActions(adj_, colors_, current_node_);
    WriteState(reward);
  }

 private:
  void GenerateRandomGraph() {
    adj_.fill(false);
    std::bernoulli_distribution edge_dist(0.8);
    for (int row = 0; row < graph_coloring::kNumNodes; ++row) {
      for (int col = 0; col < row; ++col) {
        const bool edge = edge_dist(gen_);
        adj_[graph_coloring::Offset(row, col)] = edge;
        const int transpose_row = col;
        const int transpose_col = row;
        adj_[graph_coloring::Offset(transpose_row, transpose_col)] = edge;
      }
    }
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int row = 0; row < graph_coloring::kNumNodes; ++row) {
      state["obs:colors"_][row] = colors_[row];
      for (int col = 0; col < graph_coloring::kNumNodes; ++col) {
        state["obs:adj_matrix"_](row, col) =
            adj_[graph_coloring::Offset(row, col)];
      }
    }
    for (int color = 0; color < graph_coloring::kNumNodes; ++color) {
      state["obs:action_mask"_][color] = action_mask_[color];
    }
    state["obs:current_node_index"_] = current_node_;
    state["reward"_] = reward;
  }
};

using GraphColoringEnvPool = AsyncEnvPool<GraphColoringEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_GRAPH_COLORING_ENV_H_
