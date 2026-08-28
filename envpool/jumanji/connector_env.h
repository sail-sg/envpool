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

#ifndef ENVPOOL_JUMANJI_CONNECTOR_ENV_H_
#define ENVPOOL_JUMANJI_CONNECTOR_ENV_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace connector {

constexpr int kGridSize = 10;
constexpr int kNumAgents = 10;
constexpr int kTimeLimit = 50;
constexpr int kEmpty = 0;
constexpr std::array<std::array<int, 2>, 5> kMoves = {
    {{{0, 0}}, {{-1, 0}}, {{0, 1}}, {{1, 0}}, {{0, -1}}}};  // NOLINT

inline int Offset(int row, int col) { return row * kGridSize + col; }
inline int PathValue(int agent) { return 1 + 3 * agent; }
inline int PositionValue(int agent) { return 2 + 3 * agent; }
inline int TargetValue(int agent) { return 3 + 3 * agent; }

using Grid = std::array<int, kGridSize * kGridSize>;

inline Grid ParseGrid(const std::string& text) {
  Grid grid{};
  if (text.empty()) {
    return grid;
  }
  std::stringstream stream(text);
  std::string token;
  int index = 0;
  while (std::getline(stream, token, ',') && index < kGridSize * kGridSize) {
    grid[index++] = std::stoi(token);
  }
  return grid;
}

inline bool InGrid(int row, int col) {
  return 0 <= row && row < kGridSize && 0 <= col && col < kGridSize;
}

}  // namespace connector

class ConnectorEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("connector_grid"_.Bind(std::string("")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict(
        "obs:grid"_.Bind(Spec<int>({10, 10}, {0, 31})),
        "obs:action_mask"_.Bind(Spec<bool>({10, 5}, {false, true})),
        "obs:step_count"_.Bind(Spec<int>({}, {0, 50})),
        "info:agent_rewards"_.Bind(Spec<float>({connector::kNumAgents})),
        "info:num_connections"_.Bind(Spec<int>({}, {0, 10})),
        "info:ratio_connections"_.Bind(Spec<float>({}, {0.0f, 1.0f})),
        "info:total_path_length"_.Bind(Spec<int>({}, {0, 100})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1, 10}, {0, 4})));
  }
};

using ConnectorEnvSpec = EnvSpec<ConnectorEnvFns>;

class ConnectorEnv : public Env<ConnectorEnvSpec>, public RenderableEnv {
 protected:
  connector::Grid grid_{};
  connector::Grid configured_grid_{};
  std::array<int, connector::kNumAgents> row_{};
  std::array<int, connector::kNumAgents> col_{};
  std::array<int, connector::kNumAgents> target_row_{};
  std::array<int, connector::kNumAgents> target_col_{};
  std::array<bool, connector::kNumAgents> connected_{};
  std::array<float, connector::kNumAgents> agent_rewards_{};
  bool use_configured_grid_;
  int step_count_{0};
  bool done_{true};

 public:
  using Spec = ConnectorEnvSpec;
  using Action = typename Env<ConnectorEnvSpec>::Action;

  ConnectorEnv(const Spec& spec, int env_id)
      : Env<ConnectorEnvSpec>(spec, env_id),
        configured_grid_(connector::ParseGrid(spec.config["connector_grid"_])),
        use_configured_grid_(!spec.config["connector_grid"_].empty()) {}

  bool IsDone() override { return done_; }

  int CurrentMaxEpisodeSteps() const override {
    return connector::kTimeLimit + 1;
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, render::kWhite, rgb);
    for (int row = 0; row < connector::kGridSize; ++row) {
      for (int col = 0; col < connector::kGridSize; ++col) {
        const int value = grid_[connector::Offset(row, col)];
        if (value != connector::kEmpty) {
          const int agent = std::max(0, (value - 1) / 3);
          render::FillCell(width, height, connector::kGridSize,
                           connector::kGridSize, row, col,
                           render::Palette(agent), rgb, value % 3 == 1 ? 3 : 1);
        }
      }
    }
    render::DrawGrid(width, height, connector::kGridSize, connector::kGridSize,
                     {180, 180, 180}, rgb);
  }

  void Reset() override {
    connected_.fill(false);
    if (use_configured_grid_) {
      grid_ = configured_grid_;
      for (int agent = 0; agent < connector::kNumAgents; ++agent) {
        row_[agent] = agent;
        col_[agent] = 0;
        target_row_[agent] = agent;
        target_col_[agent] = connector::kGridSize - 1;
      }
      for (int row = 0; row < connector::kGridSize; ++row) {
        for (int col = 0; col < connector::kGridSize; ++col) {
          const int value = grid_[connector::Offset(row, col)];
          for (int agent = 0; agent < connector::kNumAgents; ++agent) {
            if (value == connector::PositionValue(agent)) {
              row_[agent] = row;
              col_[agent] = col;
            } else if (value == connector::TargetValue(agent)) {
              target_row_[agent] = row;
              target_col_[agent] = col;
            }
          }
        }
      }
    } else {
      GenerateBoard();
    }
    agent_rewards_.fill(0);
    for (int agent = 0; agent < connector::kNumAgents; ++agent) {
      connected_[agent] = row_[agent] == target_row_[agent] &&
                          col_[agent] == target_col_[agent];
    }
    step_count_ = 0;
    done_ = false;
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    const auto was_connected = connected_;
    auto next_row = row_;
    auto next_col = col_;
    for (int agent = 0; agent < connector::kNumAgents; ++agent) {
      const int move =
          std::clamp(static_cast<int>(action["action"_](0, agent)), 0, 4);
      const int row = row_[agent] + connector::kMoves[move][0];
      const int col = col_[agent] + connector::kMoves[move][1];
      if (move && IsValidPosition(agent, row, col)) {
        next_row[agent] = row;
        next_col[agent] = col;
      }
    }
    // The official simultaneous update gives a contested cell to the higher ID.
    const auto proposed_row = next_row;
    const auto proposed_col = next_col;
    for (int agent = 0; agent < connector::kNumAgents; ++agent) {
      for (int other = agent + 1; other < connector::kNumAgents; ++other) {
        if (proposed_row[agent] == proposed_row[other] &&
            proposed_col[agent] == proposed_col[other]) {
          next_row[agent] = row_[agent];
          next_col[agent] = col_[agent];
        }
      }
    }
    double reward = 0;
    for (int agent = 0; agent < connector::kNumAgents; ++agent) {
      if (row_[agent] != next_row[agent] || col_[agent] != next_col[agent]) {
        grid_[connector::Offset(row_[agent], col_[agent])] =
            connector::PathValue(agent);
        row_[agent] = next_row[agent];
        col_[agent] = next_col[agent];
        grid_[connector::Offset(row_[agent], col_[agent])] =
            connector::PositionValue(agent);
      }
      connected_[agent] = row_[agent] == target_row_[agent] &&
                          col_[agent] == target_col_[agent];
      agent_rewards_[agent] = was_connected[agent]
                                  ? 0.0f
                                  : -0.03f + (connected_[agent] ? 1.0f : 0.0f);
      reward += agent_rewards_[agent];
    }
    ++step_count_;
    done_ = step_count_ >= connector::kTimeLimit || AllConnectedOrBlocked();
    // Keep EnvPool's scalar reward; expose individual rewards in info as well.
    WriteState(static_cast<float>(reward / connector::kNumAgents));
  }

 private:
  std::vector<int> EmptyNeighbors(int row, int col) const {
    std::vector<int> neighbors;
    for (int move = 1; move < 5; ++move) {
      const int r = row + connector::kMoves[move][0];
      const int c = col + connector::kMoves[move][1];
      if (connector::InGrid(r, c) &&
          grid_[connector::Offset(r, c)] == connector::kEmpty) {
        neighbors.push_back(connector::Offset(r, c));
      }
    }
    return neighbors;
  }

  void GenerateBoard() {
    grid_.fill(connector::kEmpty);
    for (int agent = 0; agent < connector::kNumAgents; ++agent) {
      std::vector<int> starts;
      for (int cell = 0; cell < connector::kGridSize * connector::kGridSize;
           ++cell) {
        if (grid_[cell] == connector::kEmpty &&
            !EmptyNeighbors(cell / connector::kGridSize,
                            cell % connector::kGridSize)
                 .empty()) {
          starts.push_back(cell);
        }
      }
      const int start = starts[std::uniform_int_distribution<int>(
          0, static_cast<int>(starts.size()) - 1)(gen_)];
      row_[agent] = start / connector::kGridSize;
      col_[agent] = start % connector::kGridSize;
      grid_[start] = connector::PathValue(agent);
      const auto neighbors = EmptyNeighbors(row_[agent], col_[agent]);
      const int first = neighbors[std::uniform_int_distribution<int>(
          0, static_cast<int>(neighbors.size()) - 1)(gen_)];
      target_row_[agent] = first / connector::kGridSize;
      target_col_[agent] = first % connector::kGridSize;
      grid_[first] = connector::PositionValue(agent);
    }
    for (int step = 1; step < 15; ++step) {
      auto next_row = target_row_;
      auto next_col = target_col_;
      for (int agent = 0; agent < connector::kNumAgents; ++agent) {
        std::array<double, 4> weights{};
        for (int move = 1; move < 5; ++move) {
          const int row = target_row_[agent] + connector::kMoves[move][0];
          const int col = target_col_[agent] + connector::kMoves[move][1];
          if (connector::InGrid(row, col) &&
              grid_[connector::Offset(row, col)] == connector::kEmpty) {
            const int projection =
                connector::kMoves[move][0] *
                    (target_row_[agent] - row_[agent]) +
                connector::kMoves[move][1] * (target_col_[agent] - col_[agent]);
            weights[move - 1] = std::exp(static_cast<double>(projection));
          }
        }
        if (std::any_of(weights.begin(), weights.end(),
                        [](double weight) { return weight > 0; })) {
          const int move = 1 + std::discrete_distribution<int>(
                                   weights.begin(), weights.end())(gen_);
          next_row[agent] += connector::kMoves[move][0];
          next_col[agent] += connector::kMoves[move][1];
        }
      }
      const auto proposed_row = next_row;
      const auto proposed_col = next_col;
      for (int agent = 0; agent < connector::kNumAgents; ++agent) {
        for (int other = agent + 1; other < connector::kNumAgents; ++other) {
          if (proposed_row[agent] == proposed_row[other] &&
              proposed_col[agent] == proposed_col[other]) {
            next_row[agent] = target_row_[agent];
            next_col[agent] = target_col_[agent];
          }
        }
        grid_[connector::Offset(target_row_[agent], target_col_[agent])] =
            connector::PathValue(agent);
        grid_[connector::Offset(next_row[agent], next_col[agent])] =
            connector::PositionValue(agent);
      }
      target_row_ = next_row;
      target_col_ = next_col;
    }
    // Keep only endpoints; the sampled disjoint walks prove a solution exists.
    grid_.fill(connector::kEmpty);
    for (int agent = 0; agent < connector::kNumAgents; ++agent) {
      grid_[connector::Offset(row_[agent], col_[agent])] =
          connector::PositionValue(agent);
      grid_[connector::Offset(target_row_[agent], target_col_[agent])] =
          connector::TargetValue(agent);
    }
  }

  bool IsValidPosition(int agent, int row, int col) const {
    if (!connector::InGrid(row, col) || connected_[agent]) {
      return false;
    }
    const int value = grid_[connector::Offset(row, col)];
    return value == connector::kEmpty || value == connector::TargetValue(agent);
  }

  bool IsBlocked(int agent) const {
    if (connected_[agent]) {
      return true;
    }
    for (int action = 1; action < 5; ++action) {
      const int row = row_[agent] + connector::kMoves[action][0];
      const int col = col_[agent] + connector::kMoves[action][1];
      if (IsValidPosition(agent, row, col)) {
        return false;
      }
    }
    return true;
  }

  bool AllConnectedOrBlocked() const {
    for (int agent = 0; agent < connector::kNumAgents; ++agent) {
      if (!IsBlocked(agent)) {
        return false;
      }
    }
    return true;
  }

  int NumConnections() const {
    return static_cast<int>(
        std::count(connected_.begin(), connected_.end(), true));
  }

  int TotalPathLength() const {
    int total = connector::kNumAgents;
    for (int value : grid_) {
      if (value > 0 && (value - 1) % 3 == 0) {
        ++total;
      }
    }
    return total;
  }

  void WriteState(float reward) {
    auto state = Allocate();
    state["info:agent_rewards"_].Assign(agent_rewards_.data(),
                                        agent_rewards_.size());
    for (int row = 0; row < connector::kGridSize; ++row) {
      for (int col = 0; col < connector::kGridSize; ++col) {
        state["obs:grid"_](row, col) = grid_[connector::Offset(row, col)];
      }
    }
    for (int agent = 0; agent < connector::kNumAgents; ++agent) {
      state["obs:action_mask"_](agent, 0) = true;
      for (int action = 1; action < 5; ++action) {
        const int row = row_[agent] + connector::kMoves[action][0];
        const int col = col_[agent] + connector::kMoves[action][1];
        state["obs:action_mask"_](agent, action) =
            IsValidPosition(agent, row, col);
      }
    }
    const int num_connections = NumConnections();
    state["obs:step_count"_] = step_count_;
    state["info:num_connections"_] = num_connections;
    state["info:ratio_connections"_] =
        static_cast<float>(num_connections) / connector::kNumAgents;
    state["info:total_path_length"_] = TotalPathLength();
    state["reward"_] = reward;
  }
};

using ConnectorEnvPool = AsyncEnvPool<ConnectorEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_CONNECTOR_ENV_H_
