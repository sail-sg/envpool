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

#ifndef ENVPOOL_JUMANJI_CLEANER_ENV_H_
#define ENVPOOL_JUMANJI_CLEANER_ENV_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace cleaner {

constexpr int kRows = 10;
constexpr int kCols = 10;
constexpr int kCellCount = kRows * kCols;
constexpr int kNumAgents = 3;
constexpr int kTimeLimit = 100;
constexpr std::int8_t kDirty = 0;
constexpr std::int8_t kClean = 1;
constexpr std::int8_t kWall = 2;
constexpr std::array<std::array<int, 2>, 4> kMoves = {
    {{{-1, 0}}, {{0, 1}}, {{1, 0}}, {{0, -1}}}};  // NOLINT

using Grid = std::array<std::int8_t, kCellCount>;
using AgentLocations = std::array<int, kNumAgents * 2>;

inline int Offset(int row, int col) { return row * kCols + col; }

inline bool InGrid(int row, int col) {
  return 0 <= row && row < kRows && 0 <= col && col < kCols;
}

inline Grid ParseGrid(const std::string& text) {
  Grid grid{};
  grid.fill(kDirty);
  if (text.empty()) {
    return grid;
  }
  std::stringstream stream(text);
  std::string token;
  int index = 0;
  while (std::getline(stream, token, ',') && index < kCellCount) {
    grid[index++] = static_cast<std::int8_t>(std::stoi(token));
  }
  return grid;
}

inline AgentLocations ParseAgentLocations(const std::string& text) {
  AgentLocations locations{};
  if (text.empty()) {
    return locations;
  }
  std::stringstream stream(text);
  std::string token;
  int index = 0;
  while (std::getline(stream, token, ',') && index < kNumAgents * 2) {
    locations[index++] = std::stoi(token);
  }
  return locations;
}

}  // namespace cleaner

class CleanerEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("cleaner_grid"_.Bind(std::string("")),
                    "cleaner_agent_locations"_.Bind(std::string("")),
                    "cleaner_penalty_per_timestep"_.Bind(0.5f));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict(
        "obs:grid"_.Bind(Spec<std::int8_t>({10, 10}, {0, 2})),
        "obs:agents_locations"_.Bind(Spec<int>({3, 2}, {0, 10})),
        "obs:action_mask"_.Bind(Spec<bool>({3, 4}, {false, true})),
        "obs:step_count"_.Bind(Spec<int>({}, {0, 100})),
        "info:ratio_dirty_tiles"_.Bind(Spec<float>({}, {0.0f, 1.0f})),
        "info:num_dirty_tiles"_.Bind(Spec<int>({}, {0, 100})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1, 3}, {0, 3})));
  }
};

using CleanerEnvSpec = EnvSpec<CleanerEnvFns>;

class CleanerEnv : public Env<CleanerEnvSpec>, public RenderableEnv {
 protected:
  cleaner::Grid grid_{};
  cleaner::Grid configured_grid_{};
  cleaner::AgentLocations configured_agent_locations_{};
  bool use_configured_grid_;
  bool use_configured_agent_locations_;
  float penalty_;
  std::array<int, cleaner::kNumAgents> agent_rows_{};
  std::array<int, cleaner::kNumAgents> agent_cols_{};
  int step_count_{0};
  bool done_{true};

 public:
  using Spec = CleanerEnvSpec;
  using Action = typename Env<CleanerEnvSpec>::Action;

  CleanerEnv(const Spec& spec, int env_id)
      : Env<CleanerEnvSpec>(spec, env_id),
        configured_grid_(cleaner::ParseGrid(spec.config["cleaner_grid"_])),
        configured_agent_locations_(cleaner::ParseAgentLocations(
            spec.config["cleaner_agent_locations"_])),
        use_configured_grid_(!spec.config["cleaner_grid"_].empty()),
        use_configured_agent_locations_(
            !spec.config["cleaner_agent_locations"_].empty()),
        penalty_(spec.config["cleaner_penalty_per_timestep"_]) {}

  bool IsDone() override { return done_; }

  int CurrentMaxEpisodeSteps() const override {
    return cleaner::kTimeLimit + 1;
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, render::kWhite, rgb);
    for (int row = 0; row < cleaner::kRows; ++row) {
      for (int col = 0; col < cleaner::kCols; ++col) {
        render::Color color = {0, 240, 40};
        const auto cell = grid_[cleaner::Offset(row, col)];
        if (cell == cleaner::kClean) {
          color = {250, 250, 250};
        } else if (cell == cleaner::kWall) {
          color = render::kBlack;
        }
        render::FillCell(width, height, cleaner::kRows, cleaner::kCols, row,
                         col, color, rgb, 1);
      }
    }
    render::DrawGrid(width, height, cleaner::kRows, cleaner::kCols,
                     {175, 175, 175}, rgb);
    for (int agent = 0; agent < cleaner::kNumAgents; ++agent) {
      auto [x, y] =
          render::CellCenter(width, height, cleaner::kRows, cleaner::kCols,
                             agent_rows_[agent], agent_cols_[agent]);
      render::FillCircle(width, height, x, y,
                         std::max(3, std::min(width, height) / 40),
                         render::Palette(agent), rgb);
    }
  }

  void Reset() override {
    if (use_configured_grid_) {
      grid_ = configured_grid_;
    } else {
      grid_.fill(cleaner::kDirty);
      std::bernoulli_distribution wall_dist(0.15);
      for (auto& cell : grid_) {
        cell = wall_dist(gen_) ? cleaner::kWall : cleaner::kDirty;
      }
    }
    if (use_configured_agent_locations_) {
      for (int agent = 0; agent < cleaner::kNumAgents; ++agent) {
        agent_rows_[agent] = std::clamp(configured_agent_locations_[agent * 2],
                                        0, cleaner::kRows - 1);
        agent_cols_[agent] = std::clamp(
            configured_agent_locations_[agent * 2 + 1], 0, cleaner::kCols - 1);
        grid_[cleaner::Offset(agent_rows_[agent], agent_cols_[agent])] =
            cleaner::kClean;
      }
    } else {
      agent_rows_.fill(0);
      agent_cols_.fill(0);
      grid_[cleaner::Offset(0, 0)] = cleaner::kClean;
    }
    step_count_ = 0;
    done_ = NoDirtyTiles();
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    std::array<bool, cleaner::kNumAgents> valid{};
    for (int agent = 0; agent < cleaner::kNumAgents; ++agent) {
      const int action_id =
          std::clamp(static_cast<int>(action["action"_](0, agent)), 0, 3);
      const int row = agent_rows_[agent] + cleaner::kMoves[action_id][0];
      const int col = agent_cols_[agent] + cleaner::kMoves[action_id][1];
      valid[agent] = cleaner::InGrid(row, col) &&
                     grid_[cleaner::Offset(row, col)] != cleaner::kWall;
      if (valid[agent]) {
        agent_rows_[agent] = row;
        agent_cols_[agent] = col;
      }
    }
    int cleaned = 0;
    for (int agent = 0; agent < cleaner::kNumAgents; ++agent) {
      const int offset =
          cleaner::Offset(agent_rows_[agent], agent_cols_[agent]);
      if (grid_[offset] == cleaner::kDirty) {
        grid_[offset] = cleaner::kClean;
        ++cleaned;
      }
    }
    ++step_count_;
    const bool all_valid = std::all_of(valid.begin(), valid.end(),
                                       [](bool value) { return value; });
    done_ = !all_valid || NoDirtyTiles() || step_count_ >= cleaner::kTimeLimit;
    WriteState(static_cast<float>(cleaned) - penalty_);
  }

 private:
  bool NoDirtyTiles() const {
    return std::none_of(grid_.begin(), grid_.end(), [](std::int8_t cell) {
      return cell == cleaner::kDirty;
    });
  }

  bool IsActionValid(int agent, int action) const {
    const int row = agent_rows_[agent] + cleaner::kMoves[action][0];
    const int col = agent_cols_[agent] + cleaner::kMoves[action][1];
    return cleaner::InGrid(row, col) &&
           grid_[cleaner::Offset(row, col)] != cleaner::kWall;
  }

  void WriteState(float reward) {
    auto state = Allocate();
    int dirty = 0;
    int non_wall = 0;
    for (int row = 0; row < cleaner::kRows; ++row) {
      for (int col = 0; col < cleaner::kCols; ++col) {
        const auto cell = grid_[cleaner::Offset(row, col)];
        state["obs:grid"_](row, col) = cell;
        dirty += cell == cleaner::kDirty ? 1 : 0;
        non_wall += cell != cleaner::kWall ? 1 : 0;
      }
    }
    for (int agent = 0; agent < cleaner::kNumAgents; ++agent) {
      state["obs:agents_locations"_](agent, 0) = agent_rows_[agent];
      state["obs:agents_locations"_](agent, 1) = agent_cols_[agent];
      for (int action = 0; action < 4; ++action) {
        state["obs:action_mask"_](agent, action) = IsActionValid(agent, action);
      }
    }
    state["obs:step_count"_] = step_count_;
    state["info:ratio_dirty_tiles"_] =
        non_wall == 0 ? 0.0f : static_cast<float>(dirty) / non_wall;
    state["info:num_dirty_tiles"_] = dirty;
    state["reward"_] = reward;
  }
};

using CleanerEnvPool = AsyncEnvPool<CleanerEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_CLEANER_ENV_H_
