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

#ifndef ENVPOOL_JUMANJI_ROBOT_WAREHOUSE_ENV_H_
#define ENVPOOL_JUMANJI_ROBOT_WAREHOUSE_ENV_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/parse_utils.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace robotwarehouse {

constexpr int kRows = 20;
constexpr int kCols = 10;
constexpr int kNumShelves = 80;
constexpr int kNumAgents = 4;
constexpr int kViewSize = 66;
constexpr int kTimeLimit = 500;
constexpr std::array<std::array<int, 2>, 4> kMoves = {
    {{{-1, 0}}, {{0, 1}}, {{1, 0}}, {{0, -1}}}};  // NOLINT

inline bool InGrid(int row, int col) {
  return 0 <= row && row < kRows && 0 <= col && col < kCols;
}

inline bool IsHighway(int row, int col) {
  return col % 3 == 0 || row % 9 == 0 || row == kRows - 1 ||
         (row > 9 && (col == 4 || col == 5));
}

}  // namespace robotwarehouse

class RobotWarehouseEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "robot_warehouse_agents_view"_.Bind(std::string("")),
        "robot_warehouse_action_mask"_.Bind(std::string("")),
        "robot_warehouse_render_grid"_.Bind(std::string("")),
        "robot_warehouse_render_agent_x"_.Bind(std::string("")),
        "robot_warehouse_render_agent_y"_.Bind(std::string("")),
        "robot_warehouse_render_agent_direction"_.Bind(std::string("")),
        "robot_warehouse_render_agent_carrying"_.Bind(std::string("")),
        "robot_warehouse_render_shelf_x"_.Bind(std::string("")),
        "robot_warehouse_render_shelf_y"_.Bind(std::string("")),
        "robot_warehouse_render_shelf_requested"_.Bind(std::string("")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("info:shelf_positions"_.Bind(Spec<int>({80, 2})),
                    "info:shelf_requested"_.Bind(Spec<bool>({80})),
                    "obs:agents_view"_.Bind(Spec<int>({4, 66})),
                    "obs:action_mask"_.Bind(Spec<bool>({4, 5}, {false, true})),
                    "obs:step_count"_.Bind(Spec<int>({}, {0, 500})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1, 4}, {0, 4})));
  }
};

using RobotWarehouseEnvSpec = EnvSpec<RobotWarehouseEnvFns>;

class RobotWarehouseEnv : public Env<RobotWarehouseEnvSpec>,
                          public RenderableEnv {
 protected:
  std::array<int, robotwarehouse::kNumAgents> row_{};
  std::array<int, robotwarehouse::kNumAgents> col_{};
  std::array<int, robotwarehouse::kNumAgents> direction_{};
  std::array<bool, robotwarehouse::kNumAgents> carrying_{};
  std::array<int, robotwarehouse::kNumShelves> shelf_row_{};
  std::array<int, robotwarehouse::kNumShelves> shelf_col_{};
  std::array<bool, robotwarehouse::kNumShelves> requested_{};
  std::array<int, robotwarehouse::kRows * robotwarehouse::kCols> agents_{};
  std::array<int, robotwarehouse::kRows * robotwarehouse::kCols> shelves_{};
  int step_count_{0};
  bool done_{true};

 public:
  using Spec = RobotWarehouseEnvSpec;
  using Action = typename Env<RobotWarehouseEnvSpec>::Action;

  RobotWarehouseEnv(const Spec& spec, int env_id)
      : Env<RobotWarehouseEnvSpec>(spec, env_id) {}

  bool IsDone() override { return done_; }

  int CurrentMaxEpisodeSteps() const override {
    return robotwarehouse::kTimeLimit + 1;
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, render::kWhite, rgb);
    for (int shelf = 0; shelf < robotwarehouse::kNumShelves; ++shelf) {
      render::FillCell(width, height, robotwarehouse::kRows,
                       robotwarehouse::kCols, shelf_row_[shelf],
                       shelf_col_[shelf],
                       requested_[shelf] ? render::Color{230, 130, 40}
                                         : render::Color{100, 90, 150},
                       rgb, 1);
    }
    render::DrawGrid(width, height, robotwarehouse::kRows,
                     robotwarehouse::kCols, {140, 140, 170}, rgb);
    for (int agent = 0; agent < robotwarehouse::kNumAgents; ++agent) {
      auto [x, y] =
          render::CellCenter(width, height, robotwarehouse::kRows,
                             robotwarehouse::kCols, row_[agent], col_[agent]);
      render::FillCircle(width, height, x, y,
                         std::max(2, std::min(width, height) / 50),
                         render::Palette(agent), rgb);
      if (carrying_[agent]) {
        render::FillRect(width, height, x - 2, y - 2, x + 3, y + 3,
                         {180, 110, 40}, rgb);
      }
    }
  }

  void Reset() override {
    const auto& conf = spec_.config;
    agents_.fill(0);
    shelves_.fill(0);
    carrying_.fill(false);
    requested_.fill(false);
    int shelf = 0;
    for (int row = 0; row < robotwarehouse::kRows; ++row) {
      for (int col = 0; col < robotwarehouse::kCols; ++col) {
        if (!robotwarehouse::IsHighway(row, col)) {
          shelf_row_[shelf] = row;
          shelf_col_[shelf++] = col;
        }
      }
    }
    if (!conf["robot_warehouse_render_shelf_x"_].empty()) {
      shelf_row_ = parse::CsvArray<int, robotwarehouse::kNumShelves>(
          conf["robot_warehouse_render_shelf_x"_]);
      shelf_col_ = parse::CsvArray<int, robotwarehouse::kNumShelves>(
          conf["robot_warehouse_render_shelf_y"_]);
      requested_ = parse::CsvArray<bool, robotwarehouse::kNumShelves>(
          conf["robot_warehouse_render_shelf_requested"_]);
    } else {
      std::array<int, robotwarehouse::kNumShelves> ids{};
      std::iota(ids.begin(), ids.end(), 0);
      std::shuffle(ids.begin(), ids.end(), gen_);
      for (int i = 0; i < 8; ++i) {
        requested_[ids[i]] = true;
      }
    }
    if (!conf["robot_warehouse_render_agent_x"_].empty()) {
      row_ = parse::CsvArray<int, robotwarehouse::kNumAgents>(
          conf["robot_warehouse_render_agent_x"_]);
      col_ = parse::CsvArray<int, robotwarehouse::kNumAgents>(
          conf["robot_warehouse_render_agent_y"_]);
      direction_ = parse::CsvArray<int, robotwarehouse::kNumAgents>(
          conf["robot_warehouse_render_agent_direction"_]);
      carrying_ = parse::CsvArray<bool, robotwarehouse::kNumAgents>(
          conf["robot_warehouse_render_agent_carrying"_]);
    } else {
      std::array<int, robotwarehouse::kRows * robotwarehouse::kCols> cells{};
      std::iota(cells.begin(), cells.end(), 0);
      std::shuffle(cells.begin(), cells.end(), gen_);
      for (int agent = 0; agent < robotwarehouse::kNumAgents; ++agent) {
        row_[agent] = cells[agent] / robotwarehouse::kCols;
        col_[agent] = cells[agent] % robotwarehouse::kCols;
        direction_[agent] = std::uniform_int_distribution<int>(0, 3)(gen_);
      }
    }
    for (int id = 0; id < robotwarehouse::kNumShelves; ++id) {
      shelves_[Cell(shelf_row_[id], shelf_col_[id])] = id + 1;
    }
    for (int id = 0; id < robotwarehouse::kNumAgents; ++id) {
      agents_[Cell(row_[id], col_[id])] = id + 1;
    }
    step_count_ = 0;
    done_ = false;
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    std::array<int, robotwarehouse::kNumAgents> actions{};
    for (int agent = 0; agent < robotwarehouse::kNumAgents; ++agent) {
      const int value =
          std::clamp(static_cast<int>(action["action"_](0, agent)), 0, 4);
      actions[agent] = IsValid(agent, value) ? value : 0;
    }
    for (int agent = 0; agent < robotwarehouse::kNumAgents; ++agent) {
      const int value = actions[agent];
      const int cell = Cell(row_[agent], col_[agent]);
      if (value == 1) {
        const auto [row, col] = Forward(agent);
        const int next = Cell(row, col);
        agents_[cell] = 0;
        agents_[next] = agent + 1;
        if (carrying_[agent]) {
          const int shelf = shelves_[cell] - 1;
          if (shelf >= 0) {
            shelves_[cell] = 0;
            shelves_[next] = shelf + 1;
            shelf_row_[shelf] = row;
            shelf_col_[shelf] = col;
          }
        }
        row_[agent] = row;
        col_[agent] = col;
      } else if (value == 2 || value == 3) {
        direction_[agent] = (direction_[agent] + (value == 2 ? 3 : 1)) % 4;
      } else if (value == 4 && !carrying_[agent]) {
        carrying_[agent] = shelves_[cell] > 0;
      } else if (!robotwarehouse::IsHighway(row_[agent], col_[agent])) {
        // Jumanji 1.1.2 also unloads on NOOP outside a highway.
        carrying_[agent] = false;
      }
    }
    float reward = 0.0f;
    for (int col : {4, 5}) {
      const int shelf = shelves_[Cell(robotwarehouse::kRows - 1, col)] - 1;
      if (shelf >= 0 && requested_[shelf]) {
        std::vector<int> candidates;
        for (int id = 0; id < robotwarehouse::kNumShelves; ++id) {
          if (!requested_[id]) {
            candidates.push_back(id);
          }
        }
        const int replacement = candidates[std::uniform_int_distribution<int>(
            0, static_cast<int>(candidates.size()) - 1)(gen_)];
        requested_[shelf] = false;
        requested_[replacement] = true;
        reward += 1.0f;
      }
    }
    ++step_count_;
    done_ = step_count_ >= robotwarehouse::kTimeLimit;
    for (int agent = 0; agent < robotwarehouse::kNumAgents; ++agent) {
      done_ |= agents_[Cell(row_[agent], col_[agent])] != agent + 1;
    }
    WriteState(reward);
  }

 private:
  static int Cell(int row, int col) {
    return row * robotwarehouse::kCols + col;
  }

  std::pair<int, int> Forward(int agent) const {
    return {
        std::clamp(row_[agent] + robotwarehouse::kMoves[direction_[agent]][0],
                   0, robotwarehouse::kRows - 1),
        std::clamp(col_[agent] + robotwarehouse::kMoves[direction_[agent]][1],
                   0, robotwarehouse::kCols - 1)};
  }

  bool IsValid(int agent, int action) const {
    const auto [row, col] = Forward(agent);
    return action != 1 || !carrying_[agent] ||
           (row == row_[agent] && col == col_[agent]) ||
           shelves_[Cell(row, col)] == 0;
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int shelf = 0; shelf < robotwarehouse::kNumShelves; ++shelf) {
      state["info:shelf_positions"_](shelf, 0) = shelf_row_[shelf];
      state["info:shelf_positions"_](shelf, 1) = shelf_col_[shelf];
      state["info:shelf_requested"_][shelf] = requested_[shelf];
    }
    for (int agent = 0; agent < robotwarehouse::kNumAgents; ++agent) {
      std::array<int, robotwarehouse::kViewSize> view{};
      view[0] = row_[agent];
      view[1] = col_[agent];
      view[2] = carrying_[agent] ? 1 : 0;
      view[3 + direction_[agent]] = 1;
      view[7] = robotwarehouse::IsHighway(row_[agent], col_[agent]) ? 1 : 0;
      int index = 8;
      for (int row = row_[agent] - 1; row <= row_[agent] + 1; ++row) {
        for (int col = col_[agent] - 1; col <= col_[agent] + 1; ++col) {
          const int id = robotwarehouse::InGrid(row, col)
                             ? agents_[Cell(row, col)] - 1
                             : -1;
          if (id == agent) {
            continue;
          }
          if (id >= 0 && index + 4 < robotwarehouse::kViewSize) {
            view[index] = 1;
            view[index + 1 + direction_[id]] = 1;
          }
          index += 5;
        }
      }
      for (int row = row_[agent] - 1; row <= row_[agent] + 1; ++row) {
        for (int col = col_[agent] - 1; col <= col_[agent] + 1; ++col) {
          const int id = robotwarehouse::InGrid(row, col)
                             ? shelves_[Cell(row, col)] - 1
                             : -1;
          if (id >= 0) {
            // Match dynamic_update_slice's clamping if a collision removes
            // the self cell from the agent channel.
            const int start = std::min(index, robotwarehouse::kViewSize - 2);
            view[start] = 1;
            view[start + 1] = requested_[id] ? 1 : 0;
          }
          index += 2;
        }
      }
      for (int i = 0; i < robotwarehouse::kViewSize; ++i) {
        state["obs:agents_view"_](agent, i) = view[i];
      }
      for (int action = 0; action < 5; ++action) {
        state["obs:action_mask"_](agent, action) = IsValid(agent, action);
      }
    }
    state["obs:step_count"_] = step_count_;
    state["reward"_] = reward;
  }
};

using RobotWarehouseEnvPool = AsyncEnvPool<RobotWarehouseEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_ROBOT_WAREHOUSE_ENV_H_
