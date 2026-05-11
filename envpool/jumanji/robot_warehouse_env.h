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
#include <string>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/parse_utils.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace robotwarehouse {

constexpr int kGridSize = 8;
constexpr int kNumAgents = 4;
constexpr int kViewSize = 66;
constexpr int kTimeLimit = 500;
constexpr int kAgentsViewSize = kNumAgents * kViewSize;
constexpr int kActionMaskSize = kNumAgents * 5;
constexpr std::array<std::array<int, 2>, 5> kMoves = {
    {{{0, 0}}, {{-1, 0}}, {{0, 1}}, {{1, 0}}, {{0, -1}}}};  // NOLINT

inline bool InGrid(int row, int col) {
  return 0 <= row && row < kGridSize && 0 <= col && col < kGridSize;
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
    return MakeDict("obs:agents_view"_.Bind(Spec<int>({4, 66})),
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
  std::array<bool, robotwarehouse::kNumAgents> carrying_{};
  std::array<int, robotwarehouse::kAgentsViewSize> configured_agents_view_{};
  std::array<bool, robotwarehouse::kActionMaskSize> configured_action_mask_{};
  bool use_configured_state_;
  int step_count_{0};
  bool done_{true};

 public:
  using Spec = RobotWarehouseEnvSpec;
  using Action = typename Env<RobotWarehouseEnvSpec>::Action;

  RobotWarehouseEnv(const Spec& spec, int env_id)
      : Env<RobotWarehouseEnvSpec>(spec, env_id),
        configured_agents_view_(
            parse::CsvArray<int, robotwarehouse::kAgentsViewSize>(
                spec.config["robot_warehouse_agents_view"_])),
        configured_action_mask_(
            parse::CsvArray<bool, robotwarehouse::kActionMaskSize>(
                spec.config["robot_warehouse_action_mask"_])),
        use_configured_state_(
            !spec.config["robot_warehouse_agents_view"_].empty()) {}

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
    for (int row = 0; row < robotwarehouse::kGridSize; ++row) {
      for (int col = 0; col < robotwarehouse::kGridSize; ++col) {
        if (col == 2 || col == 5) {
          render::FillCell(width, height, robotwarehouse::kGridSize,
                           robotwarehouse::kGridSize, row, col, {100, 90, 150},
                           rgb, 1);
        }
      }
    }
    render::DrawGrid(width, height, robotwarehouse::kGridSize,
                     robotwarehouse::kGridSize, {140, 140, 170}, rgb);
    for (int agent = 0; agent < robotwarehouse::kNumAgents; ++agent) {
      auto [x, y] = render::CellCenter(width, height, robotwarehouse::kGridSize,
                                       robotwarehouse::kGridSize, row_[agent],
                                       col_[agent]);
      render::FillCircle(width, height, x, y,
                         std::max(3, std::min(width, height) / 38),
                         render::Palette(agent), rgb);
      if (carrying_[agent]) {
        render::FillRect(width, height, x - 3, y - 3, x + 4, y + 4,
                         {180, 110, 40}, rgb);
      }
    }
  }

  void Reset() override {
    for (int agent = 0; agent < robotwarehouse::kNumAgents; ++agent) {
      row_[agent] =
          use_configured_state_
              ? configured_agents_view_[agent * robotwarehouse::kViewSize]
              : agent;
      col_[agent] =
          use_configured_state_
              ? configured_agents_view_[agent * robotwarehouse::kViewSize + 1]
              : 0;
      carrying_[agent] = false;
    }
    step_count_ = 0;
    done_ = false;
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    if (use_configured_state_) {
      ++step_count_;
      done_ = step_count_ >= robotwarehouse::kTimeLimit;
      WriteState(0.0f);
      return;
    }
    std::array<int, robotwarehouse::kNumAgents> next_row = row_;
    std::array<int, robotwarehouse::kNumAgents> next_col = col_;
    bool valid = true;
    for (int agent = 0; agent < robotwarehouse::kNumAgents; ++agent) {
      const int action_id =
          std::clamp(static_cast<int>(action["action"_](0, agent)), 0, 4);
      const int row = row_[agent] + robotwarehouse::kMoves[action_id][0];
      const int col = col_[agent] + robotwarehouse::kMoves[action_id][1];
      if (IsFree(agent, row, col)) {
        next_row[agent] = row;
        next_col[agent] = col;
      } else {
        valid = false;
      }
    }
    for (int a = 0; a < robotwarehouse::kNumAgents; ++a) {
      for (int b = a + 1; b < robotwarehouse::kNumAgents; ++b) {
        if (next_row[a] == next_row[b] && next_col[a] == next_col[b]) {
          valid = false;
        }
      }
    }
    if (valid) {
      row_ = next_row;
      col_ = next_col;
    }
    ++step_count_;
    done_ = !valid || step_count_ >= robotwarehouse::kTimeLimit;
    WriteState(valid ? 0.0f : -1.0f);
  }

 private:
  bool IsFree(int moving_agent, int row, int col) const {
    if (!robotwarehouse::InGrid(row, col)) {
      return false;
    }
    for (int agent = 0; agent < robotwarehouse::kNumAgents; ++agent) {
      if (agent != moving_agent && row_[agent] == row && col_[agent] == col) {
        return false;
      }
    }
    return true;
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int agent = 0; agent < robotwarehouse::kNumAgents; ++agent) {
      if (use_configured_state_) {
        for (int i = 0; i < robotwarehouse::kViewSize; ++i) {
          state["obs:agents_view"_](agent, i) =
              configured_agents_view_[agent * robotwarehouse::kViewSize + i];
        }
      } else {
        state["obs:agents_view"_](agent, 0) = row_[agent];
        state["obs:agents_view"_](agent, 1) = col_[agent];
        state["obs:agents_view"_](agent, 2) = carrying_[agent] ? 1 : 0;
        state["obs:agents_view"_](agent, 3) = robotwarehouse::kGridSize - 1;
        state["obs:agents_view"_](agent, 4) = robotwarehouse::kGridSize - 1;
        for (int i = 5; i < robotwarehouse::kViewSize; ++i) {
          state["obs:agents_view"_](agent, i) = 0;
        }
      }
      for (int action = 0; action < 5; ++action) {
        const int row = row_[agent] + robotwarehouse::kMoves[action][0];
        const int col = col_[agent] + robotwarehouse::kMoves[action][1];
        state["obs:action_mask"_](agent, action) =
            use_configured_state_ ? configured_action_mask_[agent * 5 + action]
                                  : IsFree(agent, row, col);
      }
    }
    state["obs:step_count"_] = step_count_;
    state["reward"_] = reward;
  }
};

using RobotWarehouseEnvPool = AsyncEnvPool<RobotWarehouseEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_ROBOT_WAREHOUSE_ENV_H_
