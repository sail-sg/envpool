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

#ifndef ENVPOOL_JUMANJI_MAZE_ENV_H_
#define ENVPOOL_JUMANJI_MAZE_ENV_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace maze {

constexpr int kRows = 10;
constexpr int kCols = 10;
constexpr int kCellCount = kRows * kCols;
constexpr int kTimeLimit = 100;
constexpr std::array<std::array<int, 2>, 4> kMoves = {
    {{{-1, 0}}, {{0, 1}}, {{1, 0}}, {{0, -1}}}};  // NOLINT

using Walls = std::array<bool, kCellCount>;

inline int Offset(int row, int col) { return row * kCols + col; }

inline bool InGrid(int row, int col) {
  return 0 <= row && row < kRows && 0 <= col && col < kCols;
}

inline Walls ParseWalls(const std::string& text) {
  Walls walls{};
  if (text.empty()) {
    return walls;
  }
  std::stringstream stream(text);
  std::string token;
  int index = 0;
  while (std::getline(stream, token, ',') && index < kCellCount) {
    walls[index++] = std::stoi(token) != 0;
  }
  return walls;
}

inline std::pair<int, int> ParsePosition(const std::string& text,
                                         int default_row, int default_col) {
  if (text.empty()) {
    return {default_row, default_col};
  }
  const std::size_t sep = text.find(',');
  if (sep == std::string::npos) {
    return {default_row, default_col};
  }
  return {std::clamp(std::stoi(text.substr(0, sep)), 0, kRows - 1),
          std::clamp(std::stoi(text.substr(sep + 1)), 0, kCols - 1)};
}

}  // namespace maze

class MazeEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("maze_walls"_.Bind(std::string("")),
                    "maze_agent_position"_.Bind(std::string("")),
                    "maze_target_position"_.Bind(std::string("")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("obs:agent_position.row"_.Bind(Spec<int>({}, {0, 9})),
                    "obs:agent_position.col"_.Bind(Spec<int>({}, {0, 9})),
                    "obs:target_position.row"_.Bind(Spec<int>({}, {0, 9})),
                    "obs:target_position.col"_.Bind(Spec<int>({}, {0, 9})),
                    "obs:walls"_.Bind(Spec<bool>({10, 10}, {false, true})),
                    "obs:step_count"_.Bind(Spec<int>({}, {0, 100})),
                    "obs:action_mask"_.Bind(Spec<bool>({4}, {false, true})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 3})));
  }
};

using MazeEnvSpec = EnvSpec<MazeEnvFns>;

class MazeEnv : public Env<MazeEnvSpec>, public RenderableEnv {
 protected:
  maze::Walls walls_{};
  maze::Walls configured_walls_{};
  bool use_configured_walls_;
  int agent_row_{0};
  int agent_col_{0};
  int configured_agent_row_{0};
  int configured_agent_col_{0};
  int target_row_{maze::kRows - 1};
  int target_col_{maze::kCols - 1};
  int configured_target_row_{maze::kRows - 1};
  int configured_target_col_{maze::kCols - 1};
  int step_count_{0};
  bool done_{true};

 public:
  using Spec = MazeEnvSpec;
  using Action = typename Env<MazeEnvSpec>::Action;

  MazeEnv(const Spec& spec, int env_id)
      : Env<MazeEnvSpec>(spec, env_id),
        configured_walls_(maze::ParseWalls(spec.config["maze_walls"_])),
        use_configured_walls_(!spec.config["maze_walls"_].empty()) {
    std::tie(configured_agent_row_, configured_agent_col_) =
        maze::ParsePosition(spec.config["maze_agent_position"_], 0, 0);
    std::tie(configured_target_row_, configured_target_col_) =
        maze::ParsePosition(spec.config["maze_target_position"_],
                            maze::kRows - 1, maze::kCols - 1);
  }

  bool IsDone() override { return done_; }

  int CurrentMaxEpisodeSteps() const override { return maze::kTimeLimit + 1; }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, render::kWhite, rgb);
    for (int row = 0; row < maze::kRows; ++row) {
      for (int col = 0; col < maze::kCols; ++col) {
        render::Color color = render::kWhite;
        if (walls_[maze::Offset(row, col)]) {
          color = render::kBlack;
        }
        render::FillCell(width, height, maze::kRows, maze::kCols, row, col,
                         color, rgb);
      }
    }
    render::DrawGrid(width, height, maze::kRows, maze::kCols, {210, 210, 210},
                     rgb);
    auto [tx, ty] = render::CellCenter(width, height, maze::kRows, maze::kCols,
                                       target_row_, target_col_);
    render::FillRect(width, height, tx - 5, ty - 5, tx + 6, ty + 6,
                     {0, 210, 70}, rgb);
    auto [ax, ay] = render::CellCenter(width, height, maze::kRows, maze::kCols,
                                       agent_row_, agent_col_);
    render::FillCircle(width, height, ax, ay,
                       std::max(3, std::min(width, height) / 40), {220, 30, 30},
                       rgb);
  }

  void Reset() override {
    walls_ = use_configured_walls_ ? configured_walls_ : maze::Walls{};
    if (!use_configured_walls_) {
      std::bernoulli_distribution wall_dist(0.2);
      for (bool& wall : walls_) {
        wall = wall_dist(gen_);
      }
    }
    agent_row_ = configured_agent_row_;
    agent_col_ = configured_agent_col_;
    target_row_ = configured_target_row_;
    target_col_ = configured_target_col_;
    walls_[maze::Offset(agent_row_, agent_col_)] = false;
    walls_[maze::Offset(target_row_, target_col_)] = false;
    step_count_ = 0;
    done_ = agent_row_ == target_row_ && agent_col_ == target_col_;
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    const int action_id = std::clamp(static_cast<int>(action["action"_]), 0, 3);
    const int next_row = agent_row_ + maze::kMoves[action_id][0];
    const int next_col = agent_col_ + maze::kMoves[action_id][1];
    if (maze::InGrid(next_row, next_col) &&
        !walls_[maze::Offset(next_row, next_col)]) {
      agent_row_ = next_row;
      agent_col_ = next_col;
    }
    ++step_count_;
    const bool target_reached =
        agent_row_ == target_row_ && agent_col_ == target_col_;
    done_ = target_reached || step_count_ >= maze::kTimeLimit ||
            !AnyActionAvailable();
    WriteState(target_reached ? 1.0f : 0.0f);
  }

 private:
  bool AnyActionAvailable() const {
    for (int action = 0; action < 4; ++action) {
      const int row = agent_row_ + maze::kMoves[action][0];
      const int col = agent_col_ + maze::kMoves[action][1];
      if (maze::InGrid(row, col) && !walls_[maze::Offset(row, col)]) {
        return true;
      }
    }
    return false;
  }

  void WriteState(float reward) {
    auto state = Allocate();
    state["obs:agent_position.row"_] = agent_row_;
    state["obs:agent_position.col"_] = agent_col_;
    state["obs:target_position.row"_] = target_row_;
    state["obs:target_position.col"_] = target_col_;
    for (int row = 0; row < maze::kRows; ++row) {
      for (int col = 0; col < maze::kCols; ++col) {
        state["obs:walls"_](row, col) = walls_[maze::Offset(row, col)];
      }
    }
    state["obs:step_count"_] = step_count_;
    for (int action = 0; action < 4; ++action) {
      const int row = agent_row_ + maze::kMoves[action][0];
      const int col = agent_col_ + maze::kMoves[action][1];
      state["obs:action_mask"_][action] =
          maze::InGrid(row, col) && !walls_[maze::Offset(row, col)];
    }
    state["reward"_] = reward;
  }
};

using MazeEnvPool = AsyncEnvPool<MazeEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_MAZE_ENV_H_
