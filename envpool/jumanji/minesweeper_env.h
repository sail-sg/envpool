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

#ifndef ENVPOOL_JUMANJI_MINESWEEPER_ENV_H_
#define ENVPOOL_JUMANJI_MINESWEEPER_ENV_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/parse_utils.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace minesweeper {

constexpr int kRows = 10;
constexpr int kCols = 10;
constexpr int kCellCount = kRows * kCols;
constexpr int kDefaultMines = 10;
constexpr int kUnexplored = -1;
constexpr int kReplaySteps = 32;

using Board = std::array<int, kCellCount>;
using MineMask = std::array<bool, kCellCount>;

inline int Offset(int row, int col) { return row * kCols + col; }

inline std::vector<int> ParseMineLocations(const std::string& text) {
  std::vector<int> mines;
  if (text.empty()) {
    return mines;
  }
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    const int location = std::stoi(token);
    if (0 <= location && location < kCellCount) {
      mines.push_back(location);
    }
  }
  std::sort(mines.begin(), mines.end());
  mines.erase(std::unique(mines.begin(), mines.end()), mines.end());
  return mines;
}

inline int CountAdjacentMines(const MineMask& mines, int row, int col) {
  int count = 0;
  for (int dr = -1; dr <= 1; ++dr) {
    for (int dc = -1; dc <= 1; ++dc) {
      if (dr == 0 && dc == 0) {
        continue;
      }
      const int r = row + dr;
      const int c = col + dc;
      if (0 <= r && r < kRows && 0 <= c && c < kCols && mines[Offset(r, c)]) {
        ++count;
      }
    }
  }
  return count;
}

}  // namespace minesweeper

class MinesweeperEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("minesweeper_mine_locations"_.Bind(std::string("")),
                    "minesweeper_replay_boards"_.Bind(std::string("")),
                    "minesweeper_replay_rewards"_.Bind(std::string("")),
                    "minesweeper_replay_done"_.Bind(std::string("")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict(
        "obs:board"_.Bind(Spec<int>({10, 10}, {-1, 8})),
        "obs:action_mask"_.Bind(Spec<bool>({10, 10}, {false, true})),
        "obs:num_mines"_.Bind(Spec<int>({}, {0, 99})),
        "obs:step_count"_.Bind(Spec<int>({}, {0, 90})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1, 2}, {0, 9})));
  }
};

using MinesweeperEnvSpec = EnvSpec<MinesweeperEnvFns>;

class MinesweeperEnv : public Env<MinesweeperEnvSpec>, public RenderableEnv {
 protected:
  minesweeper::Board board_{};
  minesweeper::MineMask mines_{};
  std::vector<int> configured_mines_;
  std::array<int, minesweeper::kReplaySteps * minesweeper::kCellCount>
      replay_boards_{};
  std::array<float, minesweeper::kReplaySteps> replay_rewards_{};
  std::array<bool, minesweeper::kReplaySteps> replay_done_{};
  int num_mines_;
  int step_count_{0};
  bool use_replay_;
  bool done_{true};

 public:
  using Spec = MinesweeperEnvSpec;
  using Action = typename Env<MinesweeperEnvSpec>::Action;

  MinesweeperEnv(const Spec& spec, int env_id)
      : Env<MinesweeperEnvSpec>(spec, env_id),
        configured_mines_(minesweeper::ParseMineLocations(
            spec.config["minesweeper_mine_locations"_])),
        replay_boards_(parse::CsvArray<int, minesweeper::kReplaySteps *
                                                minesweeper::kCellCount>(
            spec.config["minesweeper_replay_boards"_],
            minesweeper::kUnexplored)),
        replay_rewards_(parse::CsvArray<float, minesweeper::kReplaySteps>(
            spec.config["minesweeper_replay_rewards"_])),
        replay_done_(parse::CsvArray<bool, minesweeper::kReplaySteps>(
            spec.config["minesweeper_replay_done"_])),
        num_mines_(configured_mines_.empty()
                       ? minesweeper::kDefaultMines
                       : static_cast<int>(configured_mines_.size())),
        use_replay_(!spec.config["minesweeper_replay_boards"_].empty()) {}

  bool IsDone() override { return done_; }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, render::kWhite, rgb);
    for (int row = 0; row < minesweeper::kRows; ++row) {
      for (int col = 0; col < minesweeper::kCols; ++col) {
        const int value = board_[minesweeper::Offset(row, col)];
        render::Color color = {206, 206, 206};
        if (value >= 0) {
          color = {246, 246, 246};
        }
        render::FillCell(width, height, minesweeper::kRows, minesweeper::kCols,
                         row, col, color, rgb, 1);
        if (value > 0) {
          const int left = col * width / minesweeper::kCols;
          const int right = (col + 1) * width / minesweeper::kCols;
          const int top = row * height / minesweeper::kRows;
          const int bottom = (row + 1) * height / minesweeper::kRows;
          render::DrawNumber(width, height, value, left + 6, top + 5, right - 6,
                             bottom - 5, render::Palette(value), rgb);
        }
      }
    }
    render::DrawGrid(width, height, minesweeper::kRows, minesweeper::kCols,
                     {150, 150, 150}, rgb);
  }

  void Reset() override {
    board_.fill(minesweeper::kUnexplored);
    mines_.fill(false);
    if (configured_mines_.empty()) {
      std::array<int, minesweeper::kCellCount> locations{};
      std::iota(locations.begin(), locations.end(), 0);
      std::shuffle(locations.begin(), locations.end(), gen_);
      for (int i = 0; i < num_mines_; ++i) {
        mines_[locations[i]] = true;
      }
    } else {
      for (int location : configured_mines_) {
        mines_[location] = true;
      }
    }
    step_count_ = 0;
    done_ = false;
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    if (use_replay_ && step_count_ < minesweeper::kReplaySteps) {
      ++step_count_;
      for (int i = 0; i < minesweeper::kCellCount; ++i) {
        board_[i] =
            replay_boards_[(step_count_ - 1) * minesweeper::kCellCount + i];
      }
      done_ = replay_done_[step_count_ - 1];
      WriteState(replay_rewards_[step_count_ - 1]);
      return;
    }
    const int row = std::clamp(static_cast<int>(action["action"_](0, 0)), 0,
                               minesweeper::kRows - 1);
    const int col = std::clamp(static_cast<int>(action["action"_](0, 1)), 0,
                               minesweeper::kCols - 1);
    const int offset = minesweeper::Offset(row, col);
    const bool valid_action = board_[offset] == minesweeper::kUnexplored;
    const bool hit_mine = mines_[offset];
    float reward = 0.0f;
    if (valid_action) {
      Reveal(row, col);
      reward = hit_mine ? 0.0f : 1.0f;
    }
    ++step_count_;
    done_ = !valid_action || hit_mine || IsSolved();
    WriteState(reward);
  }

 private:
  bool IsSolved() const {
    int explored = 0;
    for (int cell : board_) {
      explored += cell >= 0 ? 1 : 0;
    }
    return explored == minesweeper::kCellCount - num_mines_;
  }

  void Reveal(int row, int col) {
    std::queue<std::pair<int, int>> frontier;
    frontier.emplace(row, col);
    while (!frontier.empty()) {
      const auto [cell_row, cell_col] = frontier.front();
      frontier.pop();
      if (cell_row < 0 || cell_row >= minesweeper::kRows || cell_col < 0 ||
          cell_col >= minesweeper::kCols) {
        continue;
      }
      const int offset = minesweeper::Offset(cell_row, cell_col);
      if (board_[offset] != minesweeper::kUnexplored) {
        continue;
      }
      const int adjacent =
          minesweeper::CountAdjacentMines(mines_, cell_row, cell_col);
      board_[offset] = adjacent;
      if (adjacent != 0 || mines_[offset]) {
        continue;
      }
      for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
          if (dr != 0 || dc != 0) {
            frontier.emplace(cell_row + dr, cell_col + dc);
          }
        }
      }
    }
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int row = 0; row < minesweeper::kRows; ++row) {
      for (int col = 0; col < minesweeper::kCols; ++col) {
        const int offset = minesweeper::Offset(row, col);
        state["obs:board"_](row, col) = board_[offset];
        state["obs:action_mask"_](row, col) =
            board_[offset] == minesweeper::kUnexplored;
      }
    }
    state["obs:num_mines"_] = num_mines_;
    state["obs:step_count"_] = step_count_;
    state["reward"_] = reward;
  }
};

using MinesweeperEnvPool = AsyncEnvPool<MinesweeperEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_MINESWEEPER_ENV_H_
