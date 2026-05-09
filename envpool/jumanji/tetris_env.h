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

#ifndef ENVPOOL_JUMANJI_TETRIS_ENV_H_
#define ENVPOOL_JUMANJI_TETRIS_ENV_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <sstream>
#include <string>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/parse_utils.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace tetris {

constexpr int kRows = 10;
constexpr int kCols = 10;
constexpr int kTimeLimit = 400;
constexpr int kActionMaskSize = 4 * kCols;
constexpr int kReplaySteps = 32;

using Grid = std::array<int, kRows * kCols>;
using Tetromino = std::array<int, 16>;

inline int Offset(int row, int col) { return row * kCols + col; }

inline Grid ParseGrid(const std::string& text) {
  Grid grid{};
  if (text.empty()) {
    return grid;
  }
  std::stringstream stream(text);
  std::string token;
  int index = 0;
  while (std::getline(stream, token, ',') && index < kRows * kCols) {
    grid[index++] = std::stoi(token) != 0 ? 1 : 0;
  }
  return grid;
}

inline Tetromino SquareTetromino() {
  Tetromino tetromino{};
  tetromino[0] = 1;
  tetromino[1] = 1;
  tetromino[4] = 1;
  tetromino[5] = 1;
  return tetromino;
}

inline Tetromino ParseTetromino(const std::string& text) {
  Tetromino tetromino = SquareTetromino();
  if (text.empty()) {
    return tetromino;
  }
  tetromino.fill(0);
  std::stringstream stream(text);
  std::string token;
  int index = 0;
  while (std::getline(stream, token, ',') && index < 16) {
    tetromino[index++] = std::stoi(token) != 0 ? 1 : 0;
  }
  return tetromino;
}

}  // namespace tetris

class TetrisEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("tetris_initial_grid"_.Bind(std::string("")),
                    "tetris_tetromino"_.Bind(std::string("")),
                    "tetris_action_mask"_.Bind(std::string("")),
                    "tetris_replay_grids"_.Bind(std::string("")),
                    "tetris_replay_tetrominoes"_.Bind(std::string("")),
                    "tetris_replay_action_masks"_.Bind(std::string("")),
                    "tetris_replay_done"_.Bind(std::string("")),
                    "tetris_render_grid_padded_replay"_.Bind(std::string("")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("obs:grid"_.Bind(Spec<int>({10, 10}, {0, 1})),
                    "obs:tetromino"_.Bind(Spec<int>({4, 4}, {0, 1})),
                    "obs:action_mask"_.Bind(Spec<bool>({4, 10}, {false, true})),
                    "obs:step_count"_.Bind(Spec<int>({}, {0, 399})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1, 2}, {{0, 0}, {3, 9}})));
  }
};

using TetrisEnvSpec = EnvSpec<TetrisEnvFns>;

class TetrisEnv : public Env<TetrisEnvSpec>, public RenderableEnv {
 protected:
  tetris::Grid grid_{};
  tetris::Grid configured_grid_{};
  tetris::Tetromino tetromino_{};
  tetris::Tetromino configured_tetromino_{};
  std::array<bool, tetris::kActionMaskSize> configured_action_mask_{};
  std::array<int, tetris::kReplaySteps * tetris::kRows * tetris::kCols>
      replay_grids_{};
  std::array<int, tetris::kReplaySteps * 16> replay_tetrominoes_{};
  std::array<bool, tetris::kReplaySteps * tetris::kActionMaskSize>
      replay_action_masks_{};
  std::array<bool, tetris::kReplaySteps> replay_done_{};
  bool use_configured_grid_;
  bool use_configured_action_mask_;
  bool use_replay_;
  int step_count_{0};
  bool done_{true};

 public:
  using Spec = TetrisEnvSpec;
  using Action = typename Env<TetrisEnvSpec>::Action;

  TetrisEnv(const Spec& spec, int env_id)
      : Env<TetrisEnvSpec>(spec, env_id),
        configured_grid_(
            tetris::ParseGrid(spec.config["tetris_initial_grid"_])),
        tetromino_(tetris::SquareTetromino()),
        configured_tetromino_(
            tetris::ParseTetromino(spec.config["tetris_tetromino"_])),
        configured_action_mask_(parse::CsvArray<bool, tetris::kActionMaskSize>(
            spec.config["tetris_action_mask"_])),
        replay_grids_(parse::CsvArray<int, tetris::kReplaySteps *
                                               tetris::kRows * tetris::kCols>(
            spec.config["tetris_replay_grids"_])),
        replay_tetrominoes_(parse::CsvArray<int, tetris::kReplaySteps * 16>(
            spec.config["tetris_replay_tetrominoes"_])),
        replay_action_masks_(parse::CsvArray<bool, tetris::kReplaySteps *
                                                       tetris::kActionMaskSize>(
            spec.config["tetris_replay_action_masks"_])),
        replay_done_(parse::CsvArray<bool, tetris::kReplaySteps>(
            spec.config["tetris_replay_done"_])),
        use_configured_grid_(!spec.config["tetris_initial_grid"_].empty()),
        use_configured_action_mask_(
            !spec.config["tetris_action_mask"_].empty()),
        use_replay_(!spec.config["tetris_replay_grids"_].empty()) {}

  bool IsDone() override { return done_; }

  int CurrentMaxEpisodeSteps() const override { return tetris::kTimeLimit + 1; }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, render::kWhite, rgb);
    const int preview_w = width / 4;
    const int board_left = width / 5;
    const int board_top = height / 5;
    const int board_right = width * 4 / 5;
    const int board_bottom = height - 4;
    render::StrokeRect(width, height, board_left, board_top, board_right,
                       board_bottom, {80, 80, 80}, rgb);
    for (int row = 0; row < tetris::kRows; ++row) {
      for (int col = 0; col < tetris::kCols; ++col) {
        const bool filled = grid_[tetris::Offset(row, col)] != 0;
        const int left =
            board_left + col * (board_right - board_left) / tetris::kCols;
        const int right =
            board_left + (col + 1) * (board_right - board_left) / tetris::kCols;
        const int top =
            board_top + row * (board_bottom - board_top) / tetris::kRows;
        const int bottom =
            board_top + (row + 1) * (board_bottom - board_top) / tetris::kRows;
        if (filled) {
          render::FillRect(width, height, left + 1, top + 1, right - 1,
                           bottom - 1, {31, 119, 180}, rgb);
        }
        render::StrokeRect(width, height, left, top, right, bottom,
                           {225, 225, 225}, rgb);
      }
    }
    const int block = std::max(4, preview_w / 6);
    const int px = width / 2 - 2 * block;
    const int py = height / 18;
    for (int row = 0; row < 4; ++row) {
      for (int col = 0; col < 4; ++col) {
        if (tetromino_[row * 4 + col] != 0) {
          render::FillRect(width, height, px + col * block, py + row * block,
                           px + (col + 1) * block, py + (row + 1) * block,
                           {255, 127, 14}, rgb);
        }
      }
    }
  }

  void Reset() override {
    grid_ = use_configured_grid_ ? configured_grid_ : tetris::Grid{};
    tetromino_ = configured_tetromino_;
    step_count_ = 0;
    done_ = false;
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    const int col = std::clamp(static_cast<int>(action["action"_](0, 1)), 0,
                               tetris::kCols - 1);
    const bool valid = CanPlace(col);
    float reward = 0.0f;
    if (valid) {
      const int row = DropRow(col);
      for (int dr = 0; dr < 2; ++dr) {
        for (int dc = 0; dc < 2; ++dc) {
          grid_[tetris::Offset(row + dr, col + dc)] = 1;
        }
      }
      reward = static_cast<float>(ClearFullRows());
    } else {
      reward = -1.0f;
    }
    ++step_count_;
    if (use_replay_ && step_count_ <= tetris::kReplaySteps) {
      for (int i = 0; i < tetris::kRows * tetris::kCols; ++i) {
        grid_[i] =
            replay_grids_[(step_count_ - 1) * tetris::kRows * tetris::kCols +
                          i];
      }
      for (int i = 0; i < 16; ++i) {
        tetromino_[i] = replay_tetrominoes_[(step_count_ - 1) * 16 + i];
      }
    }
    done_ = !valid || !HasValidPlacement() || step_count_ >= tetris::kTimeLimit;
    if (use_replay_ && step_count_ <= tetris::kReplaySteps) {
      done_ = replay_done_[step_count_ - 1];
    }
    WriteState(reward);
  }

 private:
  bool CanPlace(int col) const {
    return col + 1 < tetris::kCols && DropRow(col) >= 0;
  }

  int DropRow(int col) const {
    if (col + 1 >= tetris::kCols) {
      return -1;
    }
    for (int row = tetris::kRows - 2; row >= 0; --row) {
      bool collision = false;
      for (int dr = 0; dr < 2; ++dr) {
        for (int dc = 0; dc < 2; ++dc) {
          if (grid_[tetris::Offset(row + dr, col + dc)] != 0) {
            collision = true;
          }
        }
      }
      if (!collision) {
        return row;
      }
    }
    return -1;
  }

  bool HasValidPlacement() const {
    for (int col = 0; col < tetris::kCols; ++col) {
      if (CanPlace(col)) {
        return true;
      }
    }
    return false;
  }

  int ClearFullRows() {
    int cleared = 0;
    for (int row = tetris::kRows - 1; row >= 0; --row) {
      bool full = true;
      for (int col = 0; col < tetris::kCols; ++col) {
        full = full && grid_[tetris::Offset(row, col)] != 0;
      }
      if (!full) {
        continue;
      }
      ++cleared;
      for (int pull = row; pull > 0; --pull) {
        for (int col = 0; col < tetris::kCols; ++col) {
          grid_[tetris::Offset(pull, col)] =
              grid_[tetris::Offset(pull - 1, col)];
        }
      }
      for (int col = 0; col < tetris::kCols; ++col) {
        grid_[tetris::Offset(0, col)] = 0;
      }
      ++row;
    }
    return cleared;
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int row = 0; row < tetris::kRows; ++row) {
      for (int col = 0; col < tetris::kCols; ++col) {
        state["obs:grid"_](row, col) = grid_[tetris::Offset(row, col)];
      }
    }
    for (int row = 0; row < 4; ++row) {
      for (int col = 0; col < 4; ++col) {
        state["obs:tetromino"_](row, col) = tetromino_[row * 4 + col];
      }
    }
    for (int rotation = 0; rotation < 4; ++rotation) {
      for (int col = 0; col < tetris::kCols; ++col) {
        state["obs:action_mask"_](rotation, col) =
            use_replay_ && step_count_ > 0 &&
                    step_count_ <= tetris::kReplaySteps
                ? replay_action_masks_[((step_count_ - 1) * 4 + rotation) *
                                           tetris::kCols +
                                       col]
            : use_configured_action_mask_ && step_count_ == 0
                ? configured_action_mask_[rotation * tetris::kCols + col]
                : CanPlace(col);
      }
    }
    state["obs:step_count"_] = use_replay_ ? 0 : step_count_;
    state["reward"_] = reward;
  }
};

using TetrisEnvPool = AsyncEnvPool<TetrisEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_TETRIS_ENV_H_
