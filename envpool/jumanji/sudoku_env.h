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

#ifndef ENVPOOL_JUMANJI_SUDOKU_ENV_H_
#define ENVPOOL_JUMANJI_SUDOKU_ENV_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/npy_utils.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace sudoku {

constexpr int kBoardWidth = 9;
constexpr int kCellCount = kBoardWidth * kBoardWidth;

using Board = std::array<int, kCellCount>;

struct Dataset {
  std::vector<Board> boards;
};

inline int Offset(int row, int col) { return row * kBoardWidth + col; }

inline Board SampleBoard() {
  return Board{{
      -1, -1, -1, 7,  -1, 0,  -1, -1, -1,  //
      -1, -1, -1, -1, -1, -1, -1, 3,  2,   //
      4,  -1, -1, -1, -1, -1, -1, -1, -1,  //
      -1, -1, -1, -1, 6,  -1, 7,  -1, -1,  //
      -1, -1, -1, -1, -1, -1, 0,  -1, -1,  //
      -1, 1,  -1, -1, 2,  -1, -1, -1, -1,  //
      5,  -1, -1, -1, -1, -1, -1, 6,  4,   //
      -1, -1, 2,  3,  -1, -1, -1, -1, -1,  //
      -1, -1, -1, 1,  -1, -1, 5,  -1, -1,
  }};
}

inline Board ParseBoard(const std::string& text) {
  Board board = SampleBoard();
  if (text.empty()) {
    return board;
  }
  std::stringstream stream(text);
  std::string token;
  int index = 0;
  while (std::getline(stream, token, ',') && index < kCellCount) {
    board[index++] = std::stoi(token);
  }
  return board;
}

inline std::string DatabaseFile(const std::string& name) {
  if (name == "mixed") {
    return "10000_mixed_puzzles.npy";
  }
  if (name == "very-easy") {
    return "1000_very_easy_puzzles.npy";
  }
  throw std::runtime_error("unknown Sudoku database: " + name);
}

inline Dataset LoadNpyDataset(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open Sudoku database: " + path);
  }
  std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                          std::istreambuf_iterator<char>());
  const std::size_t data_offset = npy::HeaderLength(bytes, "Sudoku");
  if (bytes.size() < data_offset ||
      (bytes.size() - data_offset) % kCellCount != 0) {
    throw std::runtime_error("unexpected Sudoku npy data size");
  }
  const std::size_t board_count = (bytes.size() - data_offset) / kCellCount;
  Dataset dataset;
  dataset.boards.reserve(board_count);
  for (std::size_t board_id = 0; board_id < board_count; ++board_id) {
    Board board;
    const std::size_t board_offset = data_offset + board_id * kCellCount;
    for (int cell = 0; cell < kCellCount; ++cell) {
      board[cell] = static_cast<int>(static_cast<unsigned char>(
                        bytes[board_offset + cell])) -
                    1;
    }
    dataset.boards.push_back(board);
  }
  if (dataset.boards.empty()) {
    throw std::runtime_error("empty Sudoku database");
  }
  return dataset;
}

inline std::shared_ptr<const Dataset> GetDataset(const std::string& base_path,
                                                 const std::string& database) {
  return std::make_shared<Dataset>(LoadNpyDataset(
      base_path + "/jumanji/assets/sudoku/" + DatabaseFile(database)));
}

inline bool IsPlacementValid(const Board& board, int row, int col, int value) {
  if (board[Offset(row, col)] != -1) {
    return false;
  }
  for (int i = 0; i < kBoardWidth; ++i) {
    if (board[Offset(row, i)] == value || board[Offset(i, col)] == value) {
      return false;
    }
  }
  const int box_row = (row / 3) * 3;
  const int box_col = (col / 3) * 3;
  for (int r = box_row; r < box_row + 3; ++r) {
    for (int c = box_col; c < box_col + 3; ++c) {
      if (board[Offset(r, c)] == value) {
        return false;
      }
    }
  }
  return true;
}

inline bool IsSolved(const Board& board) {
  for (int i = 0; i < kBoardWidth; ++i) {
    std::array<bool, kBoardWidth> row_seen{};
    std::array<bool, kBoardWidth> col_seen{};
    for (int j = 0; j < kBoardWidth; ++j) {
      const int row_value = board[Offset(i, j)];
      const int col_value = board[Offset(j, i)];
      if (row_value < 0 || row_value >= kBoardWidth || col_value < 0 ||
          col_value >= kBoardWidth || row_seen[row_value] ||
          col_seen[col_value]) {
        return false;
      }
      row_seen[row_value] = true;
      col_seen[col_value] = true;
    }
  }
  for (int box = 0; box < kBoardWidth; ++box) {
    std::array<bool, kBoardWidth> seen{};
    const int box_row = (box / 3) * 3;
    const int box_col = (box % 3) * 3;
    for (int r = box_row; r < box_row + 3; ++r) {
      for (int c = box_col; c < box_col + 3; ++c) {
        const int value = board[Offset(r, c)];
        if (value < 0 || value >= kBoardWidth || seen[value]) {
          return false;
        }
        seen[value] = true;
      }
    }
  }
  return true;
}

}  // namespace sudoku

class SudokuEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("sudoku_initial_board"_.Bind(std::string("")),
                    "sudoku_database"_.Bind(std::string("mixed")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict(
        "obs:board"_.Bind(Spec<int>({9, 9}, {-1, 9})),
        "obs:action_mask"_.Bind(Spec<bool>({9, 9, 9}, {false, true})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1, 3}, {0, 8})));
  }
};

using SudokuEnvSpec = EnvSpec<SudokuEnvFns>;

class SudokuEnv : public Env<SudokuEnvSpec>, public RenderableEnv {
 protected:
  sudoku::Board board_{};
  sudoku::Board initial_board_{};
  std::shared_ptr<const sudoku::Dataset> dataset_;
  bool use_configured_board_;
  bool done_{true};

 public:
  using Spec = SudokuEnvSpec;
  using Action = typename Env<SudokuEnvSpec>::Action;

  SudokuEnv(const Spec& spec, int env_id)
      : Env<SudokuEnvSpec>(spec, env_id),
        initial_board_(
            sudoku::ParseBoard(spec.config["sudoku_initial_board"_])),
        dataset_(sudoku::GetDataset(spec.config["base_path"_],
                                    spec.config["sudoku_database"_])),
        use_configured_board_(!spec.config["sudoku_initial_board"_].empty()) {}

  bool IsDone() override { return done_; }

  int CurrentMaxEpisodeSteps() const override { return sudoku::kCellCount + 1; }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, render::kWhite, rgb);
    for (int row = 0; row < sudoku::kBoardWidth; ++row) {
      for (int col = 0; col < sudoku::kBoardWidth; ++col) {
        const int value = board_[sudoku::Offset(row, col)];
        render::FillCell(width, height, sudoku::kBoardWidth,
                         sudoku::kBoardWidth, row, col, {252, 252, 252}, rgb,
                         1);
        if (value >= 0) {
          const int left = col * width / sudoku::kBoardWidth;
          const int right = (col + 1) * width / sudoku::kBoardWidth;
          const int top = row * height / sudoku::kBoardWidth;
          const int bottom = (row + 1) * height / sudoku::kBoardWidth;
          render::DrawNumber(width, height, value + 1, left + 6, top + 5,
                             right - 6, bottom - 5, {70, 70, 70}, rgb);
        }
      }
    }
    render::DrawGrid(width, height, sudoku::kBoardWidth, sudoku::kBoardWidth,
                     {170, 170, 170}, rgb);
    for (int i = 0; i <= sudoku::kBoardWidth; i += 3) {
      const int x = i * width / sudoku::kBoardWidth;
      const int y = i * height / sudoku::kBoardWidth;
      render::FillRect(width, height, x - 1, 0, x + 1, height, {20, 20, 20},
                       rgb);
      render::FillRect(width, height, 0, y - 1, width, y + 1, {20, 20, 20},
                       rgb);
    }
  }

  void Reset() override {
    if (use_configured_board_) {
      board_ = initial_board_;
    } else {
      std::uniform_int_distribution<std::size_t> dist(
          0, dataset_->boards.size() - 1);
      board_ = dataset_->boards[dist(gen_)];
    }
    done_ = !AnyActionAvailable() || sudoku::IsSolved(board_);
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    const int row = std::clamp(static_cast<int>(action["action"_](0, 0)), 0, 8);
    const int col = std::clamp(static_cast<int>(action["action"_](0, 1)), 0, 8);
    const int value =
        std::clamp(static_cast<int>(action["action"_](0, 2)), 0, 8);
    const bool invalid = !sudoku::IsPlacementValid(board_, row, col, value);
    board_[sudoku::Offset(row, col)] = value;
    done_ = invalid || !AnyActionAvailable() || sudoku::IsSolved(board_);
    WriteState(sudoku::IsSolved(board_) ? 1.0f : 0.0f);
  }

 private:
  bool AnyActionAvailable() const {
    for (int row = 0; row < sudoku::kBoardWidth; ++row) {
      for (int col = 0; col < sudoku::kBoardWidth; ++col) {
        for (int value = 0; value < sudoku::kBoardWidth; ++value) {
          if (sudoku::IsPlacementValid(board_, row, col, value)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int row = 0; row < sudoku::kBoardWidth; ++row) {
      for (int col = 0; col < sudoku::kBoardWidth; ++col) {
        state["obs:board"_](row, col) = board_[sudoku::Offset(row, col)];
        for (int value = 0; value < sudoku::kBoardWidth; ++value) {
          state["obs:action_mask"_](row, col, value) =
              sudoku::IsPlacementValid(board_, row, col, value);
        }
      }
    }
    state["reward"_] = reward;
  }
};

using SudokuEnvPool = AsyncEnvPool<SudokuEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_SUDOKU_ENV_H_
