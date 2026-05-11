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

#ifndef ENVPOOL_JUMANJI_SLIDING_TILE_PUZZLE_ENV_H_
#define ENVPOOL_JUMANJI_SLIDING_TILE_PUZZLE_ENV_H_

#include <algorithm>
#include <array>
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
namespace sliding_tile_puzzle {

constexpr int kGridSize = 5;
constexpr int kCellCount = kGridSize * kGridSize;
constexpr int kTimeLimit = 500;
constexpr std::array<std::array<int, 2>, 4> kMoves = {
    {{{-1, 0}}, {{0, 1}}, {{1, 0}}, {{0, -1}}}};  // NOLINT

using Puzzle = std::array<int, kCellCount>;

inline int Offset(int row, int col) { return row * kGridSize + col; }

inline Puzzle SolvedPuzzle() {
  Puzzle puzzle{};
  for (int i = 0; i < kCellCount - 1; ++i) {
    puzzle[i] = i + 1;
  }
  puzzle[kCellCount - 1] = 0;
  return puzzle;
}

inline Puzzle ParsePuzzle(const std::string& text) {
  Puzzle puzzle = SolvedPuzzle();
  if (text.empty()) {
    return puzzle;
  }
  std::stringstream stream(text);
  std::string token;
  int index = 0;
  while (std::getline(stream, token, ',') && index < kCellCount) {
    puzzle[index++] = std::stoi(token);
  }
  return puzzle;
}

inline std::pair<int, int> FindEmpty(const Puzzle& puzzle) {
  for (int row = 0; row < kGridSize; ++row) {
    for (int col = 0; col < kGridSize; ++col) {
      if (puzzle[Offset(row, col)] == 0) {
        return {row, col};
      }
    }
  }
  return {kGridSize - 1, kGridSize - 1};
}

inline bool InGrid(int row, int col) {
  return 0 <= row && row < kGridSize && 0 <= col && col < kGridSize;
}

inline int CountCorrect(const Puzzle& puzzle) {
  const Puzzle solved = SolvedPuzzle();
  int correct = 0;
  for (int i = 0; i < kCellCount; ++i) {
    correct += puzzle[i] == solved[i] ? 1 : 0;
  }
  return correct;
}

}  // namespace sliding_tile_puzzle

class SlidingTilePuzzleEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("sliding_tile_initial_puzzle"_.Bind(std::string("")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict(
        "obs:puzzle"_.Bind(Spec<int>({5, 5}, {0, 24})),
        "obs:empty_tile_position"_.Bind(Spec<int>({2}, {0, 4})),
        "obs:action_mask"_.Bind(Spec<bool>({4}, {false, true})),
        "obs:step_count"_.Bind(Spec<int>({}, {0, 500})),
        "info:prop_correctly_placed"_.Bind(Spec<float>({}, {0.0f, 1.0f})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 3})));
  }
};

using SlidingTilePuzzleEnvSpec = EnvSpec<SlidingTilePuzzleEnvFns>;

class SlidingTilePuzzleEnv : public Env<SlidingTilePuzzleEnvSpec>,
                             public RenderableEnv {
 protected:
  sliding_tile_puzzle::Puzzle puzzle_{};
  sliding_tile_puzzle::Puzzle initial_puzzle_{};
  bool use_initial_puzzle_;
  int empty_row_{sliding_tile_puzzle::kGridSize - 1};
  int empty_col_{sliding_tile_puzzle::kGridSize - 1};
  int step_count_{0};
  bool done_{true};

 public:
  using Spec = SlidingTilePuzzleEnvSpec;
  using Action = typename Env<SlidingTilePuzzleEnvSpec>::Action;

  SlidingTilePuzzleEnv(const Spec& spec, int env_id)
      : Env<SlidingTilePuzzleEnvSpec>(spec, env_id),
        initial_puzzle_(sliding_tile_puzzle::ParsePuzzle(
            spec.config["sliding_tile_initial_puzzle"_])),
        use_initial_puzzle_(
            !spec.config["sliding_tile_initial_puzzle"_].empty()) {}

  bool IsDone() override { return done_; }

  int CurrentMaxEpisodeSteps() const override {
    return sliding_tile_puzzle::kTimeLimit + 1;
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, render::kWhite, rgb);
    for (int row = 0; row < sliding_tile_puzzle::kGridSize; ++row) {
      for (int col = 0; col < sliding_tile_puzzle::kGridSize; ++col) {
        const int tile = puzzle_[sliding_tile_puzzle::Offset(row, col)];
        const int left = col * width / sliding_tile_puzzle::kGridSize;
        const int right = (col + 1) * width / sliding_tile_puzzle::kGridSize;
        const int top = row * height / sliding_tile_puzzle::kGridSize;
        const int bottom = (row + 1) * height / sliding_tile_puzzle::kGridSize;
        if (tile == 0) {
          render::FillRect(width, height, left + 1, top + 1, right - 1,
                           bottom - 1, {48, 54, 61}, rgb);
        } else {
          render::Color color = render::Blend(
              {224, 228, 255}, {36, 74, 235},
              static_cast<float>(tile) / sliding_tile_puzzle::kCellCount);
          render::FillRect(width, height, left + 1, top + 1, right - 1,
                           bottom - 1, color, rgb);
          render::DrawNumber(width, height, tile, left + 7, top + 7, right - 7,
                             bottom - 7, {25, 25, 40}, rgb);
        }
      }
    }
    render::DrawGrid(width, height, sliding_tile_puzzle::kGridSize,
                     sliding_tile_puzzle::kGridSize, {150, 150, 150}, rgb);
  }

  void Reset() override {
    puzzle_ = use_initial_puzzle_ ? initial_puzzle_
                                  : sliding_tile_puzzle::SolvedPuzzle();
    auto [row, col] = sliding_tile_puzzle::FindEmpty(puzzle_);
    empty_row_ = row;
    empty_col_ = col;
    if (!use_initial_puzzle_) {
      RandomWalk(200);
    }
    step_count_ = 0;
    done_ = puzzle_ == sliding_tile_puzzle::SolvedPuzzle();
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    const int action_id = std::clamp(static_cast<int>(action["action"_]), 0, 3);
    const sliding_tile_puzzle::Puzzle before = puzzle_;
    ApplyMove(action_id);
    ++step_count_;
    const float reward = DenseReward(before, puzzle_);
    done_ = puzzle_ == sliding_tile_puzzle::SolvedPuzzle() ||
            step_count_ >= sliding_tile_puzzle::kTimeLimit;
    WriteState(reward);
  }

 private:
  void RandomWalk(int num_moves) {
    for (int i = 0; i < num_moves; ++i) {
      std::vector<int> valid;
      for (int action = 0; action < 4; ++action) {
        const int row = empty_row_ + sliding_tile_puzzle::kMoves[action][0];
        const int col = empty_col_ + sliding_tile_puzzle::kMoves[action][1];
        if (sliding_tile_puzzle::InGrid(row, col)) {
          valid.push_back(action);
        }
      }
      std::uniform_int_distribution<int> dist(
          0, static_cast<int>(valid.size()) - 1);
      ApplyMove(valid[dist(gen_)]);
    }
  }

  void ApplyMove(int action_id) {
    const int row = empty_row_ + sliding_tile_puzzle::kMoves[action_id][0];
    const int col = empty_col_ + sliding_tile_puzzle::kMoves[action_id][1];
    if (!sliding_tile_puzzle::InGrid(row, col)) {
      return;
    }
    std::swap(puzzle_[sliding_tile_puzzle::Offset(empty_row_, empty_col_)],
              puzzle_[sliding_tile_puzzle::Offset(row, col)]);
    empty_row_ = row;
    empty_col_ = col;
  }

  static float DenseReward(const sliding_tile_puzzle::Puzzle& before,
                           const sliding_tile_puzzle::Puzzle& after) {
    const auto solved = sliding_tile_puzzle::SolvedPuzzle();
    int new_correct = 0;
    int new_incorrect = 0;
    for (int i = 0; i < sliding_tile_puzzle::kCellCount; ++i) {
      new_correct += after[i] == solved[i] && before[i] != solved[i] ? 1 : 0;
      new_incorrect += after[i] != solved[i] && before[i] == solved[i] ? 1 : 0;
    }
    return static_cast<float>(new_correct - new_incorrect);
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int row = 0; row < sliding_tile_puzzle::kGridSize; ++row) {
      for (int col = 0; col < sliding_tile_puzzle::kGridSize; ++col) {
        state["obs:puzzle"_](row, col) =
            puzzle_[sliding_tile_puzzle::Offset(row, col)];
      }
    }
    state["obs:empty_tile_position"_][0] = empty_row_;
    state["obs:empty_tile_position"_][1] = empty_col_;
    for (int action = 0; action < 4; ++action) {
      const int row = empty_row_ + sliding_tile_puzzle::kMoves[action][0];
      const int col = empty_col_ + sliding_tile_puzzle::kMoves[action][1];
      state["obs:action_mask"_][action] = sliding_tile_puzzle::InGrid(row, col);
    }
    state["obs:step_count"_] = step_count_;
    state["info:prop_correctly_placed"_] =
        static_cast<float>(sliding_tile_puzzle::CountCorrect(puzzle_)) /
        sliding_tile_puzzle::kCellCount;
    state["reward"_] = reward;
  }
};

using SlidingTilePuzzleEnvPool = AsyncEnvPool<SlidingTilePuzzleEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_SLIDING_TILE_PUZZLE_ENV_H_
