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

#ifndef ENVPOOL_JUMANJI_GAME2048_ENV_H_
#define ENVPOOL_JUMANJI_GAME2048_ENV_H_

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
#include "envpool/jumanji/parse_utils.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace game2048 {

constexpr int kBoardSize = 4;
constexpr int kCellCount = kBoardSize * kBoardSize;
constexpr int kReplaySteps = 32;

using Board = std::array<int, kCellCount>;

inline int Offset(int row, int col) { return row * kBoardSize + col; }

inline Board ParseBoard(const std::string& text) {
  Board board{};
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

inline float MoveLineLeft(std::array<int, kBoardSize>* line) {
  std::array<int, kBoardSize> compact{};
  int compact_size = 0;
  for (int value : *line) {
    if (value != 0) {
      compact[compact_size++] = value;
    }
  }
  std::array<int, kBoardSize> moved{};
  int out = 0;
  float reward = 0.0f;
  for (int i = 0; i < compact_size; ++i) {
    if (i + 1 < compact_size && compact[i] == compact[i + 1]) {
      const int merged = compact[i] + 1;
      moved[out++] = merged;
      reward += std::ldexp(1.0f, merged);
      ++i;
    } else {
      moved[out++] = compact[i];
    }
  }
  *line = moved;
  return reward;
}

inline float Move(Board* board, int action) {
  Board next = *board;
  float reward = 0.0f;
  for (int i = 0; i < kBoardSize; ++i) {
    std::array<int, kBoardSize> line{};
    for (int j = 0; j < kBoardSize; ++j) {
      if (action == 0) {
        line[j] = (*board)[Offset(j, i)];
      } else if (action == 1) {
        line[j] = (*board)[Offset(i, kBoardSize - 1 - j)];
      } else if (action == 2) {
        line[j] = (*board)[Offset(kBoardSize - 1 - j, i)];
      } else {
        line[j] = (*board)[Offset(i, j)];
      }
    }
    reward += MoveLineLeft(&line);
    for (int j = 0; j < kBoardSize; ++j) {
      if (action == 0) {
        next[Offset(j, i)] = line[j];
      } else if (action == 1) {
        next[Offset(i, kBoardSize - 1 - j)] = line[j];
      } else if (action == 2) {
        next[Offset(kBoardSize - 1 - j, i)] = line[j];
      } else {
        next[Offset(i, j)] = line[j];
      }
    }
  }
  *board = next;
  return reward;
}

inline bool CanMove(const Board& board, int action) {
  Board moved = board;
  Move(&moved, action);
  return moved != board;
}

inline int HighestTile(const Board& board) {
  int exponent = 0;
  for (int value : board) {
    exponent = std::max(exponent, value);
  }
  return exponent == 0 ? 1 : (1 << exponent);
}

}  // namespace game2048

class Game2048EnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("game2048_initial_board"_.Bind(std::string("")),
                    "game2048_replay_boards"_.Bind(std::string("")),
                    "game2048_add_random_cell"_.Bind(true));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("obs:board"_.Bind(Spec<int>({4, 4})),
                    "obs:action_mask"_.Bind(Spec<bool>({4}, {false, true})),
                    "info:highest_tile"_.Bind(Spec<int>({}, {1, 1 << 30})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 3})));
  }
};

using Game2048EnvSpec = EnvSpec<Game2048EnvFns>;

class Game2048Env : public Env<Game2048EnvSpec>, public RenderableEnv {
 protected:
  game2048::Board board_{};
  game2048::Board initial_board_{};
  std::array<int, game2048::kReplaySteps * game2048::kCellCount>
      replay_boards_{};
  bool use_initial_board_;
  bool use_replay_boards_;
  bool add_random_cell_;
  bool done_{true};
  int step_count_{0};

 public:
  using Spec = Game2048EnvSpec;
  using Action = typename Env<Game2048EnvSpec>::Action;

  Game2048Env(const Spec& spec, int env_id)
      : Env<Game2048EnvSpec>(spec, env_id),
        initial_board_(
            game2048::ParseBoard(spec.config["game2048_initial_board"_])),
        replay_boards_(
            parse::CsvArray<int, game2048::kReplaySteps * game2048::kCellCount>(
                spec.config["game2048_replay_boards"_])),
        use_initial_board_(!spec.config["game2048_initial_board"_].empty()),
        use_replay_boards_(!spec.config["game2048_replay_boards"_].empty()),
        add_random_cell_(spec.config["game2048_add_random_cell"_]) {}

  bool IsDone() override { return done_; }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, {187, 173, 160}, rgb);
    for (int row = 0; row < game2048::kBoardSize; ++row) {
      for (int col = 0; col < game2048::kBoardSize; ++col) {
        const int value = board_[game2048::Offset(row, col)];
        const int left = col * width / game2048::kBoardSize + 3;
        const int right = (col + 1) * width / game2048::kBoardSize - 3;
        const int top = row * height / game2048::kBoardSize + 3;
        const int bottom = (row + 1) * height / game2048::kBoardSize - 3;
        render::Color color = {205, 193, 180};
        if (value == 1) {
          color = {238, 228, 218};
        } else if (value == 2) {
          color = {237, 224, 200};
        } else if (value > 2) {
          color = render::Blend({242, 177, 121}, {237, 94, 66},
                                std::min(1.0f, (value - 3) / 8.0f));
        }
        render::FillRect(width, height, left, top, right, bottom, color, rgb);
        if (value > 0) {
          render::DrawNumber(width, height, 1 << value, left + 8, top + 8,
                             right - 8, bottom - 8, {90, 80, 70}, rgb);
        }
      }
    }
  }

  void Reset() override {
    board_ = use_initial_board_ ? initial_board_ : game2048::Board{};
    if (!use_initial_board_) {
      AddRandomCell();
    }
    step_count_ = 0;
    done_ = !AnyActionAvailable();
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    const int action_id = std::clamp(static_cast<int>(action["action"_]), 0, 3);
    float reward = 0.0f;
    if (CanMove(action_id)) {
      reward = game2048::Move(&board_, action_id);
      if (add_random_cell_) {
        AddRandomCell();
      }
    }
    ++step_count_;
    if (use_replay_boards_ && step_count_ <= game2048::kReplaySteps) {
      for (int i = 0; i < game2048::kCellCount; ++i) {
        board_[i] =
            replay_boards_[(step_count_ - 1) * game2048::kCellCount + i];
      }
    }
    done_ = !AnyActionAvailable();
    WriteState(reward);
  }

 private:
  bool CanMove(int action_id) const {
    return game2048::CanMove(board_, action_id);
  }

  bool AnyActionAvailable() const {
    for (int action = 0; action < 4; ++action) {
      if (CanMove(action)) {
        return true;
      }
    }
    return false;
  }

  void AddRandomCell() {
    std::vector<int> empty;
    for (int i = 0; i < game2048::kCellCount; ++i) {
      if (board_[i] == 0) {
        empty.push_back(i);
      }
    }
    if (empty.empty()) {
      return;
    }
    std::uniform_int_distribution<int> position_dist(
        0, static_cast<int>(empty.size()) - 1);
    std::bernoulli_distribution two_dist(0.1);
    board_[empty[position_dist(gen_)]] = two_dist(gen_) ? 2 : 1;
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int row = 0; row < game2048::kBoardSize; ++row) {
      for (int col = 0; col < game2048::kBoardSize; ++col) {
        state["obs:board"_](row, col) = board_[game2048::Offset(row, col)];
      }
    }
    for (int action = 0; action < 4; ++action) {
      state["obs:action_mask"_][action] = CanMove(action);
    }
    state["reward"_] = reward;
    state["info:highest_tile"_] = game2048::HighestTile(board_);
  }
};

using Game2048EnvPool = AsyncEnvPool<Game2048Env>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_GAME2048_ENV_H_
