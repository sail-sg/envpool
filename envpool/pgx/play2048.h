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

#ifndef ENVPOOL_PGX_PLAY2048_H_
#define ENVPOOL_PGX_PLAY2048_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/pgx/board_games.h"

namespace pgx {
namespace play2048 {

constexpr int kSize = 4;
constexpr int kCells = kSize * kSize;
using Board = std::array<int, kCells>;

inline int Offset(int row, int col) { return row * kSize + col; }

inline int Power2(int exponent) { return 1 << exponent; }

inline int MoveLineLeft(std::array<int, kSize>* line) {
  std::array<int, kSize> compact{};
  int compact_size = 0;
  for (int value : *line) {
    if (value != 0) {
      compact[compact_size++] = value;
    }
  }
  std::array<int, kSize> moved{};
  int out = 0;
  int reward = 0;
  for (int i = 0; i < compact_size; ++i) {
    if (i + 1 < compact_size && compact[i] == compact[i + 1]) {
      const int merged = compact[i] + 1;
      moved[out++] = merged;
      reward += Power2(merged);
      ++i;
    } else {
      moved[out++] = compact[i];
    }
  }
  *line = moved;
  return reward;
}

inline int Move(Board* board, int action) {
  Board next = *board;
  int reward = 0;
  for (int i = 0; i < kSize; ++i) {
    std::array<int, kSize> line{};
    for (int j = 0; j < kSize; ++j) {
      if (action == 0) {
        line[j] = (*board)[Offset(i, j)];
      } else if (action == 1) {
        line[j] = (*board)[Offset(j, i)];
      } else if (action == 2) {
        line[j] = (*board)[Offset(i, kSize - 1 - j)];
      } else {
        line[j] = (*board)[Offset(kSize - 1 - j, i)];
      }
    }
    reward += MoveLineLeft(&line);
    for (int j = 0; j < kSize; ++j) {
      if (action == 0) {
        next[Offset(i, j)] = line[j];
      } else if (action == 1) {
        next[Offset(j, i)] = line[j];
      } else if (action == 2) {
        next[Offset(i, kSize - 1 - j)] = line[j];
      } else {
        next[Offset(kSize - 1 - j, i)] = line[j];
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

}  // namespace play2048

class Play2048EnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("task"_.Bind(std::string("2048")));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("obs"_.Bind(Spec<bool>({4, 4, 31})),
                    "info:board"_.Bind(Spec<int>({4, 4}, {0, 30})),
                    "info:legal_action_mask"_.Bind(Spec<bool>({4})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 3})));
  }
};

using Play2048EnvSpec = EnvSpec<Play2048EnvFns>;

class Play2048Env : public Env<Play2048EnvSpec>, public RenderableEnv {
 public:
  using Spec = Play2048EnvSpec;
  using Action = typename Env<Play2048EnvSpec>::Action;

  Play2048Env(const Spec& spec, int env_id)
      : Env<Play2048EnvSpec>(spec, env_id) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    board_ = {};
    AddRandomNum();
    AddRandomNum();
    done_ = false;
    UpdateLegalActionMask();
    done_ = !AnyLegal();
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    const int act = static_cast<int>(action["action"_]);
    const bool illegal =
        act < 0 || act > 3 || (act >= 0 && !legal_action_mask_[act]);
    float reward = 0.0f;
    if (act >= 0 && act <= 3) {
      reward = static_cast<float>(play2048::Move(&board_, act));
      AddRandomNum();
    }
    UpdateLegalActionMask();
    done_ = !AnyLegal();
    if (illegal) {
      done_ = true;
      reward = -1.0f;
    }
    if (done_) {
      std::fill(legal_action_mask_.begin(), legal_action_mask_.end(), true);
    }
    WriteState(reward);
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int camera_id,
              unsigned char* rgb) override {
    (void)camera_id;
    board_games::Fill(rgb, width, height, {187, 173, 160});
    for (int row = 0; row < play2048::kSize; ++row) {
      for (int col = 0; col < play2048::kSize; ++col) {
        const int value = board_[play2048::Offset(row, col)];
        const int left = col * width / play2048::kSize + 3;
        const int right = (col + 1) * width / play2048::kSize - 3;
        const int top = row * height / play2048::kSize + 3;
        const int bottom = (row + 1) * height / play2048::kSize - 3;
        board_games::Rgb color{205, 193, 180};
        if (value == 1) {
          color = {238, 228, 218};
        } else if (value == 2) {
          color = {237, 224, 200};
        } else if (value > 2) {
          const int red = std::max(80, 242 - value * 10);
          const int green = std::max(70, 177 - value * 8);
          color = {static_cast<unsigned char>(red),
                   static_cast<unsigned char>(green), 90};
        }
        board_games::FillRect(rgb, width, height, left, top, right, bottom,
                              color);
      }
    }
  }

 private:
  play2048::Board board_{};
  std::array<bool, 4> legal_action_mask_{true, true, true, true};
  bool done_{true};

  void AddRandomNum() {
    std::vector<int> empty;
    for (int i = 0; i < play2048::kCells; ++i) {
      if (board_[i] == 0) {
        empty.push_back(i);
      }
    }
    if (empty.empty()) {
      return;
    }
    std::uniform_int_distribution<int> pos_dist(
        0, static_cast<int>(empty.size()) - 1);
    std::bernoulli_distribution four_dist(0.1);
    board_[empty[pos_dist(gen_)]] = four_dist(gen_) ? 2 : 1;
  }

  void UpdateLegalActionMask() {
    for (int action = 0; action < 4; ++action) {
      legal_action_mask_[action] = play2048::CanMove(board_, action);
    }
  }

  bool AnyLegal() const {
    return std::any_of(legal_action_mask_.begin(), legal_action_mask_.end(),
                       [](bool value) { return value; });
  }

  void WriteState(float reward) {
    State state = Allocate();
    for (int row = 0; row < play2048::kSize; ++row) {
      for (int col = 0; col < play2048::kSize; ++col) {
        const int cell = play2048::Offset(row, col);
        state["info:board"_](row, col) = board_[cell];
        state["info:legal_action_mask"_][col] = legal_action_mask_[col];
        for (int channel = 0; channel < 31; ++channel) {
          state["obs"_](row, col, channel) = board_[cell] == channel;
        }
      }
    }
    state["reward"_] = reward;
  }
};

using Play2048EnvPool = AsyncEnvPool<Play2048Env>;

}  // namespace pgx

#endif  // ENVPOOL_PGX_PLAY2048_H_
