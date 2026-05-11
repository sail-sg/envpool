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

#ifndef ENVPOOL_PGX_ANIMAL_SHOGI_H_
#define ENVPOOL_PGX_ANIMAL_SHOGI_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/pgx/board_games.h"

namespace pgx {
namespace animal_shogi {

constexpr int kEmpty = -1;
constexpr int kPawn = 0;
constexpr int kBishop = 1;
constexpr int kRook = 2;
constexpr int kKing = 3;
constexpr int kGold = 4;
constexpr int kMaxSteps = 256;
constexpr std::array<int, 12> kInitBoard = {6, -1, -1, 2,  8,  5,
                                            0, 3,  7,  -1, -1, 1};
constexpr std::array<int, 8> kDx = {-1, -1, -1, 0, 0, 1, 1, 1};
constexpr std::array<int, 8> kDy = {-1, 0, 1, -1, 1, -1, 0, 1};

using Board = std::array<int, 12>;
using Hand = std::array<std::array<int, 3>, 2>;
using BoardHistory = std::array<int, 8 * 12>;
using HandHistory = std::array<int, 8 * 6>;

struct DecodedAction {
  bool is_drop;
  int from;
  int to;
  int drop_piece;
};

inline int To(int from, int dir) {
  const int x = from / 4;
  const int y = from % 4;
  const int nx = x + kDx[dir];
  const int ny = y + kDy[dir];
  if (nx < 0 || nx >= 3 || ny < 0 || ny >= 4) {
    return -1;
  }
  return nx * 4 + ny;
}

inline DecodedAction Decode(int label) {
  const int x = label / 12;
  const int sq = label % 12;
  if (x >= 8) {
    return {true, -1, sq, x - 8};
  }
  return {false, sq, To(sq, x), -1};
}

inline bool CanMove(int piece, int from, int to) {
  const int x0 = from / 4;
  const int y0 = from % 4;
  const int x1 = to / 4;
  const int y1 = to % 4;
  const int dx = x1 - x0;
  const int dy = y1 - y0;
  const bool is_neighbour =
      ((dx != 0) || (dy != 0)) && std::abs(dx) <= 1 && std::abs(dy) <= 1;
  if (piece == kPawn) {
    return dx == 0 && dy == -1;
  }
  if (piece == kBishop) {
    return is_neighbour && (dx == dy || dx == -dy);
  }
  if (piece == kRook) {
    return is_neighbour && (dx == 0 || dy == 0);
  }
  if (piece == kKing) {
    return is_neighbour;
  }
  return is_neighbour && (dx == 0 || dy != 1);
}

template <std::size_t N>
inline void FlatRoll(std::array<int, N>* values, int shift) {
  std::array<int, N> old = *values;
  for (std::size_t i = 0; i < N; ++i) {
    (*values)[(i + shift) % N] = old[i];
  }
}

inline int HandIndexForCapture(int captured) { return (captured % 5) % 4; }

}  // namespace animal_shogi

class AnimalShogiEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("task"_.Bind(std::string("animal_shogi")));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("obs"_.Bind(Spec<float>({-1, 4, 3, 194})),
                    "info:board"_.Bind(Spec<int>({4, 3}, {-1, 9})),
                    "info:current_player"_.Bind(Spec<int>({}, {0, 1})),
                    "info:hand"_.Bind(Spec<int>({2, 3})),
                    "info:legal_action_mask"_.Bind(Spec<bool>({132})),
                    "info:players.id"_.Bind(Spec<int>({-1}, {0, 1})),
                    "info:turn"_.Bind(Spec<int>({}, {0, 1})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 131})));
  }
};

using AnimalShogiEnvSpec = EnvSpec<AnimalShogiEnvFns>;

class AnimalShogiEnv : public Env<AnimalShogiEnvSpec>, public RenderableEnv {
 public:
  using Spec = AnimalShogiEnvSpec;
  using Action = typename Env<AnimalShogiEnvSpec>::Action;

  AnimalShogiEnv(const Spec& spec, int env_id)
      : Env<AnimalShogiEnvSpec>(spec, env_id) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    current_player_ = static_cast<int>(gen_() & 1U);
    turn_ = 0;
    step_count_ = 0;
    board_ = animal_shogi::kInitBoard;
    hand_ = {};
    std::fill(board_history_.begin(), board_history_.end(),
              animal_shogi::kEmpty);
    for (int i = 0; i < 12; ++i) {
      board_history_[i] = board_[i];
    }
    std::fill(hand_history_.begin(), hand_history_.end(), 0);
    std::fill(rep_history_.begin(), rep_history_.end(), 0);
    done_ = false;
    rewards_ = {0.0f, 0.0f};
    UpdateLegalActionMask();
    WriteState(rewards_);
  }

  void Step(const Action& action) override {
    const int act = action["action"_][0];
    const int loser = current_player_;
    const bool illegal =
        act < 0 || act >= 132 || (act >= 0 && !legal_action_mask_[act]);
    if (act >= 0 && act < 132) {
      StepGame(act);
    }
    std::array<float, 2> rewards = rewards_;
    if (illegal) {
      done_ = true;
      std::fill(legal_action_mask_.begin(), legal_action_mask_.end(), true);
      rewards = board_games::IllegalRewards(loser);
    } else if (done_) {
      std::fill(legal_action_mask_.begin(), legal_action_mask_.end(), true);
    }
    WriteState(rewards);
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 180, height > 0 ? height : 240};
  }

  void Render(int width, int height, int camera_id,
              unsigned char* rgb) override {
    (void)camera_id;
    board_games::DrawGrid(rgb, width, height, 4, 3);
    const int radius = std::max(3, std::min(width / 3, height / 4) / 4);
    for (int sq = 0; sq < 12; ++sq) {
      if (board_[sq] == animal_shogi::kEmpty) {
        continue;
      }
      const auto [row, col] = RenderCoord(sq);
      const int cx = col * width / 3 + width / 6;
      const int cy = row * height / 4 + height / 8;
      const bool mine = board_[sq] < 5;
      board_games::DrawCircle(
          rgb, width, height, cx, cy, radius,
          mine ? board_games::Rgb{40, 40, 40} : board_games::Rgb{220, 70, 70});
    }
  }

 private:
  animal_shogi::Board board_{};
  animal_shogi::Hand hand_{};
  animal_shogi::BoardHistory board_history_{};
  animal_shogi::HandHistory hand_history_{};
  std::array<int, 8> rep_history_{};
  std::array<bool, 132> legal_action_mask_{};
  std::array<float, 2> rewards_{0.0f, 0.0f};
  int current_player_{0};
  int turn_{0};
  int step_count_{0};
  bool done_{true};

  static std::pair<int, int> RenderCoord(int sq) {
    const int x = sq / 4;
    const int y = sq % 4;
    return {y, 2 - x};
  }

  void StepGame(int label) {
    ++step_count_;
    const animal_shogi::DecodedAction action = animal_shogi::Decode(label);
    if (action.is_drop) {
      ApplyDrop(action, &board_, &hand_);
    } else {
      ApplyMove(action, &board_, &hand_);
    }
    const bool is_try = board_[0] == animal_shogi::kKing ||
                        board_[4] == animal_shogi::kKing ||
                        board_[8] == animal_shogi::kKing;

    animal_shogi::FlatRoll(&board_history_, 8);
    for (int i = 0; i < 12; ++i) {
      board_history_[i] = board_[i];
    }
    animal_shogi::FlatRoll(&hand_history_, 8);
    WriteCurrentHandHistoryRow();

    Flip();
    const int rep = RepetitionCount() - 1;
    const bool is_rep_draw = rep >= 2;
    UpdateLegalActionMask();
    const bool any_legal =
        std::any_of(legal_action_mask_.begin(), legal_action_mask_.end(),
                    [](bool value) { return value; });
    done_ = !any_legal || is_try || is_rep_draw ||
            step_count_ >= animal_shogi::kMaxSteps;
    rewards_ = {0.0f, 0.0f};
    if (done_ && !is_rep_draw && step_count_ < animal_shogi::kMaxSteps) {
      rewards_ = {1.0f, 1.0f};
      rewards_[current_player_] = -1.0f;
    }
    RollRepHistory(rep);
  }

  static void ApplyMove(const animal_shogi::DecodedAction& action,
                        animal_shogi::Board* board, animal_shogi::Hand* hand) {
    int piece = (*board)[action.from];
    (*board)[action.from] = animal_shogi::kEmpty;
    const int captured = (*board)[action.to];
    if (captured != animal_shogi::kEmpty) {
      const int hand_index = animal_shogi::HandIndexForCapture(captured);
      if (hand_index >= 0 && hand_index < 3) {
        ++(*hand)[0][hand_index];
      }
    }
    if (action.from % 4 == 1 && piece == animal_shogi::kPawn) {
      piece = animal_shogi::kGold;
    }
    (*board)[action.to] = piece;
  }

  static void ApplyDrop(const animal_shogi::DecodedAction& action,
                        animal_shogi::Board* board, animal_shogi::Hand* hand) {
    (*board)[action.to] = action.drop_piece;
    --(*hand)[0][action.drop_piece];
  }

  void Flip() {
    for (int& piece : board_) {
      if (piece != animal_shogi::kEmpty) {
        piece = (piece + 5) % 10;
      }
    }
    std::reverse(board_.begin(), board_.end());
    std::swap(hand_[0], hand_[1]);
    for (int& piece : board_history_) {
      if (piece != animal_shogi::kEmpty) {
        piece = (piece + 5) % 10;
      }
    }
    for (int row = 0; row < 8; ++row) {
      std::reverse(board_history_.begin() + row * 12,
                   board_history_.begin() + (row + 1) * 12);
      std::rotate(hand_history_.begin() + row * 6,
                  hand_history_.begin() + row * 6 + 3,
                  hand_history_.begin() + (row + 1) * 6);
    }
    current_player_ = 1 - current_player_;
    turn_ = 1 - turn_;
  }

  void WriteCurrentHandHistoryRow() {
    for (int i = 0; i < 3; ++i) {
      hand_history_[i] = hand_[0][i];
      hand_history_[3 + i] = hand_[1][i];
    }
  }

  int RepetitionCount() const {
    int count = 0;
    for (int row = 0; row < 8; ++row) {
      bool same_board = true;
      for (int i = 0; i < 12; ++i) {
        same_board = same_board && board_history_[row * 12 + i] == board_[i];
      }
      bool same_hand = true;
      for (int i = 0; i < 6; ++i) {
        const int hand_value = i < 3 ? hand_[0][i] : hand_[1][i - 3];
        same_hand = same_hand && hand_history_[row * 6 + i] == hand_value;
      }
      if (same_board && same_hand) {
        ++count;
      }
    }
    return count;
  }

  void RollRepHistory(int rep) {
    for (int i = 7; i > 0; --i) {
      rep_history_[i] = rep_history_[i - 1];
    }
    rep_history_[0] = rep;
  }

  bool IsChecked(const animal_shogi::Board& board) const {
    int king_pos = 0;
    int best = 100;
    for (int i = 0; i < 12; ++i) {
      const int dist = std::abs(board[i] - animal_shogi::kKing);
      if (dist < best) {
        best = dist;
        king_pos = i;
      }
    }
    for (int from = 0; from < 12; ++from) {
      const int piece = board[from];
      if (piece >= 5 && animal_shogi::CanMove(piece % 5, king_pos, from)) {
        return true;
      }
    }
    return false;
  }

  bool IsLegalMove(const animal_shogi::DecodedAction& action) const {
    if (action.to == -1) {
      return false;
    }
    const int piece = board_[action.from];
    bool ok = animal_shogi::kPawn <= piece && piece <= animal_shogi::kGold;
    ok = ok && (board_[action.to] == animal_shogi::kEmpty ||
                animal_shogi::kGold < board_[action.to]);
    ok = ok && animal_shogi::CanMove(piece, action.from, action.to);
    if (!ok) {
      return false;
    }
    animal_shogi::Board next_board = board_;
    animal_shogi::Hand next_hand = hand_;
    ApplyMove(action, &next_board, &next_hand);
    return !IsChecked(next_board);
  }

  bool IsLegalDrop(const animal_shogi::DecodedAction& action) const {
    bool ok = board_[action.to] == animal_shogi::kEmpty;
    ok = ok && hand_[0][action.drop_piece] > 0;
    if (!ok) {
      return false;
    }
    animal_shogi::Board next_board = board_;
    animal_shogi::Hand next_hand = hand_;
    ApplyDrop(action, &next_board, &next_hand);
    return !IsChecked(next_board);
  }

  void UpdateLegalActionMask() {
    for (int label = 0; label < 132; ++label) {
      const animal_shogi::DecodedAction action = animal_shogi::Decode(label);
      legal_action_mask_[label] =
          action.is_drop ? IsLegalDrop(action) : IsLegalMove(action);
    }
  }

  void WriteState(const std::array<float, 2>& rewards) {
    State state = Allocate(2);
    state["info:current_player"_] = current_player_;
    state["info:turn"_] = turn_;
    for (int sq = 0; sq < 12; ++sq) {
      const auto [row, col] = RenderCoord(sq);
      state["info:board"_](row, col) = board_[sq];
      state["info:legal_action_mask"_][sq] = legal_action_mask_[sq];
    }
    for (int label = 12; label < 132; ++label) {
      state["info:legal_action_mask"_][label] = legal_action_mask_[label];
    }
    for (int row = 0; row < 2; ++row) {
      for (int col = 0; col < 3; ++col) {
        state["info:hand"_](row, col) = hand_[row][col];
      }
    }
    for (int player = 0; player < 2; ++player) {
      state["reward"_][player] = rewards[player];
      state["info:players.id"_][player] = player;
      WriteObservation(player, &state);
    }
  }

  void WriteObservation(int player, State* state) const {
    animal_shogi::BoardHistory board_history = board_history_;
    animal_shogi::HandHistory hand_history = hand_history_;
    const int color = player == current_player_ ? turn_ : 1 - turn_;
    if (player != current_player_) {
      for (int& piece : board_history) {
        if (piece != animal_shogi::kEmpty) {
          piece = (piece + 5) % 10;
        }
      }
      for (int row = 0; row < 8; ++row) {
        std::reverse(board_history.begin() + row * 12,
                     board_history.begin() + (row + 1) * 12);
        std::rotate(hand_history.begin() + row * 6,
                    hand_history.begin() + row * 6 + 3,
                    hand_history.begin() + (row + 1) * 6);
      }
    }
    for (int row = 0; row < 4; ++row) {
      for (int col = 0; col < 3; ++col) {
        for (int ch = 0; ch < 194; ++ch) {
          (*state)["obs"_](player, row, col, ch) = 0.0f;
        }
      }
    }
    int channel = 0;
    for (int hist = 0; hist < 8; ++hist) {
      for (int piece = 0; piece < 10; ++piece, ++channel) {
        for (int sq = 0; sq < 12; ++sq) {
          const auto [row, col] = RenderCoord(sq);
          (*state)["obs"_](player, row, col, channel) =
              board_history[hist * 12 + sq] == piece ? 1.0f : 0.0f;
        }
      }
      for (int p = 0; p < 6; ++p) {
        for (int n = 1; n <= 2; ++n, ++channel) {
          const float value = hand_history[hist * 6 + p] >= n ? 1.0f : 0.0f;
          for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 3; ++col) {
              (*state)["obs"_](player, row, col, channel) = value;
            }
          }
        }
      }
      for (int rep_value = 0; rep_value < 2; ++rep_value, ++channel) {
        const float value = rep_history_[hist] == rep_value ? 1.0f : 0.0f;
        for (int row = 0; row < 4; ++row) {
          for (int col = 0; col < 3; ++col) {
            (*state)["obs"_](player, row, col, channel) = value;
          }
        }
      }
    }
    for (int row = 0; row < 4; ++row) {
      for (int col = 0; col < 3; ++col) {
        (*state)["obs"_](player, row, col, channel) = static_cast<float>(color);
        (*state)["obs"_](player, row, col, channel + 1) =
            static_cast<float>(step_count_) /
            static_cast<float>(animal_shogi::kMaxSteps);
      }
    }
  }
};

using AnimalShogiEnvPool = AsyncEnvPool<AnimalShogiEnv>;

}  // namespace pgx

#endif  // ENVPOOL_PGX_ANIMAL_SHOGI_H_
