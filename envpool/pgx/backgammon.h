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

#ifndef ENVPOOL_PGX_BACKGAMMON_H_
#define ENVPOOL_PGX_BACKGAMMON_H_

#include <algorithm>
#include <array>
#include <random>
#include <string>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/pgx/board_games.h"

namespace pgx {
namespace backgammon {

using Board = std::array<int, 28>;
using Dice = std::array<int, 2>;
using PlayableDice = std::array<int, 4>;

// clang-format off
constexpr Board kInitBoard = {
    2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0,
    0, 0, 3, 0, 5, 0, 0, 0, 0, -2, 0, 0, 0, 0,
};
// clang-format on

inline int BarIdx() { return 24; }
inline int OffIdx() { return 26; }
inline bool Exists(const Board& board, int point) { return board[point] >= 1; }
inline bool IsOpen(const Board& board, int point) { return board[point] >= -1; }
inline int CalcSrc(int src) { return src == 1 ? BarIdx() : src - 2; }
inline int FromBoard(int src, int die) {
  const int target = src + die;
  return 0 <= target && target <= 23 ? target : OffIdx();
}
inline int CalcTgt(int src, int die) {
  return src >= 24 ? die - 1 : FromBoard(src, die);
}
inline std::pair<int, int> Decompose(int action) {
  const int src = CalcSrc(action / 6);
  const int die = action % 6 + 1;
  return {src, CalcTgt(src, die)};
}

inline Board FlipBoard(const Board& input) {
  Board board = input;
  for (int i = 0; i < 24; ++i) {
    board[i] = -input[23 - i];
  }
  board[24] = -input[25];
  board[25] = -input[24];
  board[26] = -input[27];
  board[27] = -input[26];
  return board;
}

inline int RearDistance(const Board& board) {
  for (int i = 0; i < 24; ++i) {
    if (board[i] > 0) {
      return 24 - i;
    }
  }
  return -76;
}

inline bool IsAllOnHomeBoard(const Board& board) {
  int on_home = 0;
  for (int i = 18; i < 24; ++i) {
    on_home += std::min(std::max(board[i], 0), 15);
  }
  return 15 - board[OffIdx()] == on_home;
}

inline bool IsToOffLegal(const Board& board, int src, int die) {
  const int rear_distance = RearDistance(board);
  const int distance = 24 - src;
  return src >= 0 && Exists(board, src) && IsAllOnHomeBoard(board) &&
         (distance == die ||
          (rear_distance <= die && rear_distance == distance));
}

inline bool IsToPointLegal(const Board& board, int src, int tgt) {
  if (src < 0 || tgt < 0 || tgt > 23) {
    return false;
  }
  const bool exists = Exists(board, src);
  const bool open = IsOpen(board, tgt);
  return (src >= 24 && exists && open) ||
         (src < 24 && exists && open && board[BarIdx()] == 0);
}

inline bool IsActionLegal(const Board& board, int action) {
  const auto [src, tgt] = Decompose(action);
  const int die = action % 6 + 1;
  const bool to_point = 0 <= tgt && tgt <= 23 && src >= 0;
  return to_point ? IsToPointLegal(board, src, tgt)
                  : IsToOffLegal(board, src, die);
}

inline Board Move(Board board, int action) {
  const auto [src, tgt] = Decompose(action);
  if (0 <= tgt && tgt < static_cast<int>(board.size())) {
    board[BarIdx() + 1] += board[tgt] == -1 ? -1 : 0;
    board[src] -= 1;
    board[tgt] += 1 + (board[tgt] == -1 ? 1 : 0);
  }
  return board;
}

inline bool IsAllOff(const Board& board) { return board[OffIdx()] == 15; }
inline bool IsGammon(const Board& board) { return board[OffIdx() + 1] == 0; }
inline bool RemainsAtInner(const Board& board) {
  int sum = 0;
  for (int i = 18; i < 24; ++i) {
    sum += board[i];
  }
  return sum != 0;
}
inline int CalcWinScore(const Board& board) {
  const bool gammon = IsGammon(board);
  return 1 + static_cast<int>(gammon) +
         static_cast<int>(gammon && RemainsAtInner(board));
}

}  // namespace backgammon

class BackgammonEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("task"_.Bind(std::string("backgammon")));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("obs"_.Bind(Spec<int>({-1, 34})),
                    "info:board"_.Bind(Spec<int>({28})),
                    "info:current_player"_.Bind(Spec<int>({}, {0, 1})),
                    "info:dice"_.Bind(Spec<int>({2}, {0, 5})),
                    "info:legal_action_mask"_.Bind(Spec<bool>({156})),
                    "info:playable_dice"_.Bind(Spec<int>({4}, {-1, 5})),
                    "info:played_dice_num"_.Bind(Spec<int>({}, {0, 4})),
                    "info:players.id"_.Bind(Spec<int>({-1}, {0, 1})),
                    "info:turn"_.Bind(Spec<int>({}, {0, 1})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 155})));
  }
};

using BackgammonEnvSpec = EnvSpec<BackgammonEnvFns>;

class BackgammonEnv : public Env<BackgammonEnvSpec>, public RenderableEnv {
 public:
  using Spec = BackgammonEnvSpec;
  using Action = typename Env<BackgammonEnvSpec>::Action;

  BackgammonEnv(const Spec& spec, int env_id)
      : Env<BackgammonEnvSpec>(spec, env_id) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    current_player_ = static_cast<int>(gen_() & 1U);
    board_ = backgammon::kInitBoard;
    RollInitDice();
    playable_dice_ = SetPlayableDice(dice_);
    played_dice_num_ = 0;
    turn_ = dice_[1] - dice_[0] > 0 ? 1 : 0;
    done_ = false;
    UpdateLegalActionMask(playable_dice_);
    WriteState({0.0f, 0.0f});
  }

  void Step(const Action& action) override {
    const int act = action["action"_][0];
    const int loser = current_player_;
    const bool illegal =
        act < 0 || act >= 156 || (act >= 0 && !legal_action_mask_[act]);
    if (act >= 0 && act < 156) {
      StepGame(act);
    }
    std::array<float, 2> rewards = rewards_;
    if (illegal) {
      done_ = true;
      std::fill(legal_action_mask_.begin(), legal_action_mask_.end(), true);
      rewards = {3.0f, 3.0f};
      rewards[loser] = -3.0f;
    } else if (done_) {
      std::fill(legal_action_mask_.begin(), legal_action_mask_.end(), true);
    }
    WriteState(rewards);
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 360, height > 0 ? height : 180};
  }

  void Render(int width, int height, int camera_id,
              unsigned char* rgb) override {
    (void)camera_id;
    board_games::Fill(rgb, width, height, {210, 180, 130});
    for (int point = 0; point < 24; ++point) {
      const int count = std::abs(board_[point]);
      const int col = point % 12;
      const bool top = point < 12;
      const int x = col * width / 12 + width / 24;
      for (int i = 0; i < std::min(count, 5); ++i) {
        const int y = top ? i * height / 12 + height / 18
                          : height - (i * height / 12 + height / 18);
        board_games::DrawCircle(rgb, width, height, x, y, height / 30,
                                board_[point] > 0
                                    ? board_games::Rgb{35, 35, 35}
                                    : board_games::Rgb{235, 235, 225});
      }
    }
  }

 private:
  backgammon::Board board_{};
  backgammon::Dice dice_{};
  backgammon::PlayableDice playable_dice_{};
  std::array<bool, 156> legal_action_mask_{};
  std::array<float, 2> rewards_{0.0f, 0.0f};
  int current_player_{0};
  int played_dice_num_{0};
  int turn_{1};
  bool done_{true};

  void RollInitDice() {
    do {
      RollDice();
    } while (dice_[0] == dice_[1]);
  }

  void RollDice() {
    std::uniform_int_distribution<int> dist(0, 5);
    dice_[0] = dist(gen_);
    dice_[1] = dist(gen_);
  }

  static backgammon::PlayableDice SetPlayableDice(
      const backgammon::Dice& dice) {
    if (dice[0] == dice[1]) {
      return {dice[0], dice[0], dice[0], dice[0]};
    }
    return {dice[0], dice[1], -1, -1};
  }

  void StepGame(int action) {
    rewards_ = {0.0f, 0.0f};
    UpdateByAction(action);
    if (backgammon::IsAllOff(board_)) {
      const int score = backgammon::CalcWinScore(board_);
      rewards_ = {-static_cast<float>(score), -static_cast<float>(score)};
      rewards_[current_player_] = static_cast<float>(score);
      done_ = true;
      return;
    }
    if (IsTurnEnd() || action / 6 == 0) {
      ChangeTurn();
    }
  }

  bool IsTurnEnd() const {
    int sum = 0;
    for (int die : playable_dice_) {
      sum += die;
    }
    return sum == -4;
  }

  void UpdateByAction(int action) {
    if (action / 6 == 0) {
      return;
    }
    board_ = backgammon::Move(board_, action);
    UpdatePlayableDice(action, played_dice_num_);
    ++played_dice_num_;
    UpdateLegalActionMask(playable_dice_);
  }

  void UpdatePlayableDice(int action, int old_played_dice_num) {
    if (dice_[0] == dice_[1]) {
      if (3 - old_played_dice_num >= 0) {
        playable_dice_[3 - old_played_dice_num] = -1;
      }
      return;
    }
    const int die = action % 6;
    for (int& playable_die : playable_dice_) {
      if (playable_die == die) {
        playable_die = -1;
      }
    }
  }

  void ChangeTurn() {
    board_ = backgammon::FlipBoard(board_);
    turn_ = (turn_ + 1) % 2;
    current_player_ = 1 - current_player_;
    RollDice();
    playable_dice_ = SetPlayableDice(dice_);
    played_dice_num_ = 0;
    UpdateLegalActionMaskForDice(dice_);
  }

  void UpdateLegalActionMask(const backgammon::PlayableDice& dice) {
    std::fill(legal_action_mask_.begin(), legal_action_mask_.end(), false);
    bool any = false;
    for (int die : dice) {
      if (die < 0) {
        continue;
      }
      for (int src = 0; src < 26; ++src) {
        const int action = src * 6 + die;
        const bool legal = backgammon::IsActionLegal(board_, action);
        legal_action_mask_[action] = legal_action_mask_[action] || legal;
        any = any || legal;
      }
    }
    if (!any) {
      for (int action = 0; action < 6; ++action) {
        legal_action_mask_[action] = true;
      }
    }
  }

  void UpdateLegalActionMaskForDice(const backgammon::Dice& dice) {
    std::fill(legal_action_mask_.begin(), legal_action_mask_.end(), false);
    bool any = false;
    for (int die : dice) {
      for (int src = 0; src < 26; ++src) {
        const int action = src * 6 + die;
        const bool legal = backgammon::IsActionLegal(board_, action);
        legal_action_mask_[action] = legal_action_mask_[action] || legal;
        any = any || legal;
      }
    }
    if (!any) {
      for (int action = 0; action < 6; ++action) {
        legal_action_mask_[action] = true;
      }
    }
  }

  std::array<int, 6> PlayableDiceCount() const {
    std::array<int, 6> out{};
    for (int die : playable_dice_) {
      if (die >= 0) {
        ++out[die];
      }
    }
    return out;
  }

  void WriteState(const std::array<float, 2>& rewards) {
    State state = Allocate(2);
    state["info:current_player"_] = current_player_;
    state["info:played_dice_num"_] = played_dice_num_;
    state["info:turn"_] = turn_;
    for (int i = 0; i < 28; ++i) {
      state["info:board"_][i] = board_[i];
    }
    for (int i = 0; i < 2; ++i) {
      state["info:dice"_][i] = dice_[i];
    }
    for (int i = 0; i < 4; ++i) {
      state["info:playable_dice"_][i] = playable_dice_[i];
    }
    for (int i = 0; i < 156; ++i) {
      state["info:legal_action_mask"_][i] = legal_action_mask_[i];
    }
    for (int player = 0; player < 2; ++player) {
      state["reward"_][player] = rewards[player];
      state["info:players.id"_][player] = player;
      WriteObservation(player, &state);
    }
  }

  void WriteObservation(int player, State* state) const {
    for (int i = 0; i < 28; ++i) {
      (*state)["obs"_](player, i) = board_[i];
    }
    const std::array<int, 6> dice_count = PlayableDiceCount();
    for (int i = 0; i < 6; ++i) {
      (*state)["obs"_](player, 28 + i) =
          player == current_player_ ? dice_count[i] : 0;
    }
  }
};

using BackgammonEnvPool = AsyncEnvPool<BackgammonEnv>;

}  // namespace pgx

#endif  // ENVPOOL_PGX_BACKGAMMON_H_
