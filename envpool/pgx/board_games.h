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

#ifndef ENVPOOL_PGX_BOARD_GAMES_H_
#define ENVPOOL_PGX_BOARD_GAMES_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace pgx {
namespace board_games {

using Rgb = std::array<unsigned char, 3>;

constexpr std::array<int, 8> kOthelloShifts = {1, -1, 8, -8, 7, -7, 9, -9};

inline constexpr std::array<std::array<int, 3>, 8> TicTacToeLines() {
  return {{{0, 1, 2},
           {3, 4, 5},
           {6, 7, 8},
           {0, 3, 6},
           {1, 4, 7},
           {2, 5, 8},
           {0, 4, 8},
           {2, 4, 6}}};
}

inline int Sign(int value) {
  if (value > 0) {
    return 1;
  }
  if (value < 0) {
    return -1;
  }
  return 0;
}

inline void SetPixel(unsigned char* rgb, int width, int height, int x, int y,
                     Rgb color) {
  if (x < 0 || y < 0 || x >= width || y >= height) {
    return;
  }
  unsigned char* px = rgb + (y * width + x) * 3;
  px[0] = color[0];
  px[1] = color[1];
  px[2] = color[2];
}

inline void Fill(unsigned char* rgb, int width, int height, Rgb color) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      SetPixel(rgb, width, height, x, y, color);
    }
  }
}

inline void FillRect(unsigned char* rgb, int width, int height, int left,
                     int top, int right, int bottom, Rgb color) {
  for (int y = top; y < bottom; ++y) {
    for (int x = left; x < right; ++x) {
      SetPixel(rgb, width, height, x, y, color);
    }
  }
}

inline void DrawCircle(unsigned char* rgb, int width, int height, int cx,
                       int cy, int radius, Rgb color) {
  const int r2 = radius * radius;
  for (int y = cy - radius; y <= cy + radius; ++y) {
    for (int x = cx - radius; x <= cx + radius; ++x) {
      const int d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      if (d2 <= r2) {
        SetPixel(rgb, width, height, x, y, color);
      }
    }
  }
}

inline void DrawGrid(unsigned char* rgb, int width, int height, int rows,
                     int cols) {
  Fill(rgb, width, height, {236, 232, 220});
  for (int row = 0; row <= rows; ++row) {
    const int y = row * height / rows;
    FillRect(rgb, width, height, 0, std::max(0, y - 1), width,
             std::min(height, y + 1), {70, 70, 70});
  }
  for (int col = 0; col <= cols; ++col) {
    const int x = col * width / cols;
    FillRect(rgb, width, height, std::max(0, x - 1), 0, std::min(width, x + 1),
             height, {70, 70, 70});
  }
}

inline std::array<float, 2> IllegalRewards(int loser) {
  std::array<float, 2> rewards{1.0f, 1.0f};
  rewards[loser] = -1.0f;
  return rewards;
}

inline int OpponentBoardValue(bool my, bool opp) {
  if (opp) {
    return 1;
  }
  if (my) {
    return -1;
  }
  return 0;
}

template <typename Mask>
inline void MaybeSetTerminalMask(Mask* legal_action_mask, bool done) {
  if (done) {
    std::fill(legal_action_mask->begin(), legal_action_mask->end(), true);
  }
}

}  // namespace board_games

class TicTacToeEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("task"_.Bind(std::string("tic_tac_toe")));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("obs"_.Bind(Spec<bool>({-1, 3, 3, 2})),
                    "info:board"_.Bind(Spec<int>({3, 3})),
                    "info:current_player"_.Bind(Spec<int>({})),
                    "info:legal_action_mask"_.Bind(Spec<bool>({9})),
                    "info:players.id"_.Bind(Spec<int>({-1}, {0, 1})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 8})));
  }
};

using TicTacToeEnvSpec = EnvSpec<TicTacToeEnvFns>;

class TicTacToeEnv : public Env<TicTacToeEnvSpec>, public RenderableEnv {
 public:
  using Spec = TicTacToeEnvSpec;
  using Action = typename Env<TicTacToeEnvSpec>::Action;

  TicTacToeEnv(const Spec& spec, int env_id)
      : Env<TicTacToeEnvSpec>(spec, env_id) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    std::fill(board_.begin(), board_.end(), -1);
    color_ = 0;
    current_player_ = static_cast<int>(gen_() & 1U);
    winner_ = -1;
    done_ = false;
    std::fill(legal_action_mask_.begin(), legal_action_mask_.end(), true);
    WriteState({0.0f, 0.0f});
  }

  void Step(const Action& action) override {
    const int act = action["action"_][0];
    const int loser = current_player_;
    const bool illegal =
        act < 0 || act >= 9 || (act >= 0 && !legal_action_mask_[act]);
    if (act >= 0 && act < 9) {
      StepGame(act);
    }
    std::array<float, 2> rewards{};
    if (illegal) {
      done_ = true;
      std::fill(legal_action_mask_.begin(), legal_action_mask_.end(), true);
      rewards = board_games::IllegalRewards(loser);
    } else {
      UpdateLegalActionMask();
      done_ = winner_ >= 0 || std::all_of(board_.begin(), board_.end(),
                                          [](int v) { return v >= 0; });
      rewards = done_ ? PlayerRewards(ColorRewards())
                      : std::array<float, 2>{0.0f, 0.0f};
      board_games::MaybeSetTerminalMask(&legal_action_mask_, done_);
    }
    WriteState(rewards);
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 192, height > 0 ? height : 192};
  }

  void Render(int width, int height, int camera_id,
              unsigned char* rgb) override {
    (void)camera_id;
    board_games::DrawGrid(rgb, width, height, 3, 3);
    DrawPieces(rgb, width, height, 3, 3);
  }

 private:
  std::array<int, 9> board_{};
  std::array<bool, 9> legal_action_mask_{};
  int color_{0};
  int current_player_{0};
  int winner_{-1};
  bool done_{true};

  void StepGame(int action) {
    board_[action] = color_;
    bool won = false;
    for (const auto& line : board_games::TicTacToeLines()) {
      won = won || (board_[line[0]] == color_ && board_[line[1]] == color_ &&
                    board_[line[2]] == color_);
    }
    winner_ = won ? color_ : -1;
    color_ = 1 - color_;
    current_player_ = 1 - current_player_;
  }

  void UpdateLegalActionMask() {
    for (int i = 0; i < 9; ++i) {
      legal_action_mask_[i] = board_[i] < 0;
    }
  }

  std::array<float, 2> ColorRewards() const {
    if (winner_ < 0) {
      return {0.0f, 0.0f};
    }
    std::array<float, 2> rewards{-1.0f, -1.0f};
    rewards[winner_] = 1.0f;
    return rewards;
  }

  std::array<float, 2> PlayerRewards(
      const std::array<float, 2>& color_rewards) const {
    if (current_player_ == color_) {
      return color_rewards;
    }
    return {color_rewards[1], color_rewards[0]};
  }

  void WriteState(const std::array<float, 2>& rewards) {
    State state = Allocate(2);
    state["info:current_player"_] = current_player_;
    for (int i = 0; i < 9; ++i) {
      state["info:board"_](i / 3, i % 3) = board_[i];
      state["info:legal_action_mask"_][i] = legal_action_mask_[i];
    }
    for (int player = 0; player < 2; ++player) {
      state["reward"_][player] = rewards[player];
      state["info:players.id"_][player] = player;
      WriteObservation(player, &state);
    }
  }

  void WriteObservation(int player, State* state) const {
    const int my_color = player == current_player_ ? color_ : 1 - color_;
    for (int i = 0; i < 9; ++i) {
      (*state)["obs"_](player, i / 3, i % 3, 0) = board_[i] == my_color;
      (*state)["obs"_](player, i / 3, i % 3, 1) = board_[i] == 1 - my_color;
    }
  }

  void DrawPieces(unsigned char* rgb, int width, int height, int rows,
                  int cols) const {
    const int radius = std::max(4, std::min(width / cols, height / rows) / 4);
    for (int i = 0; i < 9; ++i) {
      if (board_[i] < 0) {
        continue;
      }
      const int cx = (i % cols) * width / cols + width / (cols * 2);
      const int cy = (i / cols) * height / rows + height / (rows * 2);
      board_games::DrawCircle(rgb, width, height, cx, cy, radius,
                              board_[i] == 0 ? board_games::Rgb{30, 30, 30}
                                             : board_games::Rgb{230, 70, 70});
    }
  }
};

class ConnectFourEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("task"_.Bind(std::string("connect_four")));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("obs"_.Bind(Spec<bool>({-1, 6, 7, 2})),
                    "info:board"_.Bind(Spec<int>({6, 7})),
                    "info:current_player"_.Bind(Spec<int>({})),
                    "info:legal_action_mask"_.Bind(Spec<bool>({7})),
                    "info:players.id"_.Bind(Spec<int>({-1}, {0, 1})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 6})));
  }
};

using ConnectFourEnvSpec = EnvSpec<ConnectFourEnvFns>;

class ConnectFourEnv : public Env<ConnectFourEnvSpec>, public RenderableEnv {
 public:
  using Spec = ConnectFourEnvSpec;
  using Action = typename Env<ConnectFourEnvSpec>::Action;

  ConnectFourEnv(const Spec& spec, int env_id)
      : Env<ConnectFourEnvSpec>(spec, env_id) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    std::fill(board_.begin(), board_.end(), -1);
    color_ = 0;
    current_player_ = static_cast<int>(gen_() & 1U);
    winner_ = -1;
    done_ = false;
    UpdateLegalActionMask();
    WriteState({0.0f, 0.0f});
  }

  void Step(const Action& action) override {
    const int act = action["action"_][0];
    const int loser = current_player_;
    const bool illegal =
        act < 0 || act >= 7 || (act >= 0 && !legal_action_mask_[act]);
    if (act >= 0 && act < 7) {
      StepGame(act);
    }
    std::array<float, 2> rewards{};
    if (illegal) {
      done_ = true;
      std::fill(legal_action_mask_.begin(), legal_action_mask_.end(), true);
      rewards = board_games::IllegalRewards(loser);
    } else {
      UpdateLegalActionMask();
      done_ = winner_ >= 0 ||
              std::all_of(legal_action_mask_.begin(), legal_action_mask_.end(),
                          [](bool v) { return !v; });
      rewards = done_ ? PlayerRewards(ColorRewards())
                      : std::array<float, 2>{0.0f, 0.0f};
      board_games::MaybeSetTerminalMask(&legal_action_mask_, done_);
    }
    WriteState(rewards);
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 280, height > 0 ? height : 240};
  }

  void Render(int width, int height, int camera_id,
              unsigned char* rgb) override {
    (void)camera_id;
    board_games::DrawGrid(rgb, width, height, 6, 7);
    const int radius = std::max(3, std::min(width / 7, height / 6) / 3);
    for (int i = 0; i < 42; ++i) {
      if (board_[i] < 0) {
        continue;
      }
      const int cx = (i % 7) * width / 7 + width / 14;
      const int cy = (i / 7) * height / 6 + height / 12;
      board_games::DrawCircle(rgb, width, height, cx, cy, radius,
                              board_[i] == 0 ? board_games::Rgb{30, 30, 30}
                                             : board_games::Rgb{220, 60, 60});
    }
  }

 private:
  std::array<int, 42> board_{};
  std::vector<bool> legal_action_mask_ = std::vector<bool>(7, true);
  int color_{0};
  int current_player_{0};
  int winner_{-1};
  bool done_{true};

  void StepGame(int action) {
    int num_filled = 0;
    for (int row = 0; row < 6; ++row) {
      if (board_[row * 7 + action] >= 0) {
        ++num_filled;
      }
    }
    const int row = 5 - num_filled;
    if (row >= 0) {
      board_[row * 7 + action] = color_;
    }
    winner_ = HasWon(color_) ? color_ : -1;
    color_ = 1 - color_;
    current_player_ = 1 - current_player_;
  }

  bool HasWon(int color) const {
    for (int row = 0; row < 6; ++row) {
      for (int col = 0; col < 7; ++col) {
        if (board_[row * 7 + col] != color) {
          continue;
        }
        const std::array<std::pair<int, int>, 4> dirs{
            std::pair<int, int>{1, 0}, {0, 1}, {1, 1}, {1, -1}};
        for (auto [dr, dc] : dirs) {
          bool ok = true;
          for (int k = 1; k < 4; ++k) {
            const int rr = row + dr * k;
            const int cc = col + dc * k;
            ok = ok && rr >= 0 && rr < 6 && cc >= 0 && cc < 7 &&
                 board_[rr * 7 + cc] == color;
          }
          if (ok) {
            return true;
          }
        }
      }
    }
    return false;
  }

  void UpdateLegalActionMask() {
    for (int col = 0; col < 7; ++col) {
      int filled = 0;
      for (int row = 0; row < 6; ++row) {
        if (board_[row * 7 + col] >= 0) {
          ++filled;
        }
      }
      legal_action_mask_[col] = filled < 6;
    }
  }

  std::array<float, 2> ColorRewards() const {
    if (winner_ < 0) {
      return {0.0f, 0.0f};
    }
    std::array<float, 2> rewards{-1.0f, -1.0f};
    rewards[winner_] = 1.0f;
    return rewards;
  }

  std::array<float, 2> PlayerRewards(
      const std::array<float, 2>& color_rewards) const {
    if (current_player_ == color_) {
      return color_rewards;
    }
    return {color_rewards[1], color_rewards[0]};
  }

  void WriteState(const std::array<float, 2>& rewards) {
    State state = Allocate(2);
    state["info:current_player"_] = current_player_;
    for (int i = 0; i < 42; ++i) {
      state["info:board"_](i / 7, i % 7) = board_[i];
    }
    for (int action = 0; action < 7; ++action) {
      state["info:legal_action_mask"_][action] = legal_action_mask_[action];
    }
    for (int player = 0; player < 2; ++player) {
      state["reward"_][player] = rewards[player];
      state["info:players.id"_][player] = player;
      WriteObservation(player, &state);
    }
  }

  void WriteObservation(int player, State* state) const {
    const int my_color = player == current_player_ ? color_ : 1 - color_;
    for (int i = 0; i < 42; ++i) {
      (*state)["obs"_](player, i / 7, i % 7, 0) = board_[i] == my_color;
      (*state)["obs"_](player, i / 7, i % 7, 1) = board_[i] == 1 - my_color;
    }
  }
};

class HexEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("task"_.Bind(std::string("hex")));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("obs"_.Bind(Spec<bool>({-1, 11, 11, 4})),
                    "info:board"_.Bind(Spec<int>({11, 11})),
                    "info:current_player"_.Bind(Spec<int>({})),
                    "info:legal_action_mask"_.Bind(Spec<bool>({122})),
                    "info:players.id"_.Bind(Spec<int>({-1}, {0, 1})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 121})));
  }
};

using HexEnvSpec = EnvSpec<HexEnvFns>;

class HexEnv : public Env<HexEnvSpec>, public RenderableEnv {
 public:
  using Spec = HexEnvSpec;
  using Action = typename Env<HexEnvSpec>::Action;

  HexEnv(const Spec& spec, int env_id) : Env<HexEnvSpec>(spec, env_id) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    std::fill(board_.begin(), board_.end(), 0);
    step_count_ = 0;
    done_ = false;
    const bool swap_players = (gen_() & 1U) != 0;
    player_order_[0] = swap_players ? 1 : 0;
    player_order_[1] = swap_players ? 0 : 1;
    UpdateLegalActionMask();
    WriteState({0.0f, 0.0f});
  }

  void Step(const Action& action) override {
    const int act = action["action"_][0];
    const int loser = CurrentPlayer();
    const bool illegal =
        act < 0 || act > 121 || (act >= 0 && !legal_action_mask_[act]);
    if (act >= 0 && act <= 121) {
      StepGame(act);
    }
    std::array<float, 2> rewards{};
    if (illegal) {
      done_ = true;
      std::fill(legal_action_mask_.begin(), legal_action_mask_.end(), true);
      rewards = board_games::IllegalRewards(loser);
    } else {
      UpdateLegalActionMask();
      rewards = PlayerRewards(ColorRewards());
      board_games::MaybeSetTerminalMask(&legal_action_mask_, done_);
    }
    WriteState(rewards);
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 352, height > 0 ? height : 352};
  }

  void Render(int width, int height, int camera_id,
              unsigned char* rgb) override {
    (void)camera_id;
    board_games::DrawGrid(rgb, width, height, 11, 11);
    const int radius = std::max(3, std::min(width, height) / 36);
    for (int xy = 0; xy < 121; ++xy) {
      const int sign = board_games::Sign(board_[xy]);
      if (sign == 0) {
        continue;
      }
      const int row = xy / 11;
      const int col = xy % 11;
      const int cx = col * width / 11 + width / 22 + row * width / 180;
      const int cy = row * height / 11 + height / 22;
      board_games::DrawCircle(rgb, width, height, cx, cy, radius,
                              sign > 0 ? board_games::Rgb{35, 35, 35}
                                       : board_games::Rgb{220, 60, 60});
    }
  }

 private:
  std::array<int, 121> board_{};
  std::vector<bool> legal_action_mask_ = std::vector<bool>(122, true);
  std::array<int, 2> player_order_{0, 1};
  int step_count_{0};
  bool done_{true};

  int Color() const { return step_count_ % 2; }
  int CurrentPlayer() const { return player_order_[Color()]; }

  std::array<int, 6> Neighbour(int xy) const {
    const int x = xy / 11;
    const int y = xy % 11;
    std::array<int, 6> out{};
    const std::array<int, 6> xs = {x, x + 1, x - 1, x + 1, x - 1, x};
    const std::array<int, 6> ys = {y - 1, y - 1, y, y, y + 1, y + 1};
    for (int i = 0; i < 6; ++i) {
      out[i] = (0 <= xs[i] && xs[i] < 11 && 0 <= ys[i] && ys[i] < 11)
                   ? xs[i] * 11 + ys[i]
                   : -1;
    }
    return out;
  }

  void StepGame(int action) {
    if (action != 121) {
      Place(action);
    } else {
      Swap();
    }
  }

  void Place(int action) {
    const int set_place_id = action + 1;
    board_[action] = set_place_id;
    const auto neighbour = Neighbour(action);
    for (int adj : neighbour) {
      if (adj < 0 || board_[adj] <= 0) {
        continue;
      }
      const int adj_id = board_[adj];
      for (int& value : board_) {
        if (value == adj_id) {
          value = set_place_id;
        }
      }
    }
    ++step_count_;
    for (int& value : board_) {
      value = -value;
    }
    done_ = IsTerminal(action);
  }

  void Swap() {
    int ix = -1;
    for (int i = 0; i < 121; ++i) {
      if (board_[i] != 0) {
        ix = i;
        break;
      }
    }
    if (ix >= 0) {
      const int row = ix / 11;
      const int col = ix % 11;
      const int swapped_ix = col * 11 + row;
      board_[ix] = 0;
      board_[swapped_ix] = swapped_ix + 1;
    }
    ++step_count_;
    for (int& value : board_) {
      value = -value;
    }
  }

  bool IsTerminal(int action) const {
    const int target_id = board_[action];
    bool top = false;
    bool bottom = false;
    if (Color() == 0) {
      for (int row = 0; row < 11; ++row) {
        top = top || board_[row * 11] == target_id;
        bottom = bottom || board_[row * 11 + 10] == target_id;
      }
    } else {
      for (int col = 0; col < 11; ++col) {
        top = top || board_[col] == target_id;
        bottom = bottom || board_[110 + col] == target_id;
      }
    }
    return top && bottom;
  }

  void UpdateLegalActionMask() {
    for (int xy = 0; xy < 121; ++xy) {
      legal_action_mask_[xy] = board_[xy] == 0;
    }
    legal_action_mask_[121] = step_count_ == 1;
  }

  std::array<float, 2> ColorRewards() const {
    if (!done_) {
      return {0.0f, 0.0f};
    }
    std::array<float, 2> rewards{1.0f, 1.0f};
    rewards[Color()] = -1.0f;
    return rewards;
  }

  std::array<float, 2> PlayerRewards(
      const std::array<float, 2>& color_rewards) const {
    std::array<float, 2> rewards{};
    for (int color = 0; color < 2; ++color) {
      rewards[player_order_[color]] = color_rewards[color];
    }
    return rewards;
  }

  void WriteState(const std::array<float, 2>& rewards) {
    State state = Allocate(2);
    state["info:current_player"_] = CurrentPlayer();
    for (int xy = 0; xy < 121; ++xy) {
      state["info:board"_](xy / 11, xy % 11) = board_games::Sign(board_[xy]);
      state["info:legal_action_mask"_][xy] = legal_action_mask_[xy];
    }
    state["info:legal_action_mask"_][121] = legal_action_mask_[121];
    for (int player = 0; player < 2; ++player) {
      state["reward"_][player] = rewards[player];
      state["info:players.id"_][player] = player;
      WriteObservation(player, &state);
    }
  }

  void WriteObservation(int player, State* state) const {
    const int color = player == CurrentPlayer() ? Color() : 1 - Color();
    const int mul = color == Color() ? 1 : -1;
    const bool can_swap = step_count_ == 1;
    for (int xy = 0; xy < 121; ++xy) {
      const int value = board_[xy] * mul;
      (*state)["obs"_](player, xy / 11, xy % 11, 0) = value > 0;
      (*state)["obs"_](player, xy / 11, xy % 11, 1) = value < 0;
      (*state)["obs"_](player, xy / 11, xy % 11, 2) = color == 1;
      (*state)["obs"_](player, xy / 11, xy % 11, 3) = can_swap;
    }
  }
};

class OthelloEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("task"_.Bind(std::string("othello")));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("obs"_.Bind(Spec<bool>({-1, 8, 8, 2})),
                    "info:board"_.Bind(Spec<int>({8, 8})),
                    "info:current_player"_.Bind(Spec<int>({})),
                    "info:legal_action_mask"_.Bind(Spec<bool>({65})),
                    "info:players.id"_.Bind(Spec<int>({-1}, {0, 1})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 64})));
  }
};

using OthelloEnvSpec = EnvSpec<OthelloEnvFns>;

class OthelloEnv : public Env<OthelloEnvSpec>, public RenderableEnv {
 public:
  using Spec = OthelloEnvSpec;
  using Action = typename Env<OthelloEnvSpec>::Action;

  OthelloEnv(const Spec& spec, int env_id)
      : Env<OthelloEnvSpec>(spec, env_id) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    std::fill(board_.begin(), board_.end(), 0);
    board_[28] = 1;
    board_[35] = 1;
    board_[27] = -1;
    board_[36] = -1;
    turn_ = 0;
    passed_ = false;
    current_player_ = static_cast<int>(gen_() & 1U);
    done_ = false;
    std::fill(legal_action_mask_.begin(), legal_action_mask_.end(), false);
    legal_action_mask_[19] = true;
    legal_action_mask_[26] = true;
    legal_action_mask_[37] = true;
    legal_action_mask_[44] = true;
    WriteState({0.0f, 0.0f});
  }

  void Step(const Action& action) override {
    const int act = action["action"_][0];
    const int loser = current_player_;
    const bool illegal =
        act < 0 || act > 64 || (act >= 0 && !legal_action_mask_[act]);
    if (act >= 0 && act <= 64) {
      StepGame(act);
    }
    std::array<float, 2> rewards{};
    if (illegal) {
      done_ = true;
      std::fill(legal_action_mask_.begin(), legal_action_mask_.end(), true);
      rewards = board_games::IllegalRewards(loser);
    } else {
      rewards = rewards_;
      board_games::MaybeSetTerminalMask(&legal_action_mask_, done_);
    }
    WriteState(rewards);
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int camera_id,
              unsigned char* rgb) override {
    (void)camera_id;
    board_games::DrawGrid(rgb, width, height, 8, 8);
    const int radius = std::max(3, std::min(width, height) / 28);
    for (int xy = 0; xy < 64; ++xy) {
      const int sign = board_games::Sign(board_[xy]);
      if (sign == 0) {
        continue;
      }
      const int cx = (xy % 8) * width / 8 + width / 16;
      const int cy = (xy / 8) * height / 8 + height / 16;
      board_games::DrawCircle(rgb, width, height, cx, cy, radius,
                              sign > 0 ? board_games::Rgb{35, 35, 35}
                                       : board_games::Rgb{240, 240, 230});
    }
  }

 private:
  std::array<int, 64> board_{};
  std::vector<bool> legal_action_mask_ = std::vector<bool>(65, false);
  std::array<float, 2> rewards_{0.0f, 0.0f};
  int turn_{0};
  int current_player_{0};
  bool passed_{false};
  bool done_{true};

  static bool EdgeOk(int from, int to, int shift) {
    if (to < 0 || to >= 64) {
      return false;
    }
    const int fc = from % 8;
    const int tc = to % 8;
    if ((shift == 1 || shift == -7 || shift == 9) && tc != fc + 1) {
      return false;
    }
    if ((shift == -1 || shift == 7 || shift == -9) && tc != fc - 1) {
      return false;
    }
    return true;
  }

  static std::vector<int> Captures(const std::array<int, 64>& board, int pos,
                                   int shift) {
    std::vector<int> out;
    int cur = pos + shift;
    while (EdgeOk(cur - shift, cur, shift) && board[cur] < 0) {
      out.push_back(cur);
      cur += shift;
    }
    if (!EdgeOk(cur - shift, cur, shift) || board[cur] <= 0) {
      out.clear();
    }
    return out;
  }

  void StepGame(int action) {
    std::array<int, 64> next_board = board_;
    std::array<bool, 64> my{};
    std::array<bool, 64> opp{};
    for (int xy = 0; xy < 64; ++xy) {
      my[xy] = board_[xy] > 0;
      opp[xy] = board_[xy] < 0;
    }

    std::vector<int> reverse;
    if (action < 64) {
      for (int shift : board_games::kOthelloShifts) {
        std::vector<int> line = Captures(board_, action, shift);
        reverse.insert(reverse.end(), line.begin(), line.end());
      }
      for (int xy : reverse) {
        my[xy] = true;
        opp[xy] = false;
      }
      my[action] = true;
    }

    std::array<bool, 64> emp{};
    for (int xy = 0; xy < 64; ++xy) {
      emp[xy] = !(my[xy] || opp[xy]);
    }
    std::array<bool, 64> legal{};
    std::array<int, 64> opponent_board{};
    for (int xy = 0; xy < 64; ++xy) {
      opponent_board[xy] = board_games::OpponentBoardValue(my[xy], opp[xy]);
    }
    for (int xy = 0; xy < 64; ++xy) {
      if (!emp[xy]) {
        continue;
      }
      for (int shift : board_games::kOthelloShifts) {
        if (!Captures(opponent_board, xy, shift).empty()) {
          legal[xy] = true;
        }
      }
    }

    const bool board_full =
        std::none_of(emp.begin(), emp.end(), [](bool value) { return value; });
    const bool opp_empty =
        std::none_of(opp.begin(), opp.end(), [](bool value) { return value; });
    done_ = board_full || opp_empty || (passed_ && action == 64);
    rewards_ = done_ ? GetReward(my, opp) : std::array<float, 2>{0.0f, 0.0f};

    for (int xy = 0; xy < 64; ++xy) {
      next_board[xy] = board_games::OpponentBoardValue(my[xy], opp[xy]);
    }
    board_ = next_board;
    turn_ = 1 - turn_;
    current_player_ = 1 - current_player_;
    passed_ = action == 64;
    bool any_legal = false;
    for (int xy = 0; xy < 64; ++xy) {
      legal_action_mask_[xy] = legal[xy];
      any_legal = any_legal || legal[xy];
    }
    legal_action_mask_[64] = !any_legal;
  }

  std::array<float, 2> GetReward(const std::array<bool, 64>& my,
                                 const std::array<bool, 64>& opp) const {
    const int my_count =
        static_cast<int>(std::count(my.begin(), my.end(), true));
    const int opp_count =
        static_cast<int>(std::count(opp.begin(), opp.end(), true));
    if (my_count == opp_count) {
      return {0.0f, 0.0f};
    }
    std::array<float, 2> reward{-1.0f, -1.0f};
    reward[my_count > opp_count ? current_player_ : 1 - current_player_] = 1.0f;
    return reward;
  }

  void WriteState(const std::array<float, 2>& rewards) {
    State state = Allocate(2);
    state["info:current_player"_] = current_player_;
    for (int xy = 0; xy < 64; ++xy) {
      state["info:board"_](xy / 8, xy % 8) = board_[xy];
      state["info:legal_action_mask"_][xy] = legal_action_mask_[xy];
    }
    state["info:legal_action_mask"_][64] = legal_action_mask_[64];
    for (int player = 0; player < 2; ++player) {
      state["reward"_][player] = rewards[player];
      state["info:players.id"_][player] = player;
      WriteObservation(player, &state);
    }
  }

  void WriteObservation(int player, State* state) const {
    const int mul = player == current_player_ ? 1 : -1;
    for (int xy = 0; xy < 64; ++xy) {
      const int value = board_[xy] * mul;
      (*state)["obs"_](player, xy / 8, xy % 8, 0) = value > 0;
      (*state)["obs"_](player, xy / 8, xy % 8, 1) = value < 0;
    }
  }
};

using TicTacToeEnvPool = AsyncEnvPool<TicTacToeEnv>;
using ConnectFourEnvPool = AsyncEnvPool<ConnectFourEnv>;
using HexEnvPool = AsyncEnvPool<HexEnv>;
using OthelloEnvPool = AsyncEnvPool<OthelloEnv>;

}  // namespace pgx

#endif  // ENVPOOL_PGX_BOARD_GAMES_H_
