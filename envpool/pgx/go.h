// Copyright 2026 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ENVPOOL_PGX_GO_H_
#define ENVPOOL_PGX_GO_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace pgx {

class GoEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("board_size"_.Bind(19), "komi"_.Bind(7.5),
                    "history_length"_.Bind(8), "max_terminal_steps"_.Bind(0),
                    "rules"_.Bind(std::string("pgx")),
                    "task"_.Bind(std::string("go_19x19")));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const int size = conf["board_size"_];
    const int history_length = conf["history_length"_];
    const int num_actions = size * size + 1;
    return MakeDict(
        "obs"_.Bind(Spec<bool>({-1, size, size, history_length * 2 + 1})),
        "info:board"_.Bind(Spec<int>({size, size}, {-1, 1})),
        "info:current_player"_.Bind(Spec<int>({}, {0, 1})),
        "info:legal_action_mask"_.Bind(Spec<bool>({num_actions})),
        "info:ko"_.Bind(Spec<int>({}, {-1, size * size - 1})),
        "info:is_psk"_.Bind(Spec<bool>({})),
        "info:consecutive_pass_count"_.Bind(Spec<int>({})),
        "info:black_area"_.Bind(Spec<int>({})),
        "info:white_area"_.Bind(Spec<int>({})),
        "info:players.id"_.Bind(Spec<int>({-1}, {0, 1})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    const int size = conf["board_size"_];
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, size * size})));
  }
};

using GoEnvSpec = EnvSpec<GoEnvFns>;

class GoEnv : public Env<GoEnvSpec>, public RenderableEnv {
 public:
  GoEnv(const Spec& spec, int env_id)
      : Env<GoEnvSpec>(spec, env_id),
        size_(spec.config["board_size"_]),
        komi_(spec.config["komi"_]),
        history_length_(spec.config["history_length"_]),
        max_terminal_steps_(spec.config["max_terminal_steps"_]),
        rules_(ParseRules(std::string(spec.config["rules"_]))),
        board_area_(size_ * size_),
        board_(board_area_),
        board_history_(history_length_ * board_area_),
        hash_history_(MaxTerminalSteps()) {
    CHECK_GT(size_, 1);
    CHECK_GT(history_length_, 0);
    CHECK_GE(max_terminal_steps_, 0);
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    std::fill(board_.begin(), board_.end(), 0);
    std::fill(board_history_.begin(), board_history_.end(), 2);
    std::fill(num_captured_.begin(), num_captured_.end(), 0);
    std::fill(hash_history_.begin(), hash_history_.end(),
              std::pair<uint64_t, uint64_t>{0, 0});
    consecutive_pass_count_ = 0;
    ko_ = -1;
    is_psk_ = false;
    step_count_ = 0;
    done_ = false;
    legal_action_mask_.assign(board_area_ + 1, true);
    const bool swap_players = (gen_() & 2U) != 0;
    player_order_[0] = swap_players ? 1 : 0;
    player_order_[1] = swap_players ? 0 : 1;
    WriteState({0.0f, 0.0f});
  }

  void Step(const Action& action) override {
    CHECK_EQ(action["action"_].Shape(0), 1)
        << "PGX Go expects one action per environment for the current player.";
    const int act = action["action"_][0];
    const int loser = CurrentPlayer();
    const bool illegal = act < 0 || act > board_area_ ||
                         (act <= board_area_ && !legal_action_mask_[act]);
    if (act >= 0 && act <= board_area_) {
      StepGame(act);
    }
    std::array<float, 2> player_rewards;
    if (illegal) {
      done_ = true;
      std::fill(legal_action_mask_.begin(), legal_action_mask_.end(), true);
      player_rewards = {1.0f, 1.0f};
      player_rewards[loser] = -1.0f;
    } else {
      UpdateLegalActionMask();
      player_rewards = PlayerRewards(ColorRewards());
      done_ = IsTerminal();
      if (done_) {
        std::fill(legal_action_mask_.begin(), legal_action_mask_.end(), true);
      }
    }
    WriteState(player_rewards);
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    const int default_size = std::max(160, size_ * 32);
    return {width > 0 ? width : default_size,
            height > 0 ? height : default_size};
  }

  void Render(int width, int height, int camera_id,
              unsigned char* rgb) override {
    (void)camera_id;
    const int width_stride = width * 3;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        unsigned char* px = rgb + y * width_stride + x * 3;
        px[0] = 219;
        px[1] = 181;
        px[2] = 111;
      }
    }
    const int margin = std::max(12, std::min(width, height) / 14);
    const double x_step = size_ == 1 ? 0.0
                                     : static_cast<double>(width - 2 * margin) /
                                           static_cast<double>(size_ - 1);
    const double y_step = size_ == 1
                              ? 0.0
                              : static_cast<double>(height - 2 * margin) /
                                    static_cast<double>(size_ - 1);
    const auto xpos = [&](int col) {
      return static_cast<int>(std::lround(margin + col * x_step));
    };
    const auto ypos = [&](int row) {
      return static_cast<int>(std::lround(margin + row * y_step));
    };
    const int radius =
        std::max(3, static_cast<int>(std::min(x_step, y_step) * 0.42));

    for (int i = 0; i < size_; ++i) {
      DrawLine(rgb, width, height, xpos(0), ypos(i), xpos(size_ - 1), ypos(i),
               {70, 50, 30});
      DrawLine(rgb, width, height, xpos(i), ypos(0), xpos(i), ypos(size_ - 1),
               {70, 50, 30});
    }
    DrawStarPoints(rgb, width, height, radius / 4 + 1, xpos, ypos);
    for (int idx = 0; idx < board_area_; ++idx) {
      const int sign = Sign(board_[idx]);
      if (sign == 0) {
        continue;
      }
      const int row = idx / size_;
      const int col = idx % size_;
      if (sign > 0) {
        DrawCircle(rgb, width, height, xpos(col), ypos(row), radius,
                   {24, 24, 24}, {5, 5, 5});
      } else {
        DrawCircle(rgb, width, height, xpos(col), ypos(row), radius,
                   {236, 232, 220}, {150, 145, 135});
      }
    }
  }

 private:
  enum class Rules : uint8_t { kPgx, kChinese };

  using Hash = std::pair<uint64_t, uint64_t>;
  using Rgb = std::array<unsigned char, 3>;

  int size_;
  double komi_;
  int history_length_;
  int max_terminal_steps_;
  Rules rules_;
  int board_area_;
  std::vector<int> board_;
  std::vector<int> board_history_;
  std::array<int, 2> num_captured_{0, 0};
  int consecutive_pass_count_{0};
  int ko_{-1};
  bool is_psk_{false};
  int step_count_{0};
  bool done_{true};
  std::array<int, 2> player_order_{0, 1};
  std::vector<Hash> hash_history_;
  std::vector<bool> legal_action_mask_;

  static Rules ParseRules(const std::string& rules) {
    if (rules == "pgx" || rules == "tromp_taylor") {
      return Rules::kPgx;
    }
    if (rules == "chinese") {
      return Rules::kChinese;
    }
    CHECK(false) << "Unknown PGX Go rules: " << rules;
    return Rules::kPgx;
  }

  bool UsesChineseRules() const { return rules_ == Rules::kChinese; }

  int MaxTerminalSteps() const {
    if (max_terminal_steps_ > 0) {
      return max_terminal_steps_;
    }
    return board_area_ * 2;
  }

  int Color() const { return step_count_ % 2; }

  int CurrentPlayer() const { return player_order_[Color()]; }

  int ColorForPlayer(int player) const {
    return player_order_[0] == player ? 0 : 1;
  }

  static int Sign(int value) {
    if (value > 0) {
      return 1;
    }
    if (value < 0) {
      return -1;
    }
    return 0;
  }

  static std::pair<int, int> Signs(int color) {
    return color == 0 ? std::pair<int, int>{1, -1} : std::pair<int, int>{-1, 1};
  }

  std::array<int, 4> Adjacent(int xy) const {
    const int x = xy / size_;
    const int y = xy % size_;
    return {
        x > 0 ? (x - 1) * size_ + y : -1,
        x + 1 < size_ ? (x + 1) * size_ + y : -1,
        y > 0 ? x * size_ + y - 1 : -1,
        y + 1 < size_ ? x * size_ + y + 1 : -1,
    };
  }

  struct Counts {
    std::vector<int64_t> num_pseudo;
    std::vector<int64_t> idx_sum;
    std::vector<int64_t> idx_squared_sum;
  };

  Counts CountLiberties() const { return CountLiberties(board_); }

  Counts CountLiberties(const std::vector<int>& board) const {
    std::vector<int64_t> per_point_num(board_area_, 0);
    std::vector<int64_t> per_point_sum(board_area_, 0);
    std::vector<int64_t> per_point_squared_sum(board_area_, 0);
    for (int xy = 0; xy < board_area_; ++xy) {
      for (int adj : Adjacent(xy)) {
        if (adj == -1 || board[adj] != 0) {
          continue;
        }
        const int64_t idx = static_cast<int64_t>(adj) + 1;
        ++per_point_num[xy];
        per_point_sum[xy] += idx;
        per_point_squared_sum[xy] += idx * idx;
      }
    }

    Counts counts{
        std::vector<int64_t>(board_area_, 0),
        std::vector<int64_t>(board_area_, 0),
        std::vector<int64_t>(board_area_, 0),
    };
    for (int xy = 0; xy < board_area_; ++xy) {
      const int chain_ix = std::abs(board[xy]) - 1;
      if (chain_ix < 0) {
        continue;
      }
      counts.num_pseudo[chain_ix] += per_point_num[xy];
      counts.idx_sum[chain_ix] += per_point_sum[xy];
      counts.idx_squared_sum[chain_ix] += per_point_squared_sum[xy];
    }
    return counts;
  }

  static bool InAtari(const Counts& counts, int chain_ix) {
    if (chain_ix < 0) {
      chain_ix = static_cast<int>(counts.num_pseudo.size()) - 1;
    }
    const int64_t idx_sum = counts.idx_sum[chain_ix];
    return idx_sum * idx_sum ==
           counts.idx_squared_sum[chain_ix] * counts.num_pseudo[chain_ix];
  }

  void UpdateLegalActionMask() {
    legal_action_mask_.assign(board_area_ + 1, false);
    const auto [my_sign, opp_sign] = Signs(Color());
    Counts counts = CountLiberties();
    std::vector<bool> has_liberty(board_area_, false);
    std::vector<bool> can_kill(board_area_, false);
    for (int xy = 0; xy < board_area_; ++xy) {
      const int chain_ix = std::abs(board_[xy]) - 1;
      const bool in_atari = InAtari(counts, chain_ix);
      has_liberty[xy] = board_[xy] * my_sign > 0 && !in_atari;
      can_kill[xy] = board_[xy] * opp_sign > 0 && in_atari;
    }
    for (int xy = 0; xy < board_area_; ++xy) {
      if (board_[xy] != 0) {
        continue;
      }
      bool adj_ok = false;
      for (int adj : Adjacent(xy)) {
        if (adj == -1) {
          continue;
        }
        if (board_[adj] == 0 || can_kill[adj] || has_liberty[adj]) {
          adj_ok = true;
          break;
        }
      }
      legal_action_mask_[xy] = adj_ok;
    }
    if (ko_ >= 0) {
      legal_action_mask_[ko_] = false;
    }
    if (UsesChineseRules()) {
      MaskRepeatedPositions();
    }
    legal_action_mask_[board_area_] = true;
  }

  void MaskRepeatedPositions() {
    for (int xy = 0; xy < board_area_; ++xy) {
      if (legal_action_mask_[xy] && WouldRepeatPosition(xy)) {
        legal_action_mask_[xy] = false;
      }
    }
  }

  void StepGame(int action) {
    ko_ = -1;
    if (action < board_area_) {
      ApplyAction(action);
    } else {
      ++consecutive_pass_count_;
    }
    UpdateBoardHistory();
    const Hash hash = ComputeHash();
    if (step_count_ < static_cast<int>(hash_history_.size())) {
      hash_history_[step_count_] = hash;
    }
    is_psk_ = UsesChineseRules() ? false : IsPsk(hash);
    ++step_count_;
  }

  struct ApplyResult {
    int num_captured;
    int ko;
  };

  void ApplyAction(int action) {
    consecutive_pass_count_ = 0;
    const ApplyResult result = ApplyActionToBoard(action, &board_);
    num_captured_[Color()] += result.num_captured;
    ko_ = result.ko;
  }

  ApplyResult ApplyActionToBoard(int action, std::vector<int>* board) const {
    const int color = Color();
    const auto [my_sign, opp_sign] = Signs(color);
    const std::array<int, 4> adj_ixs = Adjacent(action);
    const Counts counts = CountLiberties(*board);

    std::array<int, 4> adj_ids{};
    std::array<bool, 4> is_killed{};
    for (int i = 0; i < 4; ++i) {
      const int adj = adj_ixs[i];
      adj_ids[i] = adj == -1 ? board->back() : (*board)[adj];
      if (adj == -1 || adj_ids[i] * opp_sign <= 0) {
        is_killed[i] = false;
        continue;
      }
      const int chain_ix = std::abs(adj_ids[i]) - 1;
      int single_liberty = -1;
      if (chain_ix >= 0 && counts.idx_sum[chain_ix] != 0) {
        single_liberty = static_cast<int>(counts.idx_squared_sum[chain_ix] /
                                          counts.idx_sum[chain_ix]) -
                         1;
      }
      is_killed[i] = InAtari(counts, chain_ix) && single_liberty == action;
    }

    std::vector<bool> surrounded(board_area_, false);
    int num_captured = 0;
    for (int xy = 0; xy < board_area_; ++xy) {
      for (int i = 0; i < 4; ++i) {
        if (is_killed[i] && (*board)[xy] == adj_ids[i]) {
          surrounded[xy] = true;
        }
      }
      if (surrounded[xy]) {
        ++num_captured;
      }
    }

    int ko_ix = 0;
    for (int i = 0; i < 4; ++i) {
      if (is_killed[i]) {
        ko_ix = i;
        break;
      }
    }
    bool ko_may_occur = true;
    for (int adj : adj_ixs) {
      if (adj != -1 && (*board)[adj] * opp_sign <= 0) {
        ko_may_occur = false;
        break;
      }
    }
    for (int xy = 0; xy < board_area_; ++xy) {
      if (surrounded[xy]) {
        (*board)[xy] = 0;
      }
    }

    (*board)[action] = (action + 1) * my_sign;
    MergeAdjacentChains(action, adj_ixs, my_sign, board);
    return {
        num_captured,
        ko_may_occur && num_captured == 1 ? adj_ixs[ko_ix] : -1,
    };
  }

  void MergeAdjacentChains(int action, const std::array<int, 4>& adj_ixs,
                           int my_sign, std::vector<int>* board) const {
    std::array<bool, 4> should_merge{};
    std::array<int, 4> target_ids{};
    int smallest_id = 9999;
    for (int i = 0; i < 4; ++i) {
      const int adj = adj_ixs[i];
      target_ids[i] = adj == -1 ? board->back() : (*board)[adj];
      should_merge[i] = adj != -1 && target_ids[i] * my_sign > 0;
      if (should_merge[i]) {
        smallest_id = std::min(smallest_id, std::abs(target_ids[i]));
      }
    }
    const int new_id = (*board)[action];
    smallest_id = std::min(std::abs(new_id), smallest_id) * my_sign;
    for (int xy = 0; xy < board_area_; ++xy) {
      bool merge = (*board)[xy] == new_id;
      for (int i = 0; i < 4; ++i) {
        merge = merge || (should_merge[i] && (*board)[xy] == target_ids[i]);
      }
      if (merge) {
        (*board)[xy] = smallest_id;
      }
    }
  }

  void UpdateBoardHistory() {
    for (int h = history_length_ - 1; h > 0; --h) {
      for (int xy = 0; xy < board_area_; ++xy) {
        board_history_[h * board_area_ + xy] =
            board_history_[(h - 1) * board_area_ + xy];
      }
    }
    for (int xy = 0; xy < board_area_; ++xy) {
      board_history_[xy] = Sign(board_[xy]);
    }
  }

  static uint64_t SplitMix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }

  Hash ComputeHash() const { return ComputeHash(board_); }

  Hash ComputeHash(const std::vector<int>& board) const {
    uint64_t h0 = 0x243f6a8885a308d3ULL;
    uint64_t h1 = 0x13198a2e03707344ULL;
    for (int xy = 0; xy < board_area_; ++xy) {
      const int stone_sign = Sign(board[xy]) + 1;
      const auto stone = static_cast<uint64_t>(stone_sign);
      h0 ^= SplitMix64(stone * 0x100000001b3ULL + xy);
      h1 ^= SplitMix64(stone * 0x9e3779b97f4a7c15ULL + xy * 17ULL);
    }
    return {h0, h1};
  }

  bool IsPsk(const Hash& hash) const {
    if (consecutive_pass_count_ != 0) {
      return false;
    }
    int same_hash_count = 0;
    for (const Hash& seen : hash_history_) {
      if (seen == hash) {
        ++same_hash_count;
      }
    }
    return same_hash_count > 1;
  }

  bool WouldRepeatPosition(int action) const {
    std::vector<int> next_board = board_;
    ApplyActionToBoard(action, &next_board);
    const Hash hash = ComputeHash(next_board);
    if (hash == EmptyBoardHash()) {
      return true;
    }
    for (int i = 0;
         i < step_count_ && i < static_cast<int>(hash_history_.size()); ++i) {
      if (hash_history_[i] == hash) {
        return true;
      }
    }
    return false;
  }

  Hash EmptyBoardHash() const {
    uint64_t h0 = 0x243f6a8885a308d3ULL;
    uint64_t h1 = 0x13198a2e03707344ULL;
    for (int xy = 0; xy < board_area_; ++xy) {
      h0 ^= SplitMix64(0x100000001b3ULL + xy);
      h1 ^= SplitMix64(0x9e3779b97f4a7c15ULL + xy * 17ULL);
    }
    return {h0, h1};
  }

  bool IsTerminal() const {
    return consecutive_pass_count_ >= 2 || is_psk_ ||
           MaxTerminalSteps() <= step_count_;
  }

  int CountTerritory(int color_sign) const {
    std::vector<int> territory(board_area_);
    for (int xy = 0; xy < board_area_; ++xy) {
      territory[xy] = Sign(board_[xy] * color_sign);
    }
    bool changed = true;
    while (changed) {
      changed = false;
      std::vector<int> next = territory;
      for (int xy = 0; xy < board_area_; ++xy) {
        if (territory[xy] != 0) {
          continue;
        }
        bool adjacent_opp = false;
        for (int adj : Adjacent(xy)) {
          if (adj != -1 && territory[adj] == -1) {
            adjacent_opp = true;
            break;
          }
        }
        if (adjacent_opp) {
          next[xy] = -1;
          changed = true;
        }
      }
      territory.swap(next);
    }
    return std::count(territory.begin(), territory.end(), 0);
  }

  std::array<float, 2> ColorRewards() const {
    const auto [black_score, white_score] = AreaScores();
    std::array<float, 2> rewards = black_score - komi_ > white_score
                                       ? std::array<float, 2>{1.0f, -1.0f}
                                       : std::array<float, 2>{-1.0f, 1.0f};
    if (is_psk_) {
      rewards = {-1.0f, -1.0f};
      rewards[Color()] = 1.0f;
    }
    if (!IsTerminal()) {
      rewards = {0.0f, 0.0f};
    }
    return rewards;
  }

  std::array<int, 2> AreaScores() const {
    if (UsesChineseRules()) {
      return ChineseAreaScores();
    }
    return {
        CountTerritory(1) + CountStones(1),
        CountTerritory(-1) + CountStones(-1),
    };
  }

  std::array<int, 2> ChineseAreaScores() const {
    std::array<int, 2> area{CountStones(1), CountStones(-1)};
    std::vector<bool> seen(board_area_, false);
    for (int start = 0; start < board_area_; ++start) {
      if (seen[start] || board_[start] != 0) {
        continue;
      }
      std::vector<int> stack{start};
      seen[start] = true;
      int empty_count = 0;
      bool touches_black = false;
      bool touches_white = false;
      while (!stack.empty()) {
        const int xy = stack.back();
        stack.pop_back();
        ++empty_count;
        for (int adj : Adjacent(xy)) {
          if (adj == -1) {
            continue;
          }
          const int sign = Sign(board_[adj]);
          if (sign > 0) {
            touches_black = true;
          } else if (sign < 0) {
            touches_white = true;
          } else if (!seen[adj]) {
            seen[adj] = true;
            stack.push_back(adj);
          }
        }
      }
      if (touches_black != touches_white) {
        area[touches_black ? 0 : 1] += empty_count;
      }
    }
    return area;
  }

  int CountStones(int color_sign) const {
    int count = 0;
    for (int value : board_) {
      if (value * color_sign > 0) {
        ++count;
      }
    }
    return count;
  }

  std::array<float, 2> PlayerRewards(
      const std::array<float, 2>& color_rewards) const {
    std::array<float, 2> player_rewards{};
    for (int color = 0; color < 2; ++color) {
      player_rewards[player_order_[color]] = color_rewards[color];
    }
    return player_rewards;
  }

  void WriteState(const std::array<float, 2>& player_rewards) {
    State state = Allocate(2);
    const auto [black_area, white_area] = AreaScores();
    state["info:current_player"_] = CurrentPlayer();
    state["info:ko"_] = ko_;
    state["info:is_psk"_] = is_psk_;
    state["info:consecutive_pass_count"_] = consecutive_pass_count_;
    state["info:black_area"_] = black_area;
    state["info:white_area"_] = white_area;
    for (int xy = 0; xy < board_area_; ++xy) {
      state["info:board"_](xy / size_, xy % size_) = Sign(board_[xy]);
      state["info:legal_action_mask"_][xy] =
          legal_action_mask_.empty() ? true : legal_action_mask_[xy];
    }
    state["info:legal_action_mask"_][board_area_] = true;
    for (int player = 0; player < 2; ++player) {
      state["info:players.id"_][player] = player;
      state["reward"_][player] = player_rewards[player];
      WriteObservation(player, &state);
    }
  }

  void WriteObservation(int player, State* state) const {
    auto obs = (*state)["obs"_];
    const int curr_color = Color();
    const int color = player == CurrentPlayer() ? curr_color : 1 - curr_color;
    const auto [my_sign, unused_opp_sign] = Signs(color);
    (void)unused_opp_sign;
    for (int xy = 0; xy < board_area_; ++xy) {
      for (int h = 0; h < history_length_; ++h) {
        const int history_value = board_history_[h * board_area_ + xy];
        obs(player, xy / size_, xy % size_, h * 2) = history_value == my_sign;
        obs(player, xy / size_, xy % size_, h * 2 + 1) =
            history_value == -my_sign;
      }
      obs(player, xy / size_, xy % size_, history_length_ * 2) = color == 1;
    }
  }

  void SetPixel(unsigned char* rgb, int width, int height, int x, int y,
                Rgb color) const {
    if (x < 0 || y < 0 || x >= width || y >= height) {
      return;
    }
    unsigned char* px = rgb + (y * width + x) * 3;
    px[0] = color[0];
    px[1] = color[1];
    px[2] = color[2];
  }

  void DrawLine(unsigned char* rgb, int width, int height, int x0, int y0,
                int x1, int y1, Rgb color) const {
    const int dx = std::abs(x1 - x0);
    const int sx = x0 < x1 ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    for (;;) {
      SetPixel(rgb, width, height, x0, y0, color);
      if (x0 == x1 && y0 == y1) {
        break;
      }
      const int e2 = 2 * err;
      if (e2 >= dy) {
        err += dy;
        x0 += sx;
      }
      if (e2 <= dx) {
        err += dx;
        y0 += sy;
      }
    }
  }

  void DrawCircle(unsigned char* rgb, int width, int height, int cx, int cy,
                  int radius, Rgb fill, Rgb edge) const {
    const int edge_band = std::max(1, radius / 5);
    const int r2 = radius * radius;
    const int inner2 = (radius - edge_band) * (radius - edge_band);
    for (int y = cy - radius; y <= cy + radius; ++y) {
      for (int x = cx - radius; x <= cx + radius; ++x) {
        const int d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        if (d2 > r2) {
          continue;
        }
        SetPixel(rgb, width, height, x, y, d2 >= inner2 ? edge : fill);
      }
    }
  }

  template <typename XPos, typename YPos>
  void DrawStarPoints(unsigned char* rgb, int width, int height, int radius,
                      XPos xpos, YPos ypos) const {
    std::vector<int> points;
    if (size_ == 19) {
      points = {3, 9, 15};
    } else if (size_ == 9) {
      points = {2, 4, 6};
    }
    for (int row : points) {
      for (int col : points) {
        DrawCircle(rgb, width, height, xpos(col), ypos(row), radius,
                   {70, 50, 30}, {70, 50, 30});
      }
    }
  }
};

using GoEnvPool = AsyncEnvPool<GoEnv>;

}  // namespace pgx

#endif  // ENVPOOL_PGX_GO_H_
