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

#ifndef ENVPOOL_PGX_SPARROW_MAHJONG_H_
#define ENVPOOL_PGX_SPARROW_MAHJONG_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <string>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/pgx/board_games.h"

namespace pgx {
namespace sparrow_mahjong {

constexpr int kNumTiles = 44;
constexpr int kNumTileTypes = 11;
constexpr int kPlayers = 3;
constexpr int kMaxRiverLength = 10;
constexpr int kNumCache = 160;
constexpr int kMaxScore = 26;

using Hands = std::array<std::array<int, kNumTileTypes>, kPlayers>;
using Rivers = std::array<std::array<int, kMaxRiverLength>, kPlayers>;
using RedRivers = std::array<std::array<bool, kMaxRiverLength>, kPlayers>;
using Wall = std::array<int, kNumTiles>;
using TileMask = std::array<bool, kNumTileTypes>;

constexpr std::array<int, kNumCache> kWinHands = {
    18,       78,       90,       378,      390,      450,      778,
    790,      850,      1150,     1550,     1878,     1890,     1950,
    2250,     2650,     3878,     3890,     3950,     4250,     4650,
    5750,     7750,     9378,     9390,     9450,     9750,     10150,
    11250,    13250,    19378,    19390,    19450,    19750,    20150,
    21250,    23250,    28750,    38750,    46878,    46890,    46950,
    47250,    47650,    48750,    50750,    56250,    66250,    96878,
    96890,    96950,    97250,    97650,    98750,    100750,   106250,
    116250,   143750,   193750,   234378,   234390,   234450,   234750,
    235150,   236250,   238250,   243750,   253750,   281250,   331250,
    484378,   484390,   484450,   484750,   485150,   486250,   488250,
    493750,   503750,   531250,   581250,   718750,   968750,   1171878,
    1171890,  1171950,  1172250,  1172650,  1173750,  1175750,  1181250,
    1191250,  1218750,  1268750,  1406250,  1656250,  2421878,  2421890,
    2421950,  2422250,  2422650,  2423750,  2425750,  2431250,  2441250,
    2468750,  2518750,  2656250,  2906250,  3593750,  4843750,  5859378,
    5859390,  5859450,  5859750,  5860150,  5861250,  5863250,  5868750,
    5878750,  5906250,  5956250,  6093750,  6343750,  7031250,  8281250,
    12109378, 12109390, 12109450, 12109750, 12110150, 12111250, 12113250,
    12118750, 12128750, 12156250, 12206250, 12343750, 12593750, 13281250,
    14531250, 17968750, 24218750, 29296878, 29296890, 29296950, 29297250,
    29297650, 29298750, 29300750, 29306250, 29316250, 29343750, 29393750,
    29531250, 29781250, 30468750, 31718750, 35156250, 41406250,
};

constexpr std::array<int, kNumCache> kBaseScores = {
    4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 3, 2,
    4, 4, 4, 4, 3, 4, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 4, 4, 4, 4, 3, 4, 3,
    4, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 4, 4, 4, 4, 3, 4, 3, 4, 3, 4,
    3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 4, 4, 4, 4, 3, 4, 3, 4, 3,
    4, 3, 4, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 4, 4, 4, 4,
    3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3,
    2, 3, 2, 3, 2, 4, 4, 4, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3,
};

constexpr std::array<int, kNumCache> kYakuScores = {
    15, 15, 15, 0,  10, 0,  0,  0,  0, 0,  0,  0, 0,  0, 1,  0, 0, 0,  0,  1,
    0,  1,  1,  0,  10, 0,  10, 0,  1, 1,  0,  0, 0,  1, 0,  1, 1, 1,  1,  0,
    0,  0,  1,  0,  1,  1,  1,  1,  0, 0,  0,  1, 0,  1, 1,  1, 1, 1,  1,  0,
    10, 0,  10, 0,  1,  1,  10, 1,  1, 1,  0,  0, 0,  1, 0,  1, 1, 1,  1,  1,
    1,  1,  1,  0,  10, 0,  10, 0,  1, 1,  10, 1, 1,  1, 10, 1, 0, 10, 0,  10,
    0,  1,  1,  10, 1,  1,  1,  10, 1, 10, 10, 0, 10, 0, 10, 0, 1, 1,  10, 1,
    1,  1,  10, 1,  10, 10, 0,  0,  0, 0,  0,  0, 0,  0, 0,  0, 0, 0,  0,  0,
    0,  0,  0,  15, 15, 15, 0,  0,  0, 0,  0,  0, 0,  0, 0,  0, 0, 0,  0,  0,
};

inline bool IsRed(int tile_id) {
  return ((tile_id % 4 == 0) && tile_id != 36) || tile_id >= 40;
}

inline int ToBase5(const std::array<int, kNumTileTypes>& hand) {
  static constexpr std::array<int, kNumTileTypes> kBase = {
      9765625, 1953125, 390625, 78125, 15625, 3125, 625, 125, 25, 5, 1,
  };
  int value = 0;
  for (int i = 0; i < kNumTileTypes; ++i) {
    value += hand[i] * kBase[i];
  }
  return value;
}

inline bool IsCompleted(const std::array<int, kNumTileTypes>& hand) {
  const int value = ToBase5(hand);
  return std::find(kWinHands.begin(), kWinHands.end(), value) !=
         kWinHands.end();
}

inline std::pair<int, int> HandToScore(
    const std::array<int, kNumTileTypes>& hand) {
  const int value = ToBase5(hand);
  int best_ix = 0;
  int best_distance = std::abs(kWinHands[0] - value);
  for (int i = 1; i < kNumCache; ++i) {
    const int distance = std::abs(kWinHands[i] - value);
    if (distance < best_distance) {
      best_distance = distance;
      best_ix = i;
    }
  }
  return {kBaseScores[best_ix], kYakuScores[best_ix]};
}

}  // namespace sparrow_mahjong

class SparrowMahjongEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("task"_.Bind(std::string("sparrow_mahjong")));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("obs"_.Bind(Spec<bool>({-1, 11, 15})),
                    "info:current_player"_.Bind(Spec<int>({}, {0, 2})),
                    "info:dora"_.Bind(Spec<int>({}, {0, 10})),
                    "info:draw_ix"_.Bind(Spec<int>({}, {0, 44})),
                    "info:hands"_.Bind(Spec<int>({3, 11}, {0, 6})),
                    "info:is_red_in_river"_.Bind(Spec<bool>({3, 10})),
                    "info:last_discard"_.Bind(Spec<int>({}, {-1, 10})),
                    "info:legal_action_mask"_.Bind(Spec<bool>({11})),
                    "info:n_red_in_hands"_.Bind(Spec<int>({3, 11}, {0, 4})),
                    "info:players.id"_.Bind(Spec<int>({-1}, {0, 2})),
                    "info:rivers"_.Bind(Spec<int>({3, 10}, {-1, 10})),
                    "info:scores"_.Bind(Spec<int>({3})),
                    "info:shuffled_players"_.Bind(Spec<int>({3}, {0, 2})),
                    "info:turn"_.Bind(Spec<int>({})),
                    "info:wall"_.Bind(Spec<int>({44}, {0, 43})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 10})));
  }
};

using SparrowMahjongEnvSpec = EnvSpec<SparrowMahjongEnvFns>;

class SparrowMahjongEnv : public Env<SparrowMahjongEnvSpec>,
                          public RenderableEnv {
 public:
  using Spec = SparrowMahjongEnvSpec;
  using Action = typename Env<SparrowMahjongEnvSpec>::Action;

  SparrowMahjongEnv(const Spec& spec, int env_id)
      : Env<SparrowMahjongEnvSpec>(spec, env_id) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    do {
      InitOnce();
    } while (done_);
    WriteState(rewards_);
  }

  void Step(const Action& action) override {
    const int act = action["action"_][0];
    const int loser = current_player_;
    const bool illegal = act < 0 || act >= sparrow_mahjong::kNumTileTypes ||
                         (act >= 0 && !legal_action_mask_[act]);
    if (act >= 0 && act < sparrow_mahjong::kNumTileTypes) {
      StepGame(act);
    }
    if (illegal) {
      done_ = true;
      rewards_.fill(1.0f);
      rewards_[loser] = -1.0f;
      legal_action_mask_.fill(true);
    }
    WriteState(rewards_);
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 220, height > 0 ? height : 160};
  }

  void Render(int width, int height, int camera_id,
              unsigned char* rgb) override {
    (void)camera_id;
    board_games::Fill(rgb, width, height, {36, 96, 76});
    for (int player = 0; player < sparrow_mahjong::kPlayers; ++player) {
      const int top = player * height / 3;
      const int bottom = (player + 1) * height / 3;
      const bool active = !done_ && turn_ % 3 == player;
      board_games::FillRect(rgb, width, height, 0, top, width, top + 2,
                            active ? board_games::Rgb{245, 220, 80}
                                   : board_games::Rgb{22, 72, 58});
      for (int tile = 0; tile < sparrow_mahjong::kNumTileTypes; ++tile) {
        const int count = hands_[player][tile];
        if (count == 0) {
          continue;
        }
        const int left = tile * width / sparrow_mahjong::kNumTileTypes;
        const int right = (tile + 1) * width / sparrow_mahjong::kNumTileTypes;
        const int bar_h = std::max(2, count * (bottom - top) / 8);
        board_games::FillRect(rgb, width, height, left + 1, bottom - bar_h - 2,
                              right - 1, bottom - 2, {230, 235, 218});
      }
    }
  }

 private:
  sparrow_mahjong::Hands hands_{};
  sparrow_mahjong::Hands n_red_in_hands_{};
  sparrow_mahjong::Rivers rivers_{};
  sparrow_mahjong::RedRivers is_red_in_river_{};
  sparrow_mahjong::Wall wall_{};
  sparrow_mahjong::TileMask legal_action_mask_{};
  std::array<int, sparrow_mahjong::kPlayers> shuffled_players_{};
  std::array<int, sparrow_mahjong::kPlayers> scores_{};
  std::array<float, sparrow_mahjong::kPlayers> rewards_{};
  int current_player_{0};
  int turn_{0};
  int last_discard_{-1};
  int draw_ix_{sparrow_mahjong::kPlayers * 5};
  int dora_{0};
  int step_count_{0};
  bool done_{true};

  template <typename Array>
  void Shuffle(Array* values) {
    for (int i = static_cast<int>(values->size()) - 1; i > 0; --i) {
      const int j = static_cast<int>(gen_() % static_cast<uint32_t>(i + 1));
      std::swap((*values)[i], (*values)[j]);
    }
  }

  void InitOnce() {
    std::iota(shuffled_players_.begin(), shuffled_players_.end(), 0);
    std::iota(wall_.begin(), wall_.end(), 0);
    Shuffle(&shuffled_players_);
    Shuffle(&wall_);
    hands_ = {};
    n_red_in_hands_ = {};
    for (auto& river : rivers_) {
      river.fill(-1);
    }
    for (auto& river : is_red_in_river_) {
      river.fill(false);
    }
    legal_action_mask_.fill(false);
    scores_.fill(0);
    rewards_.fill(0.0f);
    current_player_ = shuffled_players_[0];
    turn_ = 0;
    last_discard_ = -1;
    dora_ = wall_.back() / 4;
    draw_ix_ = sparrow_mahjong::kPlayers * 5;
    step_count_ = 0;
    done_ = false;

    for (int i = 0; i < sparrow_mahjong::kPlayers * 5; ++i) {
      const int player = i / 5;
      const int tile = wall_[i] / 4;
      ++hands_[player][tile];
      if (sparrow_mahjong::IsRed(wall_[i])) {
        ++n_red_in_hands_[player][tile];
      }
    }
    DrawTileForTurn(0);
    const auto scores = HandsToScore();
    if (CheckTsumo(scores)) {
      StepByTsumo(scores);
    }
  }

  void DrawTileForTurn(int turn) {
    const int player = turn % sparrow_mahjong::kPlayers;
    const int tile_id = wall_[draw_ix_];
    const int tile = tile_id / 4;
    ++hands_[player][tile];
    if (sparrow_mahjong::IsRed(tile_id)) {
      ++n_red_in_hands_[player][tile];
    }
    ++draw_ix_;
    legal_action_mask_.fill(false);
    for (int tile_type = 0; tile_type < sparrow_mahjong::kNumTileTypes;
         ++tile_type) {
      legal_action_mask_[tile_type] = hands_[player][tile_type] > 0;
    }
  }

  std::array<int, sparrow_mahjong::kPlayers> HandsToScore() const {
    std::array<int, sparrow_mahjong::kPlayers> out{};
    const int discard = last_discard_ >= 0 ? last_discard_ : 10;
    for (int player = 0; player < sparrow_mahjong::kPlayers; ++player) {
      std::array<int, sparrow_mahjong::kNumTileTypes> hand = hands_[player];
      const int hand_sum = std::accumulate(hand.begin(), hand.end(), 0);
      if (hand_sum == 5) {
        ++hand[discard];
      }
      auto [base_score, yaku_score] = sparrow_mahjong::HandToScore(hand);
      const int n_doras = hand[dora_];
      int n_red_doras = 0;
      for (int value : n_red_in_hands_[player]) {
        n_red_doras += value;
      }
      if (n_red_doras >= 6) {
        yaku_score = 20;
      }
      out[player] = yaku_score >= 10
                        ? base_score + yaku_score
                        : base_score + yaku_score + n_doras + n_red_doras;
    }
    return out;
  }

  std::array<bool, sparrow_mahjong::kPlayers> CheckRon(
      const std::array<int, sparrow_mahjong::kPlayers>& scores) const {
    std::array<bool, sparrow_mahjong::kPlayers> winners{};
    const int discard = last_discard_ >= 0 ? last_discard_ : 10;
    for (int player = 0; player < sparrow_mahjong::kPlayers; ++player) {
      std::array<int, sparrow_mahjong::kNumTileTypes> hand = hands_[player];
      ++hand[discard];
      bool furiten = false;
      for (int tile : rivers_[player]) {
        furiten = furiten || tile == last_discard_;
      }
      winners[player] = player != turn_ % sparrow_mahjong::kPlayers &&
                        !furiten && scores[player] >= 5 &&
                        sparrow_mahjong::IsCompleted(hand);
    }
    return winners;
  }

  bool CheckTsumo(
      const std::array<int, sparrow_mahjong::kPlayers>& scores) const {
    const int player = turn_ % sparrow_mahjong::kPlayers;
    return scores[player] >= 0 && sparrow_mahjong::IsCompleted(hands_[player]);
  }

  std::array<float, sparrow_mahjong::kPlayers> RewardsFromScores(
      const std::array<int, sparrow_mahjong::kPlayers>& scores) const {
    std::array<float, sparrow_mahjong::kPlayers> rewards{};
    for (int i = 0; i < sparrow_mahjong::kPlayers; ++i) {
      rewards[shuffled_players_[i]] =
          static_cast<float>(scores[i]) /
          static_cast<float>(sparrow_mahjong::kMaxScore);
    }
    return rewards;
  }

  void StepByRon(std::array<int, sparrow_mahjong::kPlayers> scores,
                 const std::array<bool, sparrow_mahjong::kPlayers>& winners) {
    scores[0] += 2;
    for (int i = 0; i < sparrow_mahjong::kPlayers; ++i) {
      scores[i] = winners[i] ? scores[i] : 0;
    }
    int sum = 0;
    for (int score : scores) {
      sum += score;
    }
    scores[turn_ % sparrow_mahjong::kPlayers] = -sum;
    scores_ = scores;
    rewards_ = RewardsFromScores(scores);
    legal_action_mask_.fill(true);
    done_ = true;
  }

  void StepByTsumo(std::array<int, sparrow_mahjong::kPlayers> scores) {
    scores[0] += 2;
    int winner_score = scores[turn_ % sparrow_mahjong::kPlayers];
    const int loser_score = (winner_score + sparrow_mahjong::kPlayers - 2) /
                            (sparrow_mahjong::kPlayers - 1);
    winner_score = loser_score * (sparrow_mahjong::kPlayers - 1);
    scores.fill(-loser_score);
    scores[turn_ % sparrow_mahjong::kPlayers] = winner_score;
    scores_ = scores;
    rewards_ = RewardsFromScores(scores);
    legal_action_mask_.fill(true);
    done_ = true;
  }

  void StepByTie() {
    legal_action_mask_.fill(true);
    rewards_.fill(0.0f);
    done_ = true;
  }

  void StepNonTied() {
    ++turn_;
    current_player_ = shuffled_players_[turn_ % sparrow_mahjong::kPlayers];
    DrawTileForTurn(turn_);
    const auto scores = HandsToScore();
    if (CheckTsumo(scores)) {
      StepByTsumo(scores);
    } else {
      rewards_.fill(0.0f);
      done_ = false;
    }
  }

  void StepGame(int action) {
    const int player = turn_ % sparrow_mahjong::kPlayers;
    --hands_[player][action];
    const bool is_red_discarded =
        hands_[player][action] < n_red_in_hands_[player][action];
    if (is_red_discarded) {
      --n_red_in_hands_[player][action];
    }
    rivers_[player][turn_ / sparrow_mahjong::kPlayers] = action;
    is_red_in_river_[player][turn_ / sparrow_mahjong::kPlayers] =
        is_red_discarded;
    last_discard_ = action;

    const auto scores = HandsToScore();
    const auto winners = CheckRon(scores);
    const bool has_winner =
        std::any_of(winners.begin(), winners.end(), [](bool v) { return v; });
    if (has_winner) {
      StepByRon(scores, winners);
    } else if (sparrow_mahjong::kNumTiles - 1 <= draw_ix_) {
      StepByTie();
    } else {
      StepNonTied();
    }
    ++step_count_;
  }

  int PlayerTurnIndex(int player_id) const {
    for (int i = 0; i < sparrow_mahjong::kPlayers; ++i) {
      if (shuffled_players_[i] == player_id) {
        return i;
      }
    }
    return 0;
  }

  int RiverCount(int player) const {
    int count = 0;
    for (int tile : rivers_[player]) {
      if (tile >= 0) {
        ++count;
      }
    }
    return count;
  }

  void SetObs(State* state, int player, int tile, int feat, bool value) const {
    (*state)["obs"_](player, tile, feat) = value;
  }

  void WriteObservationForPlayer(State* state, int player_id) const {
    const int player = PlayerTurnIndex(player_id);
    for (int tile = 0; tile < sparrow_mahjong::kNumTileTypes; ++tile) {
      for (int feat = 0; feat < 15; ++feat) {
        SetObs(state, player_id, tile, feat, false);
      }
    }
    for (int tile = 0; tile < sparrow_mahjong::kNumTileTypes; ++tile) {
      for (int n = 1; n <= 4; ++n) {
        SetObs(state, player_id, tile, n - 1, hands_[player][tile] >= n);
      }
      SetObs(state, player_id, tile, 4, n_red_in_hands_[player][tile] >= 1);
    }
    SetObs(state, player_id, dora_, 5, true);
    for (int offset = 0; offset < sparrow_mahjong::kPlayers; ++offset) {
      const int river_player = (player + offset) % sparrow_mahjong::kPlayers;
      for (int tile : rivers_[river_player]) {
        if (tile >= 0) {
          SetObs(state, player_id, tile, 6 + offset, true);
        }
      }
    }
    for (int offset = 1; offset <= 2; ++offset) {
      const int river_player = (player + offset) % sparrow_mahjong::kPlayers;
      const int count = RiverCount(river_player);
      for (int back = 1; back <= 3; ++back) {
        if (count - back >= 0) {
          const int tile = rivers_[river_player][count - back];
          SetObs(state, player_id, tile, 9 + (offset - 1) * 3 + (back - 1),
                 true);
        }
      }
    }
  }

  void WriteState(const std::array<float, 3>& rewards) {
    State state = Allocate(3);
    state["info:current_player"_] = current_player_;
    state["info:dora"_] = dora_;
    state["info:draw_ix"_] = draw_ix_;
    state["info:last_discard"_] = last_discard_;
    state["info:turn"_] = turn_;
    for (int tile = 0; tile < sparrow_mahjong::kNumTileTypes; ++tile) {
      state["info:legal_action_mask"_][tile] = legal_action_mask_[tile];
    }
    for (int i = 0; i < sparrow_mahjong::kPlayers; ++i) {
      state["info:players.id"_][i] = i;
      state["info:scores"_][i] = scores_[i];
      state["info:shuffled_players"_][i] = shuffled_players_[i];
      state["reward"_][i] = rewards[i];
      WriteObservationForPlayer(&state, i);
      for (int tile = 0; tile < sparrow_mahjong::kNumTileTypes; ++tile) {
        state["info:hands"_](i, tile) = hands_[i][tile];
        state["info:n_red_in_hands"_](i, tile) = n_red_in_hands_[i][tile];
      }
      for (int ix = 0; ix < sparrow_mahjong::kMaxRiverLength; ++ix) {
        state["info:rivers"_](i, ix) = rivers_[i][ix];
        state["info:is_red_in_river"_](i, ix) = is_red_in_river_[i][ix];
      }
    }
    for (int i = 0; i < sparrow_mahjong::kNumTiles; ++i) {
      state["info:wall"_][i] = wall_[i];
    }
  }
};

using SparrowMahjongEnvPool = AsyncEnvPool<SparrowMahjongEnv>;

}  // namespace pgx

#endif  // ENVPOOL_PGX_SPARROW_MAHJONG_H_
