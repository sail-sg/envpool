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

#ifndef ENVPOOL_PGX_CARD_GAMES_H_
#define ENVPOOL_PGX_CARD_GAMES_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <string>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/pgx/board_games.h"

namespace pgx {

class KuhnPokerEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("task"_.Bind(std::string("kuhn_poker")));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("obs"_.Bind(Spec<bool>({-1, 7})),
                    "info:cards"_.Bind(Spec<int>({2}, {-1, 2})),
                    "info:current_player"_.Bind(Spec<int>({}, {0, 1})),
                    "info:last_action"_.Bind(Spec<int>({}, {-1, 1})),
                    "info:legal_action_mask"_.Bind(Spec<bool>({2})),
                    "info:players.id"_.Bind(Spec<int>({-1}, {0, 1})),
                    "info:pot"_.Bind(Spec<int>({2})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 1})));
  }
};

using KuhnPokerEnvSpec = EnvSpec<KuhnPokerEnvFns>;

class KuhnPokerEnv : public Env<KuhnPokerEnvSpec>, public RenderableEnv {
 public:
  using Spec = KuhnPokerEnvSpec;
  using Action = typename Env<KuhnPokerEnvSpec>::Action;

  KuhnPokerEnv(const Spec& spec, int env_id)
      : Env<KuhnPokerEnvSpec>(spec, env_id) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    static constexpr int kDeals[6][2] = {
        {0, 1}, {0, 2}, {1, 0}, {1, 2}, {2, 0}, {2, 1},
    };
    current_player_ = static_cast<int>(gen_() & 1U);
    const int deal = static_cast<int>(gen_() % 6);
    cards_[0] = kDeals[deal][0];
    cards_[1] = kDeals[deal][1];
    last_action_ = -1;
    pot_ = {0, 0};
    legal_action_mask_ = {true, true};
    done_ = false;
    WriteState({0.0f, 0.0f});
  }

  void Step(const Action& action) override {
    const int act = action["action"_][0];
    const int loser = current_player_;
    const bool illegal =
        act < 0 || act > 1 || (act >= 0 && !legal_action_mask_[act]);
    if (act >= 0 && act <= 1) {
      StepGame(act);
    }

    std::array<float, 2> rewards = rewards_;
    if (illegal) {
      done_ = true;
      legal_action_mask_ = {true, true};
      rewards = board_games::IllegalRewards(loser);
    } else if (done_) {
      legal_action_mask_ = {true, true};
    }
    WriteState(rewards);
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 192, height > 0 ? height : 128};
  }

  void Render(int width, int height, int camera_id,
              unsigned char* rgb) override {
    (void)camera_id;
    board_games::Fill(rgb, width, height, {38, 96, 76});
    const int card_w = std::max(16, width / 5);
    const int card_h = std::max(24, height / 2);
    for (int player = 0; player < 2; ++player) {
      const int left = (player + 1) * width / 3 - card_w / 2;
      const int top = height / 2 - card_h / 2;
      const int shade = 215 - cards_[player] * 35;
      board_games::FillRect(
          rgb, width, height, left, top, left + card_w, top + card_h,
          board_games::Rgb{static_cast<unsigned char>(shade), 230, 245});
      const int bar = std::min(width / 6, pot_[player] * width / 24);
      board_games::FillRect(rgb, width, height, left, top + card_h + 4,
                            left + bar, top + card_h + 10, {245, 220, 80});
    }
  }

 private:
  std::array<int, 2> cards_{-1, -1};
  std::array<int, 2> pot_{0, 0};
  std::array<bool, 2> legal_action_mask_{true, true};
  std::array<float, 2> rewards_{0.0f, 0.0f};
  int current_player_{0};
  int last_action_{-1};
  bool done_{true};

  void StepGame(int action) {
    std::array<int, 2> pot = pot_;
    if (action == 0) {
      ++pot[current_player_];
    }

    std::array<float, 2> rewards{0.0f, 0.0f};
    bool terminated = false;
    if (last_action_ == 0 && action == 1) {
      terminated = true;
      rewards = {-1.0f, -1.0f};
      rewards[1 - current_player_] = 1.0f;
    } else if (last_action_ == 0 && action == 0) {
      terminated = true;
      rewards = UnitReward(2.0f);
    } else if (last_action_ == 1 && action == 1) {
      terminated = true;
      rewards = UnitReward(1.0f);
    }

    current_player_ = 1 - current_player_;
    last_action_ = action;
    pot_ = pot;
    done_ = terminated;
    rewards_ = rewards;
    legal_action_mask_ = terminated ? std::array<bool, 2>{false, false}
                                    : std::array<bool, 2>{true, true};
  }

  std::array<float, 2> UnitReward(float scale) const {
    std::array<float, 2> rewards{-scale, -scale};
    const int winner = cards_[current_player_] > cards_[1 - current_player_]
                           ? current_player_
                           : 1 - current_player_;
    rewards[winner] = scale;
    return rewards;
  }

  void WriteState(const std::array<float, 2>& rewards) {
    State state = Allocate(2);
    state["info:current_player"_] = current_player_;
    state["info:last_action"_] = last_action_;
    for (int i = 0; i < 2; ++i) {
      state["info:cards"_][i] = cards_[i];
      state["info:legal_action_mask"_][i] = legal_action_mask_[i];
      state["info:pot"_][i] = pot_[i];
    }
    for (int player = 0; player < 2; ++player) {
      state["reward"_][player] = rewards[player];
      state["info:players.id"_][player] = player;
      WriteObservation(player, &state);
    }
  }

  void WriteObservation(int player, State* state) const {
    for (int i = 0; i < 7; ++i) {
      (*state)["obs"_](player, i) = false;
    }
    (*state)["obs"_](player, cards_[player]) = true;
    (*state)["obs"_](player, 3 + pot_[player]) = true;
    (*state)["obs"_](player, 5 + pot_[1 - player]) = true;
  }
};

class LeducHoldemEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("task"_.Bind(std::string("leduc_holdem")));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("obs"_.Bind(Spec<bool>({-1, 34})),
                    "info:cards"_.Bind(Spec<int>({3}, {-1, 2})),
                    "info:chips"_.Bind(Spec<int>({2})),
                    "info:current_player"_.Bind(Spec<int>({}, {0, 1})),
                    "info:first_player"_.Bind(Spec<int>({}, {0, 1})),
                    "info:last_action"_.Bind(Spec<int>({}, {-1, 2})),
                    "info:legal_action_mask"_.Bind(Spec<bool>({3})),
                    "info:players.id"_.Bind(Spec<int>({-1}, {0, 1})),
                    "info:raise_count"_.Bind(Spec<int>({})),
                    "info:round"_.Bind(Spec<int>({}, {0, 2})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 2})));
  }
};

using LeducHoldemEnvSpec = EnvSpec<LeducHoldemEnvFns>;

class LeducHoldemEnv : public Env<LeducHoldemEnvSpec>, public RenderableEnv {
 public:
  using Spec = LeducHoldemEnvSpec;
  using Action = typename Env<LeducHoldemEnvSpec>::Action;

  LeducHoldemEnv(const Spec& spec, int env_id)
      : Env<LeducHoldemEnvSpec>(spec, env_id) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    current_player_ = static_cast<int>(gen_() & 1U);
    first_player_ = current_player_;
    std::array<int, 6> deck{0, 0, 1, 1, 2, 2};
    std::shuffle(deck.begin(), deck.end(), gen_);
    cards_[0] = deck[0];
    cards_[1] = deck[1];
    cards_[2] = deck[2];
    last_action_ = -1;
    chips_ = {1, 1};
    round_ = 0;
    raise_count_ = 0;
    legal_action_mask_ = {true, true, false};
    rewards_ = {0.0f, 0.0f};
    done_ = false;
    WriteState(rewards_);
  }

  void Step(const Action& action) override {
    const int act = action["action"_][0];
    const int loser = current_player_;
    const bool illegal =
        act < 0 || act > 2 || (act >= 0 && !legal_action_mask_[act]);
    if (act >= 0 && act <= 2) {
      StepGame(act);
    }

    std::array<float, 2> rewards = rewards_;
    if (illegal) {
      done_ = true;
      legal_action_mask_ = {true, true, true};
      rewards = board_games::IllegalRewards(loser);
    } else if (done_) {
      legal_action_mask_ = {true, true, true};
    }
    WriteState(rewards);
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 224, height > 0 ? height : 144};
  }

  void Render(int width, int height, int camera_id,
              unsigned char* rgb) override {
    (void)camera_id;
    board_games::Fill(rgb, width, height, {45, 90, 70});
    const int card_w = std::max(16, width / 6);
    const int card_h = std::max(24, height / 2);
    for (int i = 0; i < 3; ++i) {
      if (i == 2 && round_ == 0) {
        continue;
      }
      const int left = (i + 1) * width / 4 - card_w / 2;
      const int top = height / 2 - card_h / 2;
      const int shade = 215 - cards_[i] * 35;
      board_games::FillRect(
          rgb, width, height, left, top, left + card_w, top + card_h,
          board_games::Rgb{245, static_cast<unsigned char>(shade), 220});
    }
    for (int player = 0; player < 2; ++player) {
      const int bar = std::min(width / 3, chips_[player] * width / 40);
      const int y = player == 0 ? height - 16 : height - 8;
      board_games::FillRect(rgb, width, height, width / 2 - bar / 2, y,
                            width / 2 + bar / 2, y + 4, {235, 220, 70});
    }
  }

 private:
  std::array<int, 3> cards_{-1, -1, -1};
  std::array<int, 2> chips_{1, 1};
  std::array<bool, 3> legal_action_mask_{true, true, false};
  std::array<float, 2> rewards_{0.0f, 0.0f};
  int current_player_{0};
  int first_player_{0};
  int last_action_{-1};
  int round_{0};
  int raise_count_{0};
  bool done_{true};

  int RaiseChips() const { return (round_ + 1) * 2; }

  void StepGame(int action) {
    std::array<int, 2> chips = chips_;
    if (action == 0) {
      chips[current_player_] = chips[1 - current_player_];
    } else if (action == 1) {
      chips[current_player_] = std::max(chips[0], chips[1]) + RaiseChips();
    }

    const bool fold = action == 2;
    const bool call = last_action_ != -1 && action == 0;
    const bool continue_game = round_ == 0 && call;
    const bool round_over = fold || call;
    const bool terminated = round_over && !continue_game;
    std::array<float, 2> reward{0.0f, 0.0f};
    if (fold) {
      reward = {-1.0f, -1.0f};
      reward[1 - current_player_] = 1.0f;
    }
    if (terminated && call) {
      reward = UnitReward();
    }
    const int min_chips = std::min(chips[0], chips[1]);
    reward[0] *= static_cast<float>(min_chips);
    reward[1] *= static_cast<float>(min_chips);

    const int new_last_action = round_over ? -1 : action;
    const int new_current_player =
        round_over ? first_player_ : 1 - current_player_;
    int new_raise_count = round_over ? 0 : raise_count_ + (action == 1);
    std::array<bool, 3> legal{};
    if (action == 0) {
      legal = {true, true, false};
    } else if (action == 1) {
      legal = {true, true, true};
    } else {
      legal = {false, false, false};
    }
    legal[1] = new_raise_count < 2;

    chips_ = chips;
    current_player_ = new_current_player;
    last_action_ = new_last_action;
    raise_count_ = new_raise_count;
    round_ += round_over ? 1 : 0;
    legal_action_mask_ = legal;
    rewards_ = reward;
    done_ = terminated;
  }

  std::array<float, 2> UnitReward() const {
    const bool win_by_pair = cards_[current_player_] == cards_[2];
    const bool lose_by_pair = cards_[1 - current_player_] == cards_[2];
    const bool win =
        win_by_pair || (!lose_by_pair &&
                        cards_[current_player_] > cards_[1 - current_player_]);
    if (cards_[current_player_] == cards_[1 - current_player_]) {
      return {0.0f, 0.0f};
    }
    std::array<float, 2> reward{-1.0f, -1.0f};
    reward[win ? current_player_ : 1 - current_player_] = 1.0f;
    return reward;
  }

  void WriteState(const std::array<float, 2>& rewards) {
    State state = Allocate(2);
    state["info:current_player"_] = current_player_;
    state["info:first_player"_] = first_player_;
    state["info:last_action"_] = last_action_;
    state["info:raise_count"_] = raise_count_;
    state["info:round"_] = round_;
    for (int i = 0; i < 3; ++i) {
      state["info:cards"_][i] = cards_[i];
      state["info:legal_action_mask"_][i] = legal_action_mask_[i];
    }
    for (int i = 0; i < 2; ++i) {
      state["info:chips"_][i] = chips_[i];
    }
    for (int player = 0; player < 2; ++player) {
      state["reward"_][player] = rewards[player];
      state["info:players.id"_][player] = player;
      WriteObservation(player, &state);
    }
  }

  void WriteObservation(int player, State* state) const {
    for (int i = 0; i < 34; ++i) {
      (*state)["obs"_](player, i) = false;
    }
    (*state)["obs"_](player, cards_[player]) = true;
    if (round_ == 1) {
      (*state)["obs"_](player, 3 + cards_[2]) = true;
    }
    (*state)["obs"_](player, 6 + chips_[player]) = true;
    (*state)["obs"_](player, 20 + chips_[1 - player]) = true;
  }
};

using KuhnPokerEnvPool = AsyncEnvPool<KuhnPokerEnv>;
using LeducHoldemEnvPool = AsyncEnvPool<LeducHoldemEnv>;

}  // namespace pgx

#endif  // ENVPOOL_PGX_CARD_GAMES_H_
