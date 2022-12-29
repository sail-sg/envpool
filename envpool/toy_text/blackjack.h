/*
 * Copyright 2021 Garena Online Private Limited
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
// https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py

#ifndef ENVPOOL_TOY_TEXT_BLACKJACK_H_
#define ENVPOOL_TOY_TEXT_BLACKJACK_H_

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace toy_text {

class BlackjackEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("natural"_.Bind(false), "sab"_.Bind(true));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.Bind(Spec<int>({3}, {0, 31})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 1})));
  }
};

using BlackjackEnvSpec = EnvSpec<BlackjackEnvFns>;

class BlackjackEnv : public Env<BlackjackEnvSpec> {
 protected:
  bool natural_, sab_;
  std::vector<int> player_, dealer_;
  std::uniform_int_distribution<> dist_;
  bool done_{true};

 public:
  BlackjackEnv(const Spec& spec, int env_id)
      : Env<BlackjackEnvSpec>(spec, env_id),
        natural_(spec.config["natural"_]),
        sab_(spec.config["sab"_]),
        dist_(1, 13) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    player_.clear();
    player_.push_back(DrawCard());
    player_.push_back(DrawCard());
    dealer_.clear();
    dealer_.push_back(DrawCard());
    dealer_.push_back(DrawCard());
    done_ = false;
    WriteState(0.0);
  }

  void Step(const Action& action) override {
    int act = action["action"_];
    float reward = 0.0;
    if (act != 0) {  // hit: add a card to players hand and return
      player_.push_back(DrawCard());
      if (IsBust(player_)) {
        done_ = true;
        reward = -1.0;
      }
    } else {  // stick: play out the dealers hand, and score
      done_ = true;
      while (SumHand(dealer_) < 17) {
        dealer_.push_back(DrawCard());
      }
      int player_score = Score(player_);
      int dealer_score = Score(dealer_);
      reward = (player_score > dealer_score ? 1.0f : 0.0f) -
               (player_score < dealer_score ? 1.0f : 0.0f);
      if (sab_ && IsNatural(player_) && !IsNatural(dealer_)) {
        reward = 1.0;
      } else if (!sab_ && natural_ && IsNatural(player_) && reward == 1.0) {
        reward = 1.5;
      }
    }
    WriteState(reward);
  }

 private:
  void WriteState(float reward) {
    State state = Allocate();
    state["obs"_][0] = SumHand(player_);
    state["obs"_][1] = dealer_[0];
    state["obs"_][2] = UsableAce(player_);
    state["reward"_] = reward;
  }

  int DrawCard() { return std::min(10, dist_(gen_)); }

  static int UsableAce(const std::vector<int>& hand) {
    for (auto i : hand) {
      if (i == 1) {
        return 1;
      }
    }
    return 0;
  }

  static int SumHand(const std::vector<int>& hand) {
    int sum = 0;
    for (auto i : hand) {
      sum += i;
    }
    if (UsableAce(hand) != 0 && sum + 10 <= 21) {
      return sum + 10;
    }
    return sum;
  }

  static bool IsBust(const std::vector<int>& hand) {
    return SumHand(hand) > 21;
  }

  static int Score(const std::vector<int>& hand) {
    int result = SumHand(hand);
    return result > 21 ? 0 : result;
  }

  static bool IsNatural(const std::vector<int>& hand) {
    return hand.size() == 2 &&
           ((hand[0] == 1 && hand[1] == 10) || (hand[0] == 10 && hand[1] == 1));
  }
};

using BlackjackEnvPool = AsyncEnvPool<BlackjackEnv>;

}  // namespace toy_text

#endif  // ENVPOOL_TOY_TEXT_BLACKJACK_H_
