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

#ifndef ENVPOOL_JUMANJI_KNAPSACK_ENV_H_
#define ENVPOOL_JUMANJI_KNAPSACK_ENV_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <sstream>
#include <string>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace knapsack {

constexpr int kNumItems = 50;

using Values = std::array<float, kNumItems>;
using Packed = std::array<bool, kNumItems>;

inline Values ParseValues(const std::string& text, float fill_value) {
  Values values{};
  values.fill(fill_value);
  if (text.empty()) {
    return values;
  }
  std::stringstream stream(text);
  std::string token;
  int index = 0;
  while (std::getline(stream, token, ',') && index < kNumItems) {
    values[index++] = std::stof(token);
  }
  return values;
}

}  // namespace knapsack

class KnapsackEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("knapsack_weights"_.Bind(std::string("")),
                    "knapsack_values"_.Bind(std::string("")),
                    "knapsack_total_budget"_.Bind(12.5f));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict(
        "obs:weights"_.Bind(Spec<float>({50}, {0.0f, 1.0f})),
        "obs:values"_.Bind(Spec<float>({50}, {0.0f, 1.0f})),
        "obs:packed_items"_.Bind(Spec<bool>({50}, {false, true})),
        "obs:action_mask"_.Bind(Spec<bool>({50}, {false, true})),
        "info:remaining_budget"_.Bind(Spec<float>({}, {0.0f, 12.5f})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 49})));
  }
};

using KnapsackEnvSpec = EnvSpec<KnapsackEnvFns>;

class KnapsackEnv : public Env<KnapsackEnvSpec>, public RenderableEnv {
 protected:
  knapsack::Values weights_{};
  knapsack::Values values_{};
  knapsack::Values configured_weights_{};
  knapsack::Values configured_values_{};
  knapsack::Packed packed_{};
  float total_budget_;
  float remaining_budget_;
  bool use_configured_items_;
  bool done_{true};

 public:
  using Spec = KnapsackEnvSpec;
  using Action = typename Env<KnapsackEnvSpec>::Action;

  KnapsackEnv(const Spec& spec, int env_id)
      : Env<KnapsackEnvSpec>(spec, env_id),
        configured_weights_(
            knapsack::ParseValues(spec.config["knapsack_weights"_], 1.0f)),
        configured_values_(
            knapsack::ParseValues(spec.config["knapsack_values"_], 0.0f)),
        total_budget_(spec.config["knapsack_total_budget"_]),
        remaining_budget_(total_budget_),
        use_configured_items_(!spec.config["knapsack_weights"_].empty() ||
                              !spec.config["knapsack_values"_].empty()) {}

  bool IsDone() override { return done_; }

  int CurrentMaxEpisodeSteps() const override {
    return knapsack::kNumItems + 1;
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, render::kWhite, rgb);
    const int bag_left = width / 4;
    const int bag_right = width * 3 / 4;
    const int bag_top = height / 5;
    const int bag_bottom = height * 4 / 5;
    render::FillRect(width, height, bag_left, bag_top + height / 8, bag_right,
                     bag_bottom, {35, 135, 95}, rgb);
    render::StrokeRect(width, height, bag_left, bag_top + height / 8, bag_right,
                       bag_bottom, {10, 80, 55}, rgb, 2);
    render::DrawLine(width, height, bag_left + width / 8, bag_top + height / 8,
                     width / 2, bag_top, {40, 90, 170}, rgb, 4);
    render::DrawLine(width, height, bag_right - width / 8, bag_top + height / 8,
                     width / 2, bag_top, {40, 90, 170}, rgb, 4);
    const int item_area_top = bag_top + height / 5;
    const int packed_count =
        static_cast<int>(std::count(packed_.begin(), packed_.end(), true));
    for (int item = 0; item < knapsack::kNumItems; ++item) {
      if (!packed_[item]) {
        continue;
      }
      const int row = item / 10;
      const int col = item % 10;
      const int x = bag_left + 8 + col * (bag_right - bag_left - 16) / 10;
      const int y = item_area_top + row * (bag_bottom - item_area_top - 8) / 5;
      render::FillCircle(width, height, x, y, 3, render::Palette(item), rgb);
    }
    const int budget_h = static_cast<int>(
        (bag_bottom - bag_top) *
        std::clamp(remaining_budget_ / total_budget_, 0.0f, 1.0f));
    render::FillRect(width, height, width - 14, bag_bottom - budget_h,
                     width - 6, bag_bottom, {80, 160, 240}, rgb);
    render::StrokeRect(width, height, width - 14, bag_top, width - 6,
                       bag_bottom, {100, 100, 100}, rgb);
    render::DrawNumber(width, height, packed_count, 6, 6, width / 5, height / 5,
                       {60, 60, 60}, rgb);
  }

  void Reset() override {
    packed_.fill(false);
    remaining_budget_ = total_budget_;
    done_ = false;
    if (use_configured_items_) {
      weights_ = configured_weights_;
      values_ = configured_values_;
    } else {
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      for (int i = 0; i < knapsack::kNumItems; ++i) {
        weights_[i] = dist(gen_);
        values_[i] = dist(gen_);
      }
    }
    done_ = !AnyActionAvailable();
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    const int item = std::clamp(static_cast<int>(action["action"_]), 0,
                                knapsack::kNumItems - 1);
    const bool valid = !packed_[item] && weights_[item] <= remaining_budget_;
    float reward = 0.0f;
    if (valid) {
      packed_[item] = true;
      remaining_budget_ -= weights_[item];
      reward = values_[item];
    }
    done_ = !valid || !AnyActionAvailable();
    WriteState(reward);
  }

 private:
  bool AnyActionAvailable() const {
    for (int i = 0; i < knapsack::kNumItems; ++i) {
      if (!packed_[i] && weights_[i] <= remaining_budget_) {
        return true;
      }
    }
    return false;
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int i = 0; i < knapsack::kNumItems; ++i) {
      state["obs:weights"_][i] = weights_[i];
      state["obs:values"_][i] = values_[i];
      state["obs:packed_items"_][i] = packed_[i];
      state["obs:action_mask"_][i] =
          !packed_[i] && weights_[i] <= remaining_budget_;
    }
    state["info:remaining_budget"_] = remaining_budget_;
    state["reward"_] = reward;
  }
};

using KnapsackEnvPool = AsyncEnvPool<KnapsackEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_KNAPSACK_ENV_H_
