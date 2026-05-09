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

#ifndef ENVPOOL_JUMANJI_LEVEL_BASED_FORAGING_ENV_H_
#define ENVPOOL_JUMANJI_LEVEL_BASED_FORAGING_ENV_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <sstream>
#include <string>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace lbf {

constexpr int kGridSize = 8;
constexpr int kNumAgents = 2;
constexpr int kNumFood = 2;
constexpr int kViewSize = 3 * (kNumAgents + kNumFood);
constexpr int kTimeLimit = 100;
constexpr std::array<std::array<int, 2>, 6> kMoves = {
    {{{0, 0}}, {{-1, 0}}, {{1, 0}}, {{0, -1}}, {{0, 1}}, {{0, 0}}}};  // NOLINT

using AgentEntities = std::array<int, kNumAgents * 3>;
using FoodEntities = std::array<int, kNumFood * 3>;

inline AgentEntities ParseAgents(const std::string& text) {
  AgentEntities agents{};
  if (text.empty()) {
    return agents;
  }
  std::stringstream stream(text);
  std::string token;
  int index = 0;
  while (std::getline(stream, token, ',') && index < kNumAgents * 3) {
    agents[index++] = std::stoi(token);
  }
  return agents;
}

inline FoodEntities ParseFood(const std::string& text) {
  FoodEntities food{};
  if (text.empty()) {
    return food;
  }
  std::stringstream stream(text);
  std::string token;
  int index = 0;
  while (std::getline(stream, token, ',') && index < kNumFood * 3) {
    food[index++] = std::stoi(token);
  }
  return food;
}

inline bool InGrid(int row, int col) {
  return 0 <= row && row < kGridSize && 0 <= col && col < kGridSize;
}

inline bool Adjacent(int row0, int col0, int row1, int col1) {
  return std::abs(row0 - row1) + std::abs(col0 - col1) == 1;
}

}  // namespace lbf

class LevelBasedForagingEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("lbf_agents"_.Bind(std::string("")),
                    "lbf_food"_.Bind(std::string("")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict(
        "obs:agents_view"_.Bind(Spec<int>({2, 12}, {-1, 8})),
        "obs:action_mask"_.Bind(Spec<bool>({2, 6}, {false, true})),
        "obs:step_count"_.Bind(Spec<int>({}, {0, 100})),
        "info:percent_eaten"_.Bind(Spec<float>({}, {0.0f, 100.0f})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1, 2}, {0, 5})));
  }
};

using LevelBasedForagingEnvSpec = EnvSpec<LevelBasedForagingEnvFns>;

class LevelBasedForagingEnv : public Env<LevelBasedForagingEnvSpec>,
                              public RenderableEnv {
 protected:
  std::array<int, lbf::kNumAgents> agent_row_{};
  std::array<int, lbf::kNumAgents> agent_col_{};
  std::array<int, lbf::kNumAgents> agent_level_{};
  std::array<bool, lbf::kNumAgents> loading_{};
  std::array<int, lbf::kNumFood> food_row_{};
  std::array<int, lbf::kNumFood> food_col_{};
  std::array<int, lbf::kNumFood> food_level_{};
  std::array<bool, lbf::kNumFood> eaten_{};
  lbf::AgentEntities configured_agents_{};
  lbf::FoodEntities configured_food_{};
  bool use_configured_agents_;
  bool use_configured_food_;
  int step_count_{0};
  bool done_{true};

 public:
  using Spec = LevelBasedForagingEnvSpec;
  using Action = typename Env<LevelBasedForagingEnvSpec>::Action;

  LevelBasedForagingEnv(const Spec& spec, int env_id)
      : Env<LevelBasedForagingEnvSpec>(spec, env_id),
        configured_agents_(lbf::ParseAgents(spec.config["lbf_agents"_])),
        configured_food_(lbf::ParseFood(spec.config["lbf_food"_])),
        use_configured_agents_(!spec.config["lbf_agents"_].empty()),
        use_configured_food_(!spec.config["lbf_food"_].empty()) {}

  bool IsDone() override { return done_; }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, {12, 12, 12}, rgb);
    render::DrawGrid(width, height, lbf::kGridSize, lbf::kGridSize,
                     {55, 55, 55}, rgb);
    for (int food = 0; food < lbf::kNumFood; ++food) {
      if (eaten_[food]) {
        continue;
      }
      auto [x, y] =
          render::CellCenter(width, height, lbf::kGridSize, lbf::kGridSize,
                             food_row_[food], food_col_[food]);
      render::FillCircle(width, height, x, y,
                         std::max(3, std::min(width, height) / 35),
                         {210, 30, 30}, rgb);
      render::DrawNumber(width, height, food_level_[food], x - 4, y - 4, x + 5,
                         y + 5, render::kWhite, rgb);
    }
    for (int agent = 0; agent < lbf::kNumAgents; ++agent) {
      auto [x, y] =
          render::CellCenter(width, height, lbf::kGridSize, lbf::kGridSize,
                             agent_row_[agent], agent_col_[agent]);
      render::FillCircle(width, height, x, y,
                         std::max(3, std::min(width, height) / 32),
                         render::Palette(agent), rgb);
      render::DrawNumber(width, height, agent_level_[agent], x - 4, y - 4,
                         x + 5, y + 5, render::kWhite, rgb);
    }
  }

  void Reset() override {
    if (use_configured_agents_) {
      for (int agent = 0; agent < lbf::kNumAgents; ++agent) {
        agent_row_[agent] =
            std::clamp(configured_agents_[agent * 3], 0, lbf::kGridSize - 1);
        agent_col_[agent] = std::clamp(configured_agents_[agent * 3 + 1], 0,
                                       lbf::kGridSize - 1);
        agent_level_[agent] = std::max(1, configured_agents_[agent * 3 + 2]);
      }
    } else {
      agent_row_ = {0, 0};
      agent_col_ = {0, 1};
      agent_level_ = {1, 1};
    }
    loading_.fill(false);
    if (use_configured_food_) {
      for (int food = 0; food < lbf::kNumFood; ++food) {
        food_row_[food] =
            std::clamp(configured_food_[food * 3], 0, lbf::kGridSize - 1);
        food_col_[food] =
            std::clamp(configured_food_[food * 3 + 1], 0, lbf::kGridSize - 1);
        food_level_[food] = std::max(1, configured_food_[food * 3 + 2]);
      }
    } else {
      food_row_ = {1, 7};
      food_col_ = {0, 7};
      food_level_ = {2, 2};
    }
    eaten_.fill(false);
    step_count_ = 0;
    done_ = false;
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    std::array<int, lbf::kNumAgents> next_row = agent_row_;
    std::array<int, lbf::kNumAgents> next_col = agent_col_;
    for (int agent = 0; agent < lbf::kNumAgents; ++agent) {
      const int action_id =
          std::clamp(static_cast<int>(action["action"_](0, agent)), 0, 5);
      loading_[agent] = action_id == 5;
      if (action_id == 5) {
        continue;
      }
      const int row = agent_row_[agent] + lbf::kMoves[action_id][0];
      const int col = agent_col_[agent] + lbf::kMoves[action_id][1];
      if (IsFree(agent, row, col)) {
        next_row[agent] = row;
        next_col[agent] = col;
      }
    }
    if (next_row[0] == next_row[1] && next_col[0] == next_col[1]) {
      next_row = agent_row_;
      next_col = agent_col_;
    }
    agent_row_ = next_row;
    agent_col_ = next_col;

    float reward = 0.0f;
    const int total_food_level = food_level_[0] + food_level_[1];
    for (int food = 0; food < lbf::kNumFood; ++food) {
      if (eaten_[food]) {
        continue;
      }
      int loading_level = 0;
      for (int agent = 0; agent < lbf::kNumAgents; ++agent) {
        if (loading_[agent] &&
            lbf::Adjacent(agent_row_[agent], agent_col_[agent], food_row_[food],
                          food_col_[food])) {
          loading_level += agent_level_[agent];
        }
      }
      if (loading_level >= food_level_[food]) {
        eaten_[food] = true;
        reward += static_cast<float>(food_level_[food]) / total_food_level;
      }
    }
    ++step_count_;
    done_ = AllFoodEaten() || step_count_ >= lbf::kTimeLimit;
    WriteState(reward);
  }

 private:
  bool IsFree(int moving_agent, int row, int col) const {
    if (!lbf::InGrid(row, col)) {
      return false;
    }
    for (int food = 0; food < lbf::kNumFood; ++food) {
      if (!eaten_[food] && row == food_row_[food] && col == food_col_[food]) {
        return false;
      }
    }
    for (int agent = 0; agent < lbf::kNumAgents; ++agent) {
      if (agent != moving_agent && row == agent_row_[agent] &&
          col == agent_col_[agent]) {
        return false;
      }
    }
    return true;
  }

  bool LoadAvailable(int agent) const {
    for (int food = 0; food < lbf::kNumFood; ++food) {
      if (!eaten_[food] && lbf::Adjacent(agent_row_[agent], agent_col_[agent],
                                         food_row_[food], food_col_[food])) {
        return true;
      }
    }
    return false;
  }

  bool AllFoodEaten() const {
    return std::all_of(eaten_.begin(), eaten_.end(),
                       [](bool value) { return value; });
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int viewer = 0; viewer < lbf::kNumAgents; ++viewer) {
      int cursor = 0;
      for (int food = 0; food < lbf::kNumFood; ++food) {
        state["obs:agents_view"_](viewer, cursor++) =
            eaten_[food] ? -1 : food_row_[food];
        state["obs:agents_view"_](viewer, cursor++) =
            eaten_[food] ? -1 : food_col_[food];
        state["obs:agents_view"_](viewer, cursor++) =
            eaten_[food] ? 0 : food_level_[food];
      }
      state["obs:agents_view"_](viewer, cursor++) = agent_row_[viewer];
      state["obs:agents_view"_](viewer, cursor++) = agent_col_[viewer];
      state["obs:agents_view"_](viewer, cursor++) = agent_level_[viewer];
      for (int agent = 0; agent < lbf::kNumAgents; ++agent) {
        if (agent == viewer) {
          continue;
        }
        state["obs:agents_view"_](viewer, cursor++) = agent_row_[agent];
        state["obs:agents_view"_](viewer, cursor++) = agent_col_[agent];
        state["obs:agents_view"_](viewer, cursor++) = agent_level_[agent];
      }
      for (int action = 0; action < 6; ++action) {
        if (action == 5) {
          state["obs:action_mask"_](viewer, action) = LoadAvailable(viewer);
        } else {
          const int row = agent_row_[viewer] + lbf::kMoves[action][0];
          const int col = agent_col_[viewer] + lbf::kMoves[action][1];
          state["obs:action_mask"_](viewer, action) = IsFree(viewer, row, col);
        }
      }
    }
    state["obs:step_count"_] = step_count_;
    const int eaten =
        static_cast<int>(std::count(eaten_.begin(), eaten_.end(), true));
    state["info:percent_eaten"_] =
        100.0f * static_cast<float>(eaten) / lbf::kNumFood;
    state["reward"_] = reward;
  }
};

using LevelBasedForagingEnvPool = AsyncEnvPool<LevelBasedForagingEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_LEVEL_BASED_FORAGING_ENV_H_
