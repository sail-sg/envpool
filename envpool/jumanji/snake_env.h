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

#ifndef ENVPOOL_JUMANJI_SNAKE_ENV_H_
#define ENVPOOL_JUMANJI_SNAKE_ENV_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace snake {

constexpr int kRows = 12;
constexpr int kCols = 12;
constexpr int kCellCount = kRows * kCols;
constexpr int kTimeLimit = 4000;
constexpr std::array<std::array<int, 2>, 4> kMoves = {
    {{{-1, 0}}, {{0, 1}}, {{1, 0}}, {{0, -1}}}};  // NOLINT

using BodyState = std::array<int, kCellCount>;

inline int Offset(int row, int col) { return row * kCols + col; }

inline bool InGrid(int row, int col) {
  return 0 <= row && row < kRows && 0 <= col && col < kCols;
}

inline std::pair<int, int> ParsePosition(const std::string& text,
                                         int default_row, int default_col) {
  if (text.empty()) {
    return {default_row, default_col};
  }
  const std::size_t sep = text.find(',');
  if (sep == std::string::npos) {
    return {default_row, default_col};
  }
  return {std::clamp(std::stoi(text.substr(0, sep)), 0, kRows - 1),
          std::clamp(std::stoi(text.substr(sep + 1)), 0, kCols - 1)};
}

}  // namespace snake

class SnakeEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("snake_head_position"_.Bind(std::string("")),
                    "snake_fruit_position"_.Bind(std::string("")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("obs:grid"_.Bind(Spec<float>({12, 12, 5}, {0.0f, 1.0f})),
                    "obs:step_count"_.Bind(Spec<int>({}, {0, 3999})),
                    "obs:action_mask"_.Bind(Spec<bool>({4}, {false, true})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 3})));
  }
};

using SnakeEnvSpec = EnvSpec<SnakeEnvFns>;

class SnakeEnv : public Env<SnakeEnvSpec>, public RenderableEnv {
 protected:
  snake::BodyState body_state_{};
  int head_row_{0};
  int head_col_{0};
  int tail_row_{0};
  int tail_col_{0};
  int fruit_row_{0};
  int fruit_col_{1};
  int configured_head_row_{0};
  int configured_head_col_{0};
  int configured_fruit_row_{0};
  int configured_fruit_col_{1};
  int length_{1};
  int step_count_{0};
  bool done_{true};

 public:
  using Spec = SnakeEnvSpec;
  using Action = typename Env<SnakeEnvSpec>::Action;

  SnakeEnv(const Spec& spec, int env_id) : Env<SnakeEnvSpec>(spec, env_id) {
    std::tie(configured_head_row_, configured_head_col_) =
        snake::ParsePosition(spec.config["snake_head_position"_], 0, 0);
    std::tie(configured_fruit_row_, configured_fruit_col_) =
        snake::ParsePosition(spec.config["snake_fruit_position"_], 0, 1);
  }

  bool IsDone() override { return done_; }

  int CurrentMaxEpisodeSteps() const override { return snake::kTimeLimit + 1; }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, render::kWhite, rgb);
    render::StrokeRect(width, height, 0, 0, width, height, {170, 170, 170},
                       rgb);
    for (int row = 0; row < snake::kRows; ++row) {
      for (int col = 0; col < snake::kCols; ++col) {
        if (body_state_[snake::Offset(row, col)] > 0) {
          render::FillCell(width, height, snake::kRows, snake::kCols, row, col,
                           {40, 170, 40}, rgb, 2);
        }
      }
    }
    auto [fx, fy] = render::CellCenter(width, height, snake::kRows,
                                       snake::kCols, fruit_row_, fruit_col_);
    render::FillRect(width, height, fx - 4, fy - 4, fx + 5, fy + 5,
                     {65, 180, 40}, rgb);
    auto [hx, hy] = render::CellCenter(width, height, snake::kRows,
                                       snake::kCols, head_row_, head_col_);
    render::FillCircle(width, height, hx, hy,
                       std::max(3, std::min(width, height) / 40), {220, 50, 50},
                       rgb);
  }

  void Reset() override {
    body_state_.fill(0);
    head_row_ = configured_head_row_;
    head_col_ = configured_head_col_;
    fruit_row_ = configured_fruit_row_;
    fruit_col_ = configured_fruit_col_;
    if (spec_.config["snake_head_position"_].empty()) {
      std::uniform_int_distribution<int> row_dist(0, snake::kRows - 1);
      std::uniform_int_distribution<int> col_dist(0, snake::kCols - 1);
      head_row_ = row_dist(gen_);
      head_col_ = col_dist(gen_);
      do {
        fruit_row_ = row_dist(gen_);
        fruit_col_ = col_dist(gen_);
      } while (fruit_row_ == head_row_ && fruit_col_ == head_col_);
    }
    length_ = 1;
    step_count_ = 0;
    body_state_[snake::Offset(head_row_, head_col_)] = 1;
    UpdateTail();
    done_ = !AnyActionAvailable();
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    const int action_id = std::clamp(static_cast<int>(action["action"_]), 0, 3);
    const bool valid = IsActionValid(action_id);
    float reward = 0.0f;
    if (valid) {
      const int next_row = head_row_ + snake::kMoves[action_id][0];
      const int next_col = head_col_ + snake::kMoves[action_id][1];
      const bool fruit_eaten = next_row == fruit_row_ && next_col == fruit_col_;
      if (fruit_eaten) {
        ++length_;
        reward = 1.0f;
      } else {
        for (int& cell : body_state_) {
          cell = std::max(0, cell - 1);
        }
      }
      head_row_ = next_row;
      head_col_ = next_col;
      body_state_[snake::Offset(head_row_, head_col_)] = length_;
      if (fruit_eaten) {
        PlaceFruit();
      }
      UpdateTail();
    }
    ++step_count_;
    done_ = !valid || IsComplete() || step_count_ >= snake::kTimeLimit ||
            !AnyActionAvailable();
    WriteState(reward);
  }

 private:
  bool IsActionValid(int action) const {
    const int row = head_row_ + snake::kMoves[action][0];
    const int col = head_col_ + snake::kMoves[action][1];
    if (!snake::InGrid(row, col)) {
      return false;
    }
    const int body_value = body_state_[snake::Offset(row, col)];
    return body_value <= 1;
  }

  bool AnyActionAvailable() const {
    for (int action = 0; action < 4; ++action) {
      if (IsActionValid(action)) {
        return true;
      }
    }
    return false;
  }

  bool IsComplete() const {
    return std::all_of(body_state_.begin(), body_state_.end(),
                       [](int value) { return value > 0; });
  }

  void UpdateTail() {
    for (int row = 0; row < snake::kRows; ++row) {
      for (int col = 0; col < snake::kCols; ++col) {
        if (body_state_[snake::Offset(row, col)] == 1) {
          tail_row_ = row;
          tail_col_ = col;
          return;
        }
      }
    }
  }

  void PlaceFruit() {
    std::vector<int> empty;
    for (int i = 0; i < snake::kCellCount; ++i) {
      if (body_state_[i] == 0) {
        empty.push_back(i);
      }
    }
    if (empty.empty()) {
      return;
    }
    std::uniform_int_distribution<int> dist(0,
                                            static_cast<int>(empty.size()) - 1);
    const int offset = empty[dist(gen_)];
    fruit_row_ = offset / snake::kCols;
    fruit_col_ = offset % snake::kCols;
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int row = 0; row < snake::kRows; ++row) {
      for (int col = 0; col < snake::kCols; ++col) {
        const int offset = snake::Offset(row, col);
        const bool body = body_state_[offset] > 0;
        state["obs:grid"_](row, col, 0) = body ? 1.0f : 0.0f;
        state["obs:grid"_](row, col, 1) =
            row == head_row_ && col == head_col_ ? 1.0f : 0.0f;
        state["obs:grid"_](row, col, 2) =
            row == tail_row_ && col == tail_col_ ? 1.0f : 0.0f;
        state["obs:grid"_](row, col, 3) =
            row == fruit_row_ && col == fruit_col_ ? 1.0f : 0.0f;
        state["obs:grid"_](row, col, 4) =
            length_ > 0 ? static_cast<float>(body_state_[offset]) / length_
                        : 0.0f;
      }
    }
    state["obs:step_count"_] = step_count_;
    for (int action = 0; action < 4; ++action) {
      state["obs:action_mask"_][action] = IsActionValid(action);
    }
    state["reward"_] = reward;
  }
};

using SnakeEnvPool = AsyncEnvPool<SnakeEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_SNAKE_ENV_H_
