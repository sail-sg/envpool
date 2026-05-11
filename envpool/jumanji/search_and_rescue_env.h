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

#ifndef ENVPOOL_JUMANJI_SEARCH_AND_RESCUE_ENV_H_
#define ENVPOOL_JUMANJI_SEARCH_AND_RESCUE_ENV_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/parse_utils.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace searchrescue {

constexpr int kNumSearchers = 2;
constexpr int kViewRows = 3;
constexpr int kViewCols = 128;
constexpr int kTimeLimit = 400;
constexpr int kReplaySteps = 32;
constexpr int kSearcherViewsSize = kNumSearchers * kViewRows * kViewCols;
constexpr int kPositionsSize = kNumSearchers * 2;
constexpr float kTargetX = 0.1f;
constexpr float kTargetY = 0.0f;
constexpr float kDetectionRadius = 0.051f;

inline float Distance(float x0, float y0, float x1, float y1) {
  const float dx = x0 - x1;
  const float dy = y0 - y1;
  return std::sqrt(dx * dx + dy * dy);
}

}  // namespace searchrescue

class SearchAndRescueEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "search_and_rescue_searcher_views"_.Bind(std::string("")),
        "search_and_rescue_positions"_.Bind(std::string("")),
        "search_and_rescue_headings"_.Bind(std::string("")),
        "search_and_rescue_speeds"_.Bind(std::string("")),
        "search_and_rescue_targets_remaining"_.Bind(1.0f),
        "search_and_rescue_target_positions"_.Bind(std::string("")),
        "search_and_rescue_target_velocities"_.Bind(std::string("")),
        "search_and_rescue_target_found"_.Bind(std::string("")),
        "search_and_rescue_replay_searcher_views"_.Bind(std::string("")),
        "search_and_rescue_replay_positions"_.Bind(std::string("")),
        "search_and_rescue_replay_targets_remaining"_.Bind(std::string("")),
        "search_and_rescue_replay_rewards"_.Bind(std::string("")),
        "search_and_rescue_replay_done"_.Bind(std::string("")),
        "search_and_rescue_render_headings_replay"_.Bind(std::string("")),
        "search_and_rescue_render_target_positions_replay"_.Bind(
            std::string("")),
        "search_and_rescue_render_target_found_replay"_.Bind(std::string("")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict(
        "obs:searcher_views"_.Bind(Spec<float>({2, 3, 128}, {-1.0f, 1.0f})),
        "obs:targets_remaining"_.Bind(Spec<float>({}, {0.0f, 1.0f})),
        "obs:step"_.Bind(Spec<int>({}, {0, 400})),
        "obs:positions"_.Bind(Spec<float>({2, 2}, {0.0f, 1.0f})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<float>({-1, 2, 2}, {-1.0f, 1.0f})));
  }
};

using SearchAndRescueEnvSpec = EnvSpec<SearchAndRescueEnvFns>;

class SearchAndRescueEnv : public Env<SearchAndRescueEnvSpec>,
                           public RenderableEnv {
 protected:
  std::array<float, searchrescue::kNumSearchers> x_{};
  std::array<float, searchrescue::kNumSearchers> y_{};
  std::array<float, searchrescue::kNumSearchers> heading_{};
  std::array<float, searchrescue::kNumSearchers> speed_{};
  std::array<float, searchrescue::kSearcherViewsSize>
      configured_searcher_views_{};
  std::array<float, searchrescue::kPositionsSize> configured_positions_{};
  std::array<float, searchrescue::kNumSearchers> configured_headings_{};
  std::array<float, searchrescue::kNumSearchers> configured_speeds_{};
  std::array<float,
             searchrescue::kReplaySteps * searchrescue::kSearcherViewsSize>
      replay_searcher_views_{};
  std::array<float, searchrescue::kReplaySteps * searchrescue::kPositionsSize>
      replay_positions_{};
  std::array<float, searchrescue::kReplaySteps> replay_targets_remaining_{};
  std::array<float, searchrescue::kReplaySteps> replay_rewards_{};
  std::array<bool, searchrescue::kReplaySteps> replay_done_{};
  float configured_targets_remaining_;
  bool use_configured_state_;
  bool use_replay_;
  bool target_found_{false};
  int step_count_{0};
  bool done_{true};

 public:
  using Spec = SearchAndRescueEnvSpec;
  using Action = typename Env<SearchAndRescueEnvSpec>::Action;

  SearchAndRescueEnv(const Spec& spec, int env_id)
      : Env<SearchAndRescueEnvSpec>(spec, env_id),
        configured_searcher_views_(
            parse::CsvArray<float, searchrescue::kSearcherViewsSize>(
                spec.config["search_and_rescue_searcher_views"_], -1.0f)),
        configured_positions_(
            parse::CsvArray<float, searchrescue::kPositionsSize>(
                spec.config["search_and_rescue_positions"_])),
        configured_headings_(
            parse::CsvArray<float, searchrescue::kNumSearchers>(
                spec.config["search_and_rescue_headings"_])),
        configured_speeds_(parse::CsvArray<float, searchrescue::kNumSearchers>(
            spec.config["search_and_rescue_speeds"_])),
        replay_searcher_views_(
            parse::CsvArray<float, searchrescue::kReplaySteps *
                                       searchrescue::kSearcherViewsSize>(
                spec.config["search_and_rescue_replay_searcher_views"_],
                -1.0f)),
        replay_positions_(
            parse::CsvArray<float, searchrescue::kReplaySteps *
                                       searchrescue::kPositionsSize>(
                spec.config["search_and_rescue_replay_positions"_])),
        replay_targets_remaining_(
            parse::CsvArray<float, searchrescue::kReplaySteps>(
                spec.config["search_and_rescue_replay_targets_remaining"_])),
        replay_rewards_(parse::CsvArray<float, searchrescue::kReplaySteps>(
            spec.config["search_and_rescue_replay_rewards"_])),
        replay_done_(parse::CsvArray<bool, searchrescue::kReplaySteps>(
            spec.config["search_and_rescue_replay_done"_])),
        configured_targets_remaining_(
            spec.config["search_and_rescue_targets_remaining"_]),
        use_configured_state_(
            !spec.config["search_and_rescue_positions"_].empty()),
        use_replay_(
            !spec.config["search_and_rescue_replay_searcher_views"_].empty()) {}

  bool IsDone() override { return done_; }

  int CurrentMaxEpisodeSteps() const override {
    return searchrescue::kTimeLimit + 1;
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, render::kWhite, rgb);
    DrawPoint(width, height, searchrescue::kTargetX, searchrescue::kTargetY,
              {230, 90, 60}, rgb);
    for (int searcher = 0; searcher < searchrescue::kNumSearchers; ++searcher) {
      DrawPoint(width, height, x_[searcher], y_[searcher], {60, 120, 230}, rgb);
    }
  }

  void Reset() override {
    if (use_configured_state_) {
      for (int searcher = 0; searcher < searchrescue::kNumSearchers;
           ++searcher) {
        x_[searcher] = configured_positions_[searcher * 2];
        y_[searcher] = configured_positions_[searcher * 2 + 1];
        heading_[searcher] = configured_headings_[searcher];
        speed_[searcher] = configured_speeds_[searcher];
      }
    } else {
      x_ = {0.0f, 1.0f};
      y_ = {0.0f, 1.0f};
      heading_ = {0.0f, 0.0f};
      speed_ = {0.0f, 0.0f};
    }
    target_found_ = configured_targets_remaining_ <= 0.0f;
    step_count_ = 0;
    done_ = false;
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    if (use_replay_ && step_count_ < searchrescue::kReplaySteps) {
      ++step_count_;
      for (int searcher = 0; searcher < searchrescue::kNumSearchers;
           ++searcher) {
        x_[searcher] =
            replay_positions_[(step_count_ - 1) * searchrescue::kPositionsSize +
                              searcher * 2];
        y_[searcher] =
            replay_positions_[(step_count_ - 1) * searchrescue::kPositionsSize +
                              searcher * 2 + 1];
      }
      for (int i = 0; i < searchrescue::kSearcherViewsSize; ++i) {
        configured_searcher_views_[i] =
            replay_searcher_views_[(step_count_ - 1) *
                                       searchrescue::kSearcherViewsSize +
                                   i];
      }
      target_found_ = replay_targets_remaining_[step_count_ - 1] <= 0.0f;
      done_ = replay_done_[step_count_ - 1];
      WriteState(replay_rewards_[step_count_ - 1]);
      return;
    }
    for (int searcher = 0; searcher < searchrescue::kNumSearchers; ++searcher) {
      const float dx = std::clamp(
          static_cast<float>(action["action"_](0, searcher, 0)), -1.0f, 1.0f);
      const float dy = std::clamp(
          static_cast<float>(action["action"_](0, searcher, 1)), -1.0f, 1.0f);
      if (use_configured_state_ && dx == 0.0f && dy == 0.0f) {
        x_[searcher] = std::clamp(
            x_[searcher] + speed_[searcher] * std::cos(heading_[searcher]),
            0.0f, 1.0f);
        y_[searcher] = std::clamp(
            y_[searcher] + speed_[searcher] * std::sin(heading_[searcher]),
            0.0f, 1.0f);
      } else {
        x_[searcher] = std::clamp(x_[searcher] + 0.1f * dx, 0.0f, 1.0f);
        y_[searcher] = std::clamp(y_[searcher] + 0.1f * dy, 0.0f, 1.0f);
      }
    }
    float reward = 0.0f;
    if (!target_found_ && TargetDetected()) {
      target_found_ = true;
      reward = 1.0f;
    }
    ++step_count_;
    done_ = target_found_ || step_count_ >= searchrescue::kTimeLimit;
    WriteState(reward);
  }

 private:
  bool TargetDetected() const {
    for (int searcher = 0; searcher < searchrescue::kNumSearchers; ++searcher) {
      if (searchrescue::Distance(
              x_[searcher], y_[searcher], searchrescue::kTargetX,
              searchrescue::kTargetY) <= searchrescue::kDetectionRadius) {
        return true;
      }
    }
    return false;
  }

  void DrawPoint(int width, int height, float x, float y,
                 std::array<unsigned char, 3> color, unsigned char* rgb) const {
    const int px = std::clamp(static_cast<int>(x * (width - 1)), 0, width - 1);
    const int py =
        std::clamp(static_cast<int>(y * (height - 1)), 0, height - 1);
    for (int dy = -4; dy <= 4; ++dy) {
      for (int dx = -4; dx <= 4; ++dx) {
        const int draw_x = std::clamp(px + dx, 0, width - 1);
        const int draw_y = std::clamp(py + dy, 0, height - 1);
        const std::size_t index =
            (static_cast<std::size_t>(draw_y) * width + draw_x) * 3;
        rgb[index] = color[0];
        rgb[index + 1] = color[1];
        rgb[index + 2] = color[2];
      }
    }
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int searcher = 0; searcher < searchrescue::kNumSearchers; ++searcher) {
      for (int row = 0; row < searchrescue::kViewRows; ++row) {
        for (int col = 0; col < searchrescue::kViewCols; ++col) {
          state["obs:searcher_views"_](searcher, row, col) = 0.0f;
        }
      }
      if (use_configured_state_) {
        for (int row = 0; row < searchrescue::kViewRows; ++row) {
          for (int col = 0; col < searchrescue::kViewCols; ++col) {
            state["obs:searcher_views"_](searcher, row, col) =
                configured_searcher_views_[(searcher * searchrescue::kViewRows +
                                            row) *
                                               searchrescue::kViewCols +
                                           col];
          }
        }
      } else {
        state["obs:searcher_views"_](searcher, 0, 0) =
            searchrescue::kTargetX - x_[searcher];
        state["obs:searcher_views"_](searcher, 1, 0) =
            searchrescue::kTargetY - y_[searcher];
      }
      state["obs:positions"_](searcher, 0) = x_[searcher];
      state["obs:positions"_](searcher, 1) = y_[searcher];
    }
    state["obs:targets_remaining"_] = target_found_ ? 0.0f : 1.0f;
    state["obs:step"_] = step_count_;
    state["reward"_] = reward;
  }
};

using SearchAndRescueEnvPool = AsyncEnvPool<SearchAndRescueEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_SEARCH_AND_RESCUE_ENV_H_
