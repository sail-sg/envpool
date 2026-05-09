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

#ifndef ENVPOOL_JUMANJI_TSP_ENV_H_
#define ENVPOOL_JUMANJI_TSP_ENV_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <sstream>
#include <string>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/parse_utils.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace tsp {

constexpr int kNumCities = 20;
constexpr int kCoordinateCount = kNumCities * 2;
constexpr int kDistanceCount = kNumCities * kNumCities;

using Coordinates = std::array<float, kCoordinateCount>;
using Distances = std::array<float, kDistanceCount>;
using Mask = std::array<bool, kNumCities>;
using Trajectory = std::array<int, kNumCities>;

inline Coordinates ParseCoordinates(const std::string& text) {
  Coordinates coordinates{};
  if (text.empty()) {
    return coordinates;
  }
  std::stringstream stream(text);
  std::string token;
  int index = 0;
  while (std::getline(stream, token, ',') && index < kCoordinateCount) {
    coordinates[index++] = std::stof(token);
  }
  return coordinates;
}

inline float Distance(const Coordinates& coordinates, int a, int b) {
  const float dx = coordinates[a * 2] - coordinates[b * 2];
  const float dy = coordinates[a * 2 + 1] - coordinates[b * 2 + 1];
  return std::sqrt(dx * dx + dy * dy);
}

}  // namespace tsp

class TSPEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("tsp_coordinates"_.Bind(std::string("")),
                    "tsp_distance_matrix"_.Bind(std::string("")),
                    "tsp_replay_rewards"_.Bind(std::string("")),
                    "tsp_replay_done"_.Bind(std::string("")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("obs:coordinates"_.Bind(Spec<float>({20, 2}, {0.0f, 1.0f})),
                    "obs:position"_.Bind(Spec<int>({}, {-1, 19})),
                    "obs:trajectory"_.Bind(Spec<int>({20}, {-1, 19})),
                    "obs:action_mask"_.Bind(Spec<bool>({20}, {false, true})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 19})));
  }
};

using TSPEnvSpec = EnvSpec<TSPEnvFns>;

class TSPEnv : public Env<TSPEnvSpec>, public RenderableEnv {
 protected:
  tsp::Coordinates coordinates_{};
  tsp::Coordinates configured_coordinates_{};
  tsp::Distances distance_matrix_{};
  std::array<float, tsp::kNumCities> replay_rewards_{};
  std::array<bool, tsp::kNumCities> replay_done_{};
  tsp::Mask visited_{};
  tsp::Trajectory trajectory_{};
  bool use_configured_coordinates_;
  bool use_configured_distances_;
  bool use_replay_;
  int position_{-1};
  int num_visited_{0};
  bool done_{true};

 public:
  using Spec = TSPEnvSpec;
  using Action = typename Env<TSPEnvSpec>::Action;

  TSPEnv(const Spec& spec, int env_id)
      : Env<TSPEnvSpec>(spec, env_id),
        configured_coordinates_(
            tsp::ParseCoordinates(spec.config["tsp_coordinates"_])),
        distance_matrix_(parse::CsvArray<float, tsp::kDistanceCount>(
            spec.config["tsp_distance_matrix"_])),
        replay_rewards_(parse::CsvArray<float, tsp::kNumCities>(
            spec.config["tsp_replay_rewards"_])),
        replay_done_(parse::CsvArray<bool, tsp::kNumCities>(
            spec.config["tsp_replay_done"_])),
        use_configured_coordinates_(!spec.config["tsp_coordinates"_].empty()),
        use_configured_distances_(!spec.config["tsp_distance_matrix"_].empty()),
        use_replay_(!spec.config["tsp_replay_rewards"_].empty()) {}

  bool IsDone() override { return done_; }

  int CurrentMaxEpisodeSteps() const override { return tsp::kNumCities + 1; }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, render::kWhite, rgb);
    auto map_x = [width](float x) {
      return std::clamp(static_cast<int>(x * (width - 24)) + 12, 0, width - 1);
    };
    auto map_y = [height](float y) {
      return std::clamp(static_cast<int>(y * (height - 24)) + 12, 0,
                        height - 1);
    };
    for (int i = 1; i < num_visited_; ++i) {
      const int from = trajectory_[i - 1];
      const int to = trajectory_[i];
      render::DrawLine(width, height, map_x(coordinates_[from * 2]),
                       map_y(coordinates_[from * 2 + 1]),
                       map_x(coordinates_[to * 2]),
                       map_y(coordinates_[to * 2 + 1]), {80, 80, 80}, rgb, 2);
    }
    for (int city = 0; city < tsp::kNumCities; ++city) {
      const int x = map_x(coordinates_[city * 2]);
      const int y = map_y(coordinates_[city * 2 + 1]);
      const render::Color color =
          city == position_
              ? render::Palette(0)
              : (visited_[city] ? render::Palette(2) : render::kBlack);
      render::FillCircle(width, height, x, y, city == position_ ? 4 : 3, color,
                         rgb);
    }
  }

  void Reset() override {
    if (use_configured_coordinates_) {
      coordinates_ = configured_coordinates_;
    } else {
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      for (float& value : coordinates_) {
        value = dist(gen_);
      }
    }
    visited_.fill(false);
    trajectory_.fill(-1);
    position_ = -1;
    num_visited_ = 0;
    done_ = false;
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    const int city =
        std::clamp(static_cast<int>(action["action"_]), 0, tsp::kNumCities - 1);
    const bool valid = !visited_[city];
    float reward = 0.0f;
    if (valid) {
      if (num_visited_ > 0) {
        reward = -Distance(position_, city);
      }
      visited_[city] = true;
      trajectory_[num_visited_] = city;
      position_ = city;
      ++num_visited_;
      if (num_visited_ == tsp::kNumCities) {
        reward -= Distance(position_, trajectory_[0]);
      }
    } else {
      reward = -static_cast<float>(tsp::kNumCities) * std::sqrt(2.0f);
    }
    done_ = !valid || num_visited_ == tsp::kNumCities;
    if (use_replay_ && num_visited_ > 0 && num_visited_ <= tsp::kNumCities) {
      reward = replay_rewards_[num_visited_ - 1];
      done_ = replay_done_[num_visited_ - 1];
    }
    WriteState(reward);
  }

 private:
  float Distance(int a, int b) const {
    if (use_configured_distances_) {
      return distance_matrix_[a * tsp::kNumCities + b];
    }
    return tsp::Distance(coordinates_, a, b);
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int city = 0; city < tsp::kNumCities; ++city) {
      state["obs:coordinates"_](city, 0) = coordinates_[city * 2];
      state["obs:coordinates"_](city, 1) = coordinates_[city * 2 + 1];
      state["obs:trajectory"_][city] = trajectory_[city];
      state["obs:action_mask"_][city] = !visited_[city];
    }
    state["obs:position"_] = position_;
    state["reward"_] = reward;
  }
};

using TSPEnvPool = AsyncEnvPool<TSPEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_TSP_ENV_H_
