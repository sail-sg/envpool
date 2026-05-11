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

#ifndef ENVPOOL_JUMANJI_CVRP_ENV_H_
#define ENVPOOL_JUMANJI_CVRP_ENV_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/parse_utils.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace cvrp {

constexpr int kNumNodes = 21;
constexpr int kTrajectoryLength = 40;
constexpr int kTimeLimit = 40;
constexpr int kDistanceMatrixSize = kNumNodes * kNumNodes;

using Coordinates = std::array<float, kNumNodes * 2>;

inline Coordinates ParseCoordinates(const std::string& text) {
  Coordinates coordinates{};
  if (text.empty()) {
    return coordinates;
  }
  std::stringstream stream(text);
  std::string token;
  int index = 0;
  while (std::getline(stream, token, ',') && index < kNumNodes * 2) {
    coordinates[index++] = std::stof(token);
  }
  return coordinates;
}

inline float Distance(const std::array<float, kNumNodes * 2>& coordinates,
                      int from, int to) {
  const float dx = coordinates[from * 2] - coordinates[to * 2];
  const float dy = coordinates[from * 2 + 1] - coordinates[to * 2 + 1];
  return std::sqrt(dx * dx + dy * dy);
}

}  // namespace cvrp

class CVRPEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "cvrp_coordinates"_.Bind(std::string("")),
        "cvrp_demands"_.Bind(std::string("")),
        "cvrp_distance_matrix"_.Bind(std::string("")),
        "cvrp_replay_rewards"_.Bind(std::string("")),
        "cvrp_replay_done"_.Bind(std::string("")),
        "cvrp_render_num_total_visits_replay"_.Bind(std::string("")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict(
        "obs:coordinates"_.Bind(Spec<float>({21, 2}, {0.0f, 1.0f})),
        "obs:demands"_.Bind(Spec<float>({21}, {0.0f, 1.0f})),
        "obs:unvisited_nodes"_.Bind(Spec<bool>({21}, {false, true})),
        "obs:position"_.Bind(Spec<int>({}, {0, 20})),
        "obs:trajectory"_.Bind(Spec<int>({40}, {0, 21})),
        "obs:capacity"_.Bind(Spec<float>({}, {0.0f, 1.0f})),
        "obs:action_mask"_.Bind(Spec<bool>({21}, {false, true})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 20})));
  }
};

using CVRPEnvSpec = EnvSpec<CVRPEnvFns>;

class CVRPEnv : public Env<CVRPEnvSpec>, public RenderableEnv {
 protected:
  cvrp::Coordinates coordinates_{};
  cvrp::Coordinates configured_coordinates_{};
  std::array<float, cvrp::kNumNodes> demands_{};
  std::array<float, cvrp::kNumNodes> configured_demands_{};
  std::array<float, cvrp::kDistanceMatrixSize> configured_distances_{};
  std::array<float, cvrp::kTimeLimit> replay_rewards_{};
  std::array<bool, cvrp::kTimeLimit> replay_done_{};
  std::array<bool, cvrp::kNumNodes> unvisited_{};
  std::array<int, cvrp::kTrajectoryLength> trajectory_{};
  bool use_configured_coordinates_;
  bool use_configured_distances_;
  bool use_replay_;
  int position_{0};
  int trajectory_size_{0};
  float capacity_{1.0f};
  int step_count_{0};
  bool done_{true};

 public:
  using Spec = CVRPEnvSpec;
  using Action = typename Env<CVRPEnvSpec>::Action;

  CVRPEnv(const Spec& spec, int env_id)
      : Env<CVRPEnvSpec>(spec, env_id),
        configured_coordinates_(
            cvrp::ParseCoordinates(spec.config["cvrp_coordinates"_])),
        configured_demands_(parse::CsvArray<float, cvrp::kNumNodes>(
            spec.config["cvrp_demands"_])),
        configured_distances_(parse::CsvArray<float, cvrp::kDistanceMatrixSize>(
            spec.config["cvrp_distance_matrix"_])),
        replay_rewards_(parse::CsvArray<float, cvrp::kTimeLimit>(
            spec.config["cvrp_replay_rewards"_])),
        replay_done_(parse::CsvArray<bool, cvrp::kTimeLimit>(
            spec.config["cvrp_replay_done"_])),
        use_configured_coordinates_(!spec.config["cvrp_coordinates"_].empty()),
        use_configured_distances_(
            !spec.config["cvrp_distance_matrix"_].empty()),
        use_replay_(!spec.config["cvrp_replay_rewards"_].empty()) {}

  bool IsDone() override { return done_; }

  int CurrentMaxEpisodeSteps() const override { return cvrp::kTimeLimit + 1; }

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
    int previous = 0;
    for (int i = 0; i < trajectory_size_; ++i) {
      const int node = trajectory_[i];
      if (node >= 0 && node < cvrp::kNumNodes) {
        render::DrawLine(width, height, map_x(coordinates_[previous * 2]),
                         map_y(coordinates_[previous * 2 + 1]),
                         map_x(coordinates_[node * 2]),
                         map_y(coordinates_[node * 2 + 1]), {120, 120, 120},
                         rgb, 2);
        previous = node;
      }
    }
    for (int node = 0; node < cvrp::kNumNodes; ++node) {
      const int x = map_x(coordinates_[node * 2]);
      const int y = map_y(coordinates_[node * 2 + 1]);
      if (node == 0) {
        render::FillRect(width, height, x - 4, y - 4, x + 5, y + 5,
                         render::kBlack, rgb);
      } else {
        render::FillCircle(
            width, height, x, y, node == position_ ? 4 : 3,
            unvisited_[node] ? render::kBlack : render::Palette(2), rgb);
      }
    }
  }

  void Reset() override {
    if (use_configured_coordinates_) {
      coordinates_ = configured_coordinates_;
    }
    for (int node = 0; node < cvrp::kNumNodes; ++node) {
      if (!use_configured_coordinates_) {
        coordinates_[node * 2] = static_cast<float>(node) / 20.0f;
        coordinates_[node * 2 + 1] = 0.0f;
      }
      if (spec_.config["cvrp_demands"_].empty()) {
        demands_[node] = node == 0 ? 0.0f : 0.05f;
      } else {
        demands_[node] = configured_demands_[node];
      }
      unvisited_[node] = node != 0;
    }
    trajectory_.fill(0);
    position_ = 0;
    trajectory_size_ = 1;
    capacity_ = 1.0f;
    step_count_ = 0;
    done_ = false;
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    const int node =
        std::clamp(static_cast<int>(action["action"_]), 0, cvrp::kNumNodes - 1);
    const bool valid = IsActionValid(node);
    float reward = 0.0f;
    if (valid) {
      reward = -Distance(position_, node);
      position_ = node;
      if (node == 0) {
        capacity_ = 1.0f;
      } else {
        capacity_ -= demands_[node];
        unvisited_[node] = false;
      }
      if (trajectory_size_ < cvrp::kTrajectoryLength) {
        trajectory_[trajectory_size_++] = node;
      }
    } else {
      reward = -1.0f;
    }
    ++step_count_;
    done_ = !valid || AllVisited() || step_count_ >= cvrp::kTimeLimit;
    if (use_replay_ && step_count_ <= cvrp::kTimeLimit) {
      reward = replay_rewards_[step_count_ - 1];
      done_ = replay_done_[step_count_ - 1];
    }
    WriteState(reward);
  }

 private:
  bool IsActionValid(int node) const {
    if (node == 0) {
      return position_ != 0;
    }
    return unvisited_[node] && demands_[node] <= capacity_;
  }

  float Distance(int from, int to) const {
    if (use_configured_distances_) {
      return configured_distances_[from * cvrp::kNumNodes + to];
    }
    return cvrp::Distance(coordinates_, from, to);
  }

  bool AllVisited() const {
    return std::none_of(unvisited_.begin() + 1, unvisited_.end(),
                        [](bool value) { return value; });
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int node = 0; node < cvrp::kNumNodes; ++node) {
      state["obs:coordinates"_](node, 0) = coordinates_[node * 2];
      state["obs:coordinates"_](node, 1) = coordinates_[node * 2 + 1];
      state["obs:demands"_][node] = demands_[node];
      state["obs:unvisited_nodes"_][node] =
          node == 0 ? position_ != 0 : unvisited_[node];
      state["obs:action_mask"_][node] = IsActionValid(node);
    }
    for (int i = 0; i < cvrp::kTrajectoryLength; ++i) {
      state["obs:trajectory"_][i] = trajectory_[i];
    }
    state["obs:position"_] = position_;
    state["obs:capacity"_] = capacity_;
    state["reward"_] = reward;
  }
};

using CVRPEnvPool = AsyncEnvPool<CVRPEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_CVRP_ENV_H_
