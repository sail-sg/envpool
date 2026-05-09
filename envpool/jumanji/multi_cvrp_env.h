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

#ifndef ENVPOOL_JUMANJI_MULTI_CVRP_ENV_H_
#define ENVPOOL_JUMANJI_MULTI_CVRP_ENV_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/parse_utils.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace multicvrp {

constexpr int kNumNodes = 21;
constexpr int kNumVehicles = 2;
constexpr int kVehicleCapacity = 60;
constexpr int kTimeLimit = 40;
constexpr int kActionMaskSize = kNumVehicles * kNumNodes;

using NodeCoordinates = std::array<float, kNumNodes * 2>;

inline NodeCoordinates ParseNodeCoordinates(const std::string& text) {
  NodeCoordinates coordinates{};
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

inline float Distance(float from_x, float from_y, float to_x, float to_y) {
  const float dx = from_x - to_x;
  const float dy = from_y - to_y;
  return std::sqrt(dx * dx + dy * dy);
}

}  // namespace multicvrp

class MultiCVRPEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("multi_cvrp_node_coordinates"_.Bind(std::string("")),
                    "multi_cvrp_node_demands"_.Bind(std::string("")),
                    "multi_cvrp_windows_start"_.Bind(std::string("")),
                    "multi_cvrp_windows_end"_.Bind(std::string("")),
                    "multi_cvrp_coeffs_early"_.Bind(std::string("")),
                    "multi_cvrp_coeffs_late"_.Bind(std::string("")),
                    "multi_cvrp_vehicle_local_times"_.Bind(std::string("")),
                    "multi_cvrp_vehicle_capacities"_.Bind(std::string("")),
                    "multi_cvrp_action_mask"_.Bind(std::string("")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict(
        "obs:nodes.coordinates"_.Bind(Spec<float>({21, 2}, {0.0f, 10.0f})),
        "obs:nodes.demands"_.Bind(Spec<std::int16_t>({21}, {0, 60})),
        "obs:windows.start"_.Bind(Spec<float>({21}, {0.0f, 30.0f})),
        "obs:windows.end"_.Bind(Spec<float>({21}, {0.0f, 30.0f})),
        "obs:coeffs.early"_.Bind(Spec<float>({21}, {0.0f, 1.0f})),
        "obs:coeffs.late"_.Bind(Spec<float>({21}, {0.0f, 1.0f})),
        "obs:vehicles.coordinates"_.Bind(Spec<float>({2, 2}, {0.0f, 10.0f})),
        "obs:vehicles.local_times"_.Bind(Spec<float>({2}, {0.0f, 565.6854f})),
        "obs:vehicles.capacities"_.Bind(Spec<std::int16_t>({2}, {0, 60})),
        "obs:action_mask"_.Bind(Spec<bool>({2, 21}, {false, true})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<std::int16_t>({-1, 2}, {0, 20})));
  }
};

using MultiCVRPEnvSpec = EnvSpec<MultiCVRPEnvFns>;

class MultiCVRPEnv : public Env<MultiCVRPEnvSpec>, public RenderableEnv {
 protected:
  multicvrp::NodeCoordinates node_coordinates_{};
  multicvrp::NodeCoordinates configured_node_coordinates_{};
  std::array<std::int16_t, multicvrp::kNumNodes> demands_{};
  std::array<std::int16_t, multicvrp::kNumNodes> configured_demands_{};
  std::array<float, multicvrp::kNumNodes> windows_start_{};
  std::array<float, multicvrp::kNumNodes> windows_end_{};
  std::array<float, multicvrp::kNumNodes> coeffs_early_{};
  std::array<float, multicvrp::kNumNodes> coeffs_late_{};
  std::array<float, multicvrp::kNumNodes> configured_windows_start_{};
  std::array<float, multicvrp::kNumNodes> configured_windows_end_{};
  std::array<float, multicvrp::kNumNodes> configured_coeffs_early_{};
  std::array<float, multicvrp::kNumNodes> configured_coeffs_late_{};
  std::array<float, multicvrp::kNumVehicles * 2> vehicle_coordinates_{};
  std::array<float, multicvrp::kNumVehicles> local_times_{};
  std::array<std::int16_t, multicvrp::kNumVehicles> capacities_{};
  std::array<float, multicvrp::kNumVehicles> configured_local_times_{};
  std::array<std::int16_t, multicvrp::kNumVehicles> configured_capacities_{};
  std::array<bool, multicvrp::kActionMaskSize> configured_action_mask_{};
  bool use_configured_node_coordinates_;
  bool use_configured_state_;
  int step_count_{0};
  bool done_{true};

 public:
  using Spec = MultiCVRPEnvSpec;
  using Action = typename Env<MultiCVRPEnvSpec>::Action;

  MultiCVRPEnv(const Spec& spec, int env_id)
      : Env<MultiCVRPEnvSpec>(spec, env_id),
        configured_node_coordinates_(multicvrp::ParseNodeCoordinates(
            spec.config["multi_cvrp_node_coordinates"_])),
        configured_demands_(parse::CsvArray<std::int16_t, multicvrp::kNumNodes>(
            spec.config["multi_cvrp_node_demands"_])),
        configured_windows_start_(parse::CsvArray<float, multicvrp::kNumNodes>(
            spec.config["multi_cvrp_windows_start"_])),
        configured_windows_end_(parse::CsvArray<float, multicvrp::kNumNodes>(
            spec.config["multi_cvrp_windows_end"_])),
        configured_coeffs_early_(parse::CsvArray<float, multicvrp::kNumNodes>(
            spec.config["multi_cvrp_coeffs_early"_])),
        configured_coeffs_late_(parse::CsvArray<float, multicvrp::kNumNodes>(
            spec.config["multi_cvrp_coeffs_late"_])),
        configured_local_times_(parse::CsvArray<float, multicvrp::kNumVehicles>(
            spec.config["multi_cvrp_vehicle_local_times"_])),
        configured_capacities_(
            parse::CsvArray<std::int16_t, multicvrp::kNumVehicles>(
                spec.config["multi_cvrp_vehicle_capacities"_],
                multicvrp::kVehicleCapacity)),
        configured_action_mask_(
            parse::CsvArray<bool, multicvrp::kActionMaskSize>(
                spec.config["multi_cvrp_action_mask"_])),
        use_configured_node_coordinates_(
            !spec.config["multi_cvrp_node_coordinates"_].empty()),
        use_configured_state_(
            !spec.config["multi_cvrp_node_demands"_].empty()) {}

  bool IsDone() override { return done_; }

  int CurrentMaxEpisodeSteps() const override {
    return multicvrp::kTimeLimit + 1;
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, render::kWhite, rgb);
    auto map_x = [width](float x) {
      return std::clamp(static_cast<int>(x / 10.0f * (width - 24)) + 12, 0,
                        width - 1);
    };
    auto map_y = [height](float y) {
      return std::clamp(static_cast<int>(y / 10.0f * (height - 24)) + 12, 0,
                        height - 1);
    };
    for (int node = 0; node < multicvrp::kNumNodes; ++node) {
      const int x = map_x(node_coordinates_[node * 2]);
      const int y = map_y(node_coordinates_[node * 2 + 1]);
      if (node == 0) {
        render::FillRect(width, height, x - 4, y - 4, x + 5, y + 5,
                         render::kBlack, rgb);
      } else {
        render::FillCircle(
            width, height, x, y, demands_[node] > 0 ? 3 : 2,
            demands_[node] > 0 ? render::kBlack : render::Palette(2), rgb);
      }
    }
    for (int vehicle = 0; vehicle < multicvrp::kNumVehicles; ++vehicle) {
      render::FillCircle(width, height,
                         map_x(vehicle_coordinates_[vehicle * 2]),
                         map_y(vehicle_coordinates_[vehicle * 2 + 1]), 5,
                         render::Palette(vehicle), rgb);
    }
  }

  void Reset() override {
    if (use_configured_node_coordinates_) {
      node_coordinates_ = configured_node_coordinates_;
    }
    for (int node = 0; node < multicvrp::kNumNodes; ++node) {
      if (!use_configured_node_coordinates_) {
        node_coordinates_[node * 2] = static_cast<float>(node) / 2.0f;
        node_coordinates_[node * 2 + 1] = 0.0f;
      }
      demands_[node] = use_configured_state_ ? configured_demands_[node]
                                             : (node == 0 ? 0 : 10);
      windows_start_[node] =
          use_configured_state_ ? configured_windows_start_[node] : 0.0f;
      windows_end_[node] =
          use_configured_state_ ? configured_windows_end_[node] : 30.0f;
      coeffs_early_[node] =
          use_configured_state_ ? configured_coeffs_early_[node] : 0.0f;
      coeffs_late_[node] =
          use_configured_state_ ? configured_coeffs_late_[node] : 0.0f;
    }
    for (int vehicle = 0; vehicle < multicvrp::kNumVehicles; ++vehicle) {
      vehicle_coordinates_[vehicle * 2] =
          use_configured_node_coordinates_ ? node_coordinates_[0] : 0.0f;
      vehicle_coordinates_[vehicle * 2 + 1] =
          use_configured_node_coordinates_ ? node_coordinates_[1] : 0.0f;
      local_times_[vehicle] =
          use_configured_state_ ? configured_local_times_[vehicle] : 0.0f;
      capacities_[vehicle] = use_configured_state_
                                 ? configured_capacities_[vehicle]
                                 : multicvrp::kVehicleCapacity;
    }
    step_count_ = 0;
    done_ = false;
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    float reward = 0.0f;
    bool valid = true;
    for (int vehicle = 0; vehicle < multicvrp::kNumVehicles; ++vehicle) {
      const int node =
          std::clamp(static_cast<int>(action["action"_](0, vehicle)), 0,
                     multicvrp::kNumNodes - 1);
      if (!IsActionValid(vehicle, node)) {
        valid = false;
        continue;
      }
      const float next_x = node_coordinates_[node * 2];
      const float next_y = node_coordinates_[node * 2 + 1];
      const float distance = multicvrp::Distance(
          vehicle_coordinates_[vehicle * 2],
          vehicle_coordinates_[vehicle * 2 + 1], next_x, next_y);
      reward -= distance;
      local_times_[vehicle] += distance;
      vehicle_coordinates_[vehicle * 2] = next_x;
      vehicle_coordinates_[vehicle * 2 + 1] = next_y;
      if (node == 0) {
        capacities_[vehicle] = multicvrp::kVehicleCapacity;
      } else {
        capacities_[vehicle] -= demands_[node];
        demands_[node] = 0;
      }
    }
    ++step_count_;
    done_ = !valid || AllServed() || step_count_ >= multicvrp::kTimeLimit;
    WriteState(valid ? reward : -100.0f);
  }

 private:
  bool IsActionValid(int vehicle, int node) const {
    if (node == 0) {
      return true;
    }
    return demands_[node] > 0 && demands_[node] <= capacities_[vehicle];
  }

  bool AllServed() const {
    return std::all_of(demands_.begin() + 1, demands_.end(),
                       [](std::int16_t demand) { return demand == 0; });
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int node = 0; node < multicvrp::kNumNodes; ++node) {
      state["obs:nodes.coordinates"_](node, 0) = node_coordinates_[node * 2];
      state["obs:nodes.coordinates"_](node, 1) =
          node_coordinates_[node * 2 + 1];
      state["obs:nodes.demands"_][node] = demands_[node];
      state["obs:windows.start"_][node] = windows_start_[node];
      state["obs:windows.end"_][node] = windows_end_[node];
      state["obs:coeffs.early"_][node] = coeffs_early_[node];
      state["obs:coeffs.late"_][node] = coeffs_late_[node];
    }
    for (int vehicle = 0; vehicle < multicvrp::kNumVehicles; ++vehicle) {
      state["obs:vehicles.coordinates"_](vehicle, 0) =
          vehicle_coordinates_[vehicle * 2];
      state["obs:vehicles.coordinates"_](vehicle, 1) =
          vehicle_coordinates_[vehicle * 2 + 1];
      state["obs:vehicles.local_times"_][vehicle] = local_times_[vehicle];
      state["obs:vehicles.capacities"_][vehicle] = capacities_[vehicle];
      for (int node = 0; node < multicvrp::kNumNodes; ++node) {
        state["obs:action_mask"_](vehicle, node) =
            use_configured_state_ && step_count_ == 0
                ? configured_action_mask_[vehicle * multicvrp::kNumNodes + node]
                : IsActionValid(vehicle, node);
      }
    }
    state["reward"_] = reward;
  }
};

using MultiCVRPEnvPool = AsyncEnvPool<MultiCVRPEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_MULTI_CVRP_ENV_H_
