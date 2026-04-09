// Copyright 2026 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ENVPOOL_HIGHWAY_OFFICIAL_VEHICLE_H_
#define ENVPOOL_HIGHWAY_OFFICIAL_VEHICLE_H_

#include <array>
#include <optional>
#include <string>

#include "envpool/highway/official_lane.h"
#include "envpool/highway/official_road.h"

namespace highway {
namespace official {

constexpr double kVehicleLength = 5.0;
constexpr double kVehicleWidth = 2.0;
constexpr double kVehicleMaxSpeed = 40.0;
constexpr double kVehicleMinSpeed = -40.0;

enum class VehicleKind : int {
  kVehicle = 0,
  kControlled = 1,
  kMDP = 2,
  kIDM = 3,
};

enum class MetaAction : int {
  kLaneLeft = 0,
  kIdle = 1,
  kLaneRight = 2,
  kFaster = 3,
  kSlower = 4,
};

struct LowLevelAction {
  double steering{0.0};
  double acceleration{0.0};
};

struct Vehicle {
  VehicleKind kind{VehicleKind::kVehicle};
  Vec2 position{};
  double heading{0.0};
  double speed{0.0};
  LowLevelAction action{};
  Vec2 impact{};
  bool has_impact{false};
  bool crashed{false};
  bool check_collisions{true};
  bool collidable{true};
  bool solid{true};
  LaneIndex lane_index{};
  LaneIndex target_lane_index{};
  double target_speed{0.0};
  std::array<double, 3> target_speeds{20.0, 25.0, 30.0};
  int speed_index{0};
  Route route;
  bool enable_lane_change{true};
  double timer{0.0};
  double idm_delta{4.0};
  bool has_goal{false};
  Vec2 goal_position{};
  double goal_heading{0.0};
  double goal_speed{0.0};
  double lateral_speed{0.0};
  double yaw_rate{0.0};

  [[nodiscard]] Vec2 Direction() const;
  [[nodiscard]] Vec2 Velocity() const;
  [[nodiscard]] LaneCoordinates LaneOffset(const RoadNetwork& network) const;
  void UpdateLane(const RoadNetwork& network);
  void Act(LowLevelAction low_level_action);
  void Step(double dt, const RoadNetwork& network);
};

Vehicle MakeVehicle(const RoadNetwork& network, Vec2 position,
                    double heading = 0.0, double speed = 0.0);
Vehicle MakeVehicleOnLane(const RoadNetwork& network,
                          const LaneIndex& lane_index, double longitudinal,
                          std::optional<double> speed = std::nullopt);
Vehicle MakeControlledVehicle(
    const RoadNetwork& network, Vec2 position, double heading = 0.0,
    double speed = 0.0,
    std::optional<LaneIndex> target_lane_index = std::nullopt,
    std::optional<double> target_speed = std::nullopt, Route route = {});
Vehicle MakeMDPVehicle(
    const RoadNetwork& network, Vec2 position, double heading = 0.0,
    double speed = 0.0,
    std::optional<LaneIndex> target_lane_index = std::nullopt,
    std::optional<double> target_speed = std::nullopt,
    std::array<double, 3> target_speeds = {20.0, 25.0, 30.0}, Route route = {});

void PlanRouteTo(Vehicle* vehicle, const RoadNetwork& network,
                 const std::string& destination);
void ActControlled(Vehicle* vehicle, const RoadNetwork& network,
                   std::optional<MetaAction> action = std::nullopt);
void ActMDP(Vehicle* vehicle, const RoadNetwork& network,
            std::optional<MetaAction> action = std::nullopt);
void FollowRoad(Vehicle* vehicle, const RoadNetwork& network);

[[nodiscard]] double SteeringControl(const Vehicle& vehicle,
                                     const RoadNetwork& network,
                                     const LaneIndex& target_lane_index);
[[nodiscard]] double SpeedControl(const Vehicle& vehicle, double target_speed);
[[nodiscard]] int SpeedToIndex(const Vehicle& vehicle, double speed);
[[nodiscard]] double IndexToSpeed(const Vehicle& vehicle, int index);

}  // namespace official
}  // namespace highway

#endif  // ENVPOOL_HIGHWAY_OFFICIAL_VEHICLE_H_
