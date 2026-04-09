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

#include "envpool/highway/official_vehicle.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace highway::official {

namespace {

constexpr double kTauAcc = 0.6;
constexpr double kTauHeading = 0.2;
constexpr double kTauLateral = 0.6;
constexpr double kTauPursuit = 0.5 * kTauHeading;
constexpr double kKpA = 1.0 / kTauAcc;
constexpr double kKpHeading = 1.0 / kTauHeading;
constexpr double kKpLateral = 1.0 / kTauLateral;
constexpr double kMaxSteeringAngle = kPi / 3.0;

double NotZero(double value) {
  constexpr double kEps = 1e-2;
  if (std::abs(value) > kEps) {
    return value;
  }
  return value >= 0.0 ? kEps : -kEps;
}

double Clip(double value, double low, double high) {
  return std::clamp(value, low, high);
}

int ClipInt(int value, int low, int high) {
  return std::clamp(value, low, high);
}

}  // namespace

Vec2 Vehicle::Direction() const {
  return {std::cos(heading), std::sin(heading)};
}

Vec2 Vehicle::Velocity() const { return speed * Direction(); }

LaneCoordinates Vehicle::LaneOffset(const RoadNetwork& network) const {
  const Lane& lane = network.GetLane(lane_index);
  return lane.LocalCoordinates(position);
}

void Vehicle::UpdateLane(const RoadNetwork& network) {
  lane_index = network.GetClosestLaneIndex(position, heading);
}

void Vehicle::Act(LowLevelAction low_level_action) {
  action = low_level_action;
}

void Vehicle::Step(double dt, const RoadNetwork& network) {
  if (crashed) {
    action.steering = 0.0;
    action.acceleration = -speed;
  }
  if (speed > kVehicleMaxSpeed) {
    action.acceleration =
        std::min(action.acceleration, kVehicleMaxSpeed - speed);
  } else if (speed < kVehicleMinSpeed) {
    action.acceleration =
        std::max(action.acceleration, kVehicleMinSpeed - speed);
  }
  const double beta = std::atan(1.0 / 2.0 * std::tan(action.steering));
  const double beta_heading = heading + beta;
  const double velocity_x = speed * std::cos(beta_heading);
  const double velocity_y = speed * std::sin(beta_heading);
  const double delta_x = velocity_x * dt;
  const double delta_y = velocity_y * dt;
  position.x += delta_x;
  position.y += delta_y;
  if (has_impact) {
    position = position + impact;
    crashed = true;
    impact = {};
    has_impact = false;
  }
  const double heading_rate = speed * std::sin(beta) / (kVehicleLength / 2.0);
  const double heading_delta = heading_rate * dt;
  const double speed_delta = action.acceleration * dt;
  heading += heading_delta;
  speed += speed_delta;
  if (kind == VehicleKind::kIDM) {
    timer += dt;
  }
  UpdateLane(network);
}

Vehicle MakeVehicle(const RoadNetwork& network, Vec2 position, double heading,
                    double speed) {
  Vehicle vehicle;
  vehicle.position = position;
  vehicle.heading = heading;
  vehicle.speed = speed;
  vehicle.UpdateLane(network);
  vehicle.target_lane_index = vehicle.lane_index;
  vehicle.target_speed = vehicle.speed;
  return vehicle;
}

Vehicle MakeVehicleOnLane(const RoadNetwork& network,
                          const LaneIndex& lane_index, double longitudinal,
                          std::optional<double> speed) {
  const Lane& lane = network.GetLane(lane_index);
  const double resolved_speed = speed.has_value() ? *speed : lane.SpeedLimit();
  Vehicle vehicle = MakeVehicle(network, lane.Position(longitudinal, 0.0),
                                lane.HeadingAt(longitudinal), resolved_speed);
  vehicle.lane_index = lane_index;
  vehicle.target_lane_index = lane_index;
  return vehicle;
}

Vehicle MakeControlledVehicle(const RoadNetwork& network, Vec2 position,
                              double heading, double speed,
                              std::optional<LaneIndex> target_lane_index,
                              std::optional<double> target_speed, Route route) {
  Vehicle vehicle = MakeVehicle(network, position, heading, speed);
  vehicle.kind = VehicleKind::kControlled;
  vehicle.target_lane_index =
      target_lane_index.has_value() ? *target_lane_index : vehicle.lane_index;
  vehicle.target_speed = target_speed.has_value() ? *target_speed : speed;
  vehicle.route = std::move(route);
  return vehicle;
}

Vehicle MakeMDPVehicle(const RoadNetwork& network, Vec2 position,
                       double heading, double speed,
                       std::optional<LaneIndex> target_lane_index,
                       std::optional<double> target_speed,
                       std::array<double, 3> target_speeds, Route route) {
  Vehicle vehicle =
      MakeControlledVehicle(network, position, heading, speed,
                            target_lane_index, target_speed, std::move(route));
  vehicle.kind = VehicleKind::kMDP;
  vehicle.target_speeds = target_speeds;
  vehicle.speed_index = SpeedToIndex(vehicle, vehicle.target_speed);
  vehicle.target_speed = IndexToSpeed(vehicle, vehicle.speed_index);
  return vehicle;
}

void PlanRouteTo(Vehicle* vehicle, const RoadNetwork& network,
                 const std::string& destination) {
  const std::vector<std::string> path =
      network.ShortestPath(vehicle->lane_index.to, destination);
  vehicle->route.clear();
  if (path.empty()) {
    vehicle->route.push_back(vehicle->lane_index);
    return;
  }
  vehicle->route.push_back(vehicle->lane_index);
  for (std::size_t i = 0; i + 1 < path.size(); ++i) {
    vehicle->route.push_back({path[i], path[i + 1], kUnknownLaneId});
  }
}

void FollowRoad(Vehicle* vehicle, const RoadNetwork& network) {
  if (network.GetLane(vehicle->target_lane_index).AfterEnd(vehicle->position)) {
    vehicle->target_lane_index = network.NextLane(
        vehicle->target_lane_index, &vehicle->route, vehicle->position);
  }
}

void ActControlled(Vehicle* vehicle, const RoadNetwork& network,
                   std::optional<MetaAction> action) {
  FollowRoad(vehicle, network);
  if (action == MetaAction::kFaster) {
    vehicle->target_speed += 5.0;
  } else if (action == MetaAction::kSlower) {
    vehicle->target_speed -= 5.0;
  } else if (action == MetaAction::kLaneRight ||
             action == MetaAction::kLaneLeft) {
    const std::vector<LaneIndex> road_lanes =
        network.AllSideLanes(vehicle->target_lane_index);
    const int lane_delta = action == MetaAction::kLaneRight ? 1 : -1;
    const int next_id = ClipInt(vehicle->target_lane_index.id + lane_delta, 0,
                                static_cast<int>(road_lanes.size()) - 1);
    LaneIndex target = vehicle->target_lane_index;
    target.id = next_id;
    if (network.GetLane(target).IsReachableFrom(vehicle->position)) {
      vehicle->target_lane_index = target;
    }
  }
  LowLevelAction low_level;
  low_level.steering =
      SteeringControl(*vehicle, network, vehicle->target_lane_index);
  low_level.acceleration = SpeedControl(*vehicle, vehicle->target_speed);
  low_level.steering =
      Clip(low_level.steering, -kMaxSteeringAngle, kMaxSteeringAngle);
  vehicle->Act(low_level);
}

void ActMDP(Vehicle* vehicle, const RoadNetwork& network,
            std::optional<MetaAction> action) {
  if (action == MetaAction::kFaster || action == MetaAction::kSlower) {
    const int delta = action == MetaAction::kFaster ? 1 : -1;
    vehicle->speed_index = SpeedToIndex(*vehicle, vehicle->speed) + delta;
    vehicle->speed_index =
        ClipInt(vehicle->speed_index, 0, vehicle->target_speeds.size() - 1);
    vehicle->target_speed = IndexToSpeed(*vehicle, vehicle->speed_index);
    ActControlled(vehicle, network, std::nullopt);
    return;
  }
  ActControlled(vehicle, network, action);
}

double SteeringControl(const Vehicle& vehicle, const RoadNetwork& network,
                       const LaneIndex& target_lane_index) {
  const Lane& target_lane = network.GetLane(target_lane_index);
  const LaneCoordinates coordinates =
      target_lane.LocalCoordinates(vehicle.position);
  const double lane_next_s =
      coordinates.longitudinal + vehicle.speed * kTauPursuit;
  const double lane_future_heading = target_lane.HeadingAt(lane_next_s);
  const double lateral_speed_command = -kKpLateral * coordinates.lateral;
  const double heading_command = std::asin(
      Clip(lateral_speed_command / NotZero(vehicle.speed), -1.0, 1.0));
  const double heading_ref =
      lane_future_heading + Clip(heading_command, -kPi / 4.0, kPi / 4.0);
  const double heading_rate_command =
      kKpHeading * WrapToPi(heading_ref - vehicle.heading);
  const double slip_angle = std::asin(
      Clip(kVehicleLength / 2.0 / NotZero(vehicle.speed) * heading_rate_command,
           -1.0, 1.0));
  return Clip(std::atan(2.0 * std::tan(slip_angle)), -kMaxSteeringAngle,
              kMaxSteeringAngle);
}

double SpeedControl(const Vehicle& vehicle, double target_speed) {
  return kKpA * (target_speed - vehicle.speed);
}

int SpeedToIndex(const Vehicle& vehicle, double speed) {
  const double speed_range =
      vehicle.target_speeds.back() - vehicle.target_speeds.front();
  const double x = (speed - vehicle.target_speeds.front()) / speed_range;
  return ClipInt(
      static_cast<int>(std::llround(x * (vehicle.target_speeds.size() - 1))), 0,
      vehicle.target_speeds.size() - 1);
}

double IndexToSpeed(const Vehicle& vehicle, int index) {
  const int clipped =
      ClipInt(index, 0, static_cast<int>(vehicle.target_speeds.size()) - 1);
  return vehicle.target_speeds[clipped];
}

}  // namespace highway::official
