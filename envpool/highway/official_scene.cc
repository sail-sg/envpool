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

#include "envpool/highway/official_scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

namespace highway::official {
namespace {

constexpr double kIdmAccMax = 6.0;
constexpr double kIdmComfortAccMax = 3.0;
constexpr double kIdmComfortAccMin = -5.0;
constexpr double kIdmDistanceWanted = 5.0 + kVehicleLength;
constexpr double kIdmTimeWanted = 1.5;
constexpr double kLaneChangeMinAccGain = 0.2;
constexpr double kLaneChangeMaxBrakingImposed = 2.0;
constexpr double kLaneChangeDelay = 1.0;

double Clip(double value, double low, double high) {
  return std::clamp(value, low, high);
}

double NotZero(double value) {
  constexpr double k_eps = 1e-2;
  if (std::abs(value) > k_eps) {
    return value;
  }
  return value >= 0.0 ? k_eps : -k_eps;
}

bool DoEvery(double duration, double timer) { return duration < timer; }

double Distance(Vec2 lhs, Vec2 rhs) {
  const Vec2 delta = lhs - rhs;
  return Norm(delta);
}

struct CollisionBox {
  Vec2 center{};
  double heading{0.0};
  double length{kVehicleLength};
  double width{kVehicleWidth};
  double diagonal{std::sqrt(kVehicleLength * kVehicleLength +
                            kVehicleWidth * kVehicleWidth)};
};

CollisionBox ToCollisionBox(const Vehicle& vehicle) {
  return {vehicle.position, vehicle.heading, kVehicleLength, kVehicleWidth,
          std::sqrt(kVehicleLength * kVehicleLength +
                    kVehicleWidth * kVehicleWidth)};
}

CollisionBox ToCollisionBox(const RoadObject& object) {
  return {
      object.position, object.heading, object.length, object.width,
      std::sqrt(object.length * object.length + object.width * object.width)};
}

std::array<Vec2, 4> Corners(const CollisionBox& box) {
  const double cos_h = std::cos(box.heading);
  const double sin_h = std::sin(box.heading);
  const Vec2 longitudinal{cos_h * box.length / 2.0, sin_h * box.length / 2.0};
  const Vec2 lateral{-sin_h * box.width / 2.0, cos_h * box.width / 2.0};
  return {
      box.center - longitudinal - lateral, box.center - longitudinal + lateral,
      box.center + longitudinal + lateral, box.center + longitudinal - lateral};
}

std::pair<double, double> Project(const std::array<Vec2, 4>& polygon,
                                  Vec2 axis) {
  double low = Dot(polygon[0], axis);
  double high = low;
  for (int i = 1; i < 4; ++i) {
    const double value = Dot(polygon[i], axis);
    low = std::min(low, value);
    high = std::max(high, value);
  }
  return {low, high};
}

double IntervalDistance(double a_low, double a_high, double b_low,
                        double b_high) {
  return a_low < b_low ? b_low - a_high : a_low - b_high;
}

Vec2 BoxVelocity(const Vehicle& vehicle) { return vehicle.Velocity(); }

Vec2 BoxVelocity(const RoadObject& object) {
  return {object.speed * std::cos(object.heading),
          object.speed * std::sin(object.heading)};
}

struct CollisionResult {
  bool intersecting{false};
  bool will_intersect{false};
  Vec2 translation{};
};

CollisionResult CollidePolygons(const std::array<Vec2, 4>& a,
                                const std::array<Vec2, 4>& b,
                                Vec2 displacement_a, Vec2 displacement_b,
                                Vec2 center_a, Vec2 center_b) {
  bool intersecting = true;
  bool will_intersect = true;
  double min_distance = std::numeric_limits<double>::infinity();
  Vec2 translation_axis{};
  for (const std::array<Vec2, 4>* polygon : {&a, &b}) {
    for (int i = 0; i < 4; ++i) {
      const Vec2 edge = (*polygon)[(i + 1) % 4] - (*polygon)[i];
      Vec2 axis{-edge.y, edge.x};
      const double norm = Norm(axis);
      if (norm <= 0.0) {
        continue;
      }
      axis = (1.0 / norm) * axis;

      auto [a_low, a_high] = Project(a, axis);
      const auto [b_low, b_high] = Project(b, axis);
      if (IntervalDistance(a_low, a_high, b_low, b_high) > 0.0) {
        intersecting = false;
      }

      const double velocity_projection =
          Dot(axis, displacement_a - displacement_b);
      if (velocity_projection < 0.0) {
        a_low += velocity_projection;
      } else {
        a_high += velocity_projection;
      }

      const double distance = IntervalDistance(a_low, a_high, b_low, b_high);
      if (distance > 0.0) {
        will_intersect = false;
      }
      if (!intersecting && !will_intersect) {
        return {};
      }
      if (std::abs(distance) < min_distance) {
        min_distance = std::abs(distance);
        const Vec2 center_delta = center_a - center_b;
        translation_axis = Dot(center_delta, axis) > 0.0 ? axis : -1.0 * axis;
      }
    }
  }
  return {intersecting, will_intersect,
          will_intersect ? min_distance * translation_axis : Vec2{}};
}

CollisionResult BoxesCollide(const CollisionBox& a, const CollisionBox& b,
                             Vec2 displacement_a, Vec2 displacement_b) {
  if (Distance(a.center, b.center) >
      (a.diagonal + b.diagonal) / 2.0 + Norm(displacement_a)) {
    return {};
  }
  return CollidePolygons(Corners(a), Corners(b), displacement_a, displacement_b,
                         a.center, b.center);
}

template <typename OtherT>
CollisionResult CheckBoxCollision(const Vehicle& a, const OtherT& b,
                                  double dt) {
  return BoxesCollide(ToCollisionBox(a), ToCollisionBox(b), BoxVelocity(a) * dt,
                      BoxVelocity(b) * dt);
}

void SetImpact(Vehicle* vehicle, Vec2 impact) {
  vehicle->impact = impact;
  vehicle->has_impact = true;
}

void ApplyCollision(Vehicle* a, Vehicle* b, const CollisionResult& collision) {
  if (collision.will_intersect && a->solid && b->solid) {
    SetImpact(a, 0.5 * collision.translation);
    SetImpact(b, -0.5 * collision.translation);
  }
  if (collision.intersecting && a->solid && b->solid) {
    a->crashed = true;
    b->crashed = true;
  }
}

void ApplyCollision(Vehicle* a, RoadObject* b,
                    const CollisionResult& collision) {
  if (collision.will_intersect && a->solid && b->solid) {
    SetImpact(a, collision.translation);
  }
  if (!collision.intersecting) {
    return;
  }
  if (a->solid && b->solid) {
    a->crashed = true;
    b->crashed = true;
  }
  if (!b->solid) {
    b->hit = true;
  }
}

const Vehicle* VehiclePtr(const Road& road, int index) {
  if (index < 0) {
    return nullptr;
  }
  return &road.vehicles[index];
}

}  // namespace

std::pair<int, int> Road::NeighbourVehicles(
    const Vehicle& vehicle, std::optional<LaneIndex> lane_index) const {
  const LaneIndex query_index = lane_index.value_or(vehicle.lane_index);
  const Lane& lane = network.GetLane(query_index);
  const double s = lane.LocalCoordinates(vehicle.position).longitudinal;
  double s_front = std::numeric_limits<double>::infinity();
  double s_rear = -std::numeric_limits<double>::infinity();
  int front = -1;
  int rear = -1;
  for (int i = 0; static_cast<std::size_t>(i) < vehicles.size(); ++i) {
    const Vehicle& other = vehicles[i];
    if (&other == &vehicle) {
      continue;
    }
    const LaneCoordinates other_local = lane.LocalCoordinates(other.position);
    if (!lane.OnLane(other.position, 1.0)) {
      continue;
    }
    const double s_other = other_local.longitudinal;
    if (s <= s_other && s_other <= s_front) {
      s_front = s_other;
      front = i;
    }
    if (s_other < s && s_rear <= s_other) {
      s_rear = s_other;
      rear = i;
    }
  }
  return {front, rear};
}

void Road::Act() {
  for (int i = 0; static_cast<std::size_t>(i) < vehicles.size(); ++i) {
    Vehicle& vehicle = vehicles[i];
    if (vehicle.kind == VehicleKind::kIDM) {
      ActIDM(this, i);
    } else if (vehicle.kind == VehicleKind::kMDP) {
      ActMDP(&vehicle, network);
    } else if (vehicle.kind == VehicleKind::kControlled) {
      ActControlled(&vehicle, network);
    }
  }
}

void Road::Step(double dt) {
  for (Vehicle& vehicle : vehicles) {
    vehicle.Step(dt, network);
  }
  CheckCollisions(dt);
}

void Road::CheckCollisions(double dt) {
  for (int i = 0; static_cast<std::size_t>(i) < vehicles.size(); ++i) {
    Vehicle& a = vehicles[i];
    for (int j = i + 1; static_cast<std::size_t>(j) < vehicles.size(); ++j) {
      Vehicle& b = vehicles[j];
      if (!(a.check_collisions || b.check_collisions)) {
        continue;
      }
      if (a.collidable && b.collidable) {
        ApplyCollision(&a, &b, CheckBoxCollision(a, b, dt));
      }
    }
    for (RoadObject& b : objects) {
      if (!(a.check_collisions || b.check_collisions)) {
        continue;
      }
      if (!a.collidable || !b.collidable) {
        continue;
      }
      ApplyCollision(&a, &b, CheckBoxCollision(a, b, dt));
    }
  }
}

Vehicle MakeIDMVehicle(const RoadNetwork& network, Vec2 position,
                       double heading, double speed,
                       std::optional<LaneIndex> target_lane_index,
                       std::optional<double> target_speed, Route route,
                       bool enable_lane_change, std::optional<double> timer) {
  Vehicle vehicle = MakeControlledVehicle(network, position, heading, speed,
                                          std::move(target_lane_index),
                                          target_speed, std::move(route));
  vehicle.kind = VehicleKind::kIDM;
  vehicle.enable_lane_change = enable_lane_change;
  vehicle.timer = timer.value_or(std::fmod(
      (vehicle.position.x + vehicle.position.y) * kPi, kLaneChangeDelay));
  return vehicle;
}

Vehicle MakeIDMVehicleFrom(const Vehicle& vehicle) {
  Vehicle copy = vehicle;
  copy.kind = VehicleKind::kIDM;
  return copy;
}

void ActIDM(Road* road, int vehicle_index) {
  Vehicle& vehicle = road->vehicles[vehicle_index];
  if (vehicle.crashed) {
    return;
  }
  FollowRoad(&vehicle, road->network);
  if (vehicle.enable_lane_change) {
    ChangeLanePolicy(road, vehicle_index);
  }
  LowLevelAction action;
  action.steering =
      SteeringControl(vehicle, road->network, vehicle.target_lane_index);
  const auto [front_index, rear_index] =
      road->NeighbourVehicles(vehicle, vehicle.lane_index);
  (void)rear_index;
  action.acceleration =
      IDMAcceleration(*road, &vehicle, VehiclePtr(*road, front_index));
  if (!SameRoad(vehicle.lane_index, vehicle.target_lane_index, true)) {
    const auto [target_front_index, target_rear_index] =
        road->NeighbourVehicles(vehicle, vehicle.target_lane_index);
    (void)target_rear_index;
    action.acceleration =
        std::min(action.acceleration,
                 IDMAcceleration(*road, &vehicle,
                                 VehiclePtr(*road, target_front_index)));
  }
  action.acceleration = Clip(action.acceleration, -kIdmAccMax, kIdmAccMax);
  vehicle.Act(action);
}

void ChangeLanePolicy(Road* road, int vehicle_index) {
  Vehicle& vehicle = road->vehicles[vehicle_index];
  if (!SameRoad(vehicle.lane_index, vehicle.target_lane_index, true)) {
    if (SameRoad(vehicle.lane_index, vehicle.target_lane_index)) {
      for (const Vehicle& other : road->vehicles) {
        if (&other == &vehicle ||
            SameRoad(other.lane_index, vehicle.target_lane_index, true) ||
            !SameRoad(other.target_lane_index, vehicle.target_lane_index,
                      true)) {
          continue;
        }
        const double d = LaneDistanceTo(road->network, vehicle, other);
        if (0.0 < d && d < DesiredGap(vehicle, other)) {
          vehicle.target_lane_index = vehicle.lane_index;
          break;
        }
      }
    }
    return;
  }
  if (!DoEvery(kLaneChangeDelay, vehicle.timer)) {
    return;
  }
  vehicle.timer = 0.0;
  for (const LaneIndex& lane_index :
       road->network.SideLanes(vehicle.lane_index)) {
    if (!road->network.GetLane(lane_index).IsReachableFrom(vehicle.position) ||
        std::abs(vehicle.speed) < 1.0) {
      continue;
    }
    if (Mobil(*road, vehicle_index, lane_index)) {
      vehicle.target_lane_index = lane_index;
    }
  }
}

double IDMAcceleration(const Road& road, const Vehicle* ego,
                       const Vehicle* front) {
  if (ego == nullptr) {
    return 0.0;
  }
  double ego_target_speed = ego->target_speed;
  const Lane& lane = road.network.GetLane(ego->lane_index);
  ego_target_speed = Clip(ego_target_speed, 0.0, lane.SpeedLimit());
  double acceleration = kIdmComfortAccMax *
                        (1.0 - std::pow(std::max(ego->speed, 0.0) /
                                            std::abs(NotZero(ego_target_speed)),
                                        ego->idm_delta));
  if (front != nullptr) {
    const double d = LaneDistanceTo(road.network, *ego, *front);
    acceleration -= kIdmComfortAccMax *
                    std::pow(DesiredGap(*ego, *front) / NotZero(d), 2.0);
  }
  return acceleration;
}

double DesiredGap(const Vehicle& ego, const Vehicle& front) {
  const Vec2 dv = ego.Velocity() - front.Velocity();
  const double projected_dv = Dot(dv, ego.Direction());
  const double ab = -kIdmComfortAccMax * kIdmComfortAccMin;
  return kIdmDistanceWanted + ego.speed * kIdmTimeWanted +
         ego.speed * projected_dv / (2.0 * std::sqrt(ab));
}

bool Mobil(const Road& road, int vehicle_index, const LaneIndex& lane_index) {
  const Vehicle& self = road.vehicles[vehicle_index];
  const auto [new_preceding_index, new_following_index] =
      road.NeighbourVehicles(self, lane_index);
  const Vehicle* new_preceding = VehiclePtr(road, new_preceding_index);
  const Vehicle* new_following = VehiclePtr(road, new_following_index);
  const double new_following_pred_a =
      IDMAcceleration(road, new_following, &self);
  if (new_following_pred_a < -kLaneChangeMaxBrakingImposed) {
    return false;
  }
  const double self_pred_a = IDMAcceleration(road, &self, new_preceding);
  const auto [old_preceding_index, old_following_index] =
      road.NeighbourVehicles(self, self.lane_index);
  const Vehicle* old_preceding = VehiclePtr(road, old_preceding_index);
  const Vehicle* old_following = VehiclePtr(road, old_following_index);
  const double self_a = IDMAcceleration(road, &self, old_preceding);
  const double old_following_a = IDMAcceleration(road, old_following, &self);
  const double old_following_pred_a =
      IDMAcceleration(road, old_following, old_preceding);
  const double new_following_a =
      IDMAcceleration(road, new_following, new_preceding);
  const double politeness = 0.0;
  const double jerk = self_pred_a - self_a +
                      politeness * (new_following_pred_a - new_following_a +
                                    old_following_pred_a - old_following_a);
  return jerk >= kLaneChangeMinAccGain;
}

double LaneDistanceTo(const RoadNetwork& network, const Vehicle& self,
                      const Vehicle& other) {
  const Lane& lane = network.GetLane(self.lane_index);
  return lane.LocalCoordinates(other.position).longitudinal -
         lane.LocalCoordinates(self.position).longitudinal;
}

}  // namespace highway::official
