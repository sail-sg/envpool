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

#include "envpool/highway/official_observation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace highway::official {
namespace {

constexpr double kPerceptionDistance = 5.0 * kVehicleMaxSpeed;

double LMap(double value, double low, double high, double target_low,
            double target_high) {
  return target_low + (value - low) * (target_high - target_low) / (high - low);
}

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

Vec2 Destination(const Road& road, const Vehicle& vehicle) {
  if (vehicle.route.empty()) {
    return vehicle.position;
  }
  LaneIndex lane_index = vehicle.route.back();
  if (lane_index.id == kUnknownLaneId) {
    lane_index.id = 0;
  }
  const Lane& lane = road.network.GetLane(lane_index);
  return lane.Position(lane.Length(), 0.0);
}

Vec2 DestinationDirection(const Road& road, const Vehicle& vehicle) {
  const Vec2 destination = Destination(road, vehicle);
  const Vec2 delta = destination - vehicle.position;
  const double distance = Norm(delta);
  if (distance <= 0.0) {
    return {0.0, 0.0};
  }
  return (1.0 / distance) * delta;
}

struct CloseObjectRef {
  bool is_vehicle{true};
  int index{0};
};

double LaneDistanceToPosition(const RoadNetwork& network, const Vehicle& self,
                              Vec2 position) {
  const Lane& lane = network.GetLane(self.lane_index);
  return lane.LocalCoordinates(position).longitudinal -
         lane.LocalCoordinates(self.position).longitudinal;
}

std::vector<CloseObjectRef> CloseObjectRefs(const Road& road,
                                            const Vehicle& vehicle, int count,
                                            bool see_behind,
                                            bool include_obstacles) {
  std::vector<std::pair<double, CloseObjectRef>> candidates;
  candidates.reserve(road.vehicles.size() + road.objects.size());
  for (int i = 0; static_cast<std::size_t>(i) < road.vehicles.size(); ++i) {
    const Vehicle& other = road.vehicles[i];
    if (&other == &vehicle ||
        Norm(other.position - vehicle.position) >= kPerceptionDistance) {
      continue;
    }
    if (!see_behind &&
        LaneDistanceTo(road.network, vehicle, other) <= -2 * kVehicleLength) {
      continue;
    }
    const double lane_distance = LaneDistanceTo(road.network, vehicle, other);
    candidates.push_back({std::abs(lane_distance), {true, i}});
  }
  if (include_obstacles) {
    for (int i = 0; static_cast<std::size_t>(i) < road.objects.size(); ++i) {
      const RoadObject& object = road.objects[i];
      if (Norm(object.position - vehicle.position) >= kPerceptionDistance) {
        continue;
      }
      const double lane_distance =
          LaneDistanceToPosition(road.network, vehicle, object.position);
      if (!see_behind && lane_distance <= -2 * kVehicleLength) {
        continue;
      }
      candidates.push_back({std::abs(lane_distance), {false, i}});
    }
  }
  std::sort(
      candidates.begin(), candidates.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
  if (static_cast<int>(candidates.size()) > count) {
    candidates.resize(count);
  }
  std::vector<CloseObjectRef> refs;
  refs.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    refs.push_back(candidate.second);
  }
  return refs;
}

int TimeIndex(double time_to_collision, double time_quantization) {
  return static_cast<int>(time_to_collision / time_quantization);
}

int CeilTimeIndex(double time_to_collision, double time_quantization) {
  return static_cast<int>(std::ceil(time_to_collision / time_quantization));
}

void NormalizeRow(FeatureRow* row, const Road& road, const Vehicle& observer,
                  const KinematicObservationConfig& config) {
  const double side_lanes =
      road.network.AllSideLanes(observer.lane_index).size();
  const double y_min = config.y_min < config.y_max
                           ? config.y_min
                           : -kDefaultLaneWidth * side_lanes;
  const double y_max = config.y_min < config.y_max
                           ? config.y_max
                           : kDefaultLaneWidth * side_lanes;
  row->x = LMap(row->x, config.x_min, config.x_max, -1.0, 1.0);
  row->y = LMap(row->y, y_min, y_max, -1.0, 1.0);
  row->vx = LMap(row->vx, config.vx_min, config.vx_max, -1.0, 1.0);
  row->vy = LMap(row->vy, config.vy_min, config.vy_max, -1.0, 1.0);
  if (config.clip) {
    row->x = Clip(row->x, -1.0, 1.0);
    row->y = Clip(row->y, -1.0, 1.0);
    row->vx = Clip(row->vx, -1.0, 1.0);
    row->vy = Clip(row->vy, -1.0, 1.0);
  }
}

}  // namespace

FeatureRow VehicleToFeatureRow(const Road& road, const Vehicle& vehicle,
                               const Vehicle* origin_vehicle,
                               bool observe_intentions) {
  const Vec2 direction = vehicle.Direction();
  const Vec2 velocity = vehicle.Velocity();
  const Vec2 destination_direction = DestinationDirection(road, vehicle);
  const Lane& lane = road.network.GetLane(vehicle.lane_index);
  const LaneCoordinates offset = lane.LocalCoordinates(vehicle.position);
  FeatureRow row;
  row.x = vehicle.position.x;
  row.y = vehicle.position.y;
  row.vx = velocity.x;
  row.vy = velocity.y;
  row.heading = vehicle.heading;
  row.cos_h = direction.x;
  row.sin_h = direction.y;
  row.cos_d = observe_intentions ? destination_direction.x : 0.0;
  row.sin_d = observe_intentions ? destination_direction.y : 0.0;
  row.long_off = offset.longitudinal;
  row.lat_off = offset.lateral;
  row.ang_off = lane.LocalAngle(vehicle.heading, offset.longitudinal);
  if (origin_vehicle != nullptr) {
    const FeatureRow origin = VehicleToFeatureRow(road, *origin_vehicle);
    row.x -= origin.x;
    row.y -= origin.y;
    row.vx -= origin.vx;
    row.vy -= origin.vy;
  }
  return row;
}

FeatureRow ObjectToFeatureRow(const Road& road, const RoadObject& object,
                              const Vehicle* origin_vehicle) {
  const LaneIndex lane_index =
      object.lane_index.from.empty()
          ? road.network.GetClosestLaneIndex(object.position, object.heading)
          : object.lane_index;
  const Lane& lane = road.network.GetLane(lane_index);
  const LaneCoordinates offset = lane.LocalCoordinates(object.position);
  FeatureRow row;
  row.x = object.position.x;
  row.y = object.position.y;
  row.vx = object.speed * std::cos(object.heading);
  row.vy = object.speed * std::sin(object.heading);
  row.heading = object.heading;
  row.cos_h = std::cos(object.heading);
  row.sin_h = std::sin(object.heading);
  row.long_off = offset.longitudinal;
  row.lat_off = offset.lateral;
  row.ang_off = lane.LocalAngle(object.heading, offset.longitudinal);
  if (origin_vehicle != nullptr) {
    const FeatureRow origin = VehicleToFeatureRow(road, *origin_vehicle);
    row.x -= origin.x;
    row.y -= origin.y;
    row.vx -= origin.vx;
    row.vy -= origin.vy;
  }
  return row;
}

std::vector<float> ObserveKinematics(const Road& road, const Vehicle& observer,
                                     const KinematicObservationConfig& config) {
  std::vector<FeatureRow> rows;
  rows.reserve(config.vehicles_count);
  FeatureRow ego_row = VehicleToFeatureRow(road, observer);
  if (config.ego_x_override.has_value()) {
    ego_row.x = *config.ego_x_override;
  }
  rows.push_back(ego_row);
  const std::vector<CloseObjectRef> close =
      CloseObjectRefs(road, observer, config.vehicles_count - 1,
                      config.see_behind, config.include_obstacles);
  const Vehicle* origin = config.absolute ? nullptr : &observer;
  for (const CloseObjectRef& ref : close) {
    if (ref.is_vehicle) {
      rows.push_back(VehicleToFeatureRow(road, road.vehicles[ref.index], origin,
                                         /*observe_intentions=*/false));
    } else {
      rows.push_back(ObjectToFeatureRow(road, road.objects[ref.index], origin));
    }
  }
  for (FeatureRow& row : rows) {
    if (config.normalize) {
      NormalizeRow(&row, road, observer, config);
    }
  }

  std::vector<float> observation(config.vehicles_count * config.features.size(),
                                 0.0f);
  const int rows_to_write =
      std::min(config.vehicles_count, static_cast<int>(rows.size()));
  for (int r = 0; r < rows_to_write; ++r) {
    for (int c = 0; static_cast<std::size_t>(c) < config.features.size(); ++c) {
      observation[r * config.features.size() + c] =
          static_cast<float>(FeatureValue(rows[r], config.features[c]));
    }
  }
  return observation;
}

std::vector<float> ObserveTimeToCollision(const Road& road,
                                          const Vehicle& observer,
                                          double time_quantization,
                                          double horizon) {
  const std::vector<LaneIndex> road_lanes =
      road.network.AllSideLanes(observer.lane_index);
  const int speed_count = observer.target_speeds.size();
  const int lane_count = road_lanes.size();
  const int time_count = static_cast<int>(horizon / time_quantization);
  std::vector<float> grid(speed_count * lane_count * time_count, 0.0f);
  auto grid_at = [&](int speed, int lane, int time) -> float& {
    return grid[(speed * lane_count + lane) * time_count + time];
  };

  for (int speed_index = 0; speed_index < speed_count; ++speed_index) {
    const double ego_speed = IndexToSpeed(observer, speed_index);
    for (const Vehicle& other : road.vehicles) {
      if (&other == &observer || ego_speed == other.speed) {
        continue;
      }
      const double margin = kVehicleLength;
      const std::array<std::pair<double, float>, 3> collision_points{
          {{0.0, 1.0f}, {-margin, 0.5f}, {margin, 0.5f}}};
      for (const auto& [margin_offset, cost] : collision_points) {
        const double distance =
            LaneDistanceTo(road.network, observer, other) + margin_offset;
        const double other_projected_speed =
            other.speed * Dot(other.Direction(), observer.Direction());
        const double time_to_collision =
            distance / NotZero(ego_speed - other_projected_speed);
        if (time_to_collision < 0.0) {
          continue;
        }
        if (!road.network.IsConnectedRoad(observer.lane_index, other.lane_index,
                                          observer.route, false, 3)) {
          continue;
        }

        std::vector<int> lanes;
        if (road.network.AllSideLanes(other.lane_index).size() ==
            road_lanes.size()) {
          lanes.push_back(other.lane_index.id);
        } else {
          lanes.reserve(lane_count);
          for (int lane = 0; lane < lane_count; ++lane) {
            lanes.push_back(lane);
          }
        }
        for (const int time :
             {TimeIndex(time_to_collision, time_quantization),
              CeilTimeIndex(time_to_collision, time_quantization)}) {
          if (time < 0 || time >= time_count) {
            continue;
          }
          for (int lane : lanes) {
            if (0 <= lane && lane < lane_count) {
              grid_at(speed_index, lane, time) =
                  std::max(grid_at(speed_index, lane, time), cost);
            }
          }
        }
      }
    }
  }

  constexpr int kObsSpeeds = 3;
  constexpr int kObsLanes = 3;
  std::vector<float> obs(kObsSpeeds * kObsLanes * time_count, 1.0f);
  auto obs_at = [&](int speed, int lane, int time) -> float& {
    return obs[(speed * kObsLanes + lane) * time_count + time];
  };
  for (int obs_speed = 0; obs_speed < kObsSpeeds; ++obs_speed) {
    const int grid_speed = std::clamp(
        observer.speed_index + obs_speed - kObsSpeeds / 2, 0, speed_count - 1);
    for (int obs_lane = 0; obs_lane < kObsLanes; ++obs_lane) {
      const int grid_lane = observer.lane_index.id + obs_lane - kObsLanes / 2;
      if (grid_lane < 0 || grid_lane >= lane_count) {
        continue;
      }
      for (int time = 0; time < time_count; ++time) {
        obs_at(obs_speed, obs_lane, time) =
            grid_at(grid_speed, grid_lane, time);
      }
    }
  }
  return obs;
}

double FeatureValue(const FeatureRow& row, KinematicFeature feature) {
  switch (feature) {
    case KinematicFeature::kPresence:
      return row.presence;
    case KinematicFeature::kX:
      return row.x;
    case KinematicFeature::kY:
      return row.y;
    case KinematicFeature::kVx:
      return row.vx;
    case KinematicFeature::kVy:
      return row.vy;
    case KinematicFeature::kHeading:
      return row.heading;
    case KinematicFeature::kCosH:
      return row.cos_h;
    case KinematicFeature::kSinH:
      return row.sin_h;
    case KinematicFeature::kCosD:
      return row.cos_d;
    case KinematicFeature::kSinD:
      return row.sin_d;
    case KinematicFeature::kLongOff:
      return row.long_off;
    case KinematicFeature::kLatOff:
      return row.lat_off;
    case KinematicFeature::kAngOff:
      return row.ang_off;
  }
  return 0.0;
}

}  // namespace highway::official
