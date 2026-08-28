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

#include "envpool/highway/official_task.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace highway::official {
namespace {

using Lines = std::array<LineType, 2>;

constexpr LineType kContinuous = LineType::kContinuous;
constexpr LineType kNone = LineType::kNone;
constexpr LineType kStriped = LineType::kStriped;

double NormalNoise(std::mt19937* generator, double scale) {
  return std::normal_distribution<double>(0.0, scale)(*generator);
}

double UniformValue(std::mt19937* generator, double low, double high) {
  return std::uniform_real_distribution<double>(low, high)(*generator);
}

void AddMergeHighwayLane(RoadNetwork* net, int lane, const Lines& line_type,
                         const Lines& line_type_merge) {
  const std::array<double, 4> ends{150.0, 80.0, 80.0, 150.0};
  const double y = static_cast<double>(lane) * kDefaultLaneWidth;
  net->AddLane("a", "b",
               Lane::Straight({0.0, y}, {ends[0] + ends[1], y},
                              kDefaultLaneWidth, line_type));
  net->AddLane(
      "b", "c",
      Lane::Straight({ends[0] + ends[1], y}, {ends[0] + ends[1] + ends[2], y},
                     kDefaultLaneWidth, line_type_merge));
  net->AddLane("c", "d",
               Lane::Straight({ends[0] + ends[1] + ends[2], y},
                              {ends[0] + ends[1] + ends[2] + ends[3], y},
                              kDefaultLaneWidth, line_type));
}

Vehicle MakeMergeTraffic(const RoadNetwork& network,
                         const LaneIndex& lane_index, double longitudinal,
                         double speed) {
  Vehicle vehicle = MakeIDMVehicle(
      network, network.GetLane(lane_index).Position(longitudinal, 0.0),
      network.GetLane(lane_index).HeadingAt(longitudinal), speed);
  vehicle.lane_index = lane_index;
  vehicle.target_lane_index = lane_index;
  vehicle.target_speed = speed;
  return vehicle;
}

Vehicle MakeRoundaboutIDM(Road* road, const LaneIndex& lane_index,
                          double longitudinal, double speed,
                          const std::string& destination) {
  const Lane& lane = road->network.GetLane(lane_index);
  Vehicle vehicle =
      MakeIDMVehicle(road->network, lane.Position(longitudinal, 0.0),
                     lane.HeadingAt(longitudinal), speed);
  vehicle.lane_index = lane_index;
  vehicle.target_lane_index = lane_index;
  PlanRouteTo(&vehicle, road->network, destination);
  road->vehicles.push_back(vehicle);
  return vehicle;
}

std::vector<LaneIndex> ParkingSpots() {
  std::vector<LaneIndex> spots;
  spots.reserve(28);
  for (int k = 0; k < 14; ++k) {
    spots.push_back({"a", "b", k});
    spots.push_back({"b", "c", k});
  }
  return spots;
}

bool SameLaneIndex(const LaneIndex& lhs, const LaneIndex& rhs) {
  return lhs.from == rhs.from && lhs.to == rhs.to && lhs.id == rhs.id;
}

Vec2 Rotate(Vec2 value, double angle) {
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  return {c * value.x - s * value.y, s * value.x + c * value.y};
}

LaneIndex ParkingSpot(int spot) {
  const std::vector<LaneIndex> spots = ParkingSpots();
  const int index = ((spot % static_cast<int>(spots.size())) +
                     static_cast<int>(spots.size())) %
                    static_cast<int>(spots.size());
  return spots[index];
}

void AddParkingWalls(Road* road) {
  constexpr double width = 70.0;
  constexpr double height = 42.0;
  for (double y : {-height / 2.0, height / 2.0}) {
    RoadObject obstacle;
    obstacle.kind = RoadObjectKind::kObstacle;
    obstacle.position = {0.0, y};
    obstacle.length = width;
    obstacle.width = 1.0;
    road->objects.push_back(obstacle);
  }
  for (double x : {-width / 2.0, width / 2.0}) {
    RoadObject obstacle;
    obstacle.kind = RoadObjectKind::kObstacle;
    obstacle.position = {x, 0.0};
    obstacle.heading = kPi / 2.0;
    obstacle.length = height;
    obstacle.width = 1.0;
    road->objects.push_back(obstacle);
  }
}

}  // namespace

Road MakeMergeRoad() {
  Road road;

  AddMergeHighwayLane(&road.network, 0, {kContinuous, kStriped},
                      {kContinuous, kStriped});
  AddMergeHighwayLane(&road.network, 1, {kNone, kContinuous},
                      {kNone, kStriped});

  const std::array<double, 4> ends{150.0, 80.0, 80.0, 150.0};
  const double amplitude = 3.25;
  const Lane ljk =
      Lane::Straight({0.0, 6.5 + 4.0 + 4.0}, {ends[0], 6.5 + 4.0 + 4.0},
                     kDefaultLaneWidth, {kContinuous, kContinuous}, true);
  const Lane lkb =
      Lane::Sine(ljk.Position(ends[0], -amplitude),
                 ljk.Position(ends[0] + ends[1], -amplitude), amplitude,
                 2.0 * kPi / (2.0 * ends[1]), kPi / 2.0, kDefaultLaneWidth,
                 {kContinuous, kContinuous}, true);
  const Lane lbc =
      Lane::Straight(lkb.Position(ends[1], 0.0),
                     lkb.Position(ends[1], 0.0) + Vec2{ends[2], 0.0},
                     kDefaultLaneWidth, {kNone, kContinuous}, true);
  road.network.AddLane("j", "k", ljk);
  road.network.AddLane("k", "b", lkb);
  road.network.AddLane("b", "c", lbc);

  RoadObject obstacle;
  obstacle.kind = RoadObjectKind::kObstacle;
  obstacle.position = lbc.Position(ends[2], 0.0);
  obstacle.lane_index = {"b", "c", 2};
  road.objects.push_back(obstacle);
  return road;
}

int ResetMergeVehicles(Road* road, double position_noise0,
                       double position_noise1, double position_noise2,
                       double speed_noise0, double speed_noise1,
                       double speed_noise2) {
  road->vehicles.clear();

  const Lane& ego_lane = road->network.GetLane({"a", "b", 1});
  Vehicle ego = MakeMDPVehicle(road->network, ego_lane.Position(30.0, 0.0),
                               ego_lane.HeadingAt(30.0), 30.0);
  ego.lane_index = {"a", "b", 1};
  ego.target_lane_index = ego.lane_index;
  ego.speed_index = 2;
  ego.target_speed = 30.0;
  road->vehicles.push_back(ego);

  road->vehicles.push_back(MakeMergeTraffic(road->network, {"a", "b", 1},
                                            90.0 + position_noise0,
                                            29.0 + speed_noise0));
  road->vehicles.push_back(MakeMergeTraffic(road->network, {"a", "b", 1},
                                            70.0 + position_noise1,
                                            31.0 + speed_noise1));
  road->vehicles.push_back(MakeMergeTraffic(road->network, {"a", "b", 1},
                                            5.0 + position_noise2,
                                            31.5 + speed_noise2));

  const Lane& merging_lane = road->network.GetLane({"j", "k", 0});
  Vehicle merging_vehicle =
      MakeIDMVehicle(road->network, merging_lane.Position(110.0, 0.0),
                     merging_lane.HeadingAt(110.0), 20.0);
  merging_vehicle.lane_index = {"j", "k", 0};
  merging_vehicle.target_lane_index = merging_vehicle.lane_index;
  merging_vehicle.target_speed = 30.0;
  road->vehicles.push_back(merging_vehicle);

  return 0;
}

Road MakeMergeGenericRoad(int lanes, double before_merge, double converge_merge,
                          double parallel_merge, double after_merge) {
  if (lanes < 1 || before_merge <= 0.0 || converge_merge <= 0.0 ||
      parallel_merge <= 0.0 || after_merge < 90.0) {
    throw std::invalid_argument("invalid generic highway merge geometry");
  }
  Road road;
  const std::array<double, 4> ends = {
      0.0, before_merge + converge_merge,
      before_merge + converge_merge + parallel_merge,
      before_merge + converge_merge + parallel_merge + after_merge};
  const std::array<std::pair<const char*, const char*>, 3> edges = {
      {{"a", "b"}, {"b", "c"}, {"c", "d"}}};
  for (int segment = 0; segment < 3; ++segment) {
    for (int lane = 0; lane < lanes; ++lane) {
      const double y = lane * kDefaultLaneWidth;
      const Lines lines = {
          lane == 0 ? LineType::kContinuousLine : kStriped,
          lane == lanes - 1 ? LineType::kContinuousLine : kNone};
      road.network.AddLane(
          edges[segment].first, edges[segment].second,
          Lane::Straight({ends[segment], y}, {ends[segment + 1], y},
                         kDefaultLaneWidth, lines, false, 30.0));
    }
  }

  constexpr double amplitude = 3.25;
  const double y_parallel = lanes * kDefaultLaneWidth;
  const Lane approach =
      Lane::Straight({0.0, y_parallel + 2.0 * amplitude},
                     {before_merge, y_parallel + 2.0 * amplitude},
                     kDefaultLaneWidth, {kContinuous, kContinuous}, true, 30.0);
  const Lane converge =
      Lane::Sine({before_merge, y_parallel + amplitude},
                 {before_merge + converge_merge, y_parallel + amplitude},
                 amplitude, kPi / converge_merge, kPi / 2.0, kDefaultLaneWidth,
                 {kContinuous, kContinuous}, true, 30.0);
  const Lane merge = Lane::Straight(
      {before_merge + converge_merge, y_parallel},
      {before_merge + converge_merge + parallel_merge, y_parallel},
      kDefaultLaneWidth, {kStriped, kContinuous}, true, 30.0);
  road.network.AddLane("j", "k", approach);
  road.network.AddLane("k", "b", converge);
  road.network.AddLane("b", "c", merge);

  RoadObject obstacle;
  obstacle.kind = RoadObjectKind::kObstacle;
  obstacle.position = merge.Position(parallel_merge, 0.0);
  obstacle.lane_index = {"b", "c", lanes};
  road.objects.push_back(obstacle);
  return road;
}

int ResetMergeGenericVehicles(Road* road, int lanes, int vehicle_count,
                              double max_position, std::mt19937* generator) {
  road->vehicles.clear();
  const LaneIndex ego_index{"a", "b", lanes - 1};
  const Lane& ego_lane = road->network.GetLane(ego_index);
  Vehicle ego = MakeMDPVehicle(road->network, ego_lane.Position(30.0, 0.0),
                               ego_lane.HeadingAt(30.0), 30.0);
  ego.lane_index = ego_index;
  ego.target_lane_index = ego_index;
  road->vehicles.push_back(ego);

  std::vector<std::vector<double>> positions(lanes);
  positions[lanes - 1].push_back(30.0);
  std::uniform_int_distribution<int> lane_distribution(0, lanes - 1);
  std::uniform_real_distribution<double> longitudinal_distribution(
      0.0, max_position);
  std::uniform_real_distribution<double> speed_distribution(-2.0, 2.0);
  for (int i = 0; i < vehicle_count; ++i) {
    for (int attempt = 0; attempt < 10; ++attempt) {
      const int lane_id = lane_distribution(*generator);
      const double longitudinal = longitudinal_distribution(*generator);
      if (std::any_of(positions[lane_id].begin(), positions[lane_id].end(),
                      [&](double position) {
                        return std::abs(longitudinal - position) <= 15.0;
                      })) {
        continue;
      }
      const LaneIndex lane_index{"a", "b", lane_id};
      const Lane& lane = road->network.GetLane(lane_index);
      road->vehicles.push_back(MakeIDMVehicle(
          road->network, lane.Position(longitudinal, 0.0),
          lane.HeadingAt(longitudinal), 30.0 + speed_distribution(*generator)));
      positions[lane_id].push_back(longitudinal);
      break;
    }
  }

  const Lane& merge_lane = road->network.GetLane({"j", "k", 0});
  Vehicle merge_vehicle =
      MakeIDMVehicle(road->network, merge_lane.Position(60.0, 0.0),
                     merge_lane.HeadingAt(60.0), 20.0);
  merge_vehicle.target_speed = 30.0;
  road->vehicles.push_back(merge_vehicle);
  return 0;
}

Road MakeRoundaboutRoad() {
  Road road;
  const Vec2 center{0.0, 0.0};
  constexpr double radius = 20.0;
  constexpr double alpha = 24.0 * kPi / 180.0;
  const std::array<double, 2> radii{radius, radius + 4.0};
  const std::array<Lines, 2> line{
      {{kContinuous, kStriped}, {kNone, kContinuous}}};

  for (int lane = 0; lane < 2; ++lane) {
    road.network.AddLane(
        "se", "ex",
        Lane::Circular(center, radii[lane], kPi / 2.0 - alpha, alpha, false,
                       kDefaultLaneWidth, line[lane]));
    road.network.AddLane("ex", "ee",
                         Lane::Circular(center, radii[lane], alpha, -alpha,
                                        false, kDefaultLaneWidth, line[lane]));
    road.network.AddLane(
        "ee", "nx",
        Lane::Circular(center, radii[lane], -alpha, -kPi / 2.0 + alpha, false,
                       kDefaultLaneWidth, line[lane]));
    road.network.AddLane("nx", "ne",
                         Lane::Circular(center, radii[lane], -kPi / 2.0 + alpha,
                                        -kPi / 2.0 - alpha, false,
                                        kDefaultLaneWidth, line[lane]));
    road.network.AddLane(
        "ne", "wx",
        Lane::Circular(center, radii[lane], -kPi / 2.0 - alpha, -kPi + alpha,
                       false, kDefaultLaneWidth, line[lane]));
    road.network.AddLane(
        "wx", "we",
        Lane::Circular(center, radii[lane], -kPi + alpha, -kPi - alpha, false,
                       kDefaultLaneWidth, line[lane]));
    road.network.AddLane(
        "we", "sx",
        Lane::Circular(center, radii[lane], kPi - alpha, kPi / 2.0 + alpha,
                       false, kDefaultLaneWidth, line[lane]));
    road.network.AddLane("sx", "se",
                         Lane::Circular(center, radii[lane], kPi / 2.0 + alpha,
                                        kPi / 2.0 - alpha, false,
                                        kDefaultLaneWidth, line[lane]));
  }

  constexpr double access = 170.0;
  constexpr double dev = 85.0;
  constexpr double a = 5.0;
  constexpr double delta_st = 0.2 * dev;
  constexpr double delta_en = dev - delta_st;
  const double w = 2.0 * kPi / dev;

  road.network.AddLane(
      "ser", "ses",
      Lane::Straight({2.0, access}, {2.0, dev / 2.0}, kDefaultLaneWidth,
                     {kStriped, kContinuous}));
  road.network.AddLane(
      "ses", "se",
      Lane::Sine({2.0 + a, dev / 2.0}, {2.0 + a, dev / 2.0 - delta_st}, a, w,
                 -kPi / 2.0, kDefaultLaneWidth, {kContinuous, kContinuous}));
  road.network.AddLane(
      "sx", "sxs",
      Lane::Sine({-2.0 - a, -dev / 2.0 + delta_en}, {-2.0 - a, dev / 2.0}, a, w,
                 -kPi / 2.0 + w * delta_en, kDefaultLaneWidth,
                 {kContinuous, kContinuous}));
  road.network.AddLane("sxs", "sxr",
                       Lane::Straight({-2.0, dev / 2.0}, {-2.0, access},
                                      kDefaultLaneWidth, {kNone, kContinuous}));

  road.network.AddLane(
      "eer", "ees",
      Lane::Straight({access, -2.0}, {dev / 2.0, -2.0}, kDefaultLaneWidth,
                     {kStriped, kContinuous}));
  road.network.AddLane(
      "ees", "ee",
      Lane::Sine({dev / 2.0, -2.0 - a}, {dev / 2.0 - delta_st, -2.0 - a}, a, w,
                 -kPi / 2.0, kDefaultLaneWidth, {kContinuous, kContinuous}));
  road.network.AddLane(
      "ex", "exs",
      Lane::Sine({-dev / 2.0 + delta_en, 2.0 + a}, {dev / 2.0, 2.0 + a}, a, w,
                 -kPi / 2.0 + w * delta_en, kDefaultLaneWidth,
                 {kContinuous, kContinuous}));
  road.network.AddLane("exs", "exr",
                       Lane::Straight({dev / 2.0, 2.0}, {access, 2.0},
                                      kDefaultLaneWidth, {kNone, kContinuous}));

  road.network.AddLane(
      "ner", "nes",
      Lane::Straight({-2.0, -access}, {-2.0, -dev / 2.0}, kDefaultLaneWidth,
                     {kStriped, kContinuous}));
  road.network.AddLane(
      "nes", "ne",
      Lane::Sine({-2.0 - a, -dev / 2.0}, {-2.0 - a, -dev / 2.0 + delta_st}, a,
                 w, -kPi / 2.0, kDefaultLaneWidth, {kContinuous, kContinuous}));
  road.network.AddLane(
      "nx", "nxs",
      Lane::Sine({2.0 + a, dev / 2.0 - delta_en}, {2.0 + a, -dev / 2.0}, a, w,
                 -kPi / 2.0 + w * delta_en, kDefaultLaneWidth,
                 {kContinuous, kContinuous}));
  road.network.AddLane("nxs", "nxr",
                       Lane::Straight({2.0, -dev / 2.0}, {2.0, -access},
                                      kDefaultLaneWidth, {kNone, kContinuous}));

  road.network.AddLane(
      "wer", "wes",
      Lane::Straight({-access, 2.0}, {-dev / 2.0, 2.0}, kDefaultLaneWidth,
                     {kStriped, kContinuous}));
  road.network.AddLane(
      "wes", "we",
      Lane::Sine({-dev / 2.0, 2.0 + a}, {-dev / 2.0 + delta_st, 2.0 + a}, a, w,
                 -kPi / 2.0, kDefaultLaneWidth, {kContinuous, kContinuous}));
  road.network.AddLane(
      "wx", "wxs",
      Lane::Sine({dev / 2.0 - delta_en, -2.0 - a}, {-dev / 2.0, -2.0 - a}, a, w,
                 -kPi / 2.0 + w * delta_en, kDefaultLaneWidth,
                 {kContinuous, kContinuous}));
  road.network.AddLane("wxs", "wxr",
                       Lane::Straight({-dev / 2.0, -2.0}, {-access, -2.0},
                                      kDefaultLaneWidth, {kNone, kContinuous}));

  return road;
}

int ResetRoundaboutVehicles(Road* road, std::mt19937* generator) {
  road->vehicles.clear();
  const Lane& ego_lane = road->network.GetLane({"ser", "ses", 0});
  Vehicle ego = MakeMDPVehicle(road->network, ego_lane.Position(125.0, 0.0),
                               ego_lane.HeadingAt(140.0), 8.0, std::nullopt,
                               std::nullopt, {0.0, 8.0, 16.0});
  ego.lane_index = {"ser", "ses", 0};
  ego.target_lane_index = ego.lane_index;
  PlanRouteTo(&ego, road->network, "nxs");
  road->vehicles.push_back(ego);

  const std::array<LaneIndex, 4> lanes = {
      {{"we", "sx", 1}, {"we", "sx", 0}, {"we", "sx", 0}, {"eer", "ees", 0}}};
  const std::array<double, 4> positions = {5.0, 20.0, -20.0, 50.0};
  const std::array<const char*, 3> destinations = {"exr", "sxr", "nxr"};
  for (int i = 0; i < 4; ++i) {
    const double position = positions[i] + NormalNoise(generator, 2.0);
    const double speed = 16.0 + NormalNoise(generator, 2.0);
    const int destination =
        std::uniform_int_distribution<int>(0, 2)(*generator);
    MakeRoundaboutIDM(road, lanes[i], position, speed,
                      destinations[destination]);
    road->vehicles.back().idm_delta = UniformValue(generator, 3.5, 4.5);
  }
  return 0;
}

Road MakeRoundaboutGenericRoad(double radius, int lanes) {
  if (radius <= 0.0 || lanes < 1) {
    throw std::invalid_argument("invalid generic highway roundabout geometry");
  }
  Road road;
  constexpr double alpha = 24.0 * kPi / 180.0;
  const std::array<const char*, 9> nodes = {"se", "ex", "ee", "nx", "ne",
                                            "wx", "we", "sx", "se"};
  const std::array<std::array<double, 2>, 8> angles = {
      {{kPi / 2.0 - alpha, alpha},
       {alpha, -alpha},
       {-alpha, -kPi / 2.0 + alpha},
       {-kPi / 2.0 + alpha, -kPi / 2.0 - alpha},
       {-kPi / 2.0 - alpha, -kPi + alpha},
       {-kPi + alpha, -kPi - alpha},
       {kPi - alpha, kPi / 2.0 + alpha},
       {kPi / 2.0 + alpha, kPi / 2.0 - alpha}}};
  for (int lane = 0; lane < lanes; ++lane) {
    const Lines lines{lane == 0 ? kContinuous : kNone,
                      lane == lanes - 1 ? kContinuous : kStriped};
    for (int segment = 0; segment < 8; ++segment) {
      road.network.AddLane(
          nodes[segment], nodes[segment + 1],
          Lane::Circular({0.0, 0.0}, radius + 4.0 * lane, angles[segment][0],
                         angles[segment][1], false, kDefaultLaneWidth, lines));
    }
  }

  const double outer_radius = radius + 4.0 * (lanes - 1);
  const double half_span = std::max(100.0, 2.0 * outer_radius + 40.0) / 2.0;
  const double access = 2.0 * half_span + 40.0;
  const Vec2 entry_join{outer_radius * std::sin(alpha),
                        outer_radius * std::cos(alpha)};
  const Vec2 exit_join{-entry_join.x, entry_join.y};
  const double entry_amplitude = (entry_join.x - 2.0) / 2.0;
  const double exit_amplitude = (exit_join.x + 2.0) / 2.0;
  const double pulsation = kPi / (half_span - entry_join.y);
  const std::array<std::array<const char*, 6>, 4> arm_nodes = {
      {{"ser", "ses", "se", "sx", "sxs", "sxr"},
       {"eer", "ees", "ee", "ex", "exs", "exr"},
       {"ner", "nes", "ne", "nx", "nxs", "nxr"},
       {"wer", "wes", "we", "wx", "wxs", "wxr"}}};
  for (int arm = 0; arm < 4; ++arm) {
    const double rotation = -arm * kPi / 2.0;
    auto point = [&](double x, double y) { return Rotate({x, y}, rotation); };
    const auto& names = arm_nodes[arm];
    road.network.AddLane(
        names[0], names[1],
        Lane::Straight(point(2.0, access), point(2.0, half_span),
                       kDefaultLaneWidth, {kStriped, kContinuous}));
    road.network.AddLane(
        names[1], names[2],
        Lane::Sine(point(2.0 + entry_amplitude, half_span),
                   point(2.0 + entry_amplitude, entry_join.y), entry_amplitude,
                   pulsation, -kPi / 2.0, kDefaultLaneWidth,
                   {kContinuous, kContinuous}));
    road.network.AddLane(
        names[3], names[4],
        Lane::Sine(point(exit_join.x - exit_amplitude, exit_join.y),
                   point(exit_join.x - exit_amplitude, half_span),
                   exit_amplitude, pulsation, -kPi / 2.0, kDefaultLaneWidth,
                   {kContinuous, kContinuous}));
    road.network.AddLane(
        names[4], names[5],
        Lane::Straight(point(-2.0, half_span), point(-2.0, access),
                       kDefaultLaneWidth, {kNone, kContinuous}));
  }
  return road;
}

int ResetRoundaboutGenericVehicles(Road* road, int vehicle_count,
                                   std::mt19937* generator) {
  road->vehicles.clear();
  const LaneIndex ego_index{"ser", "ses", 0};
  const Lane& ego_lane = road->network.GetLane(ego_index);
  const double ego_longitudinal = ego_lane.Length() - 2.5;
  Vehicle ego =
      MakeMDPVehicle(road->network, ego_lane.Position(ego_longitudinal, 0.0),
                     ego_lane.HeadingAt(ego_longitudinal), 8.0, std::nullopt,
                     std::nullopt, {0.0, 8.0, 16.0});
  ego.lane_index = ego_index;
  ego.target_lane_index = ego_index;
  PlanRouteTo(&ego, road->network, "nxs");
  road->vehicles.push_back(ego);

  const std::array<std::pair<const char*, const char*>, 7> spawn_lanes = {
      {{"we", "sx"},
       {"sx", "se"},
       {"ee", "nx"},
       {"nx", "ne"},
       {"eer", "ees"},
       {"ner", "nes"},
       {"wer", "wes"}}};
  const std::array<const char*, 4> destinations = {"exr", "sxr", "nxr", "wxr"};
  std::uniform_int_distribution<int> spawn_distribution(0,
                                                        spawn_lanes.size() - 1);
  std::uniform_int_distribution<int> destination_distribution(
      0, destinations.size() - 1);
  std::normal_distribution<double> speed_distribution(14.0, 2.0);
  std::vector<Vec2> occupied = {ego.position};
  for (int i = 0; i < vehicle_count; ++i) {
    for (int attempt = 0; attempt < 10; ++attempt) {
      const auto& edge = spawn_lanes[spawn_distribution(*generator)];
      std::vector<LaneIndex> candidates;
      for (const LaneIndex& index : road->network.LaneIndexes()) {
        if (index.from == edge.first && index.to == edge.second) {
          candidates.push_back(index);
        }
      }
      std::uniform_int_distribution<int> lane_distribution(
          0, candidates.size() - 1);
      const LaneIndex lane_index = candidates[lane_distribution(*generator)];
      const Lane& lane = road->network.GetLane(lane_index);
      std::uniform_real_distribution<double> longitudinal_distribution(
          5.0, std::max(5.0, lane.Length() - 5.0));
      const double longitudinal = longitudinal_distribution(*generator);
      const Vec2 position = lane.Position(longitudinal, 0.0);
      if (std::any_of(occupied.begin(), occupied.end(), [&](Vec2 other) {
            return Norm(position - other) < 7.0;
          })) {
        continue;
      }
      Vehicle other =
          MakeIDMVehicle(road->network, position, lane.HeadingAt(longitudinal),
                         speed_distribution(*generator));
      other.lane_index = lane_index;
      other.target_lane_index = lane_index;
      PlanRouteTo(&other, road->network,
                  destinations[destination_distribution(*generator)]);
      road->vehicles.push_back(other);
      occupied.push_back(position);
      break;
    }
  }
  return 0;
}

Road MakeTwoWayRoad() {
  Road road;
  constexpr double length = 800.0;
  road.network.AddLane(
      "a", "b",
      Lane::Straight({0.0, 0.0}, {length, 0.0}, kDefaultLaneWidth,
                     {kContinuous, kStriped}));
  road.network.AddLane(
      "a", "b",
      Lane::Straight({0.0, kDefaultLaneWidth}, {length, kDefaultLaneWidth},
                     kDefaultLaneWidth, {kNone, kContinuous}));
  road.network.AddLane("b", "a",
                       Lane::Straight({length, 0.0}, {0.0, 0.0},
                                      kDefaultLaneWidth, {kNone, kNone}));
  return road;
}

Vehicle MakeTwoWayIDM(const RoadNetwork& network, const LaneIndex& lane_index,
                      double longitudinal, double speed) {
  const Lane& lane = network.GetLane(lane_index);
  Vehicle vehicle = MakeIDMVehicle(network, lane.Position(longitudinal, 0.0),
                                   lane.HeadingAt(longitudinal), speed,
                                   lane_index, speed, {}, false);
  vehicle.lane_index = lane_index;
  vehicle.target_lane_index = lane_index;
  vehicle.target_speed = speed;
  return vehicle;
}

int ResetTwoWayVehicles(Road* road, std::mt19937* generator) {
  road->vehicles.clear();
  const LaneIndex ego_lane_index{"a", "b", 1};
  const Lane& ego_lane = road->network.GetLane(ego_lane_index);
  Vehicle ego = MakeMDPVehicle(road->network, ego_lane.Position(30.0, 0.0),
                               ego_lane.HeadingAt(30.0), 30.0);
  ego.lane_index = ego_lane_index;
  ego.target_lane_index = ego_lane_index;
  ego.speed_index = 2;
  ego.target_speed = 30.0;
  road->vehicles.push_back(ego);

  for (int i = 0; i < 3; ++i) {
    const double position = 70.0 + 40.0 * i + NormalNoise(generator, 10.0);
    const double speed = 24.0 + NormalNoise(generator, 2.0);
    road->vehicles.push_back(
        MakeTwoWayIDM(road->network, {"a", "b", 1}, position, speed));
  }
  for (int i = 0; i < 2; ++i) {
    const double position = 200.0 + 100.0 * i + NormalNoise(generator, 10.0);
    const double speed = 20.0 + NormalNoise(generator, 5.0);
    road->vehicles.push_back(
        MakeTwoWayIDM(road->network, {"b", "a", 0}, position, speed));
  }
  return 0;
}

Road MakeUTurnRoad() {
  Road road;
  constexpr double length = 128.0;
  constexpr double radius = 20.0;
  constexpr double offset = 2.0 * radius;
  constexpr Vec2 center{length, kDefaultLaneWidth + 20.0};

  road.network.AddLane(
      "c", "d",
      Lane::Straight({length, kDefaultLaneWidth}, {0.0, kDefaultLaneWidth},
                     kDefaultLaneWidth, {kContinuous, kStriped}));
  road.network.AddLane("c", "d",
                       Lane::Straight({length, 0.0}, {0.0, 0.0},
                                      kDefaultLaneWidth, {kNone, kContinuous}));

  const std::array<double, 2> radii{radius, radius + kDefaultLaneWidth};
  const std::array<Lines, 2> line{
      {{kContinuous, kStriped}, {kNone, kContinuous}}};
  for (int lane = 0; lane < 2; ++lane) {
    road.network.AddLane(
        "b", "c",
        Lane::Circular(center, radii[lane], kPi / 2.0, -kPi / 2.0, false,
                       kDefaultLaneWidth, line[lane]));
  }

  road.network.AddLane(
      "a", "b",
      Lane::Straight(
          {0.0, 2.0 * kDefaultLaneWidth + offset - kDefaultLaneWidth},
          {length, 2.0 * kDefaultLaneWidth + offset - kDefaultLaneWidth},
          kDefaultLaneWidth, {kContinuous, kStriped}));
  road.network.AddLane(
      "a", "b",
      Lane::Straight({0.0, 2.0 * kDefaultLaneWidth + offset},
                     {length, 2.0 * kDefaultLaneWidth + offset},
                     kDefaultLaneWidth, {kNone, kContinuous}));
  return road;
}

Vehicle MakeUTurnIDM(const RoadNetwork& network, const LaneIndex& lane_index,
                     double longitudinal, double speed) {
  const Lane& lane = network.GetLane(lane_index);
  Vehicle vehicle = MakeIDMVehicle(network, lane.Position(longitudinal, 0.0),
                                   lane.HeadingAt(longitudinal), speed);
  vehicle.lane_index = lane_index;
  vehicle.target_lane_index = lane_index;
  vehicle.target_speed = speed;
  PlanRouteTo(&vehicle, network, "d");
  return vehicle;
}

int ResetUTurnVehicles(Road* road, std::mt19937* generator) {
  road->vehicles.clear();
  const LaneIndex ego_lane_index{"a", "b", 0};
  const Lane& ego_lane = road->network.GetLane(ego_lane_index);
  Vehicle ego = MakeMDPVehicle(road->network, ego_lane.Position(0.0, 0.0),
                               ego_lane.HeadingAt(0.0), 16.0, std::nullopt,
                               std::nullopt, {8.0, 16.0, 24.0});
  ego.lane_index = ego_lane_index;
  ego.target_lane_index = ego_lane_index;
  ego.speed_index = 1;
  ego.target_speed = 16.0;
  PlanRouteTo(&ego, road->network, "d");
  road->vehicles.push_back(ego);

  const std::array<LaneIndex, 6> lanes = {{{"a", "b", 0},
                                           {"a", "b", 1},
                                           {"b", "c", 1},
                                           {"b", "c", 0},
                                           {"c", "d", 0},
                                           {"c", "d", 1}}};
  const std::array<double, 6> positions = {25.0, 56.0, 0.5, 17.5, 1.0, 30.0};
  const std::array<double, 6> speeds = {13.5, 14.5, 4.5, 5.5, 3.5, 5.5};
  for (int i = 0; i < 6; ++i) {
    const double position = positions[i] + NormalNoise(generator, 2.0);
    const double speed = speeds[i] + NormalNoise(generator, 2.0);
    road->vehicles.push_back(
        MakeUTurnIDM(road->network, lanes[i], position, speed));
    // Upstream randomizes the leading blocker's IDM behavior only.
    if (i == 0) {
      road->vehicles.back().idm_delta = UniformValue(generator, 3.5, 4.5);
    }
  }
  return 0;
}

Road MakeParkingRoad() {
  Road road;
  constexpr int spots = 14;
  constexpr double width = 4.0;
  constexpr double y_offset = 10.0;
  constexpr double length = 8.0;
  for (int k = 0; k < spots; ++k) {
    const double x =
        (static_cast<double>(k + 1) - static_cast<double>(spots) / 2.0) *
            width -
        width / 2.0;
    road.network.AddLane("a", "b",
                         Lane::Straight({x, y_offset}, {x, y_offset + length},
                                        width, {kContinuous, kContinuous}));
    road.network.AddLane("b", "c",
                         Lane::Straight({x, -y_offset}, {x, -y_offset - length},
                                        width, {kContinuous, kContinuous}));
  }
  return road;
}

int ResetParkingVehicles(Road* road, double ego_x, double ego_heading,
                         int goal_spot, bool add_parked_vehicles) {
  road->vehicles.clear();
  road->objects.clear();

  Vehicle ego = MakeVehicle(road->network, {ego_x, 0.0}, ego_heading, 0.0);
  ego.kind = VehicleKind::kVehicle;

  const LaneIndex goal_lane_index = ParkingSpot(goal_spot);
  const Lane& goal_lane = road->network.GetLane(goal_lane_index);
  const Vec2 goal_position = goal_lane.Position(goal_lane.Length() / 2.0, 0.0);
  RoadObject goal;
  goal.kind = RoadObjectKind::kLandmark;
  goal.position = goal_position;
  goal.heading = goal_lane.HeadingAt(goal_lane.Length() / 2.0);
  goal.solid = false;
  goal.lane_index = goal_lane_index;
  road->objects.push_back(goal);

  ego.has_goal = true;
  ego.goal_position = goal.position;
  ego.goal_heading = goal.heading;
  ego.goal_speed = goal.speed;
  road->vehicles.push_back(ego);

  if (add_parked_vehicles) {
    int parked = 0;
    for (const LaneIndex& lane_index : ParkingSpots()) {
      if (SameLaneIndex(lane_index, goal_lane_index)) {
        continue;
      }
      road->vehicles.push_back(
          MakeVehicleOnLane(road->network, lane_index, 4.0, 0.0));
      ++parked;
      if (parked >= 10) {
        break;
      }
    }
  }

  AddParkingWalls(road);
  return 0;
}

Road MakeExitRoad() {
  Road road;
  constexpr int lanes_count = 6;
  constexpr double exit_position = 400.0;
  constexpr double exit_length = 100.0;
  constexpr double road_length = 1000.0;
  auto speed_limit = [](int lane) {
    return 26.0 - 3.4 * static_cast<double>(lane);
  };
  auto add_straight_segment = [&](const std::string& from,
                                  const std::string& to, double start_x,
                                  double end_x, int lanes) {
    for (int lane = 0; lane < lanes; ++lane) {
      const double y = static_cast<double>(lane) * kDefaultLaneWidth;
      Lines line_types{
          lane == 0 ? kContinuous : kNone,
          lane == lanes - 1 ? kContinuous : kStriped,
      };
      road.network.AddLane(
          from, to,
          Lane::Straight({start_x, y}, {end_x, y}, kDefaultLaneWidth,
                         line_types, false, speed_limit(lane)));
    }
  };
  add_straight_segment("0", "1", 0.0, exit_position, lanes_count);
  add_straight_segment("1", "2", exit_position, exit_position + exit_length,
                       lanes_count + 1);
  add_straight_segment("2", "3", exit_position + exit_length, road_length,
                       lanes_count);

  constexpr double radius = 150.0;
  const Vec2 exit_lane_start{
      exit_position + exit_length,
      static_cast<double>(lanes_count) * kDefaultLaneWidth,
  };
  const Vec2 exit_center = exit_lane_start + Vec2{0.0, radius};
  road.network.AddLane(
      "2", "exit",
      Lane::Circular(exit_center, radius, 3.0 * kPi / 2.0, 2.0 * kPi, true,
                     kDefaultLaneWidth, {kStriped, kStriped}, true));
  return road;
}

Vehicle MakeExitTraffic(const RoadNetwork& network, const LaneIndex& lane_index,
                        double longitudinal) {
  const Lane& lane = network.GetLane(lane_index);
  Vehicle vehicle = MakeIDMVehicle(
      network, lane.Position(longitudinal, 0.0), lane.HeadingAt(longitudinal),
      lane.SpeedLimit(), lane_index, lane.SpeedLimit(), {}, false);
  vehicle.lane_index = lane_index;
  vehicle.target_lane_index = lane_index;
  vehicle.target_speed = lane.SpeedLimit();
  PlanRouteTo(&vehicle, network, "3");
  return vehicle;
}

int ResetExitVehicles(Road* road, std::mt19937* generator) {
  road->vehicles.clear();
  const LaneIndex ego_lane_index{"0", "1", 0};
  const Lane& ego_lane = road->network.GetLane(ego_lane_index);
  const double ego_offset = 2.0 * (12.0 + 25.0) * std::exp(-5.0 / 40.0 * 6.0);
  const double ego_position =
      ego_offset * (3.0 + UniformValue(generator, 0.9, 1.1));
  Vehicle ego =
      MakeMDPVehicle(road->network, ego_lane.Position(ego_position, 0.0),
                     ego_lane.HeadingAt(ego_position), 25.0, std::nullopt,
                     std::nullopt, {18.0, 24.0, 30.0});
  ego.lane_index = ego_lane_index;
  ego.target_lane_index = ego_lane_index;
  road->vehicles.push_back(ego);

  for (int i = 0; i < 20; ++i) {
    const std::array<double, 6> weights = {0, 1, 2, 3, 4, 5};
    const int lane = std::discrete_distribution<int>(weights.begin(),
                                                     weights.end())(*generator);
    const Lane& traffic_lane = road->network.GetLane({"0", "1", lane});
    double longitudinal = -std::numeric_limits<double>::infinity();
    for (const auto& vehicle : road->vehicles) {
      longitudinal = std::max(
          longitudinal,
          traffic_lane.LocalCoordinates(vehicle.position).longitudinal);
    }
    const double offset =
        (12.0 + traffic_lane.SpeedLimit()) / 1.5 * std::exp(-5.0 / 40.0 * 6.0);
    longitudinal += offset * UniformValue(generator, 0.9, 1.1);
    road->vehicles.push_back(
        MakeExitTraffic(road->network, {"0", "1", lane}, longitudinal));
  }
  return 0;
}

Road MakeIntersectionRoad() {
  Road road;
  road.regulated = true;
  constexpr double lane_width = kDefaultLaneWidth;
  constexpr double right_turn_radius = lane_width + 5.0;
  constexpr double left_turn_radius = right_turn_radius + lane_width;
  constexpr double outer_distance = right_turn_radius + lane_width / 2.0;
  constexpr double access_length = 100.0;

  for (int corner = 0; corner < 4; ++corner) {
    const double angle = (kPi / 2.0) * static_cast<double>(corner);
    const bool is_horizontal = corner % 2 != 0;
    const int priority = is_horizontal ? 3 : 1;
    const std::string corner_id = std::to_string(corner);
    const std::string prev_corner_id = std::to_string((corner + 3) % 4);
    const std::string left_corner_id = std::to_string((corner + 1) % 4);
    const std::string straight_corner_id = std::to_string((corner + 2) % 4);

    road.network.AddLane(
        "o" + corner_id, "ir" + corner_id,
        Lane::Straight(
            Rotate({lane_width / 2.0, access_length + outer_distance}, angle),
            Rotate({lane_width / 2.0, outer_distance}, angle), lane_width,
            {kStriped, kContinuous}, false, 10.0, priority));

    road.network.AddLane(
        "ir" + corner_id, "il" + prev_corner_id,
        Lane::Circular(Rotate({outer_distance, outer_distance}, angle),
                       right_turn_radius, angle + kPi, angle + 3.0 * kPi / 2.0,
                       true, lane_width, {kNone, kContinuous}, false, 10.0,
                       priority));

    road.network.AddLane(
        "ir" + corner_id, "il" + left_corner_id,
        Lane::Circular(Rotate({-left_turn_radius + lane_width / 2.0,
                               left_turn_radius - lane_width / 2.0},
                              angle),
                       left_turn_radius, angle, angle - kPi / 2.0, false,
                       lane_width, {kNone, kNone}, false, 10.0, priority - 1));

    road.network.AddLane(
        "ir" + corner_id, "il" + straight_corner_id,
        Lane::Straight(Rotate({lane_width / 2.0, outer_distance}, angle),
                       Rotate({lane_width / 2.0, -outer_distance}, angle),
                       lane_width, {kStriped, kNone}, false, 10.0, priority));

    road.network.AddLane(
        "il" + prev_corner_id, "o" + prev_corner_id,
        Lane::Straight(
            Rotate({outer_distance, lane_width / 2.0}, angle),
            Rotate({access_length + outer_distance, lane_width / 2.0}, angle),
            lane_width, {kNone, kContinuous}, false, 10.0, priority));
  }
  return road;
}

Vehicle MakeIntersectionIDM(const RoadNetwork& network, int incoming,
                            double longitudinal, double speed,
                            const std::string& destination) {
  const LaneIndex lane_index{"o" + std::to_string(incoming),
                             "ir" + std::to_string(incoming), 0};
  const Lane& lane = network.GetLane(lane_index);
  Vehicle vehicle = MakeIDMVehicle(network, lane.Position(longitudinal, 0.0),
                                   lane.HeadingAt(longitudinal), speed);
  vehicle.lane_index = lane_index;
  vehicle.target_lane_index = lane_index;
  vehicle.target_speed = speed;
  vehicle.idm_comfort_acc_max = 6.0;
  vehicle.idm_comfort_acc_min = -3.0;
  vehicle.idm_distance_wanted = 7.0;
  PlanRouteTo(&vehicle, network, destination);
  return vehicle;
}

void SpawnIntersectionVehicle(Road* road, std::mt19937* generator,
                              double longitudinal, double probability,
                              double position_deviation, double speed_deviation,
                              bool straight) {
  if (UniformValue(generator, 0.0, 1.0) > probability) {
    return;
  }
  const int incoming = std::uniform_int_distribution<int>(0, 3)(*generator);
  int outgoing = std::uniform_int_distribution<int>(0, 2)(*generator);
  outgoing += outgoing >= incoming;
  if (straight) {
    outgoing = (incoming + 2) % 4;
  }
  const double position =
      longitudinal + 5.0 + NormalNoise(generator, position_deviation);
  const double speed = 8.0 + NormalNoise(generator, speed_deviation);
  Vehicle vehicle = MakeIntersectionIDM(road->network, incoming, position,
                                        speed, "o" + std::to_string(outgoing));
  for (const Vehicle& other : road->vehicles) {
    if (Norm(other.position - vehicle.position) < 15.0) {
      return;
    }
  }
  vehicle.idm_delta = UniformValue(generator, 3.5, 4.5);
  road->vehicles.push_back(vehicle);
}

void UpdateIntersectionTraffic(Road* road, std::mt19937* generator, int players,
                               double spawn_probability) {
  road->vehicles.erase(
      std::remove_if(
          road->vehicles.begin() + players, road->vehicles.end(),
          [&](const Vehicle& vehicle) {
            const LaneIndex& index = vehicle.lane_index;
            const Lane& lane = road->network.GetLane(index);
            return vehicle.route.empty() ||
                   (index.from.find("il") != std::string::npos &&
                    index.to.find('o') != std::string::npos &&
                    lane.LocalCoordinates(vehicle.position).longitudinal >=
                        lane.Length() - 4.0 * kVehicleLength);
          }),
      road->vehicles.end());
  SpawnIntersectionVehicle(road, generator, 0.0, spawn_probability, 1.0, 1.0,
                           false);
}

int ResetIntersectionVehicles(Road* road, std::mt19937* generator, int players,
                              int simulation_frequency) {
  road->regulated_steps = 0;
  road->vehicles.clear();
  for (int i = 0; i < 9; ++i) {
    SpawnIntersectionVehicle(road, generator, 80.0 * i / 9.0, 0.6, 1.0, 1.0,
                             false);
  }
  for (int frame = 0; frame < 3 * simulation_frequency; ++frame) {
    road->Act();
    road->Step(1.0 / simulation_frequency);
  }
  SpawnIntersectionVehicle(road, generator, 60.0, 1.0, 0.1, 0.0, true);

  std::vector<Vehicle> controlled;
  for (int player = 0; player < players; ++player) {
    const std::string index = std::to_string(player % 4);
    const LaneIndex lane_index{"o" + index, "ir" + index, 0};
    const Lane& lane = road->network.GetLane(lane_index);
    const double longitudinal = 65.0 + NormalNoise(generator, 5.0);
    Vehicle ego = MakeMDPVehicle(
        road->network, lane.Position(longitudinal, 0.0), lane.HeadingAt(60.0),
        lane.SpeedLimit(), std::nullopt, std::nullopt, {0.0, 4.5, 9.0});
    ego.lane_index = lane_index;
    ego.target_lane_index = lane_index;
    ego.speed_index = 2;
    ego.target_speed = 9.0;
    PlanRouteTo(&ego, road->network, "o1");
    road->vehicles.erase(
        std::remove_if(road->vehicles.begin(), road->vehicles.end(),
                       [&](const Vehicle& vehicle) {
                         return Norm(vehicle.position - ego.position) < 20.0;
                       }),
        road->vehicles.end());
    controlled.push_back(ego);
  }
  // EnvPool addresses controlled players by their leading vehicle indexes.
  road->vehicles.insert(road->vehicles.begin(), controlled.begin(),
                        controlled.end());
  return 0;
}

Road MakeLaneKeepingRoad() {
  Road road;
  road.network.AddLane(
      "a", "b",
      Lane::Sine({0.0, 0.0}, {500.0, 0.0}, 5.0, 2.0 * kPi / 100.0, 0.0, 10.0,
                 {kStriped, kStriped}));
  road.network.AddLane(
      "c", "d",
      Lane::Straight({50.0, 50.0}, {115.0, 15.0}, 10.0, {kStriped, kStriped}));
  road.network.AddLane(
      "d", "a",
      Lane::Straight({115.0, 15.0},
                     {135.0, 15.0 + 20.0 * (15.0 - 50.0) / (115.0 - 50.0)},
                     10.0, {kNone, kStriped}));
  return road;
}

int ResetLaneKeepingVehicle(Road* road) {
  road->vehicles.clear();
  const LaneIndex lane_index{"c", "d", 0};
  const Lane& lane = road->network.GetLane(lane_index);
  Vehicle ego = MakeVehicle(road->network, lane.Position(50.0, -4.0),
                            lane.HeadingAt(0.0), 8.3);
  ego.kind = VehicleKind::kVehicle;
  ego.lane_index = lane_index;
  ego.target_lane_index = lane_index;
  road->vehicles.push_back(ego);
  return 0;
}

Road MakeRacetrackRoad(const std::string& scenario) {
  Road road;
  const int lanes =
      scenario == "racetrack_large" || scenario == "racetrack_oval" ? 3 : 2;
  const double start_x = scenario == "racetrack_oval" ? 0.0 : 42.0;
  double end_x = 100.0;
  if (scenario == "racetrack_large") {
    end_x = 200.0;
  } else if (scenario == "racetrack_oval") {
    end_x = 101.0;
  }
  const double width = 5.0;
  for (int lane = 0; lane < lanes; ++lane) {
    Lines line_types{kStriped, kStriped};
    if (lane == 0) {
      line_types[0] = kContinuous;
    }
    if (lane == lanes - 1) {
      line_types[1] = kContinuous;
    }
    road.network.AddLane(
        "a", "b",
        Lane::Straight({start_x, lane * width}, {end_x, lane * width}, width,
                       line_types, false, 10.0));
  }

  Vec2 center1{100.0, -20.0};
  if (scenario == "racetrack_large") {
    center1 = {200.0, -20.0};
  }
  for (int lane = 0; lane < lanes; ++lane) {
    Lines line_types{kStriped, kStriped};
    if (lane == 0) {
      line_types[0] = kContinuous;
      line_types[1] = kNone;
    }
    if (lane == lanes - 1) {
      line_types[1] = kContinuous;
    }
    road.network.AddLane(
        "b", "c",
        Lane::Circular(center1, 20.0 + lane * width, kPi / 2.0,
                       scenario == "racetrack_oval" ? 0.0 : -1.0 * kPi / 180.0,
                       false, width, line_types, false, 10.0));
  }
  if (scenario == "racetrack") {
    road.network.AddLane("c", "d",
                         Lane::Straight({120.0, -20.0}, {120.0, -30.0}, width,
                                        {kContinuous, kNone}, false, 10.0));
    road.network.AddLane("c", "d",
                         Lane::Straight({125.0, -20.0}, {125.0, -30.0}, width,
                                        {kStriped, kContinuous}, false, 10.0));
    road.network.AddLane(
        "d", "e",
        Lane::Circular({105.0, -30.0}, 15.0, 0.0, -181.0 * kPi / 180.0, false,
                       width, {kContinuous, kNone}, false, 10.0));
    road.network.AddLane(
        "d", "e",
        Lane::Circular({105.0, -30.0}, 20.0, 0.0, -181.0 * kPi / 180.0, false,
                       width, {kStriped, kContinuous}, false, 10.0));
    road.network.AddLane(
        "e", "f",
        Lane::Circular({70.0, -30.0}, 20.0, 0.0, 136.0 * kPi / 180.0, true,
                       width, {kContinuous, kStriped}, false, 10.0));
    road.network.AddLane(
        "e", "f",
        Lane::Circular({70.0, -30.0}, 15.0, 0.0, 137.0 * kPi / 180.0, true,
                       width, {kNone, kContinuous}, false, 10.0));
    road.network.AddLane("f", "g",
                         Lane::Straight({55.7, -15.7}, {35.7, -35.7}, width,
                                        {kContinuous, kNone}, false, 10.0));
    road.network.AddLane(
        "f", "g",
        Lane::Straight({59.3934, -19.2}, {39.3934, -39.2}, width,
                       {kStriped, kContinuous}, false, 10.0));
    road.network.AddLane(
        "g", "h",
        Lane::Circular({18.1, -18.1}, 25.0, 315.0 * kPi / 180.0,
                       170.0 * kPi / 180.0, false, width, {kContinuous, kNone},
                       false, 10.0));
    road.network.AddLane(
        "g", "h",
        Lane::Circular({18.1, -18.1}, 30.0, 315.0 * kPi / 180.0,
                       165.0 * kPi / 180.0, false, width,
                       {kStriped, kContinuous}, false, 10.0));
    road.network.AddLane(
        "h", "i",
        Lane::Circular({18.1, -18.1}, 25.0, 170.0 * kPi / 180.0,
                       56.0 * kPi / 180.0, false, width, {kContinuous, kNone},
                       false, 10.0));
    road.network.AddLane(
        "h", "i",
        Lane::Circular({18.1, -18.1}, 30.0, 170.0 * kPi / 180.0,
                       58.0 * kPi / 180.0, false, width,
                       {kStriped, kContinuous}, false, 10.0));
    road.network.AddLane("i", "a",
                         Lane::Circular({43.2, 23.4}, 23.5, 240.0 * kPi / 180.0,
                                        270.0 * kPi / 180.0, true, width,
                                        {kContinuous, kStriped}, false, 10.0));
    road.network.AddLane("i", "a",
                         Lane::Circular({43.2, 23.4}, 18.5, 238.0 * kPi / 180.0,
                                        268.0 * kPi / 180.0, true, width,
                                        {kNone, kContinuous}, false, 10.0));
  }
  return road;
}

int ResetRacetrackVehicles(Road* road, std::mt19937* generator) {
  road->vehicles.clear();
  const int lanes = road->network.AllSideLanes({"a", "b", 0}).size();
  const int lane = std::uniform_int_distribution<int>(0, lanes - 1)(*generator);
  const double longitudinal = UniformValue(generator, 20.0, 50.0);
  const LaneIndex lane_index{"a", "b", lane};
  const double speed_limit = road->network.GetLane(lane_index).SpeedLimit();
  Vehicle ego =
      MakeVehicleOnLane(road->network, lane_index, longitudinal, speed_limit);
  ego.kind = VehicleKind::kVehicle;
  road->vehicles.push_back(ego);
  const LaneIndex front_lane{"b", "c", lane};
  const Lane& front = road->network.GetLane(front_lane);
  const double front_position = UniformValue(generator, 0.0, front.Length());
  const double front_speed = UniformValue(generator, 6.0, 9.0);
  Vehicle traffic =
      MakeIDMVehicle(road->network, front.Position(front_position, 0),
                     front.HeadingAt(front_position), front_speed);
  road->vehicles.push_back(traffic);
  return 0;
}

}  // namespace highway::official
