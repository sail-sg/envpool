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

#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace highway::official {
namespace {

using Lines = std::array<LineType, 2>;

constexpr LineType kContinuous = LineType::kContinuous;
constexpr LineType kNone = LineType::kNone;
constexpr LineType kStriped = LineType::kStriped;

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

int ResetRoundaboutVehicles(Road* road) {
  road->vehicles.clear();
  const Lane& ego_lane = road->network.GetLane({"ser", "ses", 0});
  Vehicle ego = MakeMDPVehicle(road->network, ego_lane.Position(125.0, 0.0),
                               ego_lane.HeadingAt(140.0), 8.0, std::nullopt,
                               std::nullopt, {0.0, 8.0, 16.0});
  ego.lane_index = {"ser", "ses", 0};
  ego.target_lane_index = ego.lane_index;
  PlanRouteTo(&ego, road->network, "nxs");
  road->vehicles.push_back(ego);

  MakeRoundaboutIDM(road, {"we", "sx", 1}, 5.0, 16.0, "nxr");
  MakeRoundaboutIDM(road, {"we", "sx", 0}, 20.0, 16.0, "sxr");
  MakeRoundaboutIDM(road, {"we", "sx", 0}, -20.0, 16.0, "exr");
  MakeRoundaboutIDM(road, {"eer", "ees", 0}, 50.0, 16.0, "nxr");
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

int ResetTwoWayVehicles(Road* road) {
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
    road->vehicles.push_back(
        MakeTwoWayIDM(road->network, {"a", "b", 1}, 70.0 + 40.0 * i, 24.0));
  }
  for (int i = 0; i < 2; ++i) {
    road->vehicles.push_back(
        MakeTwoWayIDM(road->network, {"b", "a", 0}, 200.0 + 100.0 * i, 20.0));
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

int ResetUTurnVehicles(Road* road) {
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

  road->vehicles.push_back(
      MakeUTurnIDM(road->network, {"a", "b", 0}, 25.0, 13.5));
  road->vehicles.push_back(
      MakeUTurnIDM(road->network, {"a", "b", 1}, 56.0, 14.5));
  road->vehicles.push_back(
      MakeUTurnIDM(road->network, {"b", "c", 1}, 0.5, 4.5));
  road->vehicles.push_back(
      MakeUTurnIDM(road->network, {"b", "c", 0}, 17.5, 5.5));
  road->vehicles.push_back(
      MakeUTurnIDM(road->network, {"c", "d", 0}, 1.0, 3.5));
  road->vehicles.push_back(
      MakeUTurnIDM(road->network, {"c", "d", 1}, 30.0, 5.5));
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

int ResetExitVehicles(Road* road) {
  road->vehicles.clear();
  const LaneIndex ego_lane_index{"0", "1", 0};
  const Lane& ego_lane = road->network.GetLane(ego_lane_index);
  Vehicle ego = MakeMDPVehicle(road->network, ego_lane.Position(30.0, 0.0),
                               ego_lane.HeadingAt(30.0), 25.0, std::nullopt,
                               std::nullopt, {18.0, 24.0, 30.0});
  ego.lane_index = ego_lane_index;
  ego.target_lane_index = ego_lane_index;
  road->vehicles.push_back(ego);

  for (int i = 0; i < 20; ++i) {
    const int lane = 1 + (i % 5);
    const double longitudinal = 45.0 + 18.0 * static_cast<double>(i);
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

int ResetIntersectionVehicles(Road* road) {
  road->regulated_steps = 3 * 15;
  road->vehicles.clear();
  const LaneIndex ego_lane_index{"o0", "ir0", 0};
  const Lane& ego_lane = road->network.GetLane(ego_lane_index);
  Vehicle ego = MakeMDPVehicle(road->network, ego_lane.Position(65.0, 0.0),
                               ego_lane.HeadingAt(60.0), ego_lane.SpeedLimit(),
                               std::nullopt, std::nullopt, {0.0, 4.5, 9.0});
  ego.lane_index = ego_lane_index;
  ego.target_lane_index = ego_lane_index;
  ego.speed_index = 2;
  ego.target_speed = 9.0;
  PlanRouteTo(&ego, road->network, "o1");
  road->vehicles.push_back(ego);

  road->vehicles.push_back(
      MakeIntersectionIDM(road->network, 1, 32.0, 8.0, "o3"));
  road->vehicles.push_back(
      MakeIntersectionIDM(road->network, 2, 44.0, 8.5, "o0"));
  road->vehicles.push_back(
      MakeIntersectionIDM(road->network, 3, 56.0, 7.5, "o1"));
  road->vehicles.push_back(
      MakeIntersectionIDM(road->network, 0, 85.0, 8.0, "o2"));
  road->vehicles.push_back(
      MakeIntersectionIDM(road->network, 1, 74.0, 8.0, "o2"));
  return 0;
}

int ResetMultiAgentIntersectionVehicles(Road* road) {
  road->regulated_steps = 3 * 15;
  road->vehicles.clear();

  const LaneIndex ego0_lane_index{"o0", "ir0", 0};
  const Lane& ego0_lane = road->network.GetLane(ego0_lane_index);
  Vehicle ego0 =
      MakeMDPVehicle(road->network, ego0_lane.Position(65.0, 0.0),
                     ego0_lane.HeadingAt(60.0), ego0_lane.SpeedLimit());
  ego0.lane_index = ego0_lane_index;
  ego0.target_lane_index = ego0_lane_index;
  ego0.speed_index = 0;
  ego0.target_speed = 20.0;
  PlanRouteTo(&ego0, road->network, "o1");
  road->vehicles.push_back(ego0);

  const LaneIndex ego1_lane_index{"o1", "ir1", 0};
  const Lane& ego1_lane = road->network.GetLane(ego1_lane_index);
  Vehicle ego1 =
      MakeMDPVehicle(road->network, ego1_lane.Position(66.0, 0.0),
                     ego1_lane.HeadingAt(60.0), ego1_lane.SpeedLimit());
  ego1.lane_index = ego1_lane_index;
  ego1.target_lane_index = ego1_lane_index;
  ego1.speed_index = 0;
  ego1.target_speed = 20.0;
  PlanRouteTo(&ego1, road->network, "o1");
  road->vehicles.push_back(ego1);

  road->vehicles.push_back(
      MakeIntersectionIDM(road->network, 2, 55.0, 8.5, "o0"));
  road->vehicles.push_back(
      MakeIntersectionIDM(road->network, 3, 70.0, 7.5, "o1"));
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

int ResetRacetrackVehicles(Road* road, double longitudinal, int lane) {
  road->vehicles.clear();
  const LaneIndex lane_index{"a", "b", lane};
  const double speed_limit = road->network.GetLane(lane_index).SpeedLimit();
  Vehicle ego =
      MakeVehicleOnLane(road->network, lane_index, longitudinal, speed_limit);
  ego.kind = VehicleKind::kVehicle;
  road->vehicles.push_back(ego);
  return 0;
}

}  // namespace highway::official
