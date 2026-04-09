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

#include "envpool/highway/highway_env.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace highway {
namespace {

constexpr double kTauAcc = 0.6;
constexpr double kTauHeading = 0.2;
constexpr double kTauLateral = 0.6;
constexpr double kKpA = 1.0 / kTauAcc;
constexpr double kKpHeading = 1.0 / kTauHeading;
constexpr double kKpLateral = 1.0 / kTauLateral;
constexpr double kMaxSteeringAngle = kPi / 3.0;

constexpr double kIDMAccMax = 6.0;
constexpr double kIDMComfortAccMax = 3.0;
constexpr double kIDMComfortAccMin = -5.0;
constexpr double kIDMDistanceWanted = 5.0 + kVehicleLength;
constexpr double kIDMTimeWanted = 1.5;
constexpr double kLaneChangeMinAccGain = 0.2;
constexpr double kLaneChangeMaxBrakingImposed = 2.0;
constexpr double kLaneChangeDelay = 1.0;

double NotZero(double x) {
  constexpr double kEps = 1e-2;
  if (std::abs(x) > kEps) {
    return x;
  }
  return x >= 0.0 ? kEps : -kEps;
}

double WrapToPi(double x) {
  x = std::fmod(x + kPi, 2.0 * kPi);
  if (x < 0.0) {
    x += 2.0 * kPi;
  }
  return x - kPi;
}

double LMap(double v, double x0, double x1, double y0, double y1) {
  return y0 + (v - x0) * (y1 - y0) / (x1 - x0);
}

double Clip(double v, double low, double high) {
  return std::clamp(v, low, high);
}

int ClipInt(int v, int low, int high) { return std::clamp(v, low, high); }

double Norm2(double x, double y) { return std::sqrt(x * x + y * y); }

struct Color {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
};

struct Point {
  int x;
  int y;
};

struct Point2d {
  double x;
  double y;
};

Point ToPoint(double x, double y) {
  return {static_cast<int>(x), static_cast<int>(y)};
}

Point2d Rotate(const Point2d& p, double theta) {
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  return {p.x * c - p.y * s, p.x * s + p.y * c};
}

void FillFrame(unsigned char* rgb, int width, int height, Color color) {
  for (int i = 0; i < width * height; ++i) {
    rgb[3 * i + 0] = color.r;
    rgb[3 * i + 1] = color.g;
    rgb[3 * i + 2] = color.b;
  }
}

void SetPixel(unsigned char* rgb, int width, int height, int x, int y,
              Color color) {
  if (x < 0 || width <= x || y < 0 || height <= y) {
    return;
  }
  const int offset = 3 * (y * width + x);
  rgb[offset + 0] = color.r;
  rgb[offset + 1] = color.g;
  rgb[offset + 2] = color.b;
}

int Pix(double length, double scaling) {
  return static_cast<int>(length * scaling);
}

Color Lighten(Color color, double ratio = 0.68) {
  return {static_cast<std::uint8_t>(
              std::min(static_cast<int>(color.r / ratio), 255)),
          static_cast<std::uint8_t>(
              std::min(static_cast<int>(color.g / ratio), 255)),
          static_cast<std::uint8_t>(
              std::min(static_cast<int>(color.b / ratio), 255))};
}

void FillRect(unsigned char* rgb, int width, int height, int x, int y, int w,
              int h, Color color) {
  for (int yy = std::max(y, 0); yy < std::min(y + h, height); ++yy) {
    for (int xx = std::max(x, 0); xx < std::min(x + w, width); ++xx) {
      SetPixel(rgb, width, height, xx, yy, color);
    }
  }
}

void DrawRectOutline(unsigned char* rgb, int width, int height, int x, int y,
                     int w, int h, Color color) {
  if (w <= 0 || h <= 0) {
    return;
  }
  FillRect(rgb, width, height, x, y, w, 1, color);
  FillRect(rgb, width, height, x, y + h - 1, w, 1, color);
  FillRect(rgb, width, height, x, y, 1, h, color);
  FillRect(rgb, width, height, x + w - 1, y, 1, h, color);
}

void DrawLine(unsigned char* rgb, int width, int height, Point start, Point end,
              Color color, int thickness) {
  const int dx = end.x - start.x;
  const int dy = end.y - start.y;
  const int steps = std::max(std::abs(dx), std::abs(dy));
  const int radius = std::max(0, thickness / 2);
  if (steps == 0) {
    SetPixel(rgb, width, height, start.x, start.y, color);
    return;
  }
  for (int i = 0; i <= steps; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(steps);
    const int x = static_cast<int>(std::lround(start.x + t * dx));
    const int y = static_cast<int>(std::lround(start.y + t * dy));
    for (int oy = -radius; oy <= radius; ++oy) {
      for (int ox = -radius; ox <= radius; ++ox) {
        SetPixel(rgb, width, height, x + ox, y + oy, color);
      }
    }
  }
}

void DrawHorizontalLine(unsigned char* rgb, int width, int height, Point start,
                        Point end, Color color, int thickness) {
  if (start.x > end.x) {
    std::swap(start, end);
  }
  if (thickness <= 1) {
    FillRect(rgb, width, height, start.x, start.y, end.x - start.x + 1, 1,
             color);
    return;
  }
  FillRect(rgb, width, height, start.x, start.y - thickness / 2,
           end.x - start.x + 1, thickness, color);
}

double Edge(const Point& a, const Point& b, double x, double y) {
  return (x - a.x) * (b.y - a.y) - (y - a.y) * (b.x - a.x);
}

bool PointInConvexPolygon(const std::vector<Point>& poly, double x, double y) {
  if (poly.size() < 3) {
    return false;
  }
  double last = 0.0;
  for (std::size_t i = 0; i < poly.size(); ++i) {
    const double edge = Edge(poly[i], poly[(i + 1) % poly.size()], x, y);
    if (std::abs(edge) < 1e-9) {
      continue;
    }
    if (last == 0.0) {
      last = edge;
    } else if ((last < 0.0) != (edge < 0.0)) {
      return false;
    }
  }
  return true;
}

void FillConvexPolygon(unsigned char* rgb, int width, int height,
                       const std::vector<Point>& poly, Color color) {
  if (poly.empty()) {
    return;
  }
  int min_x = poly.front().x;
  int max_x = poly.front().x;
  int min_y = poly.front().y;
  int max_y = poly.front().y;
  for (const auto& p : poly) {
    min_x = std::min(min_x, p.x);
    max_x = std::max(max_x, p.x);
    min_y = std::min(min_y, p.y);
    max_y = std::max(max_y, p.y);
  }
  min_x = std::max(min_x, 0);
  min_y = std::max(min_y, 0);
  max_x = std::min(max_x, width - 1);
  max_y = std::min(max_y, height - 1);
  for (int y = min_y; y <= max_y; ++y) {
    for (int x = min_x; x <= max_x; ++x) {
      if (PointInConvexPolygon(poly, static_cast<double>(x) + 0.5,
                               static_cast<double>(y) + 0.5)) {
        SetPixel(rgb, width, height, x, y, color);
      }
    }
  }
}

void DrawPolygonOutline(unsigned char* rgb, int width, int height,
                        const std::vector<Point>& poly, Color color) {
  for (std::size_t i = 0; i < poly.size(); ++i) {
    DrawLine(rgb, width, height, poly[i], poly[(i + 1) % poly.size()], color,
             1);
  }
}

}  // namespace

HighwayEnv::HighwayEnv(const Spec& spec, int env_id)
    : Env<HighwayEnvSpec>(spec, env_id),
      lanes_count_(spec.config["lanes_count"_]),
      traffic_vehicle_count_(spec.config["vehicles_count"_]),
      obs_vehicle_count_(spec.config["observation_vehicles_count"_]),
      initial_lane_id_(spec.config["initial_lane_id"_]),
      max_episode_steps_(
          std::max(1, static_cast<int>(spec.config["duration"_]) *
                          static_cast<int>(spec.config["policy_frequency"_]))),
      simulation_frequency_(spec.config["simulation_frequency"_]),
      policy_frequency_(spec.config["policy_frequency"_]),
      ego_spacing_(spec.config["ego_spacing"_]),
      vehicles_density_(spec.config["vehicles_density"_]),
      collision_reward_(spec.config["collision_reward"_]),
      right_lane_reward_(spec.config["right_lane_reward"_]),
      high_speed_reward_(spec.config["high_speed_reward"_]),
      reward_speed_low_(spec.config["reward_speed_low"_]),
      reward_speed_high_(spec.config["reward_speed_high"_]),
      normalize_reward_(spec.config["normalize_reward"_]),
      offroad_terminal_(spec.config["offroad_terminal"_]),
      other_vehicles_check_collisions_(
          spec.config["other_vehicles_check_collisions"_]),
      uniform01_(0.0, 1.0) {
  if (lanes_count_ < 1) {
    throw std::invalid_argument("lanes_count must be positive");
  }
  if (policy_frequency_ < 1 || simulation_frequency_ < policy_frequency_) {
    throw std::invalid_argument(
        "simulation_frequency must be >= positive policy_frequency");
  }
}

bool HighwayEnv::IsDone() { return done_; }

int HighwayEnv::CurrentMaxEpisodeSteps() const { return max_episode_steps_; }

double HighwayEnv::Uniform(double low, double high) {
  return low + (high - low) * uniform01_(gen_);
}

int HighwayEnv::RandomLane() {
  std::uniform_int_distribution<int> dist(0, lanes_count_ - 1);
  return dist(gen_);
}

int HighwayEnv::ClosestLane(double y) const {
  return ClipInt(static_cast<int>(std::lround(y / kLaneWidth)), 0,
                 lanes_count_ - 1);
}

double HighwayEnv::LaneCenter(int lane_index) const {
  return static_cast<double>(lane_index) * kLaneWidth;
}

std::pair<double, double> HighwayEnv::LaneCoordinates(const Vehicle& vehicle,
                                                      int lane_index) const {
  return {vehicle.x, vehicle.y - LaneCenter(lane_index)};
}

bool HighwayEnv::LaneIsReachableFrom(const Vehicle& vehicle,
                                     int lane_index) const {
  const auto [longitudinal, lateral] = LaneCoordinates(vehicle, lane_index);
  return std::abs(lateral) <= 2.0 * kLaneWidth && 0.0 <= longitudinal &&
         longitudinal < kLaneLength + kVehicleLength;
}

bool HighwayEnv::LaneOnRoad(const Vehicle& vehicle) const {
  const auto [longitudinal, lateral] =
      LaneCoordinates(vehicle, vehicle.lane_index);
  return std::abs(lateral) <= kLaneWidth / 2.0 &&
         -kVehicleLength <= longitudinal &&
         longitudinal < kLaneLength + kVehicleLength;
}

void HighwayEnv::UpdateVehicleLane(Vehicle* vehicle) {
  vehicle->lane_index = ClosestLane(vehicle->y);
}

void HighwayEnv::ResetRoad() { vehicles_.clear(); }

Vehicle HighwayEnv::CreateRandomVehicle(VehicleKind kind,
                                        std::optional<double> speed,
                                        int lane_id, double spacing) {
  Vehicle vehicle;
  vehicle.kind = kind;
  vehicle.lane_index =
      lane_id >= 0 ? ClipInt(lane_id, 0, lanes_count_ - 1) : RandomLane();
  if (!speed.has_value()) {
    speed = Uniform(0.7 * 30.0, 0.8 * 30.0);
  }
  vehicle.speed = *speed;
  vehicle.target_speed = *speed;
  vehicle.target_lane_index = vehicle.lane_index;

  const double default_spacing = 12.0 + vehicle.speed;
  const double offset =
      spacing * default_spacing * std::exp(-5.0 / 40.0 * lanes_count_);
  double max_x = 0.0;
  if (vehicles_.empty()) {
    max_x = 3.0 * offset;
  } else {
    max_x = vehicles_.front().x;
    for (const auto& other : vehicles_) {
      max_x = std::max(max_x, other.x);
    }
  }
  vehicle.x = max_x + offset * Uniform(0.9, 1.1);
  vehicle.y = LaneCenter(vehicle.lane_index);
  vehicle.timer = std::fmod((vehicle.x + vehicle.y) * kPi, kLaneChangeDelay);

  if (kind == VehicleKind::kMDP) {
    vehicle.speed_index = SpeedToIndex(vehicle, vehicle.target_speed);
    vehicle.target_speed = IndexToSpeed(vehicle, vehicle.speed_index);
  } else {
    vehicle.idm_delta = Uniform(3.5, 4.5);
    vehicle.check_collisions = other_vehicles_check_collisions_;
  }
  return vehicle;
}

void HighwayEnv::CreateVehicles() {
  const int ego_lane = initial_lane_id_;
  Vehicle ego =
      CreateRandomVehicle(VehicleKind::kMDP, 25.0, ego_lane, ego_spacing_);
  vehicles_.push_back(ego);

  const double traffic_spacing = 1.0 / std::max(vehicles_density_, 1e-6);
  for (int i = 0; i < traffic_vehicle_count_; ++i) {
    vehicles_.push_back(CreateRandomVehicle(VehicleKind::kIDM, std::nullopt, -1,
                                            traffic_spacing));
  }
}

void HighwayEnv::Reset() {
  elapsed_step_ = 0;
  time_ = 0.0;
  done_ = false;
  ResetRoad();
  CreateVehicles();
  WriteState(0.0f);
}

void HighwayEnv::ApplyMetaAction(int action) {
  static const std::array<std::string, 5> kActions = {
      "LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"};
  action = ClipInt(action, 0, static_cast<int>(kActions.size()) - 1);
  ActMDP(&vehicles_[0], kActions[action]);
}

void HighwayEnv::Step(const Action& action) {
  const int act = action["action"_];
  ++elapsed_step_;
  time_ += 1.0 / static_cast<double>(policy_frequency_);
  const int frames = simulation_frequency_ / policy_frequency_;
  const double dt = 1.0 / static_cast<double>(simulation_frequency_);

  for (int frame = 0; frame < frames; ++frame) {
    if (frame == 0) {
      ApplyMetaAction(act);
    }
    RoadAct();
    RoadStep(dt);
  }

  done_ = vehicles_[0].crashed || elapsed_step_ >= max_episode_steps_ ||
          (offroad_terminal_ && !EgoOnRoad());
  WriteState(static_cast<float>(Reward(act)));
}

int HighwayEnv::SpeedToIndex(const Vehicle& vehicle, double speed) const {
  const double x =
      (speed - vehicle.target_speeds.front()) /
      (vehicle.target_speeds.back() - vehicle.target_speeds.front());
  const auto index = static_cast<int>(
      std::lround(x * static_cast<double>(vehicle.target_speeds.size() - 1)));
  return ClipInt(index, 0, static_cast<int>(vehicle.target_speeds.size()) - 1);
}

double HighwayEnv::IndexToSpeed(const Vehicle& vehicle, int index) const {
  index = ClipInt(index, 0, static_cast<int>(vehicle.target_speeds.size()) - 1);
  return vehicle.target_speeds[index];
}

void HighwayEnv::ActMDP(Vehicle* vehicle, std::optional<std::string> action) {
  if (action == "FASTER") {
    vehicle->speed_index = SpeedToIndex(*vehicle, vehicle->speed) + 1;
    vehicle->speed_index =
        ClipInt(vehicle->speed_index, 0,
                static_cast<int>(vehicle->target_speeds.size()) - 1);
    vehicle->target_speed = IndexToSpeed(*vehicle, vehicle->speed_index);
    ActControlled(vehicle, std::nullopt);
    return;
  }
  if (action == "SLOWER") {
    vehicle->speed_index = SpeedToIndex(*vehicle, vehicle->speed) - 1;
    vehicle->speed_index =
        ClipInt(vehicle->speed_index, 0,
                static_cast<int>(vehicle->target_speeds.size()) - 1);
    vehicle->target_speed = IndexToSpeed(*vehicle, vehicle->speed_index);
    ActControlled(vehicle, std::nullopt);
    return;
  }
  ActControlled(vehicle, std::move(action));
}

void HighwayEnv::ActControlled(Vehicle* vehicle,
                               std::optional<std::string> action) {
  if (action == "LANE_RIGHT") {
    const int lane =
        ClipInt(vehicle->target_lane_index + 1, 0, lanes_count_ - 1);
    if (LaneIsReachableFrom(*vehicle, lane)) {
      vehicle->target_lane_index = lane;
    }
  } else if (action == "LANE_LEFT") {
    const int lane =
        ClipInt(vehicle->target_lane_index - 1, 0, lanes_count_ - 1);
    if (LaneIsReachableFrom(*vehicle, lane)) {
      vehicle->target_lane_index = lane;
    }
  }
  vehicle->steering = SteeringControl(*vehicle, vehicle->target_lane_index);
  vehicle->acceleration = SpeedControl(*vehicle, vehicle->target_speed);
}

void HighwayEnv::RoadAct() {
  for (std::size_t i = 0; i < vehicles_.size(); ++i) {
    if (vehicles_[i].kind == VehicleKind::kMDP) {
      ActMDP(&vehicles_[i], std::nullopt);
    } else {
      ActIDM(static_cast<int>(i));
    }
  }
}

double HighwayEnv::SteeringControl(const Vehicle& vehicle,
                                   int target_lane_index) const {
  const auto [longitudinal, lateral] =
      LaneCoordinates(vehicle, target_lane_index);
  const double lane_future_heading = 0.0;
  (void)longitudinal;

  const double lateral_speed_command = -kKpLateral * lateral;
  const double heading_command = std::asin(
      Clip(lateral_speed_command / NotZero(vehicle.speed), -1.0, 1.0));
  const double heading_ref =
      lane_future_heading + Clip(heading_command, -kPi / 4.0, kPi / 4.0);
  const double heading_rate_command =
      kKpHeading * WrapToPi(heading_ref - vehicle.heading);
  const double slip_angle = std::asin(
      Clip(kVehicleLength / 2.0 / NotZero(vehicle.speed) * heading_rate_command,
           -1.0, 1.0));
  const double steering_angle = std::atan(2.0 * std::tan(slip_angle));
  return Clip(steering_angle, -kMaxSteeringAngle, kMaxSteeringAngle);
}

double HighwayEnv::SpeedControl(const Vehicle& vehicle,
                                double target_speed) const {
  return kKpA * (target_speed - vehicle.speed);
}

void HighwayEnv::ActIDM(int vehicle_index) {
  Vehicle& vehicle = vehicles_[vehicle_index];
  if (vehicle.crashed) {
    return;
  }
  ChangeLanePolicy(vehicle_index);
  vehicle.steering = SteeringControl(vehicle, vehicle.target_lane_index);

  auto [front_index, rear_index] =
      NeighbourVehicles(vehicle, vehicle.lane_index);
  (void)rear_index;
  const Vehicle* front = front_index >= 0
                             ? &vehicles_[front_index]
                             : static_cast<const Vehicle*>(nullptr);
  double acc = IDMAcceleration(vehicle_index, &vehicle, front);
  if (vehicle.lane_index != vehicle.target_lane_index) {
    auto [target_front_index, target_rear_index] =
        NeighbourVehicles(vehicle, vehicle.target_lane_index);
    (void)target_rear_index;
    const Vehicle* target_front =
        target_front_index >= 0 ? &vehicles_[target_front_index] : nullptr;
    acc = std::min(acc, IDMAcceleration(vehicle_index, &vehicle, target_front));
  }
  vehicle.acceleration = Clip(acc, -kIDMAccMax, kIDMAccMax);
}

void HighwayEnv::StepVehicle(Vehicle* vehicle, double dt) {
  if (vehicle->crashed) {
    vehicle->steering = 0.0;
    vehicle->acceleration = -vehicle->speed;
  }
  vehicle->steering = static_cast<double>(vehicle->steering);
  vehicle->acceleration = static_cast<double>(vehicle->acceleration);
  if (vehicle->speed > kVehicleMaxSpeed) {
    vehicle->acceleration =
        std::min(vehicle->acceleration, kVehicleMaxSpeed - vehicle->speed);
  } else if (vehicle->speed < kVehicleMinSpeed) {
    vehicle->acceleration =
        std::max(vehicle->acceleration, kVehicleMinSpeed - vehicle->speed);
  }

  const double beta = std::atan(0.5 * std::tan(vehicle->steering));
  vehicle->x += vehicle->speed * std::cos(vehicle->heading + beta) * dt;
  vehicle->y += vehicle->speed * std::sin(vehicle->heading + beta) * dt;
  vehicle->heading +=
      vehicle->speed * std::sin(beta) / (kVehicleLength / 2.0) * dt;
  vehicle->speed += vehicle->acceleration * dt;
  if (vehicle->kind == VehicleKind::kIDM) {
    vehicle->timer += dt;
  }
  UpdateVehicleLane(vehicle);
}

void HighwayEnv::RoadStep(double dt) {
  for (auto& vehicle : vehicles_) {
    StepVehicle(&vehicle, dt);
  }
  CheckCollisions();
}

void HighwayEnv::CheckCollisions() {
  for (std::size_t i = 0; i < vehicles_.size(); ++i) {
    for (std::size_t j = i + 1; j < vehicles_.size(); ++j) {
      if (!(vehicles_[i].check_collisions || vehicles_[j].check_collisions)) {
        continue;
      }
      if (RectanglesIntersect(vehicles_[i], vehicles_[j])) {
        vehicles_[i].crashed = true;
        vehicles_[j].crashed = true;
      }
    }
  }
}

std::pair<int, int> HighwayEnv::NeighbourVehicles(const Vehicle& vehicle,
                                                  int lane_index) const {
  const double s = vehicle.x;
  std::optional<double> s_front;
  std::optional<double> s_rear;
  int v_front = -1;
  int v_rear = -1;
  for (int i = 0; i < static_cast<int>(vehicles_.size()); ++i) {
    const Vehicle& other = vehicles_[i];
    if (&other == &vehicle) {
      continue;
    }
    const auto [s_v, lat_v] = LaneCoordinates(other, lane_index);
    if (!(std::abs(lat_v) <= kLaneWidth / 2.0 + 1.0 && -kVehicleLength <= s_v &&
          s_v < kLaneLength + kVehicleLength)) {
      continue;
    }
    if (s <= s_v && (!s_front.has_value() || s_v <= *s_front)) {
      s_front = s_v;
      v_front = i;
    }
    if (s_v < s && (!s_rear.has_value() || s_v > *s_rear)) {
      s_rear = s_v;
      v_rear = i;
    }
  }
  return {v_front, v_rear};
}

double HighwayEnv::LaneDistanceTo(const Vehicle& self,
                                  const Vehicle& other) const {
  return other.x - self.x;
}

double HighwayEnv::IDMDistanceTo(const Vehicle& ego,
                                 const Vehicle& front) const {
  return LaneDistanceTo(ego, front);
}

double HighwayEnv::IDMAcceleration(int idm_index, const Vehicle* ego,
                                   const Vehicle* front) const {
  if (ego == nullptr) {
    return 0.0;
  }
  const Vehicle& idm = vehicles_[idm_index];
  double ego_target_speed = Clip(ego->target_speed, 0.0, 30.0);
  double acc = kIDMComfortAccMax *
               (1.0 - std::pow(std::max(ego->speed, 0.0) /
                                   std::abs(NotZero(ego_target_speed)),
                               idm.idm_delta));
  if (front != nullptr) {
    const double d = IDMDistanceTo(*ego, *front);
    acc -= kIDMComfortAccMax *
           std::pow(DesiredGap(*ego, *front) / NotZero(d), 2.0);
  }
  return acc;
}

double HighwayEnv::DesiredGap(const Vehicle& ego, const Vehicle& front) const {
  const double dvx = ego.vx() - front.vx();
  const double dvy = ego.vy() - front.vy();
  const double dv = dvx * std::cos(ego.heading) + dvy * std::sin(ego.heading);
  const double ab = -kIDMComfortAccMax * kIDMComfortAccMin;
  return kIDMDistanceWanted + ego.speed * kIDMTimeWanted +
         ego.speed * dv / (2.0 * std::sqrt(ab));
}

void HighwayEnv::ChangeLanePolicy(int vehicle_index) {
  Vehicle& vehicle = vehicles_[vehicle_index];
  if (vehicle.lane_index != vehicle.target_lane_index) {
    return;
  }
  if (!(kLaneChangeDelay < vehicle.timer)) {
    return;
  }
  vehicle.timer = 0.0;

  const std::array<int, 2> candidate_lanes = {vehicle.lane_index - 1,
                                              vehicle.lane_index + 1};
  for (int lane : candidate_lanes) {
    if (lane < 0 || lane >= lanes_count_) {
      continue;
    }
    if (!LaneIsReachableFrom(vehicle, lane) || std::abs(vehicle.speed) < 1.0) {
      continue;
    }
    if (Mobil(vehicle_index, lane)) {
      vehicle.target_lane_index = lane;
      return;
    }
  }
}

bool HighwayEnv::Mobil(int vehicle_index, int lane_index) const {
  const Vehicle& self = vehicles_[vehicle_index];
  auto [new_preceding_index, new_following_index] =
      NeighbourVehicles(self, lane_index);
  const Vehicle* new_preceding =
      new_preceding_index >= 0 ? &vehicles_[new_preceding_index] : nullptr;
  const Vehicle* new_following =
      new_following_index >= 0 ? &vehicles_[new_following_index] : nullptr;

  const double new_following_pred_a =
      IDMAcceleration(vehicle_index, new_following, &self);
  if (new_following_pred_a < -kLaneChangeMaxBrakingImposed) {
    return false;
  }

  auto [old_preceding_index, old_following_index] =
      NeighbourVehicles(self, self.lane_index);
  const Vehicle* old_preceding =
      old_preceding_index >= 0 ? &vehicles_[old_preceding_index] : nullptr;
  const Vehicle* old_following =
      old_following_index >= 0 ? &vehicles_[old_following_index] : nullptr;

  const double self_pred_a =
      IDMAcceleration(vehicle_index, &self, new_preceding);
  const double self_a = IDMAcceleration(vehicle_index, &self, old_preceding);
  const double new_following_a =
      IDMAcceleration(vehicle_index, new_following, new_preceding);
  const double old_following_a =
      IDMAcceleration(vehicle_index, old_following, &self);
  const double old_following_pred_a =
      IDMAcceleration(vehicle_index, old_following, old_preceding);

  const double politeness = 0.0;
  const double jerk = self_pred_a - self_a +
                      politeness * (new_following_pred_a - new_following_a +
                                    old_following_pred_a - old_following_a);
  return jerk >= kLaneChangeMinAccGain;
}

bool HighwayEnv::RectanglesIntersect(const Vehicle& a, const Vehicle& b) const {
  if (Norm2(a.x - b.x, a.y - b.y) > std::sqrt(kVehicleLength * kVehicleLength +
                                              kVehicleWidth * kVehicleWidth)) {
    return false;
  }
  const double dx = std::abs(a.x - b.x);
  const double dy = std::abs(a.y - b.y);
  // Highway vehicles are almost axis-aligned; inflate laterally to remain
  // conservative during lane changes.
  return dx < kVehicleLength && dy < kVehicleWidth * 1.25;
}

bool HighwayEnv::EgoOnRoad() const { return LaneOnRoad(vehicles_[0]); }

double HighwayEnv::Reward(int action) const {
  (void)action;
  const Vehicle& ego = vehicles_[0];
  const double lane = static_cast<double>(ego.target_lane_index);
  const double forward_speed = ego.speed * std::cos(ego.heading);
  const double scaled_speed =
      LMap(forward_speed, reward_speed_low_, reward_speed_high_, 0.0, 1.0);
  double reward = collision_reward_ * static_cast<double>(ego.crashed) +
                  right_lane_reward_ * lane / std::max(lanes_count_ - 1, 1) +
                  high_speed_reward_ * Clip(scaled_speed, 0.0, 1.0);
  if (normalize_reward_) {
    reward = LMap(reward, collision_reward_,
                  high_speed_reward_ + right_lane_reward_, 0.0, 1.0);
  }
  return reward * static_cast<double>(EgoOnRoad());
}

void HighwayEnv::WriteState(float reward) {
  auto state = Allocate();
  const Vehicle& ego = vehicles_[0];
  auto obs = state["obs"_];

  auto write_row = [&](int row, double presence, double x, double y, double vx,
                       double vy) {
    x = Clip(LMap(x, -kPerceptionDistance, kPerceptionDistance, -1.0, 1.0),
             -1.0, 1.0);
    y = Clip(LMap(y, -kLaneWidth * lanes_count_, kLaneWidth * lanes_count_,
                  -1.0, 1.0),
             -1.0, 1.0);
    vx = Clip(
        LMap(vx, -2.0 * kVehicleMaxSpeed, 2.0 * kVehicleMaxSpeed, -1.0, 1.0),
        -1.0, 1.0);
    vy = Clip(
        LMap(vy, -2.0 * kVehicleMaxSpeed, 2.0 * kVehicleMaxSpeed, -1.0, 1.0),
        -1.0, 1.0);
    obs(row, 0) = static_cast<float>(presence);
    obs(row, 1) = static_cast<float>(x);
    obs(row, 2) = static_cast<float>(y);
    obs(row, 3) = static_cast<float>(vx);
    obs(row, 4) = static_cast<float>(vy);
  };

  write_row(0, 1.0, ego.x, ego.y, ego.vx(), ego.vy());

  std::vector<int> close;
  close.reserve(vehicles_.size());
  for (int i = 1; i < static_cast<int>(vehicles_.size()); ++i) {
    const Vehicle& other = vehicles_[i];
    if (Norm2(other.x - ego.x, other.y - ego.y) < kPerceptionDistance &&
        LaneDistanceTo(ego, other) > -2.0 * kVehicleLength) {
      close.push_back(i);
    }
  }
  std::sort(close.begin(), close.end(), [&](int lhs, int rhs) {
    return std::abs(LaneDistanceTo(ego, vehicles_[lhs])) <
           std::abs(LaneDistanceTo(ego, vehicles_[rhs]));
  });
  if (static_cast<int>(close.size()) > obs_vehicle_count_ - 1) {
    close.resize(obs_vehicle_count_ - 1);
  }

  int row = 1;
  for (int index : close) {
    const Vehicle& other = vehicles_[index];
    write_row(row++, 1.0, other.x - ego.x, other.y - ego.y,
              other.vx() - ego.vx(), other.vy() - ego.vy());
  }
  for (; row < obs_vehicle_count_; ++row) {
    write_row(row, 0.0, 0.0, 0.0, 0.0, 0.0);
  }

  state["reward"_] = reward;
  state["info:speed"_] = static_cast<float>(ego.speed);
  state["info:crashed"_] = ego.crashed;
}

std::pair<int, int> HighwayEnv::RenderSize(int width, int height) const {
  const int default_width = spec_.config["screen_width"_];
  const int default_height = spec_.config["screen_height"_];
  return {width > 0 ? width : default_width,
          height > 0 ? height : default_height};
}

void HighwayEnv::Render(int width, int height, int /*camera_id*/,
                        unsigned char* rgb) {
  FillFrame(rgb, width, height, {100, 100, 100});
  const Vehicle& ego = vehicles_[0];
  const double scaling = spec_.config["scaling"_];
  const double center_x = spec_.config["centering_position_x"_];
  const double center_y = spec_.config["centering_position_y"_];
  const double origin_x =
      ego.x - center_x * static_cast<double>(width) / scaling;
  const double origin_y =
      ego.y - center_y * static_cast<double>(height) / scaling;

  auto pos2pix = [&](double x, double y) {
    return ToPoint((x - origin_x) * scaling, (y - origin_y) * scaling);
  };

  const double stripe_spacing = 4.33;
  const double stripe_length = 3.0;
  const double stripe_width = 0.3;
  const double road_length = 10000.0;
  const int line_thickness = std::max(1, Pix(stripe_width, scaling));
  const int stripes_count =
      static_cast<int>(2.0 * (height + width) / (stripe_spacing * scaling));
  const double s0 =
      (std::floor(static_cast<double>(static_cast<int>(origin_x)) /
                  stripe_spacing) -
       static_cast<double>(stripes_count / 2)) *
      stripe_spacing;

  auto draw_line_segment = [&](double x0, double x1, double y,
                               bool continuous) {
    x0 = Clip(x0, 0.0, road_length);
    x1 = Clip(x1, 0.0, road_length);
    if (continuous) {
      DrawHorizontalLine(rgb, width, height, pos2pix(x0, y), pos2pix(x1, y),
                         {255, 255, 255}, line_thickness);
      return;
    }
    for (int k = 0; k < stripes_count; ++k) {
      const double start =
          Clip(x0 + static_cast<double>(k) * stripe_spacing, 0.0, road_length);
      const double end = Clip(start + stripe_length, 0.0, road_length);
      if (std::abs(start - end) > 0.5 * stripe_length) {
        DrawHorizontalLine(rgb, width, height, pos2pix(start, y),
                           pos2pix(end, y), {255, 255, 255}, line_thickness);
      }
    }
  };

  for (int lane = 0; lane < lanes_count_; ++lane) {
    const double left_y = LaneCenter(lane) - kLaneWidth / 2.0;
    const double right_y = LaneCenter(lane) + kLaneWidth / 2.0;
    draw_line_segment(s0, s0 + stripes_count * stripe_spacing + stripe_length,
                      left_y, lane == 0);
    if (lane == lanes_count_ - 1) {
      draw_line_segment(s0, s0 + stripes_count * stripe_spacing + stripe_length,
                        right_y, true);
    } else {
      draw_line_segment(s0, s0 + stripes_count * stripe_spacing + stripe_length,
                        right_y, false);
    }
  }

  auto draw_vehicle = [&](const Vehicle& v) {
    const Point center = pos2pix(v.x, v.y);
    const double tire_length = 1.0;
    const double headlight_length = 0.72;
    const double headlight_width = 0.6;
    const double sprite_length = kVehicleLength + 2.0 * tire_length;
    const int sprite_px = Pix(sprite_length, scaling);
    const int sprite_left =
        static_cast<int>(center.x - static_cast<double>(sprite_px) / 2.0);
    const int sprite_top =
        static_cast<int>(center.y - static_cast<double>(sprite_px) / 2.0);
    const Color fill =
        v.crashed ? Color{255, 100, 100}
                  : (v.kind == VehicleKind::kMDP ? Color{50, 200, 0}
                                                 : Color{100, 200, 255});

    const int body_x = Pix(tire_length, scaling);
    const int body_y = Pix(sprite_length / 2.0 - kVehicleWidth / 2.0, scaling);
    const int body_w = Pix(kVehicleLength, scaling);
    const int body_h = Pix(kVehicleWidth, scaling);
    const int light_x =
        Pix(tire_length + kVehicleLength - headlight_length, scaling);
    const int light_left_y =
        Pix(sprite_length / 2.0 - (1.4 * kVehicleWidth) / 3.0, scaling);
    const int light_right_y =
        Pix(sprite_length / 2.0 + (0.6 * kVehicleWidth) / 5.0, scaling);
    const int light_w = Pix(headlight_length, scaling);
    const int light_h = Pix(headlight_width, scaling);
    const double heading = std::abs(v.heading) > 2.0 * kPi / 180.0
                               ? static_cast<double>(v.heading)
                               : 0.0;

    auto blit_rect = [&](int local_x, int local_y, int local_w, int local_h,
                         Color color) {
      FillRect(rgb, width, height, sprite_left + local_x, sprite_top + local_y,
               local_w, local_h, color);
    };

    auto draw_rotated_rect = [&](int local_x, int local_y, int local_w,
                                 int local_h, Color color, bool outline) {
      const std::vector<Point2d> local = {
          {static_cast<double>(local_x) - static_cast<double>(sprite_px) / 2.0,
           static_cast<double>(local_y) - static_cast<double>(sprite_px) / 2.0},
          {static_cast<double>(local_x + local_w) -
               static_cast<double>(sprite_px) / 2.0,
           static_cast<double>(local_y) - static_cast<double>(sprite_px) / 2.0},
          {static_cast<double>(local_x + local_w) -
               static_cast<double>(sprite_px) / 2.0,
           static_cast<double>(local_y + local_h) -
               static_cast<double>(sprite_px) / 2.0},
          {static_cast<double>(local_x) - static_cast<double>(sprite_px) / 2.0,
           static_cast<double>(local_y + local_h) -
               static_cast<double>(sprite_px) / 2.0},
      };
      std::vector<Point> poly;
      poly.reserve(local.size());
      for (const auto& p : local) {
        Point2d r = Rotate(p, heading);
        poly.push_back({center.x + static_cast<int>(std::lround(r.x)),
                        center.y + static_cast<int>(std::lround(r.y))});
      }
      if (outline) {
        DrawPolygonOutline(rgb, width, height, poly, color);
      } else {
        FillConvexPolygon(rgb, width, height, poly, color);
      }
    };

    if (heading == 0.0) {
      blit_rect(body_x, body_y, body_w, body_h, fill);
      blit_rect(light_x, light_left_y, light_w, light_h, Lighten(fill));
      blit_rect(light_x, light_right_y, light_w, light_h, Lighten(fill));
      DrawRectOutline(rgb, width, height, sprite_left + body_x,
                      sprite_top + body_y, body_w, body_h, {60, 60, 60});
      return;
    }

    draw_rotated_rect(body_x, body_y, body_w, body_h, fill, false);
    draw_rotated_rect(light_x, light_left_y, light_w, light_h, Lighten(fill),
                      false);
    draw_rotated_rect(light_x, light_right_y, light_w, light_h, Lighten(fill),
                      false);
    draw_rotated_rect(body_x, body_y, body_w, body_h, {60, 60, 60}, true);
  };
  for (const auto& v : vehicles_) {
    const Point pix = pos2pix(v.x, v.y);
    if (-80 <= pix.x && pix.x <= width + 80 && -80 <= pix.y &&
        pix.y <= height + 80) {
      draw_vehicle(v);
    }
  }
}

HighwayDebugState HighwayEnv::DebugState() const {
  HighwayDebugState state;
  state.lanes_count = lanes_count_;
  state.simulation_frequency = simulation_frequency_;
  state.policy_frequency = policy_frequency_;
  state.elapsed_step = elapsed_step_;
  state.time = time_;
  state.vehicles.reserve(vehicles_.size());
  for (const auto& v : vehicles_) {
    HighwayVehicleDebugState vehicle;
    vehicle.kind = static_cast<int>(v.kind);
    vehicle.lane_index = v.lane_index;
    vehicle.target_lane_index = v.target_lane_index;
    vehicle.speed_index = v.speed_index;
    vehicle.x = v.x;
    vehicle.y = v.y;
    vehicle.heading = v.heading;
    vehicle.speed = v.speed;
    vehicle.target_speed = v.target_speed;
    vehicle.target_speed0 = v.target_speeds[0];
    vehicle.target_speed1 = v.target_speeds[1];
    vehicle.target_speed2 = v.target_speeds[2];
    vehicle.idm_delta = v.idm_delta;
    vehicle.timer = v.timer;
    vehicle.crashed = v.crashed;
    vehicle.on_road = LaneOnRoad(v);
    vehicle.check_collisions = v.check_collisions;
    state.vehicles.push_back(vehicle);
  }
  return state;
}

std::vector<HighwayDebugState> HighwayEnvPool::DebugStates(
    const std::vector<int>& env_ids) const {
  std::vector<HighwayDebugState> states;
  states.reserve(env_ids.size());
  for (int env_id : env_ids) {
    states.emplace_back(envs_[env_id]->DebugState());
  }
  return states;
}

}  // namespace highway
