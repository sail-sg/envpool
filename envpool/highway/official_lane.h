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
//
// Native port of highway_env.road.lane primitives.

#ifndef ENVPOOL_HIGHWAY_OFFICIAL_LANE_H_
#define ENVPOOL_HIGHWAY_OFFICIAL_LANE_H_

#include <array>
#include <cmath>

namespace highway {
namespace official {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDefaultLaneWidth = 4.0;
constexpr double kLaneVehicleLength = 5.0;

struct Vec2 {
  double x{0.0};
  double y{0.0};
};

Vec2 operator+(Vec2 lhs, Vec2 rhs);
Vec2 operator-(Vec2 lhs, Vec2 rhs);
Vec2 operator*(double scale, Vec2 rhs);
Vec2 operator*(Vec2 lhs, double scale);

double Dot(Vec2 lhs, Vec2 rhs);
double Norm(Vec2 value);
double WrapToPi(double angle);

enum class LineType : int {
  kNone = 0,
  kStriped = 1,
  kContinuous = 2,
  kContinuousLine = 3,
};

enum class LaneKind : int {
  kStraight = 0,
  kSine = 1,
  kCircular = 2,
};

struct LaneCoordinates {
  double longitudinal{0.0};
  double lateral{0.0};
};

class Lane {
 public:
  static Lane Straight(
      Vec2 start, Vec2 end, double width = kDefaultLaneWidth,
      std::array<LineType, 2> line_types = {LineType::kStriped,
                                            LineType::kStriped},
      bool forbidden = false, double speed_limit = 20.0, int priority = 0);
  static Lane Sine(Vec2 start, Vec2 end, double amplitude, double pulsation,
                   double phase, double width = kDefaultLaneWidth,
                   std::array<LineType, 2> line_types = {LineType::kStriped,
                                                         LineType::kStriped},
                   bool forbidden = false, double speed_limit = 20.0,
                   int priority = 0);
  static Lane Circular(
      Vec2 center, double radius, double start_phase, double end_phase,
      bool clockwise = true, double width = kDefaultLaneWidth,
      std::array<LineType, 2> line_types = {LineType::kStriped,
                                            LineType::kStriped},
      bool forbidden = false, double speed_limit = 20.0, int priority = 0);

  [[nodiscard]] Vec2 Position(double longitudinal, double lateral) const;
  [[nodiscard]] LaneCoordinates LocalCoordinates(Vec2 position) const;
  [[nodiscard]] double HeadingAt(double longitudinal) const;
  [[nodiscard]] double WidthAt(double longitudinal) const;
  [[nodiscard]] bool OnLane(Vec2 position, double margin = 0.0) const;
  [[nodiscard]] bool IsReachableFrom(Vec2 position) const;
  [[nodiscard]] bool AfterEnd(Vec2 position) const;
  [[nodiscard]] double Distance(Vec2 position) const;
  [[nodiscard]] double DistanceWithHeading(Vec2 position, double heading,
                                           double heading_weight = 1.0) const;
  [[nodiscard]] double LocalAngle(double heading, double longitudinal) const;

  [[nodiscard]] LaneKind kind() const { return kind_; }
  [[nodiscard]] double length() const { return length_; }
  [[nodiscard]] double width() const { return width_; }
  [[nodiscard]] bool forbidden() const { return forbidden_; }
  [[nodiscard]] double speed_limit() const { return speed_limit_; }
  [[nodiscard]] int priority() const { return priority_; }
  [[nodiscard]] std::array<LineType, 2> line_types() const {
    return line_types_;
  }

 private:
  LaneKind kind_{LaneKind::kStraight};
  Vec2 start_{};
  Vec2 end_{};
  Vec2 center_{};
  Vec2 direction_{1.0, 0.0};
  Vec2 direction_lateral_{0.0, 1.0};
  double heading_{0.0};
  double length_{0.0};
  double width_{kDefaultLaneWidth};
  std::array<LineType, 2> line_types_{LineType::kStriped, LineType::kStriped};
  bool forbidden_{false};
  double speed_limit_{20.0};
  int priority_{0};

  double amplitude_{0.0};
  double pulsation_{0.0};
  double phase_{0.0};

  double radius_{1.0};
  double start_phase_{0.0};
  double end_phase_{0.0};
  bool clockwise_{true};
  double circular_direction_{1.0};

  [[nodiscard]] Vec2 StraightPosition(double longitudinal,
                                      double lateral) const;
  [[nodiscard]] LaneCoordinates StraightLocalCoordinates(Vec2 position) const;
};

}  // namespace official
}  // namespace highway

#endif  // ENVPOOL_HIGHWAY_OFFICIAL_LANE_H_
