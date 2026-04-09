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

#include "envpool/highway/official_lane.h"

#include <algorithm>
#include <cmath>

namespace highway::official {

Vec2 operator+(Vec2 lhs, Vec2 rhs) { return {lhs.x + rhs.x, lhs.y + rhs.y}; }

Vec2 operator-(Vec2 lhs, Vec2 rhs) { return {lhs.x - rhs.x, lhs.y - rhs.y}; }

Vec2 operator*(double scale, Vec2 rhs) {
  return {scale * rhs.x, scale * rhs.y};
}

Vec2 operator*(Vec2 lhs, double scale) { return scale * lhs; }

double Dot(Vec2 lhs, Vec2 rhs) { return lhs.x * rhs.x + lhs.y * rhs.y; }

double Norm(Vec2 value) { return std::sqrt(Dot(value, value)); }

double WrapToPi(double angle) {
  angle = std::fmod(angle + kPi, 2.0 * kPi);
  if (angle < 0.0) {
    angle += 2.0 * kPi;
  }
  return angle - kPi;
}

Lane Lane::Straight(Vec2 start, Vec2 end, double width,
                    std::array<LineType, 2> line_types, bool forbidden,
                    double speed_limit, int priority) {
  Lane lane;
  lane.kind_ = LaneKind::kStraight;
  lane.start_ = start;
  lane.end_ = end;
  lane.width_ = width;
  lane.line_types_ = line_types;
  lane.forbidden_ = forbidden;
  lane.priority_ = priority;
  lane.speed_limit_ = speed_limit;
  lane.heading_ = std::atan2(end.y - start.y, end.x - start.x);
  lane.length_ = Norm(end - start);
  lane.direction_ = (1.0 / lane.length_) * (end - start);
  lane.direction_lateral_ = {-lane.direction_.y, lane.direction_.x};
  return lane;
}

Lane Lane::Sine(Vec2 start, Vec2 end, double amplitude, double pulsation,
                double phase, double width, std::array<LineType, 2> line_types,
                bool forbidden, double speed_limit, int priority) {
  Lane lane = Lane::Straight(start, end, width, line_types, forbidden,
                             speed_limit, priority);
  lane.kind_ = LaneKind::kSine;
  lane.amplitude_ = amplitude;
  lane.pulsation_ = pulsation;
  lane.phase_ = phase;
  return lane;
}

Lane Lane::Circular(Vec2 center, double radius, double start_phase,
                    double end_phase, bool clockwise, double width,
                    std::array<LineType, 2> line_types, bool forbidden,
                    double speed_limit, int priority) {
  Lane lane;
  lane.kind_ = LaneKind::kCircular;
  lane.center_ = center;
  lane.radius_ = radius;
  lane.start_phase_ = start_phase;
  lane.end_phase_ = end_phase;
  lane.clockwise_ = clockwise;
  lane.circular_direction_ = clockwise ? 1.0 : -1.0;
  lane.width_ = width;
  lane.line_types_ = line_types;
  lane.forbidden_ = forbidden;
  lane.length_ = radius * (end_phase - start_phase) * lane.circular_direction_;
  lane.priority_ = priority;
  lane.speed_limit_ = speed_limit;
  return lane;
}

Vec2 Lane::StraightPosition(double longitudinal, double lateral) const {
  return start_ + longitudinal * direction_ + lateral * direction_lateral_;
}

LaneCoordinates Lane::StraightLocalCoordinates(Vec2 position) const {
  const Vec2 delta = position - start_;
  return {Dot(delta, direction_), Dot(delta, direction_lateral_)};
}

Vec2 Lane::Position(double longitudinal, double lateral) const {
  if (kind_ == LaneKind::kSine) {
    const double offset =
        amplitude_ * std::sin(pulsation_ * longitudinal + phase_);
    return StraightPosition(longitudinal, lateral + offset);
  }
  if (kind_ == LaneKind::kCircular) {
    const double phi =
        circular_direction_ * longitudinal / radius_ + start_phase_;
    const double r = radius_ - lateral * circular_direction_;
    return center_ + r * Vec2{std::cos(phi), std::sin(phi)};
  }
  return StraightPosition(longitudinal, lateral);
}

LaneCoordinates Lane::LocalCoordinates(Vec2 position) const {
  if (kind_ == LaneKind::kSine) {
    LaneCoordinates coordinates = StraightLocalCoordinates(position);
    coordinates.lateral -=
        amplitude_ * std::sin(pulsation_ * coordinates.longitudinal + phase_);
    return coordinates;
  }
  if (kind_ == LaneKind::kCircular) {
    const Vec2 delta = position - center_;
    double phi = std::atan2(delta.y, delta.x);
    phi = start_phase_ + WrapToPi(phi - start_phase_);
    const double r = Norm(delta);
    return {circular_direction_ * (phi - start_phase_) * radius_,
            circular_direction_ * (radius_ - r)};
  }
  return StraightLocalCoordinates(position);
}

double Lane::HeadingAt(double longitudinal) const {
  if (kind_ == LaneKind::kSine) {
    return heading_ + std::atan(amplitude_ * pulsation_ *
                                std::cos(pulsation_ * longitudinal + phase_));
  }
  if (kind_ == LaneKind::kCircular) {
    const double phi =
        circular_direction_ * longitudinal / radius_ + start_phase_;
    return phi + kPi / 2.0 * circular_direction_;
  }
  return heading_;
}

double Lane::WidthAt(double longitudinal) const {
  (void)longitudinal;
  return width_;
}

bool Lane::OnLane(Vec2 position, double margin) const {
  const LaneCoordinates coordinates = LocalCoordinates(position);
  return std::abs(coordinates.lateral) <=
             WidthAt(coordinates.longitudinal) / 2.0 + margin &&
         -kLaneVehicleLength <= coordinates.longitudinal &&
         coordinates.longitudinal < length_ + kLaneVehicleLength;
}

bool Lane::IsReachableFrom(Vec2 position) const {
  if (forbidden_) {
    return false;
  }
  const LaneCoordinates coordinates = LocalCoordinates(position);
  return std::abs(coordinates.lateral) <=
             2.0 * WidthAt(coordinates.longitudinal) &&
         0.0 <= coordinates.longitudinal &&
         coordinates.longitudinal < length_ + kLaneVehicleLength;
}

bool Lane::AfterEnd(Vec2 position) const {
  const LaneCoordinates coordinates = LocalCoordinates(position);
  return coordinates.longitudinal > length_ - kLaneVehicleLength / 2.0;
}

double Lane::Distance(Vec2 position) const {
  const LaneCoordinates coordinates = LocalCoordinates(position);
  return std::abs(coordinates.lateral) +
         std::max(coordinates.longitudinal - length_, 0.0) +
         std::max(0.0 - coordinates.longitudinal, 0.0);
}

double Lane::DistanceWithHeading(Vec2 position, double heading,
                                 double heading_weight) const {
  const LaneCoordinates coordinates = LocalCoordinates(position);
  const double angle = std::abs(LocalAngle(heading, coordinates.longitudinal));
  return std::abs(coordinates.lateral) +
         std::max(coordinates.longitudinal - length_, 0.0) +
         std::max(0.0 - coordinates.longitudinal, 0.0) + heading_weight * angle;
}

double Lane::LocalAngle(double heading, double longitudinal) const {
  return WrapToPi(heading - HeadingAt(longitudinal));
}

}  // namespace highway::official
