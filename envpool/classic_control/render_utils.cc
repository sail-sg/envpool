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

#include "envpool/classic_control/render_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <opencv2/opencv.hpp>

namespace classic_control {
namespace rendering {
namespace {

constexpr double kPi = 3.14159265358979323846;

cv::Point ToPoint(double x, double y) {
  return {static_cast<int>(std::lround(x)), static_cast<int>(std::lround(y))};
}

cv::Point2d RotatePoint(const cv::Point2d& point, double theta) {
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  return {point.x * c - point.y * s, point.x * s + point.y * c};
}

std::vector<cv::Point> TransformPolygonPoints(
    const std::vector<cv::Point2d>& points, double origin_x, double origin_y,
    double theta) {
  std::vector<cv::Point> polygon;
  polygon.reserve(points.size());
  for (const auto& point : points) {
    const cv::Point2d rotated = RotatePoint(point, theta);
    polygon.push_back(ToPoint(rotated.x + origin_x, rotated.y + origin_y));
  }
  return polygon;
}

double MountainHeight(double pos) { return std::sin(3.0 * pos) * 0.45 + 0.55; }

void FinalizeRgbFrame(cv::Mat* surf, std::uint8_t* rgb) {
  cv::flip(*surf, *surf, 0);
  cv::Mat output(surf->rows, surf->cols, CV_8UC3, rgb);
  surf->copyTo(output);
}

}  // namespace

void RenderCartPole(double x, double theta, int width, int height,
                    std::uint8_t* rgb) {
  constexpr double kXThreshold = 2.4;
  constexpr double kPoleLength = 1.0;
  constexpr double kPoleWidth = 10.0;
  constexpr double kCartWidth = 50.0;
  constexpr double kCartHeight = 30.0;
  constexpr double kCartY = 100.0;
  constexpr double kAxleOffset = kCartHeight / 4.0;

  cv::Mat surf(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
  const double scale = static_cast<double>(width) / (2.0 * kXThreshold);
  const double cart_x = x * scale + static_cast<double>(width) / 2.0;

  const std::vector<cv::Point2d> cart = {
      {-kCartWidth / 2.0, -kCartHeight / 2.0},
      {-kCartWidth / 2.0, kCartHeight / 2.0},
      {kCartWidth / 2.0, kCartHeight / 2.0},
      {kCartWidth / 2.0, -kCartHeight / 2.0},
  };
  const std::vector<cv::Point> cart_poly =
      TransformPolygonPoints(cart, cart_x, kCartY, 0.0);
  cv::fillConvexPoly(surf, cart_poly, cv::Scalar(0, 0, 0), cv::LINE_AA);

  const double pole_len = scale * kPoleLength;
  const std::vector<cv::Point2d> pole = {
      {-kPoleWidth / 2.0, -kPoleWidth / 2.0},
      {-kPoleWidth / 2.0, pole_len - kPoleWidth / 2.0},
      {kPoleWidth / 2.0, pole_len - kPoleWidth / 2.0},
      {kPoleWidth / 2.0, -kPoleWidth / 2.0},
  };
  const std::vector<cv::Point> pole_poly =
      TransformPolygonPoints(pole, cart_x, kCartY + kAxleOffset, -theta);
  cv::fillConvexPoly(surf, pole_poly, cv::Scalar(202, 152, 101), cv::LINE_AA);
  cv::circle(surf, ToPoint(cart_x, kCartY + kAxleOffset),
             static_cast<int>(kPoleWidth / 2.0), cv::Scalar(129, 132, 203),
             cv::FILLED, cv::LINE_AA);
  cv::line(surf, ToPoint(0, kCartY), ToPoint(width, kCartY),
           cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

  FinalizeRgbFrame(&surf, rgb);
}

void RenderPendulum(double theta, bool has_last_u, double last_u, int width,
                    int height, std::uint8_t* rgb) {
  cv::Mat surf(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
  const double bound = 2.2;
  const double scale = static_cast<double>(width) / (bound * 2.0);
  const double offset_x = static_cast<double>(width) / 2.0;
  const double offset_y = static_cast<double>(height) / 2.0;
  const double rod_length = scale;
  const double rod_width = 0.2 * scale;

  const std::vector<cv::Point2d> rod = {
      {0.0, -rod_width / 2.0},
      {0.0, rod_width / 2.0},
      {rod_length, rod_width / 2.0},
      {rod_length, -rod_width / 2.0},
  };
  const std::vector<cv::Point> rod_poly =
      TransformPolygonPoints(rod, offset_x, offset_y, theta + kPi / 2.0);
  cv::fillConvexPoly(surf, rod_poly, cv::Scalar(204, 77, 77), cv::LINE_AA);
  cv::circle(surf, ToPoint(offset_x, offset_y),
             static_cast<int>(std::lround(rod_width / 2.0)),
             cv::Scalar(204, 77, 77), cv::FILLED, cv::LINE_AA);
  const cv::Point2d rod_end =
      RotatePoint(cv::Point2d(rod_length, 0.0), theta + kPi / 2.0);
  cv::circle(surf, ToPoint(rod_end.x + offset_x, rod_end.y + offset_y),
             static_cast<int>(std::lround(rod_width / 2.0)),
             cv::Scalar(204, 77, 77), cv::FILLED, cv::LINE_AA);
  cv::circle(surf, ToPoint(offset_x, offset_y),
             static_cast<int>(std::lround(0.05 * scale)), cv::Scalar(0, 0, 0),
             cv::FILLED, cv::LINE_AA);

  if (has_last_u && std::abs(last_u) > 1e-6) {
    const double arrow_scale = scale * std::abs(last_u) / 2.0;
    const double direction = last_u > 0.0 ? 1.0 : -1.0;
    const cv::Point arrow_start = ToPoint(
        offset_x - direction * 0.35 * arrow_scale, offset_y - 0.3 * scale);
    const cv::Point arrow_end = ToPoint(
        offset_x + direction * 0.35 * arrow_scale, offset_y - 0.3 * scale);
    cv::arrowedLine(surf, arrow_start, arrow_end, cv::Scalar(0, 0, 0), 2,
                    cv::LINE_AA, 0, 0.35);
  }

  FinalizeRgbFrame(&surf, rgb);
}

void RenderMountainCar(double pos, double goal_pos, int width, int height,
                       std::uint8_t* rgb) {
  constexpr double kMinPos = -1.2;
  constexpr double kMaxPos = 0.6;
  constexpr double kCarWidth = 40.0;
  constexpr double kCarHeight = 20.0;
  constexpr double kClearance = 10.0;

  cv::Mat surf(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
  const double scale = static_cast<double>(width) / (kMaxPos - kMinPos);

  std::vector<cv::Point> path;
  path.reserve(100);
  for (int i = 0; i < 100; ++i) {
    const double x =
        kMinPos + (kMaxPos - kMinPos) * static_cast<double>(i) / 99.0;
    const double y = MountainHeight(x);
    path.push_back(ToPoint((x - kMinPos) * scale, y * scale));
  }
  cv::polylines(surf, path, false, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

  const double angle = std::cos(3.0 * pos);
  const std::vector<cv::Point2d> car = {
      {-kCarWidth / 2.0, 0.0},
      {-kCarWidth / 2.0, kCarHeight},
      {kCarWidth / 2.0, kCarHeight},
      {kCarWidth / 2.0, 0.0},
  };
  const double car_x = (pos - kMinPos) * scale;
  const double car_y = kClearance + MountainHeight(pos) * scale;
  const std::vector<cv::Point> car_poly =
      TransformPolygonPoints(car, car_x, car_y, angle);
  cv::fillConvexPoly(surf, car_poly, cv::Scalar(0, 0, 0), cv::LINE_AA);

  for (double wheel_offset : {kCarWidth / 4.0, -kCarWidth / 4.0}) {
    const cv::Point2d rotated = RotatePoint({wheel_offset, 0.0}, angle);
    cv::circle(surf, ToPoint(rotated.x + car_x, rotated.y + car_y),
               static_cast<int>(std::lround(kCarHeight / 2.5)),
               cv::Scalar(128, 128, 128), cv::FILLED, cv::LINE_AA);
  }

  const int flag_x =
      static_cast<int>(std::lround((goal_pos - kMinPos) * scale));
  const int flag_y1 =
      static_cast<int>(std::lround(MountainHeight(goal_pos) * scale));
  const int flag_y2 = flag_y1 + 50;
  cv::line(surf, cv::Point(flag_x, flag_y1), cv::Point(flag_x, flag_y2),
           cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  const std::vector<cv::Point> flag = {
      cv::Point(flag_x, flag_y2),
      cv::Point(flag_x, flag_y2 - 10),
      cv::Point(flag_x + 25, flag_y2 - 5),
  };
  cv::fillConvexPoly(surf, flag, cv::Scalar(204, 204, 0), cv::LINE_AA);

  FinalizeRgbFrame(&surf, rgb);
}

void RenderAcrobot(double theta1, double theta2, int width, int height,
                   std::uint8_t* rgb) {
  constexpr double kLinkLength1 = 1.0;
  constexpr double kLinkLength2 = 1.0;

  cv::Mat surf(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
  const double bound = kLinkLength1 + kLinkLength2 + 0.2;
  const double scale = static_cast<double>(width) / (bound * 2.0);
  const double offset = static_cast<double>(width) / 2.0;

  const cv::Point2d p1 = {
      kLinkLength1 * std::sin(theta1) * scale,
      -kLinkLength1 * std::cos(theta1) * scale,
  };
  const cv::Point2d p2 = {
      p1.x + kLinkLength2 * std::sin(theta1 + theta2) * scale,
      p1.y - kLinkLength2 * std::cos(theta1 + theta2) * scale,
  };

  cv::line(surf, ToPoint(-2.2 * scale + offset, scale + offset),
           ToPoint(2.2 * scale + offset, scale + offset), cv::Scalar(0, 0, 0),
           1, cv::LINE_AA);

  const std::array<cv::Point2d, 3> joints = {
      cv::Point2d(0.0, 0.0), p1, p2};
  const std::array<double, 2> link_lengths = {kLinkLength1 * scale,
                                              kLinkLength2 * scale};
  const std::array<double, 2> link_thetas = {theta1 - kPi / 2.0,
                                             theta1 + theta2 - kPi / 2.0};
  for (int i = 0; i < 2; ++i) {
    const cv::Point2d joint = joints[i];
    const std::vector<cv::Point2d> link = {
        {0.0, -0.1 * scale},
        {0.0, 0.1 * scale},
        {link_lengths[i], 0.1 * scale},
        {link_lengths[i], -0.1 * scale},
    };
    const std::vector<cv::Point> link_poly =
        TransformPolygonPoints(link, joint.x + offset, joint.y + offset,
                               link_thetas[i]);
    cv::fillConvexPoly(surf, link_poly, cv::Scalar(0, 204, 204), cv::LINE_AA);
    cv::circle(surf, ToPoint(joint.x + offset, joint.y + offset),
               static_cast<int>(std::lround(0.1 * scale)),
               cv::Scalar(204, 204, 0), cv::FILLED, cv::LINE_AA);
  }

  FinalizeRgbFrame(&surf, rgb);
}

}  // namespace rendering
}  // namespace classic_control
