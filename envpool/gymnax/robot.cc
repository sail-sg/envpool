// Copyright 2026 Garena Online Private Limited
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Native adaptation of RobertTLange/gymnax 1.1.0 (Apache-2.0).

#include <algorithm>
#include <cmath>

#include "envpool/gymnax/games.h"
#include "opencv2/imgproc.hpp"

namespace gymnax {
namespace {

constexpr float kPi = 3.14159265358979323846f;

float DistanceSquared(Robot::Vec a, Robot::Vec b) {
  float x = a[0] - b[0];
  float y = a[1] - b[1];
  return x * x + y * y;
}

}  // namespace

Robot::Vec Robot::OnCircle(float radius) {
  float angle = Uniform(0, kPi);
  return {radius * std::cos(angle), radius * std::sin(angle)};
}

void Robot::Reset() {
  auto pair = [this](float r) { return Vec{Uniform(-r, r), Uniform(-r, r)}; };
  vel = {};
  if (task.kind == Kind::PointRobot) {
    goal = OnCircle(1);
    pos = OnCircle(Uniform(0, 1));
    Record(0, goal[0]);
    Record(1, goal[1]);
  } else if (task.kind == Kind::Reacher) {
    pos = pair(kPi);
    vel = pair(1);
    goal = pair(2);
  } else {
    for (auto& u : urchins) {
      u = pair(1);
    }
    pos = pair(1);
    goal = pair(1);
  }
}

void Robot::Step(const float* action) {
  if (task.kind == Kind::PointRobot) {
    // The old position determines the reward. Observe the unclipped action.
    vel = {action[0], action[1]};
    if (std::sqrt(DistanceSquared(pos, goal)) <= 0.2f) {
      reward = 1;
      pos = OnCircle(Uniform(0, 1));
    } else {
      for (int i = 0; i < 2; ++i) {
        pos[i] += std::clamp(action[i], -task.force, task.force);
      }
    }
    return;
  }
  for (int i = 0; i < 2; ++i) {
    vel[i] += 0.05f * action[i];
    pos[i] += 0.05f * vel[i];
  }
  if (task.kind == Kind::Reacher) {
    Vec tip{std::cos(pos[0]) + std::cos(pos[1]),
            std::sin(pos[0]) + std::sin(pos[1])};
    reward = std::exp(-DistanceSquared(tip, goal));
  } else {
    reward = std::exp(-0.5f * DistanceSquared(pos, goal));
    for (const auto& u : urchins) {
      reward *= 1 - std::exp(-30 * DistanceSquared(pos, u));
    }
  }
}

void Robot::Observe(float* obs) const {
  if (task.kind == Kind::PointRobot) {
    obs[0] = pos[0];
    obs[1] = pos[1];
    obs[2] = reward;
    obs[3] = vel[0];
    obs[4] = vel[1];
    obs[5] = time / 50.0f - 1;
    return;
  }
  if (task.kind == Kind::Reacher) {
    for (float angle : pos) {
      *obs++ = std::cos(angle);
    }
    for (float angle : pos) {
      *obs++ = std::sin(angle);
    }
  } else {
    for (const auto& u : urchins) {
      obs = std::copy(u.begin(), u.end(), obs);
    }
    obs = std::copy(pos.begin(), pos.end(), obs);
  }
  obs = std::copy(vel.begin(), vel.end(), obs);
  std::copy(goal.begin(), goal.end(), obs);
}

void Robot::Render(int width, int height, unsigned char* rgb) const {
  cv::Mat frame(height, width, CV_8UC3, rgb);
  frame.setTo(cv::Scalar::all(255));
  const double sx = width / 2.5;
  const double sy = height / 1.5;
  auto point = [=](Vec v) {
    return cv::Point(cvRound(sx * (1.25 + v[0])), cvRound(sy * (1.25 - v[1])));
  };
  auto radius = [=](double r) {
    return cv::Size(cvRound(sx * r), cvRound(sy * r));
  };
  cv::ellipse(frame, point(goal), radius(0.2), 0, 0, 360, {188, 214, 233},
              cv::FILLED, cv::LINE_AA);
  cv::ellipse(frame, point(pos), radius(0.05), 0, 0, 360, {255, 0, 0},
              cv::FILLED, cv::LINE_AA);
  cv::ellipse(frame, point({0, 0}), radius(1), 0, 180, 360, cv::Scalar::all(0),
              1, cv::LINE_AA);
}

}  // namespace gymnax
