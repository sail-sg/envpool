/*
 * Copyright 2021 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py

#ifndef ENVPOOL_BOX2D_CAR_DYNAMICS_H_
#define ENVPOOL_BOX2D_CAR_DYNAMICS_H_

#include <box2d/box2d.h>

#include <cmath>
#include <memory>
#include <random>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "opencv2/opencv.hpp"
#include "utils.h"

namespace box2d {

static const float kSize = 0.02;
static const float kEnginePower = 100000000 * kSize * kSize;
static const float kWheelMomentOfInertia = 4000 * kSize * kSize;
static const float kFrictionLimit = 1000000 * kSize * kSize;
static const float kWheelR = 27;
static const float kWheelW = 14;
static const float kWheelPos[8] = {-55, 80, 55, 80, -55, -82, 55, -82};
static const float kHullPoly1[8] = {-60, 130, 60, 130, 60, 110, -60, 110};
static const float kHullPoly2[8] = {-15, 120, 15, 120, 20, 20, -20, 20};
static const float kHullPoly3[16] = {25,  20,  50,  -10, 50,  -40, 20,  -90,
                                     -20, -90, -50, -40, -50, -10, -25, 20};
static const float kHullPoly4[8] = {-50, -120, 50, -120, 50, -90, -50, -90};
static const float wheelPoly[8] = {-kWheelW, +kWheelR, +kWheelW, +kWheelR,
                                   +kWheelW, -kWheelR, -kWheelW, -kWheelR};

static const float kRoadColor[3] = {102, 102, 102};
static const cv::Scalar kBgColor(102, 204, 102);
static const cv::Scalar kGrassColor(102, 230, 102);

enum UserDataType { INVALID = 1000, WHEEL_TYPE, TILE_TYPE };

class UserData {
 public:
  UserDataType type{INVALID};
  b2Body* body;
  int idx{-1};
};

class Tile : public UserData {
 public:
  bool tileRoadVisited{false};
  float roadFriction;
  std::array<float, 3> RoadColor;
};

class Wheel : public UserData {
 public:
  float wheel_rad{0};
  float gas{0};
  float brake{0};
  float steer{0};
  float phase{0};
  float omega{0};
  b2RevoluteJoint* joint;
  std::unordered_set<Tile*> tiles;
  // body will be wheel object
};

class Car {
 public:
  Car(const std::shared_ptr<b2World>& world, float init_angle, float init_x,
      float init_y);
  void gas(float g);
  void brake(float b);
  void steer(float s);
  void step(float dt);
  void draw(const cv::Mat& surf, float zoom,
            const std::array<float, 2>& translation, float angle);
  void destroy();
  float GetFuelSpent();
  std::vector<float> GetGas();
  std::vector<float> GetSteer();
  std::vector<float> GetBrake();

 protected:
  std::vector<b2Body*> drawlist_;
  std::shared_ptr<b2World> world_;
  b2Body* hull_;
  std::vector<Wheel*> wheels_;
  float fuel_spent_{0};

  friend class CarRacingBox2dEnv;
};

}  // namespace box2d

#endif  // ENVPOOL_BOX2D_CAR_DYNAMICS_H_
