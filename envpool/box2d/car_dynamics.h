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
#include <random>
#include <tuple>
#include <unordered_set>
#include <vector>
#include <memory>

namespace box2d {

class UserData {
  public: 
    b2Body* body;
    bool isTile;
    bool tileRoadVisited;
    double roadFriction;
    std::unordered_set<UserData*> tiles;
};

class Wheel: public UserData {
  public: 
    double wheel_rad{0};
    double gas{0};
    double brake{0};
    double steer{0};
    double phase{0};
    double omega{0};
    b2RevoluteJoint* joint;
  // body will be wheel object
};

static const double kSize = 0.02;
static const double kEnginePower = 100000000 * kSize * kSize;
static const double kWheelMomentOfInertia = 4000 * kSize * kSize;
static const double kFrictionLimit = 1000000 * kSize * kSize;
static const double kWheelR = 27;
static const double kWheelW = 14;
static const double kWheelPos[8] = {-55, 80, 55, 80, -55, -82, 55, -82};
static const double kHullPoly1[8] = {-60, 130, 60, 130, 60, 110, -60, 110};
static const double kHullPoly2[8] = {-15, 120, 15, 120, 20, 20, -20, 20};
static const double kHullPoly3[16] = {25, 20, 50, -10, 50, -40, 20, -90,
                                      -20, -90, -50, -40, -50, -10, -25, 20};
static const double kHullPoly4[8] = {-50, -120, 50, -120,50, -90, -50, -90};
static const double wheelPoly[8] = {-kWheelW, +kWheelR, +kWheelW, +kWheelR,
                              +kWheelW, -kWheelR, -kWheelW, -kWheelR};


class Car {
 public:
  Car(std::shared_ptr<b2World>& world, double init_angle, double init_x, double init_y);
  void gas(double g);
  void brake(double b);
  void steer(double s);
  void step(double dt);
  void destroy();

 protected:
  std::shared_ptr<b2World> world_;
  b2Body* hull_;
  std::vector<Wheel*> wheels_;
  std::vector<b2Fixture*> hullFixtures;
  double fuel_spent{0};
};

}  // namespace box2d

#endif  // ENVPOOL_BOX2D_CAR_DYNAMICS_H_