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

namespace box2d {

static const double kSize = 0.02;
static const double kEnginePower = 100000000 * kSize * kSize;
static const double kWheelMomentOfInertia = 4000 * kSize * kSize;
static const double kFrictionLimit = 1000000 * kSize * kSize;
static const double kWheelR = 27;
static const double kWheelW = 14;
static const double kWheelPos[4][2] = {{-55, 80}, {55, 80}, {-55, -82}, {55, -82}};
static const double kHullPoly1[4][2] = {{-60, 130}, {60, 130}, {60, 110}, {-60, 110}};
static const double kHullPoly2[4][2] = {{-15, 120}, {15, 120}, {20, 20}, {-20, 20}};
static const double kHullPoly3[8][2] = {{25, 20}, {50, -10}, {50, -40}, {20, -90},
                               {-20, -90}, {-50, -40}, {-50, -10}, {-25, 20}};
static const double kHullPoly4[4][2] = {{-50, -120}, {50, -120}, {50, -90}, {-50, -90}};
static const double kWheelColor[3] = {0.0, 0.0, 0.0};
static const double kWheelWhite[3] = {0.3, 0.3, 0.3};
static const double kMudColor[3] = {0.4, 0.4, 0.0};

struct UserData {
  b2Body* body;
  bool isTile;
  bool tileRoadVisited;
  float tileColor[3];
  float roadFriction;
  std::unordered_set<UserData*> objTiles;
};

class Particle {
 public:
  float color[3];
  float ttl;
  std::vector<std::tuple<float, float> > poly;
  bool isGrass;
  Particle(b2Vec2 point1, b2Vec2 point2, bool _isGrass) {
    if (!_isGrass) {
      memcpy(color, kWheelColor, 3 * sizeof(float));
    } else {
      memcpy(color, kMudColor, 3 * sizeof(float));
    }
    ttl = 1;
    poly.push_back({point1.x, point1.y});
    poly.push_back({point2.x, point2.y});
    isGrass = _isGrass;
  }
};

struct WheelData {
  float wheel_rad;
  float color[3];
  float gas;
  float brake;
  float steer;
  float phase;
  float omega;
  b2Vec2* skid_start;
  Particle* skid_particle;
  b2RevoluteJoint* joint;
  std::unordered_set<UserData*> tiles;
  b2Body* wheel;
};

b2PolygonShape generatePolygon(float* array, int size);

class Car {
 public:
  Car(b2World* _world, float init_angle, float init_x, float init_y);
  void gas(float g);
  void brake(float b);
  void steer(float s);
  void step(float dt);
  void destroy();

 protected:
  b2World* world;
  b2Body* hull;
  std::vector<WheelData*> wheels;
  std::vector<b2Fixture*> hullFixtures;
  std::vector<Particle*> particles;
  float fuel_spent = 0.f;
  const float color[3] = {0.8, 0.0, 0.0};
  const float wheelPoly[8] = {-kWheelW, +kWheelR, +kWheelW, +kWheelR,
                              +kWheelW, -kWheelR, -kWheelW, -kWheelR};
  Particle* createParticle(b2Vec2 point1, b2Vec2 point2, bool isGrass) {
    Particle* p = new Particle(point1, point2, isGrass);
    particles.push_back(p);
    while (particles.size() > 30) {
      particles.erase(particles.begin());
    }
    return p;
  }
};

}  // namespace box2d

#endif  // ENVPOOL_BOX2D_CAR_DYNAMICS_H_
