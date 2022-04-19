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

#ifndef ENVPOOL_BOX2D_CAR_RACING_H_
#define ENVPOOL_BOX2D_CAR_RACING_H_

#include <cmath>
#include <random>
#include <vector>
#include <unordered_set>
#include <box2d/box2d.h>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace box2d {

class CarRacingEnvFns;
class CarRacingEnv;


typedef struct UserData{
  b2Body* body;
  bool isTile;
  bool tileRoadVisited;
  float tileColor[3];
  float roadFriction;
  std::unordered_set<struct UserData*> objTiles;
} UserData;

float kRoadColor[3] = {0.4, 0.4, 0.4};

float kSize = 0.02;
float kEnginePower = 100000000 * kSize * kSize;
float kWheelMomentOfInertia = 4000 * kSize * kSize;
float kFrictionLimit = 1000000 * kSize * kSize;
// friction ~= mass ~= size^2 (calculated implicitly using density)
int kWheelR = 27;
int kWheelW = 14;
// WHEELPOS = [(-55, +80), (+55, +80), (-55, -82), (+55, -82)]
float kHullPoly1[8] = {-60, +130, +60, +130, +60, +110, -60, +110};
float kHullPoly2[8] = {-15, +120, +15, +120, +20, +20, -20, 20};
float kHullPoly3[16] = {+25, +20, +50, -10, +50, -40, +20, -90, -20, -90, -50, -40, -50, -10, -25, +20};
float kHullPoly4[8] = {-50, -120, +50, -120, +50, -90, -50, -90};

float kWheelColor[3] = {0.0, 0.0, 0.0};
float kWheelWhite[3] = {0.3, 0.3, 0.3};
float kMudColor[3] = {0.4, 0.4, 0.0};

b2PolygonShape generatePolygon(float* array, int size);

class FrictionDetector: public b2ContactListener {
  public:
    FrictionDetector(CarRacingEnv* _env);
    void BeginContact (b2Contact *contact);
    void EndContact (b2Contact *contact);
  protected:
    box2d::CarRacingEnv* env;
  private:
    void _Contact(b2Contact *contact, bool begin);

};

class Car {
  public:
    Car(b2World* _world, float init_angle, float init_x, float init_y)
      : world(_world), fuel_spent(0.0) {
        b2BodyDef bd;
        bd.position.Set(init_x, init_y);
        bd.angle = init_angle;
        bd.type = b2_dynamicBody;
        hull = world->CreateBody(&bd);

        b2PolygonShape polygon1 = generatePolygon(kHullPoly1, 4);
        fixtures.emplace_back(hull->CreateFixture(&polygon1, 1.f));

        b2PolygonShape polygon2 = generatePolygon(kHullPoly2, 4);
        fixtures.emplace_back(hull->CreateFixture(&polygon2, 1.f));

        b2PolygonShape polygon3 = generatePolygon(kHullPoly3, 8);
        fixtures.emplace_back(hull->CreateFixture(&polygon3, 1.f));

        b2PolygonShape polygon4 = generatePolygon(kHullPoly4, 4);
        fixtures.emplace_back(hull->CreateFixture(&polygon4, 1.f));

    }
  protected:
    b2World* world;
    b2Body* hull;
    std::vector<b2Body*> wheels;
    std::vector<b2Fixture*> fixtures;
    float fuel_spent;
};


class CarRacingEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.bind(1000),
                    "reward_threshold"_.bind(900.0));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.bind(Spec<uint8_t>({96, 96, 3}, {0, 256})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    // TODO(alicia): specify range for steer, gas, brake:
    // np.array([-1, 0, 0]).astype(np.float32),
    // np.array([+1, +1, +1]).astype(np.float32),
    return MakeDict("action"_.bind(Spec<float>({3})));
  }
};

typedef class EnvSpec<CarRacingEnvFns> CarRacingEnvSpec;

class CarRacingEnv : public Env<CarRacingEnvSpec> {
 protected:
    const int stateW = 96;
    const int stateH = 96;
    const int videoW = 600;
    const int videoH = 400;
    const int windowW = 1000;
    const int windowH = 800;
    const float scale = 6.0; //Track scale
    const float trackRAD = 900 / scale; // Track is heavily morphed circle with this radius
    const float playfiled = 2000 / scale;  // Game over boundary
    const int fps = 50; // Frames per second
    const float zoom = 2.7; // Camera zoom
    const bool zoomFollow = true;  // Set to False for fixed view (don't use zoom)

    const float trackDetailStep = 21 / scale;
    const float trackTurnRate = 0.31;
    const float trackWidth = 40 / scale;
    const float border = 8 / scale;
    const int borderMinCount = 4;

    float reward = 0.0;
    float prev_reward = 0.0;

    bool verbose; // true
    FrictionDetector* contactListener_keepref;
    b2World* world;
    std::vector<std::vector<float>> track;
    int tile_visited_count;

    int max_episode_steps_, elapsed_step_;
    std::uniform_real_distribution<> dist_;
    bool done_;
    
 public:
  CarRacingEnv(const Spec& spec, int env_id)
      : Env<CarRacingEnvSpec>(spec, env_id),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        elapsed_step_(max_episode_steps_ + 1),
        done_(true) {
          contactListener_keepref = new FrictionDetector(this);
          b2Vec2 gravity(0.0f, 0.0f);
          world = new b2World(gravity);
          world->SetContactListener(contactListener_keepref);
        }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    State state = Allocate();
    WriteObs(state, 0.0f);
  }

  void Step(const Action& action) override {
    State state = Allocate();
    WriteObs(state, 1.0f);
  }

 private:
  void WriteObs(State& state, float reward) {  // NOLINT
  }
};

typedef AsyncEnvPool<CarRacingEnv> CarRacingEnvPool;

}  // namespace box2d

#endif  // ENVPOOL_BOX2D_CarRacing_H_
