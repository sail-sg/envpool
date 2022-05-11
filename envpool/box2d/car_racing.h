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

#include <box2d/box2d.h>

#include <cmath>
#include <random>
#include <unordered_set>
#include <vector>

#include "car_dynamics.h"
#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace box2d {

static float kRoadColor[3] = {0.4, 0.4, 0.4};

struct CarEnvWrapper {
  float reward;
  float prev_reward;
  int tile_visited_count;
  std::vector<std::vector<float>> track;
  CarEnvWrapper() {
    reward = 0.f;
    prev_reward = 0.f;
    tile_visited_count = 0;
  }
};

class FrictionDetector : public b2ContactListener {
 public:
  explicit FrictionDetector(CarEnvWrapper* _env) : env(_env) {}
  void BeginContact(b2Contact* contact) { _Contact(contact, true); }
  void EndContact(b2Contact* contact) { _Contact(contact, false); }

 protected:
  CarEnvWrapper* env;

 private:
  void _Contact(b2Contact* contact, bool begin) {
    UserData* tile = nullptr;
    UserData* obj = nullptr;
    void* u1 = (void*)contact->GetFixtureA()->GetBody()->GetUserData().pointer;
    void* u2 = (void*)contact->GetFixtureB()->GetBody()->GetUserData().pointer;

    if (u1 && static_cast<UserData*>(u1)->isTile) {
      tile = static_cast<UserData*>(u1);
      obj = static_cast<UserData*>(u2);
    }
    if (u2 && static_cast<UserData*>(u2)->isTile) {
      tile = static_cast<UserData*>(u2);
      obj = static_cast<UserData*>(u1);
    }
    if (tile == nullptr) return;

    tile->tileColor[0] = kRoadColor[0];
    tile->tileColor[1] = kRoadColor[1];
    tile->tileColor[2] = kRoadColor[2];

    // if not obj or "tiles" not in obj.__dict__:
    //     return
    if (obj == nullptr) return;
    if (begin) {
      obj->objTiles.insert(tile);
      if (!tile->tileRoadVisited) {
        tile->tileRoadVisited = true;
        env->reward += 1000.0 / env->track.size();
        env->tile_visited_count += 1;
      }
    } else {
      obj->objTiles.erase(tile);
    }
  }
};

class CarRacingEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.Bind(1000),
                    "reward_threshold"_.Bind(900.0));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.Bind(Spec<uint8_t>({96, 96, 3}, {0, 256})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    // TODO(alicia): specify range for steer, gas, brake:
    // np.array([-1, 0, 0]).astype(np.float32),
    // np.array([+1, +1, +1]).astype(np.float32),
    return MakeDict("action"_.Bind(Spec<float>({3})));
  }
};

using CarRacingEnvSpec = EnvSpec<CarRacingEnvFns>;

class CarRacingEnv : public Env<CarRacingEnvSpec> {
 protected:
  const int stateW = 96;
  const int stateH = 96;
  const int videoW = 600;
  const int videoH = 400;
  const int windowW = 1000;
  const int windowH = 800;
  const float scale = 6.0;  // Track scale
  const float trackRAD =
      900 / scale;  // Track is heavily morphed circle with this radius
  const float playfiled = 2000 / scale;  // Game over boundary
  const int fps = 50;                    // Frames per second
  const float zoom = 2.7;                // Camera zoom
  const bool zoomFollow = true;  // Set to False for fixed view (don't use zoom)

  const float trackDetailStep = 21 / scale;
  const float trackTurnRate = 0.31;
  const float trackWidth = 40 / scale;
  const float border = 8 / scale;
  const int borderMinCount = 4;

  bool verbose;  // true
  FrictionDetector* contactListener_keepref;
  b2World* world;

  int max_episode_steps_, elapsed_step_;
  std::uniform_real_distribution<> dist_;
  bool done_;

 public:
  CarEnvWrapper* carEnvWrapper = new CarEnvWrapper();

  CarRacingEnv(const Spec& spec, int env_id)
      : Env<CarRacingEnvSpec>(spec, env_id),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        elapsed_step_(max_episode_steps_ + 1),
        done_(true) {
    contactListener_keepref = new FrictionDetector(carEnvWrapper);
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
  void WriteObs(State& state, float reward) {}
};

using CarRacingEnvPool = AsyncEnvPool<CarRacingEnv>;

}  // namespace box2d

#endif  // ENVPOOL_BOX2D_CAR_RACING_H_
