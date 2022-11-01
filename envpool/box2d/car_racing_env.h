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

#ifndef ENVPOOL_BOX2D_CAR_RACING_ENV_H_
#define ENVPOOL_BOX2D_CAR_RACING_ENV_H_

#include <box2d/box2d.h>
#include "car_dynamics.h"

#include <cmath>
#include <random>
#include <unordered_set>
#include <vector>

namespace box2d {

class CarRacingBox2dEnv;

class CarRacingFrictionDetector : public b2ContactListener {
 public:
  explicit CarRacingFrictionDetector(CarRacingBox2dEnv* _env);
  void BeginContact(b2Contact* contact) override { _Contact(contact, true); }
  void EndContact(b2Contact* contact) override { _Contact(contact, false); }

 protected:
  CarRacingBox2dEnv* env_;
  double lap_complete_percent_{0.95};
  void _Contact(b2Contact* contact, bool begin);
};

class CarRacingBox2dEnv{
  const int stateW = 96;
  const int stateH = 96;
  const int videoW = 600;
  const int videoH = 400;
  const int windowW = 1000;
  const int windowH = 800;
  const double kScale = 6.0;  // Track scale
  const double kFps = 50; // Frames per second
  const double trackRAD =
      900 / kScale;  // Track is heavily morphed circle with this radius
  const double kPlayfiled = 2000 / kScale;  // Game over boundary
  const double kTrackTurnRate = 0.31;
  const double kTrackDetailStep =21 / kScale;
  const double kTrackWidth = 40 / kScale;


  friend class CarRacingFrictionDetector;

 protected:
  int max_episode_steps_, elapsed_step_{0};
  float reward_{0};
  float prev_reward_{0};
  float step_reward_{0};
  bool done_{false};

  std::unique_ptr<CarRacingFrictionDetector> listener_;
  std::shared_ptr<b2World> world_;
  std::unique_ptr<Car> car_;
  int tile_visited_count_{0};
  float start_alpha_{0};
  bool new_lap_{false};
  b2FixtureDef fd_tile_;
  std::vector<std::array<double, 4>> track_;
  std::vector<UserData*> roads_;
  
 public:
  CarRacingBox2dEnv(int max_episode_steps);
  void CarRacingReset(std::mt19937* gen);
  void CarRacingStep(std::mt19937* gen, float action0, float action1, float action2);

 private:
  bool CreateTrack();
  void ResetBox2d(std::mt19937* gen);
  void StepBox2d(std::mt19937* gen, float action0, float action1, float action2,
  bool isAction);


};

}  // namespace box2d

#endif  // ENVPOOL_BOX2D_CAR_RACING_ENV_H_