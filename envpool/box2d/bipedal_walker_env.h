/*
 * Copyright 2022 Garena Online Private Limited
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
// https://github.com/openai/gym/blob/0.23.1/gym/envs/box2d/bipedal_walker.py

#ifndef ENVPOOL_BOX2D_BIPEDAL_WALKER_ENV_H_
#define ENVPOOL_BOX2D_BIPEDAL_WALKER_ENV_H_

#include <box2d/box2d.h>

#include <array>
#include <memory>
#include <random>
#include <vector>

namespace box2d {

class BipedalWalkerContactDetector;

class BipedalWalkerBox2dEnv {
  const double kFPS = 50;
  const double kScale = 30.0;
  const double kMotorsTorque = 80;
  const double kSpeedHip = 4;
  const double kSpeedKnee = 6;
  const double kLidarRange = 160 / kScale;
  const double kInitialRandom = 5;
  const double kHullPoly[5][2] = {  // NOLINT
      {-30, 9},
      {6, 9},
      {34, 1},
      {34, -8},
      {-30, -8}};
  const double kLegDown = -8 / kScale;
  const double kLegW = 8 / kScale;
  const double kLegH = 34 / kScale;
  const double kViewportW = 600;
  const double kViewportH = 400;
  const double kTerrainStep = 14 / kScale;
  const double kTerrainLength = 200;
  const double kTerrainHeight = kViewportH / kScale / 4;
  const double kTerrainGrass = 10;
  const double kTerrainStartpad = 20;
  const double kFriction = 2.5;

  friend class BipedalWalkerContactDetector;

 protected:
  int max_episode_steps_, elapsed_step_;
  float reward_;
  bool hardcore_, done_;
  std::array<float, 24> obs_;
  std::uniform_real_distribution<> dist_uniform_;

  // box2d related
  std::unique_ptr<b2World> world_;
  b2Body* hull_;
  std::array<b2Body*, 4> legs_;
  std::array<float, 4> ground_contact_;
  std::unique_ptr<BipedalWalkerContactDetector> listener_;

 public:
  BipedalWalkerBox2dEnv(bool hardcore, int max_episode_steps);
  void BipedalWalkerReset(std::mt19937* gen);
  void BipedalWalkerStep(std::mt19937* gen, float action0, float action1,
                         float action2, float action3);

 private:
  void ResetBox2d(std::mt19937* gen);
  void StepBox2d(std::mt19937* gen, float action0, float action1, float action2,
                 float action3);
};

class BipedalWalkerContactDetector : public b2ContactListener {
  BipedalWalkerBox2dEnv* env_;

 public:
  explicit BipedalWalkerContactDetector(BipedalWalkerBox2dEnv* env);
  void BeginContact(b2Contact* contact) override;
  void EndContact(b2Contact* contact) override;
};

}  // namespace box2d

#endif  // ENVPOOL_BOX2D_BIPEDAL_WALKER_ENV_H_
