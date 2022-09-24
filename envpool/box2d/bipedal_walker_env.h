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

class BipedalWalkerLidarCallback : public b2RayCastCallback {
 public:
  float fraction;

  float ReportFixture(b2Fixture* fixture, const b2Vec2& point,
                      const b2Vec2& normal, float fraction) override;
};

class BipedalWalkerBox2dEnv {
  const float kFPS = 50;
  const float kScaleFloat = 30.0;
  const double kScaleDouble = 30.0;
  const float kMotorsTorque = 80;
  const float kSpeedHip = 4;
  const float kSpeedKnee = 6;
  const double kLidarRange = 160 / kScaleDouble;
  const double kInitialRandom = 5;
  const double kHullPoly[5][2] = {  // NOLINT
      {-30, 9},
      {6, 9},
      {34, 1},
      {34, -8},
      {-30, -8}};
  const double kLegDown = -8 / kScaleDouble;
  const double kLegW = 8 / kScaleDouble;
  const double kLegH = 34 / kScaleDouble;
  const double kViewportW = 600;
  const double kViewportH = 400;
  const double kTerrainStep = 14 / kScaleDouble;
  const double kTerrainHeight = kViewportH / kScaleDouble / 4;
  const float kFriction = 2.5;
  static const int kTerrainLength = 200;
  static const int kTerrainGrass = 10;
  static const int kTerrainStartpad = 20;
  static const int kLidarNum = 10;
  static const int kGrass = 0;
  static const int kStump = 1;
  static const int kStairs = 2;
  static const int kPit = 3;
  static const int kStates = 4;

  friend class BipedalWalkerContactDetector;

 protected:
  int max_episode_steps_, elapsed_step_;
  float reward_, prev_shaping_;
  bool hardcore_, done_;
  std::array<float, 24> obs_;
  // info
  float scroll_;
  std::vector<float> path2_, path4_, path5_;

  // box2d related
  std::unique_ptr<b2World> world_;
  b2Body* hull_;
  std::vector<b2Vec2> hull_poly_;
  std::vector<b2Body*> terrain_;
  std::array<b2Body*, 4> legs_;
  std::array<float, 4> ground_contact_;
  std::array<b2RevoluteJoint*, 4> joints_;
  std::array<BipedalWalkerLidarCallback, kLidarNum> lidar_;
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
  void CreateTerrain(std::vector<b2Vec2> poly);
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
