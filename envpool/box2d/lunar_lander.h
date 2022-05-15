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
// https://github.com/openai/gym/blob/0.23.1/gym/envs/box2d/lunar_lander.py

#ifndef ENVPOOL_BOX2D_LUNAR_LANDER_H_
#define ENVPOOL_BOX2D_LUNAR_LANDER_H_

#include <box2d/box2d.h>

#include <array>
#include <memory>
#include <random>
#include <vector>

namespace box2d {

class ContactDetector;

class LunarLanderEnv {
  static constexpr double kFPS = 50;
  static constexpr double kScale = 30.0;
  static constexpr double kMainEnginePower = 13.0;
  static constexpr double kSideEnginePower = 0.6;
  static constexpr double kInitialRandom = 1000.0;
  static constexpr double kLanderPoly[6][2] = {{-14, 17},  {-17, 0},  // NOLINT
                                               {-17, -10}, {17, -10},
                                               {17, 0},    {14, 17}};
  static constexpr double kLegAway = 20;
  static constexpr double kLegDown = 18;
  static constexpr double kLegW = 2;
  static constexpr double kLegH = 8;
  static constexpr double kLegSpringTorque = 40;
  static constexpr double kSideEngineHeight = 14.0;
  static constexpr double kSideEngineAway = 12.0;
  static constexpr double kViewportW = 600;
  static constexpr double kViewportH = 400;
  static const int kChunks = 11;

  friend class ContactDetector;

 protected:
  int max_episode_steps_, elapsed_step_;
  float reward_, prev_shaping_;
  bool continuous_, done_;
  std::array<float, 8> obs_;

  // box2d related
  std::unique_ptr<b2World> world_;
  b2Body *moon_, *lander_;
  std::vector<b2Body*> particles_;
  std::vector<b2Vec2> lander_poly_;
  std::array<b2Body*, 2> legs_;
  std::array<float, 2> ground_contact_;
  std::unique_ptr<ContactDetector> listener_;
  std::uniform_real_distribution<> dist_;

 public:
  LunarLanderEnv(bool continuous, int max_episode_steps);
  void LunarLanderReset(std::mt19937* gen);
  // discrete action space: action
  // continuous action space: action0 and action1
  void LunarLanderStep(std::mt19937* gen, int action, float action0,
                       float action1);

 private:
  void ResetBox2d(std::mt19937* gen);
  void StepBox2d(std::mt19937* gen, int action, float action0, float action1);
  b2Body* CreateParticle(float mass, b2Vec2 pos);
};

class ContactDetector : public b2ContactListener {
  LunarLanderEnv* env_;

 public:
  explicit ContactDetector(LunarLanderEnv* env);
  void BeginContact(b2Contact* contact) override;
  void EndContact(b2Contact* contact) override;
};

}  // namespace box2d

#endif  // ENVPOOL_BOX2D_LUNAR_LANDER_H_
