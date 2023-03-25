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

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include "car_dynamics.h"
#include "utils.h"

namespace box2d {

class CarRacingBox2dEnv;

class CarRacingFrictionDetector : public b2ContactListener {
 public:
  explicit CarRacingFrictionDetector(CarRacingBox2dEnv* _env,
                                     float lap_complete_percent);
  void BeginContact(b2Contact* contact) override { Contact(contact, true); }
  void EndContact(b2Contact* contact) override { Contact(contact, false); }

 protected:
  CarRacingBox2dEnv* env_;
  float lap_complete_percent_;
  void Contact(b2Contact* contact, bool begin);
};

class CarRacingBox2dEnv {
  const int kStateW = 96;
  const int kStateH = 96;
  const int kWindowW = 1000;
  const int kWindowH = 800;
  const float kScale = 6.0;  // Track scale
  const float kFps = 50;     // Frames per second
  const float kZoom = 2.7;
  const float kTrackRad =
      900 / kScale;  // Track is heavily morphed circle with this radius
  const float kPlayfiled = 2000 / kScale;  // Game over boundary
  const float kTrackTurnRate = 0.31;
  const float kTrackDetailStep = 21 / kScale;
  const float kTrackWidth = 40 / kScale;

  const float kBorder = 8.f / kScale;
  const int kBorderMinCount = 4;

  const float kGrassDim = kPlayfiled / 20;
  const float kMaxShapeDim =
      std::max(kGrassDim, std::max(kTrackWidth, kTrackDetailStep)) * sqrt(2.f) *
      kZoom * kScale;
  const int kCheckPoint = 12;

  friend class CarRacingFrictionDetector;

 protected:
  float lap_complete_percent_;
  int max_episode_steps_, elapsed_step_{0};
  float reward_{0};
  float prev_reward_{0};
  float step_reward_{0};
  bool done_{true};

  cv::Mat surf_;
  cv::Mat img_array_;

  std::unique_ptr<CarRacingFrictionDetector> listener_;
  std::shared_ptr<b2World> world_;
  std::unique_ptr<Car> car_;
  int tile_visited_count_{0};
  float start_alpha_{0};
  float t_{0};
  bool new_lap_{false};
  b2FixtureDef fd_tile_;
  std::vector<std::array<float, 4>> track_;
  std::vector<UserData*> roads_;
  // pair of position and color
  std::vector<std::pair<std::array<b2Vec2, 4>, cv::Scalar>> roads_poly_;

 public:
  CarRacingBox2dEnv(int max_episode_steps, float lap_complete_percent);
  void Render();

  void RenderRoad(float zoom, const std::array<float, 2>& translation,
                  float angle);
  void RenderIndicators();
  void DrawColoredPolygon(const std::array<std::array<float, 2>, 4>& field,
                          const cv::Scalar& color, float zoom,
                          const std::array<float, 2>& translation, float angle,
                          bool clip = true);
  void CarRacingReset(std::mt19937* gen);
  void CarRacingStep(std::mt19937* gen, float action0, float action1,
                     float action2);
  void CreateImageArray();

 private:
  [[nodiscard]] std::vector<cv::Point> VerticalInd(int place, int s, int h,
                                                   float val) const;
  [[nodiscard]] std::vector<cv::Point> HorizInd(int place, int s, int h,
                                                float val) const;
  void RenderIfMin(float value, const std::vector<cv::Point>& points,
                   const cv::Scalar& color);
  bool CreateTrack(std::mt19937* gen);
  void ResetBox2d(std::mt19937* gen);
  void StepBox2d(std::mt19937* gen, float action0, float action1, float action2,
                 bool isAction);
};

}  // namespace box2d

#endif  // ENVPOOL_BOX2D_CAR_RACING_ENV_H_
