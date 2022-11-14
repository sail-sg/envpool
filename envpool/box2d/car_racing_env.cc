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

#include "car_racing_env.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace box2d {

CarRacingFrictionDetector::CarRacingFrictionDetector(CarRacingBox2dEnv* env,
                                                     float lap_complete_percent)
    : env_(env), lap_complete_percent_(lap_complete_percent) {}

void CarRacingFrictionDetector::_Contact(b2Contact* contact, bool begin) {
  Tile* tile = nullptr;
  Wheel* obj = nullptr;

  auto u1 = reinterpret_cast<UserData*>(
      contact->GetFixtureA()->GetBody()->GetUserData().pointer);
  auto u2 = reinterpret_cast<UserData*>(
      contact->GetFixtureB()->GetBody()->GetUserData().pointer);

  if (u1 == nullptr || u2 == nullptr) {
    return;
  }

  if (u1->type != WHEEL_TYPE && u1->type != TILE_TYPE) return;
  if (u2->type != WHEEL_TYPE && u2->type != TILE_TYPE) return;

  if (u1->type == TILE_TYPE) {
    tile = reinterpret_cast<Tile*>(u1);
    obj = reinterpret_cast<Wheel*>(u2);
  }
  if (u2->type == TILE_TYPE) {
    tile = reinterpret_cast<Tile*>(u2);
    obj = reinterpret_cast<Wheel*>(u1);
  }

  if (tile->type != TILE_TYPE || obj->type != WHEEL_TYPE) return;

  tile->road_color = {kRoadColor[0], kRoadColor[1], kRoadColor[2]};
  if (begin) {
    obj->tiles.insert(tile);
    if (!tile->tile_road_visited) {
      tile->tile_road_visited = true;
      env_->reward_ += 1000.0 / env_->track_.size();
      env_->tile_visited_count_ += 1;
      // Lap is considered completed if enough % of the track was covered
      if (tile->idx == 0 && env_->tile_visited_count_ / env_->track_.size() >
                                lap_complete_percent_) {
        env_->new_lap_ = true;
      }
    }
  } else {
    obj->tiles.erase(tile);
  }
}

CarRacingBox2dEnv::CarRacingBox2dEnv(int max_episode_steps,
                                     float lap_complete_percent)
    : lap_complete_percent_(lap_complete_percent),
      max_episode_steps_(max_episode_steps),
      elapsed_step_(max_episode_steps + 1),
      done_(true),
      world_(new b2World(b2Vec2(0.0, 0.0))) {
  b2PolygonShape shape;
  b2Vec2 vertices[4] = {b2Vec2(0, 0), b2Vec2(1, 0), b2Vec2(1, -1),
                        b2Vec2(0, -1)};
  shape.Set(vertices, 4);
  fd_tile_.shape = &shape;
}

bool CarRacingBox2dEnv::CreateTrack(std::mt19937* gen) {
  // Create checkpoints
  std::vector<std::array<float, 3>> checkpoints;
  for (int c = 0; c < checkpointInt; c++) {
    float noise = RandUniform(0, 2 * M_PI * 1 / checkpointFloat)(*gen);
    float alpha = 2 * M_PI * c / checkpointFloat + noise;
    float rad = RandUniform(kTrackRad / 3, kTrackRad)(*gen);

    if (c == 0) {
      alpha = 0;
      rad = 1.5 * kTrackRad;
    }
    if (c == checkpointInt - 1) {
      alpha = 2 * M_PI * c / checkpointFloat;
      start_alpha_ = -M_PI / checkpointFloat;
      rad = 1.5 * kTrackRad;
    }
    std::array<float, 3> cp = {alpha, rad * std::cos(alpha),
                               rad * std::sin(alpha)};
    checkpoints.emplace_back(cp);
  }
  roads_.clear();
  // Go from one checkpoint to another to create track
  float x = 1.5 * kTrackRad;
  float y = 0;
  float beta = 0;
  int dest_i = 0;
  int laps = 0;
  std::vector<std::array<float, 4>> current_track;
  int no_freeze = 2500;
  bool visited_other_side = false;
  while (true) {
    float alpha = std::atan2(y, x);
    if (visited_other_side && alpha > 0) {
      laps++;
      visited_other_side = false;
    }
    if (alpha < 0) {
      visited_other_side = true;
      alpha += 2 * M_PI;
    }
    float dest_alpha, dest_x, dest_y;
    while (true) {  // Find destination from checkpoints
      bool failed = true;
      while (true) {
        dest_alpha = checkpoints[dest_i % checkpoints.size()][0];
        dest_x = checkpoints[dest_i % checkpoints.size()][1];
        dest_y = checkpoints[dest_i % checkpoints.size()][2];
        if (alpha <= dest_alpha) {
          failed = false;
          break;
        }
        dest_i++;
        if (dest_i % checkpoints.size() == 0) {
          break;
        }
      }
      if (!failed) {
        break;
      }
      alpha -= 2 * M_PI;
      continue;
    }
    float r1x = std::cos(beta);
    float r1y = std::sin(beta);
    float p1x = -r1y;
    float p1y = r1x;
    float dest_dx = dest_x - x;  // vector towards destination
    float dest_dy = dest_y - y;
    // destination vector projected on rad:
    float proj = r1x * dest_dx + r1y * dest_dy;
    while (beta - alpha > 1.5 * M_PI) {
      beta -= 2 * M_PI;
    }
    while (beta - alpha < -1.5 * M_PI) {
      beta += 2 * M_PI;
    }
    float prev_beta = beta;
    proj *= kScale;
    if (proj > 0.3) {
      beta -= std::min(kTrackTurnRate, abs(0.001f * proj));
    }
    if (proj < -0.3) {
      beta += std::min(kTrackTurnRate, abs(0.001f * proj));
    }
    x += p1x * kTrackDetailStep;
    y += p1y * kTrackDetailStep;
    std::array<float, 4> track = {alpha, prev_beta * 0.5f + beta * 0.5f, x, y};
    current_track.emplace_back(track);
    if (laps > 4) {
      break;
    }
    no_freeze--;
    if (no_freeze == 0) {
      break;
    }
  }

  // Find closed loop range i1..i2, first loop should be ignored, second is OK
  int i1 = -1;
  int i2 = -1;
  int i = static_cast<int>(current_track.size());
  while (true) {
    i--;
    if (i == 0) {
      return false;  // failed
    }
    bool pass_through_start = current_track[i][0] > start_alpha_ &&
                              current_track[i - 1][0] <= start_alpha_;
    if (pass_through_start && i2 == -1) {
      i2 = i;
    } else if (pass_through_start && i1 == -1) {
      i1 = i;
      break;
    }
  }

  assert(i1 != -1);
  assert(i2 != -1);

  current_track = std::vector<std::array<float, 4>>(
      current_track.begin() + i1, current_track.begin() + i2 - 1);
  auto first_beta = current_track[0][1];
  auto first_perp_x = std::cos(first_beta);
  auto first_perp_y = std::sin(first_beta);
  // Length of perpendicular jump to put together head and tail
  auto well_glued_together = std::sqrt(
      std::pow(first_perp_x * (current_track[0][2] - current_track.back()[2]),
               2) +
      std::pow(first_perp_y * (current_track[0][3] - current_track.back()[3]),
               2));
  if (well_glued_together > kTrackDetailStep) {
    return false;
  }
  // Create tiles
  for (int i = 0; i < static_cast<int>(current_track.size()); i++) {
    auto [alpha1, beta1, x1, y1] = current_track[i];
    int last_i = i - 1;
    if (last_i < 0) {
      last_i = current_track.size() - 1;
    }
    auto [alpha2, beta2, x2, y2] = current_track[last_i];
    b2Vec2 road1_l{static_cast<float>(x1 - kTrackWidth * std::cos(beta1)),
                   static_cast<float>(y1 - kTrackWidth * std::sin(beta1))};
    b2Vec2 road1_r = {static_cast<float>(x1 + kTrackWidth * std::cos(beta1)),
                      static_cast<float>(y1 + kTrackWidth * std::sin(beta1))};
    b2Vec2 road2_l = {static_cast<float>(x2 - kTrackWidth * std::cos(beta2)),
                      static_cast<float>(y2 - kTrackWidth * std::sin(beta2))};
    b2Vec2 road2_r = {static_cast<float>(x2 + kTrackWidth * std::cos(beta2)),
                      static_cast<float>(y2 + kTrackWidth * std::sin(beta2))};
    b2Vec2 vertices[4] = {road1_l, road1_r, road2_r, road2_l};
    std::array<b2Vec2, 4> roads_vertices = {road1_l, road1_r, road2_r, road2_l};
    b2PolygonShape shape;
    shape.Set(vertices, 4);
    fd_tile_.shape = &shape;

    b2BodyDef bd;
    bd.type = b2_staticBody;

    auto* t = new Tile();
    t->body = world_->CreateBody(&bd);
    t->body->CreateFixture(&fd_tile_);

    // t->body->SetUserData(t); // recently removed from 2.4.1
    t->body->GetUserData().pointer = reinterpret_cast<uintptr_t>(t);

    float c = 0.01 * (i % 3) * 255;
    t->road_color = {kRoadColor[0] + c, kRoadColor[1] + c, kRoadColor[2] + c};

    t->type = TILE_TYPE;
    t->tile_road_visited = false;
    t->road_friction = 1.0;
    t->idx = i;
    t->body->GetFixtureList()[0].SetSensor(true);
    roads_.push_back(t);
    roads_poly_.emplace_back(std::make_pair(roads_vertices, t->road_color));
  }
  track_ = current_track;
  return true;
}

void CarRacingBox2dEnv::CarRacingReset(std::mt19937* gen) {
  elapsed_step_ = 0;
  done_ = false;
  ResetBox2d(gen);
  StepBox2d(gen, 0.0, 0.0, 0.0, false);
}

void CarRacingBox2dEnv::ResetBox2d(std::mt19937* gen) {
  // clean all body in world
  if (!roads_.empty()) {
    world_->SetContactListener(nullptr);
    for (auto& t : roads_) {
      world_->DestroyBody(t->body);
    }
    roads_.clear();
    assert(car_ != nullptr);
    car_->Destroy();
  }
  listener_ =
      std::make_unique<CarRacingFrictionDetector>(this, lap_complete_percent_);
  world_->SetContactListener(listener_.get());
  reward_ = 0;
  prev_reward_ = 0;
  tile_visited_count_ = 0;
  new_lap_ = false;
  t_ = 0;
  roads_poly_.clear();

  bool success = false;
  while (!success) {
    success = CreateTrack(gen);
  }
  car_ =
      std::make_unique<Car>(world_, track_[0][1], track_[0][2], track_[0][3]);
}

void CarRacingBox2dEnv::CarRacingStep(std::mt19937* gen, float action0,
                                      float action1, float action2) {
  ++elapsed_step_;
  StepBox2d(gen, action0, action1, action2, true);
}

void CarRacingBox2dEnv::StepBox2d(std::mt19937* gen, float action0,
                                  float action1, float action2, bool isAction) {
  assert(car_ != nullptr);
  assert(-1 <= action0 && action0 <= 1);
  assert(0 <= action1 && action1 <= 1);
  assert(0 <= action2 && action2 <= 1);
  if (isAction) {
    car_->Steer(-action0);
    car_->Gas(action1);
    car_->Brake(action2);
  }

  car_->Step(1.0 / kFps);
  world_->Step(1.0 / kFps, 6 * 30, 2 * 30);
  t_ += 1.0 / kFps;

  step_reward_ = 0;
  done_ = false;
  // First step without action, called from reset()
  if (isAction) {
    reward_ -= 0.1;
    // We actually don't want to count fuel spent, we want car to be faster.
    car_->fuel_spent_ = 0.0;
    step_reward_ = reward_ - prev_reward_;
    prev_reward_ = reward_;
    if (tile_visited_count_ == static_cast<int>(track_.size()) || new_lap_) {
      // Truncation due to finishing lap
      // This should not be treated as a failure
      // but like a timeout
      // truncated = true;
      done_ = true;
    }
    float x = car_->hull_->GetPosition().x;
    float y = car_->hull_->GetPosition().y;

    if (abs(x) > kPlayfiled || abs(y) > kPlayfiled) {
      // terminated = true;
      done_ = true;
      step_reward_ = -100;
    }

    if (elapsed_step_ >= max_episode_steps_) {
      done_ = true;
    }
  }
  Render(HUMAN);
}

void CarRacingBox2dEnv::CreateImageArray() {
  // cv::resize(surf_, img_array_, cv::Size(stateW, stateH), 0, 0,
  // cv::INTER_AREA);
  cv::resize(surf_, img_array_, cv::Size(stateW, stateH));
  cv::cvtColor(img_array_, img_array_, cv::COLOR_BGR2RGB);
}

void CarRacingBox2dEnv::DrawColoredPolygon(
    const std::array<std::array<float, 2>, 4>& field, cv::Scalar color,
    float zoom, const std::array<float, 2>& translation, float angle,
    bool clip) {
  // This checks if the polygon is out of bounds of the screen, and we skip
  // drawing if so. Instead of calculating exactly if the polygon and screen
  // overlap, we simply check if the polygon is in a larger bounding box whose
  // dimension is greater than the screen by MAX_SHAPE_DIM, which is the maximum
  // diagonal length of an environment object

  bool exist = false;
  std::vector<cv::Point> poly;
  for (const auto& f : field) {
    auto f_roated = RotateRad(f, angle);
    f_roated = {f_roated[0] * zoom + translation[0],
                f_roated[1] * zoom + translation[1]};
    poly.push_back(cv::Point(f_roated[0], f_roated[1]));
    if (-kMaxShapeDim <= f_roated[0] && f_roated[0] <= windowW + kMaxShapeDim &&
        -kMaxShapeDim <= f_roated[1] && f_roated[1] <= windowH + kMaxShapeDim) {
      exist = true;
    }
  }

  if (!clip || exist) {
    cv::fillPoly(surf_, poly, color);
  }
}

void CarRacingBox2dEnv::RenderRoad(float zoom,
                                   const std::array<float, 2>& translation,
                                   float angle) {
  std::array<std::array<float, 2>, 4> field;
  field[0] = {kPlayfiled, kPlayfiled};
  field[1] = {kPlayfiled, -kPlayfiled};
  field[2] = {-kPlayfiled, -kPlayfiled};
  field[3] = {-kPlayfiled, kPlayfiled};

  // draw background
  DrawColoredPolygon(field, kBgColor, zoom, translation, angle, false);

  // draw grass patches
  std::vector<std::array<std::array<float, 2>, 4>> grass;
  for (int x = -20; x < 20; x += 2) {
    for (int y = -20; y < 20; y += 2) {
      std::array<std::array<float, 2>, 4> grass;

      grass[0] = {kGrassDim * x + kGrassDim, kGrassDim * y + 0};
      grass[1] = {kGrassDim * x + 0, kGrassDim * y + 0};
      grass[2] = {kGrassDim * x + 0, kGrassDim * y + kGrassDim};
      grass[3] = {kGrassDim * x + kGrassDim, kGrassDim * y + kGrassDim};
      DrawColoredPolygon(grass, kGrassColor, zoom, translation, angle);
    }
  }

  // draw road
  for (auto& [poly, color] : roads_poly_) {
    std::array<std::array<float, 2>, 4> field;
    field[0] = {poly[0].x, poly[0].y};
    field[1] = {poly[1].x, poly[1].y};
    field[2] = {poly[2].x, poly[2].y};
    field[3] = {poly[3].x, poly[3].y};

    cv::Scalar c(static_cast<int>(color[0]), static_cast<int>(color[1]),
                 static_cast<int>(color[2]));
    DrawColoredPolygon(field, c, zoom, translation, angle);
  }
}

std::vector<cv::Point> CarRacingBox2dEnv::VerticalInd(int place, int s, int h,
                                                      float val) {
  return {
      cv::Point(place * s, windowH - static_cast<int>(h + h * val)),
      cv::Point((place + 1) * s, windowH - static_cast<int>(h + h * val)),
      cv::Point((place + 1) * s, windowH - h),
      cv::Point((place + 0) * s, windowH - h),
  };
}

std::vector<cv::Point> CarRacingBox2dEnv::HorizInd(int place, int s, int h,
                                                   float val) {
  return {
      cv::Point((place + 0) * s, windowH - 4 * h),
      cv::Point((place + val) * s, windowH - 4 * h),
      cv::Point((place + val) * s, windowH - 2 * h),
      cv::Point((place + 0) * s, windowH - 2 * h),
  };
}

void CarRacingBox2dEnv::RenderIfMin(float value, std::vector<cv::Point> points,
                                    cv::Scalar color) {
  if (abs(value) > 1e-4) {
    cv::fillPoly(surf_, points, color);
  }
}

void CarRacingBox2dEnv::RenderIndicators() {
  int h = windowH / 40;
  int s = windowW / 40;
  std::vector<cv::Point> poly = {
      cv::Point(windowW, windowH), cv::Point(windowW, windowH - 5 * h),
      cv::Point(0, windowH - 5 * h), cv::Point(0, windowH)};
  cv::fillPoly(surf_, poly, cv::Scalar(0, 0, 0));

  assert(car_ != nullptr);

  float true_speed = std::sqrt(std::pow(car_->hull_->GetLinearVelocity().x, 2) +
                               std::pow(car_->hull_->GetLinearVelocity().y, 2));

  RenderIfMin(true_speed, VerticalInd(5, s, h, 0.02f * true_speed),
              cv::Scalar(255, 255, 255));
  // ABS sensors
  RenderIfMin(car_->wheels_[0]->omega,
              VerticalInd(7, s, h, 0.01 * car_->wheels_[0]->omega),
              cv::Scalar(255, 0, 0));
  RenderIfMin(car_->wheels_[1]->omega,
              VerticalInd(8, s, h, 0.01 * car_->wheels_[1]->omega),
              cv::Scalar(255, 0, 0));
  RenderIfMin(car_->wheels_[2]->omega,
              VerticalInd(9, s, h, 0.01 * car_->wheels_[2]->omega),
              cv::Scalar(255, 0, 51));
  RenderIfMin(car_->wheels_[3]->omega,
              VerticalInd(10, s, h, 0.01 * car_->wheels_[3]->omega),
              cv::Scalar(255, 0, 51));

  RenderIfMin(
      car_->wheels_[0]->joint->GetJointAngle(),
      HorizInd(20, s, h, -10.0 * car_->wheels_[0]->joint->GetJointAngle()),
      cv::Scalar(0, 255, 0));
  RenderIfMin(car_->hull_->GetAngularVelocity(),
              HorizInd(30, s, h, -0.8f * car_->hull_->GetAngularVelocity()),
              cv::Scalar(0, 0, 255));
}

void CarRacingBox2dEnv::Render(RenderMode mode) {
  surf_ = cv::Mat(windowH, windowW, CV_8UC3);

  assert(car_ != nullptr);
  // computing transformations
  float angle = -car_->hull_->GetAngle();
  // Animating first second zoom.
  float zoom =
      0.1 * kScale * std::max(1 - t_, 0.f) + kZoom * kScale * std::min(t_, 1.f);
  float scroll_x = -car_->hull_->GetPosition().x * zoom;
  float scroll_y = -car_->hull_->GetPosition().y * zoom;

  std::array<float, 2> scroll = {scroll_x, scroll_y};
  std::array<float, 2> trans = RotateRad(scroll, angle);
  trans = {windowW / 2 + trans[0], windowH / 4 + trans[1]};

  RenderRoad(zoom, trans, angle);
  car_->Draw(surf_, zoom, trans, angle);

  cv::flip(surf_, surf_, 0);
  RenderIndicators();

  auto reward = static_cast<int>(reward_);
  cv::putText(surf_, cv::format("%04d", reward),
              cv::Point(20, windowH - windowH * 2 / 40.0),
              cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 255, 255), 2, false);
}

}  // namespace box2d
