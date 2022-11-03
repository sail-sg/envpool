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

#include <cmath>
#include <random>
#include <unordered_set>
#include <iostream>
#include <vector>

namespace box2d {

CarRacingFrictionDetector::CarRacingFrictionDetector(CarRacingBox2dEnv* env)
  : env_(env) {}

void CarRacingFrictionDetector::_Contact(b2Contact* contact, bool begin) {
    Tile* tile = nullptr;
    Wheel* obj = nullptr;

    auto u1 = reinterpret_cast<UserData*>(contact->GetFixtureA()->GetBody()->GetUserData().pointer);
    auto u2 = reinterpret_cast<UserData*>(contact->GetFixtureB()->GetBody()->GetUserData().pointer);

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

    tile->RoadColor = {kRoadColor[0]/255, kRoadColor[1]/255, kRoadColor[2]/255};
    
    if (begin) {
      obj->tiles.insert(tile);
      if (!tile->tileRoadVisited) {
        tile->tileRoadVisited = true;
        env_->reward_ += 1000.0 / env_->track_.size();
        env_->tile_visited_count_ += 1;
        // Lap is considered completed if enough % of the track was covered
        if (tile->idx == 0 && env_->tile_visited_count_ / env_->track_.size() > lap_complete_percent_) {
          env_->new_lap_ = true;
        }
      }
    } else {
      obj->tiles.erase(tile);
    }
  }

CarRacingBox2dEnv::CarRacingBox2dEnv(int max_episode_steps)
      : max_episode_steps_(max_episode_steps),
        world_(new b2World(b2Vec2(0.0, 0.0))) {
  b2PolygonShape shape;
  b2Vec2 vertices[4] = {b2Vec2(0, 0), b2Vec2(1, 0), b2Vec2(1, -1), b2Vec2(0, -1)};
  shape.Set(vertices, 4); 
  fd_tile_.shape = &shape;
}

bool CarRacingBox2dEnv::CreateTrack() {
  int checkpointInt = 12;
  float checkpointFloat = 12;
  // Create checkpoints
  std::vector<std::array<float, 3>> checkpoints;
  for (int c = 0; c < checkpointInt; c++) {
    float noise = 2 * M_PI * 1 / checkpointFloat / 3; // self.np_random.uniform(0, 2 * M_PI * 1 / checkpointFloat)
    float alpha = 2 * M_PI * c / checkpointFloat + noise;
    float rad = trackRAD / 3 * 2; //self.np_random.uniform(trackRAD / 3, trackRAD);

    if (c == 0) {
        alpha = 0;
        rad = 1.5 * trackRAD;
    }
    if (c == checkpointInt - 1) {
        alpha = 2 * M_PI * c / checkpointFloat;
        start_alpha_ = 2 * M_PI * (-0.5) / checkpointFloat;
        rad = 1.5 * trackRAD;
    }
    std::array<float, 3> cp = {alpha, rad *std::cos(alpha), rad * std::sin(alpha)};
    checkpoints.emplace_back(cp);
  }
  roads_.clear();
  // Go from one checkpoint to another to create track
  float x = 1.5 * trackRAD;
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
    while (true) { // Find destination from checkpoints
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
          if(dest_i % checkpoints.size() == 0) {
            break;
          }

      }
      if (!failed) {
        break;
      }
      alpha -= 2 * M_PI;
      continue;
    }
    float r1x =std::cos(beta);
    float r1y =std::sin(beta);
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
      return false; // failed
    }
    bool pass_through_start = current_track[i][0] > start_alpha_ && current_track[i - 1][0] <= start_alpha_;
    if (pass_through_start && i2 == -1) {
      i2 = i;
    } else if (pass_through_start && i1 == -1) {
      i1 = i;
      break;
    }
  }

  // if self.verbose:
  //     print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
  assert(i1 != -1);
  assert(i2 != -1);

  // todo: check current_track[i1 : i2 - 1]
  current_track = std::vector<std::array<float, 4>>(current_track.begin() + i1, current_track.begin() + i2 - 1);
  auto first_beta = current_track[0][1];
  auto first_perp_x =std::cos(first_beta);
  auto first_perp_y =std::sin(first_beta);
  // Length of perpendicular jump to put together head and tail
  auto well_glued_together = std::sqrt(
      std::pow(first_perp_x * (current_track[0][2] - current_track.back()[2]), 2)
      + std::pow(first_perp_y * (current_track[0][3] - current_track.back()[3]), 2)
  );
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
    b2Vec2 road1_l{
      static_cast<float>(x1 - kTrackWidth *std::cos(beta1)),
      static_cast<float>(y1 - kTrackWidth *std::sin(beta1))
    };
    b2Vec2 road1_r = {
      static_cast<float>(x1 + kTrackWidth *std::cos(beta1)),
      static_cast<float>(y1 + kTrackWidth *std::sin(beta1))
    };
    b2Vec2 road2_l = {
      static_cast<float>(x2 - kTrackWidth *std::cos(beta2)),
      static_cast<float>(y2 - kTrackWidth *std::sin(beta2))
    };
    b2Vec2 road2_r =  {
      static_cast<float>(x2 + kTrackWidth *std::cos(beta2)),
      static_cast<float>(y2 + kTrackWidth *std::sin(beta2))
    };
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
    t->RoadColor = {kRoadColor[0] + c, kRoadColor[1] + c, kRoadColor[2] + c};

    t->type = TILE_TYPE;
    t->tileRoadVisited = false;
    t->roadFriction = 1.0;
    t->idx = i;
    t->body->GetFixtureList()[0].SetSensor(true);
    roads_.push_back(t);
    roads_poly_.emplace_back(std::make_pair(roads_vertices, t->RoadColor));
  }
  track_ = current_track;
  return true;
}


void CarRacingBox2dEnv::CarRacingReset(std::mt19937* gen){
  ResetBox2d(gen);
  StepBox2d(gen, 0.0, 0.0, 0.0, false); // todo
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
    car_->destroy();
  }
  listener_ = std::make_unique<CarRacingFrictionDetector>(this);
  world_->SetContactListener(listener_.get());
  reward_ = 0;
  prev_reward_ = 0;
  tile_visited_count_ = 0;
  new_lap_ = false;
  t_ = 0;
  roads_poly_.clear();

  bool success = false;
  while (!success) {
    success = CreateTrack();
  }
  car_ = std::make_unique<Car>(world_, track_[0][1],track_[0][2], track_[0][3]);
}

void CarRacingBox2dEnv::CarRacingStep(std::mt19937* gen, float action0, float action1, float action2) {
  ++elapsed_step_;
  StepBox2d(gen, action0, action1, action2, true);
}
void CarRacingBox2dEnv::StepBox2d(std::mt19937* gen, float action0, float action1, float action2,
  bool isAction) {
  assert(car_ != nullptr);
  assert(-1 <= action0 && action0 <= 1);
  assert(0 <= action1 && action1 <= 1);
  assert(0 <= action2 && action2 <= 1);
  if (isAction) {
    car_->steer(-action0);
    car_->gas(action1);
    car_->brake(action2);
  }

  car_->step(1.0 / kFps);
  world_->Step(1.0 / kFps, 6*30, 2 * 30);
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
    printf("x %f y %f step_reward %f\n", x, y, step_reward_);
  }
  Render(HUMAN);
}

cv::Mat CarRacingBox2dEnv::CreateImageArray() {
  cv::Mat state;
  cv::resize(surf_, state, cv::Size(stateW, stateH));
  return state;
}
void CarRacingBox2dEnv::DrawColoredPolygon(std::array<std::array<float, 2>, 4>& field,
                                           cv::Scalar  color,
                                           float zoom, std::array<float, 2>& translation, float angle, bool clip) {

  // This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
  // Instead of calculating exactly if the polygon and screen overlap,
  // we simply check if the polygon is in a larger bounding box whose dimension
  // is greater than the screen by MAX_SHAPE_DIM, which is the maximum
  // diagonal length of an environment object
  
  bool exist = false;
  std::vector<cv::Point> poly;
  for (int i = 0; i < 4; i++) {
    field[i] = RotateRad(field[i], angle);
    field[i] = {field[i][0] * zoom + translation[0], field[i][1] * zoom + translation[1]};
    poly.push_back(cv::Point(field[i][0],field[i][1]));
    if (-kMaxShapeDim <= field[i][0] && field[i][0] <= windowW + kMaxShapeDim &&
        -kMaxShapeDim <= field[i][1] && field[i][1] <= windowH + kMaxShapeDim) {
      exist = true;
    }
  }

  if (!clip || exist) {
    cv::fillPoly(surf_, poly, color);
    // gfxdraw.aapolygon(self.surf, poly, color)
    // gfxdraw.filled_polygon(self.surf, poly, color)
  }

}

void CarRacingBox2dEnv::RenderRoad(float zoom, std::array<float, 2>& translation, float angle) {
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
  for (auto& [poly, color]: roads_poly_) {
    std::array<std::array<float, 2>, 4> field;
    field[0] = {poly[0].x, poly[0].y};
    field[1] = {poly[1].x, poly[1].y};
    field[2] = {poly[2].x, poly[2].y};
    field[3] = {poly[3].x, poly[3].y};

    cv::Scalar c(static_cast<int>(color[0]), static_cast<int>(color[1]), static_cast<int>(color[2]));
    DrawColoredPolygon(field, c, zoom, translation, angle);
  }
}
void CarRacingBox2dEnv::Render(RenderMode mode){
  surf_ = cv::Mat(windowH, windowW,  CV_8UC3);
  // self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

  assert(car_ != nullptr);
  // computing transformations
  float angle = -car_->hull_->GetAngle();
  // Animating first second zoom.
  float zoom = 0.1 * kScale * std::max(1 - t_, 0.f) + kZoom * kScale * std::min(t_, 1.f);
  float scroll_x = -car_->hull_->GetPosition().x * zoom;
  float scroll_y = -car_->hull_->GetPosition().y * zoom;

  // trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
  std::array<float, 2> trans = RotateRad({scroll_x, scroll_y}, angle);
  trans = {windowW / 2 + trans[0], windowH / 4 + trans[1]};

  RenderRoad(zoom, trans, angle);
  car_->draw(surf_, zoom, trans, angle);

  cv::flip(surf_, surf_, 0);

  // if mode == "human":
  //     pygame.event.pump()
  //     self.clock.tick(self.metadata["render_fps"])
  //     assert self.screen is not None
  //     self.screen.fill(0)
  //     self.screen.blit(self.surf, (0, 0))
  //     pygame.display.flip()

  // if mode == "rgb_array":
  //     return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
  // elif mode == "state_pixels":
  //     return self._create_image_array(self.surf, (STATE_W, STATE_H))
  // else:
  //     return self.isopen

  }
}  // namespace box2d
