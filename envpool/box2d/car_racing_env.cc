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
       world_->SetContactListener(listener_.get());

  b2PolygonShape shape;
  b2Vec2 vertices[4] = {b2Vec2(0, 0), b2Vec2(1, 0), b2Vec2(1, -1), b2Vec2(0, -1)};
  shape.Set(vertices, 4); 
  fd_tile_.shape = &shape;
}

bool CarRacingBox2dEnv::CreateTrack() {
  int checkpointInt = 12;
  double checkpointDouble = 12;
  // Create checkpoints
  std::vector<std::array<double, 3>> checkpoints;
  for (int c = 0; c < checkpointInt; c++) {
    double noise = 2 * M_PI * 1 / checkpointDouble / 3; // self.np_random.uniform(0, 2 * M_PI * 1 / checkpointDouble)
    double alpha = 2 * M_PI * c / checkpointDouble + noise;
    double rad = trackRAD / 3 * 2; //self.np_random.uniform(trackRAD / 3, trackRAD);

    if (c == 0) {
        alpha = 0;
        rad = 1.5 * trackRAD;
    }
    if (c == checkpointInt - 1) {
        alpha = 2 * M_PI * c / checkpointDouble;
        start_alpha_ = 2 * M_PI * (-0.5) / checkpointDouble;
        rad = 1.5 * trackRAD;
    }
    std::array<double, 3> cp = {alpha, rad *std::cos(alpha), rad * std::sin(alpha)};
    checkpoints.emplace_back(cp);
  }
  roads_.clear();
  // Go from one checkpoint to another to create track
  double x = 1.5 * trackRAD;
  double y = 0;
  double beta = 0;
  int dest_i = 0;
  int laps = 0;
  std::vector<std::array<double, 4>> current_track;
  int no_freeze = 2500;
  bool visited_other_side = false;
  while (true) {
    double alpha = std::atan2(y, x);
    if (visited_other_side && alpha > 0) {
        laps++;
        visited_other_side = false;
    }
    if (alpha < 0) {
      visited_other_side = true;
      alpha += 2 * M_PI;
    }
    double dest_alpha, dest_x, dest_y;
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
    double r1x =std::cos(beta);
    double r1y =std::sin(beta);
    double p1x = -r1y;
    double p1y = r1x;
    double dest_dx = dest_x - x;  // vector towards destination
    double dest_dy = dest_y - y;
    // destination vector projected on rad:
    double proj = r1x * dest_dx + r1y * dest_dy;
    while (beta - alpha > 1.5 * M_PI) {
      beta -= 2 * M_PI;
    }
    while (beta - alpha < -1.5 * M_PI) {
      beta += 2 * M_PI;
    }
    double prev_beta = beta;
    proj *= kScale;
    if (proj > 0.3) {
      beta -= std::min(kTrackTurnRate, abs(0.001 * proj));
    }
    if (proj < -0.3) {
      beta += std::min(kTrackTurnRate, abs(0.001 * proj));
    } 
    x += p1x * kTrackDetailStep;
    y += p1y * kTrackDetailStep;
    std::array<double, 4> track = {alpha, prev_beta * 0.5 + beta * 0.5, x, y};
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
  current_track = std::vector<std::array<double, 4>>(current_track.begin() + i1, current_track.begin() + i2 - 1);
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

    t->type = TILE_TYPE;
    t->tileRoadVisited = false;
    t->roadFriction = 1.0;
    t->idx = i;
    t->body->GetFixtureList()[0].SetSensor(true);
    roads_.push_back(t);
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
}

}  // namespace box2d
