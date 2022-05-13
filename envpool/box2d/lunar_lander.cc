// Copyright 2022 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "envpool/box2d/lunar_lander.h"

namespace box2d {

ContactDetector::ContactDetector(LunarLanderEnv* env) : env_(env) {}

void ContactDetector::BeginContact(b2Contact* contact) {
  b2Body* body_a = contact->GetFixtureA()->GetBody();
  b2Body* body_b = contact->GetFixtureB()->GetBody();
  if (env_->lander_ == body_a || env_->lander_ == body_b) {
    env_->done_ = true;
  }
  if (env_->legs_[0] == body_a || env_->legs_[0] == body_b) {
    env_->ground_contact_[0] = true;
  }
  if (env_->legs_[1] == body_a || env_->legs_[1] == body_b) {
    env_->ground_contact_[1] = true;
  }
}

void ContactDetector::EndContact(b2Contact* contact) {
  b2Body* body_a = contact->GetFixtureA()->GetBody();
  b2Body* body_b = contact->GetFixtureB()->GetBody();
  if (env_->legs_[0] == body_a || env_->legs_[0] == body_b) {
    env_->ground_contact_[0] = false;
  }
  if (env_->legs_[1] == body_a || env_->legs_[1] == body_b) {
    env_->ground_contact_[1] = false;
  }
}

LunarLanderEnv::LunarLanderEnv(bool continuous, int max_episode_steps)
    : max_episode_steps_(max_episode_steps),
      elapsed_step_(max_episode_steps + 1),
      continuous_(continuous),
      done_(true),
      world_(new b2World(b2Vec2(0.0, -10.0))),
      moon_(nullptr),
      lander_(nullptr),
      dist_(0, 1) {}

void LunarLanderEnv::ResetBox2d(std::mt19937* gen) {
  // clean all body in world
  if (moon_ != nullptr) {
    world_->SetContactListener(nullptr);
    for (auto& p : particles_) {
      world_->DestroyBody(p);
    }
    particles_.clear();
    world_->DestroyBody(moon_);
    world_->DestroyBody(lander_);
    world_->DestroyBody(legs_[0]);
    world_->DestroyBody(legs_[1]);
  }
  listener_.reset(new ContactDetector(this));
  world_->SetContactListener(listener_.get());
  double h = kViewportH / kScale;
  double w = kViewportW / kScale;
  // moon
  std::array<double, kChunks + 1> height;
  std::array<double, kChunks> chunk_x, smooth_y;
  helipad_y_ = h / 4;
  for (int i = 0; i <= kChunks; ++i) {
    if (kChunks / 2 - 2 <= i && i <= kChunks / 2 + 2) {
      height[i] = helipad_y_;
    } else {
      height[i] = dist_(*gen) * h / 2;
    }
  }
  for (int i = 0; i < kChunks; ++i) {
    chunk_x[i] = w / (kChunks - 1) * i;
    smooth_y[i] =
        (height[i == 0 ? kChunks : i - 1] + height[i] + height[i + 1]) / 3;
  }
  {
    b2BodyDef bd;
    bd.type = b2_staticBody;
    moon_ = world_->CreateBody(&bd);
    b2EdgeShape shape;
    shape.SetTwoSided(b2Vec2(0, 0), b2Vec2(w, 0));
    moon_->CreateFixture(&shape, 0);
  }
  for (int i = 0; i < kChunks - 1; ++i) {
    b2EdgeShape shape;
    shape.SetTwoSided(b2Vec2(chunk_x[i], smooth_y[i]),
                      b2Vec2(chunk_x[i + 1], smooth_y[i + 1]));
    b2FixtureDef bd;
    bd.shape = &shape;
    bd.friction = 0.1;
    bd.density = 0;
    moon_->CreateFixture(&bd);
  }
  // lander
  // double initial_y = kViewportH / kScale;
}

void LunarLanderEnv::LunarLanderReset(std::mt19937* gen) {
  elapsed_step_ = -1;  // because of the step(0)
  done_ = false;
  ResetBox2d(gen);
  LunarLanderStep(gen, 0, 0, 0);
}

void LunarLanderEnv::LunarLanderStep(std::mt19937* gen, int action0,
                                     float action1, float action2) {
  ++elapsed_step_;
}

}  // namespace box2d
