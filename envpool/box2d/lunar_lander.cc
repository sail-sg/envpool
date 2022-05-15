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
      dist_(0, 1) {
  for (int i = 0; i < 6; ++i) {
    lander_poly_.emplace_back(
        b2Vec2(kLanderPoly[i][0] / kScale, kLanderPoly[i][1] / kScale));
  }
}

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
  std::array<double, kChunks> chunk_x;
  std::array<double, kChunks> smooth_y;
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

    b2EdgeShape shape;
    shape.SetTwoSided(b2Vec2(0, 0), b2Vec2(w, 0));

    moon_ = world_->CreateBody(&bd);
    moon_->CreateFixture(&shape, 0);
  }
  for (int i = 0; i < kChunks - 1; ++i) {
    b2EdgeShape shape;
    shape.SetTwoSided(b2Vec2(chunk_x[i], smooth_y[i]),
                      b2Vec2(chunk_x[i + 1], smooth_y[i + 1]));

    b2FixtureDef fd;
    fd.shape = &shape;
    fd.friction = 0.1;
    fd.density = 0;

    moon_->CreateFixture(&fd);
  }

  // lander
  double initial_x = kViewportW / kScale / 2;
  double initial_y = kViewportH / kScale;
  {
    b2BodyDef bd;
    bd.type = b2_dynamicBody;
    bd.position.Set(initial_x, initial_y);
    bd.angle = 0;

    b2PolygonShape polygon;
    polygon.Set(lander_poly_.data(), lander_poly_.size());

    b2FixtureDef fd;
    fd.shape = &polygon;
    fd.density = 5.0;
    fd.friction = 0.1;
    fd.filter.categoryBits = 0x0010;
    fd.filter.maskBits = 0x001;
    fd.restitution = 0.0;

    lander_ = world_->CreateBody(&bd);
    lander_->CreateFixture(&fd);
    b2Vec2 force(dist_(*gen) * 2 * kInitialRandom - kInitialRandom,
                 dist_(*gen) * 2 * kInitialRandom - kInitialRandom);
    lander_->ApplyForceToCenter(force, true);
  }
  // legs
  for (int index = 0; index < 2; ++index) {
    int i = index == 0 ? -1 : 1;

    b2BodyDef bd;
    bd.type = b2_dynamicBody;
    bd.position.Set(initial_x - i * kLegAway, initial_y);
    bd.angle = i * 0.05;

    b2PolygonShape polygon;
    polygon.SetAsBox(kLegW / kScale, kLegH / kScale);

    b2FixtureDef fd;
    fd.shape = &polygon;
    fd.density = 1.0;
    fd.filter.categoryBits = 0x0020;
    fd.filter.maskBits = 0x001;
    fd.restitution = 0.0;

    legs_[index] = world_->CreateBody(&bd);
    legs_[index]->CreateFixture(&fd);
    ground_contact_[index] = false;

    b2RevoluteJointDef rjd;
    rjd.bodyA = lander_;
    rjd.bodyB = legs_[index];
    rjd.localAnchorA.Set(0, 0);
    rjd.localAnchorB.Set(i * kLegAway / kScale, kLegDown / kScale);
    rjd.enableMotor = true;
    rjd.enableLimit = true;
    rjd.maxMotorTorque = kLegSpringTorque;
    rjd.motorSpeed = 0.3 * i;
    rjd.lowerAngle = i == -1 ? 0.4 : -0.9;
    rjd.upperAngle = i == -1 ? 0.9 : -0.4;
    world_->CreateJoint(&rjd);
  }
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
