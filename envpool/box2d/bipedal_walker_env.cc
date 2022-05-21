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

#include "envpool/box2d/bipedal_walker_env.h"

#include <algorithm>

#include "envpool/box2d/utils.h"

namespace box2d {

float BipedalWalkerLidarCallback::ReportFixture(b2Fixture* fixture,
                                                const b2Vec2& point,
                                                const b2Vec2& normal,
                                                float frac) {
  if ((fixture->GetFilterData().categoryBits & 1) == 0) {
    return -1;
  }
  p2 = point;
  fraction = frac;
  return frac;
}

BipedalWalkerContactDetector::BipedalWalkerContactDetector(
    BipedalWalkerBox2dEnv* env)
    : env_(env) {}

void BipedalWalkerContactDetector::BeginContact(b2Contact* contact) {
  b2Body* body_a = contact->GetFixtureA()->GetBody();
  b2Body* body_b = contact->GetFixtureB()->GetBody();
  if (env_->hull_ == body_a || env_->hull_ == body_b) {
    env_->done_ = true;
  }
  if (env_->legs_[1] == body_a || env_->legs_[1] == body_b) {
    env_->ground_contact_[1] = 1;
  }
  if (env_->legs_[3] == body_a || env_->legs_[3] == body_b) {
    env_->ground_contact_[3] = 1;
  }
}

void BipedalWalkerContactDetector::EndContact(b2Contact* contact) {
  b2Body* body_a = contact->GetFixtureA()->GetBody();
  b2Body* body_b = contact->GetFixtureB()->GetBody();
  if (env_->legs_[1] == body_a || env_->legs_[1] == body_b) {
    env_->ground_contact_[1] = 0;
  }
  if (env_->legs_[3] == body_a || env_->legs_[3] == body_b) {
    env_->ground_contact_[3] = 0;
  }
}

BipedalWalkerBox2dEnv::BipedalWalkerBox2dEnv(bool hardcore,
                                             int max_episode_steps)
    : max_episode_steps_(max_episode_steps),
      elapsed_step_(max_episode_steps + 1),
      hardcore_(hardcore),
      done_(true),
      world_(new b2World(b2Vec2(0.0, -10.0))),
      hull_(nullptr) {
  for (const auto* p : kHullPoly) {
    hull_poly_.emplace_back(Vec2(p[0] / kScale, p[1] / kScale));
  }
}

void BipedalWalkerBox2dEnv::CreateTerrain(std::vector<b2Vec2> poly) {
  b2BodyDef bd;
  bd.type = b2_staticBody;

  b2PolygonShape shape;
  shape.Set(poly.data(), poly.size());

  b2FixtureDef fd;
  fd.shape = &shape;
  fd.friction = kFriction;

  auto* t = world_->CreateBody(&bd);
  t->CreateFixture(&fd);
  terrain_.emplace_back(t);
}

void BipedalWalkerBox2dEnv::ResetBox2d(std::mt19937* gen) {
  // clean all body in world
  if (hull_ != nullptr) {
    world_->SetContactListener(nullptr);
    for (auto& t : terrain_) {
      world_->DestroyBody(t);
    }
    terrain_.clear();
    world_->DestroyBody(hull_);
    for (auto& l : legs_) {
      world_->DestroyBody(l);
    }
    lidar_.clear();
  }
  listener_ = std::make_unique<BipedalWalkerContactDetector>(this);
  world_->SetContactListener(listener_.get());

  // terrain
  {
    int state = kGrass;
    double velocity = 0.0;
    double y = kTerrainHeight;
    int counter = kTerrainStartpad;
    bool oneshot = false;
    std::vector<double> terrain_x;
    std::vector<double> terrain_y;
    double original_y = 0.0;
    int stair_steps = 0;
    int stair_width = 0;
    int stair_height = 0;
    for (int i = 0; i < kTerrainLength; ++i) {
      double x = i * kTerrainStep;
      terrain_x.emplace_back(x);
      if (state == kGrass && !oneshot) {
        velocity = 0.8 * velocity + 0.01 * Sign(kTerrainHeight - y);
        if (i > kTerrainStartpad) {
          velocity += RandUniform(-1, 1)(*gen) / kScale;
        }
        y += velocity;
      } else if (state == kPit && oneshot) {
        counter = RandInt(3, 5)(*gen);
        std::vector<b2Vec2> poly{Vec2(x, y), Vec2(x + kTerrainStep, y),
                                 Vec2(x + kTerrainStep, y - 4 * kTerrainStep),
                                 Vec2(x, y - 4 * kTerrainStep)};
        CreateTerrain(poly);
        for (auto& p : poly) {
          p = Vec2(p.x + kTerrainStep * counter, p.y);
        }
        CreateTerrain(poly);
        counter += 2;
        original_y = y;
      } else if (state == kPit && !oneshot) {
        y = original_y;
        if (counter > 1) {
          y -= 4 * kTerrainStep;
        }
      } else if (state == kStump && oneshot) {
        counter = RandInt(1, 3)(*gen);
        auto size = kTerrainStep * counter;
        std::vector<b2Vec2> poly{Vec2(x, y), Vec2(x + size, y),
                                 Vec2(x + size, y + size), Vec2(x, y + size)};
        CreateTerrain(poly);
      } else if (state == kStairs && oneshot) {
        stair_height = RandUniform(0, 1)(*gen) > 0.5 ? 1 : -1;
        stair_width = RandInt(4, 5)(*gen);
        stair_steps = RandInt(3, 5)(*gen);
        original_y = y;
        for (int s = 0; s < stair_steps; ++s) {
          std::vector<b2Vec2> poly{
              Vec2(x + kTerrainStep * s * stair_width,
                   y + kTerrainStep * s * stair_height),
              Vec2(x + kTerrainStep * (1 + s) * stair_width,
                   y + kTerrainStep * s * stair_height),
              Vec2(x + kTerrainStep * (1 + s) * stair_width,
                   y + kTerrainStep * (s * stair_height - 1)),
              Vec2(x + kTerrainStep * s * stair_width,
                   y + kTerrainStep * (s * stair_height - 1))};
          CreateTerrain(poly);
        }
        counter = stair_steps * stair_width;
      } else if (state == kStairs && !oneshot) {
        int s = stair_steps * stair_width - counter - stair_height;
        y = original_y + kTerrainStep * s / stair_width * stair_height;
      }
      oneshot = false;
      terrain_y.emplace_back(y);
      if (--counter == 0) {
        counter = RandInt(kTerrainGrass / 2, kTerrainGrass)(*gen);
        if (state == kGrass && hardcore_) {
          state = RandInt(1, kStates)(*gen);
        } else {
          state = kGrass;
        }
        oneshot = true;
      }
    }
    for (int i = 0; i < kTerrainLength - 1; ++i) {
      b2BodyDef bd;
      bd.type = b2_staticBody;

      b2EdgeShape shape;
      shape.SetTwoSided(Vec2(terrain_x[i], terrain_y[i]),
                        Vec2(terrain_x[i + 1], terrain_y[i + 1]));

      b2FixtureDef fd;
      fd.shape = &shape;
      fd.friction = kFriction;
      fd.filter.categoryBits = 0x0001;

      auto* t = world_->CreateBody(&bd);
      t->CreateFixture(&fd);
      terrain_.emplace_back(t);
    }
  }

  // hull
  double init_x = kTerrainStep * kTerrainStartpad / 2;
  double init_y = kTerrainHeight + 2 * kLegH;
  {
    b2BodyDef bd;
    bd.type = b2_dynamicBody;
    bd.position = Vec2(init_x, init_y);

    b2PolygonShape shape;
    shape.Set(hull_poly_.data(), hull_poly_.size());

    b2FixtureDef fd;
    fd.shape = &shape;
    fd.density = 5.0;
    fd.friction = 0.1;
    fd.filter.categoryBits = 0x0020;
    fd.filter.maskBits = 0x001;
    fd.restitution = 0.0;

    hull_ = world_->CreateBody(&bd);
    hull_->CreateFixture(&fd);
    b2Vec2 force = Vec2(RandUniform(-kInitialRandom, kInitialRandom)(*gen), 0);
    hull_->ApplyForceToCenter(force, true);
  }
  // leg
  for (int index = 0; index < 2; ++index) {
    float sign = index == 0 ? -1 : 1;

    // upper leg
    b2BodyDef bd;
    bd.type = b2_dynamicBody;
    bd.position = Vec2(init_x, init_y - kLegH / 2 - kLegDown);
    bd.angle = sign * 0.05f;

    b2PolygonShape shape;
    shape.SetAsBox(static_cast<float>(kLegW / 2),
                   static_cast<float>(kLegH / 2));

    b2FixtureDef fd;
    fd.shape = &shape;
    fd.density = 1.0;
    fd.filter.categoryBits = 0x0020;
    fd.filter.maskBits = 0x001;
    fd.restitution = 0.0;

    legs_[index * 2] = world_->CreateBody(&bd);
    legs_[index * 2]->CreateFixture(&fd);
    ground_contact_[index * 2] = 0;

    b2RevoluteJointDef rjd;
    rjd.bodyA = hull_;
    rjd.bodyB = legs_[index * 2];
    rjd.localAnchorA = Vec2(0, kLegDown);
    rjd.localAnchorB = Vec2(0, kLegH / 2);
    rjd.enableMotor = true;
    rjd.enableLimit = true;
    rjd.maxMotorTorque = static_cast<float>(kMotorsTorque);
    rjd.motorSpeed = sign;
    rjd.lowerAngle = -0.8;
    rjd.upperAngle = 1.1;
    joints_[index * 2] = world_->CreateJoint(&rjd);

    // lower leg
    bd.position = Vec2(init_x, init_y - kLegH * 3 / 2 - kLegDown);
    shape.SetAsBox(static_cast<float>(0.8 * kLegW / 2),
                   static_cast<float>(kLegH / 2));
    legs_[index * 2 + 1] = world_->CreateBody(&bd);
    legs_[index * 2 + 1]->CreateFixture(&fd);
    ground_contact_[index * 2 + 1] = 0;

    rjd.bodyA = legs_[index * 2];
    rjd.bodyB = legs_[index * 2 + 1];
    rjd.localAnchorA = Vec2(0, -kLegH / 2);
    rjd.localAnchorB = Vec2(0, kLegH / 2);
    rjd.motorSpeed = 1;
    rjd.lowerAngle = -1.6;
    rjd.upperAngle = -0.1;
    joints_[index * 2 + 1] = world_->CreateJoint(&rjd);
  }

  // lidar
  for (int i = 0; i < kLidarNum; ++i) {
    lidar_.emplace_back(BipedalWalkerLidarCallback());
  }
}

void BipedalWalkerBox2dEnv::StepBox2d(std::mt19937* gen, float action0,
                                      float action1, float action2,
                                      float action3) {
  if (elapsed_step_ >= max_episode_steps_) {
    done_ = true;
  }
}

void BipedalWalkerBox2dEnv::BipedalWalkerReset(std::mt19937* gen) {
  elapsed_step_ = 0;
  done_ = false;
  ResetBox2d(gen);
  StepBox2d(gen, 0, 0, 0, 0);
}

void BipedalWalkerBox2dEnv::BipedalWalkerStep(std::mt19937* gen, float action0,
                                              float action1, float action2,
                                              float action3) {
  ++elapsed_step_;
  StepBox2d(gen, action0, action1, action2, action3);
}

}  // namespace box2d
