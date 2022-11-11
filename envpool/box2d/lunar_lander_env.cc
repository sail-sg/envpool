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

#include "envpool/box2d/lunar_lander_env.h"

#include <algorithm>

#include "envpool/box2d/utils.h"

namespace box2d {

LunarLanderContactDetector::LunarLanderContactDetector(LunarLanderBox2dEnv* env)
    : env_(env) {}

void LunarLanderContactDetector::BeginContact(b2Contact* contact) {
  b2Body* body_a = contact->GetFixtureA()->GetBody();
  b2Body* body_b = contact->GetFixtureB()->GetBody();
  if (env_->lander_ == body_a || env_->lander_ == body_b) {
    env_->done_ = true;
  }
  if (env_->legs_[0] == body_a || env_->legs_[0] == body_b) {
    env_->ground_contact_[0] = 1;
  }
  if (env_->legs_[1] == body_a || env_->legs_[1] == body_b) {
    env_->ground_contact_[1] = 1;
  }
}

void LunarLanderContactDetector::EndContact(b2Contact* contact) {
  b2Body* body_a = contact->GetFixtureA()->GetBody();
  b2Body* body_b = contact->GetFixtureB()->GetBody();
  if (env_->legs_[0] == body_a || env_->legs_[0] == body_b) {
    env_->ground_contact_[0] = 0;
  }
  if (env_->legs_[1] == body_a || env_->legs_[1] == body_b) {
    env_->ground_contact_[1] = 0;
  }
}

LunarLanderBox2dEnv::LunarLanderBox2dEnv(bool continuous, int max_episode_steps)
    : max_episode_steps_(max_episode_steps),
      elapsed_step_(max_episode_steps + 1),
      continuous_(continuous),
      done_(true),
      world_(new b2World(b2Vec2(0.0, -10.0))),
      moon_(nullptr),
      lander_(nullptr) {
  for (const auto* p : kLanderPoly) {
    lander_poly_.emplace_back(Vec2(p[0] / kScale, p[1] / kScale));
  }
}

void LunarLanderBox2dEnv::ResetBox2d(std::mt19937* gen) {
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
  listener_ = std::make_unique<LunarLanderContactDetector>(this);
  world_->SetContactListener(listener_.get());
  double w = kViewportW / kScale;
  double h = kViewportH / kScale;

  // moon
  std::array<double, kChunks + 1> height;
  std::array<double, kChunks> chunk_x;
  std::array<double, kChunks> smooth_y;
  double helipad_y = h / 4;
  for (int i = 0; i <= kChunks; ++i) {
    if (kChunks / 2 - 2 <= i && i <= kChunks / 2 + 2) {
      height[i] = helipad_y;
    } else {
      height[i] = RandUniform(0, h / 2)(*gen);
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
    shape.SetTwoSided(b2Vec2(0, 0), Vec2(w, 0));

    b2FixtureDef fd;
    fd.shape = &shape;

    moon_ = world_->CreateBody(&bd);
    moon_->CreateFixture(&fd);
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
  double initial_x = w / 2;
  double initial_y = h;
  {
    b2BodyDef bd;
    bd.type = b2_dynamicBody;
    bd.position = Vec2(initial_x, initial_y);
    bd.angle = 0.0;

    b2PolygonShape shape;
    shape.Set(lander_poly_.data(), lander_poly_.size());

    b2FixtureDef fd;
    fd.shape = &shape;
    fd.density = 5.0;
    fd.friction = 0.1;
    fd.filter.categoryBits = 0x0010;
    fd.filter.maskBits = 0x001;
    fd.restitution = 0.0;

    lander_ = world_->CreateBody(&bd);
    lander_->CreateFixture(&fd);
    b2Vec2 force = Vec2(RandUniform(-kInitialRandom, kInitialRandom)(*gen),
                        RandUniform(-kInitialRandom, kInitialRandom)(*gen));
    lander_->ApplyForceToCenter(force, true);
  }

  // legs
  for (int index = 0; index < 2; ++index) {
    float sign = index == 0 ? -1 : 1;

    b2BodyDef bd;
    bd.type = b2_dynamicBody;
    bd.position = Vec2(initial_x - sign * kLegAway / kScale, initial_y);
    bd.angle = sign * 0.05f;

    b2PolygonShape shape;
    shape.SetAsBox(static_cast<float>(kLegW / kScale),
                   static_cast<float>(kLegH / kScale));

    b2FixtureDef fd;
    fd.shape = &shape;
    fd.density = 1.0;
    fd.filter.categoryBits = 0x0020;
    fd.filter.maskBits = 0x001;
    fd.restitution = 0.0;

    legs_[index] = world_->CreateBody(&bd);
    legs_[index]->CreateFixture(&fd);
    ground_contact_[index] = 0;

    b2RevoluteJointDef rjd;
    rjd.bodyA = lander_;
    rjd.bodyB = legs_[index];
    rjd.localAnchorA.SetZero();
    rjd.localAnchorB = Vec2(sign * kLegAway / kScale, kLegDown / kScale);
    rjd.referenceAngle = sign * 0.05f;
    rjd.enableMotor = true;
    rjd.enableLimit = true;
    rjd.maxMotorTorque = static_cast<float>(kLegSpringTorque);
    rjd.motorSpeed = sign * 0.3f;
    rjd.lowerAngle = index == 0 ? 0.4 : -0.9;
    rjd.upperAngle = index == 0 ? 0.9 : -0.4;
    world_->CreateJoint(&rjd);
  }
}

b2Body* LunarLanderBox2dEnv::CreateParticle(float mass, b2Vec2 pos) {
  b2BodyDef bd;
  bd.type = b2_dynamicBody;
  bd.position = pos;
  bd.angle = 0.0;

  b2CircleShape shape;
  shape.m_radius = static_cast<float>(2 / kScale);
  shape.m_p.SetZero();

  b2FixtureDef fd;
  fd.shape = &shape;
  fd.density = mass;
  fd.friction = 0.1;
  fd.filter.categoryBits = 0x0100;
  fd.filter.maskBits = 0x001;
  fd.restitution = 0.3;

  auto* p = world_->CreateBody(&bd);
  p->CreateFixture(&fd);
  particles_.emplace_back(p);
  return p;
}

void LunarLanderBox2dEnv::StepBox2d(std::mt19937* gen, int action,
                                    float action0, float action1) {
  action0 = std::min(std::max(action0, -1.0f), 1.0f);
  action1 = std::min(std::max(action1, -1.0f), 1.0f);
  std::array<double, 2> tip;
  std::array<double, 2> side;
  std::array<double, 2> dispersion;
  tip[0] = std::sin(lander_->GetAngle());
  tip[1] = std::cos(lander_->GetAngle());
  side[0] = -tip[1];
  side[1] = tip[0];
  dispersion[0] = RandUniform(-1, 1)(*gen) / kScale;
  dispersion[1] = RandUniform(-1, 1)(*gen) / kScale;

  // main engine
  double m_power = 0.0;
  if ((continuous_ && action0 > 0) || (!continuous_ && action == 2)) {
    if (continuous_) {
      m_power = (std::min(std::max(action0, 0.0f), 1.0f) + 1) * 0.5;
    } else {
      m_power = 1.0;
    }
    double tmp = 4 / kScale + 2 * dispersion[0];
    double ox = tip[0] * tmp + side[0] * dispersion[1];
    double oy = -tip[1] * tmp - side[1] * dispersion[1];
    auto impulse_pos = Vec2(ox, oy);
    impulse_pos += lander_->GetPosition();
    auto* p = CreateParticle(3.5, impulse_pos);
    auto impulse =
        Vec2(ox * kMainEnginePower * m_power, oy * kMainEnginePower * m_power);
    p->ApplyLinearImpulse(impulse, impulse_pos, true);
    lander_->ApplyLinearImpulse(-impulse, impulse_pos, true);
  }

  // orientation engines
  double s_power = 0.0;
  if ((continuous_ && std::abs(action1) > 0.5) ||
      (!continuous_ && (action == 1 || action == 3))) {
    double direction;
    if (continuous_) {
      direction = Sign(action1);
      s_power = std::min(std::max(std::abs(action1), 0.5f), 1.0f);
    } else {
      direction = action - 2;
      s_power = 1.0;
    }
    double tmp = 3 * dispersion[1] + direction * kSideEngineAway / kScale;
    double ox = tip[0] * dispersion[0] + side[0] * tmp;
    double oy = -tip[1] * dispersion[0] - side[1] * tmp;
    auto impulse_pos = Vec2(ox - tip[0] * 17 / kScale,
                            oy + tip[1] * kSideEngineHeight / kScale);
    impulse_pos += lander_->GetPosition();
    auto* p = CreateParticle(0.7, impulse_pos);
    auto impulse =
        Vec2(ox * kSideEnginePower * s_power, oy * kSideEnginePower * s_power);
    p->ApplyLinearImpulse(impulse, impulse_pos, true);
    lander_->ApplyLinearImpulse(-impulse, impulse_pos, true);
  }

  world_->Step(static_cast<float>(1.0 / kFPS), 6 * 30, 2 * 30);

  // state and reward
  auto pos = lander_->GetPosition();
  auto vel = lander_->GetLinearVelocity();
  double w = kViewportW / kScale;
  double h = kViewportH / kScale;
  obs_[0] = (pos.x - w / 2) / (w / 2);
  obs_[1] = (pos.y - h / 4 - kLegDown / kScale) / (h / 2);
  obs_[2] = vel.x * w / 2 / kFPS;
  obs_[3] = vel.y * h / 2 / kFPS;
  obs_[4] = lander_->GetAngle();
  obs_[5] = lander_->GetAngularVelocity() * 20 / kFPS;
  obs_[6] = ground_contact_[0];
  obs_[7] = ground_contact_[1];
  reward_ = 0;
  float shaping = -100 * (std::sqrt(obs_[0] * obs_[0] + obs_[1] * obs_[1]) +
                          std::sqrt(obs_[2] * obs_[2] + obs_[3] * obs_[3]) +
                          std::abs(obs_[4])) +
                  10 * (obs_[6] + obs_[7]);
  if (elapsed_step_ > 0) {
    reward_ = shaping - prev_shaping_;
  }
  prev_shaping_ = shaping;
  reward_ -= static_cast<float>(m_power * 0.3 + s_power * 0.03);
  if (done_ || std::abs(obs_[0]) >= 1) {
    done_ = true;
    reward_ = -100;
  }
  if (!lander_->IsAwake()) {
    done_ = true;
    reward_ = 100;
  }
  if (elapsed_step_ >= max_episode_steps_) {
    done_ = true;
  }
}

void LunarLanderBox2dEnv::LunarLanderReset(std::mt19937* gen) {
  elapsed_step_ = 0;
  done_ = false;
  ResetBox2d(gen);
  StepBox2d(gen, 0, 0, 0);
}

void LunarLanderBox2dEnv::LunarLanderStep(std::mt19937* gen, int action,
                                          float action0, float action1) {
  ++elapsed_step_;
  StepBox2d(gen, action, action0, action1);
}

}  // namespace box2d
