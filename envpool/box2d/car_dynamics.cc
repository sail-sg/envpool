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

#include "car_dynamics.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace box2d {

b2PolygonShape GeneratePolygon(const float (*poly)[2], int size) {  // NOLINT
  std::vector<b2Vec2> vec_list;
  vec_list.resize(size);
  for (int i = 0; i < size; ++i) {
    vec_list[i] = b2Vec2(poly[i][0] * kSize, poly[i][1] * kSize);
  }
  b2PolygonShape polygon;
  polygon.Set(vec_list.data(), vec_list.size());
  return polygon;
}

Car::Car(std::shared_ptr<b2World> world, float init_angle, float init_x,
         float init_y)
    : world_(std::move(world)) {
  // Create hull
  b2BodyDef bd;
  bd.position.Set(init_x, init_y);
  bd.angle = init_angle;
  bd.type = b2_dynamicBody;

  hull_ = world_->CreateBody(&bd);
  drawlist_.push_back(hull_);

  b2PolygonShape polygon1 = GeneratePolygon(kHullPoly1, 4);
  hull_->CreateFixture(&polygon1, 1.f);

  b2PolygonShape polygon2 = GeneratePolygon(kHullPoly2, 4);
  hull_->CreateFixture(&polygon2, 1.f);

  b2PolygonShape polygon3 = GeneratePolygon(kHullPoly3, 8);
  hull_->CreateFixture(&polygon3, 1.f);

  b2PolygonShape polygon4 = GeneratePolygon(kHullPoly4, 4);
  hull_->CreateFixture(&polygon4, 1.f);

  for (const auto* p : kWheelPos) {
    float wx = p[0];
    float wy = p[1];

    b2BodyDef bd;
    bd.position.Set(init_x + wx * kSize, init_y + wy * kSize);
    bd.angle = init_angle;
    bd.type = b2_dynamicBody;

    b2PolygonShape polygon = GeneratePolygon(kWheelPoly, 4);
    b2FixtureDef fd;
    fd.shape = &polygon;
    fd.density = 0.1;
    fd.filter.categoryBits = 0x0020;
    fd.filter.maskBits = 0x001;
    fd.restitution = 0.0;

    auto* w = new Wheel();
    w->type = WHEEL_TYPE;
    w->body = world_->CreateBody(&bd);

    drawlist_.push_back(w->body);

    w->body->CreateFixture(&fd);
    w->wheel_rad = kWheelR * kSize;

    b2RevoluteJointDef rjd;
    rjd.bodyA = hull_;
    rjd.bodyB = w->body;
    rjd.localAnchorA.Set(wx * kSize, wy * kSize);
    rjd.localAnchorB.Set(0, 0);
    rjd.referenceAngle = rjd.bodyB->GetAngle() - rjd.bodyA->GetAngle();
    rjd.enableMotor = true;
    rjd.enableLimit = true;
    rjd.maxMotorTorque = 180 * 900 * kSize * kSize;
    rjd.motorSpeed = 0;
    rjd.referenceAngle = rjd.bodyB->GetAngle() - rjd.bodyA->GetAngle();
    rjd.lowerAngle = -0.4;
    rjd.upperAngle = +0.4;
    rjd.type = b2JointType::e_revoluteJoint;
    w->joint = static_cast<b2RevoluteJoint*>(world_->CreateJoint(&rjd));
    // w->body->SetUserData(&w);
    w->body->GetUserData().pointer = reinterpret_cast<uintptr_t>(w);
    wheels_.push_back(w);
  }
}
void Car::Gas(float g) {
  g = std::min(std::max(g, 0.0f), 1.0f);
  for (int i = 2; i < 4; i++) {
    auto* w = wheels_[i];
    w->gas += std::min(g - w->gas, 0.1f);
  }
}

void Car::Brake(float b) {
  for (auto& w : wheels_) {
    w->brake = b;
  }
}
void Car::Steer(float s) {
  wheels_[0]->steer = s;
  wheels_[1]->steer = s;
}
void Car::Step(float dt) {
  for (auto* w : wheels_) {
    // Steer each wheel
    float dir = Sign(w->steer - w->joint->GetJointAngle());
    float val = abs(w->steer - w->joint->GetJointAngle());
    w->joint->SetMotorSpeed(dir * std::min(50.0f * val, 3.0f));

    // Position => friction_limit
    bool grass = true;
    float friction_limit = kFrictionLimit * 0.6f;  // Grass friction if no tile
    for (auto* t : w->tiles) {
      friction_limit =
          std::max(friction_limit, kFrictionLimit * t->road_friction);
      grass = false;
    }
    // Force
    auto forw = w->body->GetWorldVector({0, 1});
    auto side = w->body->GetWorldVector({1, 0});
    auto v = w->body->GetLinearVelocity();
    auto vf = forw.x * v.x + forw.y * v.y;  // forward speed
    auto vs = side.x * v.x + side.y * v.y;  // side speed
    // WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
    // WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W -- power
    // domega = dt*W/WHEEL_MOMENT_OF_INERTIA/w.omega

    // add small coef not to divide by zero
    w->omega += (dt * kEnginePower * w->gas / kWheelMomentOfInertia /
                 (abs(w->omega) + 5.0f));
    fuel_spent_ += dt * kEnginePower * w->gas;
    if (w->brake >= 0.9) {
      w->omega = 0;
    } else if (w->brake > 0) {
      dir = -Sign(w->omega);
      val = kBrakeForce * w->brake;
      if (abs(val) > abs(w->omega)) {
        val = abs(w->omega);  // low speed => same as = 0
      }
      w->omega += dir * val;
    }
    w->phase += w->omega * dt;

    auto vr = w->omega * w->wheel_rad;  // rotating wheel speed
    // force direction is direction of speed difference
    auto f_force = -vf + vr;
    auto p_force = -vs;

    // Physically correct is to always apply friction_limit until speed is
    // equal. But dt is finite, that will lead to oscillations if difference is
    // already near zero.

    // Random coefficient to cut oscillations in few steps (have no effect on
    // friction_limit)
    f_force *= 205000 * kSize * kSize;
    p_force *= 205000 * kSize * kSize;
    auto force = sqrt(f_force * f_force + p_force * p_force);

    // Skid trace
    if (abs(force) > 2.0 * friction_limit) {
      if (w->skid_particle && w->skid_particle->grass == grass &&
          w->skid_particle->poly.size() < 30) {
        w->skid_particle->poly.emplace_back(
            Vec2(w->body->GetPosition().x, w->body->GetPosition().y));
      } else if (w->skid_start == nullptr) {
        w->skid_start = std::make_unique<b2Vec2>(w->body->GetPosition());
      } else {
        w->skid_particle = CreateParticle(*(w->skid_start.get()),  // NOLINT
                                          w->body->GetPosition(), grass);
        w->skid_start = nullptr;
      }

    } else {
      w->skid_start = nullptr;
      w->skid_particle = nullptr;
    }

    if (abs(force) > friction_limit) {
      f_force /= force;
      p_force /= force;
      force = friction_limit;  // Correct physics here
      f_force *= force;
      p_force *= force;
    }

    w->omega -= dt * f_force * w->wheel_rad / kWheelMomentOfInertia;

    w->body->ApplyForceToCenter(
        {
            static_cast<float>(p_force * side.x + f_force * forw.x),
            static_cast<float>(p_force * side.y + f_force * forw.y),
        },
        true);
  }
}

std::shared_ptr<Particle> Car::CreateParticle(b2Vec2 point1, b2Vec2 point2,
                                              bool grass) {
  cv::Scalar color = grass ? kMudColor : kWheelColor;
  auto p = std::make_shared<Particle>(point1, point2, grass, color);
  particles_.emplace_back(p);
  while (particles_.size() > 30) {
    particles_.pop_front();
  }
  return p;
}

void Car::Draw(const cv::Mat& surf, float zoom,
               const std::array<float, 2>& translation, float angle,
               bool draw_particles) {
  if (draw_particles) {
    std::vector<cv::Point> poly;
    for (const auto& p : particles_) {
      poly.clear();
      for (const auto& vec_tmp : p->poly) {
        auto v = RotateRad(vec_tmp, angle);
        poly.emplace_back(cv::Point(v.x * zoom + translation[0],
                                    v.y * zoom + translation[1]));
      }
      cv::polylines(surf, poly, false, p->color, 2);
    }
  }
  for (size_t i = 0; i < drawlist_.size(); i++) {
    auto* body = drawlist_[i];
    auto trans = body->GetTransform();
    cv::Scalar color;
    if (i == 0) {
      color = cv::Scalar(0, 0, 204);  // hull.color = (0.8, 0.0, 0.0) * 255
    } else {
      color = cv::Scalar(0, 0, 0);  // wheel.color = (0, 0, 0)
    }
    std::vector<cv::Point> poly;
    for (b2Fixture* f = body->GetFixtureList(); f != nullptr;
         f = f->GetNext()) {
      auto* shape = static_cast<b2PolygonShape*>(f->GetShape());
      poly.clear();
      for (int j = 0; j < shape->m_count; j++) {
        auto vec_tmp = Multiply(trans, shape->m_vertices[j]);
        auto v = RotateRad(vec_tmp, angle);
        poly.emplace_back(cv::Point(v.x * zoom + translation[0],
                                    v.y * zoom + translation[1]));
      }
      cv::fillPoly(surf, poly, color);

      auto* user_data =
          reinterpret_cast<UserData*>(body->GetUserData().pointer);  // NOLINT
      if (user_data == nullptr || user_data->type != WHEEL_TYPE) {
        continue;
      }
      auto* obj = reinterpret_cast<Wheel*>(user_data);

      auto a1 = obj->phase;
      auto a2 = obj->phase + 1.2f;  // radians
      auto s1 = std::sin(a1);
      auto s2 = std::sin(a2);
      auto c1 = std::cos(a1);
      auto c2 = std::cos(a2);
      if (s1 > 0 && s2 > 0) {
        continue;
      }
      if (s1 > 0) {
        c1 = Sign(c1);
      }
      if (s2 > 0) {
        c2 = Sign(c2);
      }

      poly.clear();
      std::vector<b2Vec2> white_poly = {
          Vec2(-kWheelW * kSize, +kWheelR * c1 * kSize),
          Vec2(+kWheelW * kSize, +kWheelR * c1 * kSize),
          Vec2(+kWheelW * kSize, +kWheelR * c2 * kSize),
          Vec2(-kWheelW * kSize, +kWheelR * c2 * kSize),
      };
      for (const auto& vec : white_poly) {
        auto vec_tmp = Multiply(trans, vec);
        auto v = RotateRad(vec_tmp, angle);
        poly.emplace_back(cv::Point(v.x * zoom + translation[0],
                                    v.y * zoom + translation[1]));
      }
      cv::fillPoly(surf, poly, kWheelWhite);
    }
  }
}

void Car::Destroy() {
  world_->DestroyBody(hull_);
  hull_ = nullptr;
  for (auto* w : wheels_) {
    world_->DestroyBody(w->body);
    delete w;
    w = nullptr;
  }
  wheels_.clear();
}

float Car::GetFuelSpent() const { return fuel_spent_; }

std::vector<float> Car::GetGas() { return {wheels_[2]->gas, wheels_[3]->gas}; }

std::vector<float> Car::GetSteer() {
  return {wheels_[0]->steer, wheels_[1]->steer};
}

std::vector<float> Car::GetBrake() {
  return {wheels_[0]->brake, wheels_[1]->brake, wheels_[2]->brake,
          wheels_[3]->brake};
}

}  // namespace box2d
