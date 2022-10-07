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

namespace box2d {

b2PolygonShape GeneratePolygon(const double* array, int size) {
  b2PolygonShape polygon;
  std::vector<b2Vec2> vertices;
  for (int i = 0; i < size; i += 2) {
    vertices.push_back(b2Vec2(array[i] * kSize, array[i + 1] * kSize));
  }
  polygon.Set(vertices.data(), size);
  return polygon;
}

Car::Car(std::shared_ptr<b2World>& world, double init_angle, double init_x, double init_y)
    : world_(world), hull_(nullptr){

  // Create hull
  b2BodyDef bd;
  bd.position.Set(init_x, init_y);
  bd.angle = init_angle;
  bd.type = b2_dynamicBody;
  hull_ = world_->CreateBody(&bd);

  b2PolygonShape polygon1 = GeneratePolygon(kHullPoly1, 8);
  hull_->CreateFixture(&polygon1, 1.f);

  b2PolygonShape polygon2 = GeneratePolygon(kHullPoly2, 8);
  hull_->CreateFixture(&polygon2, 1.f);

  b2PolygonShape polygon3 = GeneratePolygon(kHullPoly3, 16);
  hull_->CreateFixture(&polygon3, 1.f);

  b2PolygonShape polygon4 = GeneratePolygon(kHullPoly4, 8);
  hull_->CreateFixture(&polygon4, 1.f);

  for (int i = 0; i < 8; i += 2) {
    double wx = kWheelPos[i], wy = kWheelPos[i + 1];

    b2BodyDef bd;
    bd.position.Set(init_x + wx * kSize, init_y + wy * kSize);
    bd.angle = init_angle;
    bd.type = b2_dynamicBody;

    b2PolygonShape polygon = GeneratePolygon(wheelPoly, 8);
    b2FixtureDef fd;
    fd.shape = &polygon;
    fd.density = 0.1;
    fd.filter.categoryBits = 0x0020;
    fd.filter.maskBits = 0x001;
    fd.restitution = 0.0;

    Wheel w;
    w.body = world_->CreateBody(&bd);
    w.body->CreateFixture(&fd);
    w.wheel_rad = kWheelR * kSize;

    b2RevoluteJointDef rjd;
    rjd.bodyA = hull_;
    rjd.bodyB = w.body;
    rjd.localAnchorA.Set(wx * kSize, wy * kSize);
    rjd.localAnchorB.Set(0, 0);
    rjd.enableMotor = true;
    rjd.enableLimit = true;
    rjd.maxMotorTorque = 180 * 900 * kSize * kSize;
    rjd.motorSpeed = 0;
    rjd.referenceAngle = rjd.bodyB->GetAngle() - rjd.bodyA->GetAngle();
    rjd.lowerAngle = -0.4;
    rjd.upperAngle = +0.4;
    rjd.type = b2JointType::e_revoluteJoint;
    w.joint = static_cast<b2RevoluteJoint*>(world_->CreateJoint(&rjd));
    // w.body->SetUserData(&w);
    w.body->GetUserData().pointer = reinterpret_cast<uintptr_t>(&w);
    wheels_.push_back(&w);
  }
}
void Car::gas(double g) {
  if (g < 0) g = 0;
  if (g > 1) g = 1;
  for (int i = 2; i < 4; i++) {
    auto w = wheels_[i];
    auto diff = g - w->gas;
    if (diff > 0.1) diff = 0.1;
    w->gas += diff;
  }
}
void Car::brake(double b) {
  for (auto& w : wheels_) {
    w->brake = b;
  }
}
void Car::steer(double s) {
  wheels_[0]->steer = s;
  wheels_[1]->steer = s;
}
void Car::step(double dt) {
  for (auto w : wheels_) {
    // Steer each wheel
    double dir = (w->steer - w->joint->GetJointAngle() > 0) ? 1 : -1;
    double val = abs(w->steer - w->joint->GetJointAngle());
    w->joint->SetMotorSpeed(dir * std::min(50.0 * val, 3.0));
    // Position => friction_limit
    auto friction_limit = kFrictionLimit * 0.6;  // Grass friction if no tile
    for (auto t : w->tiles) {
      friction_limit =
          std::max(friction_limit,
                   static_cast<double>(kFrictionLimit * t->roadFriction));
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
                 (abs(w->omega) + 5.0));
    fuel_spent += dt * kEnginePower * w->gas;

    if (w->brake >= 0.9) {
      w->omega = 0;
    } else if (w->brake > 0) {
      auto brake_force = 15;          // radians per second
      dir = (w->omega > 0) ? -1 : 1;  // -np.sign(w.omega)
      val = brake_force * w->brake;
      if (abs(val) > abs(w->omega)) {
        val = abs(w->omega);  // low speed => same as = 0
      }
      w->omega += dir * val;
    }
    w->phase += w->omega * dt;

    auto vr = w->omega * w->wheel_rad;  // rotating wheel speed
    auto f_force = -vf + vr; // force direction is direction of speed difference
    auto p_force = -vs;

    // Physically correct is to always apply friction_limit until speed is
    // equal. But dt is finite, that will lead to oscillations if difference is
    // already near zero.

    // Random coefficient to cut oscillations in few steps (have no effect on
    // friction_limit)
    f_force *= 205000 * kSize * kSize;
    p_force *= 205000 * kSize * kSize;
    auto force = sqrt(pow(f_force, 2) + pow(p_force, 2));


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
void Car::destroy() {
  world_->DestroyBody(hull_);
  hull_ = nullptr;
  for (auto w : wheels_) {
    world_->DestroyBody(w->body);
  }
  wheels_.clear();
}

}  // namespace box2d