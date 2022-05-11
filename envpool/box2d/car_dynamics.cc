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

b2PolygonShape GeneratePolygon(float* array, int size) {
  b2Vec2* vertices = (b2Vec2*)malloc(size * sizeof(b2Vec2));
  for (int i = 0; i < 2 * size; i += 2) {
    vertices[i].Set(array[i], array[i + 1]);
  }
  b2PolygonShape polygon;
  polygon.Set(vertices, size);
  free(vertices);
  return polygon;
}

Car::Car(b2World* _world, float init_angle, float init_x, float init_y)
    : world(_world), fuel_spent(0.0) {
  b2BodyDef bd;
  bd.position.Set(init_x, init_y);
  bd.angle = init_angle;
  bd.type = b2_dynamicBody;
  hull = world->CreateBody(&bd);

  b2PolygonShape polygon1 = GeneratePolygon(kHullPoly1, 4);
  hullFixtures.emplace_back(hull->CreateFixture(&polygon1, 1.f));

  b2PolygonShape polygon2 = GeneratePolygon(kHullPoly2, 4);
  hullFixtures.emplace_back(hull->CreateFixture(&polygon2, 1.f));

  b2PolygonShape polygon3 = GeneratePolygon(kHullPoly3, 8);
  hullFixtures.emplace_back(hull->CreateFixture(&polygon3, 1.f));

  b2PolygonShape polygon4 = GeneratePolygon(kHullPoly4, 4);
  hullFixtures.emplace_back(hull->CreateFixture(&polygon4, 1.f));

  for (int i = 0; i < 8; i += 2) {
    float wx = kWheelPos[i], wy = kWheelPos[i + 1];
    float front_k = 1.0;

    b2BodyDef bd;
    bd.position.Set(init_x + wx * kSize, init_y + wy * kSize);
    bd.angle = init_angle;
    bd.type = b2_dynamicBody;

    float vertices[8];
    for (int j = 0; j < 8; j += 2) {
      vertices[j] = wheelPoly[j] * front_k * kSize;
      vertices[j + 1] = wheelPoly[j + 1] * front_k * kSize;
    }

    b2PolygonShape polygon = GeneratePolygon(vertices, 4);
    b2FixtureDef* def = new b2FixtureDef();

    def->shape = &polygon;
    def->density = 0.1;
    def->filter.categoryBits = 0x0020;
    def->filter.maskBits = 0x001;
    def->restitution = 0.0;

    WheelData* wheelData = new WheelData;
    wheelData->wheel = world->CreateBody(&bd);
    wheelData->wheel->CreateFixture(def);

    wheelData->wheel_rad = front_k * kWheelR * kSize;
    memcpy(wheelData->color, kWheelColor, 3 * sizeof(float));
    wheelData->gas = 0;
    wheelData->brake = 0;
    wheelData->steer = 0;
    wheelData->phase = 0.0;
    wheelData->omega = 0.0;
    wheelData->skid_start = NULL;
    wheelData->skid_particle = NULL;

    b2RevoluteJointDef rjd;
    rjd.bodyA = hull;
    rjd.bodyB = wheelData->wheel;
    rjd.localAnchorA.Set(wx * kSize, wy * kSize);
    rjd.localAnchorB.Set(0, 0);
    rjd.enableMotor = true;
    rjd.enableLimit = true;
    rjd.maxMotorTorque = 180 * 900 * kSize * kSize;
    rjd.motorSpeed = 0;
    rjd.lowerAngle = -0.4;
    rjd.upperAngle = +0.4;
    b2Joint* joint = world->CreateJoint(&rjd);
    wheelData->joint = dynamic_cast<b2RevoluteJoint*>(joint);
    // world->CreateJoint(&rjd);
    wheelData->wheel->SetUserData(wheelData);
    wheels.emplace_back(wheelData);
  }
}
void Car::gas(float g) {
  if (g < 0) g = 0;
  if (g > 1) g = 1;
  for (int i = 2; i < 4; i++) {
    auto w = wheels[i];
    auto diff = g - w->gas;
    if (diff > 0.1) diff = 0.1;
    w->gas += diff;
  }
}
void Car::brake(float b) {
  for (auto& w : wheels) {
    w->brake = b;
  }
}
void Car::steer(float s) {
  wheels[0]->steer = s;
  wheels[1]->steer = s;
}
void Car::step(float dt) {
  for (auto w : wheels) {
    // Steer each wheel
    float dir = (w->steer - w->joint->GetJointAngle() > 0) ? 1 : -1;
    float val = abs(w->steer - w->joint->GetJointAngle() > 0);
    w->joint->SetMotorSpeed(dir * std::min(50.0 * val, 3.0));
    // Position => friction_limit
    auto grass = true;
    auto friction_limit = kFrictionLimit * 0.6;  // Grass friction if no tile
    for (auto t : w->tiles) {
      friction_limit =
          std::max(friction_limit,
                   static_cast<double>(kFrictionLimit * t->roadFriction));
      grass = false;
    }
    // Force
    auto forw = w->wheel->GetWorldVector({0, 1});
    auto side = w->wheel->GetWorldVector({1, 0});
    auto v = w->wheel->GetLinearVelocity();
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
    auto f_force =
        -vf + vr;  // force direction is direction of speed difference
    auto p_force = -vs;

    // Physically correct is to always apply friction_limit until speed is
    // equal. But dt is finite, that will lead to oscillations if difference is
    // already near zero.

    // Random coefficient to cut oscillations in few steps (have no effect on
    // friction_limit)
    f_force *= 205000 * kSize * kSize;
    p_force *= 205000 * kSize * kSize;
    auto force = sqrt(pow(f_force, 2) + pow(p_force, 2));

    // Skid trace
    if (abs(force) > 2.0 * friction_limit) {
      if (w->skid_particle != NULL && w->skid_particle->isGrass == grass &&
          w->skid_particle->poly.size() < 30) {
        w->skid_particle->poly.push_back(
            {w->wheel->GetPosition().x, w->wheel->GetPosition().y});
      } else if (w->skid_start == NULL) {
        w->skid_start = new b2Vec2(w->wheel->GetPosition());
      } else {
        w->skid_particle =
            createParticle(*(w->skid_start), w->wheel->GetPosition(), grass);
        w->skid_start = NULL;
      }
    } else {
      w->skid_start = NULL;
      w->skid_particle = NULL;
    }

    if (abs(force) > friction_limit) {
      f_force /= force;
      p_force /= force;
      force = friction_limit;  // Correct physics here
      f_force *= force;
      p_force *= force;
    }

    w->omega -= dt * f_force * w->wheel_rad / kWheelMomentOfInertia;

    w->wheel->ApplyForceToCenter(
        {
            p_force * side.x + f_force * forw.x,
            p_force * side.y + f_force * forw.y,
        },
        true);
  }
}
void Car::destroy() {
  world->DestroyBody(hull);
  hull = NULL;
  for (auto w : wheels) {
    world->DestroyBody(w->wheel);
  }
  wheels.clear();
}

}  // namespace box2d
