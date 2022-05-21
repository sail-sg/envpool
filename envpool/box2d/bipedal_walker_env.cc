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
        //
      } else if (state == kPit && !oneshot) {
        y = original_y;
        if (counter > 1) {
          y -= 4 * kTerrainStep;
        }
      } else if (state == kStump && oneshot) {
        counter = RandInt(1, 3)(*gen);

      } else if (state == kStairs && oneshot) {
      } else if (state == kStairs && !oneshot) {
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
    }
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
