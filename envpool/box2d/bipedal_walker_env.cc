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
      dist_uniform_(0, 1),
      world_(new b2World(b2Vec2(0.0, -10.0))),
      hull_(nullptr) {}

void BipedalWalkerBox2dEnv::ResetBox2d(std::mt19937* gen) {}

void BipedalWalkerBox2dEnv::StepBox2d(std::mt19937* gen, float action0,
                                      float action1, float action2,
                                      float action3) {
  if (elapsed_step_ >= max_episode_steps_) {
    done_ = true;
  }
}

void BipedalWalkerBox2dEnv::BipedalWalkerReset(std::mt19937* gen) {
  elapsed_step_ = -1;  // because of the step(0)
  done_ = false;
  ResetBox2d(gen);
  BipedalWalkerStep(gen, 0, 0, 0, 0);
}

void BipedalWalkerBox2dEnv::BipedalWalkerStep(std::mt19937* gen, float action0,
                                              float action1, float action2,
                                              float action3) {
  ++elapsed_step_;
  StepBox2d(gen, action0, action1, action2, action3);
}

}  // namespace box2d
