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

#ifndef ENVPOOL_BOX2D_CAR_RACING_H_
#define ENVPOOL_BOX2D_CAR_RACING_H_

#include "car_racing_env.h"
#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace box2d {

class CarRacingEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.Bind(1000),
                    "reward_threshold"_.Bind(900.0));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.Bind(Spec<uint8_t>({96, 96, 3}, {0, 255})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<float>({3}, {{-1, 0, 0}, {1, 1, 1}})));
  }
};

using CarRacingEnvSpec = EnvSpec<CarRacingEnvFns>;

class CarRacingEnv : public Env<CarRacingEnvSpec>,
                     public CarRacingBox2dEnv {
 public:
  CarRacingEnv(const Spec& spec, int env_id)
      : Env<CarRacingEnvSpec>(spec, env_id),
        CarRacingBox2dEnv(spec.config["max_episode_steps"_]){}

  bool IsDone() override { return done_; }

  void Reset() override {
    CarRacingReset(&gen_);
    WriteState();
  }

  void Step(const Action& action) override {
    CarRacingStep(&gen_, action["action"_][0], action["action"_][1],
                  action["action"_][2]);
    WriteState();
  }

 private:
  void WriteState() {
    State state = Allocate();
    state["reward"_] = reward_;
    // state["obs"_].Assign(obs_.begin(), obs_.size());
  }
};

using CarRacingEnvPool = AsyncEnvPool<CarRacingEnv>;

}  // namespace box2d

#endif  // ENVPOOL_BOX2D_CAR_RACING_H_