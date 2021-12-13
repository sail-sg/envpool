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
// https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py

#ifndef ENVPOOL_TOY_TEXT_FROZEN_LAKE_H_
#define ENVPOOL_TOY_TEXT_FROZEN_LAKE_H_

#include <cmath>
#include <random>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace toy_text {

class FrozenLakeEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.bind(100),
                    "reward_threshold"_.bind(0.7f), "size"_.bind(4));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    int size = conf["size"_];
    return MakeDict("obs"_.bind(Spec<int>({-1}, {0, size * size - 1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.bind(Spec<int>({-1}, {0, 3})));
  }
};

typedef class EnvSpec<FrozenLakeEnvFns> FrozenLakeEnvSpec;

class FrozenLakeEnv : public Env<FrozenLakeEnvSpec> {
 protected:
  int x_, y_, size_, max_episode_steps_, elapsed_step_;
  std::uniform_int_distribution<> dist_;
  bool done_;
  std::vector<std::string> map_;

 public:
  FrozenLakeEnv(const Spec& spec, int env_id)
      : Env<FrozenLakeEnvSpec>(spec, env_id),
        size_(spec.config["size"_]),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        dist_(-1, 1),
        done_(true) {
    if (size_ != 8) {
      map_ = std::vector<std::string>({"SFFF", "FHFH", "FFFH", "HFFG"});
    } else {
      map_ = std::vector<std::string>({"SFFFFFFF", "FFFFFFFF", "FFFHFFFF",
                                       "FFFFFHFF", "FFFHFFFF", "FHHFFFHF",
                                       "FHFFHFHF", "FFFHFFFG"});
    }
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    x_ = y_ = 0;
    State state = Allocate();
    WriteObs(state, 0.0f);
  }

  void Step(const Action& action) override {
    done_ = (++elapsed_step_ >= max_episode_steps_);
    int act = action["action"_];
    act = (act + dist_(gen_) + 4) % 4;
    if (act == 0) {
      --y_;
    } else if (act == 1) {
      ++x_;
    } else if (act == 2) {
      ++y_;
    } else {
      --x_;
    }
    x_ = std::min(std::max(x_, 0), size_ - 1);
    y_ = std::min(std::max(y_, 0), size_ - 1);
    float reward = 0.0f;
    if (map_[x_][y_] == 'H' || map_[x_][y_] == 'G') {
      done_ = true;
      reward = map_[x_][y_] == 'G' ? 1.0f : 0.0f;
    }
    State state = Allocate();
    WriteObs(state, reward);
  }

 private:
  void WriteObs(State& state, float reward) {  // NOLINT
    state["obs"_] = x_ * size_ + y_;
    state["reward"_] = reward;
  }
};

typedef AsyncEnvPool<FrozenLakeEnv> FrozenLakeEnvPool;

}  // namespace toy_text

#endif  // ENVPOOL_TOY_TEXT_FROZEN_LAKE_H_
