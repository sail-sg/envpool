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

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace toy_text {

class FrozenLakeEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("reward_threshold"_.Bind(0.7), "size"_.Bind(4),
                    "is_slippery"_.Bind(true));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    int size = conf["size"_];
    if (size != 4 && size != 8) {
      throw std::invalid_argument("FrozenLake size must be 4 or 8");
    }
    return MakeDict(TerminationSpec(conf),
                    "obs"_.Bind(Spec<int>({-1}, {0, size * size - 1})),
                    "info:prob"_.Bind(Spec<float>({-1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 3})));
  }
};

using FrozenLakeEnvSpec = EnvSpec<FrozenLakeEnvFns>;

class FrozenLakeEnv : public TimeLimitEnv<FrozenLakeEnvSpec> {
 protected:
  int x_, y_, size_;
  std::uniform_int_distribution<> dist_;
  std::vector<std::string> map_;

 public:
  FrozenLakeEnv(const Spec& spec, int env_id)
      : TimeLimitEnv<FrozenLakeEnvSpec>(spec, env_id),
        size_(spec.config["size"_]),
        dist_(-1, 1) {
    if (size_ != 8) {
      map_ = std::vector<std::string>({"SFFF", "FHFH", "FFFH", "HFFG"});
    } else {
      map_ = std::vector<std::string>({"SFFFFFFF", "FFFFFFFF", "FFFHFFFF",
                                       "FFFFFHFF", "FFFHFFFF", "FHHFFFHF",
                                       "FHFFHFHF", "FFFHFFFG"});
    }
  }

  void Reset() override {
    x_ = y_ = 0;
    WriteState(1.0f);
  }

  void Step(const Action& action) override {
    const bool slippery = spec_.config["is_slippery"_];
    const int slip = slippery ? dist_(gen_) : 0;
    const int act = (static_cast<int>(action["action"_]) + slip + 4) % 4;
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
    WriteState(slippery ? 1.0f / 3 : 1.0f);
  }

 private:
  void WriteState(float probability) {
    const char tile = map_[x_][y_];
    auto state =
        Transition(tile == 'G' ? 1.0f : 0.0f, tile == 'G' || tile == 'H');
    state["obs"_] = x_ * size_ + y_;
    state["info:prob"_] = probability;
  }
};

using FrozenLakeEnvPool = AsyncEnvPool<FrozenLakeEnv>;

}  // namespace toy_text

#endif  // ENVPOOL_TOY_TEXT_FROZEN_LAKE_H_
