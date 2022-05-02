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

#ifndef ENVPOOL_CORE_ENV_SPEC_H_
#define ENVPOOL_CORE_ENV_SPEC_H_

#include <string>

#include "envpool/core/array.h"
#include "envpool/core/dict.h"

auto common_config =
    MakeDict("num_envs"_.Bind(1), "batch_size"_.Bind(0), "num_threads"_.Bind(0),
             "max_num_players"_.Bind(1), "thread_affinity_offset"_.Bind(-1),
             "base_path"_.Bind(std::string("envpool")), "seed"_.Bind(42),
             "gym_reset_return_info"_.Bind(false));
// Note: this action order is hardcoded in async_envpool Send function
// and env ParseAction function for performance
auto common_action_spec = MakeDict("env_id"_.Bind(Spec<int>({})),
                                   "players.env_id"_.Bind(Spec<int>({-1})));
// Note: this state order is hardcoded in async_envpool Recv function
auto common_state_spec =
    MakeDict("info:env_id"_.Bind(Spec<int>({})),
             "info:players.env_id"_.Bind(Spec<int>({-1})),
             "elapsed_step"_.Bind(Spec<int>({})), "done"_.Bind(Spec<bool>({})),
             "reward"_.Bind(Spec<float>({-1})));

/**
 * EnvSpec funciton, it constructs the env spec when a Config is passed.
 */
template <typename EnvFns>
class EnvSpec {
 public:
  using Config = decltype(ConcatDict(common_config, EnvFns::DefaultConfig()));
  using ConfigKeys = typename Config::Keys;
  using ConfigValues = typename Config::Values;
  using StateSpec = decltype(
      ConcatDict(common_state_spec, EnvFns::StateSpec(std::declval<Config>())));
  using ActionSpec = decltype(ConcatDict(
      common_action_spec, EnvFns::ActionSpec(std::declval<Config>())));
  using StateKeys = typename StateSpec::Keys;
  using ActionKeys = typename ActionSpec::Keys;

  // For C++
  Config config_;
  StateSpec state_spec_;
  ActionSpec action_spec_;
  static inline const Config DEFAULT_CONFIG =
      ConcatDict(common_config, EnvFns::DefaultConfig());

  EnvSpec() : EnvSpec(DEFAULT_CONFIG) {}
  explicit EnvSpec(const ConfigValues& conf)
      : config_(conf),
        state_spec_(ConcatDict(common_state_spec, EnvFns::StateSpec(config_))),
        action_spec_(
            ConcatDict(common_action_spec, EnvFns::ActionSpec(config_))) {
    if (config_["batch_size"_] > config_["num_envs"_]) {
      throw std::invalid_argument(
          "It is required that batch_size <= num_envs, got num_envs = " +
          std::to_string(config_["num_envs"_]) +
          ", batch_size = " + std::to_string(config_["batch_size"_]));
    }
  }
};

#endif  // ENVPOOL_CORE_ENV_SPEC_H_
