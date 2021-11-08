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

#include "envpool/core/array.h"
#include "envpool/core/dict.h"

auto common_config =
    MakeDict("num_envs"_.bind(1), "batch"_.bind(0), "num_threads"_.bind(0),
             "max_num_players"_.bind(1), "thread_affinity_offset"_.bind(-1),
             "seed"_.bind(42));
// Note: this action order is hardcoded in async_envpool Send function
// and env ParseAction function for performance
auto common_action_spec = MakeDict("env_id"_.bind(Spec<int>({})),
                                   "players.env_id"_.bind(Spec<int>({-1})));
// Note: this state order is hardcoded in async_envpool Recv function
auto common_state_spec =
    MakeDict("info:env_id"_.bind(Spec<int>({})),
             "info:players.env_id"_.bind(Spec<int>({-1})),
             "elapsed_step"_.bind(Spec<int>({})), "done"_.bind(Spec<bool>({})));

/**
 * EnvSpec funciton, it constructs the env spec when a Config is passed.
 */
template <typename EnvFns>
class EnvSpec {
 public:
  typedef decltype(ConcatDict(common_config, EnvFns::DefaultConfig())) Config;
  typedef typename Config::Keys ConfigKeys;
  typedef typename Config::Values ConfigValues;
  typedef decltype(ConcatDict(
      common_state_spec, EnvFns::StateSpec(std::declval<Config>()))) StateSpec;
  typedef decltype(
      ConcatDict(common_action_spec,
                 EnvFns::ActionSpec(std::declval<Config>()))) ActionSpec;
  typedef typename StateSpec::Keys StateKeys;
  typedef typename ActionSpec::Keys ActionKeys;

 public:
  // For C++
  Config config;
  StateSpec state_spec;
  ActionSpec action_spec;
  static inline const Config default_config =
      ConcatDict(common_config, EnvFns::DefaultConfig());

 public:
  EnvSpec()
      : config(default_config),
        state_spec(ConcatDict(common_state_spec, EnvFns::StateSpec(config))),
        action_spec(
            ConcatDict(common_action_spec, EnvFns::ActionSpec(config))) {}
  explicit EnvSpec(const ConfigValues& conf)
      : config(conf),
        state_spec(ConcatDict(common_state_spec, EnvFns::StateSpec(config))),
        action_spec(
            ConcatDict(common_action_spec, EnvFns::ActionSpec(config))) {}
};

#endif  // ENVPOOL_CORE_ENV_SPEC_H_
