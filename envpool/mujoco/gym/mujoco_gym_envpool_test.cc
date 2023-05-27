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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "envpool/mujoco/gym/half_cheetah.h"

using MjcAction = typename mujoco_gym::HalfCheetahEnv::Action;
using MjcState = typename mujoco_gym::HalfCheetahEnv::State;

TEST(MjcEnvPoolTest, CheckAction) {
  auto config = mujoco_gym::HalfCheetahEnvSpec::kDefaultConfig;
  int num_envs = 128;
  config["num_envs"_] = num_envs;
  mujoco_gym::HalfCheetahEnvSpec spec(config);
  mujoco_gym::HalfCheetahEnvPool envpool(spec);
  Array all_env_ids(Spec<int>({num_envs}));
  for (int i = 0; i < num_envs; ++i) {
    all_env_ids[i] = i;
  }
  envpool.Reset(all_env_ids);
  auto state_vec = envpool.Recv();
  // construct action
  std::vector<Array> raw_action({Array(Spec<int>({num_envs})),
                                 Array(Spec<int>({num_envs})),
                                 Array(Spec<double>({num_envs, 6}))});
  MjcAction action(raw_action);
  for (int i = 0; i < num_envs; ++i) {
    action["env_id"_][i] = i;
    action["players.env_id"_][i] = i;
    for (int j = 0; j < 6; ++j) {
      action["action"_][i][j] = (i + j + 1) / 100.0;
    }
  }
  // send
  envpool.Send(action);
  state_vec = envpool.Recv();
}
