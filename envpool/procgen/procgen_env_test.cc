// Copyright 2023 Garena Online Private Limited
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

#include "envpool/procgen/procgen_env.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <random>

using ProcgenState = procgen::ProcgenEnv::State;
using ProcgenAction = procgen::ProcgenEnv::Action;

TEST(PRocgenEnvTest, BasicStep) {
  std::srand(std::time(nullptr));
  auto config = procgen::ProcgenEnvSpec::kDefaultConfig;
  std::size_t batch = 4;
  config["num_envs"_] = batch;
  config["batch_size"_] = batch;
  config["seed"_] = 0;
  int total_iter = 10000;
  procgen::ProcgenEnvSpec spec(config);
  procgen::ProcgenEnvPool envpool(spec);
  Array all_env_ids(Spec<int>({static_cast<int>(batch)}));
  for (std::size_t i = 0; i < batch; ++i) {
    all_env_ids[i] = i;
  }
  LOG(INFO) << "init done";
  envpool.Reset(all_env_ids);
  LOG(INFO) << "init done";
  std::vector<Array> raw_action(3);
  ProcgenAction action(&raw_action);
  for (int i = 0; i < total_iter; ++i) {
    auto state_vec = envpool.Recv();
    ProcgenState state(&state_vec);
    EXPECT_EQ(state["obs"_].Shape(),
              std::vector<std::size_t>({batch, 64, 64, 3}));
    uint8_t* data = static_cast<uint8_t*>(state["obs"_].Data());
    int index = 0;
    for (std::size_t j = 0; j < batch; ++j) {
      // ensure there's no black screen in each frame
      int sum = 0;
      for (int k = 0; k < 64 * 64 * 3; ++k) {
        sum += data[index++];
      }
      EXPECT_NE(sum, 0) << i << " " << j;
    }
    action["env_id"_] = state["info:env_id"_];
    action["players.env_id"_] = state["info:env_id"_];
    action["action"_] = Array(Spec<int>({static_cast<int>(batch)}));
    for (std::size_t j = 0; j < batch; ++j) {
      action["action"_][j] = std::rand() % 15;
    }
    envpool.Send(action);
  }
}
