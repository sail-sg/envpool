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

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
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

TEST(MjcEnvPoolTest, FrameStack) {
  auto config = mujoco_gym::HalfCheetahEnvSpec::kDefaultConfig;
  constexpr int num_envs = 1;
  constexpr int frame_stack = 4;
  constexpr int obs_dim = 17;
  config["num_envs"_] = num_envs;
  config["batch_size"_] = num_envs;
  config["seed"_] = 0;
  config["frame_stack"_] = frame_stack;
  mujoco_gym::HalfCheetahEnvSpec spec(config);
  EXPECT_EQ(spec.state_spec["obs"_].shape,
            std::vector<int>({frame_stack, obs_dim}));

  mujoco_gym::HalfCheetahEnvPool envpool(spec);
  TArray<int> all_env_ids(Spec<int>({num_envs}));
  all_env_ids[0] = 0;
  envpool.Reset(all_env_ids);

  MjcState reset_state(envpool.Recv());
  EXPECT_EQ(reset_state["obs"_].Shape(),
            std::vector<std::size_t>({num_envs, frame_stack, obs_dim}));
  const auto reset_obs = TArray<mjtNum>(reset_state["obs"_][0]);
  const auto* reset_ptr = static_cast<const mjtNum*>(reset_obs.Data());
  for (int i = 1; i < frame_stack; ++i) {
    for (int j = 0; j < obs_dim; ++j) {
      EXPECT_EQ(reset_ptr[j], reset_ptr[i * obs_dim + j]);
    }
  }

  std::vector<Array> raw_action({Array(Spec<int>({num_envs})),
                                 Array(Spec<int>({num_envs})),
                                 Array(Spec<double>({num_envs, 6}))});
  MjcAction action(raw_action);
  action["env_id"_][0] = 0;
  action["players.env_id"_][0] = 0;
  for (int j = 0; j < 6; ++j) {
    action["action"_][0][j] = 0.0;
  }
  envpool.Send(action);

  MjcState step_state(envpool.Recv());
  EXPECT_EQ(step_state["obs"_].Shape(),
            std::vector<std::size_t>({num_envs, frame_stack, obs_dim}));
  const auto step_obs = TArray<mjtNum>(step_state["obs"_][0]);
  const auto* step_ptr = static_cast<const mjtNum*>(step_obs.Data());
  for (int i = 0; i < frame_stack - 1; ++i) {
    for (int j = 0; j < obs_dim; ++j) {
      EXPECT_EQ(step_ptr[i * obs_dim + j], reset_ptr[j]);
    }
  }
  bool changed = false;
  for (int j = 0; j < obs_dim; ++j) {
    changed =
        changed || (step_ptr[(frame_stack - 1) * obs_dim + j] != reset_ptr[j]);
  }
  EXPECT_TRUE(changed);
}
