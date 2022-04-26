// Copyright 2021 Garena Online Private Limited
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

#include "envpool/dummy/dummy_envpool.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <random>
#include <vector>

using DummyAction = typename dummy::DummyEnv::Action;
using DummyState = typename dummy::DummyEnv::State;

TEST(DummyEnvPoolTest, SplitZeroAction) {
  auto config = dummy::DummyEnvSpec::default_config;
  int num_envs = 4;
  config["num_envs"_] = num_envs;
  config["batch_size"_] = 4;
  config["num_threads"_] = 1;
  config["seed"_] = 42;
  config["max_num_players"_] = 4;
  dummy::DummyEnvSpec spec(config);
  dummy::DummyEnvPool envpool(spec);
  Array all_env_ids(Spec<int>({num_envs}));
  for (int i = 0; i < num_envs; ++i) {
    all_env_ids[i] = i;
  }
  envpool.Reset(all_env_ids);
  auto state_vec = envpool.Recv();
  // construct action
  std::vector<Array> raw_action({Array(Spec<int>({4})), Array(Spec<int>({8})),
                                 Array(Spec<int>({8})), Array(Spec<int>({8}))});
  DummyAction action(&raw_action);
  for (int i = 0; i < 4; ++i) {
    action["env_id"_][i] = i;
  }
  std::vector<int> player_env_id({1, 2, 0, 2, 0, 1, 1, 2});
  for (int i = 0; i < 8; ++i) {
    action["players.env_id"_][i] = player_env_id[i];
  }
  // send
  envpool.Send(action);
  state_vec = envpool.Recv();
  DummyState state(&state_vec);
  EXPECT_EQ(action["env_id"_].Shape(0), state["info:env_id"_].Shape(0));
  EXPECT_EQ(state["info:players.env_id"_].Shape(0), 8);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(static_cast<int>(state["info:env_id"_][i]), i);
  }
  auto obs = state["obs"_];
  auto peid = state["info:players.env_id"_];
  std::vector<int> counter({2, 3, 3, 0});
  for (int i = 0; i < 8; ++i) {
    int p = peid[i];
    EXPECT_EQ(static_cast<int>(obs(i, 1)), counter[p]);
  }
  // construct continuous action
  envpool.Reset(action["env_id"_]);
  envpool.Recv();
  player_env_id = std::vector<int>({0, 0, 0, 2, 2, 3, 3, 3});
  for (int i = 0; i < 8; ++i) {
    action["players.env_id"_][i] = player_env_id[i];
  }
  // send
  envpool.Send(action);
  state_vec = envpool.Recv();
  state = DummyState(&state_vec);
  EXPECT_EQ(action["env_id"_].Shape(0), state["info:env_id"_].Shape(0));
  EXPECT_EQ(state["info:players.env_id"_].Shape(0), 8);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(static_cast<int>(state["info:env_id"_][i]), i);
  }
  obs = state["obs"_];
  peid = state["info:players.env_id"_];
  counter = std::vector<int>({3, 0, 2, 3});
  for (int i = 0; i < 8; ++i) {
    int p = peid[i];
    EXPECT_EQ(static_cast<int>(obs(i, 1)), counter[p]);
  }
}

void Runner(int num_envs, int batch, int seed, int total_iter, int num_threads,
            int max_num_players) {
  LOG(INFO) << num_envs << " " << batch << " " << seed << " " << total_iter
            << " " << num_threads << " " << max_num_players;
  bool is_sync = num_envs == batch && max_num_players == 1;
  auto config = dummy::DummyEnvSpec::default_config;
  config["num_envs"_] = num_envs;
  config["batch_size"_] = batch;
  config["num_threads"_] = num_threads;
  config["seed"_] = seed;
  config["max_num_players"_] = max_num_players;
  std::vector<int> length;
  std::vector<int> counter;
  for (int i = 0; i < num_envs; ++i) {
    length.push_back(seed + i);
    counter.push_back(-1);
  }
  dummy::DummyEnvSpec spec(config);
  dummy::DummyEnvPool envpool(spec);
  Array all_env_ids(Spec<int>({num_envs}));
  for (int i = 0; i < num_envs; ++i) {
    all_env_ids[i] = i;
  }
  envpool.Reset(all_env_ids);
  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < total_iter; ++i) {
    // recv
    auto state_vec = envpool.Recv();
    DummyState state(&state_vec);
    // check state
    auto env_id = state["info:env_id"_];
    auto player_env_id = state["info:players.env_id"_];
    auto player_id = state["info:players.id"_];
    auto obs = state["obs"_];
    auto reward = state["reward"_];
    auto done = state["done"_];
    auto player_done = state["info:players.done"_];
    EXPECT_EQ(env_id.Shape(0), batch);
    int total_num_players = 0;
    for (int i = 0; i < batch; ++i) {
      int eid = env_id[i];
      if (is_sync) {
        EXPECT_EQ(eid, i);
      }
      ++counter[eid];
      int num_players =
          max_num_players <= 1 ? 1 : counter[eid] % (max_num_players - 1) + 1;
      total_num_players += num_players;
    }
    EXPECT_EQ(player_env_id.Shape(0), total_num_players);
    if (is_sync) {
      EXPECT_EQ(total_num_players, batch);
    }
    for (int i = 0; i < total_num_players; ++i) {
      int eid = player_env_id[i];
      int num_players =
          max_num_players <= 1 ? 1 : counter[eid] % (max_num_players - 1) + 1;
      int id = player_id[i];
      int state0 = obs(i, 0);
      int state1 = obs(i, 1);
      // float rew = reward[i];
      bool done_flag = player_done[i];
      EXPECT_LT(id, num_players);
      // EXPECT_LT(-rew, num_players);  <-- error caused by float(arr) := int
      EXPECT_EQ(done_flag, counter[eid] >= length[eid]);
      ASSERT_EQ(state0, counter[eid]) << eid;
      EXPECT_LE(state1, max_num_players);
      if (is_sync) {
        EXPECT_EQ(eid, i);
      }
    }

    for (int i = 0; i < batch; ++i) {
      int eid = env_id[i];
      if (counter[eid] >= length[eid]) {
        EXPECT_TRUE(static_cast<bool>(done[i]));
        counter[eid] = -1;
      } else {
        EXPECT_FALSE(static_cast<bool>(done[i]));
      }
    }
    // construct action
    std::vector<Array> raw_action(4);
    DummyAction action(&raw_action);
    action["env_id"_] = env_id;
    action["players.env_id"_] = player_env_id;
    action["players.action"_] = player_id;
    action["players.id"_] = player_id;
    // send
    envpool.Send(action);
  }
  std::chrono::duration<double> dur = std::chrono::system_clock::now() - start;
  double t = dur.count();
  double fps = (total_iter * batch) / t;
  LOG(INFO) << "time(s): " << t << ", FPS: " << fps;
}

TEST(DummyEnvPoolTest, SinglePlayer) {
  Runner(1, 1, 20, 100000, 1, 1);
  Runner(3, 1, 20, 100000, 1, 1);
  Runner(3, 1, 20, 100000, 3, 1);
  Runner(9, 4, 20, 100000, 1, 1);
  Runner(9, 4, 30, 100000, 4, 1);
  Runner(9, 4, 30, 100000, 9, 1);
  Runner(10, 10, 25, 100000, 0, 1);
}

TEST(DummyEnvPoolTest, MultiPlayers) {
  Runner(1, 1, 20, 100000, 1, 10);
  Runner(3, 1, 20, 100000, 1, 10);
  Runner(3, 1, 20, 100000, 3, 10);
  Runner(9, 4, 30, 100000, 1, 6);
  Runner(9, 4, 30, 100000, 4, 6);
  Runner(9, 4, 30, 100000, 9, 6);
  Runner(10, 10, 25, 100000, 0, 9);
}
