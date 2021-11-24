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

#include "envpool/atari/atari_env.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <random>

typedef typename atari::AtariEnv::State AtariState;
typedef typename atari::AtariEnv::Action AtariAction;

TEST(AtariEnvTest, GrayScaleMaxPoolOrder) {
  std::string rom_path = atari::GetRomPath("envpool", "pong");
  const int N = 256;
  uint8_t arr[N * N];      // grayscale -> maxpool
  uint8_t arr_ref[N * N];  // maxpool -> grayscale
  uint8_t ptr0[N * N];
  uint8_t ptr1[N * N];
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      ptr0[i * N + j] = i;
      ptr1[i * N + j] = j;
    }
  }
  Array col0(Spec<uint8_t>({N, N, 3}));
  Array col1(Spec<uint8_t>({N, N, 3}));
  Array result(Spec<uint8_t>({N, N, 1}));
  uint8_t* col0_ptr = static_cast<uint8_t*>(col0.data());
  uint8_t* col1_ptr = static_cast<uint8_t*>(col1.data());
  uint8_t* result_ptr = static_cast<uint8_t*>(result.data());
  ale::ALEInterface env;
  env.loadROM(rom_path);
  // color mapping from pixel_t to RGB
  env.theOSystem->colourPalette().applyPaletteRGB(col0_ptr, ptr0, N * N);
  env.theOSystem->colourPalette().applyPaletteRGB(col1_ptr, ptr1, N * N);
  // maxpool RGB
  for (int i = 0; i < N * N * 3; ++i) {
    col0_ptr[i] = std::max(col0_ptr[i], col1_ptr[i]);
  }
  // gray scale
  GrayScale(col0, &result);
  memcpy(arr, result_ptr, sizeof arr);
  // ref
  env.theOSystem->colourPalette().applyPaletteGrayscale(col0_ptr, ptr0, N * N);
  env.theOSystem->colourPalette().applyPaletteGrayscale(col1_ptr, ptr1, N * N);
  // maxpool
  for (int i = 0; i < N * N; ++i) {
    arr_ref[i] = std::max(col0_ptr[i], col1_ptr[i]);
  }
  // calc diff
  int diff_count = 0;
  int total_count = 0;
  int diff_sum = 0;
  for (int i = 0; i < N; i += 2) {
    for (int j = 0; j < N; j += 2) {
      int k = i * N + j;
      ++total_count;
      if (arr[k] != arr_ref[k]) {
        ++diff_count;
        if (arr[k] > arr_ref[k]) {
          diff_sum += arr[k] - arr_ref[k];
        } else {
          diff_sum += arr_ref[k] - arr[k];
        }
      }
    }
  }
  LOG(INFO) << diff_count << "/" << total_count;
  LOG(INFO) << "Mean diff " << 1.0 * diff_sum / total_count;
}

TEST(AtariEnvTest, Seed) {
  std::srand(std::time(nullptr));
  auto config = atari::AtariEnvSpec::default_config;
  std::size_t batch = 4;
  config["num_envs"_] = batch;
  config["batch_size"_] = batch;
  config["seed"_] = 0;
  int total_iter = 10000;
  atari::AtariEnvSpec spec(config);
  atari::AtariEnvPool envpool0(spec), envpool1(spec);
  Array all_env_ids(Spec<int>({static_cast<int>(batch)}));
  for (std::size_t i = 0; i < batch; ++i) {
    all_env_ids[i] = i;
  }
  envpool0.Reset(all_env_ids);
  envpool1.Reset(all_env_ids);
  std::vector<Array> raw_action(3);
  AtariAction action(&raw_action);
  for (int i = 0; i < total_iter; ++i) {
    auto state_vec0 = envpool0.Recv();
    auto state_vec1 = envpool1.Recv();
    AtariState state0(&state_vec0);
    AtariState state1(&state_vec1);
    EXPECT_EQ(state0["obs"_].Shape(),
              std::vector<std::size_t>({batch, 4, 84, 84}));
    EXPECT_EQ(state1["obs"_].Shape(),
              std::vector<std::size_t>({batch, 4, 84, 84}));
    uint8_t* data0 = static_cast<uint8_t*>(state0["obs"_].data());
    uint8_t* data1 = static_cast<uint8_t*>(state1["obs"_].data());
    int index = 0;
    for (std::size_t j = 0; j < batch * 4; ++j) {
      // ensure there's no black screen in each frame
      int sum = 0;
      for (int k = 0; k < 84 * 84; ++k) {
        EXPECT_EQ(data0[index], data1[index]);
        sum += data0[index++];
      }
      EXPECT_NE(sum, 0) << i << " " << j;
    }
    action["env_id"_] = state0["info:env_id"_];
    action["players.env_id"_] = state0["info:env_id"_];
    action["action"_] = Array(Spec<int>({static_cast<int>(batch)}));
    for (std::size_t j = 0; j < batch; ++j) {
      action["action"_][j] = std::rand() % 6;
    }
    envpool0.Send(action);
    envpool1.Send(action);
  }
}

TEST(AtariEnvTest, MaxEpisodeSteps) {
  auto config = atari::AtariEnvSpec::default_config;
  int batch = 4;
  int max_episode_steps = 10;
  config["num_envs"_] = batch;
  config["batch_size"_] = batch;
  config["seed"_] = 0;
  config["max_episode_steps"_] = max_episode_steps;
  config["repeat_action_probability"_] = 0.2f;
  int total_iter = 100;
  atari::AtariEnvSpec spec(config);
  atari::AtariEnvPool envpool(spec);
  Array all_env_ids(Spec<int>({batch}));
  for (int i = 0; i < batch; ++i) {
    all_env_ids[i] = i;
  }
  envpool.Reset(all_env_ids);
  std::vector<Array> raw_action(3);
  AtariAction action(&raw_action);
  int count = 0;
  for (int i = 0; i < total_iter; ++i) {
    auto state_vec = envpool.Recv();
    AtariState state(&state_vec);
    auto elapsed_step = state["elapsed_step"_];
    for (int j = 0; j < batch; ++j) {
      EXPECT_EQ(count, static_cast<int>(elapsed_step[j]));
    }
    if (count++ == max_episode_steps) {
      count = 0;
    }
    action["env_id"_] = state["info:env_id"_];
    action["players.env_id"_] = state["info:env_id"_];
    action["action"_] = Array(Spec<int>({batch}));
    for (int j = 0; j < batch; ++j) {
      action["action"_][j] = 0;
    }
    envpool.Send(action);
  }
}

TEST(AtariEnvTest, EpisodicLife) {
  std::srand(std::time(nullptr));
  int batch = 4;
  int total_iter = 3000;
  auto config = atari::AtariEnvSpec::default_config;
  config["num_envs"_] = batch;
  config["batch_size"_] = batch;
  config["episodic_life"_] = true;
  config["task"_] = "pong";
  atari::AtariEnvSpec spec(config);
  atari::AtariEnvPool envpool(spec);
  Array all_env_ids(Spec<int>({batch}));
  for (int i = 0; i < batch; ++i) {
    all_env_ids[i] = i;
  }
  envpool.Reset(all_env_ids);
  std::vector<Array> raw_action(3);
  AtariAction action(&raw_action);
  std::vector<bool> last_done(batch);
  std::vector<int> last_lives(batch);
  for (int i = 0; i < total_iter; ++i) {
    auto state_vec = envpool.Recv();
    AtariState state(&state_vec);
    auto done = state["done"_];
    auto lives = state["info:lives"_];
    for (int j = 0; j < batch; ++j) {
      int live = lives[j];
      bool d = done[j];
      EXPECT_EQ(live, 0);
      last_lives[j] = live;
      last_done[j] = d;
    }
    action["env_id"_] = state["info:env_id"_];
    action["players.env_id"_] = state["info:env_id"_];
    action["action"_] = Array(Spec<int>({batch}));
    for (int j = 0; j < batch; ++j) {
      action["action"_][j] = std::rand() % 6;
    }
    envpool.Send(action);
  }

  config["task"_] = "breakout";
  atari::AtariEnvSpec spec2(config);
  atari::AtariEnvPool envpool2(spec2);
  envpool2.Reset(all_env_ids);
  last_lives = std::vector<int>(4);
  last_done = std::vector<bool>(4, true);
  for (int i = 0; i < total_iter; ++i) {
    auto state_vec = envpool2.Recv();
    AtariState state(&state_vec);
    auto done = state["done"_];
    auto lives = state["info:lives"_];
    for (int j = 0; j < batch; ++j) {
      int live = lives[j];
      bool d = done[j];
      if (live == last_lives[j]) {
        EXPECT_FALSE(d);
        EXPECT_GT(live, 0);
      } else if (live == 5) {
        // init of episode
        EXPECT_EQ(last_lives[j], 0);
        EXPECT_FALSE(d);
        EXPECT_TRUE(last_done[j]);
      } else {
        EXPECT_EQ(last_lives[j], live + 1);
        EXPECT_TRUE(d);
        EXPECT_FALSE(last_done[j]);
      }
      last_lives[j] = live;
      last_done[j] = d;
    }
    action["env_id"_] = state["info:env_id"_];
    action["players.env_id"_] = state["info:env_id"_];
    action["action"_] = Array(Spec<int>({batch}));
    for (int j = 0; j < batch; ++j) {
      action["action"_][j] = i % 4;
    }
    envpool2.Send(action);
  }
}

TEST(AtariEnvTest, ZeroDiscountOnLifeLoss) {
  std::srand(std::time(nullptr));
  int batch = 4;
  int total_iter = 3000;
  auto config = atari::AtariEnvSpec::default_config;
  config["num_envs"_] = batch;
  config["batch_size"_] = batch;
  config["task"_] = "breakout";
  atari::AtariEnvSpec spec(config);
  atari::AtariEnvPool envpool(spec);
  config["zero_discount_on_life_loss"_] = true;
  atari::AtariEnvSpec spec2(config);
  atari::AtariEnvPool envpool2(spec2);
  Array all_env_ids(Spec<int>({batch}));
  for (int i = 0; i < batch; ++i) {
    all_env_ids[i] = i;
  }
  envpool.Reset(all_env_ids);
  envpool2.Reset(all_env_ids);
  std::vector<Array> raw_action(3);
  AtariAction action(&raw_action);
  std::vector<bool> last_done(batch, true);
  std::vector<int> last_lives(batch);
  for (int i = 0; i < total_iter; ++i) {
    auto state_vec = envpool.Recv();
    auto state_vec2 = envpool2.Recv();
    AtariState state(&state_vec);
    AtariState state2(&state_vec2);

    auto done = state["done"_];
    auto lives = state["info:lives"_];
    auto discount = state["discount"_];
    auto done2 = state2["done"_];
    auto lives2 = state2["info:lives"_];
    auto discount2 = state2["discount"_];
    for (int j = 0; j < batch; ++j) {
      int live = lives[j], live2 = lives2[j];
      bool d = done[j], d2 = done2[j];
      float disc = discount[j], disc2 = discount2[j];
      EXPECT_EQ(d, d2);
      EXPECT_EQ(live, live2);
      if (live == last_lives[j]) {
        EXPECT_FALSE(d);
        EXPECT_GT(live, 0);
        EXPECT_EQ(disc, 1.0f);
        EXPECT_EQ(disc2, 1.0f);
      } else if (live == 5) {
        // init of episode
        EXPECT_EQ(last_lives[j], 0);
        EXPECT_FALSE(d);
        EXPECT_TRUE(last_done[j]);
        EXPECT_EQ(disc, 1.0f);
        EXPECT_EQ(disc2, 1.0f);
      } else {
        EXPECT_EQ(last_lives[j], live + 1);
        EXPECT_EQ(d, live == 0);
        EXPECT_FALSE(last_done[j]);
        EXPECT_EQ(disc, live > 0);
        EXPECT_EQ(disc2, 0.0f);
      }
      last_lives[j] = live;
      last_done[j] = d;
    }
    action["env_id"_] = state["info:env_id"_];
    action["players.env_id"_] = state["info:env_id"_];
    action["action"_] = Array(Spec<int>({batch}));
    for (int j = 0; j < batch; ++j) {
      action["action"_][j] = i % 4;
    }
    envpool.Send(action);
    envpool2.Send(action);
  }
}

TEST(AtariEnvSpeedTest, Benchmark) {
  int num_envs = 8;
  int batch = 3;
  int num_threads = 3;
  int total_iter = 50000;
  // int num_envs = 655;
  // int batch = 252;
  // int num_threads = 252;
  // int total_iter = 50000;
  auto config = atari::AtariEnvSpec::default_config;
  config["num_envs"_] = num_envs;
  config["batch_size"_] = batch;
  config["num_threads"_] = num_threads;
  config["thread_affinity_offset"_] = 0;
  atari::AtariEnvSpec spec(config);
  atari::AtariEnvPool envpool(spec);
  Array all_env_ids(Spec<int>({num_envs}));
  for (int i = 0; i < num_envs; ++i) {
    all_env_ids[i] = i;
  }
  envpool.Reset(all_env_ids);
  std::vector<Array> raw_action(3);
  AtariAction action(&raw_action);
  action["action"_] = Array(Spec<int>({batch}));
  for (int j = 0; j < batch; ++j) {
    action["action"_][j] = 1;
  }
  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < total_iter; ++i) {
    // recv
    auto state_vec = envpool.Recv();
    AtariState state(&state_vec);
    auto env_id = state["info:env_id"_];
    // EXPECT_EQ(env_id.Shape(),
    // std::vector<std::size_t>({(std::size_t)batch}));
    // send
    action["env_id"_] = env_id;
    action["players.env_id"_] = env_id;
    envpool.Send(action);
  }
  std::chrono::duration<double> dur = std::chrono::system_clock::now() - start;
  double t = dur.count();
  double fps = (total_iter * batch) / t * 4;
  LOG(INFO) << "time(s): " << t << ", FPS: " << fps;
}
