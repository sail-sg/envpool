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

#include "envpool/core/state_buffer_queue.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <random>

#include "ThreadPool.h"

TEST(StateBufferQueueTest, Basic) {
  std::vector<ShapeSpec> specs{ShapeSpec(1, {10, 2, 4}),
                               ShapeSpec(4, {1, 2, 2})};
  std::size_t batch = 32;
  std::size_t num_envs = 50;
  std::size_t max_num_players = 10;
  StateBufferQueue queue(batch, num_envs, max_num_players, specs);
  std::srand(std::time(nullptr));
  std::size_t size = 0;
  for (std::size_t i = 0; i < batch; ++i) {
    LOG(INFO) << i << " start";
    std::size_t num_players = 1;
    auto slice = queue.Allocate(num_players);
    LOG(INFO) << i << " allocate";
    slice.done_write();
    LOG(INFO) << i << " done_write";
    EXPECT_EQ(slice.arr[0].Shape(0), 10);
    EXPECT_EQ(slice.arr[1].Shape(0), 1);
    size += num_players;
  }
  std::vector<Array> out = queue.Wait();
  LOG(INFO) << "finish wait";
  EXPECT_EQ(out[0].Shape(0), size);
  EXPECT_EQ(out[1].Shape(0), size);
  EXPECT_EQ(batch, size);
}

TEST(StateBufferQueueTest, SinglePlayerSync) {
  std::vector<ShapeSpec> specs{ShapeSpec(4, {-1}), ShapeSpec(4, {1, 2, 2})};
  std::size_t batch = 256;
  std::size_t num_envs = 256;
  std::size_t max_num_players = 1;
  StateBufferQueue queue(batch, num_envs, max_num_players, specs);
  std::mt19937 gen(0);
  std::srand(std::time(nullptr));
  std::size_t mul = 100;
  std::vector<int> order;
  std::vector<int> env_id;
  for (std::size_t i = 0; i < num_envs; ++i) {
    order.push_back(i);
    env_id.push_back(i);
  }
  for (std::size_t m = 0; m < mul; ++m) {
    std::shuffle(order.begin(), order.end(), gen);
    for (std::size_t i = 0; i < batch; ++i) {
      auto slice = queue.Allocate(1, order[i]);
      EXPECT_EQ(slice.arr[0].Shape(0), 1);
      slice.arr[0] = static_cast<int>(i);
      slice.done_write();
    }
    std::vector<Array> out = queue.Wait();
    EXPECT_EQ(out[0].Shape(0), batch);
    auto* ptr = reinterpret_cast<int*>(out[0].Data());
    for (std::size_t i = 0; i < batch; ++i) {
      EXPECT_EQ(ptr[order[i]], i);
    }
  }
  // remove env_id to see if it works with no deadlock
  while (env_id.size() > 1) {
    std::shuffle(env_id.begin(), env_id.end(), gen);
    env_id.pop_back();
    for (std::size_t i = 0; i < env_id.size(); ++i) {
      auto slice = queue.Allocate(1, i);
      slice.arr[0] = env_id[i];
      slice.done_write();
    }
    std::vector<Array> out = queue.Wait(batch - env_id.size());
    EXPECT_EQ(out[0].Shape(0), env_id.size());
    auto* ptr = reinterpret_cast<int*>(out[0].Data());
    for (std::size_t i = 0; i < env_id.size(); ++i) {
      EXPECT_EQ(ptr[i], env_id[i]);
    }
  }
}

TEST(StateBufferQueueTest, NumPlayers) {
  std::vector<ShapeSpec> specs{ShapeSpec(1, {-1, 2, 4}),
                               ShapeSpec(4, {1, 2, 2})};
  std::size_t batch = 32;
  std::size_t num_envs = 500;
  std::size_t max_num_players = 10;
  StateBufferQueue queue(batch, num_envs, max_num_players, specs);
  std::srand(std::time(nullptr));
  std::size_t size = 0;
  for (std::size_t i = 0; i < batch; ++i) {
    std::size_t num_players = 1 + std::rand() % max_num_players;
    auto slice = queue.Allocate(num_players);
    slice.done_write();
    EXPECT_EQ(slice.arr[0].Shape(0), num_players);
    EXPECT_EQ(slice.arr[1].Shape(0), 1);
    size += num_players;
  }
  std::vector<Array> out = queue.Wait(batch * max_num_players - size);
  EXPECT_EQ(out[0].Shape(0), size);
  EXPECT_EQ(out[1].Shape(0), batch);
}

TEST(StateBufferQueueTest, MultipleTimes) {
  std::vector<ShapeSpec> specs{ShapeSpec(1, {-1, 2, 4}),
                               ShapeSpec(4, {1, 2, 2})};
  std::size_t batch = 32;
  std::size_t num_envs = 500;
  std::size_t max_num_players = 10;
  StateBufferQueue queue(batch, num_envs, max_num_players, specs);
  std::srand(std::time(nullptr));
  std::size_t mul = 10000;
  for (std::size_t m = 0; m < mul; ++m) {
    std::size_t size = 0;
    for (std::size_t i = 0; i < batch; ++i) {
      std::size_t num_players = 1 + std::rand() % max_num_players;
      auto slice = queue.Allocate(num_players);
      slice.done_write();
      EXPECT_EQ(slice.arr[0].Shape(0), num_players);
      EXPECT_EQ(slice.arr[1].Shape(0), 1);
      size += num_players;
    }
    std::vector<Array> out = queue.Wait();
    EXPECT_EQ(out[0].Shape(0), size);
    EXPECT_EQ(out[1].Shape(0), batch);
  }
}

TEST(StateBufferQueueTest, ConcurrentSinglePlayer) {
  std::vector<ShapeSpec> specs{ShapeSpec(8, {-1, 2, 4}),
                               ShapeSpec(4, {1, 2, 2})};
  std::size_t batch = 31;
  std::size_t num_envs = 100;
  std::size_t max_num_players = 1;
  StateBufferQueue queue(batch, num_envs, max_num_players, specs);
  ThreadPool pool(batch);
  std::srand(std::time(nullptr));
  // reset
  for (std::size_t i = 0; i < num_envs; ++i) {
    pool.enqueue([&] {
      auto slice = queue.Allocate(1);
      slice.done_write();
    });
  }
  std::size_t total = 10000;
  for (std::size_t m = 0; m < total; ++m) {
    // recv
    auto out = queue.Wait();
    for (std::size_t i = 0; i < batch; ++i) {
      pool.enqueue([&] {
        auto slice = queue.Allocate(1);
        std::this_thread::sleep_for(
            std::chrono::nanoseconds(std::rand() % 1000 + 1));
        slice.done_write();
      });
    }
  }
}

TEST(StateBufferQueueTest, ConcurrentMultiPlayer) {
  std::vector<ShapeSpec> specs{ShapeSpec(8, {-1, 2, 4}),
                               ShapeSpec(4, {1, 2, 2})};
  std::size_t batch = 256;
  std::size_t num_envs = 1000;
  std::size_t max_num_players = 10;
  StateBufferQueue queue(batch, num_envs, max_num_players, specs);
  ThreadPool pool(batch);
  std::srand(std::time(nullptr));
  // reset
  for (std::size_t i = 0; i < num_envs; ++i) {
    pool.enqueue([&] {
      std::size_t num_players = 1 + std::rand() % max_num_players;
      auto slice = queue.Allocate(num_players);
      slice.done_write();
    });
  }
  std::size_t total = 1000;
  for (std::size_t m = 0; m < total; ++m) {
    // recv
    auto out = queue.Wait();
    for (std::size_t i = 0; i < batch; ++i) {
      pool.enqueue([&] {
        std::size_t num_players = 1 + std::rand() % max_num_players;
        auto slice = queue.Allocate(num_players);
        std::this_thread::sleep_for(
            std::chrono::nanoseconds(std::rand() % 1000 + 1));
        slice.done_write();
      });
    }
  }
}
