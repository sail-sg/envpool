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

#include "envpool/core/state_buffer.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdint>

#include "envpool/core/spec.h"

TEST(StateBufferTest, Basic) {
  int batch = 32;
  std::vector<ShapeSpec> specs{ShapeSpec(1, {batch, 10, 2, 2}),
                               ShapeSpec(4, {batch, 1, 2, 2})};
  int max_num_players = 10;
  StateBuffer buffer(batch, max_num_players, specs,
                     std::vector<bool>({false, false}));
  auto offset = buffer.Offsets();
  std::size_t total = 0;
  std::srand(std::time(nullptr));
  for (int i = 0; i < batch; ++i) {
    std::size_t num = 1;
    total += num;
    auto r = buffer.Allocate(num);
    offset = buffer.Offsets();
    EXPECT_EQ(std::get<0>(offset), std::get<1>(offset));
    r.done_write();
  }
  auto bs = buffer.Wait();
  EXPECT_EQ(bs[0].Shape(0), total);
}

TEST(StateBufferTest, SinglePlayerSync) {
  int batch = 32;
  std::vector<ShapeSpec> specs{ShapeSpec(1, {batch, 10, 2, 2}),
                               ShapeSpec(4, {batch, 1, 2, 2})};
  int max_num_players = 1;
  StateBuffer buffer(batch, max_num_players, specs,
                     std::vector<bool>({false, false}));
  auto offset = buffer.Offsets();
  std::size_t total = 0;
  std::srand(std::time(nullptr));
  for (int i = 0; i < batch; ++i) {
    std::size_t num = 1;
    total += num;
    // use reversed order to write data
    auto r = buffer.Allocate(num, batch - 1 - i);
    offset = buffer.Offsets();
    EXPECT_EQ(std::get<0>(offset), std::get<1>(offset));
    EXPECT_EQ(r.arr[0].Shape(), std::vector<std::size_t>({10, 2, 2}));
    EXPECT_EQ(r.arr[1].Shape(), std::vector<std::size_t>({1, 2, 2}));
    r.arr[1](0, 0, 0) = i;  // only the first element is modified
    r.done_write();
  }
  auto bs = buffer.Wait();
  EXPECT_EQ(bs[0].Shape(0), total);
  for (int i = 0; i < batch; ++i) {
    auto* ptr = reinterpret_cast<int*>(bs[1][i].Data());
    EXPECT_EQ(ptr[0], batch - 1 - i);
  }
}

TEST(StateBufferTest, Truncate) {
  int batch = 32;
  int max_num_players = 10;
  std::vector<ShapeSpec> specs{ShapeSpec(1, {batch, 10, 2, 2}),
                               ShapeSpec(4, {batch * max_num_players, 2, 2})};
  std::size_t player_num = 3;
  StateBuffer buffer(batch, max_num_players, specs,
                     std::vector<bool>({false, true}));
  auto r = buffer.Allocate(player_num);
  r.done_write();
  auto bs = buffer.Wait(batch - 1);
  EXPECT_EQ(bs[0].Shape(), std::vector<std::size_t>({1, 10, 2, 2}));
  EXPECT_EQ(bs[1].Shape(), std::vector<std::size_t>(
                               {static_cast<std::size_t>(player_num), 2, 2}));
}

TEST(StateBufferTest, MultiPlayers) {
  int batch = 32;
  int max_num_players = 10;
  std::vector<ShapeSpec> specs{ShapeSpec(1, {batch * max_num_players, 2, 2}),
                               ShapeSpec(4, {batch, 1, 2, 2})};
  StateBuffer buffer(batch, max_num_players, specs,
                     std::vector<bool>({true, false}));
  auto offset = buffer.Offsets();
  int total = 0;
  std::srand(std::time(nullptr));
  for (int i = 0; i < batch; ++i) {
    int num = 1 + std::rand() % max_num_players;
    total += num;
    auto r = buffer.Allocate(num);
    offset = buffer.Offsets();
    EXPECT_EQ(num, r.arr[0].Shape()[0]);
    EXPECT_EQ(std::get<0>(offset), total);
    EXPECT_EQ(std::get<1>(offset), i + 1);
    r.done_write();
  }
  auto bs = buffer.Wait();
  EXPECT_EQ(bs[0].Shape(0), total);
  EXPECT_EQ(bs[1].Shape(0), batch);
}
