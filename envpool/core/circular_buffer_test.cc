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

#include "circular_buffer.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <random>
#include <thread>

TEST(CircularBufferTest, Basic) {
  CircularBuffer<int> cb(100);
  std::vector<int> number(100000);
  for (auto& n : number) {
    n = std::rand();
  }
  std::thread t_put([&]() {
    for (auto& n : number) {
      cb.Put(n);
    }
  });

  for (auto& n : number) {
    int r = cb.Get();
    EXPECT_EQ(r, n);
  }
  t_put.join();
}
