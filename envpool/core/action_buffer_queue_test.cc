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

#include "envpool/core/action_buffer_queue.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <queue>
#include <random>
#include <tuple>
#include <utility>

#include "ThreadPool.h"
#include "envpool/core/dict.h"
#include "envpool/core/spec.h"

using ActionSlice = typename ActionBufferQueue::ActionSlice;

TEST(ActionBufferQueueTest, Concurrent) {
  std::size_t num_envs = 1000;
  ActionBufferQueue queue(num_envs);
  std::srand(std::time(nullptr));
  std::size_t mul = 2000;
  std::vector<ActionSlice> actions;
  // enqueue all envs
  for (std::size_t i = 0; i < num_envs; ++i) {
    actions.push_back(ActionSlice{
        .env_id = static_cast<int>(i), .order = -1, .force_reset = false});
  }
  queue.EnqueueBulk(actions);
  std::vector<std::atomic<std::size_t>> flag(mul);
  std::vector<std::size_t> env_num(mul);
  for (std::size_t m = 0; m < mul; ++m) {
    flag[m] = 1;
    env_num[m] = std::rand() % (num_envs - 1) + 1;
  }

  std::thread send([&] {
    for (std::size_t m = 0; m < mul; ++m) {
      while (flag[m] == 1) {
      }
      actions.clear();
      for (std::size_t i = 0; i < env_num[m]; ++i) {
        actions.push_back(ActionSlice{
            .env_id = static_cast<int>(i), .order = -1, .force_reset = false});
      }
      queue.EnqueueBulk(actions);
    }
  });
  std::thread recv([&] {
    for (std::size_t m = 0; m < mul; ++m) {
      for (std::size_t i = 0; i < env_num[m]; ++i) {
        queue.Dequeue();
      }
      flag[m] = 0;
    }
  });
  recv.join();
  send.join();
  EXPECT_EQ(queue.SizeApprox(), num_envs);
}
