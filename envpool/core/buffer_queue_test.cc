#include "envpool/core/buffer_queue.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(TestBufferQueue, Basic) {
  std::vector<ShapeSpec> specs{ShapeSpec(1, {10, 2, 4}),
                               ShapeSpec(4, {1, 2, 2})};
  std::size_t batch = 32;
  std::size_t num_envs = 500;
  std::size_t max_num_players = 10;
  BufferQueue queue(batch, num_envs, max_num_players, specs);
  std::srand(std::time(nullptr));
  std::size_t size = 0;
  for (std::size_t i = 0; i < batch; ++i) {
    std::size_t num_players = 1;
    auto slice = queue.Allocate(num_players);
    slice.done_write();
    EXPECT_EQ(slice.arr[0].Shape()[0], 10);
    EXPECT_EQ(slice.arr[1].Shape()[0], 1);
    size += num_players;
  }
  std::vector<Array> out = queue.Wait();
  EXPECT_EQ(out[0].Shape()[0], size);
  EXPECT_EQ(out[1].Shape()[0], size);
  EXPECT_EQ(batch, size);
}

TEST(TestBufferQueue, NumPlayers) {
  std::vector<ShapeSpec> specs{ShapeSpec(1, {-1, 2, 4}),
                               ShapeSpec(4, {1, 2, 2})};
  std::size_t batch = 32;
  std::size_t num_envs = 500;
  std::size_t max_num_players = 10;
  BufferQueue queue(batch, num_envs, max_num_players, specs);
  std::srand(std::time(nullptr));
  std::size_t size = 0;
  for (std::size_t i = 0; i < batch; ++i) {
    std::size_t num_players = 1 + std::rand() % max_num_players;
    auto slice = queue.Allocate(num_players);
    slice.done_write();
    EXPECT_EQ(slice.arr[0].Shape()[0], num_players);
    EXPECT_EQ(slice.arr[1].Shape()[0], 1);
    size += num_players;
  }
  std::vector<Array> out = queue.Wait();
  EXPECT_EQ(out[0].Shape()[0], size);
  EXPECT_EQ(out[1].Shape()[0], batch);
}

TEST(TestBufferQueue, MultipleTimes) {
  std::vector<ShapeSpec> specs{ShapeSpec(1, {-1, 2, 4}),
                               ShapeSpec(4, {1, 2, 2})};
  std::size_t batch = 32;
  std::size_t num_envs = 500;
  std::size_t max_num_players = 10;
  BufferQueue queue(batch, num_envs, max_num_players, specs);
  std::srand(std::time(nullptr));
  std::size_t mul = 10000;
  for (std::size_t m = 0; m < mul; ++m) {
    std::size_t size = 0;
    for (std::size_t i = 0; i < batch; ++i) {
      std::size_t num_players = 1 + std::rand() % max_num_players;
      auto slice = queue.Allocate(num_players);
      slice.done_write();
      EXPECT_EQ(slice.arr[0].Shape()[0], num_players);
      EXPECT_EQ(slice.arr[1].Shape()[0], 1);
      size += num_players;
    }
    std::vector<Array> out = queue.Wait();
    EXPECT_EQ(out[0].Shape()[0], size);
    EXPECT_EQ(out[1].Shape()[0], batch);
  }
}
