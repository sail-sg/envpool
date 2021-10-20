#include "envpool/core/state_buffer.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdint>

#include "envpool/core/spec.h"

TEST(StateBufferTest, Basic) {
  std::vector<ShapeSpec> specs{ShapeSpec(1, {10, 2, 2}),
                               ShapeSpec(4, {1, 2, 2})};
  std::size_t batch = 32;
  std::size_t max_num_players = 10;
  StateBuffer buffer(batch, max_num_players, specs);
  auto offset = buffer.Offsets();
  std::size_t total = 0;
  std::srand(std::time(nullptr));
  for (std::size_t i = 0; i < batch; ++i) {
    std::size_t num = 1;
    total += num;
    auto r = buffer.Allocate(num);
    offset = buffer.Offsets();
    EXPECT_EQ(std::get<0>(offset), std::get<1>(offset));
    r.done_write();
  }
  auto bs = buffer.Wait();
  EXPECT_EQ(bs[0].Shape()[0], total);
}

TEST(StateBufferTest, MultiPlayers) {
  std::vector<ShapeSpec> specs{ShapeSpec(1, {-1, 2, 2}),
                               ShapeSpec(4, {1, 2, 2})};
  std::size_t batch = 32;
  std::size_t max_num_players = 10;
  StateBuffer buffer(batch, max_num_players, specs);
  auto offset = buffer.Offsets();
  std::size_t total = 0;
  std::srand(std::time(nullptr));
  for (std::size_t i = 0; i < batch; ++i) {
    std::size_t num = 1 + std::rand() % max_num_players;
    total += num;
    auto r = buffer.Allocate(num);
    offset = buffer.Offsets();
    EXPECT_EQ(num, r.arr[0].Shape()[0]);
    EXPECT_EQ(std::get<0>(offset), total);
    EXPECT_EQ(std::get<1>(offset), i + 1);
    r.done_write();
  }
  auto bs = buffer.Wait();
  EXPECT_EQ(bs[0].Shape()[0], total);
  EXPECT_EQ(bs[1].Shape()[0], batch);
}
