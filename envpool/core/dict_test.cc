#include "envpool/core/dict.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>

TEST(DictTest, Keys) {
  auto d = MakeDict("abc"_.bind(0.), "xyz"_.bind(0.), "ijk"_.bind(1));
  auto keys = decltype(d)::static_keys();
  auto dkeys = d.keys();
  EXPECT_EQ(dkeys[0], "abc");
  EXPECT_EQ(dkeys[1], "xyz");
  EXPECT_EQ(dkeys[2], "ijk");
  EXPECT_EQ(std::get<0>(keys).str(), "abc");
  EXPECT_EQ(std::get<1>(keys).str(), "xyz");
  EXPECT_EQ(std::get<2>(keys).str(), "ijk");
}

TEST(DictTest, Values) {
  auto d = MakeDict("abc"_.bind(0.), "xyz"_.bind(0.), "ijk"_.bind(1));
  auto values = d.values();
  auto int_vector = d.values<int>();
  EXPECT_EQ(std::get<0>(values), 0.);
  EXPECT_EQ(std::get<1>(values), 0.);
  EXPECT_EQ(std::get<2>(values), 1);
  EXPECT_EQ(int_vector[0], 0);
  EXPECT_EQ(int_vector[1], 0);
  EXPECT_EQ(int_vector[2], 1);
}

TEST(DictTest, Lookup) {
  auto d = MakeDict("abc"_.bind(0.), "xyz"_.bind(0.), "ijk"_.bind(1));
  EXPECT_EQ(d["abc"_], 0.);
  EXPECT_EQ(d["xyz"_], 0.);
  EXPECT_EQ(d["ijk"_], 1);
}
