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

#include "envpool/core/dict.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>

TEST(DictTest, Keys) {
  auto d = MakeDict("abc"_.Bind(0.), "xyz"_.Bind(0.), "ijk"_.Bind(1));
  auto keys = decltype(d)::StaticKeys();
  auto dkeys = d.AllKeys();
  EXPECT_EQ(dkeys[0], "abc");
  EXPECT_EQ(dkeys[1], "xyz");
  EXPECT_EQ(dkeys[2], "ijk");
  EXPECT_EQ(std::get<0>(keys).Str(), "abc");
  EXPECT_EQ(std::get<1>(keys).Str(), "xyz");
  EXPECT_EQ(std::get<2>(keys).Str(), "ijk");
}

TEST(DictTest, Values) {
  auto d = MakeDict("abc"_.Bind(0.), "xyz"_.Bind(0.), "ijk"_.Bind(1));
  auto values = d.AllValues();
  auto int_vector = d.AllValues<int>();
  EXPECT_EQ(std::get<0>(values), 0.);
  EXPECT_EQ(std::get<1>(values), 0.);
  EXPECT_EQ(std::get<2>(values), 1);
  EXPECT_EQ(int_vector[0], 0);
  EXPECT_EQ(int_vector[1], 0);
  EXPECT_EQ(int_vector[2], 1);
}

TEST(DictTest, Lookup) {
  auto d = MakeDict("abc"_.Bind(0.), "xyz"_.Bind(0.), "ijk"_.Bind(1));
  EXPECT_EQ(d["abc"_], 0.);
  EXPECT_EQ(d["xyz"_], 0.);
  EXPECT_EQ(d["ijk"_], 1);
}

TEST(DictTest, Modification) {
  auto d = MakeDict("abc"_.Bind(0.), "xyz"_.Bind("123"), "ijk"_.Bind(1));
  EXPECT_EQ(d["abc"_], 0.);
  EXPECT_EQ(d["xyz"_], "123");
  EXPECT_EQ(d["ijk"_], 1);
  d["abc"_] = 1;
  d["xyz"_] = "456";
  // force convert to int
  d["ijk"_] = 0.5;  // NOLINT
  EXPECT_EQ(d["abc"_], 1);
  EXPECT_EQ(d["xyz"_], "456");
  EXPECT_EQ(d["ijk"_], 0);
}
