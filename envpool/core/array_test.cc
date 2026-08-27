// Copyright 2026 Garena Online Private Limited
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

#include "envpool/core/array.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "envpool/core/spec.h"

TEST(ArrayTest, ViewsPreserveHighDimensionalShapesAndData) {
  ShapeSpec spec(sizeof(int), {2, 3, 4, 5, 6});
  Array array(spec);

  auto view = array(1, 2);
  EXPECT_EQ(view.Shape(), std::vector<std::size_t>({4, 5, 6}));
  view(3, 4, 5) = 17;
  EXPECT_EQ(*reinterpret_cast<int*>(array(1, 2, 3, 4, 5).Data()), 17);

  auto slice = array.Slice(1, 2);
  EXPECT_EQ(slice.Shape(), std::vector<std::size_t>({1, 3, 4, 5, 6}));
  slice(0, 2, 3, 4, 5) = 23;
  EXPECT_EQ(*reinterpret_cast<int*>(array(1, 2, 3, 4, 5).Data()), 23);
}

TEST(ArrayTest, TruncateAndSharedPtrKeepOwnedStorageAlive) {
  ShapeSpec spec(sizeof(int), {4});
  int delete_count = 0;
  std::shared_ptr<char> shared;

  {
    Array truncated;
    {
      auto* data = new char[4 * sizeof(int)];
      Array parent(spec, data, [&delete_count](char* ptr) {
        ++delete_count;
        delete[] ptr;
      });
      truncated = parent.Truncate(2);
      shared = truncated.SharedPtr();
      EXPECT_EQ(shared.get(), truncated.Data());
    }
    EXPECT_EQ(delete_count, 0);
  }

  EXPECT_EQ(delete_count, 0);
  shared.reset();
  EXPECT_EQ(delete_count, 1);
}

TEST(ArrayTest, SharedPtrPreservesNonOwningDataPointer) {
  ShapeSpec spec(sizeof(int), {2});
  int data[2] = {};
  Array array(spec, reinterpret_cast<char*>(data));

  auto shared = array.SharedPtr();
  EXPECT_EQ(shared.get(), reinterpret_cast<char*>(data));
}

TEST(ArrayTest, ViewsDoNotExtendBackingStorageLifetime) {
  ShapeSpec spec(sizeof(int), {2, 2});
  int delete_count = 0;

  {
    Array indexed_view;
    Array slice;
    {
      auto* data = new char[4 * sizeof(int)];
      Array parent(spec, data, [&delete_count](char* ptr) {
        ++delete_count;
        delete[] ptr;
      });
      indexed_view = parent(1);
      slice = parent.Slice(0, 1);
    }
    EXPECT_EQ(delete_count, 1);
  }

  EXPECT_EQ(delete_count, 1);
}

TEST(ArrayTest, TypedTruncateSharesData) {
  Spec<int> spec(std::vector<int>({4}));
  TArray<int> array(spec);

  auto truncated = array.Truncate(2);
  truncated[1] = 31;

  EXPECT_EQ(static_cast<int>(array[1]), 31);
}
