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

#include <gtest/gtest.h>

#include "envpool/mujoco/myosuite/myosuite_env.h"

namespace myosuite {

TEST(MyoSuiteTaskTest, TableTennisRally) {
  // Contacts are paddle=1, own half=2, opponent half=4.
  EXPECT_EQ(EvaluateTableTennisContacts({0, 2, 2, 0, 1, 1, 0}),
            TableTennisResult::kPending);
  EXPECT_EQ(EvaluateTableTennisContacts({0, 2, 2, 0, 1, 1, 0, 4}),
            TableTennisResult::kSolved);
  EXPECT_EQ(EvaluateTableTennisContacts({0, 4}), TableTennisResult::kFault);
  EXPECT_EQ(EvaluateTableTennisContacts({1, 0, 1}), TableTennisResult::kFault);
  EXPECT_EQ(EvaluateTableTennisContacts({2, 0, 2}), TableTennisResult::kFault);
  EXPECT_EQ(EvaluateTableTennisContacts({2, 2, 2}), TableTennisResult::kFault);
  // The first conclusive event decides the rally.
  EXPECT_EQ(EvaluateTableTennisContacts({1, 0, 4, 1}),
            TableTennisResult::kSolved);
}

}  // namespace myosuite
