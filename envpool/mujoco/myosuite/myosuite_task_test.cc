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

#include <array>

#include "envpool/mujoco/myosuite/myosuite_env.h"

namespace myosuite {

TEST(MyoSuiteTaskTest, BrownNoiseHasVariationAndTemporalCorrelation) {
  double variance = 0;
  double changes = 0;
  for (int seed = 0; seed < 32; ++seed) {
    envpool::Random random(seed);
    envpool::BrownNoise noise;
    noise.Reset(2000, 1, &random);
    std::array<std::array<double, 2>, 2000> samples{};
    std::array<double, 2> mean{};
    for (auto& sample : samples) {
      sample = noise.Sample(&random);
      for (int channel = 0; channel < 2; ++channel) {
        mean[channel] += sample[channel] / samples.size();
      }
    }
    for (std::size_t i = 0; i < samples.size(); ++i) {
      for (int channel = 0; channel < 2; ++channel) {
        const double centered = samples[i][channel] - mean[channel];
        variance += centered * centered;
        if (i > 0) {
          const double delta = samples[i][channel] - samples[i - 1][channel];
          changes += delta * delta;
        }
      }
    }
  }
  // Unit-variance beta=2 noise has smooth neighboring samples. These broad
  // distribution checks reject frozen or white noise without matching a PRNG.
  EXPECT_GT(variance / (32 * 2 * 2000), 0.25);
  EXPECT_LT(variance / (32 * 2 * 2000), 4.0);
  EXPECT_LT(changes / variance, 0.05);
}

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
