// Copyright 2026 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ENVPOOL_UTILS_JAX_RANDOM_H_
#define ENVPOOL_UTILS_JAX_RANDOM_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace envpool::jax_random {

// Threefry2x32-20, with JAX's partitionable counter layout. Keys are values:
// drawing from one never implicitly advances another stream.
using Key = std::array<std::uint32_t, 2>;

inline Key Threefry(Key key, Key count) {
  constexpr std::array<int, 8> rotations{13, 15, 26, 6, 17, 29, 16, 24};
  const std::array<std::uint32_t, 3> keys{key[0], key[1],
                                          key[0] ^ key[1] ^ 0x1BD11BDAU};
  count[0] += keys[0];
  count[1] += keys[1];
  for (int round = 0; round < 20; ++round) {
    count[0] += count[1];
    const int shift = rotations[round % 8];
    count[1] = (count[1] << shift) | (count[1] >> (32 - shift));
    count[1] ^= count[0];
    if (round % 4 == 3) {
      const int injection = (round + 1) / 4;
      count[0] += keys[injection % 3];
      count[1] += keys[(injection + 1) % 3] + injection;
    }
  }
  return count;
}

inline Key Split(Key key, std::uint32_t index) {
  return Threefry(key, {0, index});
}

inline Key TakeKey(Key* key) {
  const Key draw = Split(*key, 1);
  *key = Split(*key, 0);
  return draw;
}

inline std::uint32_t Bits(Key key, std::uint32_t index = 0) {
  const Key bits = Threefry(key, {0, index});
  return bits[0] ^ bits[1];
}

inline float Uniform(Key key, std::uint32_t index = 0) {
  const std::uint32_t bits = (Bits(key, index) >> 9) | 0x3F800000U;
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value - 1.0f;
}

inline int RandInt(Key key, int low, int high, std::uint32_t index = 0) {
  const auto span = static_cast<std::uint32_t>(std::max(1, high - low));
  const std::uint32_t half = 65536U % span;
  const std::uint32_t multiplier = (half * half) % span;
  const std::uint32_t higher = Bits(Split(key, 0), index);
  const std::uint32_t lower = Bits(Split(key, 1), index);
  return low +
         static_cast<int>(((higher % span) * multiplier + lower % span) % span);
}

// Categorical sampling depends on the weights, not a backend's scan layout.
inline std::vector<float> Cumulative(const std::vector<float>& values) {
  std::vector<float> out(values.size());
  std::partial_sum(values.begin(), values.end(), out.begin());
  return out;
}

// Inputs need not sum to one.
inline int Choice(Key key, const std::vector<float>& weights) {
  const auto cumulative = Cumulative(weights);
  const float sample = cumulative.back() * (1.0f - Uniform(key));
  return static_cast<int>(
      std::lower_bound(cumulative.begin(), cumulative.end(), sample) -
      cumulative.begin());
}

}  // namespace envpool::jax_random

#endif  // ENVPOOL_UTILS_JAX_RANDOM_H_
