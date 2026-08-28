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
//
// Adapted from Craftax's MIT-licensed Perlin noise, originally by Pierre
// Vigier.

#ifndef ENVPOOL_CRAFTAX_NOISE_H_
#define ENVPOOL_CRAFTAX_NOISE_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "envpool/craftax/random.h"

namespace craftax {

inline std::vector<float> Noise(Key rng, int height, int width, int res_y,
                                int res_x,
                                const std::vector<float>& angles = {}) {
  rng = Split(Split(rng, 1), 1);
  std::vector<std::array<float, 2>> gradients((res_y + 1) * (res_x + 1));
  for (std::size_t i = 0; i < gradients.size(); ++i) {
    const float angle =
        6.283185307179586f * (angles.empty() ? Uniform(rng, i) : angles.at(i));
    gradients[i] = {std::cos(angle), std::sin(angle)};
  }
  const int dy = height / res_y;
  const int dx = width / res_x;
  std::vector<float> noise(height * width);
  auto smooth = [](float x) {
    return x * x * x * (x * (x * 6.0f - 15.0f) + 10.0f);
  };
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int gy = y / dy;
      const int gx = x / dx;
      const float fy = static_cast<float>(y % dy) / dy;
      const float fx = static_cast<float>(x % dx) / dx;
      auto dot = [&](int cy, int cx) {
        const auto& g = gradients[(gy + cy) * (res_x + 1) + gx + cx];
        return (fy - cy) * g[0] + (fx - cx) * g[1];
      };
      const float ty = smooth(fy);
      const float tx = smooth(fx);
      const float n0 = dot(0, 0) * (1.0f - ty) + ty * dot(1, 0);
      const float n1 = dot(0, 1) * (1.0f - ty) + ty * dot(1, 1);
      noise[y * width + x] = 1.4142135623730951f * ((1.0f - tx) * n0 + tx * n1);
    }
  }
  const float low = *std::min_element(noise.begin(), noise.end());
  const float high = *std::max_element(noise.begin(), noise.end());
  for (auto& value : noise) {
    value = (value - low) / (high - low);
  }
  return noise;
}

}  // namespace craftax

#endif  // ENVPOOL_CRAFTAX_NOISE_H_
