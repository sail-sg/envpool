/*
 * Copyright 2026 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "envpool/mujoco/locomotion/random.h"

namespace mujoco_locomotion {

double RandomState::Uniform(double low, double high) {
  const uint32_t a = generator_() >> 5;
  const uint32_t b = generator_() >> 6;
  const double unit = (a * 67108864.0 + b) / 9007199254740992.0;
  return low + (high - low) * unit;
}

double RandomState::Normal() {
  if (has_gaussian_) {
    has_gaussian_ = false;
    return gaussian_;
  }
  double x, y, radius;
  do {
    x = 2 * Uniform() - 1;
    y = 2 * Uniform() - 1;
    radius = x * x + y * y;
  } while (radius >= 1 || radius == 0);
  const double scale = std::sqrt(-2 * std::log(radius) / radius);
  gaussian_ = scale * x;
  has_gaussian_ = true;
  return scale * y;
}

}  // namespace mujoco_locomotion
