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

#ifndef ENVPOOL_MUJOCO_LOCOMOTION_RANDOM_H_
#define ENVPOOL_MUJOCO_LOCOMOTION_RANDOM_H_

#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace mujoco_locomotion {

// NumPy RandomState's MT19937 sampling conventions. Avoid the STL distribution
// classes: their algorithms differ across libstdc++, libc++, and MSVC.
class RandomState {
 public:
  explicit RandomState(uint32_t seed) : generator_(seed) {}

  double Uniform(double low = 0, double high = 1);

  uint32_t Int(uint32_t high) {
    if (high <= 1) {
      return 0;
    }
    uint32_t mask = high - 1;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    uint32_t value;
    do {
      value = generator_() & mask;
    } while (value >= high);
    return value;
  }

  double Normal();

  template <typename T>
  void Shuffle(std::vector<T>* values) {
    for (std::size_t i = values->size(); i > 1; --i) {
      std::swap((*values)[i - 1], (*values)[Int(i)]);
    }
  }

 private:
  std::mt19937 generator_;
  double gaussian_{0};
  bool has_gaussian_{false};
};

}  // namespace mujoco_locomotion

#endif  // ENVPOOL_MUJOCO_LOCOMOTION_RANDOM_H_
