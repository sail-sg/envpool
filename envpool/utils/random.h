// Copyright 2026 Garena Online Private Limited
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ENVPOOL_UTILS_RANDOM_H_
#define ENVPOOL_UTILS_RANDOM_H_

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace envpool {

// Native model sampling; reproducible on the same build, without a foreign
// generator's seed mapping or distribution implementation as part of the API.
class Random {
 public:
  explicit Random(std::uint32_t seed = 0) : generator_(seed) {}

  double Uniform(double low = 0, double high = 1) {
    return std::uniform_real_distribution<double>(low, high)(generator_);
  }

  template <typename T>
  T Integer(T high) {
    return std::uniform_int_distribution<T>(0, high - 1)(generator_);
  }

  template <typename T>
  void Shuffle(std::vector<T>* values) {
    std::shuffle(values->begin(), values->end(), generator_);
  }

  double Normal() { return normal_(generator_); }

 private:
  std::mt19937 generator_;
  std::normal_distribution<double> normal_;
};

}  // namespace envpool
#endif  // ENVPOOL_UTILS_RANDOM_H_
