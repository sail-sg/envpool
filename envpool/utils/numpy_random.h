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

#ifndef ENVPOOL_UTILS_NUMPY_RANDOM_H_
#define ENVPOOL_UTILS_NUMPY_RANDOM_H_

#include <array>
#include <cmath>
#include <cstdint>
#include <random>

#ifdef _MSC_VER
#include <intrin.h>
#endif
#define PCG_FORCE_EMULATED_128BIT_MATH
#include "third_party/numpy/pcg64.h"
#undef PCG_FORCE_EMULATED_128BIT_MATH
#ifdef inline
#undef inline
#endif

namespace envpool::numpy_random {

#include "third_party/numpy/ziggurat_constants.h"

// NumPy's PCG64 and sampling conventions. Seed expansion may differ; oracle
// tests synchronize the complete bit-generator state once after reset.
class Random {
 public:
  explicit Random(std::uint32_t seed = 0) {
    std::mt19937 seeds(seed);
    auto word = [&seeds] {
      auto high = static_cast<std::uint64_t>(seeds());
      return (high << 32) | seeds();
    };
    const auto state_high = word();
    const auto state_low = word();
    const auto sequence_high = word();
    const auto sequence_low = word();
    const auto state = PCG_128BIT_CONSTANT(state_high, state_low);
    const auto sequence = PCG_128BIT_CONSTANT(sequence_high, sequence_low);
    pcg64_srandom_r(&engine_, state, sequence);
  }

  double Uniform(double low = 0, double high = 1) {
    double unit = static_cast<double>(pcg64_random_r(&engine_) >> 11) *
                  (1.0 / 9007199254740992.0);
    return low + (high - low) * unit;
  }

  int Integer(std::uint32_t bound) {
    if (bound == 1) {
      return 0;
    }
    auto next = [this]() {
      if (has_word_) {
        has_word_ = false;
        return word_;
      }
      auto value = pcg64_random_r(&engine_);
      word_ = static_cast<std::uint32_t>(value >> 32);
      has_word_ = true;
      return static_cast<std::uint32_t>(value);
    };
    std::uint64_t product = static_cast<std::uint64_t>(next()) * bound;
    auto lower = static_cast<std::uint32_t>(product);
    if (lower < bound) {
      auto threshold = -bound % bound;
      while (lower < threshold) {
        product = static_cast<std::uint64_t>(next()) * bound;
        lower = static_cast<std::uint32_t>(product);
      }
    }
    return static_cast<int>(product >> 32);
  }

  // NumPy's Ziggurat sampling protocol, using ordinary native exp/log1p.
  // Adapted from NumPy 2.5.2 (BSD-3-Clause); see mpe2/NUMPY_LICENSE.
  // Matching the random stream permits independent stochastic oracle rollouts;
  // the resulting floating-point values are compared numerically.
  double Normal() {
    for (;;) {
      std::uint64_t bits = pcg64_random_r(&engine_);
      const int index = bits & 0xff;
      bits >>= 8;
      const bool negative = bits & 1;
      const auto magnitude = (bits >> 1) & 0x000fffffffffffffULL;
      double value = magnitude * wi_double[index];
      if (negative) value = -value;
      if (magnitude < ki_double[index]) return value;
      if (index == 0) {
        for (;;) {
          const double x = -ziggurat_nor_inv_r * std::log1p(-Uniform());
          const double y = -std::log1p(-Uniform());
          if (2 * y > x * x) {
            return ((magnitude >> 8) & 1) ? -(ziggurat_nor_r + x)
                                          : ziggurat_nor_r + x;
          }
        }
      } else if ((fi_double[index - 1] - fi_double[index]) * Uniform() +
                     fi_double[index] <
                 std::exp(-0.5 * value * value)) {
        return value;
      }
    }
  }

  std::array<std::uint64_t, 6> State() const {
    return {engine_.state.high,
            engine_.state.low,
            engine_.inc.high,
            engine_.inc.low,
            static_cast<std::uint64_t>(has_word_),
            word_};
  }

 private:
  pcg64_random_t engine_{};
  bool has_word_{false};
  std::uint32_t word_{0};
};

}  // namespace envpool::numpy_random
#endif  // ENVPOOL_UTILS_NUMPY_RANDOM_H_
