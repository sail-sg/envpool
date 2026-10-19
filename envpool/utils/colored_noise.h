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

#ifndef ENVPOOL_UTILS_COLORED_NOISE_H_
#define ENVPOOL_UTILS_COLORED_NOISE_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <vector>

#include "envpool/utils/numpy_random.h"
#include "third_party/numpy/pocketfft_hdronly.h"

namespace envpool::numpy_random {

// Two-channel, unit-variance beta=2 noise, as used by MyoSuite's NPCs.
// Generate spectral Gaussian coefficients and transform them to a finite
// periodic stream; keep the stream cursor as ordinary runtime state.
class BrownNoise {
 public:
  void Reset(int size, double scale, Random* random) {
    size_ = size;
    scale_ = scale;
    buffer_.resize(2 * size_);
    Refill(random);
  }

  std::array<double, 2> Sample(Random* random) {
    if (cursor_ == size_) Refill(random);
    const int index = cursor_++;
    return {scale_ * buffer_[index], scale_ * buffer_[size_ + index]};
  }

 private:
  int size_{0};
  int cursor_{0};
  double scale_{1};
  std::vector<double> buffer_;

  void Refill(Random* random) {
    const int frequencies = size_ / 2 + 1;
    std::vector<double> amplitude(frequencies);
    double variance = 0;
    for (int i = 0; i < frequencies; ++i) {
      amplitude[i] = static_cast<double>(size_) / std::max(i, 1);
      if (i > 0) {
        const double weight =
            amplitude[i] * (i == frequencies - 1 ? (1 + size_ % 2) / 2.0 : 1.0);
        variance += weight * weight;
      }
    }
    const double sigma = 2 * std::sqrt(variance) / size_;
    std::vector<std::complex<double>> spectrum(2 * frequencies);
    for (int i = 0; i < 2 * frequencies; ++i) {
      spectrum[i].real(random->Normal() * amplitude[i % frequencies]);
    }
    for (int i = 0; i < 2 * frequencies; ++i) {
      spectrum[i].imag(random->Normal() * amplitude[i % frequencies]);
    }
    for (int channel = 0; channel < 2; ++channel) {
      auto* values = spectrum.data() + channel * frequencies;
      values[0] = {values[0].real() * std::sqrt(2.0), 0};
      if (size_ % 2 == 0) {
        values[frequencies - 1] = {
            values[frequencies - 1].real() * std::sqrt(2.0), 0};
      }
      pocketfft::c2r<double>({static_cast<size_t>(size_)},
                             {sizeof(std::complex<double>)}, {sizeof(double)},
                             0, false, values, buffer_.data() + channel * size_,
                             1.0 / (size_ * sigma));
    }
    cursor_ = 0;
  }
};

}  // namespace envpool::numpy_random
#endif  // ENVPOOL_UTILS_COLORED_NOISE_H_
