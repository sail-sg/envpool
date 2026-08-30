// Copyright 2026 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "envpool/mujoco/mjlab/math.h"

#include <algorithm>
#include <array>

#include "mujoco/mujoco.h"

#ifdef MJLAB_USE_MKL
#include "mkl_vml.h"
#else
#include "sleef.h"
#endif

namespace mjlab {

// Trigonometry also sets physical orientations during mid-episode motion
// resampling. Preserve those inputs without emulating tensor CPU dispatch.
#ifdef MJLAB_USE_MKL
float Sin(float v) {
  float result;
  vmsSin(1, &v, &result, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE);
  return result;
}
float Cos(float v) {
  float result;
  vmsCos(1, &v, &result, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE);
  return result;
}
#else
float Sin(float v) { return Sleef_sinf1_u10purecfma(v); }
float Cos(float v) { return Sleef_cosf1_u10purecfma(v); }
#endif

Eigen3 SymmetricEigen(Mat3 matrix) {
  std::array<mjtNum, 9> input{}, vectors{};
  std::array<mjtNum, 3> values{};
  std::array<mjtNum, 4> quaternion{};
  std::copy(matrix.begin(), matrix.end(), input.begin());
  mju_eig3(values.data(), vectors.data(), quaternion.data(), input.data());
  std::array<int, 3> order{0, 1, 2};
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return values[a] < values[b]; });
  // Expose ascending eigenvalues and column-major eigenvectors.
  Eigen3 result{};
  for (int column = 0; column < 3; ++column) {
    const int source = order[column];
    result.values[column] = static_cast<float>(values[source]);
    for (int row = 0; row < 3; ++row) {
      result.vectors[column * 3 + row] =
          static_cast<float>(vectors[row * 3 + source]);
    }
  }
  return result;
}

}  // namespace mjlab
