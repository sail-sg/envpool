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

#include <array>
#include <stdexcept>

#ifdef MJLAB_USE_MKL
#include "mkl_lapacke.h"
#include "mkl_vml.h"
#else
// The same single-precision LAPACK interface used by the pinned CPU oracle:
// Accelerate on macOS and the official OpenBLAS distribution on Linux ARM.
extern "C" void ssyevd_(char*, char*, int*, float*, int*, float*, float*, int*,
                        int*, int*, int*);
#endif

namespace mjlab {

#ifdef MJLAB_USE_MKL
namespace {
using Unary = void (*)(MKL_INT, const float*, float*, MKL_INT64);
float Vml(Unary function, float input) {
  float output;
  // Match ATen/cpu/vml.h explicitly, independent of a process's VML mode.
  function(1, &input, &output, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE);
  return output;
}
}  // namespace
float Sin(float v) { return Vml(vmsSin, v); }
float Cos(float v) { return Vml(vmsCos, v); }
float Exp(float v) { return Vml(vmsExp, v); }
float Log(float v) { return Vml(vmsLn, v); }
float Acos(float v) { return Vml(vmsAcos, v); }
float Sqrt(float v) { return Vml(vmsSqrt, v); }
#else
float Sin(float v) { return Sleef_sinf1_u10purecfma(v); }
float Cos(float v) { return Sleef_cosf1_u10purecfma(v); }
float Exp(float v) { return Sleef_expf1_u10purecfma(v); }
float Log(float v) { return Sleef_logf1_u10purecfma(v); }
float Acos(float v) { return Sleef_acosf1_u10purecfma(v); }
float Sqrt(float v) { return std::sqrt(v); }
#endif

Eigen3 SymmetricEigen(Mat3 matrix) {
  Eigen3 result{{}, matrix};
  // SSYEVD's documented workspace minima for n=3 are 37 floats and 18 ints.
  // This small solve takes the unblocked path independently of workspace size.
  std::array<float, 128> work{};
  std::array<int, 32> iwork{};
  int info;
#ifdef MJLAB_USE_MKL
  info =
      LAPACKE_ssyevd_work(LAPACK_COL_MAJOR, 'V', 'L', 3, result.vectors.data(),
                          3, result.values.data(), work.data(), work.size(),
                          iwork.data(), iwork.size());
#else
  char job = 'V';
  char triangle = 'L';
  int n = 3;
  int leading = 3;
  int nwork = work.size();
  int niwork = iwork.size();
  ssyevd_(&job, &triangle, &n, result.vectors.data(), &leading,
          result.values.data(), work.data(), &nwork, iwork.data(), &niwork,
          &info);
#endif
  if (info != 0) {
    throw std::runtime_error("MJLab terrain eigendecomposition failed");
  }
  return result;
}

}  // namespace mjlab
