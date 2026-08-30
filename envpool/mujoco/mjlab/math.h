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

#ifndef ENVPOOL_MUJOCO_MJLAB_MATH_H_
#define ENVPOOL_MUJOCO_MJLAB_MATH_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

#include "sleef.h"

namespace mjlab {

// Tensor reductions follow PyTorch 2.9.0; quaternion expressions follow
// MJLab 1.6.0's Isaac Lab math helpers. See third_party/mjlab/NOTICE and the
// accompanying upstream license files for attribution.

using Vec3 = std::array<float, 3>;
using Quat = std::array<float, 4>;
using Mat3 = std::array<float, 9>;

// The pinned official CPU wheels use SLEEF on ARM and MKL VML on x64.
// The implementation links just the required native math, never Torch.
float Sin(float v);
float Cos(float v);
float Exp(float v);
float Log(float v);
float Acos(float v);
float Sqrt(float v);

struct Eigen3 {
  Vec3 values;
  Mat3 vectors;  // Column-major eigenvectors, with ascending eigenvalues.
};

Eigen3 SymmetricEigen(Mat3 matrix);
inline float Log1p(float v) { return Sleef_log1pf1_u10purecfma(v); }
// BinaryOpsKernel uses std::atan2 for its scalar tail. One independent
// environment is smaller than the vector loop, unlike the unary VML kernels.
inline float Atan2(float a, float b, bool vectorized = false) {
  return vectorized ? Sleef_atan2f1_u10purecfma(a, b) : std::atan2(a, b);
}

template <std::size_t N>
std::array<float, N> Read(const float* data) {
  std::array<float, N> result;
  std::copy_n(data, N, result.begin());
  return result;
}

template <std::size_t N>
std::array<float, N> operator+(std::array<float, N> a,
                               const std::array<float, N>& b) {
  for (std::size_t i = 0; i < N; ++i) {
    a[i] += b[i];
  }
  return a;
}

template <std::size_t N>
std::array<float, N> operator-(std::array<float, N> a,
                               const std::array<float, N>& b) {
  for (std::size_t i = 0; i < N; ++i) {
    a[i] -= b[i];
  }
  return a;
}

template <std::size_t N>
std::array<float, N> operator*(std::array<float, N> a, float b) {
  for (auto& v : a) {
    v *= b;
  }
  return a;
}

inline float Square(float value) { return value * value; }

inline int ReductionWidth() {
#if defined(__aarch64__) || defined(_M_ARM64)
  return 4;
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  static const int width = __builtin_cpu_supports("avx512f") &&
                                   __builtin_cpu_supports("avx512dq") &&
                                   __builtin_cpu_supports("avx512bw") &&
                                   __builtin_cpu_supports("avx512vl")
                               ? 16
                               : 8;
  return width;
#elif defined(_MSC_VER) && defined(_M_X64)
  static const int width = [] {
    std::array<int, 4> registers{};
    __cpuidex(registers.data(), 1, 0);
    if ((registers[2] & (1 << 27)) == 0 || (_xgetbv(0) & 0xe6) != 0xe6) {
      return 8;
    }
    __cpuidex(registers.data(), 7, 0);
    constexpr unsigned int flags =
        (1U << 16) | (1U << 17) | (1U << 30) | (1U << 31);
    return (static_cast<unsigned int>(registers[1]) & flags) == flags ? 16 : 8;
  }();
  return width;
#else
  return 8;
#endif
}

template <typename Range, typename Transform>
float Sum(const Range& values, Transform transform,
          int vector_width = std::min(ReductionWidth(), 8)) {
  // Match the pinned CPU SumKernel's four independent accumulators and
  // cascade, including its tail-before-lanes order. Reassociation changes
  // posture and action penalties even when physics is bitwise identical.
  const int size = values.size();
  // SumKernel deliberately disables AVX512 dispatch even on capable CPUs.
  // Other kernels (including atan2) retain their normal dispatch width.
  const int width = size >= vector_width ? vector_width : 1;
  const int vectors = size / width;
  const int groups = vectors / 4;
  int logarithm = 0;
  for (int value = groups - 1; value > 0; value >>= 1) {
    ++logarithm;
  }
  const int power = std::max(4, logarithm / 4);
  const int mask = (1 << power) - 1;
  std::array<std::array<std::array<float, 16>, 4>, 4> acc{};
  for (int i = 0; i < groups; ++i) {
    for (int k = 0; k < 4; ++k) {
      for (int lane = 0; lane < width; ++lane) {
        acc[0][k][lane] += transform(values[(i * 4 + k) * width + lane]);
      }
    }
    if (((i + 1) & mask) == 0) {
      for (int level = 1; level < 4; ++level) {
        for (int k = 0; k < 4; ++k) {
          for (int lane = 0; lane < width; ++lane) {
            acc[level][k][lane] += acc[level - 1][k][lane];
            acc[level - 1][k][lane] = 0;
          }
        }
        if (((i + 1) & (mask << (level * power))) != 0) {
          break;
        }
      }
    }
  }
  for (int level = 1; level < 4; ++level) {
    for (int k = 0; k < 4; ++k) {
      for (int lane = 0; lane < width; ++lane) {
        acc[0][k][lane] += acc[level][k][lane];
      }
    }
  }
  for (int i = groups * 4; i < vectors; ++i) {
    for (int lane = 0; lane < width; ++lane) {
      acc[0][0][lane] += transform(values[i * width + lane]);
    }
  }
  for (int k = 1; k < 4; ++k) {
    for (int lane = 0; lane < width; ++lane) {
      acc[0][0][lane] += acc[0][k][lane];
    }
  }
  float result = 0;
  for (int i = vectors * width; i < size; ++i) {
    result += transform(values[i]);
  }
  for (int lane = 0; lane < width; ++lane) {
    result += acc[0][0][lane];
  }
  return result;
}

template <typename Range>
float Sum(const Range& values) {
  return Sum(values, [](float v) { return v; });
}

template <typename Range>
float SquaredNorm(const Range& values) {
  return Sum(values, [](float v) { return v * v; });
}

template <std::size_t N>
float Norm(const std::array<float, N>& values) {
  static_assert(N >= 2 && N <= 4);
  // These tasks use 2/3-vectors and quaternions. The pinned Linux and macOS
  // kernels fuse the short scalar reduction; a four-element norm forms four
  // separate products. The official MSVC build does not contract either case.
  float result = 0;
  for (float value : values) {
#if defined(__linux__) || defined(__aarch64__)
    if constexpr (N < 4) {
      result = std::fma(value, value, result);
      continue;
    }
#endif
    result += Square(value);
  }
  return std::sqrt(result);
}

inline Vec3 Cross(const Vec3& a, const Vec3& b) {
#if defined(__linux__) || defined(__aarch64__)
  // PyTorch's Linux/macOS CrossKernel contracts the first product. Keep
  // this local: ordinary tensor multiply/add and Warp kernels do not fuse.
  return {std::fma(a[1], b[2], -a[2] * b[1]),
          std::fma(a[2], b[0], -a[0] * b[2]),
          std::fma(a[0], b[1], -a[1] * b[0])};
#else
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
#endif
}

inline Quat Conjugate(const Quat& q) { return {q[0], -q[1], -q[2], -q[3]}; }
inline Quat Inverse(const Quat& q) {
  auto result = Conjugate(q);
  const float divisor = std::max(SquaredNorm(q), 1.0e-9F);
  for (auto& value : result) {
    value /= divisor;
  }
  return result;
}
inline Vec3 Rotate(const Quat& q, const Vec3& v) {
  const Vec3 xyz{q[1], q[2], q[3]};
  const auto t = Cross(xyz, v) * 2.0F;
  return (v + t * q[0]) + Cross(xyz, t);
}
inline Vec3 RotateInverse(const Quat& q, const Vec3& v) {
  const Vec3 xyz{q[1], q[2], q[3]};
  const auto t = Cross(xyz, v) * 2.0F;
  return (v - t * q[0]) + Cross(xyz, t);
}

inline Quat Multiply(const Quat& a, const Quat& b) {
  // Preserve the expression order used by mjlab.utils.lab_api.math.quat_mul.
  const float ww = (a[3] + a[1]) * (b[1] + b[2]);
  const float yy = (a[0] - a[2]) * (b[0] + b[3]);
  const float zz = (a[0] + a[2]) * (b[0] - b[3]);
  const float xx = (ww + yy) + zz;
  const float qq = 0.5F * (xx + (a[3] - a[1]) * (b[1] - b[2]));
  return {(qq - ww) + (a[3] - a[2]) * (b[2] - b[3]),
          (qq - xx) + (a[1] + a[0]) * (b[1] + b[0]),
          (qq - yy) + (a[0] - a[1]) * (b[2] + b[3]),
          (qq - zz) + (a[3] + a[2]) * (b[0] - b[1])};
}

inline Quat Euler(float roll, float pitch, float yaw) {
  const float cy = Cos(yaw * 0.5F);
  const float sy = Sin(yaw * 0.5F);
  const float cr = Cos(roll * 0.5F);
  const float sr = Sin(roll * 0.5F);
  const float cp = Cos(pitch * 0.5F);
  const float sp = Sin(pitch * 0.5F);
  return {(cy * cr) * cp + (sy * sr) * sp, (cy * sr) * cp - (sy * cr) * sp,
          (cy * cr) * sp + (sy * sr) * cp, (sy * cr) * cp - (cy * sr) * sp};
}

inline float Yaw(const Quat& q, bool vectorized = false) {
  return Atan2(2.0F * (q[0] * q[3] + q[1] * q[2]),
               1.0F - 2.0F * (q[2] * q[2] + q[3] * q[3]), vectorized);
}
inline Quat YawQuat(const Quat& q, bool vectorized = false) {
  const float yaw = Yaw(q, vectorized) / 2.0F;
  Quat result{Cos(yaw), 0, 0, Sin(yaw)};
  const float divisor = std::max(Norm(result), 1.0e-9F);
  for (auto& value : result) {
    value /= divisor;
  }
  return result;
}

inline Mat3 Matrix(const Quat& q) {
  const float r = q[0];
  const float i = q[1];
  const float j = q[2];
  const float k = q[3];
  const float s = 2.0F / SquaredNorm(q);
  return {1.0F - s * (j * j + k * k), s * (i * j - k * r),
          s * (i * k + j * r),        s * (i * j + k * r),
          1.0F - s * (i * i + k * k), s * (j * k - i * r),
          s * (i * k - j * r),        s * (j * k + i * r),
          1.0F - s * (i * i + j * j)};
}

inline Quat Quaternion(const float* m) {
  const Quat diagonal{
      ((1.0F + m[0]) + m[4]) + m[8], ((1.0F + m[0]) - m[4]) - m[8],
      ((1.0F - m[0]) + m[4]) - m[8], ((1.0F - m[0]) - m[4]) + m[8]};
  Quat magnitude;
  for (int i = 0; i < 4; ++i) {
    magnitude[i] = Sqrt(std::max(diagonal[i], 0.0F));
  }
  const std::array<Quat, 4> candidates{
      {{Square(magnitude[0]), m[7] - m[5], m[2] - m[6], m[3] - m[1]},
       {m[7] - m[5], Square(magnitude[1]), m[3] + m[1], m[2] + m[6]},
       {m[2] - m[6], m[3] + m[1], Square(magnitude[2]), m[5] + m[7]},
       {m[3] - m[1], m[6] + m[2], m[7] + m[5], Square(magnitude[3])}}};
  const int best =
      std::max_element(magnitude.begin(), magnitude.end()) - magnitude.begin();
  Quat result = candidates[best];
  const float divisor = 2.0F * std::max(magnitude[best], 0.1F);
  for (auto& value : result) {
    value /= divisor;
  }
  return result;
}

class Random {
 public:
  explicit Random(uint32_t seed) : seed_(seed), engine_(seed) {}
  uint32_t Next() {
    ++draws_;
    return engine_();
  }
  float Unit() { return static_cast<float>(Next() & 0xffffffU) * 0x1p-24F; }
  float Uniform(float low, float high) {
    // The official Linux and macOS uniform_ kernels contract this operation.
    // MJLab's separate rand()*range+low helper remains unfused in Sample().
#if defined(__linux__) || defined(__aarch64__)
    return std::fma(Unit(), high - low, low);
#else
    return Unit() * (high - low) + low;
#endif
  }
  int Integer(int high) { return Next() % static_cast<uint32_t>(high); }
  double UnitDouble() {
    const uint64_t high = Next();
    const uint64_t low = Next();
    return static_cast<double>(((high << 32) | low) & ((1ULL << 53) - 1)) *
           0x1p-53;
  }
  uint32_t Seed() const { return seed_; }
  uint64_t Draws() const { return draws_; }

 private:
  uint32_t seed_;
  uint64_t draws_{0};
  std::mt19937 engine_;
};

}  // namespace mjlab

#endif  // ENVPOOL_MUJOCO_MJLAB_MATH_H_
