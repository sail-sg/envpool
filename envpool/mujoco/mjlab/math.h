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

namespace mjlab {

// Quaternion conventions follow MJLab 1.6.0 and its Isaac Lab math helpers.
// See third_party/mjlab/NOTICE for attribution.

using Vec3 = std::array<float, 3>;
using Quat = std::array<float, 4>;
using Mat3 = std::array<float, 9>;

inline float Sin(float v) { return std::sin(v); }
inline float Cos(float v) { return std::cos(v); }
inline float Exp(float v) { return std::exp(v); }
inline float Log(float v) { return std::log(v); }
inline float Acos(float v) { return std::acos(v); }
inline float Sqrt(float v) { return std::sqrt(v); }

struct Eigen3 {
  Vec3 values;
  Mat3 vectors;  // Column-major eigenvectors, with ascending eigenvalues.
};

Eigen3 SymmetricEigen(Mat3 matrix);
inline float Log1p(float v) { return std::log1p(v); }
inline float Atan2(float a, float b) { return std::atan2(a, b); }

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

template <typename Range, typename Transform>
float Sum(const Range& values, Transform transform) {
  // Double accumulation limits rounding error without CPU-specific ordering.
  double result = 0;
  for (const auto& value : values) {
    result += transform(value);
  }
  return static_cast<float>(result);
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
  return std::sqrt(SquaredNorm(values));
}

inline Vec3 Cross(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
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
  return {a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
          a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
          a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
          a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]};
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

inline float Yaw(const Quat& q) {
  return Atan2(2.0F * (q[0] * q[3] + q[1] * q[2]),
               1.0F - 2.0F * (q[2] * q[2] + q[3] * q[3]));
}
inline Quat YawQuat(const Quat& q) {
  const float yaw = Yaw(q) / 2.0F;
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
  float Uniform(float low, float high) { return Unit() * (high - low) + low; }
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
