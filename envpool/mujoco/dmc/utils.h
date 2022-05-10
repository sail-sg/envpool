/*
 * Copyright 2022 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENVPOOL_MUJOCO_DMC_UTILS_H_
#define ENVPOOL_MUJOCO_DMC_UTILS_H_

#include <cmath>

namespace mujoco {

enum class SigmoidType {
  kGaussian,
  kHyperbolic,
  kLongTail,
  kReciprocal,
  kCosine,
  kLinear,
  kQuadratic,
  kTanhSquared,
};

double RewardTolerance(double x, double bound_min = 0.0, double bound_max = 0.0,
                       double margin = 0.0, double value_at_margin = 0.1,
                       SigmoidType sigmoid_type = SigmoidType::kGaussian) {
  if (bound_min <= x && x <= bound_max) {
    return 1.0;
  }
  if (margin <= 0.0) {
    return 0.0;
  }
  x = (x < bound_min ? bound_min - x : x - bound_max) / margin;
  if (sigmoid_type == SigmoidType::kGaussian) {
    // scale = np.sqrt(-2 * np.log(value_at_1))
    // return np.exp(-0.5 * (x*scale)**2)
    double scaled_x = std::sqrt(-2 * std::log(value_at_margin)) * x;
    return std::exp(-0.5 * scaled_x * scaled_x);
  } else if (sigmoid_type == SigmoidType::kHyperbolic) {
    // scale = np.arccosh(1/value_at_1)
    // return 1 / np.cosh(x*scale)
    double scaled_x = std::acosh(1 / value_at_margin) * x;
    return 1 / std::cosh(scaled_x);
  } else if (sigmoid_type == SigmoidType::kLongTail) {
    // scale = np.sqrt(1/value_at_1 - 1)
    // return 1 / ((x*scale)**2 + 1)
    double scaled_x = std::sqrt(1 / value_at_margin - 1) * x;
    return 1 / (scaled_x * scaled_x + 1);
  } else if (sigmoid_type == SigmoidType::kReciprocal) {
    // scale = 1/value_at_1 - 1
    // return 1 / (abs(x)*scale + 1)
    double scale = 1 / value_at_margin - 1;
    return 1 / (std::abs(x) * scale + 1);
  } else if (sigmoid_type == SigmoidType::kCosine) {
    // scale = np.arccos(2*value_at_1 - 1) / np.pi
    // scaled_x = x*scale
    // with warnings.catch_warnings():
    //   warnings.filterwarnings(
    //       action='ignore', message='invalid value encountered in cos')
    //   cos_pi_scaled_x = np.cos(np.pi*scaled_x)
    // return np.where(abs(scaled_x) < 1, (1 + cos_pi_scaled_x)/2, 0.0)
    const double pi = std::acos(-1);
    double scaled_x = std::acos(2 * value_at_margin - 1) / pi * x;
    return std::abs(scaled_x) < 1 ? (1 + std::cos(pi * scaled_x)) / 2 : 0.0;
  } else if (sigmoid_type == SigmoidType::kLinear) {
    // scale = 1-value_at_1
    // scaled_x = x*scale
    // return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)
    double scaled_x = (1 - value_at_margin) * x;
    return std::abs(scaled_x) < 1 ? 1 - scaled_x : 0.0;
  } else if (sigmoid_type == SigmoidType::kQuadratic) {
    // scale = np.sqrt(1-value_at_1)
    // scaled_x = x*scale
    // return np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)
    double scaled_x = std::sqrt(1 - value_at_margin) * x;
    return std::abs(scaled_x) < 1 ? 1 - scaled_x * scaled_x : 0.0;
  } else if (sigmoid_type == SigmoidType::: kTanhSquared) {
    // scale = np.arctanh(np.sqrt(1-value_at_1))
    // return 1 - np.tanh(x*scale)**2
    double scaled_x = std::atanh(std::sqrt(1 - value_at_margin)) * x;
    return 1 - std::tanh(scaled_x) * std::tanh(scaled_x);
  } else {
    throw std::runtime_error("Unknown sigmoid type for RewardTolerance.");
  }
}

}  // namespace mujoco

#endif  // ENVPOOL_MUJOCO_DMC_UTILS_H_
