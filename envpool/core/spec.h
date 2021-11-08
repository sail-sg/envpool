/*
 * Copyright 2021 Garena Online Private Limited
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

#ifndef ENVPOOL_CORE_SPEC_H_
#define ENVPOOL_CORE_SPEC_H_

#include <glog/logging.h>

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

std::size_t prod(const std::size_t* shape, std::size_t ndim) {
  return std::accumulate(shape, shape + ndim, static_cast<std::size_t>(1),
                         std::multiplies<std::size_t>());
}

class ShapeSpec {
 public:
  int element_size;
  std::vector<int> shape;
  ShapeSpec() = default;
  ShapeSpec(int element_size, const std::vector<int>& shape_vec)
      : element_size(element_size), shape(shape_vec) {}
  ShapeSpec(int element_size, std::vector<int>&& shape_vec)
      : element_size(element_size), shape(std::move(shape_vec)) {}
  ShapeSpec Batch(int batch_size) const {
    std::vector<int> new_shape = {batch_size};
    new_shape.insert(new_shape.end(), shape.begin(), shape.end());
    return ShapeSpec(element_size, std::move(new_shape));
  }
  std::vector<std::size_t> Shape() const {
    auto s = std::vector<std::size_t>(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i) {
      s[i] = shape[i];
    }
    return s;
  }
};

template <typename D>
class Spec : public ShapeSpec {
 public:
  using dtype = D;
  std::tuple<dtype, dtype> bounds = {std::numeric_limits<dtype>::min(),
                                     std::numeric_limits<dtype>::max()};
  explicit Spec(std::vector<int>&& shape)
      : ShapeSpec(sizeof(dtype), std::move(shape)) {}
  explicit Spec(const std::vector<int>& shape)
      : ShapeSpec(sizeof(dtype), shape) {}
  Spec(std::vector<int>&& shape, std::tuple<dtype, dtype>&& bounds)
      : ShapeSpec(sizeof(dtype), std::move(shape)), bounds(std::move(bounds)) {}
  Spec(const std::vector<int>& shape, const std::tuple<dtype, dtype>& bounds)
      : ShapeSpec(sizeof(dtype), shape), bounds(bounds) {}
  Spec Batch(int batch_size) const {
    std::vector<int> new_shape = {batch_size};
    new_shape.insert(new_shape.end(), shape.begin(), shape.end());
    return Spec(std::move(new_shape));
  }
};

#endif  // ENVPOOL_CORE_SPEC_H_
