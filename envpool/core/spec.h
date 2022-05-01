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

std::size_t Prod(const std::size_t* shape, std::size_t ndim) {
  return std::accumulate(shape, shape + ndim, static_cast<std::size_t>(1),
                         std::multiplies<>());
}

class ShapeSpec {
 public:
  int element_size_;
  std::vector<int> shape_;
  ShapeSpec() = default;
  ShapeSpec(int element_size, std::vector<int> shape_vec)
      : element_size_(element_size), shape_(std::move(shape_vec)) {}
  [[nodiscard]] ShapeSpec Batch(int batch_size) const {
    std::vector<int> new_shape = {batch_size};
    new_shape.insert(new_shape.end(), shape_.begin(), shape_.end());
    return ShapeSpec(element_size_, std::move(new_shape));
  }
  [[nodiscard]] std::vector<std::size_t> Shape() const {
    auto s = std::vector<std::size_t>(shape_.size());
    for (std::size_t i = 0; i < shape_.size(); ++i) {
      s[i] = shape_[i];
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
  std::tuple<std::vector<dtype>, std::vector<dtype>> elementwise_bounds;
  explicit Spec(std::vector<int>&& shape)
      : ShapeSpec(sizeof(dtype), std::move(shape)) {}
  explicit Spec(const std::vector<int>& shape)
      : ShapeSpec(sizeof(dtype), shape) {}

  /* init with constant bounds */
  Spec(std::vector<int>&& shape, std::tuple<dtype, dtype>&& bounds)
      : ShapeSpec(sizeof(dtype), std::move(shape)), bounds(std::move(bounds)) {}
  Spec(const std::vector<int>& shape, const std::tuple<dtype, dtype>& bounds)
      : ShapeSpec(sizeof(dtype), shape), bounds(bounds) {}

  /* init with elementwise bounds */
  Spec(std::vector<int>&& shape,
       std::tuple<std::vector<dtype>, std::vector<dtype>>&& elementwise_bounds)
      : ShapeSpec(sizeof(dtype), std::move(shape)),
        elementwise_bounds(std::move(elementwise_bounds)) {}
  Spec(const std::vector<int>& shape,
       const std::tuple<std::vector<dtype>, std::vector<dtype>>&
           elementwise_bounds)
      : ShapeSpec(sizeof(dtype), shape),
        elementwise_bounds(elementwise_bounds) {}

  [[nodiscard]] Spec Batch(int batch_size) const {
    std::vector<int> new_shape = {batch_size};
    new_shape.insert(new_shape.end(), shape_.begin(), shape_.end());
    return Spec(std::move(new_shape));
  }
};

#endif  // ENVPOOL_CORE_SPEC_H_
