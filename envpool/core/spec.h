#ifndef ENVPOOL_CORE_SPEC_H_
#define ENVPOOL_CORE_SPEC_H_

#include <glog/logging.h>

#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

class ShapeSpec {
 public:
  int element_size;
  std::vector<int> shape;
  int size(int offset = 0) const {
    return std::accumulate(shape.begin() + offset, shape.end(), 1,
                           std::multiplies<int>());
  }
  int nbytes(int offset = 0) const { return size(offset) * element_size; }
  ShapeSpec() = default;
  ShapeSpec(int element_size, const std::vector<int> &shape_vec)
      : element_size(element_size), shape(shape_vec) {}

  ShapeSpec(int element_size, std::vector<int> &&shape_vec)
      : element_size(element_size), shape(std::move(shape_vec)) {}
  ShapeSpec Batch(int batch_size) const {
    std::vector<int> new_shape = {batch_size};
    new_shape.insert(new_shape.end(), shape.begin(), shape.end());
    return ShapeSpec(element_size, std::move(new_shape));
  }
  ShapeSpec Unbatch() const {
    DCHECK_GT(shape.size(), (std::size_t)0);
    return ShapeSpec(element_size,
                     std::vector<int>(shape.begin() + 1, shape.end()));
  }
  ShapeSpec Slice(int start, int end) const {
    DCHECK_GT(shape.size(), (std::size_t)0);
    DCHECK_GT(end, start);
    DCHECK_LE(end, shape[0]);
    DCHECK_GE(start, 0);
    std::vector<int> new_shape(shape.begin(), shape.end());
    new_shape[0] = end - start;
    return ShapeSpec(element_size, new_shape);
  }
};

template <typename D>
class Spec : public ShapeSpec {
 public:
  using dtype = D;
  explicit Spec(std::vector<int> &&shape)
      : ShapeSpec(sizeof(dtype), std::move(shape)) {}
  explicit Spec(const std::vector<int> &shape)
      : ShapeSpec(sizeof(dtype), shape) {}
  Spec Batch(int batch_size) const {
    std::vector<int> new_shape = {batch_size};
    new_shape.insert(new_shape.end(), shape.begin(), shape.end());
    return Spec(std::move(new_shape));
  }
  Spec Unbatch() const {
    DCHECK_GT(shape.size(), (std::size_t)0);
    return Spec(std::vector<int>(shape.begin() + 1, shape.end()));
  }
  Spec Slice(int start, int end) const {
    DCHECK_GT(shape.size(), (std::size_t)0);
    DCHECK_GT(end, start);
    DCHECK_LE(end, shape[0]);
    DCHECK_GE(start, 0);
    std::vector<int> new_shape(shape.begin(), shape.end());
    new_shape[0] = end - start;
    return Spec(new_shape);
  }
};

#endif  // ENVPOOL_CORE_SPEC_H_
