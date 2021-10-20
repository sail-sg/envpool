#ifndef ENVPOOL_CORE_ARRAY_H_
#define ENVPOOL_CORE_ARRAY_H_

#include <glog/logging.h>

#include <cstring>
#include <memory>

#include "envpool/core/spec.h"

class Array {
  ShapeSpec spec_;
  std::shared_ptr<char> ptr_;

 public:
  Array() {}
  template <class Deleter>
  Array(const ShapeSpec &spec, char *data, Deleter &&deleter)
      : spec_(spec), ptr_(data, std::forward<Deleter>(deleter)) {}
  Array(const ShapeSpec &spec, char *data)
      : Array(spec, data, [](char *p) {}) {}
  Array(const ShapeSpec &spec, const std::shared_ptr<char> &ptr)
      : spec_(spec), ptr_(ptr) {}
  explicit Array(const ShapeSpec &spec)
      : Array(spec, new char[spec.nbytes()](), [](char *p) { delete[] p; }) {}
  inline Array operator[](int index) const {
    DCHECK_GT(spec_.shape.size(), (std::size_t)0)
        << " can't be indexed, because it is a scalar";
    ShapeSpec spec = spec_.Unbatch();
    int offset = index * spec.nbytes();
    return Array(spec, ptr_.get() + offset);
  }
  template <typename... ArgTypes>
  inline Array operator()(ArgTypes... Idx) const {
    constexpr int argsize = sizeof...(ArgTypes);
    std::vector<int> indices = {Idx...};
    DCHECK_GE(spec_.shape.size(), argsize);
    int offset = 0;
    int shape_tail_prod = 1;
    for (int i = spec_.shape.size() - 1; i != 0; i--) {
      if (i <= argsize - 1) {
        offset += indices[i] * shape_tail_prod * spec_.element_size;
      }
      shape_tail_prod *= spec_.shape[i];
    }
    ShapeSpec new_spec(
        spec_.element_size,
        std::vector<int>(spec_.shape.begin() + argsize, spec_.shape.end()));
    return Array(new_spec, ptr_.get() + offset);
  }
  Array Slice(int start, int end) const {
    DCHECK_GT(spec_.shape.size(), (std::size_t)0);
    ShapeSpec spec = spec_.Slice(start, end);
    int offset = start * spec.nbytes(1);
    return Array(spec, ptr_.get() + offset);
  }
  void Assign(const Array &value) const {
    DCHECK_EQ(spec_.element_size, value.spec_.element_size)
        << " element size doesn't match";
    DCHECK_EQ(spec_.size(), value.spec_.size()) << " size doesn't match";
    std::memcpy(ptr_.get(), value.ptr_.get(), value.spec_.nbytes());
  }
  template <typename T>
  void operator=(const T &value) const {
    DCHECK_EQ(spec_.element_size, (int)sizeof(T))
        << " element size doesn't match";
    DCHECK_EQ(spec_.nbytes(), spec_.element_size)
        << " assigning scalar to non-scalar array";
    *reinterpret_cast<T *>(ptr_.get()) = value;
  }
  template <typename T>
  void operator=(const std::pair<const T *, int> &raw) const {
    DCHECK_EQ(raw.second * sizeof(T), spec_.nbytes())
        << " assignment size mismatch";
    std::memcpy(ptr_.get(), raw.first, raw.second * sizeof(T));
  }
  template <typename T>
  operator T() const {
    DCHECK_EQ(spec_.element_size, (int)sizeof(T))
        << " there could be a type mismatch";
    DCHECK_EQ(spec_.element_size, spec_.nbytes())
        << " Array with a shape can't be used as a scalar";
    return *reinterpret_cast<T *>(ptr_.get());
  }
  inline const std::vector<int> &Shape() const { return spec_.shape; }
  inline void *data() const { return ptr_.get(); }
  inline const ShapeSpec &Spec() const { return spec_; }
  Array Truncate(int end) const { return Array(spec_.Slice(0, end), ptr_); }
};

#endif  // ENVPOOL_CORE_ARRAY_H_
