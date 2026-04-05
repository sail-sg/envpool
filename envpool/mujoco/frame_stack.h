/*
 * Copyright 2026 Garena Online Private Limited
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

#ifndef ENVPOOL_MUJOCO_FRAME_STACK_H_
#define ENVPOOL_MUJOCO_FRAME_STACK_H_

#include <mujoco.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "envpool/core/array.h"
#include "envpool/core/spec.h"

namespace envpool::mujoco {

template <typename D>
Spec<D> StackSpec(const Spec<D>& spec, int frame_stack) {
  if (frame_stack < 1) {
    throw std::invalid_argument("frame_stack must be greater than 0");
  }
  if (frame_stack == 1) {
    return spec;
  }
  std::vector<int> stacked_shape = {frame_stack};
  stacked_shape.insert(stacked_shape.end(), spec.shape.begin(),
                       spec.shape.end());
  Spec<D> stacked(stacked_shape);
  stacked.is_discrete = spec.is_discrete;
  stacked.bounds = spec.bounds;
  const auto& [minimum, maximum] = spec.elementwise_bounds;
  if (!minimum.empty() || !maximum.empty()) {
    std::vector<D> stacked_minimum;
    std::vector<D> stacked_maximum;
    stacked_minimum.reserve(minimum.size() * frame_stack);
    stacked_maximum.reserve(maximum.size() * frame_stack);
    for (int i = 0; i < frame_stack; ++i) {
      stacked_minimum.insert(stacked_minimum.end(), minimum.begin(),
                             minimum.end());
      stacked_maximum.insert(stacked_maximum.end(), maximum.begin(),
                             maximum.end());
    }
    stacked.elementwise_bounds =
        std::make_tuple(std::move(stacked_minimum), std::move(stacked_maximum));
  }
  return stacked;
}

class FrameStackBuffer {
 private:
  struct KeyBuffer {
    std::vector<mjtNum> stacked;
    std::vector<mjtNum> scratch;
  };

  int frame_stack_;
  std::unordered_map<std::string, KeyBuffer> buffers_;

  KeyBuffer& GetBuffer(std::string_view key) {
    return buffers_[std::string(key)];
  }

 public:
  explicit FrameStackBuffer(int frame_stack) : frame_stack_(frame_stack) {
    if (frame_stack_ < 1) {
      throw std::invalid_argument("frame_stack must be greater than 0");
    }
  }

  mjtNum* Prepare(std::string_view key, Array* target) {
    if (frame_stack_ == 1) {
      return static_cast<mjtNum*>(target->Data());
    }
    DCHECK_EQ(target->Shape(0), static_cast<std::size_t>(frame_stack_));
    DCHECK_EQ(target->size % static_cast<std::size_t>(frame_stack_),
              static_cast<std::size_t>(0));
    auto& buffer = GetBuffer(key);
    buffer.scratch.resize(target->size /
                          static_cast<std::size_t>(frame_stack_));
    return buffer.scratch.data();
  }

  void Commit(std::string_view key, Array* target, bool reset) {
    if (frame_stack_ == 1) {
      return;
    }
    auto& buffer = GetBuffer(key);
    std::size_t stacked_size = target->size;
    std::size_t frame_size =
        stacked_size / static_cast<std::size_t>(frame_stack_);
    if (buffer.stacked.size() != stacked_size) {
      buffer.stacked.resize(stacked_size);
      reset = true;
    }
    if (reset) {
      for (int i = 0; i < frame_stack_; ++i) {
        std::memcpy(buffer.stacked.data() + i * frame_size,
                    buffer.scratch.data(), frame_size * sizeof(mjtNum));
      }
    } else {
      std::memmove(buffer.stacked.data(), buffer.stacked.data() + frame_size,
                   (stacked_size - frame_size) * sizeof(mjtNum));
      std::memcpy(buffer.stacked.data() + stacked_size - frame_size,
                  buffer.scratch.data(), frame_size * sizeof(mjtNum));
    }
    target->Assign(buffer.stacked.data(), stacked_size);
  }

  void Assign(std::string_view key, Array* target, const mjtNum* data,
              std::size_t size, bool reset) {
    mjtNum* scratch = Prepare(key, target);
    std::memcpy(scratch, data, size * sizeof(mjtNum));
    Commit(key, target, reset);
  }

  void AssignScalar(std::string_view key, Array* target, mjtNum value,
                    bool reset) {
    mjtNum* scratch = Prepare(key, target);
    scratch[0] = value;
    Commit(key, target, reset);
  }
};

}  // namespace envpool::mujoco

#endif  // ENVPOOL_MUJOCO_FRAME_STACK_H_
