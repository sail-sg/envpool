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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "envpool/core/array.h"
#include "envpool/core/dict.h"
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

template <typename D>
class TypedFrameStackBuffer {
 private:
  struct KeyBuffer {
    std::vector<D> stacked;
    std::vector<D> scratch;
  };

  int frame_stack_;
  std::unordered_map<std::string, KeyBuffer> buffers_;

  KeyBuffer& GetBuffer(std::string_view key) {
    return buffers_[std::string(key)];
  }

 public:
  explicit TypedFrameStackBuffer(int frame_stack) : frame_stack_(frame_stack) {
    if (frame_stack_ < 1) {
      throw std::invalid_argument("frame_stack must be greater than 0");
    }
  }

  D* Prepare(std::string_view key, Array* target) {
    if (frame_stack_ == 1) {
      return static_cast<D*>(target->Data());
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
                    buffer.scratch.data(), frame_size * sizeof(D));
      }
    } else {
      std::memmove(buffer.stacked.data(), buffer.stacked.data() + frame_size,
                   (stacked_size - frame_size) * sizeof(D));
      std::memcpy(buffer.stacked.data() + stacked_size - frame_size,
                  buffer.scratch.data(), frame_size * sizeof(D));
    }
    target->Assign(buffer.stacked.data(), stacked_size);
  }

  void Assign(std::string_view key, Array* target, const D* data,
              std::size_t size, bool reset) {
    D* scratch = Prepare(key, target);
    std::memcpy(scratch, data, size * sizeof(D));
    Commit(key, target, reset);
  }

  void AssignScalar(std::string_view key, Array* target, D value, bool reset) {
    D* scratch = Prepare(key, target);
    scratch[0] = value;
    Commit(key, target, reset);
  }
};

using FrameStackBuffer = TypedFrameStackBuffer<mjtNum>;

template <typename Config>
Spec<uint8_t> PixelObservationSpec(const Config& conf) {
  return Spec<uint8_t>(
      {3 * conf["frame_stack"_], conf["render_height"_], conf["render_width"_]},
      {static_cast<uint8_t>(0), static_cast<uint8_t>(255)});
}

template <typename Key>
inline constexpr bool IsObservationKey() {
  constexpr auto kKey = Key::kStrView;
  return kKey == "obs" ||
         (kKey.size() > 4 && kKey[0] == 'o' && kKey[1] == 'b' &&
          kKey[2] == 's' && kKey[3] == ':');
}

template <std::size_t I = 0, typename DictT>
auto NonObservationStateSpec(const DictT& spec) {
  if constexpr (I == DictT::kSize) {
    return MakeDict();
  } else {
    using Key = std::tuple_element_t<I, typename DictT::Keys>;
    auto tail = NonObservationStateSpec<I + 1>(spec);
    if constexpr (IsObservationKey<Key>()) {
      return tail;
    } else {
      const auto& values = static_cast<const typename DictT::Values&>(spec);
      auto value = std::get<I>(values);
      return ConcatDict(MakeDict(Key().Bind(std::move(value))), tail);
    }
  }
}

template <typename BaseEnvFns>
class PixelObservationEnvFns : public BaseEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return ConcatDict(
        BaseEnvFns::DefaultConfig(),
        MakeDict("render_width"_.Bind(84), "render_height"_.Bind(84),
                 "render_camera_id"_.Bind(-1)));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return ConcatDict(
        NonObservationStateSpec(BaseEnvFns::StateSpec(conf)),
        MakeDict("obs:pixels"_.Bind(PixelObservationSpec(conf))));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return BaseEnvFns::ActionSpec(conf);
  }
};

template <bool kFromPixels, typename Config>
int RenderWidthOrDefault(const Config& conf) {
  if constexpr (kFromPixels) {
    return conf["render_width"_];
  } else {
    return 84;
  }
}

template <bool kFromPixels, typename Config>
int RenderHeightOrDefault(const Config& conf) {
  if constexpr (kFromPixels) {
    return conf["render_height"_];
  } else {
    return 84;
  }
}

template <bool kFromPixels, typename Config>
int RenderCameraIdOrDefault(const Config& conf) {
  if constexpr (kFromPixels) {
    return conf["render_camera_id"_];
  } else {
    return -1;
  }
}

class PixelFrameStackBuffer {
 private:
  struct KeyBuffer {
    std::vector<uint8_t> stacked;
    std::vector<uint8_t> hwc_scratch;
    std::vector<uint8_t> chw_scratch;
  };

  int frame_stack_;
  std::unordered_map<std::string, KeyBuffer> buffers_;

  KeyBuffer& GetBuffer(std::string_view key) {
    return buffers_[std::string(key)];
  }

  static void HwcToChw(const std::vector<uint8_t>& hwc,
                       std::vector<uint8_t>* chw_out, int width, int height) {
    std::size_t plane_size = static_cast<std::size_t>(width) * height;
    chw_out->resize(3 * plane_size);
    const uint8_t* src = hwc.data();
    uint8_t* dst = chw_out->data();
    for (int channel = 0; channel < 3; ++channel) {
      uint8_t* channel_dst = dst + channel * plane_size;
      for (std::size_t pixel = 0; pixel < plane_size; ++pixel) {
        channel_dst[pixel] = src[pixel * 3 + channel];
      }
    }
  }

 public:
  explicit PixelFrameStackBuffer(int frame_stack) : frame_stack_(frame_stack) {
    if (frame_stack_ < 1) {
      throw std::invalid_argument("frame_stack must be greater than 0");
    }
  }

  uint8_t* Prepare(std::string_view key, int width, int height) {
    auto& buffer = GetBuffer(key);
    buffer.hwc_scratch.resize(static_cast<std::size_t>(width) * height * 3);
    return buffer.hwc_scratch.data();
  }

  void Commit(std::string_view key, Array* target, int width, int height,
              bool reset) {
    auto& buffer = GetBuffer(key);
    const std::size_t frame_size = static_cast<std::size_t>(width) * height * 3;
    HwcToChw(buffer.hwc_scratch, &buffer.chw_scratch, width, height);
    if (frame_stack_ == 1) {
      DCHECK_EQ(target->size, frame_size);
      target->Assign(buffer.chw_scratch.data(), frame_size);
      return;
    }
    const std::size_t stacked_size =
        frame_size * static_cast<std::size_t>(frame_stack_);
    DCHECK_EQ(target->size, stacked_size);
    if (buffer.stacked.size() != stacked_size) {
      buffer.stacked.resize(stacked_size);
      reset = true;
    }
    if (reset) {
      for (int i = 0; i < frame_stack_; ++i) {
        std::memcpy(buffer.stacked.data() + i * frame_size,
                    buffer.chw_scratch.data(), frame_size);
      }
    } else {
      std::memmove(buffer.stacked.data(), buffer.stacked.data() + frame_size,
                   stacked_size - frame_size);
      std::memcpy(buffer.stacked.data() + stacked_size - frame_size,
                  buffer.chw_scratch.data(), frame_size);
    }
    target->Assign(buffer.stacked.data(), stacked_size);
  }
};

}  // namespace envpool::mujoco

#endif  // ENVPOOL_MUJOCO_FRAME_STACK_H_
