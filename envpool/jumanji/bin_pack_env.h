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

#ifndef ENVPOOL_JUMANJI_BIN_PACK_ENV_H_
#define ENVPOOL_JUMANJI_BIN_PACK_ENV_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/parse_utils.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace binpack {

constexpr int kNumEms = 40;
constexpr int kNumItems = 20;
constexpr std::array<int, 3> kContainer{5870, 2330, 2200};
constexpr int kTimeLimit = 20;

}  // namespace binpack

class BinPackEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("bin_pack_item_x_len"_.Bind(std::string("")),
                    "bin_pack_item_y_len"_.Bind(std::string("")),
                    "bin_pack_item_z_len"_.Bind(std::string("")),
                    "bin_pack_items_mask"_.Bind(std::string("")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict(
        "obs:ems.x1"_.Bind(Spec<float>({40}, {0.0f, 1.0f})),
        "obs:ems.x2"_.Bind(Spec<float>({40}, {0.0f, 1.0f})),
        "obs:ems.y1"_.Bind(Spec<float>({40}, {0.0f, 1.0f})),
        "obs:ems.y2"_.Bind(Spec<float>({40}, {0.0f, 1.0f})),
        "obs:ems.z1"_.Bind(Spec<float>({40}, {0.0f, 1.0f})),
        "obs:ems.z2"_.Bind(Spec<float>({40}, {0.0f, 1.0f})),
        "obs:ems_mask"_.Bind(Spec<bool>({40}, {false, true})),
        "obs:items.x_len"_.Bind(Spec<float>({20}, {0.0f, 1.0f})),
        "obs:items.y_len"_.Bind(Spec<float>({20}, {0.0f, 1.0f})),
        "obs:items.z_len"_.Bind(Spec<float>({20}, {0.0f, 1.0f})),
        "obs:items_mask"_.Bind(Spec<bool>({20}, {false, true})),
        "obs:items_placed"_.Bind(Spec<bool>({20}, {false, true})),
        "obs:action_mask"_.Bind(Spec<bool>({40, 20}, {false, true})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1, 2}, {{0, 0}, {39, 19}})));
  }
};

using BinPackEnvSpec = EnvSpec<BinPackEnvFns>;

class BinPackEnv : public Env<BinPackEnvSpec>, public RenderableEnv {
 protected:
  // Upstream geometry uses integer millimeters, exact in float32. Normalize
  // only at the API boundary so touching boxes do not leave rounding slivers.
  using Box = std::array<float, 6>;
  std::array<Box, binpack::kNumEms> ems_{};
  std::array<int, binpack::kNumEms> ems_order_{};
  std::array<bool, binpack::kNumEms> ems_mask_{};
  std::array<float, binpack::kNumItems> item_x_{};
  std::array<float, binpack::kNumItems> item_y_{};
  std::array<float, binpack::kNumItems> item_z_{};
  std::array<float, binpack::kNumItems> configured_item_x_{};
  std::array<float, binpack::kNumItems> configured_item_y_{};
  std::array<float, binpack::kNumItems> configured_item_z_{};
  std::array<bool, binpack::kNumItems> configured_items_mask_{};
  std::array<bool, binpack::kNumItems> items_mask_{};
  std::array<bool, binpack::kNumItems> items_placed_{};
  std::array<std::array<float, 3>, binpack::kNumItems> item_origins_{};
  bool use_configured_items_;
  int step_count_{0};
  bool done_{true};

 public:
  using Spec = BinPackEnvSpec;
  using Action = typename Env<BinPackEnvSpec>::Action;

  BinPackEnv(const Spec& spec, int env_id)
      : Env<BinPackEnvSpec>(spec, env_id),
        configured_item_x_(parse::CsvArray<float, binpack::kNumItems>(
            spec.config["bin_pack_item_x_len"_])),
        configured_item_y_(parse::CsvArray<float, binpack::kNumItems>(
            spec.config["bin_pack_item_y_len"_])),
        configured_item_z_(parse::CsvArray<float, binpack::kNumItems>(
            spec.config["bin_pack_item_z_len"_])),
        configured_items_mask_(parse::CsvArray<bool, binpack::kNumItems>(
            spec.config["bin_pack_items_mask"_])),
        use_configured_items_(!spec.config["bin_pack_item_x_len"_].empty()) {
    for (int item = 0; item < binpack::kNumItems; ++item) {
      configured_item_x_[item] =
          std::round(configured_item_x_[item] * binpack::kContainer[0]);
      configured_item_y_[item] =
          std::round(configured_item_y_[item] * binpack::kContainer[1]);
      configured_item_z_[item] =
          std::round(configured_item_z_[item] * binpack::kContainer[2]);
    }
  }

  bool IsDone() override { return done_; }

  int CurrentMaxEpisodeSteps() const override {
    return binpack::kTimeLimit + 1;
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, render::kWhite, rgb);
    // An oblique wireframe preserves all three placement coordinates and
    // dimensions without a second scene representation or a 3-D dependency.
    const auto box = [&](const std::array<float, 3>& origin,
                         const std::array<float, 3>& size,
                         render::Color color) {
      std::array<std::pair<int, int>, 8> corners;
      for (int corner = 0; corner < 8; ++corner) {
        const float x = (origin[0] + ((corner & 1) ? size[0] : 0.0f)) /
                        binpack::kContainer[0];
        const float y = (origin[1] + ((corner & 2) ? size[1] : 0.0f)) /
                        binpack::kContainer[1];
        const float z = (origin[2] + ((corner & 4) ? size[2] : 0.0f)) /
                        binpack::kContainer[2];
        corners[corner] = {
            static_cast<int>(width * (0.1f + 0.6f * x + 0.2f * y)),
            static_cast<int>(height * (0.9f - 0.6f * z - 0.2f * y))};
      }
      for (int corner = 0; corner < 8; ++corner) {
        for (int axis = 1; axis <= 4; axis <<= 1) {
          if ((corner & axis) == 0) {
            const auto [x0, y0] = corners[corner];
            const auto [x1, y1] = corners[corner | axis];
            render::DrawLine(width, height, x0, y0, x1, y1, color, rgb);
          }
        }
      }
    };
    box({0, 0, 0},
        {binpack::kContainer[0], binpack::kContainer[1],
         binpack::kContainer[2]},
        {180, 180, 180});
    for (int item = 0; item < binpack::kNumItems; ++item) {
      if (items_placed_[item]) {
        box(item_origins_[item], {item_x_[item], item_y_[item], item_z_[item]},
            render::Palette(item));
      }
    }
  }

  void Reset() override {
    ems_.fill({});
    ems_[0] = {0, binpack::kContainer[0], 0, binpack::kContainer[1],
               0, binpack::kContainer[2]};
    ems_mask_.fill(false);
    ems_mask_[0] = true;
    std::iota(ems_order_.begin(), ems_order_.end(), 0);

    item_x_.fill(0.0f);
    item_y_.fill(0.0f);
    item_z_.fill(0.0f);
    items_mask_.fill(false);
    items_placed_.fill(false);
    if (use_configured_items_) {
      item_x_ = configured_item_x_;
      item_y_ = configured_item_y_;
      item_z_ = configured_item_z_;
      items_mask_ = configured_items_mask_;
    } else {
      GenerateItems();
    }
    step_count_ = 0;
    done_ = false;
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    const int ems = std::clamp(static_cast<int>(action["action"_](0, 0)), 0,
                               binpack::kNumEms - 1);
    const int item = std::clamp(static_cast<int>(action["action"_](0, 1)), 0,
                                binpack::kNumItems - 1);
    const bool valid = IsActionValid(ems, item);
    const int slot = ems_order_[ems];
    float reward = 0.0f;
    if (valid) {
      items_placed_[item] = true;
      item_origins_[item] = {ems_[slot][0], ems_[slot][2], ems_[slot][4]};
      reward = (item_x_[item] * item_y_[item] * item_z_[item]) /
               (static_cast<float>(binpack::kContainer[0]) *
                binpack::kContainer[1] * binpack::kContainer[2]);
      SplitEms(slot, item);
    }
    ++step_count_;
    done_ = !valid || !HasValidAction() || step_count_ >= binpack::kTimeLimit;
    WriteState(reward);
  }

 private:
  void GenerateItems() {
    // Match the pinned generator: split a 20-foot container into solvable
    // integer-sized items, choosing an axis uniformly and an item by length.
    constexpr auto container = binpack::kContainer;
    std::array<std::array<int, 3>, binpack::kNumItems> sizes{};
    sizes.fill(container);
    items_mask_[0] = true;
    while (std::count(items_mask_.begin(), items_mask_.end(), true) <
           binpack::kNumItems - 1) {
      const int axis = std::uniform_int_distribution<int>(0, 2)(gen_);
      std::array<double, binpack::kNumItems> weights{};
      for (int item = 0; item < binpack::kNumItems; ++item) {
        weights[item] = items_mask_[item] ? sizes[item][axis] : 0;
      }
      const int item =
          std::discrete_distribution<int>(weights.begin(), weights.end())(gen_);
      const int length = sizes[item][axis];
      int cut = 0;
      if (std::uniform_real_distribution<float>(0, 1)(gen_) < 0.7f) {
        const int padding = static_cast<int>(0.3f * length);
        cut = std::uniform_int_distribution<int>(padding,
                                                 length - padding - 1)(gen_);
      } else {
        if (std::uniform_int_distribution<int>(1, 2)(gen_) == 1) {
          continue;
        }
        cut = length / 2;
      }
      const auto free =
          std::find(items_mask_.begin(), items_mask_.end(), false) -
          items_mask_.begin();
      sizes[free] = sizes[item];
      sizes[item][axis] = cut;
      sizes[free][axis] = length - cut;
      items_mask_[item] = cut > 0;
      items_mask_[free] = length - cut > 0;
    }
    for (int item = 0; item < binpack::kNumItems; ++item) {
      item_x_[item] = static_cast<float>(sizes[item][0]);
      item_y_[item] = static_cast<float>(sizes[item][1]);
      item_z_[item] = static_cast<float>(sizes[item][2]);
    }
  }

  bool IsActionValid(int visible_ems, int item) const {
    const int ems = ems_order_[visible_ems];
    if (!ems_mask_[ems] || !items_mask_[item] || items_placed_[item]) {
      return false;
    }
    return item_x_[item] <= ems_[ems][1] - ems_[ems][0] &&
           item_y_[item] <= ems_[ems][3] - ems_[ems][2] &&
           item_z_[item] <= ems_[ems][5] - ems_[ems][4];
  }

  bool HasValidAction() const {
    for (int ems = 0; ems < binpack::kNumEms; ++ems) {
      for (int item = 0; item < binpack::kNumItems; ++item) {
        if (IsActionValid(ems, item)) {
          return true;
        }
      }
    }
    return false;
  }

  void SplitEms(int selected, int item) {
    const Box placed{ems_[selected][0], ems_[selected][0] + item_x_[item],
                     ems_[selected][2], ems_[selected][2] + item_y_[item],
                     ems_[selected][4], ems_[selected][4] + item_z_[item]};
    const auto includes = [](const Box& outer, const Box& inner) {
      for (int axis = 0; axis < 3; ++axis) {
        if (inner[2 * axis] < outer[2 * axis] ||
            inner[2 * axis + 1] > outer[2 * axis + 1]) {
          return false;
        }
      }
      return true;
    };
    const auto old = ems_;
    std::array<bool, binpack::kNumEms> intersected{};
    for (int ems = 0; ems < binpack::kNumEms; ++ems) {
      bool intersects = ems_mask_[ems];
      for (int axis = 0; axis < 3; ++axis) {
        intersects = intersects && old[ems][2 * axis] < placed[2 * axis + 1] &&
                     placed[2 * axis] < old[ems][2 * axis + 1];
      }
      intersected[ems] = intersects;
      if (intersects) {
        ems_mask_[ems] = false;
      }
    }
    // Match the public action ordering: direction first, then original slot.
    std::vector<Box> candidates;
    for (int direction = 0; direction < 6; ++direction) {
      const int axis = direction / 2;
      for (int ems = 0; ems < binpack::kNumEms; ++ems) {
        if (!intersected[ems]) {
          continue;
        }
        auto remaining = old[ems];
        if (direction % 2 == 0) {
          remaining[2 * axis + 1] =
              std::min(remaining[2 * axis + 1], placed[2 * axis]);
        } else {
          remaining[2 * axis] =
              std::max(remaining[2 * axis], placed[2 * axis + 1]);
        }
        if (remaining[2 * axis] < remaining[2 * axis + 1]) {
          candidates.push_back(remaining);
        }
      }
    }
    for (const auto& candidate : candidates) {
      const bool contained = std::any_of(
          candidates.begin(), candidates.end(), [&](const Box& other) {
            return other != candidate && includes(other, candidate);
          });
      if (contained) {
        continue;
      }
      bool already_present = false;
      for (int ems = 0; ems < binpack::kNumEms; ++ems) {
        already_present |= ems_mask_[ems] && includes(ems_[ems], candidate);
      }
      if (already_present) {
        continue;
      }
      const int first_free = static_cast<int>(
          std::distance(ems_mask_.begin(),
                        std::find(ems_mask_.begin(), ems_mask_.end(), false)));
      // The pinned upstream argmin selects slot zero if the buffer is full.
      const int slot = first_free == binpack::kNumEms ? 0 : first_free;
      ems_[slot] = candidate;
      ems_mask_[slot] = true;
    }
    const auto volume = [&](int slot) {
      const auto& box = ems_[slot];
      return ems_mask_[slot]
                 ? (box[1] - box[0]) * (box[3] - box[2]) * (box[5] - box[4])
                 : 0.0f;
    };
    std::iota(ems_order_.begin(), ems_order_.end(), 0);
    std::stable_sort(ems_order_.begin(), ems_order_.end(),
                     [&](int a, int b) { return volume(a) > volume(b); });
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int ems = 0; ems < binpack::kNumEms; ++ems) {
      const int slot = ems_order_[ems];
      state["obs:ems.x1"_][ems] = ems_[slot][0] / binpack::kContainer[0];
      state["obs:ems.x2"_][ems] = ems_[slot][1] / binpack::kContainer[0];
      state["obs:ems.y1"_][ems] = ems_[slot][2] / binpack::kContainer[1];
      state["obs:ems.y2"_][ems] = ems_[slot][3] / binpack::kContainer[1];
      state["obs:ems.z1"_][ems] = ems_[slot][4] / binpack::kContainer[2];
      state["obs:ems.z2"_][ems] = ems_[slot][5] / binpack::kContainer[2];
      state["obs:ems_mask"_][ems] = ems_mask_[slot];
    }
    for (int item = 0; item < binpack::kNumItems; ++item) {
      state["obs:items.x_len"_][item] = item_x_[item] / binpack::kContainer[0];
      state["obs:items.y_len"_][item] = item_y_[item] / binpack::kContainer[1];
      state["obs:items.z_len"_][item] = item_z_[item] / binpack::kContainer[2];
      state["obs:items_mask"_][item] = items_mask_[item];
      state["obs:items_placed"_][item] = items_placed_[item];
    }
    for (int ems = 0; ems < binpack::kNumEms; ++ems) {
      for (int item = 0; item < binpack::kNumItems; ++item) {
        state["obs:action_mask"_](ems, item) = IsActionValid(ems, item);
      }
    }
    state["reward"_] = reward;
  }
};

using BinPackEnvPool = AsyncEnvPool<BinPackEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_BIN_PACK_ENV_H_
