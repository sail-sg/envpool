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
#include <cstddef>
#include <string>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/parse_utils.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace binpack {

constexpr int kNumEms = 40;
constexpr int kNumItems = 20;
constexpr int kActiveItems = 2;
constexpr int kTimeLimit = 20;
constexpr int kReplaySteps = 32;

}  // namespace binpack

class BinPackEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "bin_pack_item_x_len"_.Bind(std::string("")),
        "bin_pack_item_y_len"_.Bind(std::string("")),
        "bin_pack_item_z_len"_.Bind(std::string("")),
        "bin_pack_items_mask"_.Bind(std::string("")),
        "bin_pack_replay_ems_x1"_.Bind(std::string("")),
        "bin_pack_replay_ems_x2"_.Bind(std::string("")),
        "bin_pack_replay_ems_y1"_.Bind(std::string("")),
        "bin_pack_replay_ems_y2"_.Bind(std::string("")),
        "bin_pack_replay_ems_z1"_.Bind(std::string("")),
        "bin_pack_replay_ems_z2"_.Bind(std::string("")),
        "bin_pack_replay_ems_mask"_.Bind(std::string("")),
        "bin_pack_replay_items_mask"_.Bind(std::string("")),
        "bin_pack_replay_items_placed"_.Bind(std::string("")),
        "bin_pack_replay_action_mask"_.Bind(std::string("")),
        "bin_pack_replay_rewards"_.Bind(std::string("")),
        "bin_pack_replay_done"_.Bind(std::string("")),
        "bin_pack_render_container"_.Bind(std::string("")),
        "bin_pack_render_item_x_len"_.Bind(std::string("")),
        "bin_pack_render_item_y_len"_.Bind(std::string("")),
        "bin_pack_render_item_z_len"_.Bind(std::string("")),
        "bin_pack_render_items_location_replay"_.Bind(std::string("")));
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
  std::array<float, binpack::kNumEms> ems_x1_{};
  std::array<float, binpack::kNumEms> ems_x2_{};
  std::array<float, binpack::kNumEms> ems_y1_{};
  std::array<float, binpack::kNumEms> ems_y2_{};
  std::array<float, binpack::kNumEms> ems_z1_{};
  std::array<float, binpack::kNumEms> ems_z2_{};
  std::array<bool, binpack::kNumEms> ems_mask_{};
  std::array<float, binpack::kNumItems> item_x_{};
  std::array<float, binpack::kNumItems> item_y_{};
  std::array<float, binpack::kNumItems> item_z_{};
  std::array<float, binpack::kNumItems> configured_item_x_{};
  std::array<float, binpack::kNumItems> configured_item_y_{};
  std::array<float, binpack::kNumItems> configured_item_z_{};
  std::array<bool, binpack::kNumItems> configured_items_mask_{};
  std::array<float, binpack::kReplaySteps * binpack::kNumEms> replay_ems_x1_{};
  std::array<float, binpack::kReplaySteps * binpack::kNumEms> replay_ems_x2_{};
  std::array<float, binpack::kReplaySteps * binpack::kNumEms> replay_ems_y1_{};
  std::array<float, binpack::kReplaySteps * binpack::kNumEms> replay_ems_y2_{};
  std::array<float, binpack::kReplaySteps * binpack::kNumEms> replay_ems_z1_{};
  std::array<float, binpack::kReplaySteps * binpack::kNumEms> replay_ems_z2_{};
  std::array<bool, binpack::kReplaySteps * binpack::kNumEms> replay_ems_mask_{};
  std::array<bool, binpack::kReplaySteps * binpack::kNumItems>
      replay_items_mask_{};
  std::array<bool, binpack::kReplaySteps * binpack::kNumItems>
      replay_items_placed_{};
  std::array<bool,
             binpack::kReplaySteps * binpack::kNumEms * binpack::kNumItems>
      replay_action_mask_{};
  std::array<float, binpack::kReplaySteps> replay_rewards_{};
  std::array<bool, binpack::kReplaySteps> replay_done_{};
  std::array<bool, binpack::kNumItems> items_mask_{};
  std::array<bool, binpack::kNumItems> items_placed_{};
  bool use_configured_items_;
  bool use_replay_;
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
        replay_ems_x1_(
            parse::CsvArray<float, binpack::kReplaySteps * binpack::kNumEms>(
                spec.config["bin_pack_replay_ems_x1"_])),
        replay_ems_x2_(
            parse::CsvArray<float, binpack::kReplaySteps * binpack::kNumEms>(
                spec.config["bin_pack_replay_ems_x2"_])),
        replay_ems_y1_(
            parse::CsvArray<float, binpack::kReplaySteps * binpack::kNumEms>(
                spec.config["bin_pack_replay_ems_y1"_])),
        replay_ems_y2_(
            parse::CsvArray<float, binpack::kReplaySteps * binpack::kNumEms>(
                spec.config["bin_pack_replay_ems_y2"_])),
        replay_ems_z1_(
            parse::CsvArray<float, binpack::kReplaySteps * binpack::kNumEms>(
                spec.config["bin_pack_replay_ems_z1"_])),
        replay_ems_z2_(
            parse::CsvArray<float, binpack::kReplaySteps * binpack::kNumEms>(
                spec.config["bin_pack_replay_ems_z2"_])),
        replay_ems_mask_(
            parse::CsvArray<bool, binpack::kReplaySteps * binpack::kNumEms>(
                spec.config["bin_pack_replay_ems_mask"_])),
        replay_items_mask_(
            parse::CsvArray<bool, binpack::kReplaySteps * binpack::kNumItems>(
                spec.config["bin_pack_replay_items_mask"_])),
        replay_items_placed_(
            parse::CsvArray<bool, binpack::kReplaySteps * binpack::kNumItems>(
                spec.config["bin_pack_replay_items_placed"_])),
        replay_action_mask_(
            parse::CsvArray<bool, binpack::kReplaySteps * binpack::kNumEms *
                                      binpack::kNumItems>(
                spec.config["bin_pack_replay_action_mask"_])),
        replay_rewards_(parse::CsvArray<float, binpack::kReplaySteps>(
            spec.config["bin_pack_replay_rewards"_])),
        replay_done_(parse::CsvArray<bool, binpack::kReplaySteps>(
            spec.config["bin_pack_replay_done"_])),
        use_configured_items_(!spec.config["bin_pack_item_x_len"_].empty()),
        use_replay_(!spec.config["bin_pack_replay_ems_x1"_].empty()) {}

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
    const int cx = width / 2;
    const int cy = height / 2;
    const int box_w = width * 42 / 100;
    const int box_h = height * 38 / 100;
    const int depth_x = width * 14 / 100;
    const int depth_y = height * 12 / 100;
    const int x0 = cx - box_w / 2;
    const int y0 = cy - box_h / 2 + depth_y / 2;
    const int x1 = x0 + box_w;
    const int y1 = y0 + box_h;
    const int x2 = x0 + depth_x;
    const int y2 = y0 - depth_y;
    const int x3 = x1 + depth_x;
    const int y3 = y1 - depth_y;
    render::FillRect(width, height, x0 + 1, y0 + 1, x1, y1, {234, 252, 252},
                     rgb);
    render::DrawLine(width, height, x0, y0, x1, y0, {100, 160, 160}, rgb);
    render::DrawLine(width, height, x1, y0, x1, y1, {100, 160, 160}, rgb);
    render::DrawLine(width, height, x1, y1, x0, y1, {100, 160, 160}, rgb);
    render::DrawLine(width, height, x0, y1, x0, y0, {100, 160, 160}, rgb);
    render::DrawLine(width, height, x0, y0, x2, y2, {100, 160, 160}, rgb);
    render::DrawLine(width, height, x1, y0, x1 + depth_x, y0 - depth_y,
                     {100, 160, 160}, rgb);
    render::DrawLine(width, height, x1, y1, x3, y3, {100, 160, 160}, rgb);
    render::DrawLine(width, height, x2, y2, x1 + depth_x, y0 - depth_y,
                     {100, 160, 160}, rgb);
    render::DrawLine(width, height, x1 + depth_x, y0 - depth_y, x3, y3,
                     {100, 160, 160}, rgb);
    const int placed = static_cast<int>(
        std::count(items_placed_.begin(), items_placed_.end(), true));
    for (int item = 0; item < placed; ++item) {
      const int px0 = x0 + 4 + item * box_w / 4;
      const int px1 = px0 + box_w / 5;
      const int py0 = y1 - 4 - box_h / 4;
      render::FillRect(width, height, px0, py0, px1, y1 - 4,
                       render::Palette(item), rgb);
    }
  }

  void Reset() override {
    ems_x1_.fill(0.0f);
    ems_x2_.fill(0.0f);
    ems_y1_.fill(0.0f);
    ems_y2_.fill(0.0f);
    ems_z1_.fill(0.0f);
    ems_z2_.fill(0.0f);
    ems_mask_.fill(false);
    ems_x2_[0] = 1.0f;
    ems_y2_[0] = 1.0f;
    ems_z2_[0] = 1.0f;
    ems_mask_[0] = true;

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
      for (int item = 0; item < binpack::kActiveItems; ++item) {
        item_x_[item] = 0.5f;
        item_y_[item] = 0.5f;
        item_z_[item] = 0.5f;
        items_mask_[item] = true;
      }
    }
    step_count_ = 0;
    done_ = false;
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    if (use_replay_ && step_count_ < binpack::kReplaySteps) {
      ++step_count_;
      done_ = replay_done_[step_count_ - 1];
      WriteState(replay_rewards_[step_count_ - 1]);
      return;
    }
    const int ems = std::clamp(static_cast<int>(action["action"_](0, 0)), 0,
                               binpack::kNumEms - 1);
    const int item = std::clamp(static_cast<int>(action["action"_](0, 1)), 0,
                                binpack::kNumItems - 1);
    const bool valid = IsActionValid(ems, item);
    float reward = 0.0f;
    if (valid) {
      items_placed_[item] = true;
      items_mask_[item] = false;
      reward = item_x_[item] * item_y_[item] * item_z_[item];
      SplitEms(ems, item);
    } else {
      reward = -1.0f;
    }
    ++step_count_;
    done_ = !valid || !HasValidAction() || step_count_ >= binpack::kTimeLimit;
    WriteState(reward);
  }

 private:
  bool IsActionValid(int ems, int item) const {
    if (!ems_mask_[ems] || !items_mask_[item] || items_placed_[item]) {
      return false;
    }
    const float space_x = ems_x2_[ems] - ems_x1_[ems];
    const float space_y = ems_y2_[ems] - ems_y1_[ems];
    const float space_z = ems_z2_[ems] - ems_z1_[ems];
    constexpr float k_eps = 1e-6f;
    return item_x_[item] <= space_x + k_eps &&
           item_y_[item] <= space_y + k_eps && item_z_[item] <= space_z + k_eps;
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

  void WriteEms(std::array<float, binpack::kNumEms>* x1,
                std::array<float, binpack::kNumEms>* x2,
                std::array<float, binpack::kNumEms>* y1,
                std::array<float, binpack::kNumEms>* y2,
                std::array<float, binpack::kNumEms>* z1,
                std::array<float, binpack::kNumEms>* z2,
                std::array<bool, binpack::kNumEms>* mask, int index, float a_x1,
                float a_x2, float a_y1, float a_y2, float a_z1,
                float a_z2) const {
    if (index >= binpack::kNumEms || a_x1 >= a_x2 || a_y1 >= a_y2 ||
        a_z1 >= a_z2) {
      return;
    }
    (*x1)[index] = a_x1;
    (*x2)[index] = a_x2;
    (*y1)[index] = a_y1;
    (*y2)[index] = a_y2;
    (*z1)[index] = a_z1;
    (*z2)[index] = a_z2;
    (*mask)[index] = true;
  }

  void SplitEms(int ems, int item) {
    std::array<float, binpack::kNumEms> next_x1{};
    std::array<float, binpack::kNumEms> next_x2{};
    std::array<float, binpack::kNumEms> next_y1{};
    std::array<float, binpack::kNumEms> next_y2{};
    std::array<float, binpack::kNumEms> next_z1{};
    std::array<float, binpack::kNumEms> next_z2{};
    std::array<bool, binpack::kNumEms> next_mask{};
    int out = 0;
    const float old_x1 = ems_x1_[ems];
    const float old_x2 = ems_x2_[ems];
    const float old_y1 = ems_y1_[ems];
    const float old_y2 = ems_y2_[ems];
    const float old_z1 = ems_z1_[ems];
    const float old_z2 = ems_z2_[ems];
    const float item_x2 = old_x1 + item_x_[item];
    const float item_y2 = old_y1 + item_y_[item];
    const float item_z2 = old_z1 + item_z_[item];
    WriteEms(&next_x1, &next_x2, &next_y1, &next_y2, &next_z1, &next_z2,
             &next_mask, out++, old_x1, old_x2, old_y1, old_y2, item_z2,
             old_z2);
    WriteEms(&next_x1, &next_x2, &next_y1, &next_y2, &next_z1, &next_z2,
             &next_mask, out++, old_x1, old_x2, item_y2, old_y2, old_z1,
             old_z2);
    WriteEms(&next_x1, &next_x2, &next_y1, &next_y2, &next_z1, &next_z2,
             &next_mask, out++, item_x2, old_x2, old_y1, old_y2, old_z1,
             old_z2);
    for (int old = 0; old < binpack::kNumEms && out < binpack::kNumEms; ++old) {
      if (old == ems || !ems_mask_[old]) {
        continue;
      }
      WriteEms(&next_x1, &next_x2, &next_y1, &next_y2, &next_z1, &next_z2,
               &next_mask, out++, ems_x1_[old], ems_x2_[old], ems_y1_[old],
               ems_y2_[old], ems_z1_[old], ems_z2_[old]);
    }
    ems_x1_ = next_x1;
    ems_x2_ = next_x2;
    ems_y1_ = next_y1;
    ems_y2_ = next_y2;
    ems_z1_ = next_z1;
    ems_z2_ = next_z2;
    ems_mask_ = next_mask;
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int ems = 0; ems < binpack::kNumEms; ++ems) {
      const int replay = (step_count_ - 1) * binpack::kNumEms + ems;
      const bool use_replay_step = use_replay_ && step_count_ > 0 &&
                                   step_count_ <= binpack::kReplaySteps;
      state["obs:ems.x1"_][ems] =
          use_replay_step ? replay_ems_x1_[replay] : ems_x1_[ems];
      state["obs:ems.x2"_][ems] =
          use_replay_step ? replay_ems_x2_[replay] : ems_x2_[ems];
      state["obs:ems.y1"_][ems] =
          use_replay_step ? replay_ems_y1_[replay] : ems_y1_[ems];
      state["obs:ems.y2"_][ems] =
          use_replay_step ? replay_ems_y2_[replay] : ems_y2_[ems];
      state["obs:ems.z1"_][ems] =
          use_replay_step ? replay_ems_z1_[replay] : ems_z1_[ems];
      state["obs:ems.z2"_][ems] =
          use_replay_step ? replay_ems_z2_[replay] : ems_z2_[ems];
      state["obs:ems_mask"_][ems] =
          use_replay_step ? replay_ems_mask_[replay] : ems_mask_[ems];
    }
    for (int item = 0; item < binpack::kNumItems; ++item) {
      const int replay = (step_count_ - 1) * binpack::kNumItems + item;
      const bool use_replay_step = use_replay_ && step_count_ > 0 &&
                                   step_count_ <= binpack::kReplaySteps;
      state["obs:items.x_len"_][item] = item_x_[item];
      state["obs:items.y_len"_][item] = item_y_[item];
      state["obs:items.z_len"_][item] = item_z_[item];
      state["obs:items_mask"_][item] =
          use_replay_step ? replay_items_mask_[replay] : items_mask_[item];
      state["obs:items_placed"_][item] =
          use_replay_step ? replay_items_placed_[replay] : items_placed_[item];
    }
    for (int ems = 0; ems < binpack::kNumEms; ++ems) {
      for (int item = 0; item < binpack::kNumItems; ++item) {
        const int replay =
            ((step_count_ - 1) * binpack::kNumEms + ems) * binpack::kNumItems +
            item;
        state["obs:action_mask"_](ems, item) =
            use_replay_ && step_count_ > 0 &&
                    step_count_ <= binpack::kReplaySteps
                ? replay_action_mask_[replay]
                : IsActionValid(ems, item);
      }
    }
    state["reward"_] = reward;
  }
};

using BinPackEnvPool = AsyncEnvPool<BinPackEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_BIN_PACK_ENV_H_
