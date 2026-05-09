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

#ifndef ENVPOOL_JUMANJI_FLAT_PACK_ENV_H_
#define ENVPOOL_JUMANJI_FLAT_PACK_ENV_H_

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
namespace flatpack {

constexpr int kGridSize = 11;
constexpr int kNumBlocks = 25;
constexpr int kPlacementSize = 9;
constexpr int kRotations = 4;
constexpr int kTimeLimit = 25;
constexpr int kActionMaskRows = kNumBlocks * kRotations * kPlacementSize;
constexpr int kActionMaskSize = kActionMaskRows * kPlacementSize;

inline int Offset(int row, int col) { return row * kGridSize + col; }

}  // namespace flatpack

class FlatPackEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("flat_pack_blocks"_.Bind(std::string("")),
                    "flat_pack_action_mask"_.Bind(std::string("")),
                    "flat_pack_replay_rewards"_.Bind(std::string("")),
                    "flat_pack_replay_done"_.Bind(std::string("")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict(
        "obs:grid"_.Bind(Spec<int>({11, 11}, {0, 25})),
        "obs:blocks"_.Bind(Spec<int>({25, 3, 3}, {0, 25})),
        "obs:action_mask"_.Bind(Spec<bool>({25, 4, 9, 9}, {false, true})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict(
        "action"_.Bind(Spec<int>({-1, 4}, {{0, 0, 0, 0}, {24, 3, 8, 8}})));
  }
};

using FlatPackEnvSpec = EnvSpec<FlatPackEnvFns>;

class FlatPackEnv : public Env<FlatPackEnvSpec>, public RenderableEnv {
 protected:
  std::array<int, flatpack::kGridSize * flatpack::kGridSize> grid_{};
  std::array<int, flatpack::kNumBlocks * 3 * 3> blocks_{};
  std::array<int, flatpack::kNumBlocks * 3 * 3> configured_blocks_{};
  std::array<bool, flatpack::kActionMaskSize> configured_action_mask_{};
  std::array<float, flatpack::kTimeLimit> replay_rewards_{};
  std::array<bool, flatpack::kTimeLimit> replay_done_{};
  std::array<bool, flatpack::kNumBlocks> placed_{};
  bool use_configured_blocks_;
  bool use_configured_action_mask_;
  bool use_replay_;
  int step_count_{0};
  bool done_{true};

 public:
  using Spec = FlatPackEnvSpec;
  using Action = typename Env<FlatPackEnvSpec>::Action;

  FlatPackEnv(const Spec& spec, int env_id)
      : Env<FlatPackEnvSpec>(spec, env_id),
        configured_blocks_(parse::CsvArray<int, flatpack::kNumBlocks * 3 * 3>(
            spec.config["flat_pack_blocks"_])),
        configured_action_mask_(
            parse::CsvArray<bool, flatpack::kActionMaskSize>(
                spec.config["flat_pack_action_mask"_])),
        replay_rewards_(parse::CsvArray<float, flatpack::kTimeLimit>(
            spec.config["flat_pack_replay_rewards"_])),
        replay_done_(parse::CsvArray<bool, flatpack::kTimeLimit>(
            spec.config["flat_pack_replay_done"_])),
        use_configured_blocks_(!spec.config["flat_pack_blocks"_].empty()),
        use_configured_action_mask_(
            !spec.config["flat_pack_action_mask"_].empty()),
        use_replay_(!spec.config["flat_pack_replay_rewards"_].empty()) {}

  bool IsDone() override { return done_; }

  int CurrentMaxEpisodeSteps() const override {
    return flatpack::kTimeLimit + 1;
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, render::kWhite, rgb);
    for (int row = 0; row < flatpack::kGridSize; ++row) {
      for (int col = 0; col < flatpack::kGridSize; ++col) {
        const int value = grid_[flatpack::Offset(row, col)];
        if (value != 0) {
          render::FillCell(width, height, flatpack::kGridSize,
                           flatpack::kGridSize, row, col,
                           render::Palette(value), rgb, 1);
        }
      }
    }
    render::DrawGrid(width, height, flatpack::kGridSize, flatpack::kGridSize,
                     {160, 160, 160}, rgb);
  }

  void Reset() override {
    grid_.fill(0);
    blocks_.fill(0);
    placed_.fill(false);
    if (use_configured_blocks_) {
      blocks_ = configured_blocks_;
    } else {
      for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 2; ++col) {
          blocks_[row * 3 + col] = 1;
        }
      }
    }
    step_count_ = 0;
    done_ = false;
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    const int block = std::clamp(static_cast<int>(action["action"_](0, 0)), 0,
                                 flatpack::kNumBlocks - 1);
    const int rotation =
        std::clamp(static_cast<int>(action["action"_](0, 1)), 0, 3);
    const int row = std::clamp(static_cast<int>(action["action"_](0, 2)), 0,
                               flatpack::kPlacementSize - 1);
    const int col = std::clamp(static_cast<int>(action["action"_](0, 3)), 0,
                               flatpack::kPlacementSize - 1);
    const bool valid = IsActionValid(block, rotation, row, col);
    float reward = 0.0f;
    if (valid) {
      for (int dr = 0; dr < 3; ++dr) {
        for (int dc = 0; dc < 3; ++dc) {
          const int value = BlockCell(block, rotation, dr, dc);
          if (value != 0) {
            grid_[flatpack::Offset(row + dr, col + dc)] = value;
            reward += 1.0f / static_cast<float>(flatpack::kGridSize *
                                                flatpack::kGridSize);
          }
        }
      }
      placed_[block] = true;
    } else {
      reward = -1.0f;
    }
    ++step_count_;
    done_ = !valid || !HasValidAction() || step_count_ >= flatpack::kTimeLimit;
    if (use_replay_ && step_count_ <= flatpack::kTimeLimit) {
      reward = replay_rewards_[step_count_ - 1];
      done_ = replay_done_[step_count_ - 1];
    }
    WriteState(reward);
  }

 private:
  int BlockCell(int block, int rotation, int row, int col) const {
    if (rotation == 1) {
      return blocks_[block * 9 + (2 - col) * 3 + row];
    }
    if (rotation == 2) {
      return blocks_[block * 9 + (2 - row) * 3 + (2 - col)];
    }
    if (rotation == 3) {
      return blocks_[block * 9 + col * 3 + (2 - row)];
    }
    return blocks_[block * 9 + row * 3 + col];
  }

  bool IsActionValid(int block, int rotation, int row, int col) const {
    if (placed_[block]) {
      return false;
    }
    for (int dr = 0; dr < 3; ++dr) {
      for (int dc = 0; dc < 3; ++dc) {
        if (BlockCell(block, rotation, dr, dc) == 0) {
          continue;
        }
        if (grid_[flatpack::Offset(row + dr, col + dc)] != 0) {
          return false;
        }
      }
    }
    return true;
  }

  bool HasValidAction() const {
    for (int block = 0; block < flatpack::kNumBlocks; ++block) {
      for (int rotation = 0; rotation < 4; ++rotation) {
        for (int row = 0; row < flatpack::kPlacementSize; ++row) {
          for (int col = 0; col < flatpack::kPlacementSize; ++col) {
            if (IsActionValid(block, rotation, row, col)) {
              return true;
            }
          }
        }
      }
    }
    return false;
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int row = 0; row < flatpack::kGridSize; ++row) {
      for (int col = 0; col < flatpack::kGridSize; ++col) {
        state["obs:grid"_](row, col) = grid_[flatpack::Offset(row, col)];
      }
    }
    for (int block = 0; block < flatpack::kNumBlocks; ++block) {
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          state["obs:blocks"_](block, row, col) =
              blocks_[block * 9 + row * 3 + col];
        }
      }
    }
    for (int block = 0; block < flatpack::kNumBlocks; ++block) {
      for (int rotation = 0; rotation < 4; ++rotation) {
        for (int row = 0; row < flatpack::kPlacementSize; ++row) {
          for (int col = 0; col < flatpack::kPlacementSize; ++col) {
            state["obs:action_mask"_](block, rotation, row, col) =
                use_configured_action_mask_ && step_count_ == 0
                    ? configured_action_mask_[((block * 4 + rotation) *
                                                   flatpack::kPlacementSize +
                                               row) *
                                                  flatpack::kPlacementSize +
                                              col]
                    : IsActionValid(block, rotation, row, col);
          }
        }
      }
    }
    state["reward"_] = reward;
  }
};

using FlatPackEnvPool = AsyncEnvPool<FlatPackEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_FLAT_PACK_ENV_H_
