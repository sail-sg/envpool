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

#ifndef ENVPOOL_JUMANJI_SOKOBAN_ENV_H_
#define ENVPOOL_JUMANJI_SOKOBAN_ENV_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace sokoban {

constexpr int kGridSize = 10;
constexpr int kCellCount = kGridSize * kGridSize;
constexpr int kNumBoxes = 4;
constexpr std::uint8_t kEmpty = 0;
constexpr std::uint8_t kWall = 1;
constexpr std::uint8_t kTarget = 2;
constexpr std::uint8_t kAgent = 3;
constexpr std::uint8_t kBox = 4;
constexpr std::uint8_t kTargetAgent = 5;
constexpr std::uint8_t kTargetBox = 6;
constexpr int kNoop = -1;
constexpr std::array<std::array<int, 2>, 4> kMoves = {
    {{{-1, 0}}, {{0, 1}}, {{1, 0}}, {{0, -1}}}};  // NOLINT

struct Level {
  std::array<std::uint8_t, kCellCount> fixed{};
  std::array<std::uint8_t, kCellCount> variable{};
  int agent_row{0};
  int agent_col{0};
};

struct Dataset {
  std::vector<Level> levels;
};

inline int Offset(int row, int col) { return row * kGridSize + col; }

inline bool InGrid(int row, int col) {
  return row >= 0 && row < kGridSize && col >= 0 && col < kGridSize;
}

inline Level ConvertLevel(const std::array<std::string, kGridSize>& ascii) {
  Level level;
  for (int row = 0; row < kGridSize; ++row) {
    for (int col = 0; col < kGridSize; ++col) {
      const int offset = Offset(row, col);
      switch (ascii[row][col]) {
        case '#':
          level.fixed[offset] = kWall;
          break;
        case '.':
          level.fixed[offset] = kTarget;
          break;
        case '@':
          level.variable[offset] = kAgent;
          level.agent_row = row;
          level.agent_col = col;
          break;
        case '$':
          level.variable[offset] = kBox;
          break;
        case ' ':
          break;
        default:
          throw std::runtime_error("invalid Sokoban level character");
      }
    }
  }
  return level;
}

inline Dataset FallbackDataset() {
  Dataset dataset;
  dataset.levels.push_back(ConvertLevel({
      "##########",
      "#       ##",
      "# ....   #",
      "# $$$$  ##",
      "# @    # #",
      "#   #   # ",
      "#        #",
      "##########",
      "##########",
      "##########",
  }));
  return dataset;
}

inline std::size_t ParseNpyHeaderLength(const std::vector<char>& bytes) {
  if (bytes.size() < 10 || bytes[0] != static_cast<char>(0x93) ||
      bytes[1] != 'N' || bytes[2] != 'U' || bytes[3] != 'M' ||
      bytes[4] != 'P' || bytes[5] != 'Y') {
    throw std::runtime_error("invalid Boxoban npy magic");
  }
  const unsigned char major = static_cast<unsigned char>(bytes[6]);
  if (major == 1) {
    return 10 + static_cast<unsigned char>(bytes[8]) +
           (static_cast<unsigned char>(bytes[9]) << 8);
  }
  if (major == 2 || major == 3) {
    if (bytes.size() < 12) {
      throw std::runtime_error("truncated Boxoban npy header");
    }
    std::size_t header_len = 0;
    for (int i = 0; i < 4; ++i) {
      header_len |=
          static_cast<std::size_t>(static_cast<unsigned char>(bytes[8 + i]))
          << (8 * i);
    }
    return 12 + header_len;
  }
  throw std::runtime_error("unsupported Boxoban npy version");
}

inline Dataset LoadNpyDataset(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return FallbackDataset();
  }
  std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                          std::istreambuf_iterator<char>());
  const std::size_t data_offset = ParseNpyHeaderLength(bytes);
  if (bytes.size() < data_offset ||
      (bytes.size() - data_offset) % (kCellCount * 2) != 0) {
    throw std::runtime_error("unexpected Boxoban npy data size");
  }
  const std::size_t level_count =
      (bytes.size() - data_offset) / (kCellCount * 2);
  Dataset dataset;
  dataset.levels.reserve(level_count);
  for (std::size_t level_id = 0; level_id < level_count; ++level_id) {
    Level level;
    const std::size_t level_offset =
        data_offset + level_id * static_cast<std::size_t>(kCellCount) * 2;
    for (int cell = 0; cell < kCellCount; ++cell) {
      level.fixed[cell] =
          static_cast<std::uint8_t>(bytes[level_offset + cell * 2]);
      level.variable[cell] =
          static_cast<std::uint8_t>(bytes[level_offset + cell * 2 + 1]);
      if (level.variable[cell] == kAgent) {
        level.agent_row = cell / kGridSize;
        level.agent_col = cell % kGridSize;
      }
    }
    dataset.levels.push_back(level);
  }
  if (dataset.levels.empty()) {
    throw std::runtime_error("empty Boxoban dataset");
  }
  return dataset;
}

inline std::shared_ptr<const Dataset> GetDataset(const std::string& base_path) {
  return std::make_shared<Dataset>(LoadNpyDataset(
      base_path + "/jumanji/assets/boxoban/unfiltered-test.npy"));
}

inline int CountTargets(const Level& level) {
  int count = 0;
  for (int i = 0; i < kCellCount; ++i) {
    if (level.fixed[i] == kTarget && level.variable[i] == kBox) {
      ++count;
    }
  }
  return count;
}

inline std::uint8_t CombinedCell(const Level& level, int offset) {
  const bool target = level.fixed[offset] == kTarget;
  if (target && level.variable[offset] == kAgent) {
    return kTargetAgent;
  }
  if (target && level.variable[offset] == kBox) {
    return kTargetBox;
  }
  return std::max(level.fixed[offset], level.variable[offset]);
}

}  // namespace sokoban

class SokobanEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "sokoban_level_index"_.Bind(-1),
        "sokoban_render_fixed_grid_replay"_.Bind(std::string("")),
        "sokoban_render_variable_grid_replay"_.Bind(std::string("")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict(
        "obs:grid"_.Bind(Spec<std::uint8_t>({10, 10, 2}, {0, 4})),
        "obs:step_count"_.Bind(Spec<int>({}, {0, 120})),
        "info:prop_correct_boxes"_.Bind(Spec<float>({}, {0.0f, 1.0f})),
        "info:solved"_.Bind(Spec<bool>({}, {false, true})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 3})));
  }
};

using SokobanEnvSpec = EnvSpec<SokobanEnvFns>;

class SokobanEnv : public Env<SokobanEnvSpec>, public RenderableEnv {
 protected:
  std::shared_ptr<const sokoban::Dataset> dataset_;
  sokoban::Level level_;
  int max_episode_steps_;
  int elapsed_step_{0};
  int configured_level_index_;
  bool done_{true};
  std::uniform_int_distribution<int> level_dist_;

 public:
  using Spec = SokobanEnvSpec;
  using Action = typename Env<SokobanEnvSpec>::Action;

  SokobanEnv(const Spec& spec, int env_id)
      : Env<SokobanEnvSpec>(spec, env_id),
        dataset_(sokoban::GetDataset(spec.config["base_path"_])),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        configured_level_index_(spec.config["sokoban_level_index"_]),
        level_dist_(0, static_cast<int>(dataset_->levels.size()) - 1) {}

  bool IsDone() override { return done_; }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, {36, 43, 49}, rgb);
    for (int row = 0; row < sokoban::kGridSize; ++row) {
      for (int col = 0; col < sokoban::kGridSize; ++col) {
        const std::uint8_t cell =
            sokoban::CombinedCell(level_, sokoban::Offset(row, col));
        const int left = col * width / sokoban::kGridSize;
        const int right = (col + 1) * width / sokoban::kGridSize;
        const int top = row * height / sokoban::kGridSize;
        const int bottom = (row + 1) * height / sokoban::kGridSize;
        const int cx = (left + right) / 2;
        const int cy = (top + bottom) / 2;
        if (cell == sokoban::kWall) {
          render::FillRect(width, height, left + 1, top + 1, right - 1,
                           bottom - 1, {130, 54, 24}, rgb);
          render::StrokeRect(width, height, left + 1, top + 1, right - 1,
                             bottom - 1, {210, 110, 55}, rgb);
          continue;
        }
        render::FillRect(width, height, left + 1, top + 1, right - 1,
                         bottom - 1, {232, 232, 232}, rgb);
        if (cell == sokoban::kTarget || cell == sokoban::kTargetAgent ||
            cell == sokoban::kTargetBox) {
          render::StrokeCircle(width, height, cx, cy,
                               std::max(3, (right - left) / 5), {230, 40, 40},
                               rgb);
        }
        if (cell == sokoban::kBox || cell == sokoban::kTargetBox) {
          render::FillRect(width, height, left + 5, top + 5, right - 5,
                           bottom - 5,
                           cell == sokoban::kTargetBox ? render::Palette(2)
                                                       : render::Palette(1),
                           rgb);
          render::StrokeRect(width, height, left + 5, top + 5, right - 5,
                             bottom - 5, {120, 60, 20}, rgb);
        }
        if (cell == sokoban::kAgent || cell == sokoban::kTargetAgent) {
          render::FillCircle(width, height, cx, cy,
                             std::max(3, (right - left) / 4), {20, 180, 50},
                             rgb);
          render::FillCircle(width, height, cx - (right - left) / 10,
                             cy - (bottom - top) / 12, 1, render::kBlack, rgb);
          render::FillCircle(width, height, cx + (right - left) / 10,
                             cy - (bottom - top) / 12, 1, render::kBlack, rgb);
        }
      }
    }
  }

  void Reset() override {
    elapsed_step_ = 0;
    done_ = false;
    int level_index = configured_level_index_;
    if (level_index < 0) {
      level_index = level_dist_(gen_);
    }
    level_index %= static_cast<int>(dataset_->levels.size());
    level_ = dataset_->levels[level_index];
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    int action_id = std::clamp(static_cast<int>(action["action"_]), 0, 3);
    const int previous_targets = sokoban::CountTargets(level_);
    action_id = DetectNoop(action_id);
    if (action_id != sokoban::kNoop) {
      MoveAgent(action_id);
    }
    ++elapsed_step_;
    const int next_targets = sokoban::CountTargets(level_);
    const bool solved = next_targets == sokoban::kNumBoxes;
    done_ = solved || elapsed_step_ >= max_episode_steps_;
    const float reward = static_cast<float>(next_targets - previous_targets) +
                         (solved ? 10.0f : 0.0f) - 0.1f;
    WriteState(reward);
  }

 private:
  int DetectNoop(int action_id) const {
    const int row = level_.agent_row + sokoban::kMoves[action_id][0];
    const int col = level_.agent_col + sokoban::kMoves[action_id][1];
    if (!sokoban::InGrid(row, col) ||
        level_.fixed[sokoban::Offset(row, col)] == sokoban::kWall) {
      return sokoban::kNoop;
    }
    if (level_.variable[sokoban::Offset(row, col)] != sokoban::kBox) {
      return action_id;
    }
    const int box_row = row + sokoban::kMoves[action_id][0];
    const int box_col = col + sokoban::kMoves[action_id][1];
    if (!sokoban::InGrid(box_row, box_col)) {
      return sokoban::kNoop;
    }
    const int box_offset = sokoban::Offset(box_row, box_col);
    if (level_.variable[box_offset] == sokoban::kBox ||
        level_.fixed[box_offset] == sokoban::kWall) {
      return sokoban::kNoop;
    }
    return action_id;
  }

  void MoveAgent(int action_id) {
    const int next_row = level_.agent_row + sokoban::kMoves[action_id][0];
    const int next_col = level_.agent_col + sokoban::kMoves[action_id][1];
    const int current = sokoban::Offset(level_.agent_row, level_.agent_col);
    const int next = sokoban::Offset(next_row, next_col);
    const bool pushes_box = level_.variable[next] == sokoban::kBox;
    level_.variable[current] = sokoban::kEmpty;
    if (pushes_box) {
      const int box_row = next_row + sokoban::kMoves[action_id][0];
      const int box_col = next_col + sokoban::kMoves[action_id][1];
      level_.variable[sokoban::Offset(box_row, box_col)] = sokoban::kBox;
    }
    level_.variable[next] = sokoban::kAgent;
    level_.agent_row = next_row;
    level_.agent_col = next_col;
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int row = 0; row < sokoban::kGridSize; ++row) {
      for (int col = 0; col < sokoban::kGridSize; ++col) {
        const int offset = sokoban::Offset(row, col);
        state["obs:grid"_](row, col, 0) = level_.variable[offset];
        state["obs:grid"_](row, col, 1) = level_.fixed[offset];
      }
    }
    const int targets = sokoban::CountTargets(level_);
    state["obs:step_count"_] = elapsed_step_;
    state["info:prop_correct_boxes"_] =
        static_cast<float>(targets) / sokoban::kNumBoxes;
    state["info:solved"_] = targets == sokoban::kNumBoxes;
    state["reward"_] = reward;
  }
};

using SokobanEnvPool = AsyncEnvPool<SokobanEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_SOKOBAN_ENV_H_
