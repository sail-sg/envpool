// Copyright 2026 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "envpool/minigrid/impl/minigrid_env.h"

namespace minigrid {
namespace {

using WFCBitmap = std::vector<std::vector<std::uint8_t>>;
using WFCPattern = std::vector<std::uint8_t>;
using WFCMask = std::uint64_t;

constexpr int kWFCPadding = 1;
constexpr int kWFCMaxAttempts = 1000;
constexpr std::array<Pos, 4> kWFCDirections = {
    Pos{0, -1}, Pos{1, 0}, Pos{0, 1},
    Pos{-1, 0}};  // NOLINT(whitespace/indent_namespace)

struct WFCPreset {
  std::vector<std::string> rows;
  int pattern_width{2};
  bool output_periodic{false};
};

struct WFCModel {
  int pattern_width{2};
  bool output_periodic{false};
  std::vector<WFCPattern> patterns;
  std::vector<double> weights;
  std::array<std::vector<WFCMask>, 4> adjacency;
  WFCMask full_mask{0};
};

const std::map<std::string, WFCPreset>& WFCPresets() {
  static const std::map<std::string, WFCPreset> presets = {
      {"MazeSimple",
       {{
            "....",
            ".###",
            ".#.#",
            ".###",
        },
        2,
        false}},
      {"DungeonMazeScaled",
       {{
            "........",
            "........",
            "..######",
            "..######",
            "..##..##",
            "..##..##",
            "..######",
            "..######",
        },
        2,
        true}},
      {"RoomsFabric",
       {{
            "..#..",
            ".....",
            "#####",
            ".....",
            "..#..",
        },
        3,
        false}},
      {"ObstaclesBlackdots",
       {{
            "...",
            ".#.",
            "...",
        },
        2,
        false}},
      {"ObstaclesAngular",
       {{
            "...........",
            ".....#####.",
            ".....#####.",
            ".....#####.",
            ".....#####.",
            ".#########.",
            ".#########.",
            ".#########.",
            ".#########.",
            ".#########.",
            "...........",
        },
        3,
        true}},
      {"ObstaclesHogs3",
       {{
            ".................",
            ".......#....#....",
            "..#...###..###...",
            ".###...#....#....",
            "..#..............",
            ".........#.......",
            "....#...###...#..",
            "...###...#...###.",
            "....#.........#..",
            ".................",
            ".......#....#....",
            "..#...###..###...",
            ".###...#....#....",
            "..#..#...#....#..",
            "....###.###..###.",
            ".....#...#....#..",
            ".................",
        },
        3,
        true}},
  };
  return presets;
}

const WFCPreset& GetWFCPreset(const std::string& name) {
  const auto& presets = WFCPresets();
  const auto it = presets.find(name);
  CHECK(it != presets.end()) << "Unknown WFC preset: " << name;
  return it->second;
}

WFCBitmap ParseBitmap(const std::vector<std::string>& rows) {
  CHECK(!rows.empty());
  const int height = static_cast<int>(rows.size());
  const int width = static_cast<int>(rows[0].size());
  CHECK_GT(width, 0);
  WFCBitmap bitmap(height, std::vector<std::uint8_t>(width, 0));
  for (int y = 0; y < height; ++y) {
    CHECK_EQ(static_cast<int>(rows[y].size()), width);
    for (int x = 0; x < width; ++x) {
      bitmap[y][x] = rows[y][x] == '#' ? 1 : 0;
    }
  }
  return bitmap;
}

WFCBitmap ReflectLeftRight(const WFCBitmap& bitmap) {
  const int height = static_cast<int>(bitmap.size());
  const int width = static_cast<int>(bitmap[0].size());
  WFCBitmap out(height, std::vector<std::uint8_t>(width, 0));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      out[y][x] = bitmap[y][width - 1 - x];
    }
  }
  return out;
}

WFCBitmap RotateClockwise(const WFCBitmap& bitmap) {
  const int height = static_cast<int>(bitmap.size());
  const int width = static_cast<int>(bitmap[0].size());
  WFCBitmap out(width, std::vector<std::uint8_t>(height, 0));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      out[x][height - 1 - y] = bitmap[y][x];
    }
  }
  return out;
}

void AddPatternsFromBitmap(const WFCBitmap& bitmap, int pattern_width,
                           std::map<WFCPattern, int>* pattern_counts) {
  const int height = static_cast<int>(bitmap.size());
  const int width = static_cast<int>(bitmap[0].size());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      WFCPattern pattern(pattern_width * pattern_width, 0);
      for (int py = 0; py < pattern_width; ++py) {
        for (int px = 0; px < pattern_width; ++px) {
          const int src_y = (y + py) % height;
          const int src_x = (x + px) % width;
          pattern[py * pattern_width + px] = bitmap[src_y][src_x];
        }
      }
      ++(*pattern_counts)[pattern];
    }
  }
}

bool IsCompatible(const WFCPattern& lhs, const WFCPattern& rhs,
                  int pattern_width, const Pos& dir) {
  const int left = std::max(0, dir.first);
  const int right = std::min(pattern_width, pattern_width + dir.first);
  const int top = std::max(0, dir.second);
  const int bottom = std::min(pattern_width, pattern_width + dir.second);
  for (int y = top; y < bottom; ++y) {
    for (int x = left; x < right; ++x) {
      if (lhs[y * pattern_width + x] !=
          rhs[(y - dir.second) * pattern_width + (x - dir.first)]) {
        return false;
      }
    }
  }
  return true;
}

WFCModel BuildWFCModel(const WFCPreset& preset) {
  std::map<WFCPattern, int> pattern_counts;
  WFCBitmap bitmap = ParseBitmap(preset.rows);
  AddPatternsFromBitmap(bitmap, preset.pattern_width, &pattern_counts);
  bitmap = ReflectLeftRight(bitmap);
  AddPatternsFromBitmap(bitmap, preset.pattern_width, &pattern_counts);
  bitmap = RotateClockwise(bitmap);
  AddPatternsFromBitmap(bitmap, preset.pattern_width, &pattern_counts);
  bitmap = ReflectLeftRight(bitmap);
  AddPatternsFromBitmap(bitmap, preset.pattern_width, &pattern_counts);
  bitmap = RotateClockwise(bitmap);
  AddPatternsFromBitmap(bitmap, preset.pattern_width, &pattern_counts);
  bitmap = ReflectLeftRight(bitmap);
  AddPatternsFromBitmap(bitmap, preset.pattern_width, &pattern_counts);
  bitmap = RotateClockwise(bitmap);
  AddPatternsFromBitmap(bitmap, preset.pattern_width, &pattern_counts);
  bitmap = ReflectLeftRight(bitmap);
  AddPatternsFromBitmap(bitmap, preset.pattern_width, &pattern_counts);

  WFCModel model;
  model.pattern_width = preset.pattern_width;
  model.output_periodic = preset.output_periodic;
  model.patterns.reserve(pattern_counts.size());
  model.weights.reserve(pattern_counts.size());
  for (const auto& [pattern, weight] : pattern_counts) {
    model.patterns.push_back(pattern);
    model.weights.push_back(static_cast<double>(weight));
  }
  CHECK_GT(static_cast<int>(model.patterns.size()), 0);
  CHECK_LE(static_cast<int>(model.patterns.size()), 64);

  const int num_patterns = static_cast<int>(model.patterns.size());
  model.full_mask = num_patterns == 64 ? std::numeric_limits<WFCMask>::max()
                                       : ((WFCMask{1} << num_patterns) - 1);
  for (int dir_idx = 0; dir_idx < 4; ++dir_idx) {
    model.adjacency[dir_idx].assign(num_patterns, 0);
    for (int lhs = 0; lhs < num_patterns; ++lhs) {
      WFCMask mask = 0;
      for (int rhs = 0; rhs < num_patterns; ++rhs) {
        if (IsCompatible(model.patterns[lhs], model.patterns[rhs],
                         model.pattern_width, kWFCDirections[dir_idx])) {
          mask |= WFCMask{1} << rhs;
        }
      }
      model.adjacency[dir_idx][lhs] = mask;
    }
  }
  return model;
}

int CountAllowedPatterns(WFCMask mask) {
#ifdef _MSC_VER
  return static_cast<int>(__popcnt64(mask));
#else
  return __builtin_popcountll(mask);
#endif
}

int LowestAllowedPattern(WFCMask mask) {
  CHECK_NE(mask, WFCMask{0});
#ifdef _MSC_VER
  unsigned long pattern_idx = 0;  // NOLINT(runtime/int)
  const unsigned char found = _BitScanForward64(&pattern_idx, mask);
  CHECK_NE(found, 0);
  return static_cast<int>(pattern_idx);
#else
  return __builtin_ctzll(mask);
#endif
}

bool PropagateWave(const WFCModel& model, int width, int height,
                   std::vector<WFCMask>* wave) {
  std::vector<WFCMask> next(wave->size(), 0);
  while (true) {
    bool changed = false;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int idx = y * width + x;
        const WFCMask cur_mask = (*wave)[idx];
        WFCMask next_mask = cur_mask;
        for (WFCMask allowed = cur_mask; allowed != 0; allowed &= allowed - 1) {
          const int pattern_idx = LowestAllowedPattern(allowed);
          bool supported = true;
          for (int dir_idx = 0; dir_idx < 4; ++dir_idx) {
            int next_x = x + kWFCDirections[dir_idx].first;
            int next_y = y + kWFCDirections[dir_idx].second;
            if (model.output_periodic) {
              next_x = (next_x + width) % width;
              next_y = (next_y + height) % height;
            } else if (next_x < 0 || next_x >= width || next_y < 0 ||
                       next_y >= height) {
              continue;
            }
            const WFCMask neighbor_mask = (*wave)[next_y * width + next_x];
            if ((model.adjacency[dir_idx][pattern_idx] & neighbor_mask) == 0) {
              supported = false;
              break;
            }
          }
          if (!supported) {
            next_mask &= ~(WFCMask{1} << pattern_idx);
          }
        }
        if (next_mask == 0) {
          return false;
        }
        next[idx] = next_mask;
        if (next_mask != cur_mask) {
          changed = true;
        }
      }
    }
    if (!changed) {
      return true;
    }
    *wave = next;
  }
}

int SelectPattern(WFCMask mask, const std::vector<double>& weights,
                  std::mt19937* gen_ref) {
  double total_weight = 0.0;
  for (WFCMask allowed = mask; allowed != 0; allowed &= allowed - 1) {
    total_weight += weights[LowestAllowedPattern(allowed)];
  }
  CHECK_GT(total_weight, 0.0);
  std::uniform_real_distribution<double> dist(0.0, total_weight);
  const double sample = dist(*gen_ref);

  double cumulative = 0.0;
  int fallback_idx = 0;
  for (WFCMask allowed = mask; allowed != 0; allowed &= allowed - 1) {
    const int pattern_idx = LowestAllowedPattern(allowed);
    fallback_idx = pattern_idx;
    cumulative += weights[pattern_idx];
    if (sample < cumulative) {
      return pattern_idx;
    }
  }
  return fallback_idx;
}

bool SolveWFC(const WFCModel& model, int width, int height,
              std::mt19937* gen_ref, WFCBitmap* bitmap) {
  std::vector<WFCMask> wave(width * height, model.full_mask);
  std::vector<double> preferences(width * height, 0.0);
  std::uniform_real_distribution<double> dist(0.0, 0.1);
  for (double& preference : preferences) {
    preference = dist(*gen_ref);
  }

  while (true) {
    if (!PropagateWave(model, width, height, &wave)) {
      return false;
    }

    int best_idx = -1;
    double best_score = std::numeric_limits<double>::infinity();
    for (int idx = 0; idx < width * height; ++idx) {
      const int num_allowed = CountAllowedPatterns(wave[idx]);
      if (num_allowed <= 1) {
        continue;
      }
      const double score = static_cast<double>(num_allowed) + preferences[idx];
      if (score < best_score) {
        best_score = score;
        best_idx = idx;
      }
    }
    if (best_idx < 0) {
      break;
    }
    const int pattern_idx =
        SelectPattern(wave[best_idx], model.weights, gen_ref);
    wave[best_idx] = WFCMask{1} << pattern_idx;
  }

  bitmap->assign(height, std::vector<std::uint8_t>(width, 0));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const WFCMask mask = wave[y * width + x];
      CHECK_EQ(CountAllowedPatterns(mask), 1);
      const int pattern_idx = LowestAllowedPattern(mask);
      (*bitmap)[y][x] = model.patterns[pattern_idx][0];
    }
  }
  return true;
}

bool KeepLargestConnectedComponent(WFCBitmap* bitmap,
                                   std::vector<Pos>* navigable_cells) {
  const int height = static_cast<int>(bitmap->size());
  const int width = static_cast<int>((*bitmap)[0].size());
  std::vector<std::vector<std::uint8_t>> seen(
      height, std::vector<std::uint8_t>(width, 0));
  std::vector<Pos> best_component;

  for (int start_y = 0; start_y < height; ++start_y) {
    for (int start_x = 0; start_x < width; ++start_x) {
      if ((*bitmap)[start_y][start_x] != 0 || seen[start_y][start_x] != 0) {
        continue;
      }
      std::queue<Pos> frontier;
      std::vector<Pos> component;
      frontier.emplace(start_x, start_y);
      seen[start_y][start_x] = 1;
      while (!frontier.empty()) {
        const Pos pos = frontier.front();
        frontier.pop();
        component.push_back(pos);
        for (const Pos& dir : kWFCDirections) {
          const int next_x = pos.first + dir.first;
          const int next_y = pos.second + dir.second;
          if (next_x < 0 || next_x >= width || next_y < 0 || next_y >= height) {
            continue;
          }
          if ((*bitmap)[next_y][next_x] != 0 || seen[next_y][next_x] != 0) {
            continue;
          }
          seen[next_y][next_x] = 1;
          frontier.emplace(next_x, next_y);
        }
      }
      if (component.size() > best_component.size() && component.size() > 1) {
        best_component = std::move(component);
      }
    }
  }

  if (best_component.size() < 2) {
    return false;
  }

  bitmap->assign(height, std::vector<std::uint8_t>(width, 1));
  for (const Pos& pos : best_component) {
    (*bitmap)[pos.second][pos.first] = 0;
  }
  *navigable_cells = std::move(best_component);
  return true;
}

void CollectNavigableCells(const WFCBitmap& bitmap,
                           std::vector<Pos>* navigable_cells) {
  navigable_cells->clear();
  const int height = static_cast<int>(bitmap.size());
  const int width = static_cast<int>(bitmap[0].size());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (bitmap[y][x] == 0) {
        navigable_cells->emplace_back(x, y);
      }
    }
  }
}

}  // namespace

WFCTask::WFCTask(std::string wfc_preset, int size, bool ensure_connected,
                 int max_steps)
    : MiniGridTask("wfc", max_steps, 7, false),
      wfc_preset_(std::move(wfc_preset)),
      size_(size),
      ensure_connected_(ensure_connected) {
  CHECK_GE(size_, 3);
}

void WFCTask::GenGrid() {
  const int inner_size = size_ - 2 * kWFCPadding;
  const WFCModel model = BuildWFCModel(GetWFCPreset(wfc_preset_));

  WFCBitmap bitmap;
  std::vector<Pos> navigable_cells;
  bool success = false;
  for (int attempt = 0; attempt < kWFCMaxAttempts; ++attempt) {
    if (!SolveWFC(model, inner_size, inner_size, gen_ref_, &bitmap)) {
      continue;
    }
    if (ensure_connected_) {
      if (!KeepLargestConnectedComponent(&bitmap, &navigable_cells)) {
        continue;
      }
    } else {
      CollectNavigableCells(bitmap, &navigable_cells);
      if (navigable_cells.size() < 2) {
        continue;
      }
    }
    success = true;
    break;
  }
  if (!success) {
    throw std::runtime_error("Could not generate a valid WFC pattern");
  }

  ClearGrid(size_, size_);
  WallRect(0, 0, size_, size_);
  for (int y = 0; y < inner_size; ++y) {
    for (int x = 0; x < inner_size; ++x) {
      if (bitmap[y][x] != 0) {
        PutObj(WorldObj(kWall), x + kWFCPadding, y + kWFCPadding);
      }
    }
  }

  const std::vector<Pos> endpoints = RandSubset(navigable_cells, 2);
  agent_pos_ = {endpoints[0].first + kWFCPadding,
                endpoints[0].second + kWFCPadding};
  agent_dir_ = RandInt(0, 4);
  goal_pos_ = {endpoints[1].first + kWFCPadding,
               endpoints[1].second + kWFCPadding};
  PutObj(WorldObj(kGoal), goal_pos_.first, goal_pos_.second);
  SetMission("traverse the maze to get to the goal");
}

}  // namespace minigrid
