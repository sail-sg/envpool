// Copyright 2026 Garena Online Private Limited
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Native adaptation of the room and task generators in xminigrid 0.9.3.
#include <algorithm>
#include <array>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "envpool/xminigrid/xminigrid.h"

namespace xminigrid {
namespace {

// Doors replace wall cells. The remaining floor cells are exactly the
// sampling domain for XLand's initial objects and agent.
std::vector<Tile> BaseGrid(const Task& task) {
  const int h = task.height;
  const int w = task.width;
  std::vector<Tile> grid(h * w, kBarrier);
  for (int y = 1; y < h - 1; ++y) {
    for (int x = 1; x < w - 1; ++x) {
      bool wall = false;
      switch (task.layout) {
        case Layout::kR2:
        case Layout::kUnlock:
        case Layout::kUnlockPickUp:
        case Layout::kBlockedUnlockPickUp:
          wall = x == w / 2;
          break;
        case Layout::kR4:
        case Layout::kFourRooms:
          wall = x == w / 2 || y == h / 2;
          break;
        case Layout::kR6:
        case Layout::kLockedRoom:
          wall = x == w / 2 - 2 || x == w / 2 + 2 ||
                 ((x < w / 2 - 2 || x > w / 2 + 2) &&
                  (y == h / 3 || y == 2 * (h / 3)));
          break;
        case Layout::kR9:
        case Layout::kPlayground:
          wall =
              x == w / 3 || x == 2 * (w / 3) || y == h / 3 || y == 2 * (h / 3);
          break;
        case Layout::kMemory:
          wall = ((x <= 4 || x == w - 3) && (y == 1 || y == 5)) ||
                 (x >= 4 && x < w - 2 && (y == 2 || y == 4));
          break;
        default:
          break;
      }
      if (!wall) {
        grid[y * w + x] = kGround;
      }
    }
  }
  return grid;
}

}  // namespace

const Task& XMiniGridFns::Lookup(const std::string& id) {
  for (const Task& task : kTasks) {
    if (id == task.id) {
      return task;
    }
  }
  throw std::invalid_argument("Unknown xminigrid task: " + id);
}

XMiniGridSpec::XMiniGridSpec(const ConfigValues& conf)
    : EnvSpec(conf),
      task(XMiniGridFns::Lookup(config["task"_])),
      base_grid(BaseGrid(task)),
      num_floors(std::count(base_grid.begin(), base_grid.end(), kGround)) {
  const int view = config["view_size"_];
  if (view < 1 || view > 255 || view % 2 == 0) {
    throw std::invalid_argument("view_size must be odd and in [1, 255]");
  }
  if (config["max_episode_steps"_] < 1) {
    throw std::invalid_argument("max_episode_steps must be positive");
  }
  const int count = config["num_objects"_];
  if (count < 0 || count >= num_floors ||
      (task.layout != Layout::kPlayground && count != 0)) {
    throw std::invalid_argument("Invalid num_objects for this grid");
  }
  if (const auto& ruleset = config["ruleset"_]) {
    Validate(*ruleset);
  }
}

void XMiniGridEnv::Generate() {
  const auto& t = spec_.task;
  const int h = t.height;
  const int w = t.width;
  auto draw = [&](int low, int high) {
    return std::uniform_int_distribution<int>(low, high - 1)(gen_);
  };
  auto color = [&]() { return static_cast<uint8_t>(draw(kRed, kGrey + 1)); };
  auto door = [&]() -> Tile { return {kDoorClosed, color()}; };
  auto object = [&](int types) -> Tile {
    constexpr std::array kinds = {kBall, kSquare, kPyramid, kHex,
                                  kStar, kKey,    kGoal};
    return {kinds[draw(0, types)], color()};
  };
  grid_ = spec_.base_grid;
  direction_ = draw(0, 4);
  goal_ = {};
  constexpr Goal reach_exit{2, kGoal, kGreen};

  switch (t.layout) {
    case Layout::kR1:
      break;
    case Layout::kR2:
      At({draw(1, h - 1), w / 2}) = door();
      break;
    case Layout::kEmpty:
    case Layout::kEmptyRandom:
      At({h - 2, w - 2}) = kExit;
      goal_ = reach_exit;
      if (t.layout == Layout::kEmpty) {
        agent_ = {1, 1};
        direction_ = 1;
        return;
      }
      break;
    case Layout::kR4:
    case Layout::kFourRooms:
      for (int offset : {0, w / 2}) {
        for (Pos p : {Pos{h / 2, offset + draw(1, w / 2)},
                      Pos{offset + draw(1, h / 2), w / 2}}) {
          At(p) = t.layout == Layout::kR4 ? door() : kGround;
        }
      }
      break;
    case Layout::kR6:
    case Layout::kLockedRoom: {
      auto entrance = [&](int i) -> Pos {
        return {h / 2 + (i % 3 - 1) * (h / 3), w / 2 + (i < 3 ? -2 : 2)};
      };
      if (t.layout == Layout::kR6) {
        for (int i = 0; i < 6; ++i) {
          At(entrance(i)) = door();
        }
        break;
      }
      std::array<Color, 6> colors{kRed, kGreen, kBlue, kPurple, kYellow, kGrey};
      std::shuffle(colors.begin(), colors.end(), gen_);
      for (int i = 0; i < 6; ++i) {
        At(entrance(i)) = {kDoorClosed, colors[i]};
      }
      auto in_room = [&](int i) -> Pos {
        return {i % 3 * (h / 3) + draw(1, h / 3),
                (i < 3 ? 0 : w / 2 + 2) + draw(1, w / 2 - 2)};
      };
      int locked = draw(0, 6);
      At(entrance(locked))[0] = kDoorLocked;
      At(in_room(locked)) = kExit;
      At(in_room((locked + draw(1, 6)) % 6)) = {kKey, colors[locked]};
      agent_ = {draw(1, h - 1), draw(w / 2 - 1, w / 2 + 1)};
      goal_ = reach_exit;
      return;
    }
    case Layout::kR9:
    case Layout::kPlayground:
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          int y = row * (h / 3);
          int x = col * (w / 3);
          if (col < 2) {
            At({y + draw(1, h / 3), x + w / 3}) = door();
          }
          if (row < 2) {
            At({y + h / 3, x + draw(1, w / 3)}) = door();
          }
        }
      }
      break;
    case Layout::kDoorKey:
    case Layout::kUnlock:
    case Layout::kUnlockPickUp:
    case Layout::kBlockedUnlockPickUp: {
      const bool door_key = t.layout == Layout::kDoorKey;
      int x = door_key ? draw(2, w - 2) : w / 2;
      int y = draw(1, door_key ? w - 2 : h - 1);
      uint8_t lock = door_key ? kYellow : color();
      if (door_key) {
        for (int row = 1; row < h - 1; ++row) {
          At({row, x}) = kBarrier;
        }
        At({h - 2, w - 2}) = kExit;
        goal_ = reach_exit;
      } else if (t.layout == Layout::kUnlock) {
        goal_ = {5, kDoorOpen, lock, static_cast<uint8_t>(y),
                 static_cast<uint8_t>(x)};
      } else {
        Tile target = object(5);
        At({draw(1, h - 1), draw(x + 1, w - 1)}) = target;
        goal_ = {1, target[0], target[1]};
      }
      At({y, x}) = {kDoorLocked, lock};
      At({draw(1, h - 1), draw(1, x)}) = {kKey, lock};
      // 0.9.3 places the blocker last, even when it covers the key.
      if (t.layout == Layout::kBlockedUnlockPickUp) {
        At({y, x - 1}) = object(5);
      }
      do {
        agent_ = {draw(1, h - 1), draw(1, x)};
      } while (At(agent_) != kGround);
      return;
    }
    case Layout::kMemory: {
      constexpr std::array choices = {Tile{kKey, kGreen}, Tile{kBall, kGreen}};
      int top = draw(0, 2);
      At({2, 1}) = choices[draw(0, 2)];
      At({1, w - 2}) = choices[top];
      At({5, w - 2}) = choices[1 - top];
      agent_ = {3, draw(1, w - 2)};
      direction_ = 1;
      return;
    }
  }

  // Uniform draws without replacement, including when an initial tile
  // happens to be a floor tile itself.
  std::vector<Pos> free;
  free.reserve(spec_.num_floors + 4);
  for (int y = 1; y < h - 1; ++y) {
    for (int x = 1; x < w - 1; ++x) {
      if (At({y, x}) == kGround) {
        free.push_back({y, x});
      }
    }
  }
  auto sample = [&]() {
    int i = draw(0, static_cast<int>(free.size()));
    Pos pos = free[i];
    free[i] = free.back();
    free.pop_back();
    return pos;
  };
  if (rules_) {
    goal_ = rules_->goal;
    for (Tile tile : rules_->init_tiles) {
      if (tile != kPadding) {
        At(sample()) = tile;
      }
    }
  } else if (t.layout == Layout::kPlayground) {
    for (int i = 0; i < spec_.config["num_objects"_]; ++i) {
      At(sample()) = object(7);
    }
  } else if (t.layout == Layout::kFourRooms) {
    At(sample()) = kExit;
    goal_ = reach_exit;
  }
  agent_ = sample();
}

}  // namespace xminigrid
