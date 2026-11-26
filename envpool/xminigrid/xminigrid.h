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

// Native adaptation of dunnolab/xland-minigrid 0.9.3 (Apache-2.0).
#ifndef ENVPOOL_XMINIGRID_XMINIGRID_H_
#define ENVPOOL_XMINIGRID_XMINIGRID_H_

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/env.h"

namespace xminigrid {

enum class Layout : uint8_t {
  kR1,
  kR2,
  kR4,
  kR6,
  kR9,
  kEmpty,
  kEmptyRandom,
  kFourRooms,
  kDoorKey,
  kLockedRoom,
  kPlayground,
  kUnlock,
  kUnlockPickUp,
  kBlockedUnlockPickUp,
  kMemory
};

struct Task {
  const char* id;
  Layout layout;
  int height, width, view_size, max_steps, num_objects;
};
inline constexpr std::array kTasks = {
#include "third_party/xminigrid/tasks.inc"
};

enum Type : uint8_t {
  kEmpty,
  kFloor,
  kWall,
  kBall,
  kSquare,
  kPyramid,
  kGoal,
  kKey,
  kDoorLocked,
  kDoorClosed,
  kDoorOpen,
  kHex,
  kStar
};
enum Color : uint8_t {
  kNone,
  kRed,
  kGreen,
  kBlue,
  kPurple,
  kYellow,
  kGrey,
  kBlack,
  kOrange,
  kWhite,
  kBrown,
  kPink
};
using Tile = std::array<uint8_t, 2>;
using Goal = std::array<uint8_t, 5>;
using Rule = std::array<uint8_t, 7>;
inline constexpr Tile kPadding{};
inline constexpr Tile kGround{kFloor, kBlack};
inline constexpr Tile kBarrier{kWall, kGrey};
inline constexpr Tile kExit{kGoal, kGreen};

struct RuleSet {
  Goal goal;
  std::vector<Rule> rules;
  std::vector<Tile> init_tiles;
};

struct Pos {
  int y, x;
  Pos Shift(int dir, int steps = 1) const {
    constexpr std::array dirs = {Pos{-1, 0}, Pos{0, 1}, Pos{1, 0}, Pos{0, -1}};
    return {y + steps * dirs[dir].y, x + steps * dirs[dir].x};
  }
  bool operator==(Pos p) const { return y == p.y && x == p.x; }
};

struct XMiniGridFns {
  static auto DefaultConfig() {
    return MakeDict("task"_.Bind(std::string(kTasks[0].id)),
                    "view_size"_.Bind(kTasks[0].view_size),
                    "num_objects"_.Bind(0),
                    "ruleset"_.Bind(std::shared_ptr<RuleSet>{}));
  }

  static const Task& Lookup(const std::string& id);

  template <typename Config>
  static auto StateSpec(const Config& c) {
    const int view = c["view_size"_];
    auto state =
        MakeDict("obs"_.Bind(Spec<uint8_t>({view, view, 2}, {0, kStar})),
                 "terminated"_.Bind(Spec<bool>({})),
                 "info:agent_pos"_.Bind(Spec<int>({2})));
#ifdef ENVPOOL_TEST
    const auto& t = Lookup(c["task"_]);
    return ConcatDict(
        state,
        MakeDict("info:grid"_.Bind(Spec<uint8_t>({t.height, t.width, 2})),
                 "info:direction"_.Bind(Spec<int>({})),
                 "info:pocket"_.Bind(Spec<uint8_t>({2}))));
#else
    return state;
#endif
  }

  template <typename Config>
  static auto ActionSpec(const Config&) {
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 5})));
  }
};

class XMiniGridSpec : public EnvSpec<XMiniGridFns> {
 public:
  explicit XMiniGridSpec(const ConfigValues& conf);
  void Validate(const RuleSet& ruleset) const;

  const Task& task;
  std::vector<Tile> base_grid;
  int num_floors;
};

class XMiniGridEnv : public Env<XMiniGridSpec>, public RenderableEnv {
 public:
  XMiniGridEnv(const Spec& spec, int env_id)
      : Env(spec, env_id),
        grid_(spec.base_grid),
        next_rules_(spec.config["ruleset"_]) {}
  bool IsDone() override { return done_; }
  void Reset() override;
  void Step(const Action& action) override;
  void SetRuleSet(std::shared_ptr<const RuleSet> ruleset);
  std::pair<int, int> RenderSize(int width, int height) const override;
  void Render(int width, int height, int camera_id,
              unsigned char* rgb) override;

 private:
  Tile& At(Pos p) { return grid_[p.y * spec_.task.width + p.x]; }
  bool Contains(Pos p) const {
    return p.y >= 0 && p.y < spec_.task.height && p.x >= 0 &&
           p.x < spec_.task.width;
  }
  Pos ViewCell(int row, int col) const;
  void Generate();
  void WriteState(bool success, bool terminated);
  Pos Act(int action);
  bool Match(const Goal& condition, int action, Pos target,
             const Tile* product);

  std::vector<Tile> grid_;
  Pos agent_{1, 1};
  int direction_{0}, steps_{0};
  Tile pocket_{};
  Goal goal_{};
  bool done_{true};

  // Staging a task during an episode must only affect the next reset.
  std::mutex rules_mutex_;
  std::shared_ptr<const RuleSet> next_rules_, rules_;
};

}  // namespace xminigrid

#endif  // ENVPOOL_XMINIGRID_XMINIGRID_H_
