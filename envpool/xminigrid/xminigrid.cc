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

// Native adaptation of xminigrid 0.9.3 actions, goals and production rules.
#include "envpool/xminigrid/xminigrid.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace xminigrid {
namespace {

enum Predicate : uint8_t {
  kNever,
  kHold,
  kStandOn,
  kAgentNear,
  kTileNear,
  kTileAt,
  kAgentAt,
  kPairUp,
  kPairRight,
  kPairDown,
  kPairLeft,
  kAgentUp,
  kAgentRight,
  kAgentDown,
  kAgentLeft
};

enum : uint8_t { kForward, kRight, kLeft, kPickUp, kPutDown, kToggle };

// Production rules share a subset of the goal predicates. Their last
// non-padding tile encodes the product.
constexpr std::array kRulePredicates = {
    kNever,    kHold,     kAgentNear, kTileNear,   kPairUp,    kPairRight,
    kPairDown, kPairLeft, kAgentUp,   kAgentRight, kAgentDown, kAgentLeft,
};

bool IsPair(int op) {
  return op == kTileNear || (op >= kPairUp && op <= kPairLeft);
}

void CheckTile(Tile tile) {
  if (tile[0] > kStar || tile[1] > kPink) {
    throw std::invalid_argument("Invalid ruleset tile or color");
  }
}

}  // namespace

void XMiniGridSpec::Validate(const RuleSet& ruleset) const {
  if (task.layout > Layout::kR9) {
    throw std::invalid_argument("Only XLand tasks accept a ruleset");
  }
  const auto& g = ruleset.goal;
  if (g[0] > kAgentLeft) {
    throw std::invalid_argument("Unknown goal");
  }
  if (g[0] != kAgentAt) {
    CheckTile({g[1], g[2]});
  }
  if (IsPair(g[0])) {
    CheckTile({g[3], g[4]});
  }
  if (g[0] == kAgentAt || g[0] == kTileAt) {
    int i = g[0] == kAgentAt ? 1 : 3;
    if (g[i] >= task.height || g[i + 1] >= task.width) {
      throw std::invalid_argument("Goal position lies outside the grid");
    }
  }
  for (const Rule& rule : ruleset.rules) {
    if (rule[0] >= std::size(kRulePredicates)) {
      throw std::invalid_argument("Unknown rule");
    }
    for (int i : {1, 3, 5}) {
      CheckTile({rule[i], rule[i + 1]});
    }
  }
  int count = 0;
  for (Tile tile : ruleset.init_tiles) {
    CheckTile(tile);
    if (tile != kPadding && ++count >= num_floors) {
      throw std::invalid_argument("Too many initial tiles for this grid");
    }
  }
}

void XMiniGridEnv::SetRuleSet(std::shared_ptr<const RuleSet> ruleset) {
  std::scoped_lock lock(rules_mutex_);
  next_rules_ = std::move(ruleset);
}

void XMiniGridEnv::Reset() {
  {
    std::scoped_lock lock(rules_mutex_);
    rules_ = next_rules_;
  }
  pocket_ = {};
  steps_ = 0;
  Generate();
  WriteState(false, false);
}

Pos XMiniGridEnv::Act(int action) {
  if (action == kRight || action == kLeft) {
    direction_ = (direction_ + (action == kRight ? 1 : 3)) % 4;
    return agent_;
  }
  Pos front = agent_.Shift(direction_);
  if (!Contains(front)) {
    return agent_;
  }
  Tile& tile = At(front);
  int type = tile[0];
  switch (action) {
    case kForward:
      if (type == kFloor || type == kGoal || type == kDoorOpen) {
        agent_ = front;
      }
      return agent_;
    case kPickUp:
      if (pocket_[0] == kEmpty &&
          (type == kBall || type == kSquare || type == kPyramid ||
           type == kKey || type == kHex || type == kStar)) {
        pocket_ = tile;
        tile = kGround;
      }
      break;
    case kPutDown:
      if (type == kFloor && pocket_[0] != kEmpty) {
        tile = pocket_;
        pocket_ = {};
      }
      break;
    case kToggle:
      if (type == kDoorOpen) {
        tile[0] = kDoorClosed;
      } else if (type == kDoorClosed ||
                 (type == kDoorLocked && pocket_ == Tile{kKey, tile[1]})) {
        tile[0] = kDoorOpen;
      }
      break;
    default:
      break;
  }
  // Even an unsuccessful put-down checks the cell in front.
  return front;
}

bool XMiniGridEnv::Match(const Goal& condition, int action, Pos target,
                         const Tile* product) {
  const int op = condition[0];
  const Tile a{condition[1], condition[2]};
  const Tile b{condition[3], condition[4]};
  switch (op) {
    case kNever:
      return false;
    case kHold:
      if (action != kPickUp || pocket_ != a) {
        return false;
      }
      if (product != nullptr) {
        pocket_ = *product;
      }
      return true;
    case kStandOn:
      return action == kForward && At(agent_) == a;
    case kTileAt:
      return At({b[0], b[1]}) == a;
    case kAgentAt:
      return agent_ == Pos{a[0], a[1]};
    default:
      break;
  }

  const bool pair = IsPair(op);
  if (pair) {
    if (action != kPutDown) {
      return false;
    }
  } else {
    if (action != kForward && (op < kAgentUp || action != kPutDown)) {
      return false;
    }
    target = agent_;
  }
  // For directed pairs B must lie in the specified compass direction
  // from A, regardless of which tile was dropped. A single rule can
  // transform more than one neighbour using the original target tile.
  const Tile center = At(target);
  bool found = false;
  for (int d = 0; d < 4; ++d) {
    Pos p = target.Shift(d);
    if (!Contains(p)) {
      continue;
    }
    Tile& other = At(p);
    bool matches =
        pair ? ((center == a && other == b &&
                 (op == kTileNear || op == kPairUp + d)) ||
                (center == b && other == a &&
                 (op == kTileNear || op == kPairUp + (d + 2) % 4)))
             : (other == a && (op == kAgentNear || op == kAgentUp + d));
    if (!matches) {
      continue;
    }
    if (product == nullptr) {
      return true;
    }
    other = *product;
    found = true;
  }
  if (found && pair) {
    At(target) = kGround;
  }
  return found;
}

void XMiniGridEnv::Step(const Action& action) {
  int a = action["action"_];
  ++steps_;
  if (spec_.task.layout == Layout::kMemory) {
    Act(a == kPickUp ? kToggle : a);
    int end = spec_.task.width - 2;
    bool choice = agent_.x == end && (agent_.y == 2 || agent_.y == 4);
    bool success = choice && At({2, 1}) == At({agent_.y == 2 ? 1 : 5, end});
    WriteState(success, choice);
    return;
  }

  Pos target = Act(a);
  if (rules_) {
    for (const Rule& r : rules_->rules) {
      Predicate op = kRulePredicates[r[0]];
      int i = IsPair(op) ? 5 : 3;
      Tile product{r[i], r[i + 1]};
      Match({op, r[1], r[2], r[3], r[4]}, a, target, &product);
    }
  }
  bool success = Match(goal_, a, target, nullptr);
  WriteState(success, success);
}

Pos XMiniGridEnv::ViewCell(int row, int col) const {
  const int n = spec_.config["view_size"_];
  return agent_.Shift(direction_, n - 1 - row)
      .Shift((direction_ + 1) % 4, col - n / 2);
}

void XMiniGridEnv::WriteState(bool success, bool terminated) {
  int limit = spec_.config["max_episode_steps"_];
  done_ = terminated || steps_ >= limit;
  State state = Allocate();
  state["reward"_] =
      success ? 1.0f - 0.9f * (static_cast<float>(steps_) / limit) : 0.0f;
  state["terminated"_] = terminated;
  state["discount"_] = terminated ? 0.0f : 1.0f;
  state["info:agent_pos"_][0] = agent_.y;
  state["info:agent_pos"_][1] = agent_.x;
  const int n = spec_.config["view_size"_];
  auto* obs = static_cast<uint8_t*>(state["obs"_].Data());
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      Pos pos = ViewCell(row, col);
      Tile cell = Contains(pos) ? At(pos) : Tile{};
      *obs++ = cell[0];
      *obs++ = cell[1];
    }
  }
#ifdef ENVPOOL_TEST
  static_assert(sizeof(Tile) == 2);
  std::memcpy(state["info:grid"_].Data(), grid_.data(),
              grid_.size() * sizeof(Tile));
  state["info:direction"_] = direction_;
  state["info:pocket"_][0] = pocket_[0];
  state["info:pocket"_][1] = pocket_[1];
#endif
}

}  // namespace xminigrid
