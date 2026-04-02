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

#ifndef ENVPOOL_MINIGRID_IMPL_MINIGRID_TASK_UTILS_H_
#define ENVPOOL_MINIGRID_IMPL_MINIGRID_TASK_UTILS_H_

#include <array>
#include <cmath>
#include <string>

#include "envpool/minigrid/impl/minigrid_env.h"

namespace minigrid {

inline constexpr std::array<Pos, 4> kTaskDirToVec = {
    Pos{1, 0}, Pos{0, 1}, Pos{-1, 0},
    Pos{0, -1}};  // NOLINT(whitespace/indent_namespace)

inline int Manhattan(const Pos& lhs, const Pos& rhs) {
  return std::abs(lhs.first - rhs.first) + std::abs(lhs.second - rhs.second);
}

inline bool IsAdjacent(const Pos& lhs, const Pos& rhs) {
  return (lhs.first == rhs.first && std::abs(lhs.second - rhs.second) == 1) ||
         (lhs.second == rhs.second && std::abs(lhs.first - rhs.first) == 1);
}

inline int MissionObjectIndex(Type type) {
  switch (type) {
    case kKey:
      return 0;
    case kBall:
      return 1;
    default:
      return 2;
  }
}

inline Type OtherKeyBallType(Type type) {
  CHECK(type == kKey || type == kBall);
  return type == kKey ? kBall : kKey;
}

inline std::string MissionFetch(int syntax_idx, Color color, Type type) {
  static const std::array<const char*, 5> k_syntax = {
      "get a", "go get a", "fetch a", "go fetch a", "you must fetch a"};
  return std::string(k_syntax[syntax_idx]) + " " + ColorName(color) + " " +
         TypeName(type);
}

inline std::string MissionGoToDoor(Color color) {
  return "go to the " + ColorName(color) + " door";
}

inline std::string MissionGoToObject(Color color, Type type) {
  return "go to the " + ColorName(color) + " " + TypeName(type);
}

inline std::string MissionPutNear(Color move_color, Type move_type,
                                  Color target_color, Type target_type) {
  return "put the " + ColorName(move_color) + " " + TypeName(move_type) +
         " near the " + ColorName(target_color) + " " + TypeName(target_type);
}

inline std::string MissionLockedRoom(Color locked_room_color,
                                     Color key_room_color) {
  return "get the " + ColorName(locked_room_color) + " key from the " +
         ColorName(key_room_color) + " room, unlock the " +
         ColorName(locked_room_color) + " door and go to the goal";
}

inline std::string MissionPickUp(Color color, Type type) {
  return "pick up the " + ColorName(color) + " " + TypeName(type);
}

}  // namespace minigrid

#endif  // ENVPOOL_MINIGRID_IMPL_MINIGRID_TASK_UTILS_H_
