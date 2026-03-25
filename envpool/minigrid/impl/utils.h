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

#ifndef ENVPOOL_MINIGRID_IMPL_UTILS_H_
#define ENVPOOL_MINIGRID_IMPL_UTILS_H_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/logging.h"

namespace minigrid {

enum Act : std::uint8_t {
  kLeft = 0,
  kRight = 1,
  kForward = 2,
  kPickup = 3,
  kDrop = 4,
  kToggle = 5,
  kDone = 6
};

enum Color : std::uint8_t {
  kRed = 0,
  kGreen = 1,
  kBlue = 2,
  kPurple = 3,
  kYellow = 4,
  kGrey = 5,
  kUnassigned = 6
};

enum Type : std::uint8_t {
  kUnseen = 0,
  kEmpty = 1,
  kWall = 2,
  kFloor = 3,
  kDoor = 4,
  kKey = 5,
  kBall = 6,
  kBox = 7,
  kGoal = 8,
  kLava = 9,
  kAgent = 10
};

using Pos = std::pair<int, int>;

inline constexpr int kMissionBytes = 96;
inline constexpr std::array<Color, 6> kColors = {
    kRed,    kGreen,  kBlue,
    kPurple, kYellow, kGrey};  // NOLINT(whitespace/indent_namespace)
inline constexpr std::array<Type, 3> kObjectTypes = {kKey, kBall, kBox};

inline std::string ColorName(Color color) {
  static const std::array<const char*, 6> k_names = {
      "red", "green", "blue", "purple", "yellow", "grey"};
  CHECK_NE(color, kUnassigned);
  return k_names[static_cast<int>(color)];
}

inline std::string TypeName(Type type) {
  static const std::array<const char*, 11> k_names = {
      "unseen", "empty", "wall", "floor", "door", "key",
      "ball",   "box",   "goal", "lava",  "agent"};
  return k_names[static_cast<int>(type)];
}

inline Color ParseColor(const std::string& color) {
  if (color == "red") {
    return kRed;
  }
  if (color == "green") {
    return kGreen;
  }
  if (color == "blue") {
    return kBlue;
  }
  if (color == "purple") {
    return kPurple;
  }
  if (color == "yellow") {
    return kYellow;
  }
  if (color == "grey") {
    return kGrey;
  }
  LOG(FATAL) << "Unknown color: " << color;
}

inline Type ParseType(const std::string& type) {
  if (type == "empty") {
    return kEmpty;
  }
  if (type == "wall") {
    return kWall;
  }
  if (type == "floor") {
    return kFloor;
  }
  if (type == "door") {
    return kDoor;
  }
  if (type == "key") {
    return kKey;
  }
  if (type == "ball") {
    return kBall;
  }
  if (type == "box") {
    return kBox;
  }
  if (type == "goal") {
    return kGoal;
  }
  if (type == "lava") {
    return kLava;
  }
  if (type == "agent") {
    return kAgent;
  }
  LOG(FATAL) << "Unknown type: " << type;
}

inline Color DefaultColor(Type type) {
  switch (type) {
    case kEmpty:
    case kLava:
      return kRed;
    case kWall:
      return kGrey;
    case kGoal:
      return kGreen;
    case kKey:
    case kBall:
    case kFloor:
      return kBlue;
    default:
      LOG(FATAL) << "Type " << static_cast<int>(type)
                 << " requires an explicit color";
  }
}

class WorldObj {
 public:
  explicit WorldObj(Type type = kEmpty, Color color = kUnassigned,
                    bool door_open = false, bool door_locked = false)
      : type_(type),
        color_(color == kUnassigned ? DefaultColor(type) : color),
        door_open_(door_open),
        door_locked_(door_locked) {}

  WorldObj(const WorldObj& other)
      : type_(other.type_),
        color_(other.color_),
        door_open_(other.door_open_),
        door_locked_(other.door_locked_) {
    if (other.contains_ != nullptr) {
      contains_ = std::make_unique<WorldObj>(*other.contains_);
    }
  }

  WorldObj(WorldObj&&) noexcept = default;
  WorldObj& operator=(WorldObj&&) noexcept = default;

  WorldObj& operator=(const WorldObj& other) {
    if (this == &other) {
      return *this;
    }
    type_ = other.type_;
    color_ = other.color_;
    door_open_ = other.door_open_;
    door_locked_ = other.door_locked_;
    contains_.reset();
    if (other.contains_ != nullptr) {
      contains_ = std::make_unique<WorldObj>(*other.contains_);
    }
    return *this;
  }

  [[nodiscard]] Type GetType() const { return type_; }
  [[nodiscard]] Color GetColor() const { return color_; }
  [[nodiscard]] bool GetDoorOpen() const { return door_open_; }
  [[nodiscard]] bool GetDoorLocked() const { return door_locked_; }
  void SetDoorOpen(bool open) { door_open_ = open; }
  void SetDoorLocked(bool locked) { door_locked_ = locked; }

  [[nodiscard]] bool CanSeeBehind() const {
    if (type_ == kDoor) {
      return door_open_;
    }
    switch (type_) {
      case kWall:
        return false;
      default:
        return true;
    }
  }

  [[nodiscard]] bool CanOverlap() const {
    if (type_ == kDoor) {
      return door_open_;
    }
    switch (type_) {
      case kWall:
      case kKey:
      case kBall:
      case kBox:
        return false;
      default:
        return true;
    }
  }

  [[nodiscard]] bool CanPickup() const {
    return type_ == kKey || type_ == kBall || type_ == kBox;
  }

  [[nodiscard]] int GetState() const {
    if (type_ != kDoor) {
      return 0;
    }
    if (door_locked_) {
      return 2;
    }
    return door_open_ ? 0 : 1;
  }

  [[nodiscard]] const WorldObj* GetContains() const { return contains_.get(); }
  WorldObj* GetContains() { return contains_.get(); }

  void SetContains(std::unique_ptr<WorldObj> contains) {
    if (contains != nullptr) {
      CHECK_EQ(type_, kBox);
    }
    contains_ = std::move(contains);
  }

  std::unique_ptr<WorldObj> ReleaseContains() { return std::move(contains_); }

  [[nodiscard]] std::array<uint8_t, 3> Encode() const {
    return {static_cast<uint8_t>(type_), static_cast<uint8_t>(color_),
            static_cast<uint8_t>(GetState())};
  }

  [[nodiscard]] bool operator==(const WorldObj& other) const {
    return type_ == other.type_ && color_ == other.color_ &&
           door_open_ == other.door_open_ && door_locked_ == other.door_locked_;
  }

 private:
  Type type_;
  Color color_;
  std::unique_ptr<WorldObj> contains_{nullptr};
  bool door_open_{false};
  bool door_locked_{false};
};

inline WorldObj MakeObj(Type type, Color color = kUnassigned) {
  return WorldObj(type, color);
}

inline WorldObj MakeDoor(Color color, bool locked = false, bool open = false) {
  return WorldObj(kDoor, color, open, locked);
}

}  // namespace minigrid

#endif  // ENVPOOL_MINIGRID_IMPL_UTILS_H_
