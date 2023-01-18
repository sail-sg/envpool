/*
 * Copyright 2023 Garena Online Private Limited
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

#include <unordered_map>

namespace minigrid {

enum Act {
  // Turn left, turn right, move forward
  kLeft = 0,
  kRight = 1,
  kForward = 2,
  // Pick up an object
  kPickup = 3,
  // Drop an object
  kDrop = 4,
  // Toggle/activate an object
  kToggle = 5,
  // Done completing task
  kDone = 6
};

enum Color {
  kRed = 0,
  kGreen = 1,
  kBlue = 2,
  kPurple = 3,
  kYellow = 4,
  kGrey = 5,
  kUnassigned = 6
};

enum Type {
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

// constants
static const int TILE_PIXELS = 32;
static const std::unordered_map<Type, bool> CAN_SEE_BEHIND{
  {kEmpty, true}, {kWall, false}, {kGoal, true}, {kFloor, true}, {kLava, true}, 
  {kKey, true}, {kBall, true}, {kDoor, true}, {kBox, true}};
static const std::unordered_map<Type, bool> CAN_OVERLAP{
  {kEmpty, true}, {kWall, false}, {kGoal, true}, {kFloor, true}, {kLava, true}, 
  {kKey, false}, {kBall, false}, {kDoor, true}, {kBox, false}};
static const std::unordered_map<Type, bool> CAN_PICKUP {
  {kEmpty, false}, {kWall, false}, {kGoal, false}, {kFloor, false}, {kLava, false}, 
  {kKey, true}, {kBall, true}, {kDoor, false}, {kBox, true}};

// object class

class WorldObj {
 private:
  Type type_;
  Color color_;
  bool door_open_{true};  // this variable only makes sence when type_ == kDoor
  bool door_locked_{false};  // this variable only makes sence when type_ == kDoor

 public:
  WorldObj(Type type = kEmpty, Color color = kUnassigned) : type_(type) {
    if (color == kUnassigned) {
      switch (type) {
        case kEmpty:
          color_ = kRed;
          break;
        case kWall:
          color_ = kGrey;
          break;
        case kGoal:
          color_ = kGreen;
          break;
        case kFloor:
          color_ = kBlue;
          break;
        case kLava:
          color_ = kRed;
          break;
        case kKey:
          color_ = kBlue;
          break;
        case kBall:
          color_ = kBlue;
          break;
        default:
          CHECK(false);
          break;
      }
    } else {
      color_ = color;
    }
  }
  bool CanSeeBehind() const { return door_open_ && CAN_SEE_BEHIND.at(type_); }
  bool CanOverlap() const { return door_open_ && CAN_OVERLAP.at(type_); }
  bool CanPickup() const { return CAN_PICKUP.at(type_); }
  bool GetDoorOpen() { return door_open_; }
  void SetDoorOpen(bool flag) { door_open_ = flag; }
  bool GetDoorLocked() { return door_locked_; }
  void SetDoorLocker(bool flag) { door_locked_ = flag; }
  Type GetType() { return type_; }
  Color GetColor() { return color_; }
  int GetState() {
    if (type_ != kDoor) {
      return 0;
    } else {
      if (door_locked_) return 2;
      if (door_open_) return 0;
      return 1;
    }
  }
};

}  // namespace minigrid

#endif  // ENDPOOL_MINIGRID_IMPL_UTILS_H_
