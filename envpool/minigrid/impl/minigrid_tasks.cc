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
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "envpool/minigrid/impl/minigrid_env.h"
#include "envpool/minigrid/impl/minigrid_task_utils.h"

namespace minigrid {

EmptyTask::EmptyTask(int size, Pos agent_start_pos, int agent_start_dir,
                     int max_steps, int agent_view_size)
    : MiniGridTask("empty", max_steps, agent_view_size, true, 6),
      size_(size),
      agent_start_pos_(std::move(agent_start_pos)),
      agent_start_dir_(agent_start_dir) {
}  // NOLINT(whitespace/indent_namespace)

void EmptyTask::GenGrid() {
  ClearGrid(size_, size_);
  WallRect(0, 0, size_, size_);
  PutObj(WorldObj(kGoal, kGreen), size_ - 2, size_ - 2);
  goal_pos_ = {size_ - 2, size_ - 2};
  if (agent_start_pos_.first >= 0) {
    agent_pos_ = agent_start_pos_;
    agent_dir_ = agent_start_dir_;
  } else {
    PlaceAgent(1, 1, size_ - 2, size_ - 2, true);
  }
  SetMission("get to the green goal square", 0);
}

DoorKeyTask::DoorKeyTask(int size, int max_steps)
    : MiniGridTask("doorkey", max_steps, 7, false, 6), size_(size) {}

void DoorKeyTask::GenGrid() {
  ClearGrid(size_, size_);
  WallRect(0, 0, size_, size_);
  PutObj(WorldObj(kGoal, kGreen), size_ - 2, size_ - 2);
  goal_pos_ = {size_ - 2, size_ - 2};
  int split_idx = RandInt(2, size_ - 2);
  VertWall(split_idx, 0);
  PlaceAgent(0, 0, split_idx, size_, true);
  int door_idx = RandInt(1, size_ - 2);
  PutObj(MakeDoor(kYellow, true, false), split_idx, door_idx);
  PlaceObj(WorldObj(kKey, kYellow), 0, 0, split_idx, size_);
  SetMission("use the key to open the door and then get to the goal", 0);
}

DistShiftTask::DistShiftTask(int width, int height, Pos agent_start_pos,
                             int agent_start_dir, int strip2_row, int max_steps)
    : MiniGridTask("distshift", max_steps, 7, true, 6),
      agent_start_pos_(std::move(agent_start_pos)),
      agent_start_dir_(agent_start_dir),
      strip2_row_(strip2_row) {
  width_ = width;
  height_ = height;
}

void DistShiftTask::GenGrid() {
  ClearGrid(width_, height_);
  WallRect(0, 0, width_, height_);
  PutObj(WorldObj(kGoal, kGreen), width_ - 2, 1);
  goal_pos_ = {width_ - 2, 1};
  for (int i = 0; i < width_ - 6; ++i) {
    PutObj(WorldObj(kLava), 3 + i, 1);
    PutObj(WorldObj(kLava), 3 + i, strip2_row_);
  }
  if (agent_start_pos_.first >= 0) {
    agent_pos_ = agent_start_pos_;
    agent_dir_ = agent_start_dir_;
  } else {
    PlaceAgent();
  }
  SetMission("get to the green goal square", 0);
}

LavaGapTask::LavaGapTask(int size, Type obstacle_type, int max_steps)
    : MiniGridTask("lavgap", max_steps, 7, false, 6),
      size_(size),
      obstacle_type_(obstacle_type) {}  // NOLINT(whitespace/indent_namespace)

void LavaGapTask::GenGrid() {
  ClearGrid(size_, size_);
  WallRect(0, 0, size_, size_);
  agent_pos_ = {1, 1};
  agent_dir_ = 0;
  goal_pos_ = {size_ - 2, size_ - 2};
  PutObj(WorldObj(kGoal, kGreen), goal_pos_.first, goal_pos_.second);
  Pos gap_pos{RandInt(2, size_ - 2), RandInt(1, size_ - 1)};
  VertWall(gap_pos.first, 1, size_ - 2, obstacle_type_,
           DefaultColor(obstacle_type_));
  SetEmpty(gap_pos.first, gap_pos.second);
  SetMission(obstacle_type_ == kLava
                 ? "avoid the lava and get to the green goal square"
                 : "find the opening and get to the green goal square",
             0);
}

CrossingTask::CrossingTask(int size, int num_crossings, Type obstacle_type,
                           int max_steps)
    : MiniGridTask("crossing", max_steps, 7, false, 6),
      size_(size),
      num_crossings_(num_crossings),
      obstacle_type_(obstacle_type) {}  // NOLINT(whitespace/indent_namespace)

void CrossingTask::GenGrid() {
  CHECK_EQ(size_ % 2, 1);
  ClearGrid(size_, size_);
  WallRect(0, 0, size_, size_);
  agent_pos_ = {1, 1};
  agent_dir_ = 0;
  goal_pos_ = {size_ - 2, size_ - 2};
  PutObj(WorldObj(kGoal, kGreen), goal_pos_.first, goal_pos_.second);

  std::vector<std::pair<bool, int>> rivers;
  for (int i = 2; i < size_ - 2; i += 2) {
    rivers.emplace_back(true, i);
    rivers.emplace_back(false, i);
  }
  std::shuffle(rivers.begin(), rivers.end(), *gen_ref_);
  rivers.resize(num_crossings_);
  std::vector<int> rivers_v;
  std::vector<int> rivers_h;
  for (const auto& river : rivers) {
    if (river.first) {
      rivers_v.push_back(river.second);
    } else {
      rivers_h.push_back(river.second);
    }
  }
  std::sort(rivers_v.begin(), rivers_v.end());
  std::sort(rivers_h.begin(), rivers_h.end());
  for (int y : rivers_h) {
    for (int x = 1; x < size_ - 1; ++x) {
      PutObj(WorldObj(obstacle_type_), x, y);
    }
  }
  for (int x : rivers_v) {
    for (int y = 1; y < size_ - 1; ++y) {
      PutObj(WorldObj(obstacle_type_), x, y);
    }
  }

  std::vector<bool> path;
  path.insert(path.end(), rivers_v.size(), true);
  path.insert(path.end(), rivers_h.size(), false);
  std::shuffle(path.begin(), path.end(), *gen_ref_);

  std::vector<int> limits_v = {0};
  limits_v.insert(limits_v.end(), rivers_v.begin(), rivers_v.end());
  limits_v.push_back(size_ - 1);
  std::vector<int> limits_h = {0};
  limits_h.insert(limits_h.end(), rivers_h.begin(), rivers_h.end());
  limits_h.push_back(size_ - 1);
  int room_i = 0;
  int room_j = 0;
  for (bool is_horizontal_move : path) {
    int x = 0;
    int y = 0;
    if (is_horizontal_move) {
      x = limits_v[room_i + 1];
      y = RandInt(limits_h[room_j] + 1, limits_h[room_j + 1]);
      room_i += 1;
    } else {
      x = RandInt(limits_v[room_i] + 1, limits_v[room_i + 1]);
      y = limits_h[room_j + 1];
      room_j += 1;
    }
    SetEmpty(x, y);
  }
  SetMission(obstacle_type_ == kLava
                 ? "avoid the lava and get to the green goal square"
                 : "find the opening and get to the green goal square",
             0);
}

DynamicObstaclesTask::DynamicObstaclesTask(int size, Pos agent_start_pos,
                                           int agent_start_dir, int n_obstacles,
                                           int max_steps)
    : MiniGridTask("dynamic_obstacles", max_steps, 7, true, 2),
      size_(size),
      agent_start_pos_(std::move(agent_start_pos)),
      agent_start_dir_(agent_start_dir),
      n_obstacles_(n_obstacles <= size / 2 + 1 ? n_obstacles : size / 2) {
}  // NOLINT(whitespace/indent_namespace)

void DynamicObstaclesTask::GenGrid() {
  ClearGrid(size_, size_);
  WallRect(0, 0, size_, size_);
  PutObj(WorldObj(kGoal, kGreen), size_ - 2, size_ - 2);
  goal_pos_ = {size_ - 2, size_ - 2};
  if (agent_start_pos_.first >= 0) {
    agent_pos_ = agent_start_pos_;
    agent_dir_ = agent_start_dir_;
  } else {
    PlaceAgent();
  }
  obstacle_pos_.clear();
  for (int i = 0; i < n_obstacles_; ++i) {
    obstacle_pos_.push_back(PlaceObj(WorldObj(kBall, kBlue), 0, 0, width_,
                                     height_, RejectFn(), 100));
  }
  SetMission("get to the green goal square", 0);
}

void DynamicObstaclesTask::BeforeStep(Act act, const WorldObj& pre_fwd) {
  pre_front_blocked_ = act == kForward && pre_fwd.GetType() != kGoal &&
                       pre_fwd.GetType() != kEmpty;
  std::vector<Pos> new_pos = obstacle_pos_;
  for (std::size_t i = 0; i < obstacle_pos_.size(); ++i) {
    Pos old_pos = obstacle_pos_[i];
    int top_x = std::max(old_pos.first - 1, 0);
    int top_y = std::max(old_pos.second - 1, 0);
    int end_x = std::min(old_pos.first + 2, width_);
    int end_y = std::min(old_pos.second + 2, height_);
    for (int attempt = 0; attempt < 100; ++attempt) {
      int x = RandInt(top_x, end_x);
      int y = RandInt(top_y, end_y);
      Pos pos{x, y};
      if (GetCell(x, y).GetType() != kEmpty || pos == agent_pos_) {
        continue;
      }
      PutObj(WorldObj(kBall, kBlue), x, y);
      SetEmpty(old_pos.first, old_pos.second);
      new_pos[i] = pos;
      break;
    }
  }
  obstacle_pos_ = new_pos;
}

void DynamicObstaclesTask::AfterStep(Act act, const WorldObj& pre_fwd,
                                     const Pos& fwd_pos,
                                     const WorldObj& pre_carrying,
                                     float* reward, bool* terminated) {
  if (act == kForward && pre_front_blocked_) {
    *reward = -1.0f;
    *terminated = true;
  }
}

MiniGridDebugState DynamicObstaclesTask::DebugState() const {
  MiniGridDebugState state = MiniGridTask::DebugState();
  state.obstacle_positions.reserve(obstacle_pos_.size() * 2);
  for (const Pos& pos : obstacle_pos_) {
    state.obstacle_positions.push_back(pos.first);
    state.obstacle_positions.push_back(pos.second);
  }
  return state;
}

FetchTask::FetchTask(int size, int num_objs, int max_steps)
    : MiniGridTask("fetch", max_steps, 7, true, 6),
      size_(size),
      num_objs_(num_objs) {}  // NOLINT(whitespace/indent_namespace)

void FetchTask::GenGrid() {
  ClearGrid(size_, size_);
  HorzWall(0, 0);
  HorzWall(0, size_ - 1);
  VertWall(0, 0);
  VertWall(size_ - 1, 0);
  std::vector<std::pair<Type, Color>> objs;
  while (static_cast<int>(objs.size()) < num_objs_) {
    Type type = RandElem(std::vector<Type>{kKey, kBall});
    Color color = RandColor();
    PlaceObj(WorldObj(type, color));
    objs.emplace_back(type, color);
  }
  PlaceAgent();
  int target_idx = RandInt(0, static_cast<int>(objs.size()));
  target_type_ = objs[target_idx].first;
  target_color_ = objs[target_idx].second;
  int syntax_idx = RandInt(0, 5);
  SetMission(MissionFetch(syntax_idx, target_color_, target_type_),
             syntax_idx * 12 + static_cast<int>(target_color_) * 2 +
                 (target_type_ == kBall ? 1 : 0));
}

void FetchTask::AfterStep(Act act, const WorldObj& pre_fwd, const Pos& fwd_pos,
                          const WorldObj& pre_carrying, float* reward,
                          bool* terminated) {
  if (carrying_.GetType() == kEmpty) {
    return;
  }
  *terminated = true;
  if (carrying_.GetType() == target_type_ &&
      carrying_.GetColor() == target_color_) {
    *reward = SuccessReward();
  } else {
    *reward = 0.0f;
  }
}

GoToDoorTask::GoToDoorTask(int size, int max_steps)
    : MiniGridTask("goto_door", max_steps, 7, true, 6), size_(size) {}

void GoToDoorTask::GenGrid() {
  ClearGrid(size_, size_);
  int active_width = RandInt(5, size_ + 1);
  int active_height = RandInt(5, size_ + 1);
  WallRect(0, 0, active_width, active_height);
  std::vector<Pos> door_pos = {
      {RandInt(2, active_width - 2), 0},
      {RandInt(2, active_width - 2), active_height - 1},
      {0, RandInt(2, active_height - 2)},
      {active_width - 1, RandInt(2, active_height - 2)},
  };
  std::vector<Color> door_colors;
  while (static_cast<int>(door_colors.size()) <
         static_cast<int>(door_pos.size())) {
    Color color = RandColor();
    if (std::find(door_colors.begin(), door_colors.end(), color) !=
        door_colors.end()) {
      continue;
    }
    door_colors.push_back(color);
  }
  for (std::size_t i = 0; i < door_pos.size(); ++i) {
    PutObj(MakeDoor(door_colors[i], false, false), door_pos[i].first,
           door_pos[i].second);
  }
  PlaceAgent(0, 0, active_width, active_height, true);
  int door_idx = RandInt(0, static_cast<int>(door_pos.size()));
  target_pos_ = door_pos[door_idx];
  target_type_ = kDoor;
  target_color_ = door_colors[door_idx];
  SetMission(MissionGoToDoor(target_color_), static_cast<int>(target_color_));
}

void GoToDoorTask::AfterStep(Act act, const WorldObj& pre_fwd,
                             const Pos& fwd_pos, const WorldObj& pre_carrying,
                             float* reward, bool* terminated) {
  if (act == kToggle) {
    *terminated = true;
    return;
  }
  if (act == kDone) {
    if (IsAdjacent(agent_pos_, target_pos_)) {
      *reward = SuccessReward();
    }
    *terminated = true;
  }
}

GoToObjectTask::GoToObjectTask(int size, int num_objs, int max_steps)
    : MiniGridTask("goto_object", max_steps, 7, true, 6),
      size_(size),
      num_objs_(num_objs) {}  // NOLINT(whitespace/indent_namespace)

void GoToObjectTask::GenGrid() {
  ClearGrid(size_, size_);
  WallRect(0, 0, size_, size_);
  std::vector<std::pair<Type, Color>> objs;
  std::vector<Pos> positions;
  while (static_cast<int>(objs.size()) < num_objs_) {
    Type type =
        RandElem(std::vector<Type>(kObjectTypes.begin(), kObjectTypes.end()));
    Color color = RandColor();
    if (std::find(objs.begin(), objs.end(),
                  std::pair<Type, Color>{type, color}) != objs.end()) {
      continue;
    }
    positions.push_back(PlaceObj(WorldObj(type, color)));
    objs.emplace_back(type, color);
  }
  PlaceAgent();
  int idx = RandInt(0, static_cast<int>(objs.size()));
  target_pos_ = positions[idx];
  target_type_ = objs[idx].first;
  target_color_ = objs[idx].second;
  SetMission(
      MissionGoToObject(target_color_, target_type_),
      static_cast<int>(target_color_) * 3 + MissionObjectIndex(target_type_));
}

void GoToObjectTask::AfterStep(Act act, const WorldObj& pre_fwd,
                               const Pos& fwd_pos, const WorldObj& pre_carrying,
                               float* reward, bool* terminated) {
  if (act == kToggle) {
    *terminated = true;
    return;
  }
  if (act == kDone) {
    if (IsAdjacent(agent_pos_, target_pos_)) {
      *reward = SuccessReward();
    }
    *terminated = true;
  }
}

PutNearTask::PutNearTask(int size, int num_objs, int max_steps)
    : MiniGridTask("put_near", max_steps, 7, true, 6),
      size_(size),
      num_objs_(num_objs) {}  // NOLINT(whitespace/indent_namespace)

void PutNearTask::GenGrid() {
  ClearGrid(size_, size_);
  HorzWall(0, 0);
  HorzWall(0, size_ - 1);
  VertWall(0, 0);
  VertWall(size_ - 1, 0);
  std::vector<std::pair<Type, Color>> objs;
  std::vector<Pos> positions;
  auto near_existing = [&](const Pos& candidate) {
    return std::any_of(positions.begin(), positions.end(), [&](const Pos& pos) {
      return std::abs(pos.first - candidate.first) <= 1 &&
             std::abs(pos.second - candidate.second) <= 1;
    });
  };
  while (static_cast<int>(objs.size()) < num_objs_) {
    Type type =
        RandElem(std::vector<Type>(kObjectTypes.begin(), kObjectTypes.end()));
    Color color = RandColor();
    if (std::find(objs.begin(), objs.end(),
                  std::pair<Type, Color>{type, color}) != objs.end()) {
      continue;
    }
    positions.push_back(
        PlaceObj(WorldObj(type, color), 0, 0, width_, height_, near_existing));
    objs.emplace_back(type, color);
  }
  PlaceAgent();
  int move_idx = RandInt(0, static_cast<int>(objs.size()));
  move_pos_ = positions[move_idx];
  move_type_ = objs[move_idx].first;
  move_color_ = objs[move_idx].second;
  int target_idx = move_idx;
  while (target_idx == move_idx) {
    target_idx = RandInt(0, static_cast<int>(objs.size()));
  }
  target_pos_ = positions[target_idx];
  target_type_ = objs[target_idx].first;
  target_color_ = objs[target_idx].second;
  SetMission(
      MissionPutNear(move_color_, move_type_, target_color_, target_type_),
      ((static_cast<int>(move_color_) * 3 + MissionObjectIndex(move_type_)) *
       18) +
          (static_cast<int>(target_color_) * 3) +
          MissionObjectIndex(target_type_));
}

void PutNearTask::AfterStep(Act act, const WorldObj& pre_fwd,
                            const Pos& fwd_pos, const WorldObj& pre_carrying,
                            float* reward, bool* terminated) {
  if (act == kPickup && carrying_.GetType() != kEmpty &&
      (carrying_.GetType() != move_type_ ||
       carrying_.GetColor() != move_color_)) {
    *terminated = true;
    return;
  }
  if (act == kDrop && pre_carrying.GetType() != kEmpty) {
    if (GetCell(fwd_pos.first, fwd_pos.second) == pre_carrying &&
        std::abs(fwd_pos.first - target_pos_.first) <= 1 &&
        std::abs(fwd_pos.second - target_pos_.second) <= 1) {
      *reward = SuccessReward();
    }
    *terminated = true;
  }
}

RedBlueDoorTask::RedBlueDoorTask(int size, int max_steps)
    : MiniGridTask("red_blue_door", max_steps, 7, false, 6), size_(size) {}

void RedBlueDoorTask::GenGrid() {
  ClearGrid(2 * size_, size_);
  WallRect(0, 0, 2 * size_, size_);
  WallRect(size_ / 2, 0, size_, size_);
  PlaceAgent(size_ / 2, 0, size_, size_, true);
  red_door_pos_ = {size_ / 2, RandInt(1, size_ - 1)};
  blue_door_pos_ = {size_ / 2 + size_ - 1, RandInt(1, size_ - 1)};
  PutObj(MakeDoor(kRed, false, false), red_door_pos_.first,
         red_door_pos_.second);
  PutObj(MakeDoor(kBlue, false, false), blue_door_pos_.first,
         blue_door_pos_.second);
  SetMission("open the red door then the blue door", 0);
}

void RedBlueDoorTask::BeforeStep(Act act, const WorldObj& pre_fwd) {
  red_open_before_ =
      GetCell(red_door_pos_.first, red_door_pos_.second).GetDoorOpen();
  blue_open_before_ =
      GetCell(blue_door_pos_.first, blue_door_pos_.second).GetDoorOpen();
}

void RedBlueDoorTask::AfterStep(Act act, const WorldObj& pre_fwd,
                                const Pos& fwd_pos,
                                const WorldObj& pre_carrying, float* reward,
                                bool* terminated) {
  bool red_open_after =
      GetCell(red_door_pos_.first, red_door_pos_.second).GetDoorOpen();
  bool blue_open_after =
      GetCell(blue_door_pos_.first, blue_door_pos_.second).GetDoorOpen();
  if (blue_open_after) {
    if (red_open_before_) {
      *reward = SuccessReward();
    }
    *terminated = true;
  } else if (red_open_after && blue_open_before_) {
    *terminated = true;
  }
}

LockedRoomTask::LockedRoomTask(int size, int max_steps)
    : MiniGridTask("locked_room", max_steps, 7, false, 6), size_(size) {}

void LockedRoomTask::GenGrid() {
  ClearGrid(size_, size_);
  for (int i = 0; i < size_; ++i) {
    PutObj(WorldObj(kWall, kGrey), i, 0);
    PutObj(WorldObj(kWall, kGrey), i, size_ - 1);
  }
  for (int j = 0; j < size_; ++j) {
    PutObj(WorldObj(kWall, kGrey), 0, j);
    PutObj(WorldObj(kWall, kGrey), size_ - 1, j);
  }
  int left_wall = size_ / 2 - 2;
  int right_wall = size_ / 2 + 2;
  for (int j = 0; j < size_; ++j) {
    PutObj(WorldObj(kWall, kGrey), left_wall, j);
    PutObj(WorldObj(kWall, kGrey), right_wall, j);
  }
  struct LockedRoomInfo {
    Pos top;
    Pos size;
    Pos door_pos;
    Color color{kRed};
    bool locked{false};
  };
  std::vector<LockedRoomInfo> rooms;
  for (int n = 0; n < 3; ++n) {
    int j = n * (size_ / 3);
    for (int i = 0; i < left_wall; ++i) {
      PutObj(WorldObj(kWall, kGrey), i, j);
    }
    for (int i = right_wall; i < size_; ++i) {
      PutObj(WorldObj(kWall, kGrey), i, j);
    }
    int room_w = left_wall + 1;
    int room_h = size_ / 3 + 1;
    rooms.emplace_back(
        LockedRoomInfo{{0, j}, {room_w, room_h}, {left_wall, j + 3}});
    rooms.emplace_back(
        LockedRoomInfo{{right_wall, j}, {room_w, room_h}, {right_wall, j + 3}});
  }
  int locked_idx = RandInt(0, static_cast<int>(rooms.size()));
  rooms[locked_idx].locked = true;
  goal_pos_ = {
      RandInt(rooms[locked_idx].top.first + 1,
              rooms[locked_idx].top.first + rooms[locked_idx].size.first - 1),
      RandInt(
          rooms[locked_idx].top.second + 1,
          rooms[locked_idx].top.second + rooms[locked_idx].size.second - 1)};
  PutObj(WorldObj(kGoal, kGreen), goal_pos_.first, goal_pos_.second);
  std::vector<Color> colors(kColors.begin(), kColors.end());
  for (auto& room : rooms) {
    int idx = RandInt(0, static_cast<int>(colors.size()));
    room.color = colors[idx];
    colors.erase(colors.begin() + idx);
    PutObj(MakeDoor(room.color, room.locked, false), room.door_pos.first,
           room.door_pos.second);
  }
  int key_room_idx = locked_idx;
  while (key_room_idx == locked_idx) {
    key_room_idx = RandInt(0, static_cast<int>(rooms.size()));
  }
  Pos key_pos{RandInt(rooms[key_room_idx].top.first + 1,
                      rooms[key_room_idx].top.first +
                          rooms[key_room_idx].size.first - 1),
              RandInt(rooms[key_room_idx].top.second + 1,
                      rooms[key_room_idx].top.second +
                          rooms[key_room_idx].size.second - 1)};
  PutObj(WorldObj(kKey, rooms[locked_idx].color), key_pos.first,
         key_pos.second);
  PlaceAgent(left_wall, 0, right_wall - left_wall, size_, true);
  SetMission(
      MissionLockedRoom(rooms[locked_idx].color, rooms[key_room_idx].color), 0);
}

}  // namespace minigrid
