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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "envpool/minigrid/impl/minigrid_env.h"
#include "envpool/minigrid/impl/minigrid_task_utils.h"

namespace minigrid {

MemoryTask::MemoryTask(int size, bool random_length, int max_steps)
    : MiniGridTask("memory", max_steps, 7, false, 6),
      size_(size),
      random_length_(random_length) {}  // NOLINT(whitespace/indent_namespace)

Act MemoryTask::MapAction(Act act) const {
  return act == kPickup ? kToggle : act;
}

void MemoryTask::GenGrid() {
  ClearGrid(size_, size_);
  HorzWall(0, 0);
  HorzWall(0, size_ - 1);
  VertWall(0, 0);
  VertWall(size_ - 1, 0);
  CHECK_EQ(size_ % 2, 1);
  int upper_room_wall = size_ / 2 - 2;
  int lower_room_wall = size_ / 2 + 2;
  int hallway_end = random_length_ ? RandInt(4, size_ - 2) : size_ - 3;
  for (int i = 1; i < 5; ++i) {
    PutObj(WorldObj(kWall, kGrey), i, upper_room_wall);
    PutObj(WorldObj(kWall, kGrey), i, lower_room_wall);
  }
  PutObj(WorldObj(kWall, kGrey), 4, upper_room_wall + 1);
  PutObj(WorldObj(kWall, kGrey), 4, lower_room_wall - 1);
  for (int i = 5; i < hallway_end; ++i) {
    PutObj(WorldObj(kWall, kGrey), i, upper_room_wall + 1);
    PutObj(WorldObj(kWall, kGrey), i, lower_room_wall - 1);
  }
  for (int j = 0; j < size_; ++j) {
    if (j != size_ / 2) {
      PutObj(WorldObj(kWall, kGrey), hallway_end, j);
    }
    PutObj(WorldObj(kWall, kGrey), hallway_end + 2, j);
  }
  agent_pos_ = {RandInt(1, hallway_end + 1), size_ / 2};
  agent_dir_ = 0;
  Type start_type = RandBool() ? kKey : kBall;
  Type other_type = OtherKeyBallType(start_type);
  PutObj(WorldObj(start_type, kGreen), 1, size_ / 2 - 1);
  bool first_matches = RandBool();
  Type first_type = first_matches ? start_type : other_type;
  Type second_type = first_matches ? other_type : start_type;
  Pos pos0{hallway_end + 1, size_ / 2 - 2};
  Pos pos1{hallway_end + 1, size_ / 2 + 2};
  PutObj(WorldObj(first_type, kGreen), pos0.first, pos0.second);
  PutObj(WorldObj(second_type, kGreen), pos1.first, pos1.second);
  if (start_type == first_type) {
    success_pos_ = {pos0.first, pos0.second + 1};
    failure_pos_ = {pos1.first, pos1.second - 1};
  } else {
    success_pos_ = {pos1.first, pos1.second - 1};
    failure_pos_ = {pos0.first, pos0.second + 1};
  }
  SetMission("go to the matching object at the end of the hallway", 0);
}

void MemoryTask::AfterStep(Act act, const WorldObj& pre_fwd, const Pos& fwd_pos,
                           const WorldObj& pre_carrying, float* reward,
                           bool* terminated) {
  if (agent_pos_ == success_pos_) {
    *reward = 1.0f - 0.9f * (static_cast<float>(step_count_) / max_steps_);
    *terminated = true;
  } else if (agent_pos_ == failure_pos_) {
    *reward = 0.0f;
    *terminated = true;
  }
}

MultiRoomTask::MultiRoomTask(int min_num_rooms, int max_num_rooms,
                             int max_room_size, int max_steps)
    : MiniGridTask("multi_room", max_steps, 7, false, 6),
      min_num_rooms_(min_num_rooms),
      max_num_rooms_(max_num_rooms),
      max_room_size_(max_room_size) {}  // NOLINT(whitespace/indent_namespace)

bool MultiRoomTask::PlaceRoom(int num_left, std::vector<MultiRoomDesc>* rooms,
                              int min_size, int max_size, int entry_wall,
                              const Pos& entry_door_pos) {
  int size_x = RandInt(min_size, max_size + 1);
  int size_y = RandInt(min_size, max_size + 1);
  int top_x = 0;
  int top_y = 0;
  if (rooms->empty()) {
    top_x = entry_door_pos.first;
    top_y = entry_door_pos.second;
  } else if (entry_wall == 0) {
    top_x = entry_door_pos.first - size_x + 1;
    top_y =
        RandInt(entry_door_pos.second - size_y + 2, entry_door_pos.second + 1);
  } else if (entry_wall == 1) {
    top_x =
        RandInt(entry_door_pos.first - size_x + 2, entry_door_pos.first + 1);
    top_y = entry_door_pos.second - size_y + 1;
  } else if (entry_wall == 2) {
    top_x = entry_door_pos.first;
    top_y =
        RandInt(entry_door_pos.second - size_y + 2, entry_door_pos.second + 1);
  } else {
    top_x =
        RandInt(entry_door_pos.first - size_x + 2, entry_door_pos.first + 1);
    top_y = entry_door_pos.second;
  }
  if (top_x < 0 || top_y < 0 || top_x + size_x > width_ ||
      top_y + size_y >= height_) {
    return false;
  }
  for (std::size_t idx = 0; idx + 1 < rooms->size(); ++idx) {
    const MultiRoomDesc& room = (*rooms)[idx];
    bool non_overlap = top_x + size_x < room.top.first ||
                       room.top.first + room.size.first <= top_x ||
                       top_y + size_y < room.top.second ||
                       room.top.second + room.size.second <= top_y;
    if (!non_overlap) {
      return false;
    }
  }
  rooms->push_back(
      {{top_x, top_y}, {size_x, size_y}, entry_door_pos, {-1, -1}});
  if (num_left == 1) {
    return true;
  }
  for (int itr = 0; itr < 8; ++itr) {
    std::vector<int> wall_set = {0, 1, 2, 3};
    wall_set.erase(std::remove(wall_set.begin(), wall_set.end(), entry_wall),
                   wall_set.end());
    int exit_wall = RandElem(wall_set);
    int next_entry_wall = (exit_wall + 2) % 4;
    Pos exit_door_pos;
    if (exit_wall == 0) {
      exit_door_pos = {top_x + size_x - 1, top_y + RandInt(1, size_y - 1)};
    } else if (exit_wall == 1) {
      exit_door_pos = {top_x + RandInt(1, size_x - 1), top_y + size_y - 1};
    } else if (exit_wall == 2) {
      exit_door_pos = {top_x, top_y + RandInt(1, size_y - 1)};
    } else {
      exit_door_pos = {top_x + RandInt(1, size_x - 1), top_y};
    }
    if (PlaceRoom(num_left - 1, rooms, min_size, max_size, next_entry_wall,
                  exit_door_pos)) {
      break;
    }
  }
  return true;
}

void MultiRoomTask::GenGrid() {
  ClearGrid(25, 25);
  int num_rooms = RandInt(min_num_rooms_, max_num_rooms_ + 1);
  std::vector<MultiRoomDesc> rooms;
  while (static_cast<int>(rooms.size()) < num_rooms) {
    std::vector<MultiRoomDesc> current;
    Pos entry{RandInt(0, width_ - 2), RandInt(0, width_ - 2)};
    PlaceRoom(num_rooms, &current, 4, max_room_size_, 2, entry);
    if (current.size() > rooms.size()) {
      rooms = current;
    }
  }
  Color prev_door_color = kUnassigned;
  for (std::size_t idx = 0; idx < rooms.size(); ++idx) {
    const auto& room = rooms[idx];
    for (int i = 0; i < room.size.first; ++i) {
      PutObj(WorldObj(kWall, kGrey), room.top.first + i, room.top.second);
      PutObj(WorldObj(kWall, kGrey), room.top.first + i,
             room.top.second + room.size.second - 1);
    }
    for (int j = 0; j < room.size.second; ++j) {
      PutObj(WorldObj(kWall, kGrey), room.top.first, room.top.second + j);
      PutObj(WorldObj(kWall, kGrey), room.top.first + room.size.first - 1,
             room.top.second + j);
    }
    if (idx > 0) {
      std::vector<Color> door_colors(kColors.begin(), kColors.end());
      if (prev_door_color != kUnassigned) {
        door_colors.erase(std::remove(door_colors.begin(), door_colors.end(),
                                      prev_door_color),
                          door_colors.end());
      }
      Color color = RandElem(door_colors);
      PutObj(MakeDoor(color, false, false), room.entry_door_pos.first,
             room.entry_door_pos.second);
      prev_door_color = color;
    }
  }
  PlaceAgent(rooms.front().top.first, rooms.front().top.second,
             rooms.front().size.first, rooms.front().size.second, true);
  goal_pos_ = PlaceObj(WorldObj(kGoal, kGreen), rooms.back().top.first,
                       rooms.back().top.second, rooms.back().size.first,
                       rooms.back().size.second);
  SetMission("traverse the rooms to get to the goal", 0);
}

FourRoomsTask::FourRoomsTask(int max_steps)
    : MiniGridTask("four_rooms", max_steps, 7, false, 6) {}

void FourRoomsTask::GenGrid() {
  ClearGrid(19, 19);
  HorzWall(0, 0);
  HorzWall(0, height_ - 1);
  VertWall(0, 0);
  VertWall(width_ - 1, 0);
  int room_w = width_ / 2;
  int room_h = height_ / 2;
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 2; ++i) {
      int x_l = i * room_w;
      int y_t = j * room_h;
      int x_r = x_l + room_w;
      int y_b = y_t + room_h;
      if (i + 1 < 2) {
        VertWall(x_r, y_t, room_h);
        SetEmpty(x_r, RandInt(y_t + 1, y_b));
      }
      if (j + 1 < 2) {
        HorzWall(x_l, y_b, room_w);
        SetEmpty(RandInt(x_l + 1, x_r), y_b);
      }
    }
  }
  PlaceAgent();
  goal_pos_ = PlaceObj(WorldObj(kGoal, kGreen));
  SetMission("reach the goal", 0);
}

PlaygroundTask::PlaygroundTask(int max_steps)
    : MiniGridTask("playground", max_steps, 7, false, 6) {}

void PlaygroundTask::GenGrid() {
  ClearGrid(19, 19);
  HorzWall(0, 0);
  HorzWall(0, height_ - 1);
  VertWall(0, 0);
  VertWall(width_ - 1, 0);
  int room_w = width_ / 3;
  int room_h = height_ / 3;
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      int x_l = i * room_w;
      int y_t = j * room_h;
      int x_r = x_l + room_w;
      int y_b = y_t + room_h;
      if (i + 1 < 3) {
        VertWall(x_r, y_t, room_h);
        Pos pos{x_r, RandInt(y_t + 1, y_b - 1)};
        PutObj(MakeDoor(RandColor(), false, false), pos.first, pos.second);
      }
      if (j + 1 < 3) {
        HorzWall(x_l, y_b, room_w);
        Pos pos{RandInt(x_l + 1, x_r - 1), y_b};
        PutObj(MakeDoor(RandColor(), false, false), pos.first, pos.second);
      }
    }
  }
  PlaceAgent();
  for (int i = 0; i < 12; ++i) {
    Type type =
        RandElem(std::vector<Type>(kObjectTypes.begin(), kObjectTypes.end()));
    PlaceObj(WorldObj(type, RandColor()));
  }
  SetMission("", 0);
}

UnlockTask::UnlockTask(int max_steps)
    : RoomGridTask("unlock", 6, 1, 2, max_steps, 7) {}

void UnlockTask::GenGrid() {
  RoomGridTask::GenGrid();
  target_pos_ = AddDoor(0, 0, 0, kUnassigned, true);
  target_type_ = kDoor;
  target_color_ = GetCell(target_pos_.first, target_pos_.second).GetColor();
  AddObject(0, 0, kKey, target_color_);
  PlaceAgentInRoom(0, 0);
  SetMission("open the door", 0);
}

void UnlockTask::AfterStep(Act act, const WorldObj& pre_fwd, const Pos& fwd_pos,
                           const WorldObj& pre_carrying, float* reward,
                           bool* terminated) {
  if (act == kToggle &&
      GetCell(target_pos_.first, target_pos_.second).GetDoorOpen()) {
    *reward = 1.0f - 0.9f * (static_cast<float>(step_count_) / max_steps_);
    *terminated = true;
  }
}

UnlockPickupTask::UnlockPickupTask(int max_steps)
    : RoomGridTask("unlock_pickup", 6, 1, 2, max_steps, 7) {}

void UnlockPickupTask::GenGrid() {
  RoomGridTask::GenGrid();
  auto obj = AddObject(1, 0, kBox, kUnassigned);
  target_pos_ = obj.first;
  target_type_ = obj.second.first;
  target_color_ = obj.second.second;
  Pos door_pos = AddDoor(0, 0, 0, kUnassigned, true);
  AddObject(0, 0, kKey, GetCell(door_pos.first, door_pos.second).GetColor());
  PlaceAgentInRoom(0, 0);
  SetMission(MissionPickUp(target_color_, target_type_),
             static_cast<int>(target_color_));
}

void UnlockPickupTask::AfterStep(Act act, const WorldObj& pre_fwd,
                                 const Pos& fwd_pos,
                                 const WorldObj& pre_carrying, float* reward,
                                 bool* terminated) {
  if (act == kPickup && carrying_.GetType() == target_type_ &&
      carrying_.GetColor() == target_color_) {
    *reward = 1.0f - 0.9f * (static_cast<float>(step_count_) / max_steps_);
    *terminated = true;
  }
}

BlockedUnlockPickupTask::BlockedUnlockPickupTask(int max_steps)
    : RoomGridTask("blocked_unlock_pickup", 6, 1, 2, max_steps, 7) {}

void BlockedUnlockPickupTask::GenGrid() {
  RoomGridTask::GenGrid();
  auto obj = AddObject(1, 0, kBox, kUnassigned);
  target_pos_ = obj.first;
  target_type_ = obj.second.first;
  target_color_ = obj.second.second;
  Pos door_pos = AddDoor(0, 0, 0, kUnassigned, true);
  Pos block_pos{door_pos.first - 1, door_pos.second};
  PutObj(WorldObj(kBall, RandColor()), block_pos.first, block_pos.second);
  AddObject(0, 0, kKey, GetCell(door_pos.first, door_pos.second).GetColor());
  PlaceAgentInRoom(0, 0);
  SetMission(MissionPickUp(target_color_, target_type_), 0);
}

void BlockedUnlockPickupTask::AfterStep(Act act, const WorldObj& pre_fwd,
                                        const Pos& fwd_pos,
                                        const WorldObj& pre_carrying,
                                        float* reward, bool* terminated) {
  if (act == kPickup && carrying_.GetType() == target_type_ &&
      carrying_.GetColor() == target_color_) {
    *reward = 1.0f - 0.9f * (static_cast<float>(step_count_) / max_steps_);
    *terminated = true;
  }
}

KeyCorridorTask::KeyCorridorTask(int num_rows, int room_size, Type obj_type,
                                 int max_steps)
    : RoomGridTask("key_corridor", room_size, num_rows, 3, max_steps, 7),
      obj_type_(obj_type) {}  // NOLINT(whitespace/indent_namespace)

void KeyCorridorTask::GenGrid() {
  RoomGridTask::GenGrid();
  for (int j = 1; j < num_rows_; ++j) {
    RemoveWall(1, j, 3);
  }
  int room_idx = RandInt(0, num_rows_);
  Pos door_pos = AddDoor(2, room_idx, 2, kUnassigned, true);
  auto obj = AddObject(2, room_idx, obj_type_, kUnassigned);
  target_pos_ = obj.first;
  target_type_ = obj.second.first;
  target_color_ = obj.second.second;
  AddObject(0, RandInt(0, num_rows_), kKey,
            GetCell(door_pos.first, door_pos.second).GetColor());
  PlaceAgentInRoom(1, num_rows_ / 2);
  ConnectAll();
  SetMission(MissionPickUp(target_color_, target_type_), 0);
}

void KeyCorridorTask::AfterStep(Act act, const WorldObj& pre_fwd,
                                const Pos& fwd_pos,
                                const WorldObj& pre_carrying, float* reward,
                                bool* terminated) {
  if (act == kPickup && carrying_.GetType() == target_type_ &&
      carrying_.GetColor() == target_color_) {
    *reward = 1.0f - 0.9f * (static_cast<float>(step_count_) / max_steps_);
    *terminated = true;
  }
}

ObstructedMazeTask::ObstructedMazeTask(std::string env_name, Pos agent_room,
                                       bool key_in_box, bool blocked,
                                       int num_quarters, int max_steps, bool v1)
    : RoomGridTask(std::move(env_name), 6, 3, 3, max_steps, 7),
      agent_room_(std::move(agent_room)),
      key_in_box_(key_in_box),
      blocked_(blocked),
      num_quarters_(num_quarters),
      v1_(v1) {}  // NOLINT(whitespace/indent_namespace)

void ObstructedMazeTask::AddKeyToRoom(int i, int j, Color color,
                                      bool key_in_box) {
  WorldObj obj(kKey, color);
  if (key_in_box) {
    WorldObj box(kBox, kBlue);
    box.SetContains(std::make_unique<WorldObj>(obj));
    PlaceObj(
        box, GetRoom(i, j).top.first, GetRoom(i, j).top.second,
        GetRoom(i, j).size.first, GetRoom(i, j).size.second,
        [&](const Pos& pos_candidate) {
          return Manhattan(agent_pos_, pos_candidate) < 2;
        },
        1000);
  } else {
    AddObject(i, j, kKey, color);
  }
}

void ObstructedMazeTask::AddObstructedDoor(int i, int j, int door_idx,
                                           Color color, bool locked,
                                           bool key_in_box, bool blocked,
                                           bool add_key) {
  Pos door_pos = AddDoor(i, j, door_idx, color, locked);
  if (blocked) {
    Pos vec = kTaskDirToVec[door_idx];
    PutObj(WorldObj(kBall, kGreen), door_pos.first - vec.first,
           door_pos.second - vec.second);
  }
  if (locked && add_key) {
    AddKeyToRoom(i, j, color, key_in_box);
  }
}

void ObstructedMazeTask::GenGrid() {
  RoomGridTask::GenGrid();
  door_colors_ = RandSubset(std::vector<Color>(kColors.begin(), kColors.end()),
                            static_cast<int>(kColors.size()));
  SetMission("pick up the blue ball", 0);
  if (env_name_ == "obstructed_maze_1dlhb") {
    AddObstructedDoor(0, 0, 0, door_colors_[0], true, key_in_box_, blocked_,
                      true);
    auto obj = AddObject(1, 0, kBall, kBlue);
    target_pos_ = obj.first;
    target_type_ = obj.second.first;
    target_color_ = obj.second.second;
    PlaceAgentInRoom(0, 0);
    return;
  }
  Pos middle_room{1, 1};
  std::vector<Pos> side_rooms = {{2, 1}, {1, 2}, {0, 1}, {1, 0}};
  side_rooms.resize(num_quarters_);
  for (int i = 0; i < static_cast<int>(side_rooms.size()); ++i) {
    Pos side = side_rooms[i];
    AddDoor(middle_room.first, middle_room.second, i, door_colors_[i], false);
    for (int sign : std::vector<int>{-1, 1}) {
      int door_idx = (i + sign + 4) % 4;
      Color color =
          door_colors_[door_idx % static_cast<int>(door_colors_.size())];
      if (v1_) {
        AddObstructedDoor(side.first, side.second, door_idx, color, true,
                          key_in_box_, blocked_, false);
      } else {
        AddObstructedDoor(side.first, side.second, door_idx, color, true,
                          key_in_box_, blocked_, true);
      }
    }
    if (v1_) {
      for (int sign : std::vector<int>{-1, 1}) {
        int door_idx = (i + sign + 4) % 4;
        Color color =
            door_colors_[door_idx % static_cast<int>(door_colors_.size())];
        AddKeyToRoom(side.first, side.second, color, key_in_box_);
      }
    }
  }
  std::vector<Pos> corners = {{2, 0}, {2, 2}, {0, 2}, {0, 0}};
  corners.resize(num_quarters_);
  Pos ball_room = RandElem(corners);
  auto obj = AddObject(ball_room.first, ball_room.second, kBall, kBlue);
  target_pos_ = obj.first;
  target_type_ = obj.second.first;
  target_color_ = obj.second.second;
  PlaceAgentInRoom(agent_room_.first, agent_room_.second);
}

void ObstructedMazeTask::AfterStep(Act act, const WorldObj& pre_fwd,
                                   const Pos& fwd_pos,
                                   const WorldObj& pre_carrying, float* reward,
                                   bool* terminated) {
  if (act == kPickup && carrying_.GetType() == target_type_ &&
      carrying_.GetColor() == target_color_) {
    *reward = 1.0f - 0.9f * (static_cast<float>(step_count_) / max_steps_);
    *terminated = true;
  }
}

}  // namespace minigrid
