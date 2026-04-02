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
#include <cmath>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "envpool/minigrid/impl/minigrid_env.h"

namespace minigrid {
namespace {

constexpr std::array<Pos, 4> kDirToVec = {
    Pos{1, 0}, Pos{0, 1}, Pos{-1, 0},
    Pos{0, -1}};  // NOLINT(whitespace/indent_namespace)

inline int Manhattan(const Pos& lhs, const Pos& rhs) {
  return std::abs(lhs.first - rhs.first) + std::abs(lhs.second - rhs.second);
}

}  // namespace

MiniGridTask::MiniGridTask(std::string env_name, int max_steps,
                           int agent_view_size, bool see_through_walls,
                           int action_max, int mission_bytes)
    : max_steps_(max_steps),
      action_max_(action_max),
      agent_view_size_(agent_view_size),
      mission_bytes_(mission_bytes),
      see_through_walls_(see_through_walls),
      env_name_(std::move(env_name)),
      carrying_(kEmpty) {}  // NOLINT(whitespace/indent_namespace)

void MiniGridTask::Reset() {
  CHECK(gen_ref_ != nullptr);
  step_count_ = 0;
  done_ = false;
  next_uid_ = 1;
  carrying_ = WorldObj(kEmpty);
  target_pos_ = {-1, -1};
  target_type_ = kEmpty;
  target_color_ = kRed;
  move_pos_ = {-1, -1};
  move_type_ = kEmpty;
  move_color_ = kRed;
  success_pos_ = {-1, -1};
  failure_pos_ = {-1, -1};
  goal_pos_ = {-1, -1};
  mission_.clear();
  mission_id_ = -1;
  GenGrid();
  CHECK_GE(agent_pos_.first, 0);
  CHECK_GE(agent_pos_.second, 0);
  CHECK_GE(agent_dir_, 0);
  CHECK_LT(agent_dir_, 4);
  CHECK(GetCell(agent_pos_.first, agent_pos_.second).CanOverlap());
}

float MiniGridTask::Step(Act act) {
  act = MapAction(act);
  step_count_ += 1;
  float reward = 0.0f;
  bool terminated = false;

  const Pos dir = DirVec();
  const Pos fwd_pos = {agent_pos_.first + dir.first,
                       agent_pos_.second + dir.second};
  CHECK(InBounds(fwd_pos.first, fwd_pos.second));
  const WorldObj pre_fwd = GetCell(fwd_pos.first, fwd_pos.second);
  const WorldObj pre_carrying = carrying_;
  BeforeStep(act, pre_fwd);

  if (act == kLeft) {
    agent_dir_ = (agent_dir_ + 3) % 4;
  } else if (act == kRight) {
    agent_dir_ = (agent_dir_ + 1) % 4;
  } else if (act == kForward) {
    const WorldObj cur_fwd = GetCell(fwd_pos.first, fwd_pos.second);
    if (cur_fwd.CanOverlap()) {
      agent_pos_ = fwd_pos;
    }
    if (cur_fwd.GetType() == kGoal) {
      reward = SuccessReward();
      terminated = true;
    } else if (cur_fwd.GetType() == kLava) {
      terminated = true;
    }
  } else if (act == kPickup) {
    const WorldObj cur_fwd = GetCell(fwd_pos.first, fwd_pos.second);
    if (carrying_.GetType() == kEmpty && cur_fwd.CanPickup()) {
      carrying_ = cur_fwd;
      SetEmpty(fwd_pos.first, fwd_pos.second);
    }
  } else if (act == kDrop) {
    const WorldObj cur_fwd = GetCell(fwd_pos.first, fwd_pos.second);
    if (carrying_.GetType() != kEmpty && cur_fwd.GetType() == kEmpty) {
      SetCell(fwd_pos.first, fwd_pos.second, carrying_);
      carrying_ = WorldObj(kEmpty);
    }
  } else if (act == kToggle) {
    WorldObj& cur_fwd = Cell(fwd_pos.first, fwd_pos.second);
    if (cur_fwd.GetType() == kDoor) {
      if (cur_fwd.GetDoorLocked()) {
        if (carrying_.GetType() == kKey &&
            carrying_.GetColor() == cur_fwd.GetColor()) {
          cur_fwd.SetDoorLocked(false);
          cur_fwd.SetDoorOpen(true);
        }
      } else {
        cur_fwd.SetDoorOpen(!cur_fwd.GetDoorOpen());
      }
    } else if (cur_fwd.GetType() == kBox) {
      auto contains = cur_fwd.ReleaseContains();
      if (contains != nullptr) {
        cur_fwd = *contains;
      } else {
        cur_fwd = WorldObj(kEmpty);
      }
    }
  } else {
    CHECK_EQ(act, kDone);
  }

  AfterStep(act, pre_fwd, fwd_pos, pre_carrying, &reward, &terminated);
  if (step_count_ >= max_steps_) {
    terminated = true;
  }
  done_ = terminated;
  return reward;
}

bool MiniGridTask::InBounds(int x, int y) const {
  return x >= 0 && x < width_ && y >= 0 && y < height_;
}

WorldObj MiniGridTask::GetCell(int x, int y) const {
  CHECK(InBounds(x, y));
  return grid_[y][x];
}

WorldObj& MiniGridTask::Cell(int x, int y) {
  CHECK(InBounds(x, y));
  return grid_[y][x];
}

void MiniGridTask::SetCell(int x, int y, const WorldObj& obj) {
  CHECK(InBounds(x, y));
  grid_[y][x] = obj;
}

void MiniGridTask::SetEmpty(int x, int y) { SetCell(x, y, WorldObj(kEmpty)); }

void MiniGridTask::ClearGrid(int width, int height) {
  width_ = width;
  height_ = height;
  grid_.assign(height_, std::vector<WorldObj>(width_, WorldObj(kEmpty)));
}

void MiniGridTask::HorzWall(int x, int y, int length, Type type, Color color) {
  if (length < 0) {
    length = width_ - x;
  }
  for (int i = 0; i < length; ++i) {
    SetCell(x + i, y, WorldObj(type, color));
  }
}

void MiniGridTask::VertWall(int x, int y, int length, Type type, Color color) {
  if (length < 0) {
    length = height_ - y;
  }
  for (int j = 0; j < length; ++j) {
    SetCell(x, y + j, WorldObj(type, color));
  }
}

void MiniGridTask::WallRect(int x, int y, int width, int height) {
  HorzWall(x, y, width);
  HorzWall(x, y + height - 1, width);
  VertWall(x, y, height);
  VertWall(x + width - 1, y, height);
}

void MiniGridTask::PutObj(const WorldObj& obj, int x, int y) {
  SetCell(x, y, PrepareObj(obj));
}

WorldObj MiniGridTask::PrepareObj(const WorldObj& obj) {
  WorldObj copy = obj;
  if (copy.GetType() != kEmpty && copy.GetUid() < 0) {
    copy.SetUid(next_uid_++);
  }
  if (WorldObj* contains = copy.GetContains(); contains != nullptr &&
                                               contains->GetType() != kEmpty &&
                                               contains->GetUid() < 0) {
    contains->SetUid(next_uid_++);
  }
  return copy;
}

Pos MiniGridTask::PlaceObj(const WorldObj& obj, int top_x, int top_y,
                           int size_x, int size_y, const RejectFn& reject_fn,
                           int max_tries) {
  if (size_x < 0) {
    size_x = width_;
  }
  if (size_y < 0) {
    size_y = height_;
  }
  top_x = std::max(top_x, 0);
  top_y = std::max(top_y, 0);

  int num_tries = 0;
  while (true) {
    if (num_tries > max_tries) {
      throw std::runtime_error("rejection sampling failed");
    }
    ++num_tries;
    int x = RandInt(top_x, std::min(top_x + size_x, width_));
    int y = RandInt(top_y, std::min(top_y + size_y, height_));
    Pos pos{x, y};
    if (GetCell(x, y).GetType() != kEmpty) {
      continue;
    }
    if (pos == agent_pos_) {
      continue;
    }
    if (reject_fn && reject_fn(pos)) {
      continue;
    }
    PutObj(obj, x, y);
    return pos;
  }
}

Pos MiniGridTask::PlaceAgent(int top_x, int top_y, int size_x, int size_y,
                             bool rand_dir, int max_tries) {
  agent_pos_ = {-1, -1};
  Pos pos = PlaceObj(WorldObj(kEmpty), top_x, top_y, size_x, size_y, RejectFn(),
                     max_tries);
  agent_pos_ = pos;
  if (rand_dir) {
    agent_dir_ = RandInt(0, 4);
  }
  return pos;
}

int MiniGridTask::RandInt(int low, int high) {
  CHECK_LT(low, high);
  std::uniform_int_distribution<> dist(low, high - 1);
  return dist(*gen_ref_);
}

bool MiniGridTask::RandBool() { return RandInt(0, 2) == 0; }

float MiniGridTask::RandFloat(float low, float high) {
  CHECK_LT(low, high);
  std::uniform_real_distribution<float> dist(low, high);
  return dist(*gen_ref_);
}

Pos MiniGridTask::FrontPos() const {
  const Pos dir = DirVec();
  return {agent_pos_.first + dir.first, agent_pos_.second + dir.second};
}

Pos MiniGridTask::DirVec() const { return kDirToVec[agent_dir_]; }

Pos MiniGridTask::RightVec() const {
  const Pos dir = DirVec();
  return {-dir.second, dir.first};
}

float MiniGridTask::SuccessReward() const {
  return 1.0f - 0.9f * (static_cast<float>(step_count_) / max_steps_);
}

RoomGridTask::RoomGridTask(std::string env_name, int room_size, int num_rows,
                           int num_cols, int max_steps, int agent_view_size,
                           int mission_bytes)
    : MiniGridTask(std::move(env_name), max_steps, agent_view_size, false, 6,
                   mission_bytes),
      room_size_(room_size),
      num_rows_(num_rows),
      num_cols_(num_cols) {}  // NOLINT(whitespace/indent_namespace)

void RoomGridTask::GenGrid() {
  int height = (room_size_ - 1) * num_rows_ + 1;
  int width = (room_size_ - 1) * num_cols_ + 1;
  ClearGrid(width, height);
  room_grid_.assign(num_rows_, std::vector<Room>(num_cols_));
  for (int j = 0; j < num_rows_; ++j) {
    for (int i = 0; i < num_cols_; ++i) {
      Room& room = room_grid_[j][i];
      room.top = {i * (room_size_ - 1), j * (room_size_ - 1)};
      room.size = {room_size_, room_size_};
      WallRect(room.top.first, room.top.second, room.size.first,
               room.size.second);
    }
  }

  for (int j = 0; j < num_rows_; ++j) {
    for (int i = 0; i < num_cols_; ++i) {
      Room& room = room_grid_[j][i];
      int x_l = room.top.first + 1;
      int y_l = room.top.second + 1;
      int x_m = room.top.first + room.size.first - 1;
      int y_m = room.top.second + room.size.second - 1;
      if (i < num_cols_ - 1) {
        room.has_neighbor[0] = true;
        room.neighbors[0] = {i + 1, j};
        room.door_pos[0] = {x_m, RandInt(y_l, y_m)};
      }
      if (j < num_rows_ - 1) {
        room.has_neighbor[1] = true;
        room.neighbors[1] = {i, j + 1};
        room.door_pos[1] = {RandInt(x_l, x_m), y_m};
      }
      if (i > 0) {
        room.has_neighbor[2] = true;
        room.neighbors[2] = {i - 1, j};
        room.door_pos[2] = room_grid_[j][i - 1].door_pos[0];
      }
      if (j > 0) {
        room.has_neighbor[3] = true;
        room.neighbors[3] = {i, j - 1};
        room.door_pos[3] = room_grid_[j - 1][i].door_pos[1];
      }
    }
  }
  agent_pos_ = {(num_cols_ / 2) * (room_size_ - 1) + room_size_ / 2,
                (num_rows_ / 2) * (room_size_ - 1) + room_size_ / 2};
  agent_dir_ = 0;
}

Room& RoomGridTask::GetRoom(int i, int j) {
  CHECK_GE(i, 0);
  CHECK_LT(i, num_cols_);
  CHECK_GE(j, 0);
  CHECK_LT(j, num_rows_);
  return room_grid_[j][i];
}

const Room& RoomGridTask::GetRoom(int i, int j) const {
  CHECK_GE(i, 0);
  CHECK_LT(i, num_cols_);
  CHECK_GE(j, 0);
  CHECK_LT(j, num_rows_);
  return room_grid_[j][i];
}

Room& RoomGridTask::RoomFromPos(int x, int y) {
  return GetRoom(x / (room_size_ - 1), y / (room_size_ - 1));
}

const Room& RoomGridTask::RoomFromPos(int x, int y) const {
  return GetRoom(x / (room_size_ - 1), y / (room_size_ - 1));
}

std::pair<Pos, std::pair<Type, Color>> RoomGridTask::AddObject(int i, int j,
                                                               Type type,
                                                               Color color) {
  if (type == kEmpty) {
    type =
        RandElem(std::vector<Type>(kObjectTypes.begin(), kObjectTypes.end()));
  }
  if (color == kUnassigned) {
    color = RandColor();
  }
  Pos pos = PlaceObj(
      WorldObj(type, color), GetRoom(i, j).top.first, GetRoom(i, j).top.second,
      GetRoom(i, j).size.first, GetRoom(i, j).size.second,
      [&](const Pos& pos_candidate) {
        return Manhattan(agent_pos_, pos_candidate) < 2;
      },
      1000);
  GetRoom(i, j).objs.emplace_back(type, color);
  return {pos, {type, color}};
}

std::vector<std::pair<Pos, std::pair<Type, Color>>>
RoomGridTask::AddDistractors(int i, int j, int num_distractors,
                             bool all_unique) {
  std::vector<std::pair<Type, Color>> objs;
  for (const auto& row : room_grid_) {
    for (const Room& room : row) {
      objs.insert(objs.end(), room.objs.begin(), room.objs.end());
    }
  }
  std::vector<std::pair<Pos, std::pair<Type, Color>>> dists;
  dists.reserve(num_distractors);
  while (static_cast<int>(dists.size()) < num_distractors) {
    Type type =
        RandElem(std::vector<Type>(kObjectTypes.begin(), kObjectTypes.end()));
    Color color = RandColor();
    std::pair<Type, Color> obj = {type, color};
    if (all_unique && std::find(objs.begin(), objs.end(), obj) != objs.end()) {
      continue;
    }
    int room_i = i >= 0 ? i : RandInt(0, num_cols_);
    int room_j = j >= 0 ? j : RandInt(0, num_rows_);
    dists.push_back(AddObject(room_i, room_j, type, color));
    objs.push_back(obj);
  }
  return dists;
}

Pos RoomGridTask::AddDoor(int i, int j, int door_idx, Color color,
                          bool locked) {
  Room& room = GetRoom(i, j);
  if (door_idx < 0) {
    while (true) {
      door_idx = RandInt(0, 4);
      if (room.has_neighbor[door_idx] && !room.connected[door_idx]) {
        break;
      }
    }
  }
  if (color == kUnassigned) {
    color = RandColor();
  }
  room.locked = locked;
  Pos pos = room.door_pos[door_idx];
  PutObj(MakeDoor(color, locked, false), pos.first, pos.second);
  room.connected[door_idx] = true;
  const Pos neighbor = room.neighbors[door_idx];
  GetRoom(neighbor.first, neighbor.second).connected[(door_idx + 2) % 4] = true;
  return pos;
}

void RoomGridTask::RemoveWall(int i, int j, int wall_idx) {
  Room& room = GetRoom(i, j);
  CHECK(room.has_neighbor[wall_idx]);
  CHECK(!room.connected[wall_idx]);
  int tx = room.top.first;
  int ty = room.top.second;
  int w = room.size.first;
  int h = room.size.second;
  if (wall_idx == 0) {
    for (int k = 1; k < h - 1; ++k) {
      SetEmpty(tx + w - 1, ty + k);
    }
  } else if (wall_idx == 1) {
    for (int k = 1; k < w - 1; ++k) {
      SetEmpty(tx + k, ty + h - 1);
    }
  } else if (wall_idx == 2) {
    for (int k = 1; k < h - 1; ++k) {
      SetEmpty(tx, ty + k);
    }
  } else {
    for (int k = 1; k < w - 1; ++k) {
      SetEmpty(tx + k, ty);
    }
  }
  room.connected[wall_idx] = true;
  const Pos neighbor = room.neighbors[wall_idx];
  GetRoom(neighbor.first, neighbor.second).connected[(wall_idx + 2) % 4] = true;
}

Pos RoomGridTask::PlaceAgentInRoom(int i, int j, bool rand_dir) {
  if (i < 0) {
    i = RandInt(0, num_cols_);
  }
  if (j < 0) {
    j = RandInt(0, num_rows_);
  }
  Room& room = GetRoom(i, j);
  while (true) {
    Pos pos = PlaceAgent(room.top.first, room.top.second, room.size.first,
                         room.size.second, rand_dir, 1000);
    const Pos dir = DirVec();
    Pos front{pos.first + dir.first, pos.second + dir.second};
    if (InBounds(front.first, front.second) &&
        (GetCell(front.first, front.second).GetType() == kEmpty ||
         GetCell(front.first, front.second).GetType() == kWall)) {
      return pos;
    }
  }
}

bool RoomGridTask::TryConnectAll(const std::vector<Color>& door_colors,
                                 int max_itrs) {
  CHECK(!door_colors.empty());
  auto reachable = [&]() {
    std::vector<std::vector<bool>> seen(num_rows_,
                                        std::vector<bool>(num_cols_, false));
    std::queue<Pos> queue;
    Pos start_room = {agent_pos_.first / (room_size_ - 1),
                      agent_pos_.second / (room_size_ - 1)};
    queue.push(start_room);
    while (!queue.empty()) {
      Pos cur = queue.front();
      queue.pop();
      if (seen[cur.second][cur.first]) {
        continue;
      }
      seen[cur.second][cur.first] = true;
      const Room& room = GetRoom(cur.first, cur.second);
      for (int k = 0; k < 4; ++k) {
        if (room.connected[k] && room.has_neighbor[k]) {
          queue.push(room.neighbors[k]);
        }
      }
    }
    return seen;
  };

  for (int itr = 0; itr <= max_itrs; ++itr) {
    auto seen = reachable();
    int count = 0;
    for (const auto& row : seen) {
      count += std::count(row.begin(), row.end(), true);
    }
    if (count == num_rows_ * num_cols_) {
      return true;
    }
    int i = RandInt(0, num_cols_);
    int j = RandInt(0, num_rows_);
    int k = RandInt(0, 4);
    Room& room = GetRoom(i, j);
    if (!room.has_neighbor[k] || room.connected[k]) {
      continue;
    }
    const Pos nb = room.neighbors[k];
    if (room.locked || GetRoom(nb.first, nb.second).locked) {
      continue;
    }
    AddDoor(i, j, k, RandElem(door_colors), false);
  }
  return false;
}

void RoomGridTask::ConnectAll() {
  CHECK(TryConnectAll()) << "connect_all failed";
}

bool RoomGridTask::CheckObjsReachable() const {
  std::set<Pos> reachable;
  std::vector<Pos> stack = {agent_pos_};
  while (!stack.empty()) {
    Pos pos = stack.back();
    stack.pop_back();
    if (!InBounds(pos.first, pos.second) ||
        reachable.find(pos) != reachable.end()) {
      continue;
    }
    reachable.insert(pos);
    const WorldObj cell = GetCell(pos.first, pos.second);
    if (cell.GetType() != kEmpty && cell.GetType() != kDoor) {
      continue;
    }
    stack.emplace_back(pos.first + 1, pos.second);
    stack.emplace_back(pos.first - 1, pos.second);
    stack.emplace_back(pos.first, pos.second + 1);
    stack.emplace_back(pos.first, pos.second - 1);
  }
  for (int x = 0; x < width_; ++x) {
    for (int y = 0; y < height_; ++y) {
      Type type = GetCell(x, y).GetType();
      if ((type != kEmpty && type != kWall) &&
          reachable.find({x, y}) == reachable.end()) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace minigrid
