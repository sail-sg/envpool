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

#include "envpool/minigrid/impl/babyai_core.h"

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"

namespace minigrid {

BabyAIRejectSampling::BabyAIRejectSampling(const std::string& msg)
    : std::runtime_error(msg) {}

BabyAILevelTask::BabyAILevelTask(std::string env_name, int room_size,
                                 int num_rows, int num_cols, int max_steps,
                                 int mission_bytes)
    : RoomGridTask(std::move(env_name), room_size, num_rows, num_cols,
                   max_steps, 7, mission_bytes),
      fixed_max_steps_(max_steps > 0) {}

std::pair<Pos, std::pair<Type, Color>> BabyAILevelTask::AddExistingObject(
    int i, int j, const WorldObj& obj) {
  const Room& room = GetRoom(i, j);
  Pos pos = PlaceObj(
      obj, room.top.first, room.top.second, room.size.first, room.size.second,
      [&](const Pos& pos_candidate) {
        return std::abs(agent_pos_.first - pos_candidate.first) +
                   std::abs(agent_pos_.second - pos_candidate.second) <
               2;
      },
      1000);
  GetRoom(i, j).objs.emplace_back(obj.GetType(), obj.GetColor());
  return {pos, {obj.GetType(), obj.GetColor()}};
}

std::vector<std::pair<Pos, std::pair<Type, Color>>>
BabyAILevelTask::AddDistractorsOrReject(int i, int j, int num_distractors,
                                        bool all_unique) {
  try {
    return AddDistractors(i, j, num_distractors, all_unique);
  } catch (const std::runtime_error&) {
    throw BabyAIRejectSampling("failed to add distractors");
  }
}

void BabyAILevelTask::ConnectAllOrReject(
    const std::vector<Color>& door_colors) {
  if (!TryConnectAll(door_colors)) {
    throw BabyAIRejectSampling("connect_all failed");
  }
}

void BabyAILevelTask::CheckObjsReachableOrReject() const {
  if (!CheckObjsReachable()) {
    throw BabyAIRejectSampling("unreachable object");
  }
}

void BabyAILevelTask::OpenAllDoors() {
  for (int i = 0; i < num_cols_; ++i) {
    for (int j = 0; j < num_rows_; ++j) {
      const Room& room = GetRoom(i, j);
      for (int door_idx = 0; door_idx < 4; ++door_idx) {
        if (!room.connected[door_idx]) {
          continue;
        }
        const Pos pos = room.door_pos[door_idx];
        WorldObj& obj = Cell(pos.first, pos.second);
        if (obj.GetType() == kDoor) {
          obj.SetDoorLocked(false);
          obj.SetDoorOpen(true);
        }
      }
    }
  }
}

void BabyAILevelTask::GenGrid() {
  std::string last_error = "unknown";
  for (int retry = 0; retry < 1000; ++retry) {
    RoomGridTask::GenGrid();
    locked_room_ = nullptr;
    instrs_.reset();
    try {
      GenMission();
      CHECK(instrs_ != nullptr);
      std::vector<Color> locked_colors;
      for (int i = 0; i < num_cols_; ++i) {
        for (int j = 0; j < num_rows_; ++j) {
          const Room& room = GetRoom(i, j);
          for (int door_idx = 0; door_idx < 4; ++door_idx) {
            if (!room.connected[door_idx]) {
              continue;
            }
            const Pos pos = room.door_pos[door_idx];
            const WorldObj door = GetCell(pos.first, pos.second);
            if (door.GetType() == kDoor && door.GetDoorLocked()) {
              locked_colors.push_back(door.GetColor());
            }
          }
        }
      }
      instrs_->Validate(*this, locked_colors, UnblockingEnabled());
      SetMission(instrs_->Surface(*this));
      instrs_->ResetVerifier(*this);
      if (!fixed_max_steps_) {
        max_steps_ = instrs_->NumNavsNeeded() * room_size_ * room_size_ *
                     num_rows_ * num_cols_;
      }
      AfterResetVerifier();
      return;
    } catch (const BabyAIRejectSampling& e) {
      last_error = e.what();
    } catch (const std::runtime_error& e) {
      last_error = e.what();
    }
  }
  throw std::runtime_error("BabyAI mission generation failed for " + env_name_ +
                           ": " + last_error);
}

void BabyAILevelTask::AfterStep(Act act, const WorldObj& /*pre_fwd*/,
                                const Pos& /*fwd_pos*/,
                                const WorldObj& pre_carrying, float* reward,
                                bool* terminated) {
  if (act == kDrop) {
    instrs_->UpdateObjPoss(*this);
  }
  BabyAIStatus status = instrs_->Verify(*this, act, pre_carrying);
  if (status == BabyAIStatus::kSuccess) {
    *reward = SuccessReward();
    *terminated = true;
  } else if (status == BabyAIStatus::kFailure) {
    *reward = 0.0f;
    *terminated = true;
  }
}

void BabyAILevelTask::AddLockedRoom() {
  while (true) {
    int i = RandInt(0, num_cols_);
    int j = RandInt(0, num_rows_);
    int door_idx = RandInt(0, 4);
    Room& room = GetRoom(i, j);
    if (!room.has_neighbor[door_idx]) {
      continue;
    }
    locked_room_ = &room;
    Pos door_pos = AddDoor(i, j, door_idx, kUnassigned, true);
    Color door_color = GetCell(door_pos.first, door_pos.second).GetColor();
    while (true) {
      int key_i = RandInt(0, num_cols_);
      int key_j = RandInt(0, num_rows_);
      if (key_i == i && key_j == j) {
        continue;
      }
      AddObject(key_i, key_j, kKey, door_color);
      return;
    }
  }
}

BabyAILevelGenTask::BabyAILevelGenTask(const BabyAITaskConfig& config)
    : BabyAILevelTask(config.env_name, config.room_size, config.num_rows,
                      config.num_cols, config.max_steps, config.mission_bytes),
      num_dists_(config.num_dists),
      locked_room_prob_(config.locked_room_prob),
      locations_(config.locations),
      unblocking_(config.unblocking),
      implicit_unlock_(config.implicit_unlock) {
  action_kinds_ = SplitKinds(config.action_kinds);
  instr_kinds_ = SplitKinds(config.instr_kinds);
}

std::vector<std::string> BabyAILevelGenTask::SplitKinds(
    const std::string& csv) const {
  std::vector<std::string> kinds;
  std::string cur;
  for (char ch : csv) {
    if (ch == ',') {
      if (!cur.empty()) {
        kinds.push_back(cur);
        cur.clear();
      }
      continue;
    }
    cur.push_back(ch);
  }
  if (!cur.empty()) {
    kinds.push_back(cur);
  }
  return kinds;
}

BabyAIObjDesc BabyAILevelGenTask::RandObj(const std::vector<Type>& types,
                                          const std::vector<Color>& colors,
                                          int max_tries) {
  for (int num_tries = 0; num_tries <= max_tries; ++num_tries) {
    std::optional<Color> color;
    if (RandInt(0, static_cast<int>(colors.size()) + 1) > 0) {
      color = RandElem(colors);
    }
    std::optional<Type> type = RandElem(types);
    BabyAILoc loc = BabyAILoc::kNone;
    if (locations_ && RandBool()) {
      loc = RandElem(std::vector<BabyAILoc>{BabyAILoc::kLeft, BabyAILoc::kRight,
                                            BabyAILoc::kFront,
                                            BabyAILoc::kBehind});
    }
    BabyAIObjDesc desc(type, color, loc);
    desc.FindMatchingObjs(*this);
    if (desc.ObjUids().empty()) {
      continue;
    }
    if (!implicit_unlock_ && locked_room_ != nullptr) {
      bool has_unlocked_match = false;
      for (const Pos& pos : desc.ObjPoss()) {
        if (!locked_room_->PosInside(pos.first, pos.second)) {
          has_unlocked_match = true;
          break;
        }
      }
      if (!has_unlocked_match) {
        continue;
      }
    }
    return desc;
  }
  throw BabyAIRejectSampling("failed to find suitable object");
}

std::unique_ptr<BabyAIInstr> BabyAILevelGenTask::RandActionInstr(
    const std::vector<std::string>& action_kinds) {
  const std::string& action = RandElem(action_kinds);
  if (action == "goto") {
    return std::make_unique<BabyAIGoToInstr>(RandObj());
  }
  if (action == "pickup") {
    return std::make_unique<BabyAIPickupInstr>(
        RandObj(std::vector<Type>{kKey, kBall, kBox}));
  }
  if (action == "open") {
    return std::make_unique<BabyAIOpenInstr>(RandObj(std::vector<Type>{kDoor}));
  }
  if (action == "putnext") {
    return std::make_unique<BabyAIPutNextInstr>(
        RandObj(std::vector<Type>{kKey, kBall, kBox}), RandObj());
  }
  LOG(FATAL) << "Unknown BabyAI action kind: " << action;
  return nullptr;
}

std::unique_ptr<BabyAIInstr> BabyAILevelGenTask::RandInstr(
    const std::vector<std::string>& action_kinds,
    const std::vector<std::string>& instr_kinds, int depth) {
  const std::string& kind = RandElem(instr_kinds);
  if (kind == "action") {
    return RandActionInstr(action_kinds);
  }
  if (kind == "and") {
    return std::make_unique<BabyAIAndInstr>(
        RandInstr(action_kinds, std::vector<std::string>{"action"}, depth + 1),
        RandInstr(action_kinds, std::vector<std::string>{"action"}, depth + 1));
  }
  if (kind == "seq") {
    auto instr_a = RandInstr(
        action_kinds, std::vector<std::string>{"action", "and"}, depth + 1);
    auto instr_b = RandInstr(
        action_kinds, std::vector<std::string>{"action", "and"}, depth + 1);
    if (RandBool()) {
      return std::make_unique<BabyAIBeforeInstr>(std::move(instr_a),
                                                 std::move(instr_b));
    }
    return std::make_unique<BabyAIAfterInstr>(std::move(instr_a),
                                              std::move(instr_b));
  }
  LOG(FATAL) << "Unknown BabyAI instruction kind: " << kind;
  return nullptr;
}

void BabyAILevelGenTask::GenMission() {
  if (RandFloat(0.0f, 1.0f) < locked_room_prob_) {
    AddLockedRoom();
  }
  ConnectAllOrReject();
  AddDistractorsOrReject(-1, -1, num_dists_, false);
  while (true) {
    PlaceAgentInRoom();
    if (locked_room_ == nullptr ||
        &RoomFromPos(agent_pos_.first, agent_pos_.second) != locked_room_) {
      break;
    }
  }
  if (!unblocking_) {
    CheckObjsReachableOrReject();
  }
  instrs_ = RandInstr(action_kinds_, instr_kinds_);
}

std::pair<Pos, std::pair<Type, Color>> ObjAtDoor(const BabyAILevelTask& env,
                                                 const Pos& pos) {
  const WorldObj door = env.CellAt(pos.first, pos.second);
  return {pos, {door.GetType(), door.GetColor()}};
}

BabyAITaskConfig MakeGoToSeqConfig(BabyAITaskConfig config) {
  config.action_kinds = "goto";
  config.locked_room_prob = 0.0f;
  config.locations = false;
  config.unblocking = false;
  return config;
}

BabyAITaskConfig MakePickupLocConfig(BabyAITaskConfig config) {
  config.action_kinds = "pickup";
  config.instr_kinds = "action";
  config.num_rows = 1;
  config.num_cols = 1;
  config.num_dists = 8;
  config.locked_room_prob = 0.0f;
  config.locations = true;
  config.unblocking = false;
  return config;
}

BabyAITaskConfig MakeSynthConfig(BabyAITaskConfig config) {
  config.instr_kinds = "action";
  config.locations = false;
  config.unblocking = true;
  config.implicit_unlock = false;
  return config;
}

BabyAITaskConfig MakeSynthLocConfig(BabyAITaskConfig config) {
  config.instr_kinds = "action";
  config.locations = true;
  config.unblocking = true;
  config.implicit_unlock = false;
  return config;
}

BabyAITaskConfig MakeSynthSeqConfig(BabyAITaskConfig config) {
  config.locations = true;
  config.unblocking = true;
  config.implicit_unlock = false;
  return config;
}

BabyAITaskConfig MakeMiniBossConfig(BabyAITaskConfig config) {
  config.num_cols = 2;
  config.num_rows = 2;
  config.room_size = 5;
  config.num_dists = 7;
  config.locked_room_prob = 0.25f;
  return config;
}

BabyAITaskConfig MakeBossNoUnlockConfig(BabyAITaskConfig config) {
  config.locked_room_prob = 0.0f;
  config.implicit_unlock = false;
  return config;
}

}  // namespace minigrid
