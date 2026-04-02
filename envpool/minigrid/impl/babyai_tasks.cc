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

namespace minigrid {
namespace {

class BabyAIGoToRedBallTask : public BabyAILevelTask {
 public:
  BabyAIGoToRedBallTask(const BabyAITaskConfig& config, bool grey,
                        int num_dists)
      : BabyAILevelTask(config.env_name, config.room_size, 1, 1,
                        config.max_steps, config.mission_bytes),
        grey_(grey),
        num_dists_(num_dists) {}

 protected:
  void GenMission() override {
    PlaceAgentInRoom();
    auto obj = AddObject(0, 0, kBall, kRed);
    auto dists = AddDistractorsOrReject(-1, -1, num_dists_, false);
    if (grey_) {
      for (const auto& dist : dists) {
        PutObj(WorldObj(dist.second.first, kGrey), dist.first.first,
               dist.first.second);
      }
    }
    CheckObjsReachableOrReject();
    instrs_ = std::make_unique<BabyAIGoToInstr>(
        BabyAIObjDesc(obj.second.first, obj.second.second));
  }

 private:
  bool grey_{false};
  int num_dists_{0};
};

class BabyAIGoToObjTask : public BabyAILevelTask {
 public:
  explicit BabyAIGoToObjTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, config.room_size, 1, 1,
                        config.max_steps, config.mission_bytes) {}

 protected:
  void GenMission() override {
    PlaceAgentInRoom();
    auto objs = AddDistractorsOrReject(-1, -1, 1, true);
    instrs_ = std::make_unique<BabyAIGoToInstr>(
        BabyAIObjDesc(objs[0].second.first, objs[0].second.second));
  }
};

class BabyAIGoToLocalTask : public BabyAILevelTask {
 public:
  explicit BabyAIGoToLocalTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, config.room_size, 1, 1,
                        config.max_steps, config.mission_bytes),
        num_dists_(config.num_dists) {}

 protected:
  void GenMission() override {
    PlaceAgentInRoom();
    auto objs = AddDistractorsOrReject(-1, -1, num_dists_, false);
    CheckObjsReachableOrReject();
    const auto& obj = RandElem(objs);
    instrs_ = std::make_unique<BabyAIGoToInstr>(
        BabyAIObjDesc(obj.second.first, obj.second.second));
  }

 private:
  int num_dists_{8};
};

class BabyAIGoToTask : public BabyAILevelTask {
 public:
  explicit BabyAIGoToTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, config.room_size, config.num_rows,
                        config.num_cols, config.max_steps,
                        config.mission_bytes),
        num_dists_(config.num_dists),
        doors_open_(config.doors_open) {}

 protected:
  void GenMission() override {
    PlaceAgentInRoom();
    ConnectAllOrReject();
    auto objs = AddDistractorsOrReject(-1, -1, num_dists_, false);
    CheckObjsReachableOrReject();
    const auto& obj = RandElem(objs);
    instrs_ = std::make_unique<BabyAIGoToInstr>(
        BabyAIObjDesc(obj.second.first, obj.second.second));
    if (doors_open_) {
      OpenAllDoors();
    }
  }

 private:
  int num_dists_{18};
  bool doors_open_{false};
};

class BabyAIGoToImpUnlockTask : public BabyAILevelTask {
 public:
  explicit BabyAIGoToImpUnlockTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, config.room_size, config.num_rows,
                        config.num_cols, config.max_steps,
                        config.mission_bytes) {}

 protected:
  void GenMission() override {
    int locked_i = RandInt(0, num_cols_);
    int locked_j = RandInt(0, num_rows_);
    Pos door_pos = AddDoor(locked_i, locked_j, -1, kUnassigned, true);
    const Room* locked_room = &GetRoom(locked_i, locked_j);
    const Color door_color =
        GetCell(door_pos.first, door_pos.second).GetColor();
    while (true) {
      int i = RandInt(0, num_cols_);
      int j = RandInt(0, num_rows_);
      if (i == locked_i && j == locked_j) {
        continue;
      }
      AddObject(i, j, kKey, door_color);
      break;
    }
    ConnectAllOrReject();
    for (int i = 0; i < num_cols_; ++i) {
      for (int j = 0; j < num_rows_; ++j) {
        if (i != locked_i || j != locked_j) {
          AddDistractorsOrReject(i, j, 2, false);
        }
      }
    }
    while (true) {
      PlaceAgentInRoom();
      if (&RoomFromPos(agent_pos_.first, agent_pos_.second) != locked_room) {
        break;
      }
    }
    CheckObjsReachableOrReject();
    auto objs = AddDistractorsOrReject(locked_i, locked_j, 1, false);
    instrs_ = std::make_unique<BabyAIGoToInstr>(
        BabyAIObjDesc(objs[0].second.first, objs[0].second.second));
  }
};

class BabyAIGoToRedBlueBallTask : public BabyAILevelTask {
 public:
  explicit BabyAIGoToRedBlueBallTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, config.room_size, 1, 1,
                        config.max_steps, config.mission_bytes),
        num_dists_(config.num_dists) {}

 protected:
  void GenMission() override {
    PlaceAgentInRoom();
    auto dists = AddDistractorsOrReject(-1, -1, num_dists_, false);
    for (const auto& dist : dists) {
      if (dist.second.first == kBall &&
          (dist.second.second == kRed || dist.second.second == kBlue)) {
        throw BabyAIRejectSampling("can only have one blue or red ball");
      }
    }
    auto obj =
        AddObject(0, 0, kBall, RandElem(std::vector<Color>{kRed, kBlue}));
    CheckObjsReachableOrReject();
    instrs_ = std::make_unique<BabyAIGoToInstr>(
        BabyAIObjDesc(obj.second.first, obj.second.second));
  }

 private:
  int num_dists_{7};
};

class BabyAIGoToDoorTask : public BabyAILevelTask {
 public:
  explicit BabyAIGoToDoorTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, 7, config.num_rows, config.num_cols,
                        config.max_steps, config.mission_bytes) {}

 protected:
  void GenMission() override {
    std::vector<std::pair<Pos, std::pair<Type, Color>>> objs;
    for (int i = 0; i < 4; ++i) {
      Pos pos = AddDoor(1, 1, -1, kUnassigned, RandBool());
      objs.push_back(ObjAtDoor(*this, pos));
    }
    PlaceAgentInRoom(1, 1);
    const auto& obj = RandElem(objs);
    instrs_ = std::make_unique<BabyAIGoToInstr>(
        BabyAIObjDesc(obj.second.first, obj.second.second));
  }
};

class BabyAIGoToObjDoorTask : public BabyAILevelTask {
 public:
  explicit BabyAIGoToObjDoorTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, 8, config.num_rows, config.num_cols,
                        config.max_steps, config.mission_bytes) {}

 protected:
  void GenMission() override {
    PlaceAgentInRoom(1, 1);
    auto objs = AddDistractorsOrReject(1, 1, 8, false);
    for (int i = 0; i < 4; ++i) {
      Pos pos = AddDoor(1, 1, -1, kUnassigned, RandBool());
      objs.push_back(ObjAtDoor(*this, pos));
    }
    CheckObjsReachableOrReject();
    const auto& obj = RandElem(objs);
    instrs_ = std::make_unique<BabyAIGoToInstr>(
        BabyAIObjDesc(obj.second.first, obj.second.second));
  }
};

class BabyAIGoToSeqTask : public BabyAILevelGenTask {
 public:
  explicit BabyAIGoToSeqTask(const BabyAITaskConfig& config)
      : BabyAILevelGenTask(MakeGoToSeqConfig(config)) {}
};

class BabyAIPickupTask : public BabyAILevelTask {
 public:
  explicit BabyAIPickupTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, config.room_size, config.num_rows,
                        config.num_cols, config.max_steps,
                        config.mission_bytes) {}

 protected:
  void GenMission() override {
    PlaceAgentInRoom();
    ConnectAllOrReject();
    auto objs = AddDistractorsOrReject(-1, -1, 18, false);
    CheckObjsReachableOrReject();
    const auto& obj = RandElem(objs);
    instrs_ = std::make_unique<BabyAIPickupInstr>(
        BabyAIObjDesc(obj.second.first, obj.second.second));
  }
};

class BabyAIUnblockPickupTask : public BabyAILevelTask {
 public:
  explicit BabyAIUnblockPickupTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, config.room_size, config.num_rows,
                        config.num_cols, config.max_steps,
                        config.mission_bytes) {}

 protected:
  void GenMission() override {
    PlaceAgentInRoom();
    ConnectAllOrReject();
    auto objs = AddDistractorsOrReject(-1, -1, 20, false);
    if (CheckObjsReachable()) {
      throw BabyAIRejectSampling("all objects reachable");
    }
    const auto& obj = RandElem(objs);
    instrs_ = std::make_unique<BabyAIPickupInstr>(
        BabyAIObjDesc(obj.second.first, obj.second.second));
  }
};

class BabyAIPickupLocTask : public BabyAILevelGenTask {
 public:
  explicit BabyAIPickupLocTask(const BabyAITaskConfig& config)
      : BabyAILevelGenTask(MakePickupLocConfig(config)) {}
};

class BabyAIPickupDistTask : public BabyAILevelTask {
 public:
  explicit BabyAIPickupDistTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, 7, 1, 1, config.max_steps,
                        config.mission_bytes),
        debug_(config.debug) {}

 protected:
  void GenMission() override {
    auto objs = AddDistractorsOrReject(-1, -1, 5, true);
    PlaceAgentInRoom(0, 0);
    const auto& obj = RandElem(objs);
    std::string select_by =
        RandElem(std::vector<std::string>{"type", "color", "both"});
    if (select_by == "color") {
      instrs_ = std::make_unique<BabyAIPickupInstr>(
          BabyAIObjDesc(std::nullopt, obj.second.second), debug_);
    } else if (select_by == "type") {
      instrs_ = std::make_unique<BabyAIPickupInstr>(
          BabyAIObjDesc(obj.second.first, std::nullopt), debug_);
    } else {
      instrs_ = std::make_unique<BabyAIPickupInstr>(
          BabyAIObjDesc(obj.second.first, obj.second.second), debug_);
    }
  }

 private:
  bool debug_{false};
};

class BabyAIPickupAboveTask : public BabyAILevelTask {
 public:
  explicit BabyAIPickupAboveTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, 6, config.num_rows, config.num_cols,
                        config.max_steps > 0 ? config.max_steps : 8 * 6 * 6,
                        config.mission_bytes) {}

 protected:
  void GenMission() override {
    auto obj = AddObject(1, 0);
    AddDoor(1, 1, 3, kUnassigned, false);
    PlaceAgentInRoom(1, 1);
    ConnectAllOrReject();
    instrs_ = std::make_unique<BabyAIPickupInstr>(
        BabyAIObjDesc(obj.second.first, obj.second.second));
  }
};

class BabyAIPutNextLocalTask : public BabyAILevelTask {
 public:
  explicit BabyAIPutNextLocalTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, config.room_size, 1, 1,
                        config.max_steps, config.mission_bytes),
        num_objs_(config.num_objs) {}

 protected:
  void GenMission() override {
    PlaceAgentInRoom();
    auto objs = AddDistractorsOrReject(-1, -1, num_objs_, true);
    CheckObjsReachableOrReject();
    auto pair = RandSubset(objs, 2);
    instrs_ = std::make_unique<BabyAIPutNextInstr>(
        BabyAIObjDesc(pair[0].second.first, pair[0].second.second),
        BabyAIObjDesc(pair[1].second.first, pair[1].second.second));
  }

 private:
  int num_objs_{8};
};

class BabyAIPutNextTask : public BabyAILevelTask {
 public:
  explicit BabyAIPutNextTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, config.room_size, 1, 2,
                        config.max_steps > 0
                            ? config.max_steps
                            : 8 * config.room_size * config.room_size,
                        config.mission_bytes),
        objs_per_room_(config.objs_per_room),
        start_carrying_(config.start_carrying) {}

 protected:
  void GenMission() override {
    PlaceAgentInRoom(0, 0);
    auto objs_l = AddDistractorsOrReject(0, 0, objs_per_room_, true);
    auto objs_r = AddDistractorsOrReject(1, 0, objs_per_room_, true);
    RemoveWall(0, 0, 0);
    auto obj_a = RandElem(objs_l);
    auto obj_b = RandElem(objs_r);
    if (RandBool()) {
      std::swap(obj_a, obj_b);
    }
    move_pos_ = obj_a.first;
    move_type_ = obj_a.second.first;
    move_color_ = obj_a.second.second;
    instrs_ = std::make_unique<BabyAIPutNextInstr>(
        BabyAIObjDesc(obj_a.second.first, obj_a.second.second),
        BabyAIObjDesc(obj_b.second.first, obj_b.second.second));
  }

  void AfterResetVerifier() override {
    if (!start_carrying_) {
      return;
    }
    carrying_ = GetCell(move_pos_.first, move_pos_.second);
    SetEmpty(move_pos_.first, move_pos_.second);
  }

 private:
  int objs_per_room_{4};
  bool start_carrying_{false};
};

class BabyAIMoveTwoAcrossTask : public BabyAILevelTask {
 public:
  explicit BabyAIMoveTwoAcrossTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, config.room_size, 1, 2,
                        config.max_steps > 0
                            ? config.max_steps
                            : 16 * config.room_size * config.room_size,
                        config.mission_bytes),
        objs_per_room_(config.objs_per_room) {}

 protected:
  void GenMission() override {
    PlaceAgentInRoom(0, 0);
    auto objs_l =
        RandSubset(AddDistractorsOrReject(0, 0, objs_per_room_, true), 2);
    auto objs_r =
        RandSubset(AddDistractorsOrReject(1, 0, objs_per_room_, true), 2);
    RemoveWall(0, 0, 0);
    instrs_ = std::make_unique<BabyAIBeforeInstr>(
        std::make_unique<BabyAIPutNextInstr>(
            BabyAIObjDesc(objs_l[0].second.first, objs_l[0].second.second),
            BabyAIObjDesc(objs_r[0].second.first, objs_r[0].second.second)),
        std::make_unique<BabyAIPutNextInstr>(
            BabyAIObjDesc(objs_r[1].second.first, objs_r[1].second.second),
            BabyAIObjDesc(objs_l[1].second.first, objs_l[1].second.second)));
  }

 private:
  int objs_per_room_{4};
};

class BabyAIOpenTask : public BabyAILevelTask {
 public:
  explicit BabyAIOpenTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, config.room_size, config.num_rows,
                        config.num_cols, config.max_steps,
                        config.mission_bytes) {}

 protected:
  void GenMission() override {
    PlaceAgentInRoom();
    ConnectAllOrReject();
    AddDistractorsOrReject(-1, -1, 18, false);
    CheckObjsReachableOrReject();
    std::vector<std::pair<Pos, std::pair<Type, Color>>> doors;
    for (int i = 0; i < num_cols_; ++i) {
      for (int j = 0; j < num_rows_; ++j) {
        const Room& room = GetRoom(i, j);
        for (int door_idx = 0; door_idx < 4; ++door_idx) {
          if (!room.connected[door_idx]) {
            continue;
          }
          const Pos pos = room.door_pos[door_idx];
          const WorldObj door = GetCell(pos.first, pos.second);
          if (door.GetType() == kDoor) {
            doors.push_back({pos, {door.GetType(), door.GetColor()}});
          }
        }
      }
    }
    const auto& door = RandElem(doors);
    instrs_ = std::make_unique<BabyAIOpenInstr>(
        BabyAIObjDesc(door.second.first, door.second.second));
  }
};

class BabyAIOpenRedDoorTask : public BabyAILevelTask {
 public:
  explicit BabyAIOpenRedDoorTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, 5, 1, 2, config.max_steps,
                        config.mission_bytes) {}

 protected:
  void GenMission() override {
    AddDoor(0, 0, 0, kRed, false);
    PlaceAgentInRoom(0, 0);
    instrs_ = std::make_unique<BabyAIOpenInstr>(BabyAIObjDesc(kDoor, kRed));
  }
};

class BabyAIOpenDoorTask : public BabyAILevelTask {
 public:
  explicit BabyAIOpenDoorTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, config.room_size, config.num_rows,
                        config.num_cols, config.max_steps,
                        config.mission_bytes),
        debug_(config.debug),
        select_by_(config.select_by) {}

 protected:
  void GenMission() override {
    std::vector<Color> door_colors =
        RandSubset(std::vector<Color>(kColors.begin(), kColors.end()), 4);
    std::vector<std::pair<Pos, std::pair<Type, Color>>> doors;
    for (int i = 0; i < 4; ++i) {
      Pos pos = AddDoor(1, 1, i, door_colors[i], false);
      doors.push_back({pos, {kDoor, door_colors[i]}});
    }
    std::string select_by = select_by_;
    if (select_by.empty()) {
      select_by =
          std::string(RandElem(std::vector<std::string>{"color", "loc"}));
    }
    std::unique_ptr<BabyAIInstr> instr;
    if (select_by == "color") {
      instr = std::make_unique<BabyAIOpenInstr>(
          BabyAIObjDesc(doors[0].second.first, doors[0].second.second), debug_);
    } else if (select_by == "loc") {
      instr = std::make_unique<BabyAIOpenInstr>(
          BabyAIObjDesc(kDoor, std::nullopt,
                        RandElem(std::vector<BabyAILoc>{
                            BabyAILoc::kLeft, BabyAILoc::kRight,
                            BabyAILoc::kFront, BabyAILoc::kBehind})),
          debug_);
    } else {
      throw BabyAIRejectSampling("invalid select_by");
    }
    PlaceAgentInRoom(1, 1);
    instrs_ = std::move(instr);
  }

 private:
  bool debug_{false};
  std::string select_by_;
};

class BabyAIOpenTwoDoorsTask : public BabyAILevelTask {
 public:
  explicit BabyAIOpenTwoDoorsTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, 6, config.num_rows, config.num_cols,
                        config.max_steps > 0 ? config.max_steps : 20 * 6 * 6,
                        config.mission_bytes),
        first_color_(config.first_color),
        second_color_(config.second_color),
        strict_(config.strict) {}

 protected:
  void GenMission() override {
    auto colors =
        RandSubset(std::vector<Color>(kColors.begin(), kColors.end()), 2);
    Color first_color =
        first_color_.empty() ? colors[0] : ParseColor(first_color_);
    Color second_color =
        second_color_.empty() ? colors[1] : ParseColor(second_color_);
    Pos door1 = AddDoor(1, 1, 2, first_color, false);
    Pos door2 = AddDoor(1, 1, 0, second_color, false);
    PlaceAgentInRoom(1, 1);
    instrs_ = std::make_unique<BabyAIBeforeInstr>(
        std::make_unique<BabyAIOpenInstr>(
            BabyAIObjDesc(kDoor, GetCell(door1.first, door1.second).GetColor()),
            strict_),
        std::make_unique<BabyAIOpenInstr>(BabyAIObjDesc(
            kDoor, GetCell(door2.first, door2.second).GetColor())));
  }

 private:
  std::string first_color_;
  std::string second_color_;
  bool strict_{false};
};

class BabyAIOpenDoorsOrderTask : public BabyAILevelTask {
 public:
  explicit BabyAIOpenDoorsOrderTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, 6, config.num_rows, config.num_cols,
                        config.max_steps > 0 ? config.max_steps : 20 * 6 * 6,
                        config.mission_bytes),
        num_doors_(config.num_doors),
        debug_(config.debug) {}

 protected:
  void GenMission() override {
    auto colors = RandSubset(std::vector<Color>(kColors.begin(), kColors.end()),
                             num_doors_);
    std::vector<std::pair<Pos, std::pair<Type, Color>>> doors;
    for (Color color : colors) {
      Pos pos = AddDoor(1, 1, -1, color, false);
      doors.push_back({pos, {kDoor, color}});
    }
    PlaceAgentInRoom(1, 1);
    auto selected = RandSubset(doors, 2);
    BabyAIObjDesc desc1(selected[0].second.first, selected[0].second.second);
    BabyAIObjDesc desc2(selected[1].second.first, selected[1].second.second);
    int mode = RandInt(0, 3);
    if (mode == 0) {
      instrs_ = std::make_unique<BabyAIOpenInstr>(std::move(desc1), debug_);
    } else if (mode == 1) {
      instrs_ = std::make_unique<BabyAIBeforeInstr>(
          std::make_unique<BabyAIOpenInstr>(std::move(desc1), debug_),
          std::make_unique<BabyAIOpenInstr>(std::move(desc2), debug_));
    } else {
      instrs_ = std::make_unique<BabyAIAfterInstr>(
          std::make_unique<BabyAIOpenInstr>(std::move(desc1), debug_),
          std::make_unique<BabyAIOpenInstr>(std::move(desc2), debug_));
    }
  }

 private:
  int num_doors_{2};
  bool debug_{false};
};

class BabyAIUnlockTask : public BabyAILevelTask {
 public:
  explicit BabyAIUnlockTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, config.room_size, config.num_rows,
                        config.num_cols, config.max_steps,
                        config.mission_bytes) {}

 protected:
  void GenMission() override {
    int door_i = RandInt(0, num_cols_);
    int door_j = RandInt(0, num_rows_);
    Pos door_pos = AddDoor(door_i, door_j, -1, kUnassigned, true);
    const Room* locked_room = &GetRoom(door_i, door_j);
    const Color door_color =
        GetCell(door_pos.first, door_pos.second).GetColor();
    while (true) {
      int key_i = RandInt(0, num_cols_);
      int key_j = RandInt(0, num_rows_);
      if (key_i == door_i && key_j == door_j) {
        continue;
      }
      AddObject(key_i, key_j, kKey, door_color);
      break;
    }
    if (RandBool()) {
      std::vector<Color> door_colors;
      for (Color color : kColors) {
        if (color != door_color) {
          door_colors.push_back(color);
        }
      }
      ConnectAllOrReject(door_colors);
    } else {
      ConnectAllOrReject();
    }
    for (int i = 0; i < num_cols_; ++i) {
      for (int j = 0; j < num_rows_; ++j) {
        if (i != door_i || j != door_j) {
          AddDistractorsOrReject(i, j, 3, false);
        }
      }
    }
    while (true) {
      PlaceAgentInRoom();
      if (&RoomFromPos(agent_pos_.first, agent_pos_.second) != locked_room) {
        break;
      }
    }
    CheckObjsReachableOrReject();
    instrs_ =
        std::make_unique<BabyAIOpenInstr>(BabyAIObjDesc(kDoor, door_color));
  }
};

class BabyAIUnlockLocalTask : public BabyAILevelTask {
 public:
  explicit BabyAIUnlockLocalTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, config.room_size, config.num_rows,
                        config.num_cols, config.max_steps,
                        config.mission_bytes),
        distractors_(config.distractors) {}

 protected:
  void GenMission() override {
    Pos door_pos = AddDoor(1, 1, -1, kUnassigned, true);
    AddObject(1, 1, kKey, GetCell(door_pos.first, door_pos.second).GetColor());
    if (distractors_) {
      AddDistractorsOrReject(1, 1, 3, true);
    }
    PlaceAgentInRoom(1, 1);
    instrs_ =
        std::make_unique<BabyAIOpenInstr>(BabyAIObjDesc(kDoor, std::nullopt));
  }

 private:
  bool distractors_{false};
};

class BabyAIKeyInBoxTask : public BabyAILevelTask {
 public:
  explicit BabyAIKeyInBoxTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, config.room_size, config.num_rows,
                        config.num_cols, config.max_steps,
                        config.mission_bytes) {}

 protected:
  void GenMission() override {
    Pos door_pos = AddDoor(1, 1, -1, kUnassigned, true);
    WorldObj box(kBox, RandColor());
    box.SetContains(std::make_unique<WorldObj>(
        kKey, GetCell(door_pos.first, door_pos.second).GetColor()));
    AddExistingObject(1, 1, box);
    PlaceAgentInRoom(1, 1);
    instrs_ =
        std::make_unique<BabyAIOpenInstr>(BabyAIObjDesc(kDoor, std::nullopt));
  }
};

class BabyAIUnlockPickupTask : public BabyAILevelTask {
 public:
  explicit BabyAIUnlockPickupTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, 6, 1, 2, config.max_steps,
                        config.mission_bytes),
        distractors_(config.distractors) {}

 protected:
  void GenMission() override {
    auto obj = AddObject(1, 0, kBox, kUnassigned);
    Pos door_pos = AddDoor(0, 0, 0, kUnassigned, true);
    AddObject(0, 0, kKey, GetCell(door_pos.first, door_pos.second).GetColor());
    if (distractors_) {
      AddDistractorsOrReject(-1, -1, 4, true);
    }
    PlaceAgentInRoom(0, 0);
    instrs_ = std::make_unique<BabyAIPickupInstr>(
        BabyAIObjDesc(obj.second.first, obj.second.second));
  }

 private:
  bool distractors_{false};
};

class BabyAIBlockedUnlockPickupTask : public BabyAILevelTask {
 public:
  explicit BabyAIBlockedUnlockPickupTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, 6, 1, 2,
                        config.max_steps > 0 ? config.max_steps : 16 * 6 * 6,
                        config.mission_bytes) {}

 protected:
  void GenMission() override {
    auto obj = AddObject(1, 0, kBox, kUnassigned);
    Pos door_pos = AddDoor(0, 0, 0, kUnassigned, true);
    PutObj(WorldObj(kBall, RandColor()), door_pos.first - 1, door_pos.second);
    AddObject(0, 0, kKey, GetCell(door_pos.first, door_pos.second).GetColor());
    PlaceAgentInRoom(0, 0);
    instrs_ = std::make_unique<BabyAIPickupInstr>(
        BabyAIObjDesc(obj.second.first, std::nullopt));
  }
};

class BabyAIUnlockToUnlockTask : public BabyAILevelTask {
 public:
  explicit BabyAIUnlockToUnlockTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, 6, 1, 3,
                        config.max_steps > 0 ? config.max_steps : 30 * 6 * 6,
                        config.mission_bytes) {}

 protected:
  void GenMission() override {
    auto colors =
        RandSubset(std::vector<Color>(kColors.begin(), kColors.end()), 2);
    AddDoor(0, 0, 0, colors[0], true);
    AddObject(2, 0, kKey, colors[0]);
    AddDoor(1, 0, 0, colors[1], true);
    AddObject(1, 0, kKey, colors[1]);
    auto obj = AddObject(0, 0, kBall, kUnassigned);
    PlaceAgentInRoom(1, 0);
    instrs_ = std::make_unique<BabyAIPickupInstr>(
        BabyAIObjDesc(obj.second.first, std::nullopt));
  }
};

class BabyAIActionObjDoorTask : public BabyAILevelTask {
 public:
  explicit BabyAIActionObjDoorTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, 7, config.num_rows, config.num_cols,
                        config.max_steps, config.mission_bytes) {}

 protected:
  void GenMission() override {
    auto objs = AddDistractorsOrReject(1, 1, 5, true);
    for (int i = 0; i < 4; ++i) {
      Pos pos = AddDoor(1, 1, -1, kUnassigned, RandBool());
      objs.push_back(ObjAtDoor(*this, pos));
    }
    PlaceAgentInRoom(1, 1);
    const auto& obj = RandElem(objs);
    BabyAIObjDesc desc(obj.second.first, obj.second.second);
    if (obj.second.first == kDoor) {
      if (RandBool()) {
        instrs_ = std::make_unique<BabyAIGoToInstr>(std::move(desc));
      } else {
        instrs_ = std::make_unique<BabyAIOpenInstr>(std::move(desc));
      }
    } else if (RandBool()) {
      instrs_ = std::make_unique<BabyAIGoToInstr>(std::move(desc));
    } else {
      instrs_ = std::make_unique<BabyAIPickupInstr>(std::move(desc));
    }
  }
};

class BabyAIFindObjTask : public BabyAILevelTask {
 public:
  explicit BabyAIFindObjTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(
            config.env_name, config.room_size, config.num_rows, config.num_cols,
            config.max_steps > 0 ? config.max_steps
                                 : 20 * config.room_size * config.room_size,
            config.mission_bytes) {}

 protected:
  void GenMission() override {
    int i = RandInt(0, num_cols_);
    int j = RandInt(0, num_rows_);
    auto obj = AddObject(i, j);
    PlaceAgentInRoom(1, 1);
    ConnectAllOrReject();
    instrs_ = std::make_unique<BabyAIPickupInstr>(
        BabyAIObjDesc(obj.second.first, std::nullopt));
  }
};

class BabyAIKeyCorridorTask : public BabyAILevelTask {
 public:
  explicit BabyAIKeyCorridorTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, config.room_size, config.num_rows, 3,
                        config.max_steps > 0
                            ? config.max_steps
                            : 30 * config.room_size * config.room_size,
                        config.mission_bytes),
        obj_type_(config.obj_type) {}

 protected:
  void GenMission() override {
    for (int j = 1; j < num_rows_; ++j) {
      RemoveWall(1, j, 3);
    }
    int room_idx = RandInt(0, num_rows_);
    Pos door_pos = AddDoor(2, room_idx, 2, kUnassigned, true);
    auto obj = AddObject(2, room_idx, obj_type_, kUnassigned);
    AddObject(0, RandInt(0, num_rows_), kKey,
              GetCell(door_pos.first, door_pos.second).GetColor());
    PlaceAgentInRoom(1, num_rows_ / 2);
    ConnectAllOrReject();
    instrs_ = std::make_unique<BabyAIPickupInstr>(
        BabyAIObjDesc(obj.second.first, std::nullopt));
  }

 private:
  Type obj_type_{kBall};
};

class BabyAIOneRoomTask : public BabyAILevelTask {
 public:
  explicit BabyAIOneRoomTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, config.room_size, 1, 1,
                        config.max_steps, config.mission_bytes) {}

 protected:
  void GenMission() override {
    auto obj = AddObject(0, 0, kBall, kUnassigned);
    PlaceAgentInRoom();
    instrs_ = std::make_unique<BabyAIPickupInstr>(
        BabyAIObjDesc(obj.second.first, std::nullopt));
  }
};

class BabyAIMiniBossLevelTask : public BabyAILevelGenTask {
 public:
  explicit BabyAIMiniBossLevelTask(const BabyAITaskConfig& config)
      : BabyAILevelGenTask(MakeMiniBossConfig(config)) {}
};

class BabyAIBossLevelTask : public BabyAILevelGenTask {
 public:
  explicit BabyAIBossLevelTask(const BabyAITaskConfig& config)
      : BabyAILevelGenTask(config) {}
};

class BabyAIBossLevelNoUnlockTask : public BabyAILevelGenTask {
 public:
  explicit BabyAIBossLevelNoUnlockTask(const BabyAITaskConfig& config)
      : BabyAILevelGenTask(MakeBossNoUnlockConfig(config)) {}
};

class BabyAISynthTask : public BabyAILevelGenTask {
 public:
  explicit BabyAISynthTask(const BabyAITaskConfig& config)
      : BabyAILevelGenTask(MakeSynthConfig(config)) {}
};

class BabyAISynthLocTask : public BabyAILevelGenTask {
 public:
  explicit BabyAISynthLocTask(const BabyAITaskConfig& config)
      : BabyAILevelGenTask(MakeSynthLocConfig(config)) {}
};

class BabyAISynthSeqTask : public BabyAILevelGenTask {
 public:
  explicit BabyAISynthSeqTask(const BabyAITaskConfig& config)
      : BabyAILevelGenTask(MakeSynthSeqConfig(config)) {}
};

}  // namespace

std::unique_ptr<MiniGridTask> MakeBabyAITask(const BabyAITaskConfig& config) {
  if (config.env_name == "babyai_action_obj_door") {
    return std::make_unique<BabyAIActionObjDoorTask>(config);
  }
  if (config.env_name == "babyai_blocked_unlock_pickup") {
    return std::make_unique<BabyAIBlockedUnlockPickupTask>(config);
  }
  if (config.env_name == "babyai_boss_level") {
    return std::make_unique<BabyAIBossLevelTask>(config);
  }
  if (config.env_name == "babyai_boss_level_no_unlock") {
    return std::make_unique<BabyAIBossLevelNoUnlockTask>(config);
  }
  if (config.env_name == "babyai_find_obj") {
    return std::make_unique<BabyAIFindObjTask>(config);
  }
  if (config.env_name == "babyai_goto") {
    return std::make_unique<BabyAIGoToTask>(config);
  }
  if (config.env_name == "babyai_goto_door") {
    return std::make_unique<BabyAIGoToDoorTask>(config);
  }
  if (config.env_name == "babyai_goto_imp_unlock") {
    return std::make_unique<BabyAIGoToImpUnlockTask>(config);
  }
  if (config.env_name == "babyai_goto_local") {
    return std::make_unique<BabyAIGoToLocalTask>(config);
  }
  if (config.env_name == "babyai_goto_obj") {
    return std::make_unique<BabyAIGoToObjTask>(config);
  }
  if (config.env_name == "babyai_goto_obj_door") {
    return std::make_unique<BabyAIGoToObjDoorTask>(config);
  }
  if (config.env_name == "babyai_goto_red_ball") {
    return std::make_unique<BabyAIGoToRedBallTask>(config, /*grey=*/false,
                                                   config.num_dists);
  }
  if (config.env_name == "babyai_goto_red_ball_grey") {
    return std::make_unique<BabyAIGoToRedBallTask>(config, /*grey=*/true,
                                                   config.num_dists);
  }
  if (config.env_name == "babyai_goto_red_ball_no_dists") {
    return std::make_unique<BabyAIGoToRedBallTask>(config, /*grey=*/false,
                                                   /*num_dists=*/0);
  }
  if (config.env_name == "babyai_goto_red_blue_ball") {
    return std::make_unique<BabyAIGoToRedBlueBallTask>(config);
  }
  if (config.env_name == "babyai_goto_seq") {
    return std::make_unique<BabyAIGoToSeqTask>(config);
  }
  if (config.env_name == "babyai_key_corridor") {
    return std::make_unique<BabyAIKeyCorridorTask>(config);
  }
  if (config.env_name == "babyai_key_in_box") {
    return std::make_unique<BabyAIKeyInBoxTask>(config);
  }
  if (config.env_name == "babyai_mini_boss_level") {
    return std::make_unique<BabyAIMiniBossLevelTask>(config);
  }
  if (config.env_name == "babyai_move_two_across") {
    return std::make_unique<BabyAIMoveTwoAcrossTask>(config);
  }
  if (config.env_name == "babyai_one_room") {
    return std::make_unique<BabyAIOneRoomTask>(config);
  }
  if (config.env_name == "babyai_open") {
    return std::make_unique<BabyAIOpenTask>(config);
  }
  if (config.env_name == "babyai_open_door") {
    return std::make_unique<BabyAIOpenDoorTask>(config);
  }
  if (config.env_name == "babyai_open_doors_order") {
    return std::make_unique<BabyAIOpenDoorsOrderTask>(config);
  }
  if (config.env_name == "babyai_open_red_door") {
    return std::make_unique<BabyAIOpenRedDoorTask>(config);
  }
  if (config.env_name == "babyai_open_two_doors") {
    return std::make_unique<BabyAIOpenTwoDoorsTask>(config);
  }
  if (config.env_name == "babyai_pickup") {
    return std::make_unique<BabyAIPickupTask>(config);
  }
  if (config.env_name == "babyai_pickup_above") {
    return std::make_unique<BabyAIPickupAboveTask>(config);
  }
  if (config.env_name == "babyai_pickup_dist") {
    return std::make_unique<BabyAIPickupDistTask>(config);
  }
  if (config.env_name == "babyai_pickup_loc") {
    return std::make_unique<BabyAIPickupLocTask>(config);
  }
  if (config.env_name == "babyai_put_next") {
    return std::make_unique<BabyAIPutNextTask>(config);
  }
  if (config.env_name == "babyai_put_next_local") {
    return std::make_unique<BabyAIPutNextLocalTask>(config);
  }
  if (config.env_name == "babyai_synth") {
    return std::make_unique<BabyAISynthTask>(config);
  }
  if (config.env_name == "babyai_synth_loc") {
    return std::make_unique<BabyAISynthLocTask>(config);
  }
  if (config.env_name == "babyai_synth_seq") {
    return std::make_unique<BabyAISynthSeqTask>(config);
  }
  if (config.env_name == "babyai_unblock_pickup") {
    return std::make_unique<BabyAIUnblockPickupTask>(config);
  }
  if (config.env_name == "babyai_unlock") {
    return std::make_unique<BabyAIUnlockTask>(config);
  }
  if (config.env_name == "babyai_unlock_local") {
    return std::make_unique<BabyAIUnlockLocalTask>(config);
  }
  if (config.env_name == "babyai_unlock_pickup") {
    return std::make_unique<BabyAIUnlockPickupTask>(config);
  }
  if (config.env_name == "babyai_unlock_to_unlock") {
    return std::make_unique<BabyAIUnlockToUnlockTask>(config);
  }
  return nullptr;
}

}  // namespace minigrid
