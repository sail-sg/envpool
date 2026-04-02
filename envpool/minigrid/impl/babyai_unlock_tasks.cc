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

std::unique_ptr<MiniGridTask> MakeBabyAIUnlockTask(
    const BabyAITaskConfig& config) {
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
  if (config.env_name == "babyai_key_corridor") {
    return std::make_unique<BabyAIKeyCorridorTask>(config);
  }
  if (config.env_name == "babyai_key_in_box") {
    return std::make_unique<BabyAIKeyInBoxTask>(config);
  }
  if (config.env_name == "babyai_mini_boss_level") {
    return std::make_unique<BabyAIMiniBossLevelTask>(config);
  }
  if (config.env_name == "babyai_one_room") {
    return std::make_unique<BabyAIOneRoomTask>(config);
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
