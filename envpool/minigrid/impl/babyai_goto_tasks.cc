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

#include <memory>
#include <utility>
#include <vector>

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

}  // namespace

std::unique_ptr<MiniGridTask> MakeBabyAIGoToTask(
    const BabyAITaskConfig& config) {
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
  return nullptr;
}

}  // namespace minigrid
