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

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace minigrid {
namespace {

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
            doors.emplace_back(
                pos, std::pair<Type, Color>{door.GetType(), door.GetColor()});
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
      doors.emplace_back(pos, std::pair<Type, Color>{kDoor, door_colors[i]});
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
      doors.emplace_back(pos, std::pair<Type, Color>{kDoor, color});
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

}  // namespace

std::unique_ptr<MiniGridTask> MakeBabyAIOpenTask(
    const BabyAITaskConfig& config) {
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
  return nullptr;
}

}  // namespace minigrid
