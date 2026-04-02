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
#include <string>
#include <utility>
#include <vector>

#include "envpool/minigrid/impl/babyai_core.h"

namespace minigrid {
namespace {

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

}  // namespace

std::unique_ptr<MiniGridTask> MakeBabyAIPickupTask(
    const BabyAITaskConfig& config) {
  if (config.env_name == "babyai_move_two_across") {
    return std::make_unique<BabyAIMoveTwoAcrossTask>(config);
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
  if (config.env_name == "babyai_unblock_pickup") {
    return std::make_unique<BabyAIUnblockPickupTask>(config);
  }
  return nullptr;
}

}  // namespace minigrid
