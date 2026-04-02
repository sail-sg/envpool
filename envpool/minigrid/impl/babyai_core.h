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

#pragma once

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "envpool/minigrid/impl/babyai_env.h"

namespace minigrid {

inline constexpr std::array<Pos, 4> kDirToVec = {
    Pos{1, 0}, Pos{0, 1}, Pos{-1, 0},
    Pos{0, -1}};  // NOLINT(whitespace/indent_namespace)

inline bool UseDoneActions() {
  return std::getenv("BABYAI_DONE_ACTIONS") != nullptr;
}

inline const bool kUseDoneActions = UseDoneActions();

inline bool IsNextTo(const Pos& lhs, const Pos& rhs) {
  return std::abs(lhs.first - rhs.first) + std::abs(lhs.second - rhs.second) ==
         1;
}

inline int Dot(const Pos& lhs, const Pos& rhs) {
  return lhs.first * rhs.first + lhs.second * rhs.second;
}

class BabyAIRejectSampling : public std::runtime_error {
 public:
  explicit BabyAIRejectSampling(const std::string& msg)
      : std::runtime_error(msg) {}
};

enum class BabyAIStatus { kContinue, kSuccess, kFailure };

enum class BabyAILoc { kNone, kLeft, kRight, kFront, kBehind };

inline std::string BabyAILocSurface(BabyAILoc loc) {
  switch (loc) {
    case BabyAILoc::kLeft:
      return " on your left";
    case BabyAILoc::kRight:
      return " on your right";
    case BabyAILoc::kFront:
      return " in front of you";
    case BabyAILoc::kBehind:
      return " behind you";
    case BabyAILoc::kNone:
      return "";
  }
  return "";
}

class BabyAIInstr;

class BabyAILevelTask : public RoomGridTask {
 public:
  BabyAILevelTask(std::string env_name, int room_size, int num_rows,
                  int num_cols, int max_steps, int mission_bytes)
      : RoomGridTask(std::move(env_name), room_size, num_rows, num_cols,
                     max_steps, 7, mission_bytes),
        fixed_max_steps_(max_steps > 0) {}

  [[nodiscard]] WorldObj CellAt(int x, int y) const { return GetCell(x, y); }
  [[nodiscard]] const WorldObj& Carrying() const { return carrying_; }
  [[nodiscard]] Pos FrontCellPos() const { return FrontPos(); }
  [[nodiscard]] const Room& RoomAt(int x, int y) const {
    return RoomFromPos(x, y);
  }
  [[nodiscard]] int Width() const { return width_; }
  [[nodiscard]] int Height() const { return height_; }
  [[nodiscard]] int RoomSize() const { return room_size_; }
  [[nodiscard]] int NumRows() const { return num_rows_; }
  [[nodiscard]] int NumCols() const { return num_cols_; }

 protected:
  void GenGrid() final;
  void AfterStep(Act act, const WorldObj& pre_fwd, const Pos& fwd_pos,
                 const WorldObj& pre_carrying, float* reward,
                 bool* terminated) final;

  virtual void GenMission() = 0;
  virtual void AfterResetVerifier() {}
  [[nodiscard]] virtual bool UnblockingEnabled() const { return true; }

  std::pair<Pos, std::pair<Type, Color>> AddExistingObject(
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

  std::vector<std::pair<Pos, std::pair<Type, Color>>> AddDistractorsOrReject(
      int i = -1, int j = -1, int num_distractors = 10,
      bool all_unique = true) {
    try {
      return AddDistractors(i, j, num_distractors, all_unique);
    } catch (const std::runtime_error&) {
      throw BabyAIRejectSampling("failed to add distractors");
    }
  }

  void ConnectAllOrReject(const std::vector<Color>& door_colors =
                              std::vector<Color>(kColors.begin(),
                                                 kColors.end())) {
    if (!TryConnectAll(door_colors)) {
      throw BabyAIRejectSampling("connect_all failed");
    }
  }

  void CheckObjsReachableOrReject() const {
    if (!CheckObjsReachable()) {
      throw BabyAIRejectSampling("unreachable object");
    }
  }

  void OpenAllDoors() {
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

  void AddLockedRoom();

  std::unique_ptr<BabyAIInstr> instrs_;
  const Room* locked_room_{nullptr};
  bool fixed_max_steps_{false};
};

class BabyAIObjDesc {
 public:
  BabyAIObjDesc(std::optional<Type> type, std::optional<Color> color,
                BabyAILoc loc = BabyAILoc::kNone)
      : type_(type), color_(color), loc_(loc) {}

  [[nodiscard]] std::string Surface(const BabyAILevelTask& env) {
    FindMatchingObjs(env, true);
    if (obj_uids_.empty()) {
      throw BabyAIRejectSampling("no object matching description");
    }
    std::string surface =
        type_.has_value() ? TypeName(type_.value()) : std::string("object");
    if (color_.has_value()) {
      surface = ColorName(color_.value()) + " " + surface;
    }
    surface += BabyAILocSurface(loc_);
    return (obj_uids_.size() > 1 ? "a " : "the ") + surface;
  }

  void FindMatchingObjs(const BabyAILevelTask& env, bool use_location = true) {
    if (use_location) {
      obj_uids_.clear();
    }
    obj_poss_.clear();
    obj_pos_uids_.clear();
    const Room& agent_room =
        env.RoomAt(env.AgentPos().first, env.AgentPos().second);
    for (int x = 0; x < env.Width(); ++x) {
      for (int y = 0; y < env.Height(); ++y) {
        const WorldObj cell = env.CellAt(x, y);
        if (cell.GetType() == kEmpty) {
          continue;
        }
        if (!use_location && !HasUid(cell.GetUid())) {
          continue;
        }
        if (type_.has_value() && cell.GetType() != type_.value()) {
          continue;
        }
        if (color_.has_value() && cell.GetColor() != color_.value()) {
          continue;
        }
        if (use_location && loc_ != BabyAILoc::kNone) {
          if (!agent_room.PosInside(x, y)) {
            continue;
          }
          const Pos offset{x - env.AgentPos().first, y - env.AgentPos().second};
          const Pos front = kDirToVec[env.AgentDir()];
          const Pos right{-front.second, front.first};
          bool matches = false;
          switch (loc_) {
            case BabyAILoc::kLeft:
              matches = Dot(offset, right) < 0;
              break;
            case BabyAILoc::kRight:
              matches = Dot(offset, right) > 0;
              break;
            case BabyAILoc::kFront:
              matches = Dot(offset, front) > 0;
              break;
            case BabyAILoc::kBehind:
              matches = Dot(offset, front) < 0;
              break;
            case BabyAILoc::kNone:
              matches = true;
              break;
          }
          if (!matches) {
            continue;
          }
        }
        if (use_location) {
          obj_uids_.push_back(cell.GetUid());
        }
        obj_poss_.push_back({x, y});
        obj_pos_uids_.push_back(cell.GetUid());
      }
    }
  }

  [[nodiscard]] bool HasType(Type type) const {
    return type_.has_value() && type_.value() == type;
  }

  [[nodiscard]] bool HasColor(Color color) const {
    return color_.has_value() && color_.value() == color;
  }

  [[nodiscard]] bool HasUid(int uid) const {
    return std::find(obj_uids_.begin(), obj_uids_.end(), uid) !=
           obj_uids_.end();
  }

  [[nodiscard]] const std::vector<Pos>& ObjPoss() const { return obj_poss_; }
  [[nodiscard]] const std::vector<int>& ObjUids() const { return obj_uids_; }

  [[nodiscard]] std::vector<Pos> ObjPossForUid(int uid) const {
    std::vector<Pos> poss;
    for (std::size_t i = 0; i < obj_poss_.size(); ++i) {
      if (obj_pos_uids_[i] == uid) {
        poss.push_back(obj_poss_[i]);
      }
    }
    return poss;
  }

 private:
  std::optional<Type> type_;
  std::optional<Color> color_;
  BabyAILoc loc_{BabyAILoc::kNone};
  std::vector<int> obj_uids_;
  std::vector<Pos> obj_poss_;
  std::vector<int> obj_pos_uids_;
};

class BabyAIInstr {
 public:
  virtual ~BabyAIInstr() = default;

  [[nodiscard]] virtual std::string Surface(const BabyAILevelTask& env) = 0;
  virtual void ResetVerifier(const BabyAILevelTask& env) {}
  [[nodiscard]] virtual BabyAIStatus Verify(const BabyAILevelTask& env,
                                            Act action,
                                            const WorldObj& pre_carrying) = 0;
  virtual void UpdateObjPoss(const BabyAILevelTask& env) {}
  virtual void Validate(const BabyAILevelTask& env,
                        const std::vector<Color>& locked_colors,
                        bool unblocking) = 0;
  [[nodiscard]] virtual int NumNavsNeeded() const = 0;
};

class BabyAIActionInstr : public BabyAIInstr {
 public:
  void ResetVerifier(const BabyAILevelTask& env) override {
    BabyAIInstr::ResetVerifier(env);
    last_step_match_ = false;
  }

  [[nodiscard]] BabyAIStatus Verify(const BabyAILevelTask& env, Act action,
                                    const WorldObj& pre_carrying) final {
    if (!kUseDoneActions) {
      return VerifyAction(env, action, pre_carrying);
    }
    if (action == kDone) {
      return last_step_match_ ? BabyAIStatus::kSuccess : BabyAIStatus::kFailure;
    }
    last_step_match_ =
        VerifyAction(env, action, pre_carrying) == BabyAIStatus::kSuccess;
    return BabyAIStatus::kContinue;
  }

  [[nodiscard]] int NumNavsNeeded() const override { return 1; }

 protected:
  explicit BabyAIActionInstr(bool strict = false) : strict_(strict) {}

  [[nodiscard]] virtual BabyAIStatus VerifyAction(
      const BabyAILevelTask& env, Act action, const WorldObj& pre_carrying) = 0;

  bool strict_{false};
  bool last_step_match_{false};
};

class BabyAIOpenInstr : public BabyAIActionInstr {
 public:
  explicit BabyAIOpenInstr(BabyAIObjDesc desc, bool strict = false)
      : BabyAIActionInstr(strict), desc_(std::move(desc)) {}

  [[nodiscard]] std::string Surface(const BabyAILevelTask& env) override {
    return "open " + desc_.Surface(env);
  }

  void ResetVerifier(const BabyAILevelTask& env) override {
    BabyAIActionInstr::ResetVerifier(env);
    desc_.FindMatchingObjs(env);
  }

  void UpdateObjPoss(const BabyAILevelTask& env) override {
    desc_.FindMatchingObjs(env, false);
  }

  void Validate(const BabyAILevelTask& /*env*/,
                const std::vector<Color>& locked_colors,
                bool unblocking) override {
    if (!unblocking || !desc_.HasType(kKey)) {
      return;
    }
    for (Color color : locked_colors) {
      if (desc_.HasColor(color)) {
        throw BabyAIRejectSampling("cannot refer to a locked-door key");
      }
    }
  }

 protected:
  [[nodiscard]] BabyAIStatus VerifyAction(
      const BabyAILevelTask& env, Act action,
      const WorldObj& /*pre_carrying*/) override {
    if (action != kToggle) {
      return BabyAIStatus::kContinue;
    }
    const WorldObj front_cell =
        env.CellAt(env.FrontCellPos().first, env.FrontCellPos().second);
    if (front_cell.GetType() == kDoor && front_cell.GetDoorOpen() &&
        desc_.HasUid(front_cell.GetUid())) {
      return BabyAIStatus::kSuccess;
    }
    if (strict_ && front_cell.GetType() == kDoor) {
      return BabyAIStatus::kFailure;
    }
    return BabyAIStatus::kContinue;
  }

 private:
  BabyAIObjDesc desc_;
};

class BabyAIGoToInstr : public BabyAIActionInstr {
 public:
  explicit BabyAIGoToInstr(BabyAIObjDesc desc)
      : BabyAIActionInstr(false), desc_(std::move(desc)) {}

  [[nodiscard]] std::string Surface(const BabyAILevelTask& env) override {
    return "go to " + desc_.Surface(env);
  }

  void ResetVerifier(const BabyAILevelTask& env) override {
    BabyAIActionInstr::ResetVerifier(env);
    desc_.FindMatchingObjs(env);
  }

  void UpdateObjPoss(const BabyAILevelTask& env) override {
    desc_.FindMatchingObjs(env, false);
  }

  void Validate(const BabyAILevelTask& /*env*/,
                const std::vector<Color>& locked_colors,
                bool unblocking) override {
    if (!unblocking || !desc_.HasType(kKey)) {
      return;
    }
    for (Color color : locked_colors) {
      if (desc_.HasColor(color)) {
        throw BabyAIRejectSampling("cannot refer to a locked-door key");
      }
    }
  }

 protected:
  [[nodiscard]] BabyAIStatus VerifyAction(
      const BabyAILevelTask& env, Act /*action*/,
      const WorldObj& /*pre_carrying*/) override {
    for (const Pos& pos : desc_.ObjPoss()) {
      if (pos == env.FrontCellPos()) {
        return BabyAIStatus::kSuccess;
      }
    }
    return BabyAIStatus::kContinue;
  }

 private:
  BabyAIObjDesc desc_;
};

class BabyAIPickupInstr : public BabyAIActionInstr {
 public:
  explicit BabyAIPickupInstr(BabyAIObjDesc desc, bool strict = false)
      : BabyAIActionInstr(strict), desc_(std::move(desc)) {}

  [[nodiscard]] std::string Surface(const BabyAILevelTask& env) override {
    return "pick up " + desc_.Surface(env);
  }

  void ResetVerifier(const BabyAILevelTask& env) override {
    BabyAIActionInstr::ResetVerifier(env);
    desc_.FindMatchingObjs(env);
  }

  void UpdateObjPoss(const BabyAILevelTask& env) override {
    desc_.FindMatchingObjs(env, false);
  }

  void Validate(const BabyAILevelTask& /*env*/,
                const std::vector<Color>& locked_colors,
                bool unblocking) override {
    if (!unblocking || !desc_.HasType(kKey)) {
      return;
    }
    for (Color color : locked_colors) {
      if (desc_.HasColor(color)) {
        throw BabyAIRejectSampling("cannot refer to a locked-door key");
      }
    }
  }

 protected:
  [[nodiscard]] BabyAIStatus VerifyAction(
      const BabyAILevelTask& env, Act action,
      const WorldObj& pre_carrying) override {
    if (action != kPickup) {
      return BabyAIStatus::kContinue;
    }
    if (pre_carrying.GetType() == kEmpty &&
        env.Carrying().GetType() != kEmpty &&
        desc_.HasUid(env.Carrying().GetUid())) {
      return BabyAIStatus::kSuccess;
    }
    if (strict_ && env.Carrying().GetType() != kEmpty) {
      return BabyAIStatus::kFailure;
    }
    return BabyAIStatus::kContinue;
  }

 private:
  BabyAIObjDesc desc_;
};

class BabyAIPutNextInstr : public BabyAIActionInstr {
 public:
  BabyAIPutNextInstr(BabyAIObjDesc desc_move, BabyAIObjDesc desc_fixed,
                     bool strict = false)
      : BabyAIActionInstr(strict),
        desc_move_(std::move(desc_move)),
        desc_fixed_(std::move(desc_fixed)) {}

  [[nodiscard]] std::string Surface(const BabyAILevelTask& env) override {
    return "put " + desc_move_.Surface(env) + " next to " +
           desc_fixed_.Surface(env);
  }

  void ResetVerifier(const BabyAILevelTask& env) override {
    BabyAIActionInstr::ResetVerifier(env);
    desc_move_.FindMatchingObjs(env);
    desc_fixed_.FindMatchingObjs(env);
  }

  void UpdateObjPoss(const BabyAILevelTask& env) override {
    desc_move_.FindMatchingObjs(env, false);
    desc_fixed_.FindMatchingObjs(env, false);
  }

  void Validate(const BabyAILevelTask& env,
                const std::vector<Color>& locked_colors,
                bool unblocking) override {
    desc_move_.FindMatchingObjs(env);
    desc_fixed_.FindMatchingObjs(env);
    for (int move_uid : desc_move_.ObjUids()) {
      if (desc_fixed_.HasUid(move_uid)) {
        throw BabyAIRejectSampling(
            "same object matches both PutNext descriptors");
      }
    }
    if (ObjsNext()) {
      throw BabyAIRejectSampling("objects already next to each other");
    }
    if (desc_move_.ObjUids().size() == 1 && desc_fixed_.ObjUids().size() == 1 &&
        desc_move_.ObjUids()[0] == desc_fixed_.ObjUids()[0]) {
      throw BabyAIRejectSampling("cannot move an object next to itself");
    }
    if (!unblocking) {
      return;
    }
    for (Color color : locked_colors) {
      if ((desc_move_.HasType(kKey) && desc_move_.HasColor(color)) ||
          (desc_fixed_.HasType(kKey) && desc_fixed_.HasColor(color))) {
        throw BabyAIRejectSampling("cannot refer to a locked-door key");
      }
    }
  }

  [[nodiscard]] int NumNavsNeeded() const final { return 2; }

 protected:
  [[nodiscard]] BabyAIStatus VerifyAction(
      const BabyAILevelTask& env, Act action,
      const WorldObj& pre_carrying) override {
    if (strict_ && action == kPickup && env.Carrying().GetType() != kEmpty) {
      return BabyAIStatus::kFailure;
    }
    if (action != kDrop) {
      return BabyAIStatus::kContinue;
    }
    if (pre_carrying.GetType() == kEmpty) {
      return BabyAIStatus::kContinue;
    }
    if (!desc_move_.HasUid(pre_carrying.GetUid())) {
      return BabyAIStatus::kContinue;
    }
    for (const Pos& pos_a : desc_move_.ObjPossForUid(pre_carrying.GetUid())) {
      for (const Pos& pos_b : desc_fixed_.ObjPoss()) {
        if (IsNextTo(pos_a, pos_b)) {
          return BabyAIStatus::kSuccess;
        }
      }
    }
    return BabyAIStatus::kContinue;
  }

 private:
  [[nodiscard]] bool ObjsNext() const {
    for (const Pos& pos_a : desc_move_.ObjPoss()) {
      for (const Pos& pos_b : desc_fixed_.ObjPoss()) {
        if (IsNextTo(pos_a, pos_b)) {
          return true;
        }
      }
    }
    return false;
  }

  BabyAIObjDesc desc_move_;
  BabyAIObjDesc desc_fixed_;
};

class BabyAISeqInstr : public BabyAIInstr {
 public:
  BabyAISeqInstr(std::unique_ptr<BabyAIInstr> instr_a,
                 std::unique_ptr<BabyAIInstr> instr_b, bool strict = false)
      : instr_a_(std::move(instr_a)),
        instr_b_(std::move(instr_b)),
        strict_(strict) {}

  void ResetVerifier(const BabyAILevelTask& env) override {
    instr_a_->ResetVerifier(env);
    instr_b_->ResetVerifier(env);
    a_done_ = BabyAIStatus::kContinue;
    b_done_ = BabyAIStatus::kContinue;
  }

  void UpdateObjPoss(const BabyAILevelTask& env) override {
    instr_a_->UpdateObjPoss(env);
    instr_b_->UpdateObjPoss(env);
  }

  void Validate(const BabyAILevelTask& env,
                const std::vector<Color>& locked_colors,
                bool unblocking) override {
    instr_a_->Validate(env, locked_colors, unblocking);
    instr_b_->Validate(env, locked_colors, unblocking);
  }

  [[nodiscard]] int NumNavsNeeded() const override {
    return instr_a_->NumNavsNeeded() + instr_b_->NumNavsNeeded();
  }

 protected:
  std::unique_ptr<BabyAIInstr> instr_a_;
  std::unique_ptr<BabyAIInstr> instr_b_;
  bool strict_{false};
  BabyAIStatus a_done_{BabyAIStatus::kContinue};
  BabyAIStatus b_done_{BabyAIStatus::kContinue};
};

class BabyAIBeforeInstr : public BabyAISeqInstr {
 public:
  using BabyAISeqInstr::BabyAISeqInstr;

  [[nodiscard]] std::string Surface(const BabyAILevelTask& env) override {
    return instr_a_->Surface(env) + ", then " + instr_b_->Surface(env);
  }

  [[nodiscard]] BabyAIStatus Verify(const BabyAILevelTask& env, Act action,
                                    const WorldObj& pre_carrying) override {
    if (a_done_ == BabyAIStatus::kSuccess) {
      b_done_ = instr_b_->Verify(env, action, pre_carrying);
      if (b_done_ == BabyAIStatus::kFailure) {
        return BabyAIStatus::kFailure;
      }
      if (b_done_ == BabyAIStatus::kSuccess) {
        return BabyAIStatus::kSuccess;
      }
    } else {
      a_done_ = instr_a_->Verify(env, action, pre_carrying);
      if (a_done_ == BabyAIStatus::kFailure) {
        return BabyAIStatus::kFailure;
      }
      if (a_done_ == BabyAIStatus::kSuccess) {
        return Verify(env, action, pre_carrying);
      }
      if (strict_ && instr_b_->Verify(env, action, pre_carrying) ==
                         BabyAIStatus::kSuccess) {
        return BabyAIStatus::kFailure;
      }
    }
    return BabyAIStatus::kContinue;
  }
};

class BabyAIAfterInstr : public BabyAISeqInstr {
 public:
  using BabyAISeqInstr::BabyAISeqInstr;

  [[nodiscard]] std::string Surface(const BabyAILevelTask& env) override {
    return instr_a_->Surface(env) + " after you " + instr_b_->Surface(env);
  }

  [[nodiscard]] BabyAIStatus Verify(const BabyAILevelTask& env, Act action,
                                    const WorldObj& pre_carrying) override {
    if (b_done_ == BabyAIStatus::kSuccess) {
      a_done_ = instr_a_->Verify(env, action, pre_carrying);
      if (a_done_ == BabyAIStatus::kSuccess) {
        return BabyAIStatus::kSuccess;
      }
      if (a_done_ == BabyAIStatus::kFailure) {
        return BabyAIStatus::kFailure;
      }
    } else {
      b_done_ = instr_b_->Verify(env, action, pre_carrying);
      if (b_done_ == BabyAIStatus::kFailure) {
        return BabyAIStatus::kFailure;
      }
      if (b_done_ == BabyAIStatus::kSuccess) {
        return Verify(env, action, pre_carrying);
      }
      if (strict_ && instr_a_->Verify(env, action, pre_carrying) ==
                         BabyAIStatus::kSuccess) {
        return BabyAIStatus::kFailure;
      }
    }
    return BabyAIStatus::kContinue;
  }
};

class BabyAIAndInstr : public BabyAISeqInstr {
 public:
  using BabyAISeqInstr::BabyAISeqInstr;

  [[nodiscard]] std::string Surface(const BabyAILevelTask& env) override {
    return instr_a_->Surface(env) + " and " + instr_b_->Surface(env);
  }

  [[nodiscard]] BabyAIStatus Verify(const BabyAILevelTask& env, Act action,
                                    const WorldObj& pre_carrying) override {
    if (a_done_ != BabyAIStatus::kSuccess) {
      a_done_ = instr_a_->Verify(env, action, pre_carrying);
    }
    if (b_done_ != BabyAIStatus::kSuccess) {
      b_done_ = instr_b_->Verify(env, action, pre_carrying);
    }
    if (kUseDoneActions && action == kDone &&
        a_done_ == BabyAIStatus::kFailure &&
        b_done_ == BabyAIStatus::kFailure) {
      return BabyAIStatus::kFailure;
    }
    if (a_done_ == BabyAIStatus::kSuccess &&
        b_done_ == BabyAIStatus::kSuccess) {
      return BabyAIStatus::kSuccess;
    }
    return BabyAIStatus::kContinue;
  }
};

inline void BabyAILevelTask::GenGrid() {
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

inline void BabyAILevelTask::AfterStep(Act act, const WorldObj& /*pre_fwd*/,
                                       const Pos& /*fwd_pos*/,
                                       const WorldObj& pre_carrying,
                                       float* reward, bool* terminated) {
  if (act == kDrop) {
    instrs_->UpdateObjPoss(*this);
  }
  BabyAIStatus status = instrs_->Verify(*this, act, pre_carrying);
  if (status == BabyAIStatus::kSuccess) {
    *reward = 1.0f - 0.9f * (static_cast<float>(step_count_) / max_steps_);
    *terminated = true;
  } else if (status == BabyAIStatus::kFailure) {
    *reward = 0.0f;
    *terminated = true;
  }
}

inline void BabyAILevelTask::AddLockedRoom() {
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

class BabyAILevelGenTask : public BabyAILevelTask {
 public:
  explicit BabyAILevelGenTask(const BabyAITaskConfig& config)
      : BabyAILevelTask(config.env_name, config.room_size, config.num_rows,
                        config.num_cols, config.max_steps,
                        config.mission_bytes),
        num_dists_(config.num_dists),
        locked_room_prob_(config.locked_room_prob),
        locations_(config.locations),
        unblocking_(config.unblocking),
        implicit_unlock_(config.implicit_unlock) {
    action_kinds_ = SplitKinds(config.action_kinds);
    instr_kinds_ = SplitKinds(config.instr_kinds);
  }

 protected:
  void GenMission() override;
  [[nodiscard]] bool UnblockingEnabled() const override { return unblocking_; }

  [[nodiscard]] std::vector<std::string> SplitKinds(
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

  [[nodiscard]] std::unique_ptr<BabyAIInstr> RandActionInstr(
      const std::vector<std::string>& action_kinds);
  [[nodiscard]] std::unique_ptr<BabyAIInstr> RandInstr(
      const std::vector<std::string>& action_kinds,
      const std::vector<std::string>& instr_kinds, int depth = 0);
  [[nodiscard]] BabyAIObjDesc RandObj(
      const std::vector<Type>& types = std::vector<Type>{kDoor, kKey, kBall,
                                                         kBox},
      const std::vector<Color>& colors = std::vector<Color>(kColors.begin(),
                                                            kColors.end()),
      int max_tries = 100);

  int num_dists_{18};
  float locked_room_prob_{0.5f};
  bool locations_{true};
  bool unblocking_{true};
  bool implicit_unlock_{true};
  std::vector<std::string> action_kinds_;
  std::vector<std::string> instr_kinds_;
};

inline BabyAIObjDesc BabyAILevelGenTask::RandObj(
    const std::vector<Type>& types, const std::vector<Color>& colors,
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

inline std::unique_ptr<BabyAIInstr> BabyAILevelGenTask::RandActionInstr(
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

inline std::unique_ptr<BabyAIInstr> BabyAILevelGenTask::RandInstr(
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

inline void BabyAILevelGenTask::GenMission() {
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

inline std::pair<Pos, std::pair<Type, Color>> ObjAtDoor(
    const BabyAILevelTask& env, const Pos& pos) {
  const WorldObj door = env.CellAt(pos.first, pos.second);
  return {pos, {door.GetType(), door.GetColor()}};
}

inline BabyAITaskConfig MakeGoToSeqConfig(BabyAITaskConfig config) {
  config.action_kinds = "goto";
  config.locked_room_prob = 0.0f;
  config.locations = false;
  config.unblocking = false;
  return config;
}

inline BabyAITaskConfig MakePickupLocConfig(BabyAITaskConfig config) {
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

inline BabyAITaskConfig MakeSynthConfig(BabyAITaskConfig config) {
  config.instr_kinds = "action";
  config.locations = false;
  config.unblocking = true;
  config.implicit_unlock = false;
  return config;
}

inline BabyAITaskConfig MakeSynthLocConfig(BabyAITaskConfig config) {
  config.instr_kinds = "action";
  config.locations = true;
  config.unblocking = true;
  config.implicit_unlock = false;
  return config;
}

inline BabyAITaskConfig MakeSynthSeqConfig(BabyAITaskConfig config) {
  config.locations = true;
  config.unblocking = true;
  config.implicit_unlock = false;
  return config;
}

inline BabyAITaskConfig MakeMiniBossConfig(BabyAITaskConfig config) {
  config.num_cols = 2;
  config.num_rows = 2;
  config.room_size = 5;
  config.num_dists = 7;
  config.locked_room_prob = 0.25f;
  return config;
}

inline BabyAITaskConfig MakeBossNoUnlockConfig(BabyAITaskConfig config) {
  config.locked_room_prob = 0.0f;
  config.implicit_unlock = false;
  return config;
}

}  // namespace minigrid
