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

#include "envpool/minigrid/impl/babyai_env.h"

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

namespace minigrid {
namespace {

constexpr std::array<Pos, 4> kDirToVec = {
    Pos{1, 0}, Pos{0, 1}, Pos{-1, 0},
    Pos{0, -1}};  // NOLINT(whitespace/indent_namespace)

bool UseDoneActions() { return std::getenv("BABYAI_DONE_ACTIONS") != nullptr; }

const bool kUseDoneActions = UseDoneActions();

bool IsNextTo(const Pos& lhs, const Pos& rhs) {
  return std::abs(lhs.first - rhs.first) + std::abs(lhs.second - rhs.second) ==
         1;
}

int Dot(const Pos& lhs, const Pos& rhs) {
  return lhs.first * rhs.first + lhs.second * rhs.second;
}

class BabyAIRejectSampling : public std::runtime_error {
 public:
  explicit BabyAIRejectSampling(const std::string& msg)
      : std::runtime_error(msg) {}
};

enum class BabyAIStatus { kContinue, kSuccess, kFailure };

enum class BabyAILoc { kNone, kLeft, kRight, kFront, kBehind };

std::string BabyAILocSurface(BabyAILoc loc) {
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
    *reward = 1.0f - 0.9f * (static_cast<float>(step_count_) / max_steps_);
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
