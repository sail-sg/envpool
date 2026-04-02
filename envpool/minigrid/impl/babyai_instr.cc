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
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "envpool/minigrid/impl/babyai_core.h"

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

void ValidateNoLockedDoorKeyRef(const BabyAIObjDesc& desc,
                                const std::vector<Color>& locked_colors,
                                bool unblocking) {
  if (!unblocking || !desc.HasType(kKey)) {
    return;
  }
  for (Color color : locked_colors) {
    if (desc.HasColor(color)) {
      throw BabyAIRejectSampling("cannot refer to a locked-door key");
    }
  }
}

}  // namespace

BabyAIObjDesc::BabyAIObjDesc(std::optional<Type> type,
                             std::optional<Color> color, BabyAILoc loc)
    : type_(type), color_(color), loc_(loc) {}

std::string BabyAIObjDesc::Surface(const BabyAILevelTask& env) {
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

void BabyAIObjDesc::FindMatchingObjs(const BabyAILevelTask& env,
                                     bool use_location) {
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
      obj_poss_.emplace_back(x, y);
      obj_pos_uids_.push_back(cell.GetUid());
    }
  }
}

bool BabyAIObjDesc::HasType(Type type) const {
  return type_.has_value() && type_.value() == type;
}

bool BabyAIObjDesc::HasColor(Color color) const {
  return color_.has_value() && color_.value() == color;
}

bool BabyAIObjDesc::HasUid(int uid) const {
  return std::find(obj_uids_.begin(), obj_uids_.end(), uid) != obj_uids_.end();
}

std::vector<Pos> BabyAIObjDesc::ObjPossForUid(int uid) const {
  std::vector<Pos> poss;
  for (std::size_t i = 0; i < obj_poss_.size(); ++i) {
    if (obj_pos_uids_[i] == uid) {
      poss.push_back(obj_poss_[i]);
    }
  }
  return poss;
}

void BabyAIActionInstr::ResetVerifier(const BabyAILevelTask& env) {
  BabyAIInstr::ResetVerifier(env);
  last_step_match_ = false;
}

BabyAIStatus BabyAIActionInstr::Verify(const BabyAILevelTask& env, Act action,
                                       const WorldObj& pre_carrying) {
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

int BabyAIActionInstr::NumNavsNeeded() const { return 1; }

BabyAIActionInstr::BabyAIActionInstr(bool strict) : strict_(strict) {}

BabyAIOpenInstr::BabyAIOpenInstr(BabyAIObjDesc desc, bool strict)
    : BabyAIActionInstr(strict), desc_(std::move(desc)) {}

std::string BabyAIOpenInstr::Surface(const BabyAILevelTask& env) {
  return "open " + desc_.Surface(env);
}

void BabyAIOpenInstr::ResetVerifier(const BabyAILevelTask& env) {
  BabyAIActionInstr::ResetVerifier(env);
  desc_.FindMatchingObjs(env);
}

void BabyAIOpenInstr::UpdateObjPoss(const BabyAILevelTask& env) {
  desc_.FindMatchingObjs(env, false);
}

void BabyAIOpenInstr::Validate(const BabyAILevelTask& /*env*/,
                               const std::vector<Color>& locked_colors,
                               bool unblocking) {
  ValidateNoLockedDoorKeyRef(desc_, locked_colors, unblocking);
}

BabyAIStatus BabyAIOpenInstr::VerifyAction(const BabyAILevelTask& env,
                                           Act action,
                                           const WorldObj& /*pre_carrying*/) {
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

BabyAIGoToInstr::BabyAIGoToInstr(BabyAIObjDesc desc)
    : BabyAIActionInstr(false), desc_(std::move(desc)) {}

std::string BabyAIGoToInstr::Surface(const BabyAILevelTask& env) {
  return "go to " + desc_.Surface(env);
}

void BabyAIGoToInstr::ResetVerifier(const BabyAILevelTask& env) {
  BabyAIActionInstr::ResetVerifier(env);
  desc_.FindMatchingObjs(env);
}

void BabyAIGoToInstr::UpdateObjPoss(const BabyAILevelTask& env) {
  desc_.FindMatchingObjs(env, false);
}

void BabyAIGoToInstr::Validate(const BabyAILevelTask& /*env*/,
                               const std::vector<Color>& locked_colors,
                               bool unblocking) {
  ValidateNoLockedDoorKeyRef(desc_, locked_colors, unblocking);
}

BabyAIStatus BabyAIGoToInstr::VerifyAction(const BabyAILevelTask& env,
                                           Act /*action*/,
                                           const WorldObj& /*pre_carrying*/) {
  for (const Pos& pos : desc_.ObjPoss()) {
    if (pos == env.FrontCellPos()) {
      return BabyAIStatus::kSuccess;
    }
  }
  return BabyAIStatus::kContinue;
}

BabyAIPickupInstr::BabyAIPickupInstr(BabyAIObjDesc desc, bool strict)
    : BabyAIActionInstr(strict), desc_(std::move(desc)) {}

std::string BabyAIPickupInstr::Surface(const BabyAILevelTask& env) {
  return "pick up " + desc_.Surface(env);
}

void BabyAIPickupInstr::ResetVerifier(const BabyAILevelTask& env) {
  BabyAIActionInstr::ResetVerifier(env);
  desc_.FindMatchingObjs(env);
}

void BabyAIPickupInstr::UpdateObjPoss(const BabyAILevelTask& env) {
  desc_.FindMatchingObjs(env, false);
}

void BabyAIPickupInstr::Validate(const BabyAILevelTask& /*env*/,
                                 const std::vector<Color>& locked_colors,
                                 bool unblocking) {
  ValidateNoLockedDoorKeyRef(desc_, locked_colors, unblocking);
}

BabyAIStatus BabyAIPickupInstr::VerifyAction(const BabyAILevelTask& env,
                                             Act action,
                                             const WorldObj& pre_carrying) {
  if (action != kPickup) {
    return BabyAIStatus::kContinue;
  }
  if (pre_carrying.GetType() == kEmpty && env.Carrying().GetType() != kEmpty &&
      desc_.HasUid(env.Carrying().GetUid())) {
    return BabyAIStatus::kSuccess;
  }
  if (strict_ && env.Carrying().GetType() != kEmpty) {
    return BabyAIStatus::kFailure;
  }
  return BabyAIStatus::kContinue;
}

BabyAIPutNextInstr::BabyAIPutNextInstr(BabyAIObjDesc desc_move,
                                       BabyAIObjDesc desc_fixed, bool strict)
    : BabyAIActionInstr(strict),
      desc_move_(std::move(desc_move)),
      desc_fixed_(std::move(desc_fixed)) {
}  // NOLINT(whitespace/indent_namespace)

std::string BabyAIPutNextInstr::Surface(const BabyAILevelTask& env) {
  return "put " + desc_move_.Surface(env) + " next to " +
         desc_fixed_.Surface(env);
}

void BabyAIPutNextInstr::ResetVerifier(const BabyAILevelTask& env) {
  BabyAIActionInstr::ResetVerifier(env);
  pre_carrying_ = WorldObj();
  desc_move_.FindMatchingObjs(env);
  desc_fixed_.FindMatchingObjs(env);
}

void BabyAIPutNextInstr::UpdateObjPoss(const BabyAILevelTask& env) {
  desc_move_.FindMatchingObjs(env, false);
  desc_fixed_.FindMatchingObjs(env, false);
}

void BabyAIPutNextInstr::Validate(const BabyAILevelTask& env,
                                  const std::vector<Color>& locked_colors,
                                  bool unblocking) {
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
  ValidateNoLockedDoorKeyRef(desc_move_, locked_colors, unblocking);
  ValidateNoLockedDoorKeyRef(desc_fixed_, locked_colors, unblocking);
}

int BabyAIPutNextInstr::NumNavsNeeded() const { return 2; }

BabyAIStatus BabyAIPutNextInstr::VerifyAction(const BabyAILevelTask& env,
                                              Act action, const WorldObj&
                                              /*pre_carrying*/) {
  const WorldObj pre_carrying = pre_carrying_;
  pre_carrying_ = env.Carrying();
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

bool BabyAIPutNextInstr::ObjsNext() const {
  for (const Pos& pos_a : desc_move_.ObjPoss()) {
    for (const Pos& pos_b : desc_fixed_.ObjPoss()) {
      if (IsNextTo(pos_a, pos_b)) {
        return true;
      }
    }
  }
  return false;
}

BabyAISeqInstr::BabyAISeqInstr(std::unique_ptr<BabyAIInstr> instr_a,
                               std::unique_ptr<BabyAIInstr> instr_b,
                               bool strict)
    : instr_a_(std::move(instr_a)),
      instr_b_(std::move(instr_b)),
      strict_(strict) {}  // NOLINT(whitespace/indent_namespace)

void BabyAISeqInstr::ResetVerifier(const BabyAILevelTask& env) {
  instr_a_->ResetVerifier(env);
  instr_b_->ResetVerifier(env);
  a_done_ = BabyAIStatus::kContinue;
  b_done_ = BabyAIStatus::kContinue;
}

void BabyAISeqInstr::UpdateObjPoss(const BabyAILevelTask& env) {
  instr_a_->UpdateObjPoss(env);
  instr_b_->UpdateObjPoss(env);
}

void BabyAISeqInstr::Validate(const BabyAILevelTask& env,
                              const std::vector<Color>& locked_colors,
                              bool unblocking) {
  instr_a_->Validate(env, locked_colors, unblocking);
  instr_b_->Validate(env, locked_colors, unblocking);
}

int BabyAISeqInstr::NumNavsNeeded() const {
  return instr_a_->NumNavsNeeded() + instr_b_->NumNavsNeeded();
}

std::string BabyAIBeforeInstr::Surface(const BabyAILevelTask& env) {
  return instr_a_->Surface(env) + ", then " + instr_b_->Surface(env);
}

BabyAIStatus BabyAIBeforeInstr::Verify(const BabyAILevelTask& env, Act action,
                                       const WorldObj& pre_carrying) {
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
    if (strict_ &&
        instr_b_->Verify(env, action, pre_carrying) == BabyAIStatus::kSuccess) {
      return BabyAIStatus::kFailure;
    }
  }
  return BabyAIStatus::kContinue;
}

std::string BabyAIAfterInstr::Surface(const BabyAILevelTask& env) {
  return instr_a_->Surface(env) + " after you " + instr_b_->Surface(env);
}

BabyAIStatus BabyAIAfterInstr::Verify(const BabyAILevelTask& env, Act action,
                                      const WorldObj& pre_carrying) {
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
    if (strict_ &&
        instr_a_->Verify(env, action, pre_carrying) == BabyAIStatus::kSuccess) {
      return BabyAIStatus::kFailure;
    }
  }
  return BabyAIStatus::kContinue;
}

std::string BabyAIAndInstr::Surface(const BabyAILevelTask& env) {
  return instr_a_->Surface(env) + " and " + instr_b_->Surface(env);
}

BabyAIStatus BabyAIAndInstr::Verify(const BabyAILevelTask& env, Act action,
                                    const WorldObj& pre_carrying) {
  if (a_done_ != BabyAIStatus::kSuccess) {
    a_done_ = instr_a_->Verify(env, action, pre_carrying);
  }
  if (b_done_ != BabyAIStatus::kSuccess) {
    b_done_ = instr_b_->Verify(env, action, pre_carrying);
  }
  if (kUseDoneActions && action == kDone && a_done_ == BabyAIStatus::kFailure &&
      b_done_ == BabyAIStatus::kFailure) {
    return BabyAIStatus::kFailure;
  }
  if (a_done_ == BabyAIStatus::kSuccess && b_done_ == BabyAIStatus::kSuccess) {
    return BabyAIStatus::kSuccess;
  }
  return BabyAIStatus::kContinue;
}

}  // namespace minigrid
