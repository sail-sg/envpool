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

#ifndef ENVPOOL_MINIGRID_IMPL_BABYAI_CORE_H_
#define ENVPOOL_MINIGRID_IMPL_BABYAI_CORE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "envpool/minigrid/impl/babyai_env.h"

namespace minigrid {

class BabyAIRejectSampling : public std::runtime_error {
 public:
  explicit BabyAIRejectSampling(const std::string& msg);
};

enum class BabyAIStatus : std::uint8_t { kContinue, kSuccess, kFailure };

enum class BabyAILoc : std::uint8_t { kNone, kLeft, kRight, kFront, kBehind };

class BabyAIInstr;

class BabyAILevelTask : public RoomGridTask {
 public:
  BabyAILevelTask(std::string env_name, int room_size, int num_rows,
                  int num_cols, int max_steps, int mission_bytes);

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

  std::pair<Pos, std::pair<Type, Color>> AddExistingObject(int i, int j,
                                                           const WorldObj& obj);
  std::vector<std::pair<Pos, std::pair<Type, Color>>> AddDistractorsOrReject(
      int i = -1, int j = -1, int num_distractors = 10, bool all_unique = true);
  void ConnectAllOrReject(const std::vector<Color>& door_colors =
                              std::vector<Color>(kColors.begin(),
                                                 kColors.end()));
  void CheckObjsReachableOrReject() const;
  void OpenAllDoors();
  void AddLockedRoom();

  std::unique_ptr<BabyAIInstr> instrs_;
  const Room* locked_room_{nullptr};
  bool fixed_max_steps_{false};
};

class BabyAIObjDesc {
 public:
  BabyAIObjDesc(std::optional<Type> type, std::optional<Color> color,
                BabyAILoc loc = BabyAILoc::kNone);

  [[nodiscard]] std::string Surface(const BabyAILevelTask& env);
  void FindMatchingObjs(const BabyAILevelTask& env, bool use_location = true);
  [[nodiscard]] bool HasType(Type type) const;
  [[nodiscard]] bool HasColor(Color color) const;
  [[nodiscard]] bool HasUid(int uid) const;
  [[nodiscard]] const std::vector<Pos>& ObjPoss() const { return obj_poss_; }
  [[nodiscard]] const std::vector<int>& ObjUids() const { return obj_uids_; }
  [[nodiscard]] std::vector<Pos> ObjPossForUid(int uid) const;

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
  void ResetVerifier(const BabyAILevelTask& env) override;
  [[nodiscard]] BabyAIStatus Verify(const BabyAILevelTask& env, Act action,
                                    const WorldObj& pre_carrying) final;
  [[nodiscard]] int NumNavsNeeded() const override;

 protected:
  explicit BabyAIActionInstr(bool strict = false);

  [[nodiscard]] virtual BabyAIStatus VerifyAction(
      const BabyAILevelTask& env, Act action, const WorldObj& pre_carrying) = 0;

  bool strict_{false};
  bool last_step_match_{false};
};

class BabyAIOpenInstr : public BabyAIActionInstr {
 public:
  explicit BabyAIOpenInstr(BabyAIObjDesc desc, bool strict = false);

  [[nodiscard]] std::string Surface(const BabyAILevelTask& env) override;
  void ResetVerifier(const BabyAILevelTask& env) override;
  void UpdateObjPoss(const BabyAILevelTask& env) override;
  void Validate(const BabyAILevelTask& env,
                const std::vector<Color>& locked_colors,
                bool unblocking) override;

 protected:
  [[nodiscard]] BabyAIStatus VerifyAction(
      const BabyAILevelTask& env, Act action,
      const WorldObj& pre_carrying) override;

 private:
  BabyAIObjDesc desc_;
};

class BabyAIGoToInstr : public BabyAIActionInstr {
 public:
  explicit BabyAIGoToInstr(BabyAIObjDesc desc);

  [[nodiscard]] std::string Surface(const BabyAILevelTask& env) override;
  void ResetVerifier(const BabyAILevelTask& env) override;
  void UpdateObjPoss(const BabyAILevelTask& env) override;
  void Validate(const BabyAILevelTask& env,
                const std::vector<Color>& locked_colors,
                bool unblocking) override;

 protected:
  [[nodiscard]] BabyAIStatus VerifyAction(
      const BabyAILevelTask& env, Act action,
      const WorldObj& pre_carrying) override;

 private:
  BabyAIObjDesc desc_;
};

class BabyAIPickupInstr : public BabyAIActionInstr {
 public:
  explicit BabyAIPickupInstr(BabyAIObjDesc desc, bool strict = false);

  [[nodiscard]] std::string Surface(const BabyAILevelTask& env) override;
  void ResetVerifier(const BabyAILevelTask& env) override;
  void UpdateObjPoss(const BabyAILevelTask& env) override;
  void Validate(const BabyAILevelTask& env,
                const std::vector<Color>& locked_colors,
                bool unblocking) override;

 protected:
  [[nodiscard]] BabyAIStatus VerifyAction(
      const BabyAILevelTask& env, Act action,
      const WorldObj& pre_carrying) override;

 private:
  BabyAIObjDesc desc_;
};

class BabyAIPutNextInstr : public BabyAIActionInstr {
 public:
  BabyAIPutNextInstr(BabyAIObjDesc desc_move, BabyAIObjDesc desc_fixed,
                     bool strict = false);

  [[nodiscard]] std::string Surface(const BabyAILevelTask& env) override;
  void ResetVerifier(const BabyAILevelTask& env) override;
  void UpdateObjPoss(const BabyAILevelTask& env) override;
  void Validate(const BabyAILevelTask& env,
                const std::vector<Color>& locked_colors,
                bool unblocking) override;
  [[nodiscard]] int NumNavsNeeded() const final;

 protected:
  [[nodiscard]] BabyAIStatus VerifyAction(
      const BabyAILevelTask& env, Act action,
      const WorldObj& pre_carrying) override;

 private:
  [[nodiscard]] bool ObjsNext() const;

  BabyAIObjDesc desc_move_;
  BabyAIObjDesc desc_fixed_;
  WorldObj pre_carrying_;
};

class BabyAISeqInstr : public BabyAIInstr {
 public:
  BabyAISeqInstr(std::unique_ptr<BabyAIInstr> instr_a,
                 std::unique_ptr<BabyAIInstr> instr_b, bool strict = false);

  void ResetVerifier(const BabyAILevelTask& env) override;
  void UpdateObjPoss(const BabyAILevelTask& env) override;
  void Validate(const BabyAILevelTask& env,
                const std::vector<Color>& locked_colors,
                bool unblocking) override;
  [[nodiscard]] int NumNavsNeeded() const override;

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

  [[nodiscard]] std::string Surface(const BabyAILevelTask& env) override;
  [[nodiscard]] BabyAIStatus Verify(const BabyAILevelTask& env, Act action,
                                    const WorldObj& pre_carrying) override;
};

class BabyAIAfterInstr : public BabyAISeqInstr {
 public:
  using BabyAISeqInstr::BabyAISeqInstr;

  [[nodiscard]] std::string Surface(const BabyAILevelTask& env) override;
  [[nodiscard]] BabyAIStatus Verify(const BabyAILevelTask& env, Act action,
                                    const WorldObj& pre_carrying) override;
};

class BabyAIAndInstr : public BabyAISeqInstr {
 public:
  using BabyAISeqInstr::BabyAISeqInstr;

  [[nodiscard]] std::string Surface(const BabyAILevelTask& env) override;
  [[nodiscard]] BabyAIStatus Verify(const BabyAILevelTask& env, Act action,
                                    const WorldObj& pre_carrying) override;
};

class BabyAILevelGenTask : public BabyAILevelTask {
 public:
  explicit BabyAILevelGenTask(const BabyAITaskConfig& config);

 protected:
  void GenMission() override;
  [[nodiscard]] bool UnblockingEnabled() const override { return unblocking_; }

  [[nodiscard]] std::vector<std::string> SplitKinds(
      const std::string& csv) const;
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

std::pair<Pos, std::pair<Type, Color>> ObjAtDoor(const BabyAILevelTask& env,
                                                 const Pos& pos);
BabyAITaskConfig MakeGoToSeqConfig(BabyAITaskConfig config);
BabyAITaskConfig MakePickupLocConfig(BabyAITaskConfig config);
BabyAITaskConfig MakeSynthConfig(BabyAITaskConfig config);
BabyAITaskConfig MakeSynthLocConfig(BabyAITaskConfig config);
BabyAITaskConfig MakeSynthSeqConfig(BabyAITaskConfig config);
BabyAITaskConfig MakeMiniBossConfig(BabyAITaskConfig config);
BabyAITaskConfig MakeBossNoUnlockConfig(BabyAITaskConfig config);

}  // namespace minigrid

#endif  // ENVPOOL_MINIGRID_IMPL_BABYAI_CORE_H_
