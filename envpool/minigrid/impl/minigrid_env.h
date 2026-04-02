/*
 * Copyright 2026 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENVPOOL_MINIGRID_IMPL_MINIGRID_ENV_H_
#define ENVPOOL_MINIGRID_IMPL_MINIGRID_ENV_H_

#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/array.h"
#include "envpool/minigrid/impl/utils.h"

namespace minigrid {

struct MiniGridDebugState {
  std::string env_name;
  std::string mission;
  int mission_id{-1};
  int width{0};
  int height{0};
  int action_max{6};
  int max_steps{0};
  std::vector<uint8_t> grid;
  std::vector<uint8_t> grid_contains;
  std::vector<int> obstacle_positions;
  Pos agent_pos{-1, -1};
  int agent_dir{0};
  bool has_carrying{false};
  int carrying_type{static_cast<int>(kEmpty)};
  int carrying_color{static_cast<int>(kRed)};
  int carrying_state{0};
  bool carrying_has_contains{false};
  int carrying_contains_type{static_cast<int>(kEmpty)};
  int carrying_contains_color{static_cast<int>(kRed)};
  int carrying_contains_state{0};
  Pos target_pos{-1, -1};
  int target_type{static_cast<int>(kEmpty)};
  int target_color{static_cast<int>(kRed)};
  Pos move_pos{-1, -1};
  int move_type{static_cast<int>(kEmpty)};
  int move_color{static_cast<int>(kRed)};
  Pos success_pos{-1, -1};
  Pos failure_pos{-1, -1};
  Pos goal_pos{-1, -1};
};

class MiniGridTask {
 protected:
  using RejectFn = std::function<bool(const Pos&)>;

  int width_{0};
  int height_{0};
  int max_steps_{100};
  int action_max_{6};
  int step_count_{0};
  int agent_view_size_{7};
  int mission_bytes_{kMissionBytes};
  bool see_through_walls_{false};
  bool done_{true};
  std::string env_name_;
  std::mt19937* gen_ref_{nullptr};
  std::vector<std::vector<WorldObj>> grid_;
  WorldObj carrying_;
  Pos agent_pos_{-1, -1};
  int agent_dir_{0};
  std::string mission_;
  int mission_id_{-1};
  Pos target_pos_{-1, -1};
  Type target_type_{kEmpty};
  Color target_color_{kRed};
  Pos move_pos_{-1, -1};
  Type move_type_{kEmpty};
  Color move_color_{kRed};
  Pos success_pos_{-1, -1};
  Pos failure_pos_{-1, -1};
  Pos goal_pos_{-1, -1};

 public:
  MiniGridTask(std::string env_name, int max_steps, int agent_view_size,
               bool see_through_walls, int action_max = 6,
               int mission_bytes = kMissionBytes);
  virtual ~MiniGridTask() = default;

  void SetGenerator(std::mt19937* gen_ref) { gen_ref_ = gen_ref; }
  void Reset();
  float Step(Act act);
  [[nodiscard]] bool IsDone() const { return done_; }
  [[nodiscard]] int AgentDir() const { return agent_dir_; }
  [[nodiscard]] Pos AgentPos() const { return agent_pos_; }
  [[nodiscard]] const std::string& Mission() const { return mission_; }
  [[nodiscard]] int MissionId() const { return mission_id_; }
  [[nodiscard]] int ActionMax() const { return action_max_; }
  [[nodiscard]] int MaxSteps() const { return max_steps_; }
  void GenImage(const Array& obs) const;
  void WriteMission(const Array& obs) const;
  [[nodiscard]] virtual MiniGridDebugState DebugState() const;
  [[nodiscard]] std::pair<int, int> RenderSize(int width, int height) const;
  void Render(int width, int height, unsigned char* rgb) const;

 protected:
  virtual void GenGrid() = 0;
  virtual Act MapAction(Act act) const { return act; }
  virtual void BeforeStep(Act act, const WorldObj& pre_fwd) {}
  virtual void AfterStep(Act act, const WorldObj& pre_fwd, const Pos& fwd_pos,
                         const WorldObj& pre_carrying, float* reward,
                         bool* terminated) {}

  void SetMission(const std::string& mission, int mission_id = -1) {
    mission_ = mission;
    mission_id_ = mission_id;
  }

  [[nodiscard]] bool InBounds(int x, int y) const;
  [[nodiscard]] WorldObj GetCell(int x, int y) const;
  WorldObj& Cell(int x, int y);
  void SetCell(int x, int y, const WorldObj& obj);
  void SetEmpty(int x, int y);
  void ClearGrid(int width, int height);
  void HorzWall(int x, int y, int length = -1, Type type = kWall,
                Color color = kGrey);
  void VertWall(int x, int y, int length = -1, Type type = kWall,
                Color color = kGrey);
  void WallRect(int x, int y, int width, int height);
  void PutObj(const WorldObj& obj, int x, int y);
  Pos PlaceObj(const WorldObj& obj, int top_x = 0, int top_y = 0,
               int size_x = -1, int size_y = -1,
               const RejectFn& reject_fn = RejectFn(),
               int max_tries = std::numeric_limits<int>::max());
  Pos PlaceAgent(int top_x = 0, int top_y = 0, int size_x = -1, int size_y = -1,
                 bool rand_dir = true,
                 int max_tries = std::numeric_limits<int>::max());
  [[nodiscard]] int RandInt(int low, int high);
  [[nodiscard]] bool RandBool();
  [[nodiscard]] float RandFloat(float low, float high);
  template <typename T>
  const T& RandElem(const std::vector<T>& values) {
    CHECK(!values.empty());
    return values[RandInt(0, static_cast<int>(values.size()))];
  }
  template <typename T>
  std::vector<T> RandSubset(std::vector<T> values, int num) {
    CHECK_LE(num, static_cast<int>(values.size()));
    std::vector<T> out;
    while (static_cast<int>(out.size()) < num) {
      int idx = RandInt(0, static_cast<int>(values.size()));
      out.push_back(values[idx]);
      values.erase(values.begin() + idx);
    }
    return out;
  }
  [[nodiscard]] Color RandColor() {
    return RandElem(std::vector<Color>(kColors.begin(), kColors.end()));
  }
  [[nodiscard]] Pos FrontPos() const;
  [[nodiscard]] Pos DirVec() const;
  [[nodiscard]] Pos RightVec() const;

 private:
  int next_uid_{1};
  WorldObj PrepareObj(const WorldObj& obj);
};

struct Room {
  Pos top{0, 0};
  Pos size{0, 0};
  std::array<bool, 4> connected{{false, false, false, false}};
  std::array<bool, 4> has_neighbor{{false, false, false, false}};
  std::array<Pos, 4> door_pos{Pos{-1, -1}, Pos{-1, -1}, Pos{-1, -1},
                              Pos{-1, -1}};
  std::array<Pos, 4> neighbors{Pos{-1, -1}, Pos{-1, -1}, Pos{-1, -1},
                               Pos{-1, -1}};
  bool locked{false};
  std::vector<std::pair<Type, Color>> objs;

  [[nodiscard]] bool PosInside(int x, int y) const {
    return x >= top.first && y >= top.second && x < top.first + size.first &&
           y < top.second + size.second;
  }
};

class RoomGridTask : public MiniGridTask {
 protected:
  int room_size_{7};
  int num_rows_{3};
  int num_cols_{3};
  std::vector<std::vector<Room>> room_grid_;

 public:
  RoomGridTask(std::string env_name, int room_size, int num_rows, int num_cols,
               int max_steps, int agent_view_size = 7,
               int mission_bytes = kMissionBytes);

 protected:
  void GenGrid() override;
  Room& GetRoom(int i, int j);
  const Room& GetRoom(int i, int j) const;
  Room& RoomFromPos(int x, int y);
  const Room& RoomFromPos(int x, int y) const;
  std::pair<Pos, std::pair<Type, Color>> AddObject(int i, int j,
                                                   Type type = kEmpty,
                                                   Color color = kUnassigned);
  std::vector<std::pair<Pos, std::pair<Type, Color>>> AddDistractors(
      int i = -1, int j = -1, int num_distractors = 10, bool all_unique = true);
  Pos AddDoor(int i, int j, int door_idx = -1, Color color = kUnassigned,
              bool locked = false);
  void RemoveWall(int i, int j, int wall_idx);
  Pos PlaceAgentInRoom(int i = -1, int j = -1, bool rand_dir = true);
  [[nodiscard]] bool TryConnectAll(const std::vector<Color>& door_colors =
                                       std::vector<Color>(kColors.begin(),
                                                          kColors.end()),
                                   int max_itrs = 5000);
  void ConnectAll();
  [[nodiscard]] bool CheckObjsReachable() const;
};

class EmptyTask : public MiniGridTask {
 public:
  EmptyTask(int size, Pos agent_start_pos, int agent_start_dir, int max_steps,
            int agent_view_size);

 protected:
  void GenGrid() override;

 private:
  int size_;
  Pos agent_start_pos_;
  int agent_start_dir_;
};

class DoorKeyTask : public MiniGridTask {
 public:
  explicit DoorKeyTask(int size, int max_steps);

 protected:
  void GenGrid() override;

 private:
  int size_;
};

class DistShiftTask : public MiniGridTask {
 public:
  DistShiftTask(int width, int height, Pos agent_start_pos, int agent_start_dir,
                int strip2_row, int max_steps);

 protected:
  void GenGrid() override;

 private:
  Pos agent_start_pos_;
  int agent_start_dir_;
  int strip2_row_;
};

class LavaGapTask : public MiniGridTask {
 public:
  LavaGapTask(int size, Type obstacle_type, int max_steps);

 protected:
  void GenGrid() override;

 private:
  int size_;
  Type obstacle_type_;
};

class CrossingTask : public MiniGridTask {
 public:
  CrossingTask(int size, int num_crossings, Type obstacle_type, int max_steps);

 protected:
  void GenGrid() override;

 private:
  int size_;
  int num_crossings_;
  Type obstacle_type_;
};

class DynamicObstaclesTask : public MiniGridTask {
 public:
  DynamicObstaclesTask(int size, Pos agent_start_pos, int agent_start_dir,
                       int n_obstacles, int max_steps);

 protected:
  void GenGrid() override;
  void BeforeStep(Act act, const WorldObj& pre_fwd) override;
  void AfterStep(Act act, const WorldObj& pre_fwd, const Pos& fwd_pos,
                 const WorldObj& pre_carrying, float* reward,
                 bool* terminated) override;
  [[nodiscard]] MiniGridDebugState DebugState() const override;

 private:
  int size_;
  Pos agent_start_pos_;
  int agent_start_dir_;
  int n_obstacles_;
  bool pre_front_blocked_{false};
  std::vector<Pos> obstacle_pos_;
};

class FetchTask : public MiniGridTask {
 public:
  FetchTask(int size, int num_objs, int max_steps);

 protected:
  void GenGrid() override;
  void AfterStep(Act act, const WorldObj& pre_fwd, const Pos& fwd_pos,
                 const WorldObj& pre_carrying, float* reward,
                 bool* terminated) override;

 private:
  int size_;
  int num_objs_;
};

class GoToDoorTask : public MiniGridTask {
 public:
  GoToDoorTask(int size, int max_steps);

 protected:
  void GenGrid() override;
  void AfterStep(Act act, const WorldObj& pre_fwd, const Pos& fwd_pos,
                 const WorldObj& pre_carrying, float* reward,
                 bool* terminated) override;

 private:
  int size_;
};

class GoToObjectTask : public MiniGridTask {
 public:
  GoToObjectTask(int size, int num_objs, int max_steps);

 protected:
  void GenGrid() override;
  void AfterStep(Act act, const WorldObj& pre_fwd, const Pos& fwd_pos,
                 const WorldObj& pre_carrying, float* reward,
                 bool* terminated) override;

 private:
  int size_;
  int num_objs_;
};

class PutNearTask : public MiniGridTask {
 public:
  PutNearTask(int size, int num_objs, int max_steps);

 protected:
  void GenGrid() override;
  void AfterStep(Act act, const WorldObj& pre_fwd, const Pos& fwd_pos,
                 const WorldObj& pre_carrying, float* reward,
                 bool* terminated) override;

 private:
  int size_;
  int num_objs_;
};

class RedBlueDoorTask : public MiniGridTask {
 public:
  RedBlueDoorTask(int size, int max_steps);

 protected:
  void GenGrid() override;
  void BeforeStep(Act act, const WorldObj& pre_fwd) override;
  void AfterStep(Act act, const WorldObj& pre_fwd, const Pos& fwd_pos,
                 const WorldObj& pre_carrying, float* reward,
                 bool* terminated) override;

 private:
  int size_;
  Pos red_door_pos_{-1, -1};
  Pos blue_door_pos_{-1, -1};
  bool red_open_before_{false};
  bool blue_open_before_{false};
};

class LockedRoomTask : public MiniGridTask {
 public:
  LockedRoomTask(int size, int max_steps);

 protected:
  void GenGrid() override;

 private:
  int size_;
};

class MemoryTask : public MiniGridTask {
 public:
  MemoryTask(int size, bool random_length, int max_steps);

 protected:
  Act MapAction(Act act) const override;
  void GenGrid() override;
  void AfterStep(Act act, const WorldObj& pre_fwd, const Pos& fwd_pos,
                 const WorldObj& pre_carrying, float* reward,
                 bool* terminated) override;

 private:
  int size_;
  bool random_length_;
};

class MultiRoomTask : public MiniGridTask {
 public:
  MultiRoomTask(int min_num_rooms, int max_num_rooms, int max_room_size,
                int max_steps);

 protected:
  void GenGrid() override;

 private:
  struct MultiRoomDesc {
    Pos top{0, 0};
    Pos size{0, 0};
    Pos entry_door_pos{-1, -1};
    Pos exit_door_pos{-1, -1};
  };

  int min_num_rooms_;
  int max_num_rooms_;
  int max_room_size_;
  bool PlaceRoom(int num_left, std::vector<MultiRoomDesc>* rooms, int min_size,
                 int max_size, int entry_wall, const Pos& entry_door_pos);
};

class FourRoomsTask : public MiniGridTask {
 public:
  explicit FourRoomsTask(int max_steps);

 protected:
  void GenGrid() override;
};

class PlaygroundTask : public MiniGridTask {
 public:
  explicit PlaygroundTask(int max_steps);

 protected:
  void GenGrid() override;
};

class UnlockTask : public RoomGridTask {
 public:
  explicit UnlockTask(int max_steps);

 protected:
  void GenGrid() override;
  void AfterStep(Act act, const WorldObj& pre_fwd, const Pos& fwd_pos,
                 const WorldObj& pre_carrying, float* reward,
                 bool* terminated) override;
};

class UnlockPickupTask : public RoomGridTask {
 public:
  explicit UnlockPickupTask(int max_steps);

 protected:
  void GenGrid() override;
  void AfterStep(Act act, const WorldObj& pre_fwd, const Pos& fwd_pos,
                 const WorldObj& pre_carrying, float* reward,
                 bool* terminated) override;
};

class BlockedUnlockPickupTask : public RoomGridTask {
 public:
  explicit BlockedUnlockPickupTask(int max_steps);

 protected:
  void GenGrid() override;
  void AfterStep(Act act, const WorldObj& pre_fwd, const Pos& fwd_pos,
                 const WorldObj& pre_carrying, float* reward,
                 bool* terminated) override;
};

class KeyCorridorTask : public RoomGridTask {
 public:
  KeyCorridorTask(int num_rows, int room_size, Type obj_type, int max_steps);

 protected:
  void GenGrid() override;
  void AfterStep(Act act, const WorldObj& pre_fwd, const Pos& fwd_pos,
                 const WorldObj& pre_carrying, float* reward,
                 bool* terminated) override;

 private:
  Type obj_type_;
};

class ObstructedMazeTask : public RoomGridTask {
 public:
  ObstructedMazeTask(std::string env_name, Pos agent_room, bool key_in_box,
                     bool blocked, int num_quarters, int max_steps, bool v1);

 protected:
  void GenGrid() override;
  void AfterStep(Act act, const WorldObj& pre_fwd, const Pos& fwd_pos,
                 const WorldObj& pre_carrying, float* reward,
                 bool* terminated) override;

 private:
  Pos agent_room_;
  bool key_in_box_;
  bool blocked_;
  int num_quarters_;
  bool v1_;
  std::vector<Color> door_colors_;
  void AddObstructedDoor(int i, int j, int door_idx, Color color, bool locked,
                         bool key_in_box, bool blocked, bool add_key);
  void AddKeyToRoom(int i, int j, Color color, bool key_in_box);
};

}  // namespace minigrid

#endif  // ENVPOOL_MINIGRID_IMPL_MINIGRID_ENV_H_
