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

#include "envpool/minigrid/impl/minigrid_env.h"

#include <memory>
#include <string>
#include <vector>

#include "envpool/core/py_envpool.h"
#include "envpool/minigrid/impl/babyai_env.h"
#include "envpool/minigrid/minigrid.h"

namespace minigrid {

MiniGridEnv::MiniGridEnv(const Spec& spec, int env_id)
    : Env<MiniGridEnvSpec>(spec, env_id) {
  const auto& conf = spec.config;
  const std::string env_name = conf["env_name"_];
  if (env_name == "empty") {
    task_ = std::make_unique<EmptyTask>(
        conf["size"_], conf["agent_start_pos"_], conf["agent_start_dir"_],
        conf["max_episode_steps"_], conf["agent_view_size"_]);
  } else if (env_name == "doorkey") {
    task_ = std::make_unique<DoorKeyTask>(conf["size"_],
                                          conf["max_episode_steps"_]);
  } else if (env_name == "distshift") {
    task_ = std::make_unique<DistShiftTask>(
        conf["width"_], conf["height"_], conf["agent_start_pos"_],
        conf["agent_start_dir"_], conf["strip2_row"_],
        conf["max_episode_steps"_]);
  } else if (env_name == "lava_gap") {
    task_ = std::make_unique<LavaGapTask>(conf["size"_],
                                          ParseType(conf["obstacle_type"_]),
                                          conf["max_episode_steps"_]);
  } else if (env_name == "crossing") {
    task_ = std::make_unique<CrossingTask>(
        conf["size"_], conf["num_crossings"_],
        ParseType(conf["obstacle_type"_]), conf["max_episode_steps"_]);
  } else if (env_name == "dynamic_obstacles") {
    task_ = std::make_unique<DynamicObstaclesTask>(
        conf["size"_], conf["agent_start_pos"_], conf["agent_start_dir"_],
        conf["n_obstacles"_], conf["max_episode_steps"_]);
  } else if (env_name == "fetch") {
    task_ = std::make_unique<FetchTask>(conf["size"_], conf["num_objs"_],
                                        conf["max_episode_steps"_]);
  } else if (env_name == "goto_door") {
    task_ = std::make_unique<GoToDoorTask>(conf["size"_],
                                           conf["max_episode_steps"_]);
  } else if (env_name == "goto_object") {
    task_ = std::make_unique<GoToObjectTask>(conf["size"_], conf["num_objs"_],
                                             conf["max_episode_steps"_]);
  } else if (env_name == "put_near") {
    task_ = std::make_unique<PutNearTask>(conf["size"_], conf["num_objs"_],
                                          conf["max_episode_steps"_]);
  } else if (env_name == "red_blue_door") {
    task_ = std::make_unique<RedBlueDoorTask>(conf["size"_],
                                              conf["max_episode_steps"_]);
  } else if (env_name == "locked_room") {
    task_ = std::make_unique<LockedRoomTask>(conf["size"_],
                                             conf["max_episode_steps"_]);
  } else if (env_name == "memory") {
    task_ = std::make_unique<MemoryTask>(conf["size"_], conf["random_length"_],
                                         conf["max_episode_steps"_]);
  } else if (env_name == "multi_room") {
    task_ = std::make_unique<MultiRoomTask>(
        conf["min_num_rooms"_], conf["max_num_rooms"_], conf["max_room_size"_],
        conf["max_episode_steps"_]);
  } else if (env_name == "four_rooms") {
    task_ = std::make_unique<FourRoomsTask>(conf["max_episode_steps"_]);
  } else if (env_name == "playground") {
    task_ = std::make_unique<PlaygroundTask>(conf["max_episode_steps"_]);
  } else if (env_name == "unlock") {
    task_ = std::make_unique<UnlockTask>(conf["max_episode_steps"_]);
  } else if (env_name == "unlock_pickup") {
    task_ = std::make_unique<UnlockPickupTask>(conf["max_episode_steps"_]);
  } else if (env_name == "blocked_unlock_pickup") {
    task_ =
        std::make_unique<BlockedUnlockPickupTask>(conf["max_episode_steps"_]);
  } else if (env_name == "key_corridor") {
    task_ = std::make_unique<KeyCorridorTask>(
        conf["num_rows"_], conf["room_size"_], ParseType(conf["obj_type"_]),
        conf["max_episode_steps"_]);
  } else if (env_name.rfind("babyai_", 0) == 0) {
    BabyAITaskConfig config;
    config.env_name = env_name;
    config.room_size = conf["room_size"_];
    config.num_rows = conf["num_rows"_];
    config.num_cols = conf["num_cols"_];
    config.num_dists = conf["num_dists"_];
    config.locked_room_prob = conf["locked_room_prob"_];
    config.locations = conf["locations"_];
    config.unblocking = conf["unblocking"_];
    config.implicit_unlock = conf["implicit_unlock"_];
    config.action_kinds = conf["action_kinds"_];
    config.instr_kinds = conf["instr_kinds"_];
    config.doors_open = conf["doors_open"_];
    config.debug = conf["debug"_];
    config.select_by = conf["select_by"_];
    config.first_color = conf["first_color"_];
    config.second_color = conf["second_color"_];
    config.strict = conf["strict"_];
    config.num_doors = conf["num_doors"_];
    config.num_objs = conf["num_objs"_];
    config.objs_per_room = conf["objs_per_room"_];
    config.start_carrying = conf["start_carrying"_];
    config.distractors = conf["distractors"_];
    config.obj_type = ParseType(conf["obj_type"_]);
    config.max_steps = conf["max_episode_steps"_];
    config.mission_bytes = conf["mission_bytes"_];
    task_ = MakeBabyAITask(config);
    CHECK(task_ != nullptr) << "Unknown BabyAI env_name: " << env_name;
  } else {
    CHECK(env_name == "obstructed_maze_1dlhb" ||
          env_name == "obstructed_maze_full" ||
          env_name == "obstructed_maze_full_v1")
        << "Unknown MiniGrid env_name: " << env_name;
    task_ = std::make_unique<
        ObstructedMazeTask>(  // NOLINT(whitespace/indent_namespace)
        env_name, conf["agent_room"_], conf["key_in_box"_], conf["blocked"_],
        conf["num_quarters"_], conf["max_episode_steps"_],
        env_name == "obstructed_maze_full_v1");
  }
  task_->SetGenerator(&gen_);
}

bool MiniGridEnv::IsDone() { return task_->IsDone(); }

void MiniGridEnv::Reset() {
  task_->Reset();
  WriteState(0.0f);
}

void MiniGridEnv::Step(const Action& action) {
  WriteState(task_->Step(static_cast<Act>(action["action"_])));
}

std::pair<int, int> MiniGridEnv::RenderSize(int width, int height) const {
  return task_->RenderSize(width, height);
}

void MiniGridEnv::Render(int width, int height, int /*camera_id*/,
                         unsigned char* rgb) {
  task_->Render(width, height, rgb);
}

MiniGridDebugState MiniGridEnv::DebugState() const {
  return task_->DebugState();
}

int MiniGridEnv::CurrentMaxEpisodeSteps() const { return task_->MaxSteps(); }

void MiniGridEnv::WriteState(float reward) {
  auto state = Allocate();
  task_->GenImage(state["obs:image"_]);
  task_->WriteMission(state["obs:mission"_]);
  state["obs:direction"_] = task_->AgentDir();
  state["reward"_] = reward;
  state["info:agent_pos"_](0) = task_->AgentPos().first;
  state["info:agent_pos"_](1) = task_->AgentPos().second;
  state["info:mission_id"_] = task_->MissionId();
}

std::vector<MiniGridDebugState> MiniGridEnvPool::DebugStates(
    const std::vector<int>& env_ids) const {
  std::vector<MiniGridDebugState> states;
  states.reserve(env_ids.size());
  for (int env_id : env_ids) {
    states.push_back(envs_[env_id]->DebugState());
  }
  return states;
}

using PyMiniGridEnvSpec = PyEnvSpec<MiniGridEnvSpec>;
using PyMiniGridEnvPool = PyEnvPool<MiniGridEnvPool>;

void BindMiniGrid(py::module_& m) {
  py::class_<MiniGridDebugState>(m, "_MiniGridDebugState")
      .def_readonly("env_name", &MiniGridDebugState::env_name)
      .def_readonly("mission", &MiniGridDebugState::mission)
      .def_readonly("mission_id", &MiniGridDebugState::mission_id)
      .def_readonly("width", &MiniGridDebugState::width)
      .def_readonly("height", &MiniGridDebugState::height)
      .def_readonly("action_max", &MiniGridDebugState::action_max)
      .def_readonly("max_steps", &MiniGridDebugState::max_steps)
      .def_readonly("grid", &MiniGridDebugState::grid)
      .def_readonly("grid_contains", &MiniGridDebugState::grid_contains)
      .def_readonly("obstacle_positions",
                    &MiniGridDebugState::obstacle_positions)
      .def_readonly("agent_pos", &MiniGridDebugState::agent_pos)
      .def_readonly("agent_dir", &MiniGridDebugState::agent_dir)
      .def_readonly("has_carrying", &MiniGridDebugState::has_carrying)
      .def_readonly("carrying_type", &MiniGridDebugState::carrying_type)
      .def_readonly("carrying_color", &MiniGridDebugState::carrying_color)
      .def_readonly("carrying_state", &MiniGridDebugState::carrying_state)
      .def_readonly("carrying_has_contains",
                    &MiniGridDebugState::carrying_has_contains)
      .def_readonly("carrying_contains_type",
                    &MiniGridDebugState::carrying_contains_type)
      .def_readonly("carrying_contains_color",
                    &MiniGridDebugState::carrying_contains_color)
      .def_readonly("carrying_contains_state",
                    &MiniGridDebugState::carrying_contains_state)
      .def_readonly("target_pos", &MiniGridDebugState::target_pos)
      .def_readonly("target_type", &MiniGridDebugState::target_type)
      .def_readonly("target_color", &MiniGridDebugState::target_color)
      .def_readonly("move_pos", &MiniGridDebugState::move_pos)
      .def_readonly("move_type", &MiniGridDebugState::move_type)
      .def_readonly("move_color", &MiniGridDebugState::move_color)
      .def_readonly("success_pos", &MiniGridDebugState::success_pos)
      .def_readonly("failure_pos", &MiniGridDebugState::failure_pos)
      .def_readonly("goal_pos", &MiniGridDebugState::goal_pos);

  py::class_<PyMiniGridEnvSpec>(
      m, "_MiniGridEnvSpec",
      py::metaclass(py::module_::import("abc").attr("ABCMeta")))
      .def(py::init<const PyMiniGridEnvSpec::ConfigValues&>())
      .def_readonly("_config_values", &PyMiniGridEnvSpec::py_config_values)
      .def_readonly("_state_spec", &PyMiniGridEnvSpec::py_state_spec)
      .def_readonly("_action_spec", &PyMiniGridEnvSpec::py_action_spec)
      .def_readonly_static("_state_keys", &PyMiniGridEnvSpec::py_state_keys)
      .def_readonly_static("_action_keys", &PyMiniGridEnvSpec::py_action_keys)
      .def_readonly_static("_config_keys", &PyMiniGridEnvSpec::py_config_keys)
      .def_readonly_static("_default_config_values",
                           &PyMiniGridEnvSpec::py_default_config_values);

  py::class_<PyMiniGridEnvPool>(
      m, "_MiniGridEnvPool",
      py::metaclass(py::module_::import("abc").attr("ABCMeta")))
      .def(py::init<const PyMiniGridEnvSpec&>())
      .def_readonly("_spec", &PyMiniGridEnvPool::py_spec)
      .def("_recv", &PyMiniGridEnvPool::PyRecv)
      .def("_send", &PyMiniGridEnvPool::PySend)
      .def("_reset", &PyMiniGridEnvPool::PyReset)
      .def("_render", &PyMiniGridEnvPool::PyRender)
      .def_readonly_static("_state_keys", &PyMiniGridEnvPool::py_state_keys)
      .def_readonly_static("_action_keys", &PyMiniGridEnvPool::py_action_keys)
      .def("_xla", &PyMiniGridEnvPool::Xla)
      .def("_debug_states", &PyMiniGridEnvPool::DebugStates);
}

}  // namespace minigrid
