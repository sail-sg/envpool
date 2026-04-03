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

#include "envpool/minigrid/impl/minigrid_task_factory.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "envpool/minigrid/impl/babyai_env.h"

namespace minigrid {
namespace {

using Config = MiniGridEnvSpec::Config;
using MiniGridTaskFactory = std::function<std::unique_ptr<MiniGridTask>(
    const Config&)>;  // NOLINT(whitespace/indent_namespace)

BabyAITaskConfig MakeBabyAIConfig(const Config& conf) {
  BabyAITaskConfig config;
  config.env_name = conf["env_name"_];
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
  return config;
}

std::unique_ptr<MiniGridTask> MakeObstructedMazeTask(const Config& conf) {
  const std::string env_name = conf["env_name"_];
  return std::make_unique<ObstructedMazeTask>(
      env_name, conf["agent_room"_], conf["key_in_box"_], conf["blocked"_],
      conf["num_quarters"_], conf["max_episode_steps"_],
      env_name == "obstructed_maze_full_v1");
}

const std::unordered_map<std::string, MiniGridTaskFactory>&
MiniGridTaskFactories() {
  static const std::unordered_map<std::string, MiniGridTaskFactory> factories =
      // NOLINTNEXTLINE(whitespace/braces)
      {
          {"empty",
           [](const Config& conf) {
             return std::make_unique<EmptyTask>(
                 conf["size"_], conf["agent_start_pos"_],
                 conf["agent_start_dir"_], conf["max_episode_steps"_],
                 conf["agent_view_size"_]);
           }},
          {"doorkey",
           [](const Config& conf) {
             return std::make_unique<DoorKeyTask>(conf["size"_],
                                                  conf["max_episode_steps"_]);
           }},
          {"distshift",
           [](const Config& conf) {
             return std::make_unique<DistShiftTask>(
                 conf["width"_], conf["height"_], conf["agent_start_pos"_],
                 conf["agent_start_dir"_], conf["strip2_row"_],
                 conf["max_episode_steps"_]);
           }},
          {"lava_gap",
           [](const Config& conf) {
             return std::make_unique<LavaGapTask>(
                 conf["size"_], ParseType(conf["obstacle_type"_]),
                 conf["max_episode_steps"_]);
           }},
          {"crossing",
           [](const Config& conf) {
             return std::make_unique<CrossingTask>(
                 conf["size"_], conf["num_crossings"_],
                 ParseType(conf["obstacle_type"_]), conf["max_episode_steps"_]);
           }},
          {"dynamic_obstacles",
           [](const Config& conf) {
             return std::make_unique<DynamicObstaclesTask>(
                 conf["size"_], conf["agent_start_pos"_],
                 conf["agent_start_dir"_], conf["n_obstacles"_],
                 conf["max_episode_steps"_]);
           }},
          {"fetch",
           [](const Config& conf) {
             return std::make_unique<FetchTask>(
                 conf["size"_], conf["num_objs"_], conf["max_episode_steps"_]);
           }},
          {"goto_door",
           [](const Config& conf) {
             return std::make_unique<GoToDoorTask>(conf["size"_],
                                                   conf["max_episode_steps"_]);
           }},
          {"goto_object",
           [](const Config& conf) {
             return std::make_unique<GoToObjectTask>(
                 conf["size"_], conf["num_objs"_], conf["max_episode_steps"_]);
           }},
          {"put_near",
           [](const Config& conf) {
             return std::make_unique<PutNearTask>(
                 conf["size"_], conf["num_objs"_], conf["max_episode_steps"_]);
           }},
          {"red_blue_door",
           [](const Config& conf) {
             return std::make_unique<RedBlueDoorTask>(
                 conf["size"_], conf["max_episode_steps"_]);
           }},
          {"locked_room",
           [](const Config& conf) {
             return std::make_unique<LockedRoomTask>(
                 conf["size"_], conf["max_episode_steps"_]);
           }},
          {"memory",
           [](const Config& conf) {
             return std::make_unique<MemoryTask>(conf["size"_],
                                                 conf["random_length"_],
                                                 conf["max_episode_steps"_]);
           }},
          {"multi_room",
           [](const Config& conf) {
             return std::make_unique<MultiRoomTask>(
                 conf["min_num_rooms"_], conf["max_num_rooms"_],
                 conf["max_room_size"_], conf["max_episode_steps"_]);
           }},
          {"four_rooms",
           [](const Config& conf) {
             return std::make_unique<FourRoomsTask>(conf["max_episode_steps"_]);
           }},
          {"playground",
           [](const Config& conf) {
             return std::make_unique<PlaygroundTask>(
                 conf["max_episode_steps"_]);
           }},
          {"unlock",
           [](const Config& conf) {
             return std::make_unique<UnlockTask>(conf["max_episode_steps"_]);
           }},
          {"unlock_pickup",
           [](const Config& conf) {
             return std::make_unique<UnlockPickupTask>(
                 conf["max_episode_steps"_]);
           }},
          {"blocked_unlock_pickup",
           [](const Config& conf) {
             return std::make_unique<BlockedUnlockPickupTask>(
                 conf["max_episode_steps"_]);
           }},
          {"key_corridor",
           [](const Config& conf) {
             return std::make_unique<KeyCorridorTask>(
                 conf["num_rows"_], conf["room_size"_],
                 ParseType(conf["obj_type"_]), conf["max_episode_steps"_]);
           }},
          {"obstructed_maze_1dlhb", MakeObstructedMazeTask},
          {"obstructed_maze_full", MakeObstructedMazeTask},
          {"obstructed_maze_full_v1", MakeObstructedMazeTask},
          {"wfc",
           [](const Config& conf) {
             return std::make_unique<WFCTask>(
                 conf["wfc_preset"_], conf["size"_], conf["ensure_connected"_],
                 conf["max_episode_steps"_]);
           }},
      };
  return factories;
}

}  // namespace

std::unique_ptr<MiniGridTask> MakeMiniGridTask(const Config& conf) {
  const std::string env_name = conf["env_name"_];
  if (env_name.rfind("babyai_", 0) == 0) {
    std::unique_ptr<MiniGridTask> task = MakeBabyAITask(MakeBabyAIConfig(conf));
    CHECK(task != nullptr) << "Unknown BabyAI env_name: " << env_name;
    return task;
  }
  const auto& factories = MiniGridTaskFactories();
  const auto it = factories.find(env_name);
  CHECK(it != factories.end()) << "Unknown MiniGrid env_name: " << env_name;
  return it->second(conf);
}

}  // namespace minigrid
