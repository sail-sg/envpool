/*
 * Copyright 2026 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENVPOOL_MUJOCO_LOCOMOTION_SCENE_H_
#define ENVPOOL_MUJOCO_LOCOMOTION_SCENE_H_

#include <mujoco.h>

#include <array>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "envpool/mujoco/locomotion/random.h"
#include "labmaze/cc/random_maze.h"
#include "pugixml.hpp"

namespace mujoco_locomotion {

enum class Walker { kCmu2019, kCmu2020, kRodent, kBoxhead, kAnt };
enum class Task {
  kWalls,
  kGaps,
  kTarget,
  kForage,
  kHeterogeneous,
  kBowl,
  kTwoTouch,
  kTracking,
  kSoccer
};

struct TaskConfig {
  Task task;
  Walker walker;
  double physics_timestep;
  double control_timestep;
  double time_limit;
};

TaskConfig GetTaskConfig(const std::string& name);

std::string Numbers(std::initializer_list<double> values);
std::string Numbers(const double* values, int size);
void Set(pugi::xml_node node, const char* key, const std::string& value);
void Set(pugi::xml_node node, const char* key, double value);
void Set(pugi::xml_node node, const char* key,
         std::initializer_list<double> values);
pugi::xml_node Child(pugi::xml_node parent, const char* tag);
pugi::xml_node Find(pugi::xml_node root, const char* tag,
                    const std::string& name);

// Composer scene assembly stays native, including episode-dependent geometry.
// Files are the unmodified assets fetched from the pinned official repository.
class Scene {
 public:
  Scene(std::string asset_path, std::string labmaze_asset_path);
  pugi::xml_node root() { return document_.child("mujoco"); }
  pugi::xml_node world() { return Child(root(), "worldbody"); }
  pugi::xml_node asset() { return Child(root(), "asset"); }
  void LoadArena(const std::string& name, double timestep);
  void Corridor(const TaskConfig& task, RandomState* random);
  void Floor(double size, bool outdoor = false);
  void MazeArena(const TaskConfig& task, RandomState* random);
  void AddTargets(const TaskConfig& task, RandomState* random);
  void Bowl();
  void TwoTouchTarget();
  void Soccer(const TaskConfig& task, int team_size, bool field_box,
              bool keep_aspect_ratio, bool disable_contacts,
              RandomState* random);
  void AddWalker(Walker walker, const std::string& prefix, int player = -1,
                 bool red = false);
  std::string Xml() const;
  mjModel* Compile() const;

  std::vector<std::string> ground_geoms;
  std::vector<std::array<double, 3>> target_positions;
  std::vector<std::array<double, 3>> spawn_positions;
  std::vector<std::string> targets;
  std::vector<int> target_types;
  std::string maze_entities, maze_variations;
  std::array<double, 2> pitch_size{}, field_size{};
  std::array<double, 3> goal_size{};
  const auto& virtual_files() const { return virtual_files_; }

 private:
  enum class RootJoint { kFixed, kFree, kSlides };
  void Attach(pugi::xml_node model, const std::string& prefix,
              RootJoint joints = RootJoint::kFree);
  void OutdoorTexture();
  void TargetSphere(const std::string& name, double radius, double height,
                    int color, bool two_touch = false);
  void SoccerSensors(Walker walker, int team_size);
  void Detector(const std::string& name, const std::array<double, 3>& position,
                const std::array<double, 3>& size, bool goal, int direction);
  void SoccerBall(bool humanoid, bool field_box);
  void CmuVisuals(pugi::xml_node model, Walker walker, int player, bool red);
  std::string asset_path_;
  std::string labmaze_asset_path_;
  pugi::xml_document document_;
  std::map<std::string, std::vector<unsigned char>> virtual_files_;
  std::unique_ptr<deepmind::labmaze::RandomMaze> maze_;
  std::array<int, 2> target_colors_{0, 1};
  bool maze_initialized_{false};
};

}  // namespace mujoco_locomotion

#endif  // ENVPOOL_MUJOCO_LOCOMOTION_SCENE_H_
