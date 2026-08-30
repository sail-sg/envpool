// Copyright 2026 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ENVPOOL_MUJOCO_MJLAB_SIMULATION_H_
#define ENVPOOL_MUJOCO_MJLAB_SIMULATION_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "envpool/mujoco/mjlab/math.h"
#include "envpool/mujoco/mjlab/physics.h"

namespace mjlab {

float Param(const Json& object, const std::string& name, float fallback = 0);
float SquaredParam(const Json& p, const std::string& name);
std::string Name(const Json& object, const std::string& name,
                 const std::string& fallback = "");
bool Flag(const Json& object, const std::string& name, bool fallback = false);
std::string Function(const Json& term);
Json ToJson(const std::vector<float>& values);
template <std::size_t N>
Json ToJson(const std::array<float, N>& values) {
  return ToJson(std::vector<float>(values.begin(), values.end()));
}

struct Entity {
  std::string name;
  int root, mocap{-1};
  std::vector<int> bodies, geoms, sites, joints, controls, qadr, vadr, freeq,
      freev;
  std::vector<float> root_default, joint_default, velocity_default, limits,
      bias;
};

struct Contact {
  struct Slot {
    int address, size;
    std::string field;
  };
  std::vector<Slot> slots;
  std::vector<float> found, force, history, air, last_air, contact,
      last_contact;
  int history_length{0};
  bool track_air{false};
};

struct RaySensor {
  struct Frame {
    std::string type;
    int id;
  };
  std::vector<Frame> frames;
  std::vector<float> offsets, directions, heights;
  std::vector<bool> hits;
  std::string alignment;
  float max_distance;
  bool terrain_height;
};

class Simulation;

// Common manager sequencing is implemented once. These classes contain only
// the task-specific commands, observations, rewards, and curriculum state.
class Task {
 public:
  explicit Task(Simulation* simulation) : sim_(*simulation) {}
  virtual ~Task() = default;
  virtual void Reset() {}
  virtual void Update(bool resetting) {}
  virtual void Curriculum() {}
  virtual std::vector<float> Observation(const std::string& fn, const Json& p);
  virtual float Reward(const std::string& fn, const Json& p, const Json& term);
  virtual bool Terminated(const std::string& fn, const Json& p);
  virtual Json State() const;
  std::vector<float> command;
  float time_left{0};
  int counter{0};

 protected:
  Simulation& sim_;
};

std::unique_ptr<Task> MakeManipulation(Simulation* sim);
std::unique_ptr<Task> MakeVelocity(Simulation* sim);
std::unique_ptr<Task> MakeTracking(Simulation* sim,
                                   const std::string& motion_file);

class Simulation {
 public:
  Simulation(const std::string& asset_path, uint32_t seed, int max_steps,
             const std::string& motion_file, int env_id = 0, int num_envs = 1);
  void Reset();
  void Step(const float* input);
  Json State() const;
  void Observe();
  void Sense();
  void UpdateContacts(bool substep);
  void Events(const std::string& mode);
  void Curriculum();
  void WriteActions(bool resetting);

  Entity& Asset(const Json& params, const std::string& key = "asset_cfg");
  const Entity& Asset(const Json& params,
                      const std::string& key = "asset_cfg") const;
  std::vector<int> Select(const Entity& entity, const Json& params,
                          const std::string& kind,
                          const std::string& key = "asset_cfg") const;
  Vec3 Position(int body) const;
  Quat Orientation(int body) const;
  Vec3 LinearVelocity(int body, int root, const Vec3& position) const;
  Vec3 AngularVelocity(int body) const;
  Vec3 SitePosition(int site) const;
  Vec3 SiteVelocity(int site, int root) const;
  Quat SiteOrientation(int site) const;
  void WritePose(const Entity& entity, const Vec3& pos, const Quat& quat);
  void WriteVelocity(const Entity& entity, const Vec3& lin, const Vec3& ang);
  std::vector<float> JointPositions(const Entity& entity,
                                    bool biased = false) const;
  std::vector<float> JointVelocities(const Entity& entity) const;
  std::vector<float> Sensor(const std::string& name) const;
  std::vector<float> Heights(const std::string& name, float offset = 0) const;
  float Sample(const Json& range, bool tensor_bounds = false);
  float Uniform(const Json& range);
  std::vector<float> SamplePose(const Json& ranges);
  float EvaluateReward(const Json& term);
  bool EvaluateTermination(const Json& term);
  std::vector<float> EvaluateObservation(const Json& term);

  Physics physics;
  Json cfg;
  Random random;
  const int kEnvId, kNumEnvs;
  std::map<std::string, Entity> entities;
  std::map<std::string, Contact> contacts;
  std::map<std::string, RaySensor> rays;
  std::unique_ptr<Task> task;
  Vec3 origin{};
  std::vector<float> action, previous_action, previous_previous_action;
  std::vector<float> action_scale, action_offset;
  std::vector<int> action_joints, action_controls;
  bool position_action{false};
  std::map<std::string, std::vector<float>> observations;
  std::map<std::string, float> reward_terms, episode_rewards, event_timers;
  std::map<std::string, float> metrics;
  int steps{0}, total_steps{0}, max_steps, decimation;
  float physics_dt, step_dt, reward{0};
  bool terminated{false}, truncated{false}, initialized{false};
  std::string action_entity;
};

}  // namespace mjlab

#endif  // ENVPOOL_MUJOCO_MJLAB_SIMULATION_H_
