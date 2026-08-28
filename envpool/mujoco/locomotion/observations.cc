/*
 * Copyright 2026 Garena Online Private Limited
 * Copyright 2019-2021 The dm_control Authors.
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

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "envpool/mujoco/locomotion/simulation.h"

namespace mujoco_locomotion {

int ActionSize(Walker walker) {
  if (walker == Walker::kBoxhead) {
    return 3;
  }
  if (walker == Walker::kAnt) {
    return 8;
  }
  if (walker == Walker::kRodent) {
    return 38;
  }
  return 56;
}

std::vector<Observation> ObservationLayout(const std::string& name,
                                           int team_size) {
  const auto task = GetTaskConfig(name);
  const bool soccer = task.task == Task::kSoccer;
  const bool rodent = task.walker == Walker::kRodent;
  const bool cmu =
      task.walker == Walker::kCmu2019 || task.walker == Walker::kCmu2020;
  const bool ant = task.walker == Walker::kAnt;
  int joints = 1;
  if (rodent) {
    joints = 30;
  } else if (cmu) {
    joints = 56;
  } else if (ant) {
    joints = 8;
  }
  const int effectors = task.walker == Walker::kBoxhead ? 3 : 12;
  std::vector<Observation> result;
  auto add = [&](const std::string& key, std::vector<int> shape,
                 int storage = 0, bool boolean = false) {
    if (soccer) {
      shape.insert(shape.begin(), 1);
    }
    const std::string prefix =
        soccer || key == "task_logic" || key.find("reference_props_") == 0
            ? ""
            : "walker/";
    result.push_back({prefix + key, std::move(shape), storage, boolean});
  };
  if (task.task == Task::kTwoTouch) {
    add("task_logic", {1}, 1);
  }
  if (task.task == Task::kTracking) {
    add("reference_props_pos_global", {0});
    add("reference_props_quat_global", {0});
  }
  if (cmu || rodent) {
    int activations = 0;
    if (rodent) {
      activations = 38;
    } else if (task.walker == Walker::kCmu2020) {
      activations = 56;
    }
    add("actuator_activation", {activations});
  }
  if (cmu || rodent || ant) {
    add("appendages_pos", {ant ? 12 : 15});
  }
  if (ant) {
    add("bodies_pos", {39});
    add("bodies_quats", {52});
  }
  add("body_height", {});
  if (!soccer && task.task != Task::kTarget && task.task != Task::kTracking) {
    add("egocentric_camera", {64, 64, 3}, 2);
  }
  add("end_effectors_pos", {effectors});
  add("joints_pos", {joints});
  add("joints_vel", {joints});
  if (soccer) {
    add("prev_action", {ActionSize(task.walker)});
  }
  add("sensors_accelerometer", {3});
  if (!soccer) {
    add("sensors_force", {0});
  }
  add("sensors_gyro", {3});
  if (!soccer) {
    add("sensors_torque", {cmu ? 6 : 0});
    add("sensors_touch", {cmu ? 10 : 4});
  }
  add("sensors_velocimeter", {3});
  if (rodent) {
    add("tendons_pos", {8});
    add("tendons_vel", {8});
  }
  add("world_zaxis", {3});
  if (task.task == Task::kTarget) {
    add("target", {3});
  }
  if (task.task == Task::kBowl) {
    add("origin", {3});
  }
  if (task.task == Task::kTracking) {
    add("reference_rel_joints", {280});
    add("reference_rel_bodies_pos_global", {465});
    add("reference_rel_bodies_quats", {620});
    add("reference_rel_bodies_pos_local", {465});
    add("reference_ego_bodies_quats", {620});
    add("reference_rel_root_quat", {20});
    add("reference_rel_root_pos_local", {15});
    add("reference_appendages_pos", {75});
    add("clip_id", {1}, 1);
    add("velocimeter_control", {3});
    add("gyro_control", {3});
    add("joints_vel_control", {56});
    add("time_in_clip", {1});
  }
  if (soccer) {
    add("ball_ego_angular_velocity", {3});
    add("ball_ego_position", {3});
    add("ball_ego_linear_velocity", {3});
    for (const std::string group : {"teammate", "opponent"}) {
      const int count = group == "teammate" ? team_size - 1 : team_size;
      for (int i = 0; i < count; ++i) {
        const auto prefix = group + "_" + std::to_string(i) + "_";
        add(prefix + "ego_end_effectors_pos", {effectors});
        add(prefix + "ego_linear_velocity", {3});
        add(prefix + "ego_position", {3});
        add(prefix + "ego_orientation", {9});
        add(prefix + "end_effectors_pos", {effectors});
      }
    }
    for (const std::string key :
         {"team_goal_back_right", "team_goal_mid", "team_goal_front_left",
          "field_front_left", "opponent_goal_back_left", "opponent_goal_mid",
          "opponent_goal_front_right", "field_back_right"}) {
      add(key, {key.find("mid") == std::string::npos ? 2 : 3});
    }
    for (const char* key : {"stats_vel_to_ball", "stats_closest_vel_to_ball",
                            "stats_veloc_forward", "stats_vel_ball_to_goal",
                            "stats_home_avg_teammate_dist"}) {
      add(key, {});
    }
    add("stats_teammate_spread_out", {}, 1, true);
    add("stats_home_score", {});
    add("stats_away_score", {});
  }
  std::array<int, 3> offsets{};
  for (auto& observation : result) {
    observation.size =
        std::accumulate(observation.shape.begin(), observation.shape.end(), 1,
                        std::multiplies<>());
    observation.offset = offsets[observation.storage];
    offsets[observation.storage] += observation.size;
  }
  return result;
}

int StorageSize(const std::vector<Observation>& layout, int storage) {
  int size = 0;
  for (const auto& observation : layout) {
    if (observation.storage == storage) {
      size += observation.size;
    }
  }
  return size;
}

std::vector<double> Simulation::RelativePositions(
    const WalkerIds& walker, const std::vector<int>& bodies) const {
  std::vector<double> result(3 * bodies.size());
  const auto* origin = data_->xpos + 3 * walker.root;
  const auto* matrix = data_->xmat + 9 * walker.root;
  for (std::size_t i = 0; i < bodies.size(); ++i) {
    std::array<double, 3> delta;
    mju_sub3(delta.data(), data_->xpos + 3 * bodies[i], origin);
    mju_mulMatTVec3(result.data() + 3 * i, matrix, delta.data());
  }
  return result;
}

std::vector<double> Simulation::WalkerObservation(
    const WalkerIds& walker, const std::string& key) const {
  std::vector<double> result;
  if (key == "body_height") {
    return {data_->xpos[3 * walker.root + 2]};
  }
  if (key == "world_zaxis") {
    return {data_->xmat + 9 * walker.root + 6,
            data_->xmat + 9 * walker.root + 9};
  }
  if (key == "joints_pos" || key == "joints_vel") {
    for (int joint : walker.joints) {
      result.push_back(key == "joints_pos"
                           ? data_->qpos[model_->jnt_qposadr[joint]]
                           : data_->qvel[model_->jnt_dofadr[joint]]);
    }
  } else if (key == "tendons_pos" || key == "tendons_vel") {
    for (int tendon : walker.tendons) {
      result.push_back(key == "tendons_pos" ? data_->ten_length[tendon]
                                            : data_->ten_velocity[tendon]);
    }
  } else if (key == "actuator_activation") {
    for (int actuator : walker.actuators) {
      const int address = model_->actuator_actadr[actuator];
      if (address >= 0) {
        result.push_back(data_->act[address]);
      }
    }
  } else if (key == "end_effectors_pos") {
    for (int sensor : walker.effector_sensors) {
      const auto* values = data_->sensordata + model_->sensor_adr[sensor];
      result.insert(result.end(), values, values + 3);
    }
  } else if (key == "appendages_pos") {
    if (task_.walker == Walker::kAnt) {
      for (int body : walker.effectors) {
        const auto values = SensorValues(
            std::string(mj_id2name(model_.get(), mjOBJ_BODY, body)) +
            "_appendage");
        result.insert(result.end(), values.begin(), values.end());
      }
      return result;
    }
    auto bodies = walker.effectors;
    if (walker.head >= 0) {
      bodies.push_back(walker.head);
    }
    result = RelativePositions(walker, bodies);
  } else if (key == "bodies_pos" || key == "bodies_quats") {
    for (int body : walker.bodies) {
      const auto values = SensorValues(
          std::string(mj_id2name(model_.get(), mjOBJ_BODY, body)) +
          (key == "bodies_pos" ? "_ego_body_pos" : "_ego_body_quat"));
      result.insert(result.end(), values.begin(), values.end());
    }
  } else if (key == "target" || key == "origin") {
    std::array<double, 3> delta;
    if (key == "target") {
      mju_sub3(delta.data(), model_->site_pos + 3 * Id(mjOBJ_SITE, "target"),
               data_->xpos + 3 * walker.root);
    } else {
      mju_scl3(delta.data(), data_->xpos + 3 * walker.root, -1);
    }
    result.resize(3);
    mju_mulMatTVec3(result.data(), data_->xmat + 9 * walker.root, delta.data());
  } else if (key.find("sensors_") == 0) {
    const std::map<std::string, int> types{
        {"sensors_accelerometer", mjSENS_ACCELEROMETER},
        {"sensors_gyro", mjSENS_GYRO},
        {"sensors_velocimeter", mjSENS_VELOCIMETER},
        {"sensors_force", mjSENS_FORCE},
        {"sensors_torque", mjSENS_TORQUE},
        {"sensors_touch", mjSENS_TOUCH}};
    const auto found = walker.sensors.find(types.at(key));
    if (found != walker.sensors.end()) {
      for (int sensor : found->second) {
        for (int i = 0; i < model_->sensor_dim[sensor]; ++i) {
          double value = data_->sensordata[model_->sensor_adr[sensor] + i];
          if (key == "sensors_touch") {
            value = static_cast<double>(value > .001);
          }
          if (key == "sensors_torque") {
            value = std::tanh(2 * value / 60);
          }
          result.push_back(value);
        }
      }
    }
  } else {
    throw std::runtime_error("Unimplemented locomotion observation: " + key);
  }
  return result;
}

void Simulation::Observe() {
  const int stride = StorageSize(layout_, 0);
  for (int player = 0; player < players_; ++player) {
    const auto& walker = walkers_[player];
    for (const auto& observation : layout_) {
      const auto slash = observation.name.find('/');
      const auto key =
          observation.name.substr(slash == std::string::npos ? 0 : slash + 1);
      if (observation.storage == 2) {
        mjvOption option;
        mjv_defaultOption(&option);
        if (task_.walker == Walker::kRodent) {
          option.geomgroup[1] = option.geomgroup[2] = 0;
        } else {
          option.geomgroup[1] = 0;
        }
        Render(64, 64, walker.camera,
               pixels.data() + player * StorageSize(layout_, 2) +
                   observation.offset,
               &option);
      } else if (observation.storage == 0) {
        const auto found = tracking_observations_.find(key);
        std::vector<double> values;
        if (task_.task == Task::kSoccer) {
          values = SoccerObservation(player, key);
        } else if (found != tracking_observations_.end()) {
          values = found->second;
        } else {
          values = WalkerObservation(walker, key);
        }
        if (values.size() != static_cast<std::size_t>(observation.size)) {
          throw std::runtime_error("Wrong native observation shape: " +
                                   observation.name);
        }
        std::copy(values.begin(), values.end(),
                  continuous.begin() + player * stride + observation.offset);
      } else if (key == "task_logic") {
        discrete[observation.offset] = observed_touch_state_;
      } else if (key == "clip_id") {
        discrete[observation.offset] = clip_id_;
      } else if (task_.task == Task::kSoccer) {
        discrete[player * StorageSize(layout_, 1) + observation.offset] =
            static_cast<int64_t>(
                SoccerObservation(player, "stats_home_avg_teammate_dist")[0] >
                5);
      }
    }
  }
}

}  // namespace mujoco_locomotion
