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
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "envpool/mujoco/locomotion/simulation.h"

namespace mujoco_locomotion {

void Simulation::InitializeSoccer() {
  for (int attempt = 0; attempt <= 100; ++attempt) {
    double* ball = data_->qpos + model_->jnt_qposadr[ball_joint_];
    ball[0] =
        random_.Uniform(-scene_.pitch_size[0] * .6, scene_.pitch_size[0] * .6);
    ball[1] =
        random_.Uniform(-scene_.pitch_size[1] * .6, scene_.pitch_size[1] * .6);
    ball[2] = .5;
    mju_zero(data_->qvel + model_->jnt_dofadr[ball_joint_], 6);
    for (int player = 0; player < players_; ++player) {
      const auto& walker = walkers_[player];
      ResetWalker(player);
      const double x = random_.Uniform(-scene_.pitch_size[0] * .6,
                                       scene_.pitch_size[0] * .6);
      const double y = random_.Uniform(-scene_.pitch_size[1] * .6,
                                       scene_.pitch_size[1] * .6);
      const double rotation = random_.Uniform(-std::acos(-1), std::acos(-1));
      if (walker.freejoint >= 0) {
        double* qpos = data_->qpos + model_->jnt_qposadr[walker.freejoint];
        const double norm = std::sqrt(mju_dot(qpos + 3, qpos + 3, 4));
        for (int i = 3; i < 7; ++i) qpos[i] /= norm;
        qpos[0] = x;
        qpos[1] = y;
        ShiftWalker(player, {0, 0, 0}, rotation);
        mju_zero(data_->qvel + model_->jnt_dofadr[walker.freejoint], 6);
      } else {
        data_->qpos[model_->jnt_qposadr[Id(mjOBJ_JOINT,
                                           walker.prefix + "root_x/")]] = x;
        data_->qpos[model_->jnt_qposadr[Id(mjOBJ_JOINT,
                                           walker.prefix + "root_y/")]] = y;
        // BoxHead.set_pose extracts yaw from the quaternion and leaves the
        // translational root joints independent of its steering hinge.
        const double c = std::cos(rotation / 2), s = std::sin(rotation / 2);
        data_->qpos[model_->jnt_qposadr[Id(mjOBJ_JOINT,
                                           walker.prefix + "steer")]] =
            std::atan2(2 * c * s, 1 - 2 * s * s);
      }
    }
    mj_forward(model_.get(), data_.get());
    bool retry = false;
    for (int i = 0; i < data_->ncon; ++i) {
      const auto& contact = data_->contact[i];
      const int a = geom_player_[contact.geom1],
                b = geom_player_[contact.geom2];
      // The pinned UniformInitializer's ball set contains a Binding rather
      // than a geom ID, so its retry loop only rejects player-player contacts.
      if (a >= 0 && b >= 0 && a != b) {
        retry = true;
        break;
      }
    }
    if (!retry) return;
  }
  throw std::runtime_error("Soccer initializer exceeded 100 collision retries");
}

void Simulation::BeforeSoccerStep() {
  if (off_court_) {
    double* qpos = data_->qpos + model_->jnt_qposadr[ball_joint_];
    qpos[0] = data_->geom_xpos[3 * ball_geom_] * random_.Uniform(.7, .9);
    qpos[1] = data_->geom_xpos[3 * ball_geom_ + 1] * random_.Uniform(.7, .9);
    qpos[2] = .5;
    mju_zero(data_->qvel + model_->jnt_dofadr[ball_joint_], 6);
  }
  goals_.fill(false);
  scoring_team_ = -1;
}

void Simulation::SoccerDetections() {
  const double* ball = data_->geom_xpos + 3 * ball_geom_;
  for (int detector = 0; detector < 3; ++detector) {
    const std::string name = detector == 0   ? "home_goal/"
                             : detector == 1 ? "away_goal/"
                                             : "field/";
    if (detector == 2 && options_.enable_field_box) continue;
    const auto* lower = model_->site_pos + 3 * Id(mjOBJ_SITE, name + "lower");
    const auto* upper = model_->site_pos + 3 * Id(mjOBJ_SITE, name + "upper");
    bool inside = true;
    for (int axis = 0; axis < (detector == 2 ? 2 : 3); ++axis) {
      inside = inside && lower[axis] < ball[axis] && ball[axis] < upper[axis];
    }
    const bool detected = detector == 2 ? !inside : inside;
    const bool previous = detector == 2 ? off_court_ : goals_now_[detector];
    if (detector == 2) {
      off_court_ = detected;
    } else {
      goals_[detector] = goals_[detector] || detected;
      goals_now_[detector] = detected;
    }
    if (detected == previous) continue;
    const float rgba[4]{detected        ? 0.f
                        : detector == 0 ? .2f
                                        : 1.f,
                        detected        ? 1.f
                        : detector == 2 ? 1.f
                                        : .2f,
                        detected        ? 0.f
                        : detector == 1 ? .2f
                                        : 1.f,
                        detected        ? .25f
                        : detector == 2 ? 1.f
                                        : .5f};
    std::copy_n(
        rgba, 4,
        model_->site_rgba + 4 * Id(mjOBJ_SITE, name + "detection_zone"));
    if (detector < 2) {
      for (int geom = 0; geom < model_->ngeom; ++geom) {
        const char* geom_name = mj_id2name(model_.get(), mjOBJ_GEOM, geom);
        if (geom_name && std::string(geom_name).find(name) == 0) {
          std::copy_n(rgba, 3, model_->geom_rgba + 4 * geom);
          model_->geom_rgba[4 * geom + 3] = 1;
        }
      }
    }
  }
  scoring_team_ = goals_[0] ? 1 : goals_[1] ? 0 : -1;
}

void Simulation::AfterSoccerStep() {
  if (scoring_team_ >= 0 && !options_.terminate_on_goal) InitializeSoccer();
  for (int player = 0; player < players_; ++player) {
    const int team = player / options_.team_size;
    rewards[player] = scoring_team_ < 0 ? 0 : team == scoring_team_ ? 1 : -1;
  }
  success_ = scoring_team_ >= 0 && options_.terminate_on_goal;
  discount = success_ ? 0 : 1;
}

std::vector<double> Simulation::SensorValues(const std::string& name) const {
  const int sensor = Id(mjOBJ_SENSOR, name);
  const double* values = data_->sensordata + model_->sensor_adr[sensor];
  return {values, values + model_->sensor_dim[sensor]};
}

std::vector<double> Simulation::SoccerObservation(
    int player, const std::string& key) const {
  const auto& walker = walkers_[player];
  if (key == "prev_action") {
    const auto start =
        previous_actions_.begin() + player * ActionSize(task_.walker);
    return {start, start + ActionSize(task_.walker)};
  }
  if (key.find("ball_ego_") == 0) {
    return SensorValues(walker.prefix +
                        (key == "ball_ego_angular_velocity" ? "ball_ego_angvel"
                         : key == "ball_ego_position" ? "ball_ego_pos"
                                                      : "ball_ego_linvel"));
  }
  if ((key.find("teammate_") == 0 || key.find("opponent_") == 0) &&
      key[9] >= '0' && key[9] <= '9') {
    const bool same = key.find("teammate_") == 0;
    const auto separator = key.find('_', 9);
    const int index = std::stoi(key.substr(9, separator - 9));
    const auto suffix = key.substr(separator + 1);
    const auto prefix = key.substr(0, separator);
    int other =
        same ? (player / options_.team_size) * options_.team_size + index
             : (1 - player / options_.team_size) * options_.team_size + index;
    if (same && other >= player) ++other;
    if (suffix == "end_effectors_pos")
      return WalkerObservation(walkers_[other], suffix);
    std::vector<double> result;
    auto append = [&](const std::string& sensor) {
      const auto values = SensorValues(walker.prefix + sensor);
      result.insert(result.end(), values.begin(), values.end());
    };
    if (suffix == "ego_end_effectors_pos") {
      for (int body : walkers_[other].effectors) {
        const std::string full = mj_id2name(model_.get(), mjOBJ_BODY, body);
        append(full.substr(walkers_[other].prefix.size()) + "_" + prefix +
               "_end_effector");
      }
    } else if (suffix == "ego_orientation") {
      for (char axis : std::string("xyz")) append(key + "_" + axis);
    } else {
      append(key);
    }
    return result;
  }
  const std::vector<std::string> arena_keys{
      "team_goal_back_right",      "team_goal_mid",
      "team_goal_front_left",      "field_front_left",
      "opponent_goal_back_left",   "opponent_goal_mid",
      "opponent_goal_front_right", "field_back_right"};
  const auto arena_key = std::find(arena_keys.begin(), arena_keys.end(), key);
  if (arena_key != arena_keys.end()) {
    int index = std::distance(arena_keys.begin(), arena_key);
    const int dimension = index == 1 || index == 5 ? 3 : 2;
    if (player >= options_.team_size) index = (index + 4) % 8;
    const char* sites[]{"home_goal/lower", "home_goal/mid",   "home_goal/upper",
                        "field/upper",     "away_goal/upper", "away_goal/mid",
                        "away_goal/lower", "field/lower"};
    const double* position =
        model_->site_pos + 3 * Id(mjOBJ_SITE, sites[index]);
    const double* origin = data_->xpos + 3 * walker.root;
    const double* matrix = data_->xmat + 9 * walker.root;
    std::vector<double> result(dimension, 0);
    for (int j = 0; j < dimension; ++j) {
      for (int i = 0; i < dimension; ++i)
        result[j] += (position[i] - origin[i]) * matrix[3 * i + j];
    }
    return result;
  }
  if (key.find("stats_") != 0) return WalkerObservation(walker, key);
  const double* position = data_->xpos + 3 * walker.root;
  const double* ball = data_->geom_xpos + 3 * ball_geom_;
  const int team = player / options_.team_size;
  if (key == "stats_home_score")
    return {static_cast<double>(scoring_team_ == team)};
  if (key == "stats_away_score")
    return {static_cast<double>(scoring_team_ >= 0 && scoring_team_ != team)};
  if (key == "stats_veloc_forward")
    return {WalkerObservation(walker, "sensors_velocimeter")[0]};
  if (key == "stats_home_avg_teammate_dist") {
    double sum = 0;
    for (int other = team * options_.team_size;
         other < (team + 1) * options_.team_size; ++other) {
      if (other == player) continue;
      double delta[3];
      mju_sub3(delta, position, data_->xpos + 3 * walkers_[other].root);
      sum += mju_norm3(delta);
    }
    return {options_.team_size == 1 ? 0 : sum / (options_.team_size - 1)};
  }
  if (key == "stats_vel_ball_to_goal") {
    double delta[3];
    mju_sub3(delta,
             model_->site_pos +
                 3 * Id(mjOBJ_SITE, team ? "home_goal/mid" : "away_goal/mid"),
             ball);
    const double norm = mju_norm3(delta);
    if (norm)
      for (double& value : delta) value /= norm;
    const auto velocity = SensorValues("soccer_ball/linear_velocity");
    return {mju_dot3(delta, velocity.data())};
  }
  if (key == "stats_closest_vel_to_ball") {
    double nearest = std::numeric_limits<double>::infinity();
    int closest = -1;
    for (int other = team * options_.team_size;
         other < (team + 1) * options_.team_size; ++other) {
      double delta[3];
      mju_sub3(delta, ball, data_->xpos + 3 * walkers_[other].root);
      const double distance = mju_norm3(delta);
      if (distance < nearest) {
        nearest = distance;
        closest = other;
      }
    }
    if (closest != player) return {0};
  }
  if (key == "stats_closest_vel_to_ball" || key == "stats_vel_to_ball") {
    const double dx = ball[0] - position[0], dy = ball[1] - position[1];
    const double norm = std::sqrt(dx * dx + dy * dy) + 1e-7;
    const double* velocity = data_->cvel + 6 * walker.root + 3;
    return {(dx / norm) * velocity[0] + (dy / norm) * velocity[1]};
  }
  throw std::runtime_error("Unknown soccer observation " + key);
}

}  // namespace mujoco_locomotion
