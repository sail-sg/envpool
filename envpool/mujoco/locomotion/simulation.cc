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

#include "envpool/mujoco/locomotion/simulation.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "third_party/dmc_locomotion/metadata.h"

namespace mujoco_locomotion {

Simulation::Simulation(Options options)
    : options_(std::move(options)),
      task_(GetTaskConfig(options_.task)),
      random_(options_.seed),
      scene_random_(options_.seed),
      scene_(options_.asset_path, options_.labmaze_asset_path),
      layout_(ObservationLayout(options_.task, options_.team_size)),
      players_(task_.task == Task::kSoccer ? options_.team_size * 2 : 1) {
  if (options_.team_size < 1 || options_.team_size > 11) {
    throw std::invalid_argument("Soccer team_size must be in [1, 11]");
  }
  continuous.resize(players_ * StorageSize(layout_, 0));
  discrete.resize(players_ * StorageSize(layout_, 1));
  pixels.resize(players_ * StorageSize(layout_, 2));
  rewards.resize(players_);
  previous_actions_.resize(players_ * ActionSize(task_.walker));
  if (task_.task == Task::kTracking ||
      (task_.task == Task::kSoccer && task_.walker == Walker::kCmu2019)) {
    mocap_ =
        MocapData::Load(options_.mocap_asset_path +
                        (task_.task == Task::kTracking ? "/mocap_2020.bin"
                                                       : "/mocap_2019.bin"));
  }
}

Simulation::~Simulation() = default;

int Simulation::Id(mjtObj object, const std::string& name) const {
  const int id = mj_name2id(model_.get(), object, name.c_str());
  if (id < 0)
    throw std::runtime_error("Missing locomotion model element " + name);
  return id;
}

void Simulation::Compile() {
  renderer_.reset();
  data_.reset();
  model_.reset(scene_.Compile());
  data_.reset(mj_makeData(model_.get()));
  if (!data_) throw std::runtime_error("Cannot allocate locomotion physics");
  CacheIds();
}

void Simulation::CacheIds() {
  walkers_.clear();
  ground_geoms_.clear();
  target_geoms_.clear();
  target_materials_.clear();
  target_activated_.assign(scene_.targets.size(), false);
  target_rewarded_.assign(scene_.targets.size(), false);
  for (const auto& name : scene_.targets) {
    target_geoms_.push_back(Id(mjOBJ_GEOM, name + "geom"));
    target_materials_.push_back(
        Id(mjOBJ_MATERIAL,
           name + (task_.task == Task::kTwoTouch ? "target_sphere_init"
                                                 : "target_sphere")));
  }
  hand_geoms_.clear();
  for (const auto& name : scene_.ground_geoms)
    ground_geoms_.push_back(Id(mjOBJ_GEOM, name));
  for (int player = 0; player < players_; ++player) {
    WalkerIds walker{};
    walker.prefix = players_ == 1
                        ? "walker/"
                        : (player < options_.team_size ? "home" : "away") +
                              std::to_string(player % options_.team_size) + "/";
    const bool cmu =
        task_.walker == Walker::kCmu2019 || task_.walker == Walker::kCmu2020;
    const bool rodent = task_.walker == Walker::kRodent;
    const bool boxhead = task_.walker == Walker::kBoxhead;
    const std::string torso = cmu ? "root" : boxhead ? "head_body" : "torso";
    walker.root = Id(mjOBJ_BODY, walker.prefix + torso);
    walker.frame = Id(mjOBJ_BODY, walker.prefix);
    walker.freejoint = boxhead ? -1 : Id(mjOBJ_JOINT, walker.prefix);
    walker.head = cmu || rodent
                      ? Id(mjOBJ_BODY, walker.prefix + (cmu ? "head" : "skull"))
                      : -1;
    walker.pelvis = rodent ? Id(mjOBJ_BODY, walker.prefix + "pelvis") : -1;
    walker.camera = Id(mjOBJ_CAMERA, walker.prefix + "egocentric");
    const std::vector<std::string> effectors =
        cmu ? std::vector<std::string>{"rradius", "lradius", "rfoot", "lfoot"}
        : rodent ? std::vector<std::string>{"lower_arm_R", "lower_arm_L",
                                            "foot_R", "foot_L"}
        : boxhead
            ? std::vector<std::string>{"head_body"}
            : std::vector<std::string>{"front_left_foot", "front_right_foot",
                                       "back_right_foot", "back_left_foot"};
    for (const auto& name : effectors) {
      walker.effectors.push_back(Id(mjOBJ_BODY, walker.prefix + name));
      walker.effector_sensors.push_back(
          Id(mjOBJ_SENSOR, walker.prefix + name + "_end_effector"));
    }
    auto belongs = [&](mjtObj object, int index) {
      const char* name = mj_id2name(model_.get(), object, index);
      return name && std::string(name).find(walker.prefix) == 0;
    };
    for (int i = 0; i < model_->nu; ++i) {
      if (!belongs(mjOBJ_ACTUATOR, i)) continue;
      walker.actuators.push_back(i);
      if (model_->actuator_trntype[i] == mjTRN_JOINT) {
        const int joint = model_->actuator_trnid[2 * i];
        if (!boxhead ||
            std::string(mj_id2name(model_.get(), mjOBJ_JOINT, joint)) ==
                walker.prefix + "kick") {
          walker.joints.push_back(joint);
        }
      }
    }
    for (int i = 0; i < model_->ntendon; ++i) {
      if (belongs(mjOBJ_TENDON, i)) walker.tendons.push_back(i);
    }
    for (int i = 0; i < model_->nsensor; ++i) {
      if (belongs(mjOBJ_SENSOR, i))
        walker.sensors[model_->sensor_type[i]].push_back(i);
    }
    for (int i = 0; i < model_->nbody; ++i) {
      if (i != walker.frame && belongs(mjOBJ_BODY, i))
        walker.bodies.push_back(i);
    }
    std::vector<int> feet;
    if (cmu) {
      feet = {Id(mjOBJ_BODY, walker.prefix + "lfoot"),
              Id(mjOBJ_BODY, walker.prefix + "rfoot")};
    } else if (rodent) {
      for (const char* name :
           {"foot_L", "foot_R", "hand_L", "hand_R", "vertebra_C1"})
        feet.push_back(Id(mjOBJ_BODY, walker.prefix + name));
    } else if (!boxhead) {
      feet = walker.effectors;
    }
    for (int geom = 0; geom < model_->ngeom; ++geom) {
      int body = model_->geom_bodyid[geom];
      while (body > 0 && body != walker.frame) {
        if (task_.task == Task::kTwoTouch &&
            (body == Id(mjOBJ_BODY, walker.prefix + "hand_L") ||
             body == Id(mjOBJ_BODY, walker.prefix + "hand_R"))) {
          hand_geoms_.push_back(geom);
        }
        if (std::find(feet.begin(), feet.end(), body) != feet.end()) {
          walker.ground_contact_geoms.push_back(geom);
          break;
        }
        body = model_->body_parentid[body];
      }
    }
    if (boxhead)
      walker.ground_contact_geoms.push_back(
          Id(mjOBJ_GEOM, walker.prefix + "shell"));
    walkers_.push_back(std::move(walker));
  }
  if (task_.task == Task::kSoccer) {
    ball_joint_ = Id(mjOBJ_JOINT, "soccer_ball/");
    ball_geom_ = Id(mjOBJ_GEOM, "soccer_ball/geom");
    geom_player_.assign(model_->ngeom, -1);
    for (int geom = 0; geom < model_->ngeom; ++geom) {
      const char* name = mj_id2name(model_.get(), mjOBJ_GEOM, geom);
      if (!name) continue;
      for (int player = 0; player < players_; ++player) {
        if (std::string(name).find(walkers_[player].prefix) == 0)
          geom_player_[geom] = player;
      }
    }
  }
}

void Simulation::ResetWalker(int index) {
  const auto& walker = walkers_[index];
  for (int joint = 0; joint < model_->njnt; ++joint) {
    const char* name = mj_id2name(model_.get(), mjOBJ_JOINT, joint);
    if (name && model_->jnt_bodyid[joint] != walker.frame &&
        std::string(name).find(walker.prefix) == 0) {
      const int count = model_->jnt_type[joint] == mjJNT_BALL ? 4 : 1;
      mju_copy(data_->qpos + model_->jnt_qposadr[joint],
               model_->qpos0 + model_->jnt_qposadr[joint], count);
    }
  }
  if (walker.freejoint < 0) {
    for (char axis : std::string("xyz")) {
      const int joint = Id(mjOBJ_JOINT, walker.prefix + "root_" + axis + "/");
      data_->qpos[model_->jnt_qposadr[joint]] = 0;
      data_->qvel[model_->jnt_dofadr[joint]] = 0;
    }
    const int joint = Id(mjOBJ_JOINT, walker.prefix + "steer");
    data_->qpos[model_->jnt_qposadr[joint]] = 0;
    data_->qvel[model_->jnt_dofadr[joint]] = 0;
    return;
  }
  double* position = data_->qpos + model_->jnt_qposadr[walker.freejoint];
  mju_copy(position, model_->qpos0 + model_->jnt_qposadr[walker.freejoint], 7);
  mju_zero(data_->qvel + model_->jnt_dofadr[walker.freejoint], 6);
  if (task_.walker == Walker::kCmu2019 || task_.walker == Walker::kCmu2020) {
    position[0] = position[1] = 0;
    position[2] = task_.walker == Walker::kCmu2019 ? .94 : 1.143;
    position[3] = .859;
    position[4] = 1;
    position[5] = 1;
    position[6] = .859;
    const double norm = std::sqrt(mju_dot(position + 3, position + 3, 4));
    for (int i = 3; i < 7; ++i) position[i] /= norm;
  }
  if (task_.task == Task::kSoccer && mocap_) {
    const auto& clip = mocap_->clips[0];
    const int frame = random_.Int(clip.frames);
    for (std::size_t i = 0; i < kCmuMocapJoints.size(); ++i) {
      const int joint =
          Id(mjOBJ_JOINT, walker.prefix + std::string(kCmuMocapJoints[i]));
      data_->qpos[model_->jnt_qposadr[joint]] = clip.Frame("joints", frame)[i];
      data_->qvel[model_->jnt_dofadr[joint]] =
          frame == clip.frames - 1 ? 0
                                   : clip.Frame("joints_velocity", frame)[i];
    }
    if (frame != clip.frames - 1) {
      double* velocity = data_->qvel + model_->jnt_dofadr[walker.freejoint];
      mju_copy3(velocity, clip.Frame("velocity", frame));
      mju_copy3(velocity + 3, clip.Frame("angular_velocity", frame));
    }
  }
}

void Simulation::ShiftWalker(int index, const std::array<double, 3>& position,
                             double rotation) {
  const auto& walker = walkers_[index];
  if (walker.freejoint < 0) return;
  double* qpos = data_->qpos + model_->jnt_qposadr[walker.freejoint];
  if (rotation != 0) {
    const double turn[4]{std::cos(rotation / 2), 0, 0, std::sin(rotation / 2)};
    double quat[4];
    mju_mulQuat(quat, turn, qpos + 3);
    // Entity.set_pose normalizes with elementwise division, also when it is
    // called by shift_pose. MuJoCo's normalize4 uses reciprocal multiplication.
    const double norm = std::sqrt(mju_dot(quat, quat, 4));
    for (double& value : quat) value /= norm;
    mju_copy4(qpos + 3, quat);
  }
  for (int i = 0; i < 3; ++i) qpos[i] += position[i];
}

void Simulation::Reset() {
  steps_ = 0;
  failure_ = done_ = truncated_ = success_ = false;
  discount = 1;
  std::fill(rewards.begin(), rewards.end(), 0);
  std::fill(previous_actions_.begin(), previous_actions_.end(), 0);
  const bool corridor = task_.task == Task::kWalls || task_.task == Task::kGaps;
  const bool maze =
      task_.task == Task::kForage || task_.task == Task::kHeterogeneous;
  if (task_.task == Task::kTracking) SelectTrackingClip();
  scene_.LoadArena(
      corridor
          ? (task_.task == Task::kWalls ? "walls_corridor" : "gaps_corridor")
      : maze ? (task_.task == Task::kForage ? "random_maze" : "maze")
      : task_.task == Task::kBowl   ? "bowl"
      : task_.task == Task::kSoccer ? "randomized_pitch"
                                    : "floor",
      task_.physics_timestep);
  if (corridor) {
    scene_.Corridor(task_, &random_);
  } else if (task_.task == Task::kTarget || task_.task == Task::kTracking) {
    scene_.Floor(8);
  } else if (maze) {
    scene_.MazeArena(task_, &scene_random_);
  } else if (task_.task == Task::kTwoTouch) {
    scene_.Floor(10, true);
  } else if (task_.task == Task::kBowl) {
    scene_.Bowl();
  } else if (task_.task == Task::kSoccer) {
    scene_.Soccer(task_, options_.team_size, options_.enable_field_box,
                  options_.keep_aspect_ratio, options_.disable_walker_contacts,
                  &random_);
  } else {
    throw std::runtime_error("Unsupported locomotion scene: " + options_.task);
  }
  if (task_.task != Task::kSoccer) scene_.AddWalker(task_.walker, "walker/");
  if (maze) scene_.AddTargets(task_, &random_);
  if (task_.task == Task::kTwoTouch) scene_.TwoTouchTarget();
  if (task_.task == Task::kTarget) {
    auto target = scene_.world().append_child("site");
    Set(target, "name", "target");
    Set(target, "type", "sphere");
    const double x = random_.Uniform(-4, 4);
    const double y = random_.Uniform(-4, 4);
    Set(target, "pos", {x, y, 0});
    Set(target, "size", {.1});
    Set(target, "rgba", {.9, .6, .6, 1});
  }
  Compile();
  mj_resetData(model_.get(), data_.get());
  const int flags = model_->opt.disableflags;
  model_->opt.disableflags |= mjDSBL_ACTUATION;
  mj_forward(model_.get(), data_.get());
  model_->opt.disableflags = flags;
  // TargetSphere.initialize_episode reads Composer's cached contact list.
  // Walker pose writes are still dirty there: the contacts belong to the
  // XML reset pose, not the subsequently selected maze spawn.
  if (maze) AfterSubstep();
  for (int player = 0; player < players_; ++player) ResetWalker(player);
  if (corridor) {
    ShiftWalker(0, {task_.walker == Walker::kRodent ? 5 : .5, 0, 0});
  } else if (maze) {
    RespawnMaze();
  } else if (task_.task == Task::kTwoTouch) {
    ShiftWalker(0, {0, 0, 0}, 2 * std::acos(-1) * scene_random_.Uniform());
    RandomizeTouchTarget();
    observed_touch_state_ = 0;
  } else if (task_.task == Task::kBowl) {
    ResetBowl();
  } else if (task_.task == Task::kSoccer) {
    goals_ = goals_now_ = {false, false};
    off_court_ = false;
    scoring_team_ = -1;
    InitializeSoccer();
  } else if (task_.task == Task::kTracking) {
    ResetTracking();
  } else {
    const double x = random_.Uniform(-4, 4);
    const double y = random_.Uniform(-4, 4);
    ShiftWalker(0, {x, y, 0});
  }
  model_->opt.disableflags |= mjDSBL_ACTUATION;
  mj_forward(model_.get(), data_.get());
  model_->opt.disableflags = flags;
  model_dirty_ = false;
  Observe();
}

void Simulation::RespawnMaze() {
  const auto& position =
      scene_.spawn_positions[random_.Int(scene_.spawn_positions.size())];
  double rotation = 0;
  if (task_.task != Task::kHeterogeneous) {
    ShiftWalker(0, {0, 0, 100});
    mj_forward(model_.get(), data_.get());
    double maximum = -2;
    int direction = 0;
    for (int i = 0; i < 10; ++i) {
      const double theta = 2 * std::acos(-1) * i / 10;
      const double origin[3]{position[0], position[1], .1};
      const double ray[3]{std::cos(theta), std::sin(theta), 0};
      int geom = -1;
      const double distance = mj_ray(model_.get(), data_.get(), origin, ray,
                                     nullptr, 1, -1, &geom, nullptr);
      if (distance > maximum) {
        maximum = distance;
        direction = i;
      }
    }
    rotation =
        2 * std::acos(-1) * direction / 10 +
        std::acos(-1) * (1 + std::tanh(std::atanh(random_.Uniform(-1, 1))));
    ShiftWalker(0, {0, 0, -100});
  }
  ShiftWalker(0, position, rotation);
}

void Simulation::AfterSubstep() {
  if (task_.task == Task::kSoccer) {
    SoccerDetections();
    return;
  }
  if (task_.task == Task::kTwoTouch) {
    if (touched_twice_) return;
    for (int j = 0; j < data_->ncon; ++j) {
      const auto& contact = data_->contact[j];
      int other = -1;
      if (contact.geom1 == target_geoms_[0]) other = contact.geom2;
      if (contact.geom2 == target_geoms_[0]) other = contact.geom1;
      if (std::find(hand_geoms_.begin(), hand_geoms_.end(), other) ==
          hand_geoms_.end())
        continue;
      const bool previous_once = touched_once_;
      const bool previous_twice = touched_twice_;
      if (!touched_once_) {
        touched_once_ = true;
        touch_time_ = data_->time;
      }
      if (data_->time > touch_time_ + .2) touched_twice_ = true;
      // TargetSphereTwoTouch writes texid only on a state transition. That
      // PyMJCF write marks dynamics dirty; repeated contact does not.
      if (previous_once == touched_once_ && previous_twice == touched_twice_)
        continue;
      const int texture =
          Id(mjOBJ_TEXTURE, touched_twice_ ? "target_0_0/target_sphere_final"
                                           : "target_0_0/target_sphere_inter");
      std::fill_n(model_->mat_texid + mjNTEXROLE * target_materials_[0],
                  mjNTEXROLE, texture);
      model_dirty_ = true;
    }
    return;
  }
  for (std::size_t i = 0; i < target_geoms_.size(); ++i) {
    if (target_activated_[i]) continue;
    for (int j = 0; j < data_->ncon; ++j) {
      const auto& contact = data_->contact[j];
      if (contact.geom1 == target_geoms_[i] ||
          contact.geom2 == target_geoms_[i]) {
        target_activated_[i] = true;
        model_->mat_rgba[4 * target_materials_[i] + 3] = 0;
        break;
      }
    }
  }
}

bool Simulation::DisallowedContact() const {
  for (int contact_id = 0; contact_id < data_->ncon; ++contact_id) {
    const auto& contact = data_->contact[contact_id];
    for (int orientation = 0; orientation < 2; ++orientation) {
      const int ground = orientation ? contact.geom1 : contact.geom2;
      const int geom = orientation ? contact.geom2 : contact.geom1;
      if (std::find(ground_geoms_.begin(), ground_geoms_.end(), ground) ==
          ground_geoms_.end())
        continue;
      const auto& walker = walkers_[0];
      const char* name = mj_id2name(model_.get(), mjOBJ_GEOM, geom);
      if (name && std::string(name).find(walker.prefix) == 0 &&
          std::find(walker.ground_contact_geoms.begin(),
                    walker.ground_contact_geoms.end(),
                    geom) == walker.ground_contact_geoms.end())
        return true;
    }
  }
  return false;
}

void Simulation::AfterStep() {
  const bool corridor = task_.task == Task::kWalls || task_.task == Task::kGaps;
  failure_ = false;
  if (task_.walker != Walker::kRodent && task_.task != Task::kSoccer &&
      task_.task != Task::kTracking)
    failure_ = DisallowedContact();
  if (task_.task == Task::kSoccer) {
    AfterSoccerStep();
    return;
  }
  if (task_.task == Task::kTracking) {
    AfterTrackingStep();
    return;
  }
  if (corridor) {
    const double height = task_.walker == Walker::kRodent ? -.3 : -.5;
    for (int body : walkers_[0].effectors) {
      if (data_->xpos[3 * body + 2] < height) failure_ = true;
    }
    const double velocity = data_->subtree_linvel[3 * walkers_[0].root];
    const double target = task_.walker == Walker::kRodent ? 1 : 3;
    const double distance = std::abs(velocity - target) / target;
    rewards[0] = distance < 1 ? 1 - distance : 0;
  } else if (task_.task == Task::kTarget) {
    const int target = Id(mjOBJ_SITE, "target");
    const auto* position = data_->xpos + 3 * walkers_[0].root;
    const double dx = model_->site_pos[3 * target] - position[0];
    const double dy = model_->site_pos[3 * target + 1] - position[1];
    rewards[0] = std::sqrt(dx * dx + dy * dy) < 1;
  } else if (task_.task == Task::kForage ||
             task_.task == Task::kHeterogeneous) {
    rewards[0] = task_.task == Task::kHeterogeneous ? .01 : 0;
    success_ = true;
    for (std::size_t i = 0; i < target_activated_.size(); ++i) {
      success_ = success_ && target_activated_[i];
      if (target_activated_[i] && !target_rewarded_[i]) {
        rewards[0] += task_.task == Task::kForage   ? 50
                      : scene_.target_types[i] == 0 ? 30
                                                    : -10;
        target_rewarded_[i] = true;
      }
    }
    // NullGoalMaze updates its internal discount in should_terminate_episode,
    // but Composer has already read get_discount for this final timestep.
    discount = 1;
    return;
  } else if (task_.task == Task::kTwoTouch) {
    TouchReward();
  } else if (task_.task == Task::kBowl) {
    const auto& walker = walkers_[0];
    const double* position =
        data_->site_xpos + 3 * Id(mjOBJ_SITE, "walker/head");
    const double distance = mju_norm3(position);
    const double escape = distance >= 6 ? 1 : 1 - (6 - distance) / 6;
    const double deviation = std::cos(std::acos(-1) / 6);
    const double vertical = std::min(data_->xmat[9 * walker.root + 8],
                                     data_->xmat[9 * walker.pelvis + 8]);
    const double upright =
        vertical >= deviation
            ? 1
            : std::max(0., 1 - (deviation - vertical) / (1 + deviation));
    rewards[0] = upright * escape;
  }
  discount = failure_ ? 0 : 1;
}

void Simulation::Step(const double* actions) {
  const int action_size = ActionSize(task_.walker);
  for (int player = 0; player < players_; ++player) {
    for (int i = 0; i < action_size; ++i) {
      data_->ctrl[walkers_[player].actuators[i]] =
          actions[player * action_size + i];
    }
  }
  std::copy(actions, actions + players_ * action_size,
            previous_actions_.begin());
  if (task_.task == Task::kTwoTouch && randomize_touch_) RandomizeTouchTarget();
  if (task_.task == Task::kSoccer) BeforeSoccerStep();
  const int substeps = static_cast<int>(
      std::round(task_.control_timestep / task_.physics_timestep));
  for (int step = 0; step < substeps; ++step) {
    if (model_->opt.integrator == mjINT_RK4)
      mj_step(model_.get(), data_.get());
    else
      mj_step2(model_.get(), data_.get());
    mj_step1(model_.get(), data_.get());
    mj_subtreeVel(model_.get(), data_.get());
    // Soccer's position detectors access a bound geom after the first
    // substep, so PyMJCF refreshes dynamics earlier than in corridor tasks.
    if (task_.task == Task::kSoccer && step == 0) {
      mj_forward(model_.get(), data_.get());
      model_dirty_ = false;
    }
    AfterSubstep();
  }
  ++steps_;
  // Walker.apply_action writes through PyMJCF bind(), unlike Task.before_step
  // which calls set_control directly. Only the former marks physics dirty and
  // triggers a forward pass on the next derived observation read.
  if (task_.task == Task::kWalls || task_.task == Task::kGaps ||
      task_.task == Task::kTarget || task_.task == Task::kTracking ||
      model_dirty_)
    mj_forward(model_.get(), data_.get());
  model_dirty_ = false;
  AfterStep();
  truncated_ = data_->time >= options_.time_limit ||
               steps_ >= options_.max_episode_steps ||
               (task_.task == Task::kTracking && success_);
  done_ = failure_ || success_ || truncated_;
  Observe();
}

#ifdef ENVPOOL_TEST
void Simulation::SetResetState(
    const std::vector<double>& qpos, const std::vector<double>& qvel,
    const std::map<std::string, std::array<double, 3>>& geoms) {
  if (steps_ != 0 || !model_)
    throw std::logic_error("oracle fixtures may only change reset state");
  if (qpos.size() != static_cast<std::size_t>(model_->nq) ||
      qvel.size() != static_cast<std::size_t>(model_->nv)) {
    throw std::invalid_argument("wrong reset state shape");
  }
  std::copy(qpos.begin(), qpos.end(), data_->qpos);
  std::copy(qvel.begin(), qvel.end(), data_->qvel);
  for (const auto& [name, position] : geoms) {
    std::copy(position.begin(), position.end(),
              model_->geom_pos + 3 * Id(mjOBJ_GEOM, name));
  }
  const int flags = model_->opt.disableflags;
  model_->opt.disableflags |= mjDSBL_ACTUATION;
  mj_forward(model_.get(), data_.get());
  model_->opt.disableflags = flags;
  mju_zero(data_->qacc_warmstart, model_->nv);
  if (task_.task == Task::kTracking) {
    tracking_features_ = TrackingFeatures();
    previous_features_ = tracking_features_;
    UpdateTrackingObservations();
  }
  Observe();
}
#endif

void Simulation::Render(int width, int height, int camera,
                        unsigned char* output, const mjvOption* option) {
  mjvOption settings;
  if (option)
    settings = *option;
  else
    mjv_defaultOption(&settings);
  // dm_control.wrapper.MjvOption disables rangefinder visualization, unlike
  // MuJoCo's raw default (visible on Soccer Ant's eight rangefinders).
  settings.flags[mjVIS_RANGEFINDER] = 0;
  if (!renderer_)
    renderer_ = std::make_unique<envpool::mujoco::OffscreenRenderer>(
        envpool::mujoco::CameraPolicy::kDmControl, false, false, true, false,
        false);
  renderer_->Render(model_.get(), data_.get(), width, height, camera, output,
                    nullptr, &settings);
}

}  // namespace mujoco_locomotion
