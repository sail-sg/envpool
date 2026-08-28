/*
 * Copyright 2026 Garena Online Private Limited
 * Copyright 2020-2021 The dm_control Authors.
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
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "envpool/mujoco/locomotion/simulation.h"
#include "third_party/dmc_locomotion/metadata.h"

namespace mujoco_locomotion {
namespace {

void NormalizeFeature(double* quaternion) {
  // np.linalg.norm(..., axis=-1) is a sum of separately squared values, then
  // elementwise division. Reciprocal multiplication changes CoMic's acos
  // argument near one and amplifies otherwise harmless quaternion roundoff.
  double squared = 0;
  for (int i = 0; i < 4; ++i) squared += quaternion[i] * quaternion[i];
  const double norm = std::sqrt(squared);
  for (int i = 0; i < 4; ++i) quaternion[i] /= norm;
}

std::array<double, 4> QuaternionDifference(const double* source,
                                           const double* target) {
  // Match transformations.quat_diff's Hamilton product, without normalizing
  // its operands: the reference quaternions were recorded as float32.
  const double w = source[0], x = -source[1], y = -source[2], z = -source[3];
  return {w * target[0] - x * target[1] - y * target[2] - z * target[3],
          x * target[0] + w * target[1] - z * target[2] + y * target[3],
          y * target[0] + z * target[1] + w * target[2] - x * target[3],
          z * target[0] - y * target[1] + x * target[2] + w * target[3]};
}

std::array<double, 9> QuaternionMatrix(const double* quaternion) {
  double q[4];
  const double norm = mju_dot(quaternion, quaternion, 4);
  mju_scl(q, quaternion, std::sqrt(2 / norm), 4);
  return {1 - q[2] * q[2] - q[3] * q[3], q[1] * q[2] - q[3] * q[0],
          q[1] * q[3] + q[2] * q[0],     q[1] * q[2] + q[3] * q[0],
          1 - q[1] * q[1] - q[3] * q[3], q[2] * q[3] - q[1] * q[0],
          q[1] * q[3] - q[2] * q[0],     q[2] * q[3] + q[1] * q[0],
          1 - q[1] * q[1] - q[2] * q[2]};
}

double SquaredDifference(const std::vector<double>& actual,
                         const double* expected) {
  double sum = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double delta = actual[i] - expected[i];
    sum += delta * delta;
  }
  return sum;
}

}  // namespace

void Simulation::SelectTrackingClip() {
  if (possible_starts_.empty()) {
    for (int clip = 0; clip < static_cast<int>(mocap_->clips.size()); ++clip) {
      for (int step = 0; step < mocap_->clips[clip].frames - 5 - 10; ++step) {
        possible_starts_.emplace_back(clip, step);
      }
    }
    if (possible_starts_.empty())
      throw std::runtime_error("No valid mocap starts");
    const double probability = 1. / possible_starts_.size();
    double sum = 0;
    for (std::size_t i = 0; i < possible_starts_.size(); ++i) {
      sum += probability;
      start_cdf_.push_back(sum);
    }
    for (double& value : start_cdf_) value /= sum;
  }
  const auto index = std::upper_bound(start_cdf_.begin(), start_cdf_.end(),
                                      random_.Uniform()) -
                     start_cdf_.begin();
  const auto [clip, step] = possible_starts_.at(index);
  clip_id_ = clip;
  reference_start_ = reference_step_ = step;
}

Simulation::Features Simulation::TrackingFeatures() const {
  const auto& walker = walkers_[0];
  const double* root = data_->qpos + model_->jnt_qposadr[walker.freejoint];
  const double* velocity = data_->qvel + model_->jnt_dofadr[walker.freejoint];
  std::map<std::string, std::vector<double>> features;
  features["position"] = {root, root + 3};
  features["quaternion"] = {root + 3, root + 7};
  features["velocity"] = {velocity, velocity + 3};
  features["angular_velocity"] = {velocity + 3, velocity + 6};
  features["center_of_mass"] = {data_->subtree_com + 3 * walker.frame,
                                data_->subtree_com + 3 * walker.frame + 3};
  for (auto name : kCmuMocapJoints) {
    const int joint = Id(mjOBJ_JOINT, walker.prefix + std::string(name));
    features["joints"].push_back(data_->qpos[model_->jnt_qposadr[joint]]);
    features["joints_velocity"].push_back(
        data_->qvel[model_->jnt_dofadr[joint]]);
  }
  for (int body : walker.bodies) {
    if (body == walker.root) continue;
    const double* pos = data_->xpos + 3 * body;
    const double* quat = data_->xquat + 4 * body;
    features["body_positions"].insert(features["body_positions"].end(), pos,
                                      pos + 3);
    features["body_quaternions"].insert(features["body_quaternions"].end(),
                                        quat, quat + 4);
  }
  features["end_effectors"] = WalkerObservation(walker, "end_effectors_pos");
  features["appendages"] = WalkerObservation(walker, "appendages_pos");
  return features;
}

void Simulation::ResetTracking() {
  const auto& clip = mocap_->clips[clip_id_];
  const auto& walker = walkers_[0];
  double* root = data_->qpos + model_->jnt_qposadr[walker.freejoint];
  double* velocity = data_->qvel + model_->jnt_dofadr[walker.freejoint];
  mju_copy3(root, clip.Frame("position", reference_step_));
  mju_copy4(root + 3, clip.Frame("quaternion", reference_step_));
  mju_copy3(velocity, clip.Frame("velocity", reference_step_));
  mju_copy3(velocity + 3, clip.Frame("angular_velocity", reference_step_));
  for (std::size_t i = 0; i < kCmuMocapJoints.size(); ++i) {
    const int joint =
        Id(mjOBJ_JOINT, walker.prefix + std::string(kCmuMocapJoints[i]));
    data_->qpos[model_->jnt_qposadr[joint]] =
        clip.Frame("joints", reference_step_)[i];
    data_->qvel[model_->jnt_dofadr[joint]] =
        clip.Frame("joints_velocity", reference_step_)[i];
  }
  const int flags = model_->opt.disableflags;
  model_->opt.disableflags |= mjDSBL_ACTUATION;
  mj_forward(model_.get(), data_.get());
  model_->opt.disableflags = flags;
  tracking_features_ = TrackingFeatures();
  previous_features_ = tracking_features_;
  if (TrackingError() > 1e-2)
    throw std::runtime_error("CMU reference and walker disagree at reset");
  UpdateTrackingObservations();
}

double Simulation::TrackingError() const {
  const auto& clip = mocap_->clips[clip_id_];
  double result = 0;
  for (const char* key : {"body_positions", "joints"}) {
    const auto& values = tracking_features_.at(key);
    const auto* reference = clip.Frame(key, reference_step_);
    if (values.size() != static_cast<std::size_t>(clip.features.at(key).width))
      throw std::runtime_error("CMU reference shape mismatch");
    double error = 0;
    for (std::size_t i = 0; i < values.size(); ++i)
      error += std::abs(reference[i] - values[i]);
    result += .5 * (error / values.size());
  }
  return result;
}

void Simulation::UpdateTrackingObservations() {
  const auto& clip = mocap_->clips[clip_id_];
  auto& obs = tracking_observations_;
  obs.clear();
  obs["reference_props_pos_global"] = {};
  obs["reference_props_quat_global"] = {};
  const auto& walker = walkers_[0];
  const double* matrix = data_->xmat + 9 * walker.root;
  std::vector<int> joint_order;
  for (int joint : walker.joints) {
    const std::string name = mj_id2name(model_.get(), mjOBJ_JOINT, joint);
    const auto index = std::find(kCmuMocapJoints.begin(), kCmuMocapJoints.end(),
                                 name.substr(walker.prefix.size())) -
                       kCmuMocapJoints.begin();
    joint_order.push_back(index);
  }
  auto append_quaternion = [&](const std::string& key, const double* source,
                               const double* target) {
    const auto delta = QuaternionDifference(source, target);
    obs[key].insert(obs[key].end(), delta.begin(), delta.end());
  };
  for (int lookahead = 1; lookahead <= 5; ++lookahead) {
    const int frame = reference_step_ + lookahead;
    for (int index : joint_order)
      obs["reference_rel_joints"].push_back(
          clip.Frame("joints", frame)[index] -
          tracking_features_.at("joints")[index]);
    const auto& positions = tracking_features_.at("body_positions");
    for (std::size_t i = 0; i < positions.size(); i += 3) {
      double delta[3], local[3];
      mju_sub3(delta, clip.Frame("body_positions", frame) + i,
               positions.data() + i);
      mju_mulMatTVec3(local, matrix, delta);
      obs["reference_rel_bodies_pos_global"].insert(
          obs["reference_rel_bodies_pos_global"].end(), delta, delta + 3);
      obs["reference_rel_bodies_pos_local"].insert(
          obs["reference_rel_bodies_pos_local"].end(), local, local + 3);
    }
    const auto& quaternions = tracking_features_.at("body_quaternions");
    for (std::size_t i = 0; i < quaternions.size(); i += 4) {
      append_quaternion("reference_rel_bodies_quats", quaternions.data() + i,
                        clip.Frame("body_quaternions", frame) + i);
      append_quaternion("reference_ego_bodies_quats",
                        clip.Frame("quaternion", frame),
                        clip.Frame("body_quaternions", frame) + i);
    }
    append_quaternion("reference_rel_root_quat",
                      tracking_features_.at("quaternion").data(),
                      clip.Frame("quaternion", frame));
    const auto* appendages = clip.Frame("appendages", frame);
    obs["reference_appendages_pos"].insert(
        obs["reference_appendages_pos"].end(), appendages, appendages + 15);
    double delta[3], local[3];
    mju_sub3(delta, clip.Frame("position", frame),
             tracking_features_.at("position").data());
    mju_mulMatTVec3(local, matrix, delta);
    obs["reference_rel_root_pos_local"].insert(
        obs["reference_rel_root_pos_local"].end(), local, local + 3);
  }
  const auto prev_matrix =
      QuaternionMatrix(previous_features_.at("quaternion").data());
  double velocity[3], local[3];
  mju_sub3(velocity, tracking_features_.at("position").data(),
           previous_features_.at("position").data());
  for (double& value : velocity) value /= task_.control_timestep;
  mju_mulMatTVec3(local, prev_matrix.data(), velocity);
  obs["velocimeter_control"] = {local, local + 3};
  auto quat = QuaternionDifference(previous_features_.at("quaternion").data(),
                                   tracking_features_.at("quaternion").data());
  mju_normalize4(quat.data());
  const double angle = 2 * std::acos(std::clamp(quat[0], -1., 1.));
  const double pi = std::acos(-1);
  obs["gyro_control"] = std::vector<double>(3, 0);
  if (angle >= 1e-10) {
    const double sine = std::sin(angle / 2);
    double wrapped = std::fmod(angle + pi, 2 * pi) - pi;
    for (int i = 0; i < 3; ++i)
      obs["gyro_control"][i] =
          (quat[i + 1] / sine) * wrapped / task_.control_timestep;
  }
  for (int index : joint_order)
    obs["joints_vel_control"].push_back(
        (tracking_features_.at("joints")[index] -
         previous_features_.at("joints")[index]) /
        task_.control_timestep);
  obs["time_in_clip"] = {(reference_start_ * clip.dt + data_->time) /
                         ((clip.frames - 1) * clip.dt)};
}

void Simulation::AfterTrackingStep() {
  previous_features_ = tracking_features_;
  ++reference_step_;
  tracking_features_ = TrackingFeatures();
  const double error = TrackingError();
  UpdateTrackingObservations();
  const auto& clip = mocap_->clips[clip_id_];
  const double com =
      .1 * std::exp(-10 * SquaredDifference(
                              tracking_features_.at("center_of_mass"),
                              clip.Frame("center_of_mass", reference_step_)));
  const double velocity = std::exp(
      -.1 * SquaredDifference(tracking_features_.at("joints_velocity"),
                              clip.Frame("joints_velocity", reference_step_)));
  const double appendages =
      .15 * std::exp(-40 * SquaredDifference(
                               tracking_features_.at("appendages"),
                               clip.Frame("appendages", reference_step_)));
  auto& quats = tracking_features_.at("body_quaternions");
  const double* target = clip.Frame("body_quaternions", reference_step_);
  double difference = 0;
  for (std::size_t i = 0; i < quats.size(); i += 4) {
    // The official reward normalizes its feature snapshots in place; keep
    // that mutation in the cached features used on the following timestep.
    NormalizeFeature(quats.data() + i);
    double reference[4];
    mju_copy4(reference, target + i);
    NormalizeFeature(reference);
    const double* source = quats.data() + i;
    // einsum's four-element sum-of-products kernel uses two pairwise sums.
    const double dot = (source[0] * reference[0] + source[1] * reference[1]) +
                       (source[2] * reference[2] + source[3] * reference[3]);
    const double distance = .5 * std::acos(std::min(1., 2 * dot * dot - 1));
    difference += distance * distance;
  }
  NormalizeFeature(tracking_features_.at("quaternion").data());
  const double bodies = .65 * std::exp(-2 * difference);
  rewards[0] =
      .5 * (1 - error / .3) + .5 * ((com + velocity) + appendages + bodies);
  failure_ = error > .3;
  success_ = reference_step_ == clip.frames - 6;
  discount = failure_ ? 0 : 1;
}

}  // namespace mujoco_locomotion
