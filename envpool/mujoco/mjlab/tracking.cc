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

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "envpool/mujoco/mjlab/motion.h"
#include "envpool/mujoco/mjlab/simulation.h"

namespace mjlab {
namespace {

float RotationError(const Quat& a, const Quat& b) {
  auto q = Multiply(a, Conjugate(b));
  if (q[0] < 0) {
    q = q * -1.0F;
  }
  Vec3 axis{q[1], q[2], q[3]};
  const float half_angle = Atan2(Norm(axis), q[0]);
  const float angle = 2.0F * half_angle;
  const float divisor = std::abs(angle) > 1.0e-6F
                            ? Sin(half_angle) / angle
                            : 0.5F - (angle * angle) / 48.0F;
  for (auto& value : axis) {
    value /= divisor;
  }
  return Norm(axis);
}

class Tracking : public Task {
 public:
  Tracking(Simulation* simulation, const std::string& motion_file)
      : Task(simulation),
        cfg_(sim_.cfg.at("command").at("motion")),
        robot_(sim_.entities.at(Name(cfg_, "entity_name"))) {
    const auto& body_names =
        sim_.cfg.at("entities").at(robot_.name).at("body_names").as_array();
    for (const auto& name : cfg_.at("body_names").as_array()) {
      const auto* found = std::find(body_names.begin(), body_names.end(), name);
      if (found == body_names.end()) {
        throw std::invalid_argument("missing MJLab tracked body");
      }
      const int index = found - body_names.begin();
      if (String(name) == Name(cfg_, "anchor_body_name")) {
        anchor_ = body_indices_.size();
      }
      body_indices_.push_back(index);
    }
    motion_ = LoadMotion(motion_file, robot_.qadr.size(), robot_.bodies.size());
    const int bins =
        static_cast<int>(std::floor(motion_->frames /
                                    (1.0 / Number(sim_.cfg.at("step_dt"))))) +
        1;
    failed_.resize(bins);
    // TODO(jiayi): expose shared training statistics if EnvPool gains
    // synchronous pool-wide managers. These adaptive counts intentionally match
    // independent one-world upstream instances, not its batch-wide histogram.
    current_failed_.resize(bins);
    probabilities_.resize(bins);
    relative_positions_.resize(body_indices_.size() * 3);
    relative_orientations_.resize(body_indices_.size() * 4);
    command.resize(robot_.qadr.size() * 2);
  }

  void Reset() override {
    counter = 0;
    time_left = sim_.Uniform(cfg_.at("resampling_time_range"));
    ++counter;
    Resample();
    pending_forward_ = false;
  }

  void Update(bool resetting) override {
    if (!resetting) {
      time_left -= sim_.step_dt;
    }
    if (time_left <= 0) {
      time_left = sim_.Uniform(cfg_.at("resampling_time_range"));
      ++counter;
      Resample();
    }
    // The official reset path advances the reference by one frame too.
    ++frame_;
    if (frame_ >= motion_->frames) {
      Resample();
    }
    if (pending_forward_) {
      sim_.physics.Run("forward");
      pending_forward_ = false;
    }
    const auto anchor_position = ReferencePosition(anchor_);
    const auto anchor_rotation = ReferenceOrientation(anchor_);
    const auto robot_anchor = robot_.bodies[body_indices_[anchor_]];
    Vec3 delta_position = sim_.Position(robot_anchor);
    delta_position[2] = anchor_position[2];
    const auto delta =
        Multiply(sim_.Orientation(robot_anchor), Inverse(anchor_rotation));
    const auto delta_rotation = YawQuat(delta);
    for (std::size_t i = 0; i < body_indices_.size(); ++i) {
      const auto position =
          delta_position +
          Rotate(delta_rotation, ReferencePosition(i) - anchor_position);
      const auto rotation = Multiply(delta_rotation, ReferenceOrientation(i));
      std::copy(position.begin(), position.end(),
                relative_positions_.begin() + i * 3);
      std::copy(rotation.begin(), rotation.end(),
                relative_orientations_.begin() + i * 4);
    }
    const auto* position = motion_->arrays.at("joint_pos").Frame(frame_);
    const auto* velocity = motion_->arrays.at("joint_vel").Frame(frame_);
    std::copy_n(position, robot_.qadr.size(), command.begin());
    std::copy_n(velocity, robot_.qadr.size(),
                command.begin() + robot_.qadr.size());
    if (!resetting) {
      const float alpha = Param(cfg_, "adaptive_alpha");
      const float retain = 1.0 - Number(cfg_.at("adaptive_alpha"));
      for (std::size_t i = 0; i < failed_.size(); ++i) {
        failed_[i] = alpha * current_failed_[i] + retain * failed_[i];
        current_failed_[i] = 0;
      }
    }
  }

  std::vector<float> Observation(const std::string& fn,
                                 const Json& p) override {
    const int anchor_body = robot_.bodies[body_indices_[anchor_]];
    const auto position = sim_.Position(anchor_body);
    const auto rotation = Inverse(sim_.Orientation(anchor_body));
    std::vector<float> result;
    const bool anchor =
        fn == "motion_anchor_pos_b" || fn == "motion_anchor_ori_b";
    const bool body = fn == "robot_body_pos_b" || fn == "robot_body_ori_b";
    if (!anchor && !body) {
      return Task::Observation(fn, p);
    }
    const int count = anchor ? 1 : body_indices_.size();
    for (int i = 0; i < count; ++i) {
      if (fn == "motion_anchor_pos_b" || fn == "robot_body_pos_b") {
        const auto target =
            anchor ? ReferencePosition(anchor_)
                   : sim_.Position(robot_.bodies[body_indices_[i]]);
        const auto local = Rotate(rotation, target - position);
        result.insert(result.end(), local.begin(), local.end());
      } else {
        const auto target =
            anchor ? ReferenceOrientation(anchor_)
                   : sim_.Orientation(robot_.bodies[body_indices_[i]]);
        const auto matrix = Matrix(Multiply(rotation, target));
        for (int row = 0; row < 3; ++row) {
          for (int column = 0; column < 2; ++column) {
            result.push_back(matrix[row * 3 + column]);
          }
        }
      }
    }
    return result;
  }

  float Reward(const std::string& fn, const Json& p,
               const Json& term) override {
    const int anchor_body = robot_.bodies[body_indices_[anchor_]];
    float error = 0;
    if (fn == "motion_global_anchor_position_error_exp") {
      error =
          SquaredNorm(ReferencePosition(anchor_) - sim_.Position(anchor_body));
    } else if (fn == "motion_global_anchor_orientation_error_exp") {
      error = Square(RotationError(ReferenceOrientation(anchor_),
                                   sim_.Orientation(anchor_body)));
    } else {
      const auto indices = SelectedBodies(p);
      std::vector<float> errors;
      for (int i : indices) {
        const int body = robot_.bodies[body_indices_[i]];
        if (fn == "motion_relative_body_position_error_exp") {
          error = SquaredNorm(Read<3>(relative_positions_.data() + i * 3) -
                              sim_.Position(body));
        } else if (fn == "motion_relative_body_orientation_error_exp") {
          error = Square(
              RotationError(Read<4>(relative_orientations_.data() + i * 4),
                            sim_.Orientation(body)));
        } else if (fn == "motion_global_body_linear_velocity_error_exp") {
          error = SquaredNorm(
              Read<3>(motion_->arrays.at("body_lin_vel_w").Frame(frame_) +
                      body_indices_[i] * 3) -
              sim_.LinearVelocity(body, robot_.root, sim_.Position(body)));
        } else if (fn == "motion_global_body_angular_velocity_error_exp") {
          error = SquaredNorm(
              Read<3>(motion_->arrays.at("body_ang_vel_w").Frame(frame_) +
                      body_indices_[i] * 3) -
              sim_.AngularVelocity(body));
        } else {
          return Task::Reward(fn, p, term);
        }
        errors.push_back(error);
      }
      error = Sum(errors) / static_cast<float>(indices.size());
    }
    return Exp(-error / SquaredParam(p, "std"));
  }

  bool Terminated(const std::string& fn, const Json& p) override {
    const int anchor_body = robot_.bodies[body_indices_[anchor_]];
    if (fn == "bad_anchor_pos_z_only") {
      return std::abs(ReferencePosition(anchor_)[2] -
                      sim_.Position(anchor_body)[2]) > Param(p, "threshold");
    }
    if (fn == "bad_anchor_ori") {
      const auto reference =
          RotateInverse(ReferenceOrientation(anchor_), {0, 0, -1});
      const auto robot =
          RotateInverse(sim_.Orientation(anchor_body), {0, 0, -1});
      return std::abs(reference[2] - robot[2]) > Param(p, "threshold");
    }
    if (fn == "bad_motion_body_pos_z_only") {
      const auto bodies = SelectedBodies(p);
      return std::any_of(bodies.begin(), bodies.end(), [&](int i) {
        return std::abs(relative_positions_[i * 3 + 2] -
                        sim_.Position(robot_.bodies[body_indices_[i]])[2]) >
               Param(p, "threshold");
      });
    }
    return Task::Terminated(fn, p);
  }

  Json State() const override {
    auto result = Task::State();
    auto& obj = result.as_object();
    obj["time_steps"] = frame_;
    obj["body_pos_relative_w"] = ToJson(relative_positions_);
    obj["body_quat_relative_w"] = ToJson(relative_orientations_);
    obj["bin_failed_count"] = ToJson(failed_);
    obj["current_bin_failed"] = ToJson(current_failed_);
    obj["sampling_probabilities"] = ToJson(probabilities_);
    return result;
  }

 private:
  std::vector<int> SelectedBodies(const Json& p) const {
    const auto* names = p.as_object().if_contains("body_names");
    std::vector<int> indices;
    const auto& all = cfg_.at("body_names").as_array();
    for (std::size_t i = 0; i < all.size(); ++i) {
      if (names == nullptr || names->is_null() ||
          std::find(names->as_array().begin(), names->as_array().end(),
                    all[i]) != names->as_array().end()) {
        indices.push_back(i);
      }
    }
    return indices;
  }
  Vec3 ReferencePosition(int i) const {
    return Read<3>(motion_->arrays.at("body_pos_w").Frame(frame_) +
                   body_indices_[i] * 3) +
           sim_.origin;
  }
  Quat ReferenceOrientation(int i) const {
    return Read<4>(motion_->arrays.at("body_quat_w").Frame(frame_) +
                   body_indices_[i] * 4);
  }

  void Resample() {
    const int bins = failed_.size();
    if (sim_.terminated) {
      std::fill(current_failed_.begin(), current_failed_.end(), 0);
      current_failed_[std::min(frame_ * bins / motion_->frames, bins - 1)] = 1;
    }
    const float prior = Number(cfg_.at("adaptive_uniform_ratio")) / bins;
    for (int i = 0; i < bins; ++i) {
      probabilities_[i] = failed_[i] + prior;
    }
    const float sum = Sum(probabilities_);
    for (auto& value : probabilities_) {
      value /= sum;
    }
    // Torch's one-sample multinomial uses an exponential race, including with
    // replacement=True. The CPU exponential kernel draws double uniforms from
    // two MT words, then casts the exponential to float32 before the division.
    float best = -std::numeric_limits<float>::infinity();
    int selected = 0;
    for (int i = 0; i < bins; ++i) {
      const auto exponential =
          static_cast<float>(-std::log1p(-sim_.random.UnitDouble()));
      const float score = probabilities_[i] / exponential;
      if (score > best) {
        selected = i;
        best = score;
      }
    }
    frame_ = ((static_cast<float>(selected) + sim_.random.Unit()) /
              static_cast<float>(bins)) *
             static_cast<float>(motion_->frames - 1);
    const auto pose = sim_.SamplePose(cfg_.at("pose_range"));
    const auto position = ReferencePosition(0) + Read<3>(pose.data());
    const auto orientation =
        Multiply(Euler(pose[3], pose[4], pose[5]), ReferenceOrientation(0));
    const auto velocity = sim_.SamplePose(cfg_.at("velocity_range"));
    const auto lin =
        Read<3>(motion_->arrays.at("body_lin_vel_w").Frame(frame_) +
                body_indices_[0] * 3) +
        Read<3>(velocity.data());
    const auto ang =
        Read<3>(motion_->arrays.at("body_ang_vel_w").Frame(frame_) +
                body_indices_[0] * 3) +
        Read<3>(velocity.data() + 3);
    const auto* q = motion_->arrays.at("joint_pos").Frame(frame_);
    const auto* v = motion_->arrays.at("joint_vel").Frame(frame_);
    for (std::size_t i = 0; i < robot_.qadr.size(); ++i) {
      const float value = q[i] + sim_.Sample(cfg_.at("joint_position_range"));
      sim_.physics.Get("data.qpos")[robot_.qadr[i]] =
          std::clamp(value, robot_.limits[i * 2], robot_.limits[i * 2 + 1]);
      sim_.physics.Get("data.qvel")[robot_.vadr[i]] = v[i];
    }
    sim_.WritePose(robot_, position, orientation);
    sim_.WriteVelocity(robot_, lin, ang);
    pending_forward_ = true;
  }

  Json cfg_;
  Entity& robot_;
  std::shared_ptr<const Motion> motion_;
  int anchor_{0}, frame_{0};
  std::vector<int> body_indices_;
  std::vector<float> failed_, current_failed_, probabilities_;
  std::vector<float> relative_positions_, relative_orientations_;
  bool pending_forward_{false};
};

}  // namespace

std::unique_ptr<Task> MakeTracking(Simulation* sim,
                                   const std::string& motion_file) {
  return std::make_unique<Tracking>(sim, motion_file);
}

}  // namespace mjlab
