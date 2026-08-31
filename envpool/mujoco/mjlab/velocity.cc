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
#include <string>
#include <utility>
#include <vector>

#include "envpool/mujoco/mjlab/simulation.h"
#include "envpool/mujoco/mjlab/terrain.h"

namespace mjlab {
namespace {

class Velocity : public Task {
 public:
  explicit Velocity(Simulation* simulation)
      : Task(simulation),
        cfg_(sim_.cfg.at("command").at("twist")),
        terrain_(simulation) {
    command.resize(3);
    const auto& sensors = sim_.cfg.at("sensors").as_object();
    const auto* const found = sensors.find("foot_height_scan");
    peak_heights_.resize(Number(found->value().at("state").at("_num_frames")));
  }

  void Reset() override {
    counter = 0;
    Resample();
  }

  void Update(bool resetting) override {
    if (!resetting) {
      time_left -= sim_.step_dt;
    }
    if (time_left <= 0) {
      Resample();
    }
    const auto& robot = sim_.entities.at("robot");
    const auto forward = Rotate(sim_.Orientation(robot.root), {1, 0, 0});
    const float heading = Atan2(forward[1], forward[0]);
    const auto pi = static_cast<float>(3.14159265358979323846);
    const float input = heading_target_ - heading;
    float wrapped = std::fmod(input + pi, 2.0F * pi);
    if (wrapped < 0) {
      wrapped += 2.0F * pi;
    }
    wrapped -= pi;
    heading_error_ = wrapped == -pi && input > 0 ? pi : wrapped;
    const auto& range = cfg_.at("ranges");
    if (is_heading_) {
      command[2] =
          std::clamp(Param(cfg_, "heading_control_stiffness") * heading_error_,
                     static_cast<float>(Number(range.at("ang_vel_z").at(0))),
                     static_cast<float>(Number(range.at("ang_vel_z").at(1))));
    }
    if (is_world_) {
      const float cosine = Cos(heading);
      const float sine = Sin(heading);
      command[0] = cosine * world_command_[0] + sine * world_command_[1];
      command[1] = -sine * world_command_[0] + cosine * world_command_[1];
    }
    if (is_standing_) {
      std::fill(command.begin(), command.end(), 0);
      world_command_ = {};
    }
  }

  void Curriculum() override {
    terrain_.Curriculum(command);
    for (const auto& entry : sim_.cfg.at("curriculum").as_object()) {
      const auto& term = entry.value();
      if (Function(term) != "commands_vel") {
        continue;
      }
      for (const auto& stage :
           term.at("params").at("velocity_stages").as_array()) {
        if (sim_.total_steps < Param(stage, "step")) {
          continue;
        }
        for (const auto* axis : {"lin_vel_x", "lin_vel_y", "ang_vel_z"}) {
          const auto* range = stage.as_object().if_contains(axis);
          if (range != nullptr && !range->is_null()) {
            cfg_.at("ranges").at(axis) = *range;
          }
        }
      }
    }
  }

  bool Terminated(const std::string& fn, const Json& p) override {
    if (fn == "out_of_terrain_bounds") {
      return terrain_.Outside(Param(p, "margin", 0.3F));
    }
    return Task::Terminated(fn, p);
  }

  float Reward(const std::string& fn, const Json& p,
               const Json& term) override {
    const float speed = std::sqrt(Square(command[0]) + Square(command[1])) +
                        std::abs(command[2]);
    const bool active = speed > Param(p, "command_threshold", 0.01F);
    if (fn == "track_linear_velocity" || fn == "track_angular_velocity") {
      const auto& robot = sim_.Asset(p);
      auto velocity = fn == "track_linear_velocity"
                          ? sim_.LinearVelocity(robot.root, robot.root,
                                                sim_.Position(robot.root))
                          : sim_.AngularVelocity(robot.root);
      velocity = RotateInverse(sim_.Orientation(robot.root), velocity);
      const float error = fn == "track_linear_velocity"
                              ? (Square(command[0] - velocity[0]) +
                                 Square(command[1] - velocity[1])) +
                                    Square(velocity[2])
                              : Square(command[2] - velocity[2]) +
                                    (Square(velocity[0]) + Square(velocity[1]));
      return Exp(-error / SquaredParam(p, "std"));
    }
    if (fn == "upright") {
      const auto& robot = sim_.Asset(p);
      const auto ids = sim_.Select(robot, p, "body");
      const int body = robot.bodies.at(ids.at(0));
      const auto* sensors = p.as_object().if_contains("terrain_sensor_names");
      const Vec3 target = sensors == nullptr || sensors->is_null()
                              ? Vec3{0, 0, -1}
                              : TerrainNormal(*sensors);
      const auto local = RotateInverse(sim_.Orientation(body), target);
      return Exp(-(Square(local[0]) + Square(local[1])) /
                 SquaredParam(p, "std"));
    }
    if (fn == "variable_posture") {
      const auto& robot = sim_.Asset(p);
      const auto& state = term.at("func").at("state");
      const char* name = "std_running";
      if (speed < Param(p, "walking_threshold", 0.5F)) {
        name = "std_standing";
      } else if (speed < Param(p, "running_threshold", 1.5F)) {
        name = "std_walking";
      }
      const auto deviations = Floats(state.at(name));
      const auto positions = sim_.JointPositions(robot);
      const auto ids = sim_.Select(robot, p, "joint");
      std::vector<float> errors(ids.size());
      for (std::size_t i = 0; i < ids.size(); ++i) {
        errors[i] = Square(positions[ids[i]] - robot.joint_default[ids[i]]) /
                    Square(deviations[i]);
      }
      return Exp(-(Sum(errors) / static_cast<float>(ids.size())));
    }
    if (fn == "body_angular_velocity_penalty") {
      const auto& robot = sim_.Asset(p);
      const auto velocity = sim_.AngularVelocity(
          robot.bodies.at(sim_.Select(robot, p, "body").at(0)));
      return Square(velocity[0]) + Square(velocity[1]);
    }
    if (fn == "angular_momentum_penalty") {
      const auto value = SquaredNorm(sim_.Sensor(Name(p, "sensor_name")));
      sim_.metrics["Metrics/angular_momentum_mean"] = std::sqrt(value);
      return value;
    }
    if (fn == "feet_clearance" || fn == "feet_slip") {
      const auto& robot = sim_.Asset(p);
      const auto sites = sim_.Select(robot, p, "site");
      const auto heights = fn == "feet_clearance"
                               ? sim_.Heights(Name(p, "height_sensor_name"))
                               : std::vector<float>{};
      float cost = 0;
      float metric = 0;
      float count = 0;
      for (std::size_t i = 0; i < sites.size(); ++i) {
        const auto v = sim_.SiteVelocity(robot.sites[sites[i]], robot.root);
        const float norm = Norm(std::array<float, 2>{v[0], v[1]});
        if (fn == "feet_clearance") {
          cost += std::abs(heights[i] - Param(p, "target_height")) * norm;
        } else if (sim_.contacts.at(Name(p, "sensor_name")).found[i] > 0) {
          cost += norm * norm;
          metric += norm;
          count += 1;
        }
      }
      if (fn == "feet_slip") {
        sim_.metrics["Metrics/slip_velocity_mean"] =
            metric / std::max(count, 1.0F);
      }
      return cost * (active ? 1.0F : 0.0F);
    }
    if (fn == "feet_swing_height" || fn == "soft_landing" ||
        fn == "feet_air_time") {
      const auto& sensor = sim_.contacts.at(Name(p, "sensor_name"));
      const auto heights = fn == "feet_swing_height"
                               ? sim_.Heights(Name(p, "height_sensor_name"))
                               : std::vector<float>{};
      float cost = 0;
      float metric = 0;
      float count = 0;
      for (std::size_t i = 0; i < sensor.found.size(); ++i) {
        const bool landing =
            sensor.contact[i] > 0 && sensor.contact[i] < sim_.step_dt + 1.0e-8F;
        if (fn == "feet_swing_height") {
          if (sensor.found[i] == 0) {
            peak_heights_[i] = std::max(peak_heights_[i], heights[i]);
          }
          if (landing) {
            cost += Square(peak_heights_[i] / Param(p, "target_height") - 1.0F);
            metric += peak_heights_[i];
            count += 1;
            peak_heights_[i] = 0;
          }
        } else if (fn == "soft_landing" && landing) {
          const float norm = Norm(Read<3>(sensor.force.data() + i * 3));
          cost += norm;
          metric += norm;
          count += 1;
        } else if (fn == "feet_air_time") {
          cost += static_cast<float>(
              sensor.air[i] > Param(p, "threshold_min", 0.05F) &&
              sensor.air[i] < Param(p, "threshold_max", 0.5F));
          if (sensor.air[i] > 0) {
            metric += sensor.air[i];
            count += 1;
          }
        }
      }
      const char* name = "Metrics/air_time_mean";
      if (fn == "feet_swing_height") {
        name = "Metrics/peak_height_mean";
      } else if (fn == "soft_landing") {
        name = "Metrics/landing_force_mean";
      }
      sim_.metrics[name] = metric / std::max(count, 1.0F);
      return cost * (active ? 1.0F : 0.0F);
    }
    return Task::Reward(fn, p, term);
  }

  Json State() const override {
    auto result = Task::State();
    auto& obj = result.as_object();
    obj["vel_command_b"] = ToJson(command);
    obj["vel_command_w"] = ToJson(world_command_);
    obj["heading_target"] = heading_target_;
    obj["heading_error"] = heading_error_;
    obj["is_heading_env"] = is_heading_;
    obj["is_standing_env"] = is_standing_;
    obj["is_world_env"] = is_world_;
    obj["is_forward_env"] = is_forward_;
    obj["peak_heights"] = ToJson(peak_heights_);
    obj["terrain"] = terrain_.State();
    return result;
  }

 private:
  void Resample() {
    time_left = sim_.Uniform(cfg_.at("resampling_time_range"));
    ++counter;
    const auto& ranges = cfg_.at("ranges");
    int index = 0;
    for (const auto* key : {"lin_vel_x", "lin_vel_y", "ang_vel_z"}) {
      command[index++] = sim_.Uniform(ranges.at(key));
    }
    heading_target_ = sim_.Uniform(ranges.at("heading"));
    is_heading_ = sim_.random.Unit() <= Param(cfg_, "rel_heading_envs");
    is_standing_ = sim_.random.Unit() <= Param(cfg_, "rel_standing_envs");
    is_world_ = sim_.random.Unit() <= Param(cfg_, "rel_world_envs");
    world_command_ = Read<3>(command.data());
    is_forward_ = sim_.random.Unit() <= Param(cfg_, "rel_forward_envs");
    if (is_forward_) {
      command[0] = std::max(std::abs(command[0]), 0.3F);
      command[1] = command[2] = 0;
    }
  }

  Vec3 TerrainNormal(const Json& names) const {
    std::vector<std::pair<Vec3, float>> points;
    float valid = 0;
    for (const auto& value : names.as_array()) {
      const std::string name = String(value);
      const std::string prefix = "ray." + name;
      const int count = sim_.physics.Count(prefix + "._ray_dist");
      const int samples = std::min(count, 32);
      for (int i = 0; i < samples; ++i) {
        const int id =
            count > 32
                ? static_cast<int>(i * (static_cast<float>(count - 1) / 31.0F))
                : i;
        const float dist = sim_.physics.Get(prefix + "._ray_dist")[id];
        const float mask = dist >= 0 ? 1.0F : 0.0F;
        valid += mask;
        points.emplace_back(
            Read<3>(sim_.physics.Get(prefix + "._ray_pnt") + id * 3) +
                Read<3>(sim_.physics.Get(prefix + "._ray_vec") + id * 3) * dist,
            mask);
      }
    }
    if (valid < 3) {
      return {0, 0, 1};
    }
    Vec3 mean{};
    for (int axis = 0; axis < 3; ++axis) {
      mean[axis] = Sum(points,
                       [axis](const auto& point) {
                         return point.first[axis] * point.second;
                       }) /
                   valid;
    }
    Mat3 covariance{};
    for (const auto& [point, mask] : points) {
      const auto centered = (point - mean) * mask;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          covariance[i * 3 + j] += centered[i] * centered[j];
        }
      }
    }
    for (auto v : covariance) {
      if (!std::isfinite(v)) {
        return {0, 0, 1};
      }
    }
    const auto eigen = SymmetricEigen(covariance);
    const float small = eigen.values[0];
    const float middle = eigen.values[1];
    const float large = eigen.values[2];
    const float eps = std::numeric_limits<float>::epsilon();
    if (small / std::max(middle, eps) >= 0.1F ||
        middle <= std::max(large, eps) * 1.0e-6F) {
      return {0, 0, 1};
    }
    Vec3 normal = Read<3>(eigen.vectors.data());
    const float divisor = std::max(Norm(normal), 1.0e-8F);
    for (auto& value : normal) {
      value /= divisor;
    }
    if (normal[2] < 0) {
      normal = normal * -1.0F;
    }
    return normal;
  }

  Json cfg_;
  Terrain terrain_;
  Vec3 world_command_{};
  float heading_target_{0}, heading_error_{0};
  bool is_heading_{false}, is_standing_{false}, is_world_{false},
      is_forward_{false};
  std::vector<float> peak_heights_;
};

}  // namespace

std::unique_ptr<Task> MakeVelocity(Simulation* sim) {
  return std::make_unique<Velocity>(sim);
}

}  // namespace mjlab
