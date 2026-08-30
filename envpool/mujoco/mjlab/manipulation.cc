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
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "envpool/mujoco/mjlab/simulation.h"

namespace mjlab {

float SquaredParam(const Json& p, const std::string& name) {
  const double value = Number(p.at(name));
  return static_cast<float>(value * value);
}

namespace {

class Manipulation : public Task {
 public:
  explicit Manipulation(Simulation* simulation)
      : Task(simulation), cfg_(sim_.cfg.at("command").at("lift_height")) {
    const auto* names = cfg_.as_object().if_contains("entity_names");
    if (names == nullptr) {
      cubes_.push_back(String(cfg_.at("entity_name")));
    } else {
      for (const auto& name : names->as_array()) {
        cubes_.push_back(String(name));
      }
    }
    command.resize(3);
  }

  void Reset() override {
    counter = 0;
    Resample();
  }

  void Update(bool resetting) override {
    cached_object_ = sim_.Position(sim_.entities.at(cubes_[target_]).root);
    const float error = Norm(Read<3>(command.data()) - cached_object_);
    const bool at_goal = error < Param(cfg_, "success_threshold");
    success_ = std::max(success_, at_goal ? 1.0F : 0.0F);
    sim_.metrics["position_error"] = error;
    sim_.metrics["at_goal"] = static_cast<float>(at_goal);
    sim_.metrics["episode_success"] = success_;
    sim_.metrics["object_height"] = cached_object_[2];
    if (!resetting) {
      time_left -= sim_.step_dt;
    }
    if (time_left <= 0) {
      Resample();
      // Official commands teleport objects after the environment's forward.
      // Only this command transition triggers an additional native forward.
      sim_.physics.Run("forward");
      cached_object_ = sim_.Position(sim_.entities.at(cubes_[target_]).root);
    }
  }

  std::vector<float> Observation(const std::string& fn,
                                 const Json& p) override {
    if (fn == "camera_rgb") {
      const auto* pixels = sim_.physics.Get<uint8_t>("camera.rgb");
      const int count = sim_.physics.Count<uint8_t>("camera.rgb") / 3;
      std::vector<float> result(count * 3);
      for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < count; ++i) {
          result[c * count + i] =
              static_cast<float>(pixels[i * 3 + c]) / 255.0F;
        }
      }
      return result;
    }
    if (fn == "camera_depth") {
      const auto* depth = sim_.physics.Get("camera.depth_data");
      const int count = sim_.physics.Count("camera.depth_data");
      std::vector<float> result(count);
      const float cutoff = Param(p, "cutoff_distance");
      for (int i = 0; i < count; ++i) {
        result[i] = std::clamp(
            std::clamp(depth[i], Param(p, "min_depth", 0.01F), cutoff) / cutoff,
            0.0F, 1.0F);
      }
      return result;
    }
    if (fn == "camera_target_cube_mask") {
      const auto* segmentation = sim_.physics.Get<int32_t>("camera.seg_data");
      const int count = sim_.physics.Count<int32_t>("camera.seg_data") / 2;
      const auto& geoms = sim_.entities.at(cubes_[target_]).geoms;
      std::vector<float> result(count);
      for (int i = 0; i < count; ++i) {
        result[i] =
            static_cast<float>(segmentation[i * 2 + 1] == mjOBJ_GEOM &&
                               std::find(geoms.begin(), geoms.end(),
                                         segmentation[i * 2]) != geoms.end());
      }
      return result;
    }
    if (fn == "ee_to_object_distance" || fn == "object_to_goal_distance" ||
        fn == "target_position") {
      const auto& robot = sim_.Asset(p);
      Vec3 value;
      Quat rotation;
      if (fn == "target_position") {
        const int site = robot.sites.at(sim_.Select(robot, p, "site").at(0));
        value = Read<3>(command.data()) - sim_.SitePosition(site);
        rotation = sim_.SiteOrientation(site);
      } else {
        const Vec3 pos =
            sim_.Position(sim_.entities.at(Name(p, "object_name")).root);
        if (fn == "ee_to_object_distance") {
          const int site = robot.sites.at(sim_.Select(robot, p, "site").at(0));
          value = pos - sim_.SitePosition(site);
        } else {
          value = Read<3>(command.data()) - pos;
        }
        rotation = sim_.Orientation(robot.root);
      }
      value = Rotate(Inverse(rotation), value);
      return {value.begin(), value.end()};
    }
    return Task::Observation(fn, p);
  }

  float Reward(const std::string& fn, const Json& p,
               const Json& term) override {
    const bool staged = fn == "staged_position_reward" ||
                        fn == "multi_cube_staged_position_reward";
    if (staged || fn == "bring_object_reward" ||
        fn == "multi_cube_bring_object_reward") {
      const auto pos = cubes_.size() == 1
                           ? sim_.Position(sim_.entities.at(cubes_[0]).root)
                           : cached_object_;
      const float error = SquaredNorm(Read<3>(command.data()) - pos);
      if (!staged) {
        return Exp(-error / SquaredParam(p, "std"));
      }
      const auto& robot = sim_.Asset(p);
      const int site = robot.sites.at(sim_.Select(robot, p, "site").at(0));
      const float reaching = Exp(-SquaredNorm(sim_.SitePosition(site) - pos) /
                                 SquaredParam(p, "reaching_std"));
      return reaching * (1.0F + Exp(-error / SquaredParam(p, "bringing_std")));
    }
    return Task::Reward(fn, p, term);
  }

  Json State() const override {
    auto result = Task::State();
    result.as_object()["target_selection"] = target_;
    result.as_object()["target_pos"] = ToJson(command);
    result.as_object()["cached_target_obj_pos"] = ToJson(cached_object_);
    result.as_object()["episode_success"] = success_;
    return result;
  }

 private:
  void Resample() {
    time_left = sim_.Uniform(cfg_.at("resampling_time_range"));
    ++counter;
    success_ = 0;
    if (cubes_.size() > 1) {
      target_ = sim_.random.Integer(cubes_.size());
    }
    const auto& target_range = cfg_.at("target_position_range");
    int i = 0;
    for (const auto* axis : {"x", "y", "z"}) {
      command[i] = sim_.Sample(target_range.at(axis), true) + sim_.origin[i];
      ++i;
    }
    const auto& object_range = cfg_.at("object_pose_range");
    for (const auto& name : cubes_) {
      Vec3 pos;
      i = 0;
      for (const auto* axis : {"x", "y", "z"}) {
        pos[i] = sim_.Sample(object_range.at(axis), true) + sim_.origin[i];
        ++i;
      }
      const float yaw = sim_.Sample(object_range.at("yaw"));
      const auto& entity = sim_.entities.at(name);
      sim_.WritePose(entity, pos, Euler(0, 0, yaw));
      sim_.WriteVelocity(entity, {}, {});
    }
  }

  Json cfg_;
  std::vector<std::string> cubes_;
  int target_{0};
  float success_{0};
  Vec3 cached_object_{};
};

}  // namespace

std::unique_ptr<Task> MakeManipulation(Simulation* sim) {
  return std::make_unique<Manipulation>(sim);
}

}  // namespace mjlab
