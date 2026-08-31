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
#include <stdexcept>
#include <string>
#include <vector>

#include "envpool/mujoco/mjlab/simulation.h"

namespace mjlab {

void Simulation::Events(const std::string& mode) {
  bool recompute = false;
  for (const auto& entry : cfg.at("event").as_object()) {
    const auto& term = entry.value();
    if (Name(term, "mode") != mode) {
      continue;
    }
    if (mode == "interval") {
      auto& timer = event_timers[std::string(entry.key())];
      timer -= step_dt;
      if (timer >= 1.0e-6F) {
        continue;
      }
      timer = Sample(term.at("interval_range_s"));
    }
    const auto fn = Function(term);
    const auto& p = term.at("params");
    auto& entity = Asset(p);
    if (fn == "reset_root_state_uniform") {
      const auto pose = SamplePose(p.at("pose_range"));
      const auto& root = entity.root_default;
      const Vec3 pos = (Read<3>(root.data()) + Read<3>(pose.data())) + origin;
      const auto quat =
          Multiply(Read<4>(root.data() + 3), Euler(pose[3], pose[4], pose[5]));
      WritePose(entity, pos, quat);
      if (!entity.freeq.empty()) {
        const auto velocity = SamplePose(p.at("velocity_range"));
        WriteVelocity(entity,
                      Read<3>(root.data() + 7) + Read<3>(velocity.data()),
                      Read<3>(root.data() + 10) + Read<3>(velocity.data() + 3));
      }
    } else if (fn == "reset_joints_by_offset") {
      const auto ids = Select(entity, p, "joint");
      for (int id : ids) {
        const float value =
            entity.joint_default[id] + Sample(p.at("position_range"));
        physics.Get("data.qpos")[entity.qadr[id]] =
            std::clamp(value, entity.limits[id * 2], entity.limits[id * 2 + 1]);
      }
      for (int id : ids) {
        physics.Get("data.qvel")[entity.vadr[id]] =
            entity.velocity_default[id] + Sample(p.at("velocity_range"));
      }
    } else if (fn == "push_by_setting_velocity") {
      const auto perturbation = SamplePose(p.at("velocity_range"));
      WriteVelocity(
          entity,
          LinearVelocity(entity.root, entity.root, Position(entity.root)) +
              Read<3>(perturbation.data()),
          AngularVelocity(entity.root) + Read<3>(perturbation.data() + 3));
    } else if (fn == "encoder_bias") {
      for (int id : Select(entity, p, "joint")) {
        entity.bias[id] = Sample(p.at("bias_range"));
      }
    } else if (fn == "geom_friction" || fn == "geom_rgba" ||
               fn == "body_com_offset") {
      const bool body = fn == "body_com_offset";
      const std::string field = body ? "body_ipos" : fn;
      const int width = fn == "geom_rgba" ? 4 : 3;
      const auto ids = Select(entity, p, body ? "body" : "geom");
      const auto& mapping = body ? entity.bodies : entity.geoms;
      const auto defaults = Floats(cfg.at("default_model_fields").at(field));
      auto* target = physics.Get("model." + field);
      const auto& ranges = p.at("ranges");
      std::vector<int> axes;
      const auto* explicit_axes = p.as_object().if_contains("axes");
      if (explicit_axes != nullptr && !explicit_axes->is_null()) {
        axes = Indices(*explicit_axes);
      } else if (ranges.is_object()) {
        for (const auto& range : ranges.as_object()) {
          axes.push_back(std::stoi(std::string(range.key())));
        }
      } else if (fn == "geom_friction") {
        axes = {0};
      } else {
        for (int i = 0; i < width; ++i) {
          axes.push_back(i);
        }
      }
      for (int axis : axes) {
        const auto& range =
            ranges.is_object() ? ranges.at(std::to_string(axis)) : ranges;
        float sample = 0;
        for (std::size_t i = 0; i < ids.size(); ++i) {
          if (i == 0 || !Flag(p, "shared_random")) {
            if (Name(p, "distribution", "uniform") == "log_uniform") {
              const float lo = Log(static_cast<float>(Number(range.at(0))));
              const float hi = Log(static_cast<float>(Number(range.at(1))));
              sample = Exp(random.Unit() * (hi - lo) + lo);
            } else {
              sample = Sample(range, true);
            }
          }
          const int address = mapping[ids[i]] * width + axis;
          const auto operation = Name(p, "operation", "abs");
          if (operation == "add") {
            target[address] = defaults[address] + sample;
          } else if (operation == "scale") {
            target[address] = defaults[address] * sample;
          } else if (operation == "abs") {
            target[address] = sample;
          } else {
            throw std::invalid_argument(
                "unexpected pinned randomization operation");
          }
          // The ordinary MuJoCo model is used only for public GL rendering.
          // Keep its appearance and inertial frame consistent with active Warp.
          if (body) {
            physics.Model()->body_ipos[address] = target[address];
          } else if (fn == "geom_rgba") {
            physics.Model()->geom_rgba[address] = target[address];
          } else {
            physics.Model()->geom_friction[address] = target[address];
          }
        }
      }
      recompute |= body;
    } else {
      throw std::invalid_argument("unimplemented MJLab event: " + fn);
    }
  }
  if (recompute) {
    physics.Run("set_const");
  }
}

}  // namespace mjlab
