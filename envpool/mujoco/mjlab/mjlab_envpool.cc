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

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "envpool/core/py_envpool.h"
#include "envpool/mujoco/mjlab/mjlab_env.h"

using MjlabEnvSpec = PyEnvSpec<mjlab::MjlabEnvSpec>;
using MjlabEnvPool = PyEnvPool<mjlab::MjlabEnvPool>;

PYBIND11_MODULE(mjlab_envpool, m) {
  REGISTER(m, MjlabEnvSpec, MjlabEnvPool);
  m.attr("REGISTRY_JSON") = mjlab::kRegistry;
#ifdef ENVPOOL_TEST
  auto pool =
      py::reinterpret_borrow<py::class_<MjlabEnvPool>>(m.attr("_MjlabEnvPool"));
  pool.def("_prepare_reset", [](MjlabEnvPool& self, int total_steps,
                                bool promote) {
    self.Inspect(0, [&](mjlab::Simulation& sim) {
      if (sim.steps != 0) {
        throw std::invalid_argument("prepare a curriculum fixture after reset");
      }
      // Initial conditions only, before the oracle's single synchronization.
      // Exercise late training stages without 240,000 warmup physics steps.
      if (sim.cfg.as_object().contains("terrain_state")) {
        const auto& robot = sim.entities.at("robot");
        const float terrain_size = mjlab::Number(
            sim.cfg.at("terrain").at("terrain_generator").at("size").at(0));
        const auto place = [&](float distance) {
          auto pos = sim.Position(robot.root);
          pos[0] = sim.origin[0] + distance;
          pos[1] = sim.origin[1];
          sim.WritePose(robot, pos, sim.Orientation(robot.root));
          sim.physics.Run("forward");
        };
        place(terrain_size);
        if (!promote) {
          sim.total_steps = 1;
          sim.task->Curriculum();
          place(0);
          // Demotion requires a nonzero previous movement request. A standing
          // command intentionally does not penalize zero traveled distance.
          sim.task->command[0] = 1;
          sim.task->command[1] = 0;
        }
      }
      sim.total_steps = total_steps;
    });
  });
  pool.def(
      "_snapshot",
      [](MjlabEnvPool& self, int env_id, bool include_model,
         const std::vector<std::string>& fields) {
        return self.Inspect(env_id, [&](const mjlab::Simulation& sim) {
          py::dict result;
          result["task"] = boost::json::serialize(sim.State());
          py::dict physics;
          for (const auto& field : sim.cfg.at("bindings").as_object()) {
            const std::string name(field.key());
            if (include_model) {
              if (name.rfind("data.", 0) != 0 && name.rfind("model.", 0) != 0) {
                continue;
              }
            } else if (!fields.empty()) {
              if (std::find(fields.begin(), fields.end(), name) ==
                  fields.end()) {
                continue;
              }
            } else if (name != "data.qpos" && name != "data.qvel" &&
                       name != "data.ctrl" && name != "data.xpos" &&
                       name != "data.xquat") {
              continue;
            }
            const std::size_t bytes = sim.physics.Bytes(name);
            physics[py::str(name)] = py::bytes(
                static_cast<const char*>(sim.physics.Pointer(name)), bytes);
          }
          result["physics"] = physics;
          if (include_model) {
            const mjModel* model = sim.physics.Model();
            std::vector<char> bytes(mj_sizeModel(model));
            mj_saveModel(model, nullptr, bytes.data(), bytes.size());
            result["model"] = py::bytes(bytes.data(), bytes.size());
            result["metadata"] = boost::json::serialize(sim.cfg);
          }
          return result;
        });
      },
      py::arg("env_id") = 0, py::arg("include_model") = false,
      py::arg("fields") = std::vector<std::string>{});
#endif
}
