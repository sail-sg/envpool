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

#include <array>
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "envpool/core/py_envpool.h"
#include "envpool/mujoco/locomotion/locomotion_env.h"
#include "third_party/dmc_locomotion/metadata.h"

using LocomotionEnvSpec = PyEnvSpec<mujoco_locomotion::LocomotionEnvSpec>;
using LocomotionEnvPool = PyEnvPool<mujoco_locomotion::LocomotionEnvPool>;

PYBIND11_MODULE(locomotion_envpool, m) {
  REGISTER(m, LocomotionEnvSpec, LocomotionEnvPool);
  m.attr("TASKS") = mujoco_locomotion::kTaskNames;
  m.def("_observation_layout", [](const std::string& task, int team) {
    py::list result;
    for (const auto& item : mujoco_locomotion::ObservationLayout(task, team)) {
      result.append(py::make_tuple(item.name, item.shape, item.storage,
                                   item.boolean, item.offset, item.size));
    }
    return result;
  });
#ifdef ENVPOOL_TEST
  auto pool = py::reinterpret_borrow<py::class_<LocomotionEnvPool>>(
      m.attr("_LocomotionEnvPool"));
  pool.def(
      "_snapshot",
      [](LocomotionEnvPool& self, int env_id, bool include_model) {
        return self.Inspect(
            env_id, [include_model](const mujoco_locomotion::Simulation& sim) {
              py::dict result;
              const mjModel* model = sim.Model();
              const mjData* data = sim.Data();
              for (const auto& [key, ptr, size] :
                   std::vector<std::tuple<const char*, const double*, int>>{
                       {"qpos", data->qpos, model->nq},
                       {"qvel", data->qvel, model->nv},
                       {"act", data->act, model->na},
                       {"warmstart", data->qacc_warmstart, model->nv},
                       {"ctrl", data->ctrl, model->nu},
                       {"sensordata", data->sensordata, model->nsensordata}}) {
                py::array_t<double> array(size);
                if (size > 0)
                  std::memcpy(array.mutable_data(), ptr, size * sizeof(double));
                result[py::str(key)] = array;
              }
              result["time"] = data->time;
              if (include_model) {
                std::vector<char> bytes(mj_sizeModel(model));
                mj_saveModel(model, nullptr, bytes.data(), bytes.size());
                result["model"] = py::bytes(bytes.data(), bytes.size());
                result["maze"] = sim.GetScene().maze_entities;
                result["variations"] = sim.GetScene().maze_variations;
              }
              return result;
            });
      },
      py::arg("env_id") = 0, py::arg("include_model") = false);
  pool.def(
      "_set_reset_state",
      [](LocomotionEnvPool& self, int env_id, const std::vector<double>& qpos,
         const std::vector<double>& qvel,
         const std::map<std::string, std::array<double, 3>>& geoms) {
        self.Inspect(env_id, [&](mujoco_locomotion::Simulation& sim) {
          sim.SetResetState(qpos, qvel, geoms);
        });
      },
      py::arg("env_id"), py::arg("qpos"), py::arg("qvel"),
      py::arg("geoms") = std::map<std::string, std::array<double, 3>>{});
#endif
}
