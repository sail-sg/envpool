// Copyright 2026 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/pybind11.h>

#include "envpool/core/py_envpool.h"
#include "envpool/highway/highway_env.h"
#include "envpool/highway/official_bridge.h"

namespace highway {
namespace py = pybind11;

using PyHighwayEnvSpec = PyEnvSpec<HighwayEnvSpec>;
using PyHighwayEnvPool = PyEnvPool<HighwayEnvPool>;

using PyOfficialKinematics5EnvSpec =
    PyEnvSpec<official::OfficialKinematics5EnvSpec>;
using PyOfficialKinematics5EnvPool =
    PyEnvPool<official::OfficialKinematics5EnvPool>;
using PyOfficialKinematics7Action5EnvSpec =
    PyEnvSpec<official::OfficialKinematics7Action5EnvSpec>;
using PyOfficialKinematics7Action5EnvPool =
    PyEnvPool<official::OfficialKinematics7Action5EnvPool>;
using PyOfficialKinematics7Action3EnvSpec =
    PyEnvSpec<official::OfficialKinematics7Action3EnvSpec>;
using PyOfficialKinematics7Action3EnvPool =
    PyEnvPool<official::OfficialKinematics7Action3EnvPool>;
using PyOfficialKinematics8ContinuousEnvSpec =
    PyEnvSpec<official::OfficialKinematics8ContinuousEnvSpec>;
using PyOfficialKinematics8ContinuousEnvPool =
    PyEnvPool<official::OfficialKinematics8ContinuousEnvPool>;
using PyOfficialTTC5EnvSpec = PyEnvSpec<official::OfficialTTC5EnvSpec>;
using PyOfficialTTC5EnvPool = PyEnvPool<official::OfficialTTC5EnvPool>;
using PyOfficialTTC16EnvSpec = PyEnvSpec<official::OfficialTTC16EnvSpec>;
using PyOfficialTTC16EnvPool = PyEnvPool<official::OfficialTTC16EnvPool>;
using PyOfficialGoalEnvSpec = PyEnvSpec<official::OfficialGoalEnvSpec>;
using PyOfficialGoalEnvPool = PyEnvPool<official::OfficialGoalEnvPool>;
using PyOfficialAttributesEnvSpec =
    PyEnvSpec<official::OfficialAttributesEnvSpec>;
using PyOfficialAttributesEnvPool =
    PyEnvPool<official::OfficialAttributesEnvPool>;
using PyOfficialOccupancyEnvSpec =
    PyEnvSpec<official::OfficialOccupancyEnvSpec>;
using PyOfficialOccupancyEnvPool =
    PyEnvPool<official::OfficialOccupancyEnvPool>;
using PyOfficialMultiAgentEnvSpec =
    PyEnvSpec<official::OfficialMultiAgentEnvSpec>;
using PyOfficialMultiAgentEnvPool =
    PyEnvPool<official::OfficialMultiAgentEnvPool>;

}  // namespace highway

#define REGISTER_HIGHWAY_OFFICIAL(MODULE, SPEC, ENVPOOL)                      \
  py::class_<SPEC>(MODULE, "_" #SPEC,                                         \
                   py::metaclass(py::module_::import("abc").attr("ABCMeta"))) \
      .def(py::init<const typename SPEC::ConfigValues&>())                    \
      .def_readonly("_config_values", &SPEC::py_config_values)                \
      .def_property_readonly(                                                 \
          "_state_spec", [](const SPEC& self) { return self.StateSpecPy(); }) \
      .def_property_readonly(                                                 \
          "_action_spec",                                                     \
          [](const SPEC& self) { return self.ActionSpecPy(); })               \
      .def_readonly_static("_state_keys", &SPEC::py_state_keys)               \
      .def_readonly_static("_action_keys", &SPEC::py_action_keys)             \
      .def_readonly_static("_config_keys", &SPEC::py_config_keys)             \
      .def_readonly_static("_default_config_values",                          \
                           &SPEC::py_default_config_values);                  \
  py::class_<ENVPOOL>(                                                        \
      MODULE, "_" #ENVPOOL,                                                   \
      py::metaclass(py::module_::import("abc").attr("ABCMeta")))              \
      .def(py::init<const SPEC&>(), py::call_guard<py::gil_scoped_release>()) \
      .def_readonly("_spec", &ENVPOOL::py_spec)                               \
      .def("_recv", &ENVPOOL::PyRecv)                                         \
      .def("_send", &ENVPOOL::PySend)                                         \
      .def("_reset", &ENVPOOL::PyReset)                                       \
      .def("_render", &ENVPOOL::PyRender)                                     \
      .def_readonly_static("_state_keys", &ENVPOOL::py_state_keys)            \
      .def_readonly_static("_action_keys", &ENVPOOL::py_action_keys)          \
      .def("_xla", &ENVPOOL::Xla);

PYBIND11_MODULE(highway_envpool, m) {
  using highway::HighwayDebugState;
  using highway::HighwayVehicleDebugState;
  using highway::PyHighwayEnvPool;
  using highway::PyHighwayEnvSpec;
  using highway::PyOfficialAttributesEnvPool;
  using highway::PyOfficialAttributesEnvSpec;
  using highway::PyOfficialGoalEnvPool;
  using highway::PyOfficialGoalEnvSpec;
  using highway::PyOfficialKinematics5EnvPool;
  using highway::PyOfficialKinematics5EnvSpec;
  using highway::PyOfficialKinematics7Action3EnvPool;
  using highway::PyOfficialKinematics7Action3EnvSpec;
  using highway::PyOfficialKinematics7Action5EnvPool;
  using highway::PyOfficialKinematics7Action5EnvSpec;
  using highway::PyOfficialKinematics8ContinuousEnvPool;
  using highway::PyOfficialKinematics8ContinuousEnvSpec;
  using highway::PyOfficialMultiAgentEnvPool;
  using highway::PyOfficialMultiAgentEnvSpec;
  using highway::PyOfficialOccupancyEnvPool;
  using highway::PyOfficialOccupancyEnvSpec;
  using highway::PyOfficialTTC16EnvPool;
  using highway::PyOfficialTTC16EnvSpec;
  using highway::PyOfficialTTC5EnvPool;
  using highway::PyOfficialTTC5EnvSpec;

  py::class_<HighwayVehicleDebugState>(m, "_HighwayVehicleDebugState")
      .def_readonly("kind", &HighwayVehicleDebugState::kind)
      .def_readonly("lane_index", &HighwayVehicleDebugState::lane_index)
      .def_readonly("target_lane_index",
                    &HighwayVehicleDebugState::target_lane_index)
      .def_readonly("speed_index", &HighwayVehicleDebugState::speed_index)
      .def_readonly("x", &HighwayVehicleDebugState::x)
      .def_readonly("y", &HighwayVehicleDebugState::y)
      .def_readonly("heading", &HighwayVehicleDebugState::heading)
      .def_readonly("speed", &HighwayVehicleDebugState::speed)
      .def_readonly("target_speed", &HighwayVehicleDebugState::target_speed)
      .def_readonly("target_speed0", &HighwayVehicleDebugState::target_speed0)
      .def_readonly("target_speed1", &HighwayVehicleDebugState::target_speed1)
      .def_readonly("target_speed2", &HighwayVehicleDebugState::target_speed2)
      .def_readonly("idm_delta", &HighwayVehicleDebugState::idm_delta)
      .def_readonly("timer", &HighwayVehicleDebugState::timer)
      .def_readonly("crashed", &HighwayVehicleDebugState::crashed)
      .def_readonly("on_road", &HighwayVehicleDebugState::on_road)
      .def_readonly("check_collisions",
                    &HighwayVehicleDebugState::check_collisions);

  py::class_<HighwayDebugState>(m, "_HighwayDebugState")
      .def_readonly("lanes_count", &HighwayDebugState::lanes_count)
      .def_readonly("simulation_frequency",
                    &HighwayDebugState::simulation_frequency)
      .def_readonly("policy_frequency", &HighwayDebugState::policy_frequency)
      .def_readonly("elapsed_step", &HighwayDebugState::elapsed_step)
      .def_readonly("time", &HighwayDebugState::time)
      .def_readonly("vehicles", &HighwayDebugState::vehicles);

  py::class_<PyHighwayEnvSpec>(
      m, "_HighwayEnvSpec",
      py::metaclass(py::module_::import("abc").attr("ABCMeta")))
      .def(py::init<const PyHighwayEnvSpec::ConfigValues&>())
      .def_readonly("_config_values", &PyHighwayEnvSpec::py_config_values)
      .def_property_readonly(
          "_state_spec",
          [](const PyHighwayEnvSpec& self) { return self.StateSpecPy(); })
      .def_property_readonly(
          "_action_spec",
          [](const PyHighwayEnvSpec& self) { return self.ActionSpecPy(); })
      .def_readonly_static("_state_keys", &PyHighwayEnvSpec::py_state_keys)
      .def_readonly_static("_action_keys", &PyHighwayEnvSpec::py_action_keys)
      .def_readonly_static("_config_keys", &PyHighwayEnvSpec::py_config_keys)
      .def_readonly_static("_default_config_values",
                           &PyHighwayEnvSpec::py_default_config_values);

  py::class_<PyHighwayEnvPool>(
      m, "_HighwayEnvPool",
      py::metaclass(py::module_::import("abc").attr("ABCMeta")))
      .def(py::init<const PyHighwayEnvSpec&>())
      .def_readonly("_spec", &PyHighwayEnvPool::py_spec)
      .def("_recv", &PyHighwayEnvPool::PyRecv)
      .def("_send", &PyHighwayEnvPool::PySend)
      .def("_reset", &PyHighwayEnvPool::PyReset)
      .def("_render", &PyHighwayEnvPool::PyRender)
      .def_readonly_static("_state_keys", &PyHighwayEnvPool::py_state_keys)
      .def_readonly_static("_action_keys", &PyHighwayEnvPool::py_action_keys)
      .def("_xla", &PyHighwayEnvPool::Xla)
      .def("_debug_states", &PyHighwayEnvPool::DebugStates);

  REGISTER_HIGHWAY_OFFICIAL(m, PyOfficialKinematics5EnvSpec,
                            PyOfficialKinematics5EnvPool)
  REGISTER_HIGHWAY_OFFICIAL(m, PyOfficialKinematics7Action5EnvSpec,
                            PyOfficialKinematics7Action5EnvPool)
  REGISTER_HIGHWAY_OFFICIAL(m, PyOfficialKinematics7Action3EnvSpec,
                            PyOfficialKinematics7Action3EnvPool)
  REGISTER_HIGHWAY_OFFICIAL(m, PyOfficialKinematics8ContinuousEnvSpec,
                            PyOfficialKinematics8ContinuousEnvPool)
  REGISTER_HIGHWAY_OFFICIAL(m, PyOfficialTTC5EnvSpec, PyOfficialTTC5EnvPool)
  REGISTER_HIGHWAY_OFFICIAL(m, PyOfficialTTC16EnvSpec, PyOfficialTTC16EnvPool)
  REGISTER_HIGHWAY_OFFICIAL(m, PyOfficialGoalEnvSpec, PyOfficialGoalEnvPool)
  REGISTER_HIGHWAY_OFFICIAL(m, PyOfficialAttributesEnvSpec,
                            PyOfficialAttributesEnvPool)
  REGISTER_HIGHWAY_OFFICIAL(m, PyOfficialOccupancyEnvSpec,
                            PyOfficialOccupancyEnvPool)
  REGISTER_HIGHWAY_OFFICIAL(m, PyOfficialMultiAgentEnvSpec,
                            PyOfficialMultiAgentEnvPool)
}
