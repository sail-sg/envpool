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
#include "envpool/highway/native_task_env.h"

namespace hn = highway::native;

using PyHighwayEnvSpec = PyEnvSpec<highway::HighwayEnvSpec>;
using PyHighwayEnvPool = PyEnvPool<highway::HighwayEnvPool>;
using PyNativeKinematics5EnvSpec = PyEnvSpec<hn::NativeK5Spec>;
using PyNativeKinematics5EnvPool = PyEnvPool<hn::NativeK5Pool>;
using PyNativeKinematics7Action5EnvSpec = PyEnvSpec<hn::NativeK75Spec>;
using PyNativeKinematics7Action5EnvPool = PyEnvPool<hn::NativeK75Pool>;
using PyNativeKinematics7Action3EnvSpec = PyEnvSpec<hn::NativeK73Spec>;
using PyNativeKinematics7Action3EnvPool = PyEnvPool<hn::NativeK73Pool>;
using PyNativeKinematics8ContinuousEnvSpec = PyEnvSpec<hn::NativeK8CSpec>;
using PyNativeKinematics8ContinuousEnvPool = PyEnvPool<hn::NativeK8CPool>;
using PyNativeTTC5EnvSpec = PyEnvSpec<hn::NativeTTC5Spec>;
using PyNativeTTC5EnvPool = PyEnvPool<hn::NativeTTC5Pool>;
using PyNativeTTC16EnvSpec = PyEnvSpec<hn::NativeTTC16Spec>;
using PyNativeTTC16EnvPool = PyEnvPool<hn::NativeTTC16Pool>;
using PyNativeGoalEnvSpec = PyEnvSpec<hn::NativeGoalSpec>;
using PyNativeGoalEnvPool = PyEnvPool<hn::NativeGoalPool>;
using PyNativeAttributesEnvSpec = PyEnvSpec<hn::NativeAttributesSpec>;
using PyNativeAttributesEnvPool = PyEnvPool<hn::NativeAttributesPool>;
using PyNativeOccupancyEnvSpec = PyEnvSpec<hn::NativeOccupancySpec>;
using PyNativeOccupancyEnvPool = PyEnvPool<hn::NativeOccupancyPool>;
using PyNativeMultiAgentEnvSpec = PyEnvSpec<hn::NativeMultiAgentSpec>;
using PyNativeMultiAgentEnvPool = PyEnvPool<hn::NativeMultiAgentPool>;

// Native highway variants need the same test-only state export as HighwayEnv.
// Keep the pybind class names identical to REGISTER().
#define REGISTER_HIGHWAY_NATIVE(MODULE, SPEC, ENVPOOL)                        \
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
      .def(py::init<const SPEC&>())                                           \
      .def_readonly("_spec", &ENVPOOL::py_spec)                               \
      .def("_recv", &ENVPOOL::PyRecv)                                         \
      .def("_send", &ENVPOOL::PySend)                                         \
      .def("_reset", &ENVPOOL::PyReset)                                       \
      .def("_render", &ENVPOOL::PyRender)                                     \
      .def_readonly_static("_state_keys", &ENVPOOL::py_state_keys)            \
      .def_readonly_static("_action_keys", &ENVPOOL::py_action_keys)          \
      .def("_xla", &ENVPOOL::Xla)                                             \
      .def("_debug_states", &ENVPOOL::DebugStates);

PYBIND11_MODULE(highway_envpool, m) {
  using highway::HighwayDebugState;
  using highway::HighwayLaneDebugState;
  using highway::HighwayRoadObjectDebugState;
  using highway::HighwayVehicleDebugState;

  py::class_<HighwayVehicleDebugState>(m, "_HighwayVehicleDebugState")
      .def_readonly("kind", &HighwayVehicleDebugState::kind)
      .def_readonly("lane_from", &HighwayVehicleDebugState::lane_from)
      .def_readonly("lane_to", &HighwayVehicleDebugState::lane_to)
      .def_readonly("lane_index", &HighwayVehicleDebugState::lane_index)
      .def_readonly("target_lane_from",
                    &HighwayVehicleDebugState::target_lane_from)
      .def_readonly("target_lane_to", &HighwayVehicleDebugState::target_lane_to)
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
      .def_readonly("has_goal", &HighwayVehicleDebugState::has_goal)
      .def_readonly("goal_x", &HighwayVehicleDebugState::goal_x)
      .def_readonly("goal_y", &HighwayVehicleDebugState::goal_y)
      .def_readonly("goal_heading", &HighwayVehicleDebugState::goal_heading)
      .def_readonly("goal_speed", &HighwayVehicleDebugState::goal_speed)
      .def_readonly("idm_delta", &HighwayVehicleDebugState::idm_delta)
      .def_readonly("timer", &HighwayVehicleDebugState::timer)
      .def_readonly("crashed", &HighwayVehicleDebugState::crashed)
      .def_readonly("on_road", &HighwayVehicleDebugState::on_road)
      .def_readonly("check_collisions",
                    &HighwayVehicleDebugState::check_collisions)
      .def_readonly("enable_lane_change",
                    &HighwayVehicleDebugState::enable_lane_change)
      .def_readonly("route_from", &HighwayVehicleDebugState::route_from)
      .def_readonly("route_to", &HighwayVehicleDebugState::route_to)
      .def_readonly("route_id", &HighwayVehicleDebugState::route_id);

  py::class_<HighwayLaneDebugState>(m, "_HighwayLaneDebugState")
      .def_readonly("from", &HighwayLaneDebugState::from)
      .def_readonly("to", &HighwayLaneDebugState::to)
      .def_readonly("index", &HighwayLaneDebugState::index)
      .def_readonly("kind", &HighwayLaneDebugState::kind)
      .def_readonly("start_x", &HighwayLaneDebugState::start_x)
      .def_readonly("start_y", &HighwayLaneDebugState::start_y)
      .def_readonly("end_x", &HighwayLaneDebugState::end_x)
      .def_readonly("end_y", &HighwayLaneDebugState::end_y)
      .def_readonly("center_x", &HighwayLaneDebugState::center_x)
      .def_readonly("center_y", &HighwayLaneDebugState::center_y)
      .def_readonly("width", &HighwayLaneDebugState::width)
      .def_readonly("line_type0", &HighwayLaneDebugState::line_type0)
      .def_readonly("line_type1", &HighwayLaneDebugState::line_type1)
      .def_readonly("forbidden", &HighwayLaneDebugState::forbidden)
      .def_readonly("speed_limit", &HighwayLaneDebugState::speed_limit)
      .def_readonly("priority", &HighwayLaneDebugState::priority)
      .def_readonly("amplitude", &HighwayLaneDebugState::amplitude)
      .def_readonly("pulsation", &HighwayLaneDebugState::pulsation)
      .def_readonly("phase", &HighwayLaneDebugState::phase)
      .def_readonly("radius", &HighwayLaneDebugState::radius)
      .def_readonly("start_phase", &HighwayLaneDebugState::start_phase)
      .def_readonly("end_phase", &HighwayLaneDebugState::end_phase)
      .def_readonly("clockwise", &HighwayLaneDebugState::clockwise);

  py::class_<HighwayRoadObjectDebugState>(m, "_HighwayRoadObjectDebugState")
      .def_readonly("kind", &HighwayRoadObjectDebugState::kind)
      .def_readonly("x", &HighwayRoadObjectDebugState::x)
      .def_readonly("y", &HighwayRoadObjectDebugState::y)
      .def_readonly("heading", &HighwayRoadObjectDebugState::heading)
      .def_readonly("speed", &HighwayRoadObjectDebugState::speed)
      .def_readonly("length", &HighwayRoadObjectDebugState::length)
      .def_readonly("width", &HighwayRoadObjectDebugState::width)
      .def_readonly("collidable", &HighwayRoadObjectDebugState::collidable)
      .def_readonly("solid", &HighwayRoadObjectDebugState::solid)
      .def_readonly("check_collisions",
                    &HighwayRoadObjectDebugState::check_collisions)
      .def_readonly("crashed", &HighwayRoadObjectDebugState::crashed)
      .def_readonly("hit", &HighwayRoadObjectDebugState::hit);

  py::class_<HighwayDebugState>(m, "_HighwayDebugState")
      .def_readonly("scenario", &HighwayDebugState::scenario)
      .def_readonly("lanes_count", &HighwayDebugState::lanes_count)
      .def_readonly("simulation_frequency",
                    &HighwayDebugState::simulation_frequency)
      .def_readonly("policy_frequency", &HighwayDebugState::policy_frequency)
      .def_readonly("elapsed_step", &HighwayDebugState::elapsed_step)
      .def_readonly("time", &HighwayDebugState::time)
      .def_readonly("road_lanes", &HighwayDebugState::road_lanes)
      .def_readonly("road_objects", &HighwayDebugState::road_objects)
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

  REGISTER_HIGHWAY_NATIVE(m, PyNativeKinematics5EnvSpec,
                          PyNativeKinematics5EnvPool)
  REGISTER_HIGHWAY_NATIVE(m, PyNativeKinematics7Action5EnvSpec,
                          PyNativeKinematics7Action5EnvPool)
  REGISTER_HIGHWAY_NATIVE(m, PyNativeKinematics7Action3EnvSpec,
                          PyNativeKinematics7Action3EnvPool)
  REGISTER_HIGHWAY_NATIVE(m, PyNativeKinematics8ContinuousEnvSpec,
                          PyNativeKinematics8ContinuousEnvPool)
  REGISTER_HIGHWAY_NATIVE(m, PyNativeTTC5EnvSpec, PyNativeTTC5EnvPool)
  REGISTER_HIGHWAY_NATIVE(m, PyNativeTTC16EnvSpec, PyNativeTTC16EnvPool)
  REGISTER_HIGHWAY_NATIVE(m, PyNativeGoalEnvSpec, PyNativeGoalEnvPool)
  REGISTER_HIGHWAY_NATIVE(m, PyNativeAttributesEnvSpec,
                          PyNativeAttributesEnvPool)
  REGISTER_HIGHWAY_NATIVE(m, PyNativeOccupancyEnvSpec, PyNativeOccupancyEnvPool)
  REGISTER_HIGHWAY_NATIVE(m, PyNativeMultiAgentEnvSpec,
                          PyNativeMultiAgentEnvPool)
}
