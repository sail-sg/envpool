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
#include "envpool/minigrid/minigrid.h"

namespace minigrid {

namespace py = pybind11;

using PyMiniGridEnvSpec = PyEnvSpec<MiniGridEnvSpec>;
using PyMiniGridEnvPool = PyEnvPool<MiniGridEnvPool>;

void BindMiniGrid(py::module_& m) {
  py::class_<MiniGridDebugState>(m, "_MiniGridDebugState")
      .def_readonly("env_name", &MiniGridDebugState::env_name)
      .def_readonly("mission", &MiniGridDebugState::mission)
      .def_readonly("mission_id", &MiniGridDebugState::mission_id)
      .def_readonly("width", &MiniGridDebugState::width)
      .def_readonly("height", &MiniGridDebugState::height)
      .def_readonly("action_max", &MiniGridDebugState::action_max)
      .def_readonly("max_steps", &MiniGridDebugState::max_steps)
      .def_readonly("grid", &MiniGridDebugState::grid)
      .def_readonly("grid_contains", &MiniGridDebugState::grid_contains)
      .def_readonly("obstacle_positions",
                    &MiniGridDebugState::obstacle_positions)
      .def_readonly("agent_pos", &MiniGridDebugState::agent_pos)
      .def_readonly("agent_dir", &MiniGridDebugState::agent_dir)
      .def_readonly("has_carrying", &MiniGridDebugState::has_carrying)
      .def_readonly("carrying_type", &MiniGridDebugState::carrying_type)
      .def_readonly("carrying_color", &MiniGridDebugState::carrying_color)
      .def_readonly("carrying_state", &MiniGridDebugState::carrying_state)
      .def_readonly("carrying_has_contains",
                    &MiniGridDebugState::carrying_has_contains)
      .def_readonly("carrying_contains_type",
                    &MiniGridDebugState::carrying_contains_type)
      .def_readonly("carrying_contains_color",
                    &MiniGridDebugState::carrying_contains_color)
      .def_readonly("carrying_contains_state",
                    &MiniGridDebugState::carrying_contains_state)
      .def_readonly("target_pos", &MiniGridDebugState::target_pos)
      .def_readonly("target_type", &MiniGridDebugState::target_type)
      .def_readonly("target_color", &MiniGridDebugState::target_color)
      .def_readonly("move_pos", &MiniGridDebugState::move_pos)
      .def_readonly("move_type", &MiniGridDebugState::move_type)
      .def_readonly("move_color", &MiniGridDebugState::move_color)
      .def_readonly("success_pos", &MiniGridDebugState::success_pos)
      .def_readonly("failure_pos", &MiniGridDebugState::failure_pos)
      .def_readonly("goal_pos", &MiniGridDebugState::goal_pos);

  py::class_<PyMiniGridEnvSpec>(
      m, "_MiniGridEnvSpec",
      py::metaclass(py::module_::import("abc").attr("ABCMeta")))
      .def(py::init<const PyMiniGridEnvSpec::ConfigValues&>())
      .def_readonly("_config_values", &PyMiniGridEnvSpec::py_config_values)
      .def_readonly("_state_spec", &PyMiniGridEnvSpec::py_state_spec)
      .def_readonly("_action_spec", &PyMiniGridEnvSpec::py_action_spec)
      .def_readonly_static("_state_keys", &PyMiniGridEnvSpec::py_state_keys)
      .def_readonly_static("_action_keys", &PyMiniGridEnvSpec::py_action_keys)
      .def_readonly_static("_config_keys", &PyMiniGridEnvSpec::py_config_keys)
      .def_readonly_static("_default_config_values",
                           &PyMiniGridEnvSpec::py_default_config_values);

  py::class_<PyMiniGridEnvPool>(
      m, "_MiniGridEnvPool",
      py::metaclass(py::module_::import("abc").attr("ABCMeta")))
      .def(py::init<const PyMiniGridEnvSpec&>())
      .def_readonly("_spec", &PyMiniGridEnvPool::py_spec)
      .def("_recv", &PyMiniGridEnvPool::PyRecv)
      .def("_send", &PyMiniGridEnvPool::PySend)
      .def("_reset", &PyMiniGridEnvPool::PyReset)
      .def("_render", &PyMiniGridEnvPool::PyRender)
      .def_readonly_static("_state_keys", &PyMiniGridEnvPool::py_state_keys)
      .def_readonly_static("_action_keys", &PyMiniGridEnvPool::py_action_keys)
      .def("_xla", &PyMiniGridEnvPool::Xla)
      .def("_debug_states", &PyMiniGridEnvPool::DebugStates);
}

}  // namespace minigrid
