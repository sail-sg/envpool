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

#include <tuple>

#include "envpool/core/py_envpool.h"
#include "envpool/mujoco/myosuite/myosuite_env.h"

using MyoSuiteEnvSpec = PyEnvSpec<myosuite::MyoSuiteEnvSpec>;
using MyoSuiteEnvPool = PyEnvPool<myosuite::MyoSuiteEnvPool>;
using MyoSuitePixelEnvSpec = PyEnvSpec<myosuite::MyoSuitePixelEnvSpec>;
using MyoSuitePixelEnvPool = PyEnvPool<myosuite::MyoSuitePixelEnvPool>;

namespace {

template <typename SpecT>
py::tuple ExportSpecEntry(const SpecT& spec) {
  return py::make_tuple(py::dtype::of<typename SpecT::dtype>(), spec.shape,
                        spec.bounds, spec.elementwise_bounds, spec.is_discrete);
}

template <typename DType>
py::tuple ExportSpecEntry(const Spec<Container<DType>>& spec) {
  return py::make_tuple(
      py::dtype::of<DType>(), py::make_tuple(spec.shape, spec.inner_spec.shape),
      spec.inner_spec.bounds, spec.inner_spec.elementwise_bounds,
      spec.inner_spec.is_discrete);
}

template <typename... SpecT>
py::tuple ExportSpecsDynamic(const std::tuple<SpecT...>& specs) {
  py::tuple out(sizeof...(SpecT));
  std::size_t index = 0;
  std::apply(
      [&](const auto&... spec) {
        ((out[index++] = ExportSpecEntry(spec)), ...);
      },
      specs);
  return out;
}

template <typename SPEC, typename ENVPOOL>
void RegisterMyoSuite(py::module_& m, const char* spec_name,
                      const char* envpool_name) {
  py::class_<SPEC>(m, spec_name,
                   py::metaclass(py::module_::import("abc").attr("ABCMeta")))
      .def(py::init<const typename SPEC::ConfigValues&>())
      .def_readonly("_config_values", &SPEC::py_config_values)
      .def_property_readonly(
          "_state_spec",
          [](const SPEC& self) { return ExportSpecsDynamic(self.state_spec); })
      .def_property_readonly(
          "_action_spec",
          [](const SPEC& self) { return ExportSpecsDynamic(self.action_spec); })
      .def_readonly_static("_state_keys", &SPEC::py_state_keys)
      .def_readonly_static("_action_keys", &SPEC::py_action_keys)
      .def_readonly_static("_config_keys", &SPEC::py_config_keys)
      .def_readonly_static("_default_config_values",
                           &SPEC::py_default_config_values);
  py::class_<ENVPOOL>(m, envpool_name,
                      py::metaclass(py::module_::import("abc").attr("ABCMeta")))
      .def(py::init<const SPEC&>())
      .def_readonly("_spec", &ENVPOOL::py_spec)
      .def("_recv", &ENVPOOL::PyRecv)
      .def("_send", &ENVPOOL::PySend)
      .def("_reset", &ENVPOOL::PyReset)
      .def("_render", &ENVPOOL::PyRender)
      .def_readonly_static("_state_keys", &ENVPOOL::py_state_keys)
      .def_readonly_static("_action_keys", &ENVPOOL::py_action_keys)
      .def("_xla", &ENVPOOL::Xla);
}

}  // namespace

PYBIND11_MODULE(myosuite_envpool, m) {
  RegisterMyoSuite<MyoSuiteEnvSpec, MyoSuiteEnvPool>(m, "_MyoSuiteEnvSpec",
                                                     "_MyoSuiteEnvPool");
  RegisterMyoSuite<MyoSuitePixelEnvSpec, MyoSuitePixelEnvPool>(
      m, "_MyoSuitePixelEnvSpec", "_MyoSuitePixelEnvPool");
}
