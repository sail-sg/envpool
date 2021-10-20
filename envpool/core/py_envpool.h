#ifndef ENVPOOL_CORE_PY_ENVPOOL_H_
#define ENVPOOL_CORE_PY_ENVPOOL_H_

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "envpool/core/envpool.h"

namespace py = pybind11;

// force contiguous memory
template <typename dtype>
using pyarray_t = py::array_t<dtype, py::array::c_style | py::array::forcecast>;

/**
 * Convert Array to py::array, with py::capsule
 */
template <typename dtype>
py::array ArrayToNumpy(const Array& a) {
  Array* arr = new Array(a);
  auto capsule =
      py::capsule(arr, [](void* arr) { delete reinterpret_cast<Array*>(arr); });
  return py::array(arr->Shape(), reinterpret_cast<dtype*>(arr->data()),
                   capsule);
}

template <typename dtype>
Array FromPy(const pyarray_t<dtype>& arr) {
  ShapeSpec spec(arr.itemsize(),
                 std::vector<int>(arr.shape(), arr.shape() + arr.ndim()));
  pyarray_t<dtype> arr_(arr);  // remove const
  return Array(spec, reinterpret_cast<char*>(arr_.mutable_data()));
}

template <typename dtype>
Array FromPyIncRef(const pyarray_t<dtype>& a) {
  pyarray_t<dtype>* arr_ptr = new pyarray_t<dtype>(a);
  ShapeSpec spec(
      arr_ptr->itemsize(),
      std::vector<int>(arr_ptr->shape(), arr_ptr->shape() + arr_ptr->ndim()));
  return Array(spec, reinterpret_cast<char*>(arr_ptr->mutable_data()),
               [arr_ptr](char* p) {
                 py::gil_scoped_acquire acquire;
                 delete arr_ptr;
               });
}

template <typename EnvSpec>
class PyEnvSpec : public EnvSpec {
 public:
  explicit PyEnvSpec(const typename EnvSpec::ConfigValues& conf)
      : EnvSpec(conf) {}
  typedef std::tuple<py::dtype, std::vector<int>> array_spec_t;
  // for python
  static inline const std::vector<std::string> config_keys =
      EnvSpec::Config::keys();
  static inline typename EnvSpec::ConfigValues default_config_values =
      EnvSpec::default_config.values();
  typename EnvSpec::ConfigValues config_values = EnvSpec::config.values();
  std::vector<std::tuple<std::string, array_spec_t>> _state_spec =
      PyEnvSpec::array_spec_map(EnvSpec::state_spec);
  std::vector<std::tuple<std::string, array_spec_t>> _action_spec =
      PyEnvSpec::array_spec_map(EnvSpec::action_spec);

 protected:
  /**
   * Utilities to make specs into a format convertible to python side
   */
  template <typename T>
  static array_spec_t make_array_spec(const Spec<T>& s) {
    return std::make_tuple(py::dtype::of<T>(), s.shape);
  }
  template <typename spec_t>
  static decltype(auto) array_spec_map(const spec_t& d) {
    std::vector<std::tuple<std::string, array_spec_t>> ret(d.size);
    d.apply([&](auto&&... ikv) {  // dict.apply, ikv = (index, key, value)
      ((ret[std::get<0>(ikv)] =
            std::make_tuple(std::get<1>(ikv).str(),
                            PyEnvSpec::make_array_spec(std::get<2>(ikv)))),
       ...);
    });
    return ret;
  }
};

template <typename... Types>
auto unpack_as_pyarray_t(std::tuple<Types...>)
    -> std::tuple<pyarray_t<typename Types::dtype>...>;

/**
 * Templated subclass of EnvPool,
 * to be overrided by the real EnvPool.
 */
template <typename EnvPool>
class PyEnvPool : public EnvPool {
 protected:
  template <std::size_t I>
  using StateDtypeAt = typename std::tuple_element_t<
      I, typename EnvPool::Spec::StateSpec::Values>::dtype;

  template <std::size_t I>
  using ActionDtypeAt = typename std::tuple_element_t<
      I, typename EnvPool::Spec::ActionSpec::Values>::dtype;

  typedef decltype(unpack_as_pyarray_t(
      std::declval<typename EnvPool::Spec::ActionSpec::Values>())) action_type;

  // Convert the state to numpy arrays
  template <std::size_t... I>
  void _StateToNumpy(const std::vector<Array>& arr, std::vector<py::array>* ret,
                     std::integer_sequence<std::size_t, I...>) {
    (((*ret)[I] = ArrayToNumpy<StateDtypeAt<I>>(arr[I])), ...);
  }
  void StateToNumpy(const std::vector<Array>& arr,
                    std::vector<py::array>* ret) {
    _StateToNumpy(arr, ret, std::make_index_sequence<EnvPool::State::size>{});
  }

  // Convert the numpy array to action
  template <std::size_t... I>
  void _NumpyToAction(const action_type& arr, std::vector<Array>* ret,
                      std::integer_sequence<std::size_t, I...>) {
    (((*ret)[I] = FromPy<ActionDtypeAt<I>>(std::get<I>(arr))), ...);
  }
  void NumpyToAction(const action_type& arr, std::vector<Array>* ret) {
    _NumpyToAction(arr, ret, std::make_index_sequence<EnvPool::Action::size>{});
  }

 public:
  typedef PyEnvSpec<typename EnvPool::Spec> PySpec;
  explicit PyEnvPool(const PySpec& py_spec) : EnvPool(py_spec), spec(py_spec) {}

  PySpec spec;

  static inline std::vector<std::string> state_keys =
      EnvPool::Spec::StateSpec::keys();
  static inline std::vector<std::string> action_keys =
      EnvPool::Spec::ActionSpec::keys();

  /**
   * py api
   */
  void py_send(const action_type& action) {
    std::vector<Array> arr(std::tuple_size<action_type>{});
    NumpyToAction(action, &arr);
    py::gil_scoped_release release;
    EnvPool::Send(arr);  // delegate to the c++ api
  }

  /**
   * py api
   */
  std::vector<py::array> py_recv() {
    std::vector<Array> arr;
    {
      py::gil_scoped_release release;
      arr = EnvPool::Recv();
      DCHECK_EQ(arr.size(), std::tuple_size_v<typename EnvPool::State::Keys>);
    }
    std::vector<py::array> ret(EnvPool::State::size);
    StateToNumpy(arr, &ret);
    return ret;
  }

  /**
   * py api
   */
  void py_reset(const pyarray_t<int>& env_ids) {
    // PyArray arr = PyArray::From<int>(env_ids);
    auto arr = FromPy<int>(env_ids);
    py::gil_scoped_release release;
    EnvPool::Reset(arr);
  }
};

/**
 * Call this macro in the translation unit of each envpool instance
 * It will register the envpool instance to the registry.
 * The static bool status is local to the translation unit.
 */
#define REGISTER(MODULE, SPEC, ENVPOOL)                              \
  py::module abc = py::module::import("abc");                        \
  py::object abc_meta = abc.attr("ABCMeta");                         \
  py::class_<SPEC>(MODULE, "_" #SPEC, py::metaclass(abc_meta))       \
      .def(py::init<const typename SPEC::ConfigValues&>())           \
      .def_readonly("_config_values", &SPEC::config_values)          \
      .def_readonly("_state_spec", &SPEC::_state_spec)               \
      .def_readonly("_action_spec", &SPEC::_action_spec)             \
      .def_readonly_static("_config_keys", &SPEC::config_keys)       \
      .def_readonly_static("_default_config_values",                 \
                           &SPEC::default_config_values);            \
  py::class_<ENVPOOL>(MODULE, "_" #ENVPOOL, py::metaclass(abc_meta)) \
      .def(py::init<const SPEC&>())                                  \
      .def_readonly("_spec", &ENVPOOL::spec)                         \
      .def("_recv", &ENVPOOL::py_recv)                               \
      .def("_send", &ENVPOOL::py_send)                               \
      .def("_reset", &ENVPOOL::py_reset)                             \
      .def_readonly_static("_state_keys", &ENVPOOL::state_keys)      \
      .def_readonly_static("_action_keys", &ENVPOOL::action_keys);

#endif  // ENVPOOL_CORE_PY_ENVPOOL_H_
