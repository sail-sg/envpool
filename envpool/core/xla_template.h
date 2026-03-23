/*
 * Copyright 2022 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENVPOOL_CORE_XLA_TEMPLATE_H_
#define ENVPOOL_CORE_XLA_TEMPLATE_H_

#include <cuda_runtime_api.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xla/ffi/api/ffi.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "envpool/core/spec.h"

namespace py = pybind11;
namespace xla_ffi = xla::ffi;

template <typename Spec>
static auto SpecToTuple(const Spec& spec) {
  return std::make_tuple(py::dtype::of<typename Spec::dtype>(), spec.shape);
}

template <typename Class, typename CC>
struct CustomCall {
  using InSpecs = std::invoke_result_t<decltype(CC::InSpecs), Class*>;
  using OutSpecs = std::invoke_result_t<decltype(CC::OutSpecs), Class*>;
  using In = std::array<void*, std::tuple_size_v<InSpecs>>;
  using Out = std::array<void*, std::tuple_size_v<OutSpecs>>;

  static py::bytes Handle(Class* obj) {
    return py::bytes(
        std::string(reinterpret_cast<const char*>(&obj), sizeof(Class*)));
  }

  static xla_ffi::ErrorOr<Class*> ResolveHandle(xla_ffi::Dictionary attrs) {
    auto handle = attrs.get<std::int64_t>("handle");
    if (!handle) {
      return xla_ffi::Unexpected(handle.error());
    }
    return reinterpret_cast<Class*>(
        static_cast<std::uintptr_t>(static_cast<std::int64_t>(*handle)));
  }

  static xla_ffi::Error ValidateArity(xla_ffi::RemainingArgs args,
                                      xla_ffi::RemainingRets rets) {
    constexpr std::size_t expected_args = std::tuple_size_v<In> + 1;
    constexpr std::size_t expected_rets = std::tuple_size_v<Out> + 1;
    if (args.size() != expected_args) {
      return xla_ffi::Error::InvalidArgument(
          "Expected " + std::to_string(expected_args) + " buffers, got " +
          std::to_string(args.size()));
    }
    if (rets.size() != expected_rets) {
      return xla_ffi::Error::InvalidArgument(
          "Expected " + std::to_string(expected_rets) + " results, got " +
          std::to_string(rets.size()));
    }
    return {};
  }

  static xla_ffi::Error PopulateInBuffers(xla_ffi::RemainingArgs args,
                                          In* in_arr) {
    for (std::size_t i = 0; i < in_arr->size(); ++i) {
      auto buffer = args.get<xla_ffi::AnyBuffer>(i + 1);
      if (!buffer) {
        return buffer.error();
      }
      (*in_arr)[i] = (*buffer).untyped_data();
    }
    return {};
  }

  static xla_ffi::Error PopulateOutBuffers(xla_ffi::RemainingRets rets,
                                           Out* out_arr) {
    for (std::size_t i = 0; i < out_arr->size(); ++i) {
      auto buffer = rets.get<xla_ffi::AnyBuffer>(i + 1);
      if (!buffer) {
        return buffer.error();
      }
      (*out_arr)[i] = (*buffer)->untyped_data();
    }
    return {};
  }

  static xla_ffi::Error CpuExecute(xla_ffi::RemainingArgs args,
                                   xla_ffi::RemainingRets rets,
                                   xla_ffi::Dictionary attrs) {
    if (auto err = ValidateArity(args, rets); err.failure()) {
      return err;
    }
    auto obj = ResolveHandle(attrs);
    if (!obj) {
      return obj.error();
    }
    In in_arr{};
    Out out_arr{};
    if (auto err = PopulateInBuffers(args, &in_arr); err.failure()) {
      return err;
    }
    if (auto err = PopulateOutBuffers(rets, &out_arr); err.failure()) {
      return err;
    }
    CC::Cpu(*obj, in_arr, out_arr);
    return {};
  }

  static xla_ffi::Error GpuExecute(cudaStream_t stream,
                                   xla_ffi::RemainingArgs args,
                                   xla_ffi::RemainingRets rets,
                                   xla_ffi::Dictionary attrs) {
    if (auto err = ValidateArity(args, rets); err.failure()) {
      return err;
    }
    auto obj = ResolveHandle(attrs);
    if (!obj) {
      return obj.error();
    }
    In in_arr{};
    Out out_arr{};
    if (auto err = PopulateInBuffers(args, &in_arr); err.failure()) {
      return err;
    }
    if (auto err = PopulateOutBuffers(rets, &out_arr); err.failure()) {
      return err;
    }
    CC::Gpu(*obj, stream, in_arr, out_arr);
    return {};
  }

  static auto Specs(Class* obj) {
    auto handle_spec =
        std::make_tuple(SpecToTuple(Spec<uint8_t>({sizeof(Class*)})));
    auto in_specs = CC::InSpecs(obj);
    auto in = std::apply(
        [&](auto&&... a) { return std::make_tuple(SpecToTuple(a)...); },
        in_specs);
    auto out_specs = CC::OutSpecs(obj);
    auto out = std::apply(
        [&](auto&&... a) { return std::make_tuple(SpecToTuple(a)...); },
        out_specs);
    return std::make_tuple(std::tuple_cat(handle_spec, in),
                           std::tuple_cat(handle_spec, out));
  }

  static auto Capsules() {
    XLA_FFI_DEFINE_HANDLER(cpu_handler, CpuExecute,
                           xla_ffi::Ffi::Bind()
                               .RemainingArgs()
                               .RemainingRets()
                               .Attrs<xla_ffi::Dictionary>());
    XLA_FFI_DEFINE_HANDLER(gpu_handler, GpuExecute,
                           xla_ffi::Ffi::Bind()
                               .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()
                               .RemainingArgs()
                               .RemainingRets()
                               .Attrs<xla_ffi::Dictionary>());
    // NOLINTNEXTLINE(bugprone-casting-through-void)
    auto* cpu_handler_ptr = reinterpret_cast<void*>(cpu_handler);
    // NOLINTNEXTLINE(bugprone-casting-through-void)
    auto* gpu_handler_ptr = reinterpret_cast<void*>(gpu_handler);
    return std::make_tuple(
        py::capsule(cpu_handler_ptr), py::capsule(gpu_handler_ptr));
  }

  static auto Xla(Class* obj) {
    return std::make_tuple(Handle(obj), Specs(obj), Capsules());
  }
};

#endif  // ENVPOOL_CORE_XLA_TEMPLATE_H_
