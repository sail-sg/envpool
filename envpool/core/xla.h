/*
 * Copyright 2021 Garena Online Private Limited
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

#ifndef ENVPOOL_CORE_XLA_H_
#define ENVPOOL_CORE_XLA_H_

#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "envpool/core/array.h"

template <typename D>
constexpr bool is_container_v = false;  // NOLINT
template <typename D>
constexpr bool is_container_v<Container<D>> = true;  // NOLINT
template <typename... T>
constexpr bool HasContainerType(std::tuple<T...>) {
  return (is_container_v<typename T::dtype> || ...);
}
bool HasDynamicDim(const std::vector<int>& shape) {
  LOG(ERROR) << shape.size();
  if (shape.size() > 0) {
    LOG(ERROR) << shape[0] << shape[1];
  }
  return std::any_of(shape.begin() + 1, shape.end(),
                     [](int s) { return s == -1; });
}
template <typename... T>
bool HasDynamicDim(const std::tuple<T...>& state_spec) {
  bool dyn = false;
  std::apply([&](auto&&... spec) { dyn = (HasDynamicDim(spec.shape) || ...); },
             state_spec);
  return dyn;
}

template <typename Dtype>
Array RawBufferToArray(const void* buffer, ::Spec<Dtype> spec, int batch_size,
                       int max_num_players) {
  if (!spec.shape.empty() &&
      spec.shape[0] == -1) {  // If first dim is max_num_players
    spec.shape[0] = max_num_players * batch_size;
  } else {
    spec = spec.Batch(batch_size);
  }
  Array ret(spec);
  ret.Assign(reinterpret_cast<const Dtype*>(buffer), ret.size);
  return ret;
}

template <typename EnvPool>
void XlaSend(void* out, const void** in) {
  EnvPool* envpool = *reinterpret_cast<EnvPool**>(const_cast<void*>(in[0]));
  *reinterpret_cast<EnvPool**>(out) = envpool;
  in = in + 1;
  std::vector<Array> action;
  action.reserve(std::tuple_size_v<typename EnvPool::Action::Keys>);
  int batch_size = envpool->spec.config["batch_size"_];
  int max_num_players = envpool->spec.config["max_num_players"_];
  auto action_spec = envpool->spec.action_spec.AllValues();
  std::size_t index = 0;
  std::apply(
      [&](auto&&... spec) {
        ((action.emplace_back(
             RawBufferToArray(in[index++], spec, batch_size, max_num_players))),
         ...);
      },
      action_spec);
  envpool->Send(action);
}

template <typename EnvPool>
void XlaRecv(void* out, const void** in) {
  EnvPool* envpool = *reinterpret_cast<EnvPool**>(const_cast<void*>(in[0]));
  int batch_size = envpool->spec.config["batch_size"_];
  int max_num_players = envpool->spec.config["max_num_players"_];
  void** outs = reinterpret_cast<void**>(out);
  *reinterpret_cast<EnvPool**>(outs[0]) = envpool;
  outs = outs + 1;
  std::vector<Array> recv = envpool->Recv();
  for (std::size_t i = 0; i < recv.size(); ++i) {
    CHECK_LE(recv[i].Shape(0), batch_size * max_num_players);
    std::memcpy(outs[i], recv[i].Data(), recv[i].size * recv[i].element_size);
  }
}

#endif  // ENVPOOL_CORE_XLA_H_
