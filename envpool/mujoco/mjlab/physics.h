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

#ifndef ENVPOOL_MUJOCO_MJLAB_PHYSICS_H_
#define ENVPOOL_MUJOCO_MJLAB_PHYSICS_H_

#include <mujoco.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "boost/json.hpp"

namespace mjlab {

using Json = boost::json::value;

Json ReadJson(const std::string& path);
double Number(const Json& value);
std::string String(const Json& value);
std::vector<int> Indices(const Json& value);
std::vector<float> Floats(const Json& value);

// The official CPU simulator is MuJoCo-Warp, not mj_step. Its pinned kernels
// are compiled into the extension ahead of time. A serialized operation graph
// supplies model constants and the dispatch schedule; it contains no Python,
// machine-code modules, executable downloads, or runtime compiler dependency.
class Physics {
 public:
  explicit Physics(const std::string& asset_path);
  ~Physics();
  Physics(const Physics&) = delete;
  Physics& operator=(const Physics&) = delete;

  const Json& Metadata() const;
  mjModel* Model() const;
  mjData* RenderData();
  bool Has(const std::string& name) const;
  std::size_t Bytes(const std::string& name) const;
  void* Pointer(const std::string& name) const;
  template <typename T = float>
  T* Get(const std::string& name) const {
    return static_cast<T*>(Pointer(name));
  }
  template <typename T = float>
  std::size_t Count(const std::string& name) const {
    return Bytes(name) / sizeof(T);
  }
  void Set(const std::string& name, const void* value, std::size_t bytes);
  void Run(const std::string& operation);
  void Sense();
  // Randomized hfields require new ray meshes, not just modified physics data.
  void RebuildHeightfields(const std::vector<int>& ids);
  void UpdateTexture(int id, const std::vector<uint8_t>& pixels);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mjlab

#endif  // ENVPOOL_MUJOCO_MJLAB_PHYSICS_H_
