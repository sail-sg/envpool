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

#ifndef ENVPOOL_MUJOCO_MJLAB_TERRAIN_H_
#define ENVPOOL_MUJOCO_MJLAB_TERRAIN_H_

#include <vector>

#include "envpool/mujoco/mjlab/simulation.h"

namespace mjlab {

class Terrain {
 public:
  explicit Terrain(Simulation* simulation);
  void Curriculum(const std::vector<float>& command);
  bool Outside(float margin) const;
  Json State() const;

 private:
  Simulation& sim_;
  int rows_{0}, columns_{0}, level_{0}, type_{0};
  std::vector<float> origins_;
  void Generate();
};

}  // namespace mjlab

#endif  // ENVPOOL_MUJOCO_MJLAB_TERRAIN_H_
