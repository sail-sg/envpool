/*
 * Copyright 2026 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENVPOOL_MUJOCO_LOCOMOTION_MOCAP_H_
#define ENVPOOL_MUJOCO_LOCOMOTION_MOCAP_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mujoco_locomotion {

struct MocapFeature {
  int width;
  std::vector<double> values;
};

struct MocapClip {
  std::string name;
  int frames;
  double dt;
  std::map<std::string, MocapFeature> features;
  const double* Frame(const std::string& key, int frame) const;
};

// Immutable clips are shared across vectorized environments, including workers
// which reset concurrently. No Python or HDF5 dependency enters the runtime.
struct MocapData {
  std::vector<MocapClip> clips;
  static std::shared_ptr<const MocapData> Load(const std::string& filename);
};

}  // namespace mujoco_locomotion

#endif  // ENVPOOL_MUJOCO_LOCOMOTION_MOCAP_H_
