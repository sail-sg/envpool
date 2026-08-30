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

#ifndef ENVPOOL_MUJOCO_MJLAB_MOTION_H_
#define ENVPOOL_MUJOCO_MJLAB_MOTION_H_

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mjlab {

struct Motion {
  struct Array {
    std::vector<std::size_t> shape;
    std::vector<float> data;
    const float* Frame(int frame) const {
      return data.data() + frame * (data.size() / shape[0]);
    }
  };
  std::map<std::string, Array> arrays;
  int frames, joints, bodies;
};

std::shared_ptr<const Motion> LoadMotion(const std::string& path, int joints,
                                         int bodies);

}  // namespace mjlab

#endif  // ENVPOOL_MUJOCO_MJLAB_MOTION_H_
