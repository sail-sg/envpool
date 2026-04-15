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

#ifndef ENVPOOL_MUJOCO_MYOSUITE_PATHS_H_
#define ENVPOOL_MUJOCO_MYOSUITE_PATHS_H_

#include <string>
#include <string_view>

namespace myosuite {

inline std::string MyoSuiteAssetRoot(const std::string& base_path) {
  return base_path + "/mujoco/myosuite_assets";
}

inline std::string MyoSuiteModelPath(const std::string& base_path,
                                     std::string_view relative_model_path) {
  return MyoSuiteAssetRoot(base_path) + "/" + std::string(relative_model_path);
}

}  // namespace myosuite

#endif  // ENVPOOL_MUJOCO_MYOSUITE_PATHS_H_
