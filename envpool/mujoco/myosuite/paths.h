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

#include <array>
#include <cctype>
#include <string>
#include <string_view>

#ifdef _WIN32
#include <direct.h>
#else
#include <unistd.h>
#endif

namespace myosuite {

inline std::string MyoSuiteAssetRoot(const std::string& base_path) {
  return base_path + "/mujoco/myosuite_assets";
}

inline bool IsAbsolutePath(std::string_view path) {
  if (path.empty()) {
    return false;
  }
  if (path[0] == '/') {
    return true;
  }
#ifdef _WIN32
  if (path[0] == '\\') {
    return true;
  }
  return path.size() >= 3 &&
         std::isalpha(static_cast<unsigned char>(path[0])) && path[1] == ':' &&
         (path[2] == '/' || path[2] == '\\');
#else
  return false;
#endif
}

inline std::string CurrentWorkingDirectory() {
  std::array<char, 4096> buffer{};
#ifdef _WIN32
  if (_getcwd(buffer.data(), static_cast<int>(buffer.size())) == nullptr) {
    return std::string();
  }
#else
  if (getcwd(buffer.data(), buffer.size()) == nullptr) {
    return std::string();
  }
#endif
  return std::string(buffer.data());
}

inline std::string MyoSuiteModelPath(const std::string& base_path,
                                     std::string_view relative_model_path) {
  if (IsAbsolutePath(relative_model_path)) {
    return std::string(relative_model_path);
  }
  std::string model_path =
      MyoSuiteAssetRoot(base_path) + "/" + std::string(relative_model_path);
  if (IsAbsolutePath(model_path)) {
    return model_path;
  }
  std::string cwd = CurrentWorkingDirectory();
  if (cwd.empty()) {
    return model_path;
  }
  return cwd + "/" + model_path;
}

}  // namespace myosuite

#endif  // ENVPOOL_MUJOCO_MYOSUITE_PATHS_H_
