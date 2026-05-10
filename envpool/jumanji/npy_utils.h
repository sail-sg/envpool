/*
 * Copyright 2026 Garena Online Private Limited
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

#ifndef ENVPOOL_JUMANJI_NPY_UTILS_H_
#define ENVPOOL_JUMANJI_NPY_UTILS_H_

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace jumanji::npy {

inline std::size_t HeaderLength(const std::vector<char>& bytes,
                                const std::string& name) {
  if (bytes.size() < 10 || bytes[0] != static_cast<char>(0x93) ||
      bytes[1] != 'N' || bytes[2] != 'U' || bytes[3] != 'M' ||
      bytes[4] != 'P' || bytes[5] != 'Y') {
    throw std::runtime_error("invalid " + name + " npy magic");
  }
  const auto major = static_cast<unsigned char>(bytes[6]);
  if (major == 1) {
    return 10 + static_cast<unsigned char>(bytes[8]) +
           (static_cast<unsigned char>(bytes[9]) << 8);
  }
  if (major == 2 || major == 3) {
    if (bytes.size() < 12) {
      throw std::runtime_error("truncated " + name + " npy header");
    }
    std::size_t header_len = 0;
    for (int i = 0; i < 4; ++i) {
      header_len |=
          static_cast<std::size_t>(static_cast<unsigned char>(bytes[8 + i]))
          << (8 * i);
    }
    return 12 + header_len;
  }
  throw std::runtime_error("unsupported " + name + " npy version");
}

}  // namespace jumanji::npy

#endif  // ENVPOOL_JUMANJI_NPY_UTILS_H_
