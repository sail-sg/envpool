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

#ifndef ENVPOOL_JUMANJI_PARSE_UTILS_H_
#define ENVPOOL_JUMANJI_PARSE_UTILS_H_

#include <array>
#include <cstddef>
#include <sstream>
#include <string>
#include <type_traits>

namespace jumanji::parse {

template <typename T, std::size_t N>
std::array<T, N> CsvArray(const std::string& text, T fill_value = T{}) {
  std::array<T, N> values{};
  values.fill(fill_value);
  if (text.empty()) {
    return values;
  }
  std::stringstream stream(text);
  std::string token;
  std::size_t index = 0;
  while (std::getline(stream, token, ',') && index < N) {
    if constexpr (std::is_same_v<T, bool>) {
      values[index++] = token == "1" || token == "True" || token == "true";
    } else if constexpr (std::is_floating_point_v<T>) {
      values[index++] = static_cast<T>(std::stof(token));
    } else {
      values[index++] = static_cast<T>(std::stoll(token));
    }
  }
  return values;
}

}  // namespace jumanji::parse

#endif  // ENVPOOL_JUMANJI_PARSE_UTILS_H_
