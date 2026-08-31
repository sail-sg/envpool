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

#include "envpool/mujoco/mjlab/motion.h"

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace mjlab {
namespace {

using Bytes = std::vector<uint8_t>;
constexpr std::size_t kMaxMotionBytes = 2ULL << 30;

uint64_t ReadInt(const Bytes& data, std::size_t at, int width,
                 bool big = false) {
  if (at > data.size() || data.size() - at < static_cast<std::size_t>(width)) {
    throw std::invalid_argument("truncated MJLab motion file");
  }
  uint64_t result = 0;
  for (int i = 0; i < width; ++i) {
    result |= static_cast<uint64_t>(data[at + i])
              << (8 * (big ? width - i - 1 : i));
  }
  return result;
}

Motion::Array ReadNpy(const Bytes& bytes) {
  if (bytes.size() < 12 || std::memcmp(bytes.data(), "\x93NUMPY", 6) != 0) {
    throw std::invalid_argument("invalid NumPy motion array");
  }
  const int version = bytes[6];
  if (version < 1 || version > 3) {
    throw std::invalid_argument("unsupported NumPy motion version");
  }
  const std::size_t prefix = version == 1 ? 10 : 12;
  const std::size_t header_size = ReadInt(bytes, 8, version == 1 ? 2 : 4);
  if (header_size > bytes.size() - prefix) {
    throw std::invalid_argument("truncated NumPy motion header");
  }
  const std::size_t start = prefix + header_size;
  const std::string header(reinterpret_cast<const char*>(bytes.data() + prefix),
                           header_size);
  std::smatch match;
  if (!std::regex_search(
          header, match,
          std::regex(
              R"(['"]descr['"]\s*:\s*['"]([<>=|])([fiu])([1248])['"])"))) {
    throw std::invalid_argument(
        "MJLab motion arrays must have a real numeric dtype");
  }
  const bool big = match[1] == ">";
  const char kind = match[2].str()[0];
  const int item_size = std::stoi(match[3]);
  if (kind == 'f' && item_size == 1) {
    throw std::invalid_argument("invalid motion float dtype");
  }
  if (!std::regex_search(
          header, match,
          std::regex(R"(['"]fortran_order['"]\s*:\s*(True|False))"))) {
    throw std::invalid_argument("missing NumPy motion order");
  }
  const bool fortran = match[1] == "True";
  if (!std::regex_search(header, match,
                         std::regex(R"(['"]shape['"]\s*:\s*\(([^)]*)\))"))) {
    throw std::invalid_argument("missing NumPy motion shape");
  }
  Motion::Array array;
  const auto dimensions = match[1].str();
  if (!std::regex_match(dimensions,
                        std::regex(R"(\s*[0-9]+\s*(,\s*[0-9]+\s*)*(,\s*)?)"))) {
    throw std::invalid_argument("invalid NumPy motion shape");
  }
  const std::regex dimension("[0-9]+");
  std::size_t count = 1;
  for (auto it = std::sregex_iterator(dimensions.begin(), dimensions.end(),
                                      dimension);
       it != std::sregex_iterator(); ++it) {
    const auto size = std::stoull(it->str());
    if (size == 0 || size > kMaxMotionBytes / sizeof(float) / count) {
      throw std::invalid_argument("empty or oversized MJLab motion array");
    }
    array.shape.push_back(size);
    count *= size;
  }
  if (array.shape.empty() || count > (bytes.size() - start) / item_size ||
      count * item_size != bytes.size() - start) {
    throw std::invalid_argument("invalid MJLab motion array size");
  }
  array.data.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    std::size_t index = i;
    if (fortran) {
      std::size_t cursor = i;
      std::size_t stride = count;
      index = 0;
      for (std::size_t d = 0; d < array.shape.size(); ++d) {
        stride /= array.shape[d];
        const auto coordinate = cursor / stride;
        cursor %= stride;
        std::size_t fstride = 1;
        for (std::size_t k = 0; k < d; ++k) {
          fstride *= array.shape[k];
        }
        index += coordinate * fstride;
      }
    }
    const auto bits = ReadInt(bytes, start + index * item_size, item_size, big);
    float value = 0;
    if (kind == 'f' && item_size == 8) {
      double number;
      std::memcpy(&number, &bits, 8);
      value = number;
    } else if (kind == 'f' && item_size == 4) {
      uint32_t word = bits;
      std::memcpy(&value, &word, 4);
    } else if (kind == 'f') {
      const int exponent = (bits >> 10) & 31;
      const int fraction = bits & 1023;
      if (exponent == 0) {
        value = std::ldexp(static_cast<float>(fraction), -24);
      } else if (exponent == 31) {
        value = std::numeric_limits<float>::infinity();
      } else {
        value = std::ldexp(static_cast<float>(1024 + fraction), exponent - 25);
      }
      if ((bits & 0x8000) != 0) {
        value = -value;
      }
    } else if (kind == 'u') {
      value = bits;
    } else {
      int64_t signed_value = bits;
      if (item_size < 8 && ((bits & (1ULL << (item_size * 8 - 1))) != 0u)) {
        signed_value -= 1ULL << (item_size * 8);
      }
      value = signed_value;
    }
    if (!std::isfinite(value)) {
      throw std::invalid_argument("MJLab motion contains non-finite values");
    }
    array.data[i] = value;
  }
  return array;
}

std::shared_ptr<Motion> ReadMotion(const std::string& path, int joints,
                                   int bodies) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::invalid_argument("cannot open MJLab motion_file: " + path);
  }
  const auto length = input.tellg();
  if (length < 22 || static_cast<uint64_t>(length) > kMaxMotionBytes) {
    throw std::invalid_argument("invalid or oversized MJLab motion NPZ");
  }
  Bytes archive(static_cast<std::size_t>(length));
  input.seekg(0);
  if (!input.read(reinterpret_cast<char*>(archive.data()), archive.size())) {
    throw std::invalid_argument("cannot read MJLab motion NPZ");
  }
  std::size_t end = archive.size() - 22;
  const std::size_t earliest =
      archive.size() > 65557 ? archive.size() - 65557 : 0;
  while (ReadInt(archive, end, 4) != 0x06054b50) {
    if (end == earliest) {
      throw std::invalid_argument("missing motion NPZ directory");
    }
    --end;
  }
  if (ReadInt(archive, end + 4, 4) != 0 ||
      end + 22 + ReadInt(archive, end + 20, 2) != archive.size()) {
    throw std::invalid_argument("invalid motion NPZ directory");
  }
  const int entries = ReadInt(archive, end + 10, 2);
  std::size_t cursor = ReadInt(archive, end + 16, 4);
  std::size_t total = 0;
  auto result = std::make_shared<Motion>();
  const std::map<std::string, std::size_t> fields{
      {"joint_pos", 0},   {"joint_vel", 0},      {"body_pos_w", 3},
      {"body_quat_w", 4}, {"body_lin_vel_w", 3}, {"body_ang_vel_w", 3}};
  for (int entry = 0; entry < entries; ++entry) {
    if (ReadInt(archive, cursor, 4) != 0x02014b50) {
      throw std::invalid_argument("invalid motion NPZ member");
    }
    const auto flags = ReadInt(archive, cursor + 8, 2);
    const auto method = ReadInt(archive, cursor + 10, 2);
    const auto crc = ReadInt(archive, cursor + 16, 4);
    const auto compressed = ReadInt(archive, cursor + 20, 4);
    const auto size = ReadInt(archive, cursor + 24, 4);
    const auto name_size = ReadInt(archive, cursor + 28, 2);
    const auto extra_size = ReadInt(archive, cursor + 30, 2);
    const auto comment_size = ReadInt(archive, cursor + 32, 2);
    const std::size_t local = ReadInt(archive, cursor + 42, 4);
    if (cursor + 46 + name_size + extra_size + comment_size > archive.size()) {
      throw std::invalid_argument("truncated motion NPZ member");
    }
    std::string name(
        reinterpret_cast<const char*>(archive.data() + cursor + 46), name_size);
    cursor += 46 + name_size + extra_size + comment_size;
    if (name.size() < 4 || name.substr(name.size() - 4) != ".npy") {
      continue;
    }
    name.resize(name.size() - 4);
    if (fields.count(name) == 0) {
      continue;
    }
    if ((result->arrays.count(name) != 0u) || ((flags & 1) != 0u) ||
        (method != 0 && method != 8) || size > kMaxMotionBytes - total ||
        ReadInt(archive, local, 4) != 0x04034b50) {
      throw std::invalid_argument("unsupported or oversized motion NPZ member");
    }
    total += size;
    const std::size_t data = local + 30 + ReadInt(archive, local + 26, 2) +
                             ReadInt(archive, local + 28, 2);
    if (data > archive.size() || compressed > archive.size() - data) {
      throw std::invalid_argument("truncated compressed motion array");
    }
    Bytes bytes(size);
    if (method == 0) {
      if (compressed != size) {
        throw std::invalid_argument("invalid uncompressed motion array");
      }
      std::copy_n(archive.data() + data, size, bytes.data());
    } else {
      z_stream stream{};
      stream.next_in = archive.data() + data;
      stream.avail_in = compressed;
      stream.next_out = bytes.data();
      stream.avail_out = size;
      if (inflateInit2(&stream, -MAX_WBITS) != Z_OK) {
        throw std::runtime_error("cannot initialize motion decompressor");
      }
      const int status = inflate(&stream, Z_FINISH);
      const bool valid = status == Z_STREAM_END && stream.total_out == size &&
                         stream.total_in == compressed;
      inflateEnd(&stream);
      if (!valid) {
        throw std::invalid_argument("invalid deflated motion array");
      }
    }
    if (crc32(0, bytes.data(), bytes.size()) != crc) {
      throw std::invalid_argument("motion NPZ checksum mismatch");
    }
    result->arrays.emplace(name, ReadNpy(bytes));
  }
  if (result->arrays.size() != fields.size()) {
    throw std::invalid_argument(
        "motion NPZ must contain all six MJLab motion arrays");
  }
  const auto frames = result->arrays.at("joint_pos").shape[0];
  for (const auto& field : fields) {
    const auto& shape = result->arrays.at(field.first).shape;
    const std::vector<std::size_t> expected =
        field.second == 0
            ? std::vector<std::size_t>{frames, static_cast<std::size_t>(joints)}
            : std::vector<std::size_t>{frames, static_cast<std::size_t>(bodies),
                                       field.second};
    if (shape != expected) {
      throw std::invalid_argument("wrong MJLab motion shape for " +
                                  field.first);
    }
  }
  result->frames = frames;
  result->joints = joints;
  result->bodies = bodies;
  return result;
}

}  // namespace

std::shared_ptr<const Motion> LoadMotion(const std::string& path, int joints,
                                         int bodies) {
  if (path.empty()) {
    throw std::invalid_argument(
        "MJLab tracking requires motion_file; upstream supplies no default "
        "motion");
  }
  std::error_code error;
  const auto file = std::filesystem::canonical(path, error);
  if (error || !std::filesystem::is_regular_file(file)) {
    throw std::invalid_argument("cannot open MJLab motion_file: " + path);
  }
  const auto key =
      file.string() + ":" + std::to_string(std::filesystem::file_size(file)) +
      ":" +
      std::to_string(static_cast<int64_t>(
          std::filesystem::last_write_time(file).time_since_epoch().count())) +
      ":" + std::to_string(joints) + ":" + std::to_string(bodies);
  static std::mutex mutex;
  static std::map<std::string, std::weak_ptr<const Motion>> cache;
  std::scoped_lock lock(mutex);
  if (auto found = cache[key].lock()) {
    return found;
  }
  auto motion = ReadMotion(path, joints, bodies);
  cache[key] = motion;
  return motion;
}

}  // namespace mjlab
