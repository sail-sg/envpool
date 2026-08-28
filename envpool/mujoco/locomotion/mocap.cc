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

#include "envpool/mujoco/locomotion/mocap.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mujoco_locomotion {
namespace {

class Reader {
 public:
  explicit Reader(const std::string& filename)
      : input_(filename, std::ios::binary) {
    if (!input_) throw std::runtime_error("Cannot open mocap data " + filename);
    char signature[8];
    Read(signature, 8);
    if (std::memcmp(signature, "EPMOCAP1", 8))
      throw std::runtime_error("Invalid mocap format");
  }
  void Read(char* bytes, std::size_t count) {
    if (!input_.read(bytes, count))
      throw std::runtime_error("Truncated mocap data");
  }
  uint32_t Integer() {
    unsigned char bytes[4];
    Read(reinterpret_cast<char*>(bytes), 4);
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i)
      result |= static_cast<uint32_t>(bytes[i]) << (8 * i);
    return result;
  }
  double Double() {
    uint64_t bits = Integer();
    bits |= static_cast<uint64_t>(Integer()) << 32;
    double value;
    std::memcpy(&value, &bits, 8);
    return value;
  }
  std::string String() {
    const auto size = Integer();
    if (size > 1024) throw std::runtime_error("Invalid mocap name length");
    std::string value(size, '\0');
    Read(value.data(), size);
    return value;
  }
  std::vector<double> Samples(std::size_t count) {
    std::vector<double> result(count);
    for (double& value : result) {
      const auto bits = Integer();
      float sample;
      std::memcpy(&sample, &bits, 4);
      value = sample;
    }
    return result;
  }

 private:
  std::ifstream input_;
};

}  // namespace

const double* MocapClip::Frame(const std::string& key, int frame) const {
  if (frame < 0 || frame >= frames)
    throw std::out_of_range("Mocap frame outside clip");
  const auto& feature = features.at(key);
  return feature.values.data() +
         static_cast<std::size_t>(frame) * feature.width;
}

std::shared_ptr<const MocapData> MocapData::Load(const std::string& filename) {
  static std::mutex mutex;
  static std::map<std::string, std::weak_ptr<const MocapData>> cache;
  std::lock_guard<std::mutex> lock(mutex);
  if (auto cached = cache[filename].lock()) return cached;
  auto data = std::make_shared<MocapData>();
  Reader reader(filename);
  const auto clips = reader.Integer();
  if (!clips || clips > 1024)
    throw std::runtime_error("Invalid mocap clip count");
  for (uint32_t i = 0; i < clips; ++i) {
    MocapClip clip;
    clip.name = reader.String();
    clip.frames = reader.Integer();
    clip.dt = reader.Double();
    const auto features = reader.Integer();
    if (clip.frames <= 0 || clip.frames > 100000 || features > 128)
      throw std::runtime_error("Invalid mocap clip shape");
    for (uint32_t j = 0; j < features; ++j) {
      auto name = reader.String();
      const auto width = reader.Integer();
      if (width > 4096) throw std::runtime_error("Invalid mocap feature width");
      clip.features.emplace(
          std::move(name),
          MocapFeature{
              static_cast<int>(width),
              reader.Samples(static_cast<std::size_t>(width) * clip.frames)});
    }
    data->clips.push_back(std::move(clip));
  }
  cache[filename] = data;
  return data;
}

}  // namespace mujoco_locomotion
