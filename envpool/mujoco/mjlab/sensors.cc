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

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "envpool/mujoco/mjlab/simulation.h"

namespace mjlab {

void RaySensor::GenerateGrid(const Json& pattern) {
  // Torch 2.9's arange uses float NEON lanes, double x64 lanes, and a
  // double scalar tail. Exporting one host's rounded grid into shared assets
  // changes height scans on another host, even with identical robot poses.
  const auto step = boost::json::value_to<double>(pattern.at("resolution"));
  const int width = std::min(ReductionWidth(), 8);  // arange disables AVX512.
  const auto axis = [&](double size) {
    const double start = -size / 2;
    const double end = size / 2 + step * 0.5;
    const int count = static_cast<int>(std::ceil((end - start) / step));
    const int vector_end = count / (2 * width) * (2 * width);
    const auto advance = [&](double base, int index) {
#ifdef _WIN32
      return base + step * index;
#else
      return std::fma(step, static_cast<double>(index), base);
#endif
    };
    std::vector<float> values(count);
    for (int i = 0; i < count; ++i) {
      if (i >= vector_end) {
        values[i] = static_cast<float>(advance(start, i));
        continue;
      }
      const auto base = static_cast<float>(advance(start, i / width * width));
#if defined(__aarch64__) || defined(_M_ARM64)
      values[i] = std::fma(static_cast<float>(step),
                           static_cast<float>(i % width), base);
#else
      values[i] = static_cast<float>(advance(base, i % width));
#endif
    }
    return values;
  };
  const auto x = axis(boost::json::value_to<double>(pattern.at("size").at(0)));
  const auto y = axis(boost::json::value_to<double>(pattern.at("size").at(1)));
  if (x.size() * y.size() * 3 != offsets.size()) {
    throw std::invalid_argument("MJLab grid size disagrees with ray buffers");
  }
  std::size_t index = 0;
  for (float yy : y) {
    for (float xx : x) {
      offsets[index++] = xx;
      offsets[index++] = yy;
      offsets[index++] = 0;
    }
  }
}

void Simulation::Sense() {
  std::map<std::string, std::vector<Vec3>> frame_positions;
  for (const auto& entry : rays) {
    const auto& sensor = entry.second;
    const std::string prefix = "ray." + entry.first;
    auto* points = physics.Get(prefix + "._ray_pnt");
    auto* directions = physics.Get(prefix + "._ray_vec");
    const std::size_t nray = sensor.offsets.size() / 3;
    for (std::size_t f = 0; f < sensor.frames.size(); ++f) {
      const auto& frame = sensor.frames[f];
      const auto field =
          frame.type == "body" ? "data.x" : "data." + frame.type + "_x";
      const Vec3 pos = Read<3>(physics.Get(field + "pos") + frame.id * 3);
      Mat3 rotation = Read<9>(physics.Get(field + "mat") + frame.id * 9);
      frame_positions[entry.first].push_back(pos);
      if (sensor.alignment == "world") {
        rotation = {1, 0, 0, 0, 1, 0, 0, 0, 1};
      } else if (sensor.alignment == "yaw") {
        Vec3 x{rotation[0], rotation[3], 0};
        float length = Norm(x);
        if (length < 0.1F) {
          Vec3 y{rotation[1], rotation[4], 0};
          const float divisor = std::max(Norm(y), 1.0e-6F);
          for (auto& v : y) {
            v /= divisor;
          }
          x = {y[1], -y[0], 0};
          length = 1;
        }
        for (auto& v : x) {
          v /= std::max(length, 1.0e-6F);
        }
        rotation = {x[0], -x[1], 0, x[1], x[0], 0, 0, 0, 1};
      } else if (sensor.alignment != "base") {
        throw std::invalid_argument("unknown MJLab ray alignment");
      }
      for (std::size_t n = 0; n < nray; ++n) {
        for (int i = 0; i < 3; ++i) {
          float offset = 0;
          float direction = 0;
          for (int j = 0; j < 3; ++j) {
            // The pinned Windows einsum does not contract this dot product;
            // the Linux/macOS BLAS path does. Translation is separate in both.
#ifdef _WIN32
            offset += rotation[i * 3 + j] * sensor.offsets[n * 3 + j];
            direction += rotation[i * 3 + j] * sensor.directions[n * 3 + j];
#else
            offset = std::fma(rotation[i * 3 + j], sensor.offsets[n * 3 + j],
                              offset);
            direction = std::fma(rotation[i * 3 + j],
                                 sensor.directions[n * 3 + j], direction);
#endif
          }
          points[(f * nray + n) * 3 + i] = pos[i] + offset;
          directions[(f * nray + n) * 3 + i] = direction;
        }
      }
    }
  }
  physics.Sense();
  for (auto& entry : rays) {
    auto& sensor = entry.second;
    const std::string prefix = "ray." + entry.first;
    auto* distances = physics.Get(prefix + "._ray_dist");
    auto* normals = physics.Get(prefix + "._ray_normal");
    const auto* points = physics.Get(prefix + "._ray_pnt");
    const auto* directions = physics.Get(prefix + "._ray_vec");
    const std::size_t nray = sensor.offsets.size() / 3;
    sensor.heights.clear();
    sensor.hits.clear();
    for (std::size_t f = 0; f < sensor.frames.size(); ++f) {
      const float z = frame_positions.at(entry.first)[f][2];
      bool all_miss = true;
      for (std::size_t n = 0; n < nray; ++n) {
        auto& distance = distances[f * nray + n];
        if (distance > sensor.max_distance) {
          distance = -1;
        }
        if (distance >= 0) {
          all_miss = false;
        }
      }
      float minimum = sensor.max_distance;
      for (std::size_t n = 0; n < nray; ++n) {
        const std::size_t id = f * nray + n;
        const float distance = distances[id];
        const bool hit = distance >= 0;
        float value = z - (points[id * 3 + 2] +
                           directions[id * 3 + 2] * std::max(distance, 0.0F));
        if (sensor.terrain_height && hit && normals[id * 3 + 2] < 0) {
          value = 0;
        }
        if (!hit) {
          value = sensor.terrain_height && all_miss
                      ? std::clamp(z, 0.0F, sensor.max_distance)
                      : sensor.max_distance;
          std::fill_n(normals + id * 3, 3, 0);
        }
        if (sensor.terrain_height) {
          minimum = std::min(minimum, value);
        } else {
          sensor.heights.push_back(value);
          sensor.hits.push_back(hit);
        }
      }
      if (sensor.terrain_height) {
        sensor.heights.push_back(minimum);
        sensor.hits.push_back(true);
      }
    }
  }
}

}  // namespace mjlab
