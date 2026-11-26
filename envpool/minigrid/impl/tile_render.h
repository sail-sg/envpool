// Copyright 2026 Garena Online Private Limited
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ENVPOOL_MINIGRID_IMPL_TILE_RENDER_H_
#define ENVPOOL_MINIGRID_IMPL_TILE_RENDER_H_

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#include "absl/log/check.h"

namespace minigrid {

using CoordFn = std::function<bool(float, float)>;
using Rgb = std::array<uint8_t, 3>;

inline constexpr int kTilePixels = 32;
inline constexpr int kTileSubdivs = 3;
inline constexpr Rgb kGridColor = {100, 100, 100};
inline constexpr Rgb kAgentColor = {255, 0, 0};

inline Rgb ScaleColor(const Rgb& color, float scale) {
  return {
      static_cast<uint8_t>(std::floor(color[0] * scale)),
      static_cast<uint8_t>(std::floor(color[1] * scale)),
      static_cast<uint8_t>(std::floor(color[2] * scale)),
  };
}

inline CoordFn PointInRect(float xmin, float xmax, float ymin, float ymax) {
  return [=](float x, float y) {
    return x >= xmin && x <= xmax && y >= ymin && y <= ymax;
  };
}

inline CoordFn PointInCircle(float cx, float cy, float r) {
  return [=](float x, float y) {
    const float dx = x - cx;
    const float dy = y - cy;
    return dx * dx + dy * dy <= r * r;
  };
}

inline CoordFn PointInTriangle(const std::array<float, 2>& a,
                               const std::array<float, 2>& b,
                               const std::array<float, 2>& c) {
  return [=](float x, float y) {
    const std::array<float, 2> v0 = {c[0] - a[0], c[1] - a[1]};
    const std::array<float, 2> v1 = {b[0] - a[0], b[1] - a[1]};
    const std::array<float, 2> v2 = {x - a[0], y - a[1]};
    const float dot00 = v0[0] * v0[0] + v0[1] * v0[1];
    const float dot01 = v0[0] * v1[0] + v0[1] * v1[1];
    const float dot02 = v0[0] * v2[0] + v0[1] * v2[1];
    const float dot11 = v1[0] * v1[0] + v1[1] * v1[1];
    const float dot12 = v1[0] * v2[0] + v1[1] * v2[1];
    const float inv_denom = 1.0f / (dot00 * dot11 - dot01 * dot01);
    const float u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    const float v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
    return u >= 0.0f && v >= 0.0f && u + v < 1.0f;
  };
}

inline void FillCoords(std::vector<uint8_t>* img, int width, int height,
                       const CoordFn& fn, const Rgb& color) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const auto yf = static_cast<float>(y + 0.5f) / height;
      const auto xf = static_cast<float>(x + 0.5f) / width;
      if (!fn(xf, yf)) {
        continue;
      }
      const int offset = (y * width + x) * 3;
      (*img)[offset + 0] = color[0];
      (*img)[offset + 1] = color[1];
      (*img)[offset + 2] = color[2];
    }
  }
}

inline std::vector<uint8_t> Downsample(const std::vector<uint8_t>& src,
                                       int width, int height, int factor) {
  CHECK_EQ(width % factor, 0);
  CHECK_EQ(height % factor, 0);
  const int out_width = width / factor;
  const int out_height = height / factor;
  std::vector<uint8_t> out(out_width * out_height * 3, 0);
  for (int y = 0; y < out_height; ++y) {
    for (int x = 0; x < out_width; ++x) {
      for (int channel = 0; channel < 3; ++channel) {
        int sum = 0;
        for (int dy = 0; dy < factor; ++dy) {
          for (int dx = 0; dx < factor; ++dx) {
            const int src_x = x * factor + dx;
            const int src_y = y * factor + dy;
            sum += src[(src_y * width + src_x) * 3 + channel];
          }
        }
        out[(y * out_width + x) * 3 + channel] =
            static_cast<uint8_t>(sum / (factor * factor));
      }
    }
  }
  return out;
}

inline void DrawDoor(std::vector<uint8_t>* img, int width, int height,
                     const Rgb& color, bool open, bool locked) {
  if (open) {
    FillCoords(img, width, height, PointInRect(0.88f, 1.0f, 0.0f, 1.0f), color);
    FillCoords(img, width, height, PointInRect(0.92f, 0.96f, 0.04f, 0.96f),
               Rgb{0, 0, 0});
    return;
  }
  FillCoords(img, width, height, PointInRect(0.0f, 1.0f, 0.0f, 1.0f), color);
  if (locked) {
    FillCoords(img, width, height, PointInRect(0.06f, 0.94f, 0.06f, 0.94f),
               ScaleColor(color, 0.45f));
    FillCoords(img, width, height, PointInRect(0.52f, 0.75f, 0.50f, 0.56f),
               color);
    return;
  }
  FillCoords(img, width, height, PointInRect(0.04f, 0.96f, 0.04f, 0.96f),
             Rgb{0, 0, 0});
  FillCoords(img, width, height, PointInRect(0.08f, 0.92f, 0.08f, 0.92f),
             color);
  FillCoords(img, width, height, PointInRect(0.12f, 0.88f, 0.12f, 0.88f),
             Rgb{0, 0, 0});
  FillCoords(img, width, height, PointInCircle(0.75f, 0.50f, 0.08f), color);
}

inline void DrawKey(std::vector<uint8_t>* img, int width, int height,
                    const Rgb& color) {
  FillCoords(img, width, height, PointInRect(0.50f, 0.63f, 0.31f, 0.88f),
             color);
  FillCoords(img, width, height, PointInRect(0.38f, 0.50f, 0.59f, 0.66f),
             color);
  FillCoords(img, width, height, PointInRect(0.38f, 0.50f, 0.81f, 0.88f),
             color);
  FillCoords(img, width, height, PointInCircle(0.56f, 0.28f, 0.19f), color);
  FillCoords(img, width, height, PointInCircle(0.56f, 0.28f, 0.064f),
             Rgb{0, 0, 0});
}

inline void DrawAgent(std::vector<uint8_t>* img, int width, int height,
                      int agent_dir) {
  const auto tri =
      PointInTriangle({0.12f, 0.19f}, {0.87f, 0.50f}, {0.12f, 0.81f});
  const float angle = -0.5f * std::acos(-1.0f) * agent_dir;
  const float c = std::cos(angle);
  const float s = std::sin(angle);
  auto rotated = [&](float x, float y) {
    return tri(0.5f + (x - 0.5f) * c - (y - 0.5f) * s,
               0.5f + (y - 0.5f) * c + (x - 0.5f) * s);
  };
  FillCoords(img, width, height, rotated, kAgentColor);
}

}  // namespace minigrid

#endif  // ENVPOOL_MINIGRID_IMPL_TILE_RENDER_H_
