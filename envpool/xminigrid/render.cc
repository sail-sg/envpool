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

// Native adaptation of xminigrid 0.9.3's RGB renderer.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include "envpool/minigrid/impl/tile_render.h"
#include "envpool/xminigrid/xminigrid.h"

namespace xminigrid {
namespace {

using minigrid::kTilePixels;
using minigrid::PointInRect;
using minigrid::PointInTriangle;
using minigrid::Rgb;

constexpr std::array<Rgb, kPink + 1> kColors = {{
    {255, 255, 255},
    {255, 0, 0},
    {0, 255, 0},
    {0, 0, 255},
    {112, 39, 195},
    {255, 255, 0},
    {100, 100, 100},
    {0, 0, 0},
    {255, 140, 0},
    {255, 255, 255},
    {160, 82, 45},
    {225, 20, 147},
}};

std::vector<uint8_t> DrawTile(Tile tile, int dir, bool highlight) {
  constexpr int n = kTilePixels * minigrid::kTileSubdivs;
  std::vector<uint8_t> img(n * n * 3);
  const auto& color = kColors[tile[1]];
  auto fill = [&](const minigrid::CoordFn& shape, const Rgb& c) {
    minigrid::FillCoords(&img, n, n, shape, c);
  };
  fill(PointInRect(0, 0.031f, 0, 1), minigrid::kGridColor);
  fill(PointInRect(0, 1, 0, 0.031f), minigrid::kGridColor);
  switch (static_cast<Type>(tile[0])) {
    case kWall:
      fill(PointInRect(0, 1, 0, 1), color);
      break;
    case kFloor:
    case kGoal:
      fill(PointInRect(0.031f, 1, 0.031f, 1),
           tile[0] == kGoal ? color : minigrid::ScaleColor(color, 0.5f));
      break;
    case kDoorLocked:
    case kDoorClosed:
    case kDoorOpen:
      minigrid::DrawDoor(&img, n, n, color, tile[0] == kDoorOpen,
                         tile[0] == kDoorLocked);
      break;
    case kKey:
      minigrid::DrawKey(&img, n, n, color);
      break;
    case kBall:
      fill(minigrid::PointInCircle(0.5f, 0.5f, 0.31f), color);
      break;
    case kSquare:
      fill(PointInRect(0.25f, 0.75f, 0.25f, 0.75f), color);
      break;
    case kPyramid:
      fill(PointInTriangle({0.15f, 0.8f}, {0.85f, 0.8f}, {0.5f, 0.2f}), color);
      break;
    case kHex:
      fill(
          [](float x, float y) {
            return std::abs(y - 0.5f) <
                   std::sqrt(3.0f) *
                       std::min(0.35f - std::abs(x - 0.5f), 0.175f);
          },
          color);
      break;
    case kStar:
      fill(PointInTriangle({0.15f, 0.75f}, {0.85f, 0.75f}, {0.5f, 0.15f}),
           color);
      fill(PointInTriangle({0.15f, 0.3f}, {0.85f, 0.3f}, {0.5f, 0.9f}), color);
      break;
    case kEmpty:
      std::fill(img.begin(), img.end(), 255);
      fill(
          [](float x, float y) {
            return std::abs(x - 0.5f) > 0.469f || std::abs(y - 0.5f) > 0.469f;
          },
          kColors[kRed]);
      fill(PointInRect(0.45f, 0.55f, 0.2f, 0.65f), kColors[kRed]);
      fill(PointInRect(0.45f, 0.55f, 0.7f, 0.85f), kColors[kRed]);
      break;
  }
  if (dir >= 0) {
    minigrid::DrawAgent(&img, n, n, dir - 1);
  }
  if (highlight) {
    for (auto& v : img) {
      v += (255 - v) / 5;
    }
  }
  return minigrid::Downsample(img, n, n, minigrid::kTileSubdivs);
}

}  // namespace

std::pair<int, int> XMiniGridEnv::RenderSize(int width, int height) const {
  return {width > 0 ? width : spec_.task.width * kTilePixels,
          height > 0 ? height : spec_.task.height * kTilePixels};
}

void XMiniGridEnv::Render(int width, int height, int, unsigned char* rgb) {
  const int w = spec_.task.width;
  const int h = spec_.task.height;
  const int last = spec_.config["view_size"_] - 1;
  Pos a = ViewCell(0, 0);
  Pos b = ViewCell(last, last);
  using Key = std::tuple<Tile, int, bool>;
  static thread_local std::map<Key, std::vector<uint8_t>> cache;
  std::vector<const uint8_t*> tiles;
  tiles.reserve(grid_.size());
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const Pos pos{y, x};
      const Tile t = At(pos);
      int dir = agent_ == pos ? direction_ : -1;
      bool highlight = y >= std::min(a.y, b.y) && y <= std::max(a.y, b.y) &&
                       x >= std::min(a.x, b.x) && x <= std::max(a.x, b.x);
      auto& img = cache[{t, dir, highlight}];
      if (img.empty()) {
        img = DrawTile(t, dir, highlight);
      }
      tiles.push_back(img.data());
    }
  }
  for (int y = 0; y < height; ++y) {
    int sy = static_cast<int64_t>(y) * h * kTilePixels / height;
    const auto* row = tiles.data() + sy / kTilePixels * w;
    int offset = sy % kTilePixels * kTilePixels * 3;
    if (width == w * kTilePixels) {
      for (int x = 0; x < w; ++x) {
        std::memcpy(rgb, row[x] + offset, kTilePixels * 3);
        rgb += kTilePixels * 3;
      }
    } else {
      for (int x = 0; x < width; ++x) {
        int sx = static_cast<int64_t>(x) * w * kTilePixels / width;
        std::memcpy(rgb, row[sx / kTilePixels] + offset + sx % kTilePixels * 3,
                    3);
        rgb += 3;
      }
    }
  }
}

}  // namespace xminigrid
