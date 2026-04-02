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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include "envpool/minigrid/impl/minigrid_env.h"
#include "opencv2/opencv.hpp"

namespace minigrid {
namespace {

inline int GridOffset(int x, int y, int height) { return (x * height + y) * 3; }

using CoordFn = std::function<bool(float, float)>;
using Rgb = std::array<uint8_t, 3>;

constexpr int kTilePixels = 32;
constexpr int kTileSubdivs = 3;
constexpr Rgb kGridColor = {100, 100, 100};
constexpr Rgb kAgentColor = {255, 0, 0};
constexpr Rgb kLavaColor = {255, 128, 0};
constexpr std::array<Rgb, 6> MakeObjectColors() {
  return {
      Rgb{255, 0, 0},    Rgb{0, 255, 0},   Rgb{0, 0, 255},
      Rgb{112, 39, 195}, Rgb{255, 255, 0}, Rgb{100, 100, 100},
  };
}

constexpr std::array<Rgb, 6> kObjectColors = MakeObjectColors();

Rgb ColorValue(Color color) { return kObjectColors[static_cast<int>(color)]; }

Rgb ScaleColor(const Rgb& color, float scale) {
  return {
      static_cast<uint8_t>(std::floor(color[0] * scale)),
      static_cast<uint8_t>(std::floor(color[1] * scale)),
      static_cast<uint8_t>(std::floor(color[2] * scale)),
  };
}

CoordFn PointInRect(float xmin, float xmax, float ymin, float ymax) {
  return [=](float x, float y) {
    return x >= xmin && x <= xmax && y >= ymin && y <= ymax;
  };
}

CoordFn PointInCircle(float cx, float cy, float r) {
  return [=](float x, float y) {
    const float dx = x - cx;
    const float dy = y - cy;
    return dx * dx + dy * dy <= r * r;
  };
}

CoordFn PointInTriangle(const std::array<float, 2>& a,
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

CoordFn RotateFn(const CoordFn& fn, float cx, float cy, float theta) {
  return [=](float x, float y) {
    const float centered_x = x - cx;
    const float centered_y = y - cy;
    const float x2 =
        cx + centered_x * std::cos(-theta) - centered_y * std::sin(-theta);
    const float y2 =
        cy + centered_y * std::cos(-theta) + centered_x * std::sin(-theta);
    return fn(x2, y2);
  };
}

CoordFn PointInLine(float x0, float y0, float x1, float y1, float r) {
  const std::array<float, 2> p0 = {x0, y0};
  const std::array<float, 2> p1 = {x1, y1};
  const std::array<float, 2> dir = {p1[0] - p0[0], p1[1] - p0[1]};
  const float dist = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1]);
  const std::array<float, 2> unit = {dir[0] / dist, dir[1] / dist};
  const float xmin = std::min(x0, x1) - r;
  const float xmax = std::max(x0, x1) + r;
  const float ymin = std::min(y0, y1) - r;
  const float ymax = std::max(y0, y1) + r;
  return [=](float x, float y) {
    if (x < xmin || x > xmax || y < ymin || y > ymax) {
      return false;
    }
    const std::array<float, 2> pq = {x - p0[0], y - p0[1]};
    const float a = std::clamp(pq[0] * unit[0] + pq[1] * unit[1], 0.0f, dist);
    const std::array<float, 2> proj = {p0[0] + a * unit[0],
                                       p0[1] + a * unit[1]};
    const float dx = x - proj[0];
    const float dy = y - proj[1];
    return std::sqrt(dx * dx + dy * dy) <= r;
  };
}

void FillCoords(std::vector<uint8_t>* img, int width, int height,
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

std::vector<uint8_t> Downsample(const std::vector<uint8_t>& src, int width,
                                int height, int factor) {
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

void RenderWorldObj(const WorldObj& obj, std::vector<uint8_t>* img, int width,
                    int height) {
  const Rgb color = ColorValue(obj.GetColor());
  switch (obj.GetType()) {
    case kGoal:
      FillCoords(img, width, height, PointInRect(0.0f, 1.0f, 0.0f, 1.0f),
                 color);
      return;
    case kFloor:
      FillCoords(img, width, height, PointInRect(0.031f, 1.0f, 0.031f, 1.0f),
                 ScaleColor(color, 0.5f));
      return;
    case kLava:
      FillCoords(img, width, height, PointInRect(0.0f, 1.0f, 0.0f, 1.0f),
                 kLavaColor);
      for (int i = 0; i < 3; ++i) {
        const float ylo = 0.3f + 0.2f * i;
        const float yhi = 0.4f + 0.2f * i;
        FillCoords(img, width, height, PointInLine(0.1f, ylo, 0.3f, yhi, 0.03f),
                   Rgb{0, 0, 0});
        FillCoords(img, width, height, PointInLine(0.3f, yhi, 0.5f, ylo, 0.03f),
                   Rgb{0, 0, 0});
        FillCoords(img, width, height, PointInLine(0.5f, ylo, 0.7f, yhi, 0.03f),
                   Rgb{0, 0, 0});
        FillCoords(img, width, height, PointInLine(0.7f, yhi, 0.9f, ylo, 0.03f),
                   Rgb{0, 0, 0});
      }
      return;
    case kWall:
      FillCoords(img, width, height, PointInRect(0.0f, 1.0f, 0.0f, 1.0f),
                 color);
      return;
    case kDoor:
      if (obj.GetDoorOpen()) {
        FillCoords(img, width, height, PointInRect(0.88f, 1.0f, 0.0f, 1.0f),
                   color);
        FillCoords(img, width, height, PointInRect(0.92f, 0.96f, 0.04f, 0.96f),
                   Rgb{0, 0, 0});
        return;
      }
      FillCoords(img, width, height, PointInRect(0.0f, 1.0f, 0.0f, 1.0f),
                 color);
      if (obj.GetDoorLocked()) {
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
      return;
    case kKey:
      FillCoords(img, width, height, PointInRect(0.50f, 0.63f, 0.31f, 0.88f),
                 color);
      FillCoords(img, width, height, PointInRect(0.38f, 0.50f, 0.59f, 0.66f),
                 color);
      FillCoords(img, width, height, PointInRect(0.38f, 0.50f, 0.81f, 0.88f),
                 color);
      FillCoords(img, width, height, PointInCircle(0.56f, 0.28f, 0.19f), color);
      FillCoords(img, width, height, PointInCircle(0.56f, 0.28f, 0.064f),
                 Rgb{0, 0, 0});
      return;
    case kBall:
      FillCoords(img, width, height, PointInCircle(0.5f, 0.5f, 0.31f), color);
      return;
    case kBox:
      FillCoords(img, width, height, PointInRect(0.12f, 0.88f, 0.12f, 0.88f),
                 color);
      FillCoords(img, width, height, PointInRect(0.18f, 0.82f, 0.18f, 0.82f),
                 Rgb{0, 0, 0});
      FillCoords(img, width, height, PointInRect(0.16f, 0.84f, 0.47f, 0.53f),
                 color);
      return;
    default:
      return;
  }
}

std::vector<uint8_t> RenderTile(const WorldObj* obj, int agent_dir) {
  const int hi_width = kTilePixels * kTileSubdivs;
  const int hi_height = kTilePixels * kTileSubdivs;
  std::vector<uint8_t> img(hi_width * hi_height * 3, 0);
  FillCoords(&img, hi_width, hi_height, PointInRect(0.0f, 0.031f, 0.0f, 1.0f),
             kGridColor);
  FillCoords(&img, hi_width, hi_height, PointInRect(0.0f, 1.0f, 0.0f, 0.031f),
             kGridColor);
  if (obj != nullptr) {
    RenderWorldObj(*obj, &img, hi_width, hi_height);
  }
  if (agent_dir >= 0) {
    CoordFn tri =
        PointInTriangle({0.12f, 0.19f}, {0.87f, 0.50f}, {0.12f, 0.81f});
    tri =
        RotateFn(tri, 0.5f, 0.5f, 0.5f * static_cast<float>(M_PI) * agent_dir);
    FillCoords(&img, hi_width, hi_height, tri, kAgentColor);
  }
  return Downsample(img, hi_width, hi_height, kTileSubdivs);
}

}  // namespace

void MiniGridTask::GenImage(const Array& obs) const {
  obs.Zero();
  int top_x = 0;
  int top_y = 0;
  if (agent_dir_ == 0) {
    top_x = agent_pos_.first;
    top_y = agent_pos_.second - agent_view_size_ / 2;
  } else if (agent_dir_ == 1) {
    top_x = agent_pos_.first - agent_view_size_ / 2;
    top_y = agent_pos_.second;
  } else if (agent_dir_ == 2) {
    top_x = agent_pos_.first - agent_view_size_ + 1;
    top_y = agent_pos_.second - agent_view_size_ / 2;
  } else {
    top_x = agent_pos_.first - agent_view_size_ / 2;
    top_y = agent_pos_.second - agent_view_size_ + 1;
  }

  std::vector<std::vector<WorldObj>> view(
      agent_view_size_,
      std::vector<WorldObj>(agent_view_size_, WorldObj(kWall)));
  for (int y = 0; y < agent_view_size_; ++y) {
    for (int x = 0; x < agent_view_size_; ++x) {
      int gx = top_x + x;
      int gy = top_y + y;
      if (InBounds(gx, gy)) {
        view[y][x] = GetCell(gx, gy);
      }
    }
  }

  for (int i = 0; i < agent_dir_ + 1; ++i) {
    std::vector<std::vector<WorldObj>> rotated = view;
    for (int y = 0; y < agent_view_size_; ++y) {
      for (int x = 0; x < agent_view_size_; ++x) {
        rotated[agent_view_size_ - 1 - x][y] = view[y][x];
      }
    }
    view = std::move(rotated);
  }

  std::vector<std::vector<bool>> vis_mask(
      agent_view_size_, std::vector<bool>(agent_view_size_, false));
  int agent_x = agent_view_size_ / 2;
  int agent_y = agent_view_size_ - 1;
  if (see_through_walls_) {
    for (auto& col : vis_mask) {
      std::fill(col.begin(), col.end(), true);
    }
  } else {
    vis_mask[agent_x][agent_y] = true;
    for (int y = agent_view_size_ - 1; y >= 0; --y) {
      for (int x = 0; x < agent_view_size_ - 1; ++x) {
        if (!vis_mask[x][y]) {
          continue;
        }
        if (!view[y][x].CanSeeBehind()) {
          continue;
        }
        vis_mask[x + 1][y] = true;
        if (y > 0) {
          vis_mask[x + 1][y - 1] = true;
          vis_mask[x][y - 1] = true;
        }
      }
      for (int x = agent_view_size_ - 1; x >= 1; --x) {
        if (!vis_mask[x][y]) {
          continue;
        }
        if (!view[y][x].CanSeeBehind()) {
          continue;
        }
        vis_mask[x - 1][y] = true;
        if (y > 0) {
          vis_mask[x - 1][y - 1] = true;
          vis_mask[x][y - 1] = true;
        }
      }
    }
  }

  view[agent_y][agent_x] =
      carrying_.GetType() == kEmpty ? WorldObj(kEmpty) : carrying_;

  for (int y = 0; y < agent_view_size_; ++y) {
    for (int x = 0; x < agent_view_size_; ++x) {
      if (!vis_mask[x][y]) {
        continue;
      }
      const auto enc = view[y][x].Encode();
      obs(x, y, 0) = enc[0];
      obs(x, y, 1) = enc[1];
      obs(x, y, 2) = enc[2];
    }
  }
}

void MiniGridTask::WriteMission(const Array& obs) const {
  obs.Zero();
  auto* data = reinterpret_cast<uint8_t*>(obs.Data());
  int n = std::min(static_cast<int>(mission_.size()), mission_bytes_ - 1);
  std::memcpy(data, mission_.data(), n);
}

MiniGridDebugState MiniGridTask::DebugState() const {
  MiniGridDebugState state;
  state.env_name = env_name_;
  state.mission = mission_;
  state.mission_id = mission_id_;
  state.width = width_;
  state.height = height_;
  state.action_max = action_max_;
  state.max_steps = max_steps_;
  state.grid.resize(width_ * height_ * 3);
  state.grid_contains.resize(width_ * height_ * 3, 0);
  for (int x = 0; x < width_; ++x) {
    for (int y = 0; y < height_; ++y) {
      const WorldObj& cell = GetCell(x, y);
      const auto enc = cell.Encode();
      int offset = GridOffset(x, y, height_);
      state.grid[offset + 0] = enc[0];
      state.grid[offset + 1] = enc[1];
      state.grid[offset + 2] = enc[2];
      if (const WorldObj* contains = cell.GetContains(); contains != nullptr) {
        const auto contains_enc = contains->Encode();
        state.grid_contains[offset + 0] = contains_enc[0];
        state.grid_contains[offset + 1] = contains_enc[1];
        state.grid_contains[offset + 2] = contains_enc[2];
      }
    }
  }
  state.agent_pos = agent_pos_;
  state.agent_dir = agent_dir_;
  state.has_carrying = carrying_.GetType() != kEmpty;
  state.carrying_type = static_cast<int>(carrying_.GetType());
  state.carrying_color = static_cast<int>(carrying_.GetColor());
  state.carrying_state = carrying_.GetState();
  if (const WorldObj* contains = carrying_.GetContains(); contains != nullptr) {
    state.carrying_has_contains = true;
    state.carrying_contains_type = static_cast<int>(contains->GetType());
    state.carrying_contains_color = static_cast<int>(contains->GetColor());
    state.carrying_contains_state = contains->GetState();
  }
  state.target_pos = target_pos_;
  state.target_type = static_cast<int>(target_type_);
  state.target_color = static_cast<int>(target_color_);
  state.move_pos = move_pos_;
  state.move_type = static_cast<int>(move_type_);
  state.move_color = static_cast<int>(move_color_);
  state.success_pos = success_pos_;
  state.failure_pos = failure_pos_;
  state.goal_pos = goal_pos_;
  return state;
}

std::pair<int, int> MiniGridTask::RenderSize(int width, int height) const {
  return {width > 0 ? width : width_ * kTilePixels,
          height > 0 ? height : height_ * kTilePixels};
}

void MiniGridTask::Render(int width, int height, unsigned char* rgb) const {
  const int native_width = width_ * kTilePixels;
  const int native_height = height_ * kTilePixels;
  const auto [render_width, render_height] = RenderSize(width, height);
  std::vector<uint8_t> native(native_width * native_height * 3, 0);
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      const WorldObj& cell = grid_[y][x];
      const WorldObj* cell_ptr = cell.GetType() == kEmpty ? nullptr : &cell;
      const int agent_dir =
          agent_pos_.first == x && agent_pos_.second == y ? agent_dir_ : -1;
      const std::vector<uint8_t> tile = RenderTile(cell_ptr, agent_dir);
      for (int row = 0; row < kTilePixels; ++row) {
        const int dst_offset =
            ((y * kTilePixels + row) * native_width + x * kTilePixels) * 3;
        const int src_offset = row * kTilePixels * 3;
        std::memcpy(native.data() + dst_offset, tile.data() + src_offset,
                    kTilePixels * 3);
      }
    }
  }

  if (render_width == native_width && render_height == native_height) {
    std::memcpy(rgb, native.data(), native.size());
    return;
  }

  cv::Mat native_img(native_height, native_width, CV_8UC3, native.data());
  cv::Mat output(render_height, render_width, CV_8UC3, rgb);
  cv::resize(native_img, output, cv::Size(render_width, render_height), 0, 0,
             cv::INTER_AREA);
}

}  // namespace minigrid
