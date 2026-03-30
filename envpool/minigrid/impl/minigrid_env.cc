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

#include "envpool/minigrid/impl/minigrid_env.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/py_envpool.h"
#include "envpool/minigrid/minigrid.h"
#include <opencv2/opencv.hpp>

namespace minigrid {
namespace {

constexpr std::array<Pos, 4> kDirToVec = {
    Pos{1, 0}, Pos{0, 1}, Pos{-1, 0},
    Pos{0, -1}};  // NOLINT(whitespace/indent_namespace)

inline int GridOffset(int x, int y, int height) { return (x * height + y) * 3; }

inline int Manhattan(const Pos& lhs, const Pos& rhs) {
  return std::abs(lhs.first - rhs.first) + std::abs(lhs.second - rhs.second);
}

inline bool IsAdjacent(const Pos& lhs, const Pos& rhs) {
  return (lhs.first == rhs.first && std::abs(lhs.second - rhs.second) == 1) ||
         (lhs.second == rhs.second && std::abs(lhs.first - rhs.first) == 1);
}

int MissionObjectIndex(Type type) {
  switch (type) {
    case kKey:
      return 0;
    case kBall:
      return 1;
    default:
      return 2;
  }
}

Type OtherKeyBallType(Type type) {
  CHECK(type == kKey || type == kBall);
  return type == kKey ? kBall : kKey;
}

std::string MissionFetch(int syntax_idx, Color color, Type type) {
  static const std::array<const char*, 5> k_syntax = {
      "get a", "go get a", "fetch a", "go fetch a", "you must fetch a"};
  return std::string(k_syntax[syntax_idx]) + " " + ColorName(color) + " " +
         TypeName(type);
}

std::string MissionGoToDoor(Color color) {
  return "go to the " + ColorName(color) + " door";
}

std::string MissionGoToObject(Color color, Type type) {
  return "go to the " + ColorName(color) + " " + TypeName(type);
}

std::string MissionPutNear(Color move_color, Type move_type, Color target_color,
                           Type target_type) {
  return "put the " + ColorName(move_color) + " " + TypeName(move_type) +
         " near the " + ColorName(target_color) + " " + TypeName(target_type);
}

std::string MissionLockedRoom(Color locked_room_color, Color key_room_color) {
  return "get the " + ColorName(locked_room_color) + " key from the " +
         ColorName(key_room_color) + " room, unlock the " +
         ColorName(locked_room_color) + " door and go to the goal";
}

std::string MissionPickUp(Color color, Type type) {
  return "pick up the " + ColorName(color) + " " + TypeName(type);
}

using CoordFn = std::function<bool(float, float)>;
using Rgb = std::array<uint8_t, 3>;

constexpr int kTilePixels = 32;
constexpr int kTileSubdivs = 3;
constexpr Rgb kGridColor = {100, 100, 100};
constexpr Rgb kAgentColor = {255, 0, 0};
constexpr Rgb kLavaColor = {255, 128, 0};
constexpr std::array<Rgb, 6> kObjectColors = {
    Rgb{255, 0, 0},   Rgb{0, 255, 0},   Rgb{0, 0, 255},
    Rgb{112, 39, 195}, Rgb{255, 255, 0}, Rgb{100, 100, 100}};

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

CoordFn RotateFn(CoordFn fn, float cx, float cy, float theta) {
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
    const float a =
        std::clamp(pq[0] * unit[0] + pq[1] * unit[1], 0.0f, dist);
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
      const float yf = static_cast<float>(y + 0.5f) / height;
      const float xf = static_cast<float>(x + 0.5f) / width;
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
      FillCoords(img, width, height, PointInRect(0.0f, 1.0f, 0.0f, 1.0f), color);
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
    CoordFn tri = PointInTriangle({0.12f, 0.19f}, {0.87f, 0.50f},
                                  {0.12f, 0.81f});
    tri = RotateFn(std::move(tri), 0.5f, 0.5f,
                   0.5f * static_cast<float>(M_PI) * agent_dir);
    FillCoords(&img, hi_width, hi_height, tri, kAgentColor);
  }
  return Downsample(img, hi_width, hi_height, kTileSubdivs);
}

}  // namespace

MiniGridTask::MiniGridTask(std::string env_name, int max_steps,
                           int agent_view_size, bool see_through_walls,
                           int action_max)
    : max_steps_(max_steps),
      action_max_(action_max),
      agent_view_size_(agent_view_size),
      see_through_walls_(see_through_walls),
      env_name_(std::move(env_name)),
      carrying_(kEmpty) {}  // NOLINT(whitespace/indent_namespace)

void MiniGridTask::Reset() {
  CHECK(gen_ref_ != nullptr);
  step_count_ = 0;
  done_ = false;
  carrying_ = WorldObj(kEmpty);
  target_pos_ = {-1, -1};
  target_type_ = kEmpty;
  target_color_ = kRed;
  move_pos_ = {-1, -1};
  move_type_ = kEmpty;
  move_color_ = kRed;
  success_pos_ = {-1, -1};
  failure_pos_ = {-1, -1};
  goal_pos_ = {-1, -1};
  mission_.clear();
  mission_id_ = -1;
  GenGrid();
  CHECK_GE(agent_pos_.first, 0);
  CHECK_GE(agent_pos_.second, 0);
  CHECK_GE(agent_dir_, 0);
  CHECK_LT(agent_dir_, 4);
  CHECK(GetCell(agent_pos_.first, agent_pos_.second).CanOverlap());
}

float MiniGridTask::Step(Act act) {
  act = MapAction(act);
  step_count_ += 1;
  float reward = 0.0f;
  bool terminated = false;

  const Pos dir = DirVec();
  const Pos fwd_pos = {agent_pos_.first + dir.first,
                       agent_pos_.second + dir.second};
  CHECK(InBounds(fwd_pos.first, fwd_pos.second));
  const WorldObj pre_fwd = GetCell(fwd_pos.first, fwd_pos.second);
  const WorldObj pre_carrying = carrying_;
  BeforeStep(act, pre_fwd);

  if (act == kLeft) {
    agent_dir_ = (agent_dir_ + 3) % 4;
  } else if (act == kRight) {
    agent_dir_ = (agent_dir_ + 1) % 4;
  } else if (act == kForward) {
    const WorldObj cur_fwd = GetCell(fwd_pos.first, fwd_pos.second);
    if (cur_fwd.CanOverlap()) {
      agent_pos_ = fwd_pos;
    }
    if (cur_fwd.GetType() == kGoal) {
      reward = 1.0f - 0.9f * (static_cast<float>(step_count_) / max_steps_);
      terminated = true;
    } else if (cur_fwd.GetType() == kLava) {
      terminated = true;
    }
  } else if (act == kPickup) {
    const WorldObj cur_fwd = GetCell(fwd_pos.first, fwd_pos.second);
    if (carrying_.GetType() == kEmpty && cur_fwd.CanPickup()) {
      carrying_ = cur_fwd;
      SetEmpty(fwd_pos.first, fwd_pos.second);
    }
  } else if (act == kDrop) {
    const WorldObj cur_fwd = GetCell(fwd_pos.first, fwd_pos.second);
    if (carrying_.GetType() != kEmpty && cur_fwd.GetType() == kEmpty) {
      SetCell(fwd_pos.first, fwd_pos.second, carrying_);
      carrying_ = WorldObj(kEmpty);
    }
  } else if (act == kToggle) {
    WorldObj& cur_fwd = Cell(fwd_pos.first, fwd_pos.second);
    if (cur_fwd.GetType() == kDoor) {
      if (cur_fwd.GetDoorLocked()) {
        if (carrying_.GetType() == kKey &&
            carrying_.GetColor() == cur_fwd.GetColor()) {
          cur_fwd.SetDoorLocked(false);
          cur_fwd.SetDoorOpen(true);
        }
      } else {
        cur_fwd.SetDoorOpen(!cur_fwd.GetDoorOpen());
      }
    } else if (cur_fwd.GetType() == kBox) {
      auto contains = cur_fwd.ReleaseContains();
      if (contains != nullptr) {
        cur_fwd = *contains;
      } else {
        cur_fwd = WorldObj(kEmpty);
      }
    }
  } else {
    CHECK_EQ(act, kDone);
  }

  AfterStep(act, pre_fwd, fwd_pos, pre_carrying, &reward, &terminated);
  if (step_count_ >= max_steps_) {
    terminated = true;
  }
  done_ = terminated;
  return reward;
}

bool MiniGridTask::InBounds(int x, int y) const {
  return x >= 0 && x < width_ && y >= 0 && y < height_;
}

WorldObj MiniGridTask::GetCell(int x, int y) const {
  CHECK(InBounds(x, y));
  return grid_[y][x];
}

WorldObj& MiniGridTask::Cell(int x, int y) {
  CHECK(InBounds(x, y));
  return grid_[y][x];
}

void MiniGridTask::SetCell(int x, int y, const WorldObj& obj) {
  CHECK(InBounds(x, y));
  grid_[y][x] = obj;
}

void MiniGridTask::SetEmpty(int x, int y) { SetCell(x, y, WorldObj(kEmpty)); }

void MiniGridTask::ClearGrid(int width, int height) {
  width_ = width;
  height_ = height;
  grid_.assign(height_, std::vector<WorldObj>(width_, WorldObj(kEmpty)));
}

void MiniGridTask::HorzWall(int x, int y, int length, Type type, Color color) {
  if (length < 0) {
    length = width_ - x;
  }
  for (int i = 0; i < length; ++i) {
    SetCell(x + i, y, WorldObj(type, color));
  }
}

void MiniGridTask::VertWall(int x, int y, int length, Type type, Color color) {
  if (length < 0) {
    length = height_ - y;
  }
  for (int j = 0; j < length; ++j) {
    SetCell(x, y + j, WorldObj(type, color));
  }
}

void MiniGridTask::WallRect(int x, int y, int width, int height) {
  HorzWall(x, y, width);
  HorzWall(x, y + height - 1, width);
  VertWall(x, y, height);
  VertWall(x + width - 1, y, height);
}

void MiniGridTask::PutObj(const WorldObj& obj, int x, int y) {
  SetCell(x, y, obj);
}

Pos MiniGridTask::PlaceObj(const WorldObj& obj, int top_x, int top_y,
                           int size_x, int size_y, const RejectFn& reject_fn,
                           int max_tries) {
  if (size_x < 0) {
    size_x = width_;
  }
  if (size_y < 0) {
    size_y = height_;
  }
  top_x = std::max(top_x, 0);
  top_y = std::max(top_y, 0);

  int num_tries = 0;
  while (true) {
    CHECK_LE(num_tries, max_tries) << "rejection sampling failed";
    ++num_tries;
    int x = RandInt(top_x, std::min(top_x + size_x, width_));
    int y = RandInt(top_y, std::min(top_y + size_y, height_));
    Pos pos{x, y};
    if (GetCell(x, y).GetType() != kEmpty) {
      continue;
    }
    if (pos == agent_pos_) {
      continue;
    }
    if (reject_fn && reject_fn(pos)) {
      continue;
    }
    PutObj(obj, x, y);
    return pos;
  }
}

Pos MiniGridTask::PlaceAgent(int top_x, int top_y, int size_x, int size_y,
                             bool rand_dir, int max_tries) {
  agent_pos_ = {-1, -1};
  Pos pos = PlaceObj(WorldObj(kEmpty), top_x, top_y, size_x, size_y, RejectFn(),
                     max_tries);
  agent_pos_ = pos;
  if (rand_dir) {
    agent_dir_ = RandInt(0, 4);
  }
  return pos;
}

int MiniGridTask::RandInt(int low, int high) {
  CHECK_LT(low, high);
  std::uniform_int_distribution<> dist(low, high - 1);
  return dist(*gen_ref_);
}

bool MiniGridTask::RandBool() { return RandInt(0, 2) == 0; }

Pos MiniGridTask::DirVec() const { return kDirToVec[agent_dir_]; }

Pos MiniGridTask::RightVec() const {
  const Pos dir = DirVec();
  return {-dir.second, dir.first};
}

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
  int n = std::min(static_cast<int>(mission_.size()), kMissionBytes - 1);
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
      const WorldObj* cell_ptr =
          cell.GetType() == kEmpty ? nullptr : &cell;
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

RoomGridTask::RoomGridTask(std::string env_name, int room_size, int num_rows,
                           int num_cols, int max_steps, int agent_view_size)
    : MiniGridTask(std::move(env_name), max_steps, agent_view_size, false, 6),
      room_size_(room_size),
      num_rows_(num_rows),
      num_cols_(num_cols) {}  // NOLINT(whitespace/indent_namespace)

void RoomGridTask::GenGrid() {
  int height = (room_size_ - 1) * num_rows_ + 1;
  int width = (room_size_ - 1) * num_cols_ + 1;
  ClearGrid(width, height);
  room_grid_.assign(num_rows_, std::vector<Room>(num_cols_));
  for (int j = 0; j < num_rows_; ++j) {
    for (int i = 0; i < num_cols_; ++i) {
      Room& room = room_grid_[j][i];
      room.top = {i * (room_size_ - 1), j * (room_size_ - 1)};
      room.size = {room_size_, room_size_};
      WallRect(room.top.first, room.top.second, room.size.first,
               room.size.second);
    }
  }

  for (int j = 0; j < num_rows_; ++j) {
    for (int i = 0; i < num_cols_; ++i) {
      Room& room = room_grid_[j][i];
      int x_l = room.top.first + 1;
      int y_l = room.top.second + 1;
      int x_m = room.top.first + room.size.first - 1;
      int y_m = room.top.second + room.size.second - 1;
      if (i < num_cols_ - 1) {
        room.has_neighbor[0] = true;
        room.neighbors[0] = {i + 1, j};
        room.door_pos[0] = {x_m, RandInt(y_l, y_m)};
      }
      if (j < num_rows_ - 1) {
        room.has_neighbor[1] = true;
        room.neighbors[1] = {i, j + 1};
        room.door_pos[1] = {RandInt(x_l, x_m), y_m};
      }
      if (i > 0) {
        room.has_neighbor[2] = true;
        room.neighbors[2] = {i - 1, j};
        room.door_pos[2] = room_grid_[j][i - 1].door_pos[0];
      }
      if (j > 0) {
        room.has_neighbor[3] = true;
        room.neighbors[3] = {i, j - 1};
        room.door_pos[3] = room_grid_[j - 1][i].door_pos[1];
      }
    }
  }
  agent_pos_ = {(num_cols_ / 2) * (room_size_ - 1) + room_size_ / 2,
                (num_rows_ / 2) * (room_size_ - 1) + room_size_ / 2};
  agent_dir_ = 0;
}

Room& RoomGridTask::GetRoom(int i, int j) {
  CHECK_GE(i, 0);
  CHECK_LT(i, num_cols_);
  CHECK_GE(j, 0);
  CHECK_LT(j, num_rows_);
  return room_grid_[j][i];
}

const Room& RoomGridTask::GetRoom(int i, int j) const {
  CHECK_GE(i, 0);
  CHECK_LT(i, num_cols_);
  CHECK_GE(j, 0);
  CHECK_LT(j, num_rows_);
  return room_grid_[j][i];
}

Room& RoomGridTask::RoomFromPos(int x, int y) {
  return GetRoom(x / (room_size_ - 1), y / (room_size_ - 1));
}

std::pair<Pos, std::pair<Type, Color>> RoomGridTask::AddObject(int i, int j,
                                                               Type type,
                                                               Color color) {
  if (type == kEmpty) {
    type =
        RandElem(std::vector<Type>(kObjectTypes.begin(), kObjectTypes.end()));
  }
  if (color == kUnassigned) {
    color = RandColor();
  }
  Pos pos = PlaceObj(
      WorldObj(type, color), GetRoom(i, j).top.first, GetRoom(i, j).top.second,
      GetRoom(i, j).size.first, GetRoom(i, j).size.second,
      [&](const Pos& pos_candidate) {
        return Manhattan(agent_pos_, pos_candidate) < 2;
      },
      1000);
  GetRoom(i, j).objs.emplace_back(type, color);
  return {pos, {type, color}};
}

Pos RoomGridTask::AddDoor(int i, int j, int door_idx, Color color,
                          bool locked) {
  Room& room = GetRoom(i, j);
  if (door_idx < 0) {
    while (true) {
      door_idx = RandInt(0, 4);
      if (room.has_neighbor[door_idx] && !room.connected[door_idx]) {
        break;
      }
    }
  }
  if (color == kUnassigned) {
    color = RandColor();
  }
  room.locked = locked;
  Pos pos = room.door_pos[door_idx];
  PutObj(MakeDoor(color, locked, false), pos.first, pos.second);
  room.connected[door_idx] = true;
  const Pos neighbor = room.neighbors[door_idx];
  GetRoom(neighbor.first, neighbor.second).connected[(door_idx + 2) % 4] = true;
  return pos;
}

void RoomGridTask::RemoveWall(int i, int j, int wall_idx) {
  Room& room = GetRoom(i, j);
  CHECK(room.has_neighbor[wall_idx]);
  CHECK(!room.connected[wall_idx]);
  int tx = room.top.first;
  int ty = room.top.second;
  int w = room.size.first;
  int h = room.size.second;
  if (wall_idx == 0) {
    for (int k = 1; k < h - 1; ++k) {
      SetEmpty(tx + w - 1, ty + k);
    }
  } else if (wall_idx == 1) {
    for (int k = 1; k < w - 1; ++k) {
      SetEmpty(tx + k, ty + h - 1);
    }
  } else if (wall_idx == 2) {
    for (int k = 1; k < h - 1; ++k) {
      SetEmpty(tx, ty + k);
    }
  } else {
    for (int k = 1; k < w - 1; ++k) {
      SetEmpty(tx + k, ty);
    }
  }
  room.connected[wall_idx] = true;
  const Pos neighbor = room.neighbors[wall_idx];
  GetRoom(neighbor.first, neighbor.second).connected[(wall_idx + 2) % 4] = true;
}

Pos RoomGridTask::PlaceAgentInRoom(int i, int j, bool rand_dir) {
  if (i < 0) {
    i = RandInt(0, num_cols_);
  }
  if (j < 0) {
    j = RandInt(0, num_rows_);
  }
  Room& room = GetRoom(i, j);
  while (true) {
    Pos pos = PlaceAgent(room.top.first, room.top.second, room.size.first,
                         room.size.second, rand_dir, 1000);
    const Pos dir = DirVec();
    Pos front{pos.first + dir.first, pos.second + dir.second};
    if (InBounds(front.first, front.second) &&
        (GetCell(front.first, front.second).GetType() == kEmpty ||
         GetCell(front.first, front.second).GetType() == kWall)) {
      return pos;
    }
  }
}

void RoomGridTask::ConnectAll() {
  auto reachable = [&]() {
    std::vector<std::vector<bool>> seen(num_rows_,
                                        std::vector<bool>(num_cols_, false));
    std::queue<Pos> queue;
    Pos start_room = {agent_pos_.first / (room_size_ - 1),
                      agent_pos_.second / (room_size_ - 1)};
    queue.push(start_room);
    while (!queue.empty()) {
      Pos cur = queue.front();
      queue.pop();
      if (seen[cur.second][cur.first]) {
        continue;
      }
      seen[cur.second][cur.first] = true;
      const Room& room = GetRoom(cur.first, cur.second);
      for (int k = 0; k < 4; ++k) {
        if (room.connected[k] && room.has_neighbor[k]) {
          queue.push(room.neighbors[k]);
        }
      }
    }
    return seen;
  };

  for (int itr = 0; itr <= 5000; ++itr) {
    auto seen = reachable();
    int count = 0;
    for (const auto& row : seen) {
      count += std::count(row.begin(), row.end(), true);
    }
    if (count == num_rows_ * num_cols_) {
      return;
    }
    int i = RandInt(0, num_cols_);
    int j = RandInt(0, num_rows_);
    int k = RandInt(0, 4);
    Room& room = GetRoom(i, j);
    if (!room.has_neighbor[k] || room.connected[k]) {
      continue;
    }
    const Pos nb = room.neighbors[k];
    if (room.locked || GetRoom(nb.first, nb.second).locked) {
      continue;
    }
    AddDoor(i, j, k, RandColor(), false);
  }
  LOG(FATAL) << "connect_all failed";
}

EmptyTask::EmptyTask(int size, Pos agent_start_pos, int agent_start_dir,
                     int max_steps, int agent_view_size)
    : MiniGridTask("empty", max_steps, agent_view_size, true, 6),
      size_(size),
      agent_start_pos_(std::move(agent_start_pos)),
      agent_start_dir_(agent_start_dir) {
}  // NOLINT(whitespace/indent_namespace)

void EmptyTask::GenGrid() {
  ClearGrid(size_, size_);
  WallRect(0, 0, size_, size_);
  PutObj(WorldObj(kGoal, kGreen), size_ - 2, size_ - 2);
  goal_pos_ = {size_ - 2, size_ - 2};
  if (agent_start_pos_.first >= 0) {
    agent_pos_ = agent_start_pos_;
    agent_dir_ = agent_start_dir_;
  } else {
    PlaceAgent(1, 1, size_ - 2, size_ - 2, true);
  }
  SetMission("get to the green goal square", 0);
}

DoorKeyTask::DoorKeyTask(int size, int max_steps)
    : MiniGridTask("doorkey", max_steps, 7, false, 6), size_(size) {}

void DoorKeyTask::GenGrid() {
  ClearGrid(size_, size_);
  WallRect(0, 0, size_, size_);
  PutObj(WorldObj(kGoal, kGreen), size_ - 2, size_ - 2);
  goal_pos_ = {size_ - 2, size_ - 2};
  int split_idx = RandInt(2, size_ - 2);
  VertWall(split_idx, 0);
  PlaceAgent(0, 0, split_idx, size_, true);
  int door_idx = RandInt(1, size_ - 2);
  PutObj(MakeDoor(kYellow, true, false), split_idx, door_idx);
  PlaceObj(WorldObj(kKey, kYellow), 0, 0, split_idx, size_);
  SetMission("use the key to open the door and then get to the goal", 0);
}

DistShiftTask::DistShiftTask(int width, int height, Pos agent_start_pos,
                             int agent_start_dir, int strip2_row, int max_steps)
    : MiniGridTask("distshift", max_steps, 7, true, 6),
      agent_start_pos_(std::move(agent_start_pos)),
      agent_start_dir_(agent_start_dir),
      strip2_row_(strip2_row) {
  width_ = width;
  height_ = height;
}

void DistShiftTask::GenGrid() {
  ClearGrid(width_, height_);
  WallRect(0, 0, width_, height_);
  PutObj(WorldObj(kGoal, kGreen), width_ - 2, 1);
  goal_pos_ = {width_ - 2, 1};
  for (int i = 0; i < width_ - 6; ++i) {
    PutObj(WorldObj(kLava), 3 + i, 1);
    PutObj(WorldObj(kLava), 3 + i, strip2_row_);
  }
  if (agent_start_pos_.first >= 0) {
    agent_pos_ = agent_start_pos_;
    agent_dir_ = agent_start_dir_;
  } else {
    PlaceAgent();
  }
  SetMission("get to the green goal square", 0);
}

LavaGapTask::LavaGapTask(int size, Type obstacle_type, int max_steps)
    : MiniGridTask("lavgap", max_steps, 7, false, 6),
      size_(size),
      obstacle_type_(obstacle_type) {}  // NOLINT(whitespace/indent_namespace)

void LavaGapTask::GenGrid() {
  ClearGrid(size_, size_);
  WallRect(0, 0, size_, size_);
  agent_pos_ = {1, 1};
  agent_dir_ = 0;
  goal_pos_ = {size_ - 2, size_ - 2};
  PutObj(WorldObj(kGoal, kGreen), goal_pos_.first, goal_pos_.second);
  Pos gap_pos{RandInt(2, size_ - 2), RandInt(1, size_ - 1)};
  VertWall(gap_pos.first, 1, size_ - 2, obstacle_type_,
           DefaultColor(obstacle_type_));
  SetEmpty(gap_pos.first, gap_pos.second);
  SetMission(obstacle_type_ == kLava
                 ? "avoid the lava and get to the green goal square"
                 : "find the opening and get to the green goal square",
             0);
}

CrossingTask::CrossingTask(int size, int num_crossings, Type obstacle_type,
                           int max_steps)
    : MiniGridTask("crossing", max_steps, 7, false, 6),
      size_(size),
      num_crossings_(num_crossings),
      obstacle_type_(obstacle_type) {}  // NOLINT(whitespace/indent_namespace)

void CrossingTask::GenGrid() {
  CHECK_EQ(size_ % 2, 1);
  ClearGrid(size_, size_);
  WallRect(0, 0, size_, size_);
  agent_pos_ = {1, 1};
  agent_dir_ = 0;
  goal_pos_ = {size_ - 2, size_ - 2};
  PutObj(WorldObj(kGoal, kGreen), goal_pos_.first, goal_pos_.second);

  std::vector<std::pair<bool, int>> rivers;
  for (int i = 2; i < size_ - 2; i += 2) {
    rivers.emplace_back(true, i);
    rivers.emplace_back(false, i);
  }
  std::shuffle(rivers.begin(), rivers.end(), *gen_ref_);
  rivers.resize(num_crossings_);
  std::vector<int> rivers_v;
  std::vector<int> rivers_h;
  for (const auto& river : rivers) {
    if (river.first) {
      rivers_v.push_back(river.second);
    } else {
      rivers_h.push_back(river.second);
    }
  }
  std::sort(rivers_v.begin(), rivers_v.end());
  std::sort(rivers_h.begin(), rivers_h.end());
  for (int y : rivers_h) {
    for (int x = 1; x < size_ - 1; ++x) {
      PutObj(WorldObj(obstacle_type_), x, y);
    }
  }
  for (int x : rivers_v) {
    for (int y = 1; y < size_ - 1; ++y) {
      PutObj(WorldObj(obstacle_type_), x, y);
    }
  }

  std::vector<bool> path;
  path.insert(path.end(), rivers_v.size(), true);
  path.insert(path.end(), rivers_h.size(), false);
  std::shuffle(path.begin(), path.end(), *gen_ref_);

  std::vector<int> limits_v = {0};
  limits_v.insert(limits_v.end(), rivers_v.begin(), rivers_v.end());
  limits_v.push_back(size_ - 1);
  std::vector<int> limits_h = {0};
  limits_h.insert(limits_h.end(), rivers_h.begin(), rivers_h.end());
  limits_h.push_back(size_ - 1);
  int room_i = 0;
  int room_j = 0;
  for (bool is_horizontal_move : path) {
    int x = 0;
    int y = 0;
    if (is_horizontal_move) {
      x = limits_v[room_i + 1];
      y = RandInt(limits_h[room_j] + 1, limits_h[room_j + 1]);
      room_i += 1;
    } else {
      x = RandInt(limits_v[room_i] + 1, limits_v[room_i + 1]);
      y = limits_h[room_j + 1];
      room_j += 1;
    }
    SetEmpty(x, y);
  }
  SetMission(obstacle_type_ == kLava
                 ? "avoid the lava and get to the green goal square"
                 : "find the opening and get to the green goal square",
             0);
}

DynamicObstaclesTask::DynamicObstaclesTask(int size, Pos agent_start_pos,
                                           int agent_start_dir, int n_obstacles,
                                           int max_steps)
    : MiniGridTask("dynamic_obstacles", max_steps, 7, true, 2),
      size_(size),
      agent_start_pos_(std::move(agent_start_pos)),
      agent_start_dir_(agent_start_dir),
      n_obstacles_(n_obstacles <= size / 2 + 1 ? n_obstacles : size / 2) {
}  // NOLINT(whitespace/indent_namespace)

void DynamicObstaclesTask::GenGrid() {
  ClearGrid(size_, size_);
  WallRect(0, 0, size_, size_);
  PutObj(WorldObj(kGoal, kGreen), size_ - 2, size_ - 2);
  goal_pos_ = {size_ - 2, size_ - 2};
  if (agent_start_pos_.first >= 0) {
    agent_pos_ = agent_start_pos_;
    agent_dir_ = agent_start_dir_;
  } else {
    PlaceAgent();
  }
  obstacle_pos_.clear();
  for (int i = 0; i < n_obstacles_; ++i) {
    obstacle_pos_.push_back(PlaceObj(WorldObj(kBall, kBlue), 0, 0, width_,
                                     height_, RejectFn(), 100));
  }
  SetMission("get to the green goal square", 0);
}

void DynamicObstaclesTask::BeforeStep(Act act, const WorldObj& pre_fwd) {
  pre_front_blocked_ = act == kForward && pre_fwd.GetType() != kGoal &&
                       pre_fwd.GetType() != kEmpty;
  std::vector<Pos> new_pos = obstacle_pos_;
  for (std::size_t i = 0; i < obstacle_pos_.size(); ++i) {
    Pos old_pos = obstacle_pos_[i];
    int top_x = std::max(old_pos.first - 1, 0);
    int top_y = std::max(old_pos.second - 1, 0);
    int end_x = std::min(old_pos.first + 2, width_);
    int end_y = std::min(old_pos.second + 2, height_);
    for (int attempt = 0; attempt < 100; ++attempt) {
      int x = RandInt(top_x, end_x);
      int y = RandInt(top_y, end_y);
      Pos pos{x, y};
      if (GetCell(x, y).GetType() != kEmpty || pos == agent_pos_) {
        continue;
      }
      PutObj(WorldObj(kBall, kBlue), x, y);
      SetEmpty(old_pos.first, old_pos.second);
      new_pos[i] = pos;
      break;
    }
  }
  obstacle_pos_ = new_pos;
}

void DynamicObstaclesTask::AfterStep(Act act, const WorldObj& pre_fwd,
                                     const Pos& fwd_pos,
                                     const WorldObj& pre_carrying,
                                     float* reward, bool* terminated) {
  if (act == kForward && pre_front_blocked_) {
    *reward = -1.0f;
    *terminated = true;
  }
}

MiniGridDebugState DynamicObstaclesTask::DebugState() const {
  MiniGridDebugState state = MiniGridTask::DebugState();
  state.obstacle_positions.reserve(obstacle_pos_.size() * 2);
  for (const Pos& pos : obstacle_pos_) {
    state.obstacle_positions.push_back(pos.first);
    state.obstacle_positions.push_back(pos.second);
  }
  return state;
}

FetchTask::FetchTask(int size, int num_objs, int max_steps)
    : MiniGridTask("fetch", max_steps, 7, true, 6),
      size_(size),
      num_objs_(num_objs) {}  // NOLINT(whitespace/indent_namespace)

void FetchTask::GenGrid() {
  ClearGrid(size_, size_);
  HorzWall(0, 0);
  HorzWall(0, size_ - 1);
  VertWall(0, 0);
  VertWall(size_ - 1, 0);
  std::vector<std::pair<Type, Color>> objs;
  while (static_cast<int>(objs.size()) < num_objs_) {
    Type type = RandElem(std::vector<Type>{kKey, kBall});
    Color color = RandColor();
    PlaceObj(WorldObj(type, color));
    objs.emplace_back(type, color);
  }
  PlaceAgent();
  int target_idx = RandInt(0, static_cast<int>(objs.size()));
  target_type_ = objs[target_idx].first;
  target_color_ = objs[target_idx].second;
  int syntax_idx = RandInt(0, 5);
  SetMission(MissionFetch(syntax_idx, target_color_, target_type_),
             syntax_idx * 12 + static_cast<int>(target_color_) * 2 +
                 (target_type_ == kBall ? 1 : 0));
}

void FetchTask::AfterStep(Act act, const WorldObj& pre_fwd, const Pos& fwd_pos,
                          const WorldObj& pre_carrying, float* reward,
                          bool* terminated) {
  if (carrying_.GetType() == kEmpty) {
    return;
  }
  *terminated = true;
  if (carrying_.GetType() == target_type_ &&
      carrying_.GetColor() == target_color_) {
    *reward = 1.0f - 0.9f * (static_cast<float>(step_count_) / max_steps_);
  } else {
    *reward = 0.0f;
  }
}

GoToDoorTask::GoToDoorTask(int size, int max_steps)
    : MiniGridTask("goto_door", max_steps, 7, true, 6), size_(size) {}

void GoToDoorTask::GenGrid() {
  ClearGrid(size_, size_);
  int active_width = RandInt(5, size_ + 1);
  int active_height = RandInt(5, size_ + 1);
  WallRect(0, 0, active_width, active_height);
  std::vector<Pos> door_pos = {
      {RandInt(2, active_width - 2), 0},
      {RandInt(2, active_width - 2), active_height - 1},
      {0, RandInt(2, active_height - 2)},
      {active_width - 1, RandInt(2, active_height - 2)},
  };
  std::vector<Color> door_colors;
  while (static_cast<int>(door_colors.size()) <
         static_cast<int>(door_pos.size())) {
    Color color = RandColor();
    if (std::find(door_colors.begin(), door_colors.end(), color) !=
        door_colors.end()) {
      continue;
    }
    door_colors.push_back(color);
  }
  for (std::size_t i = 0; i < door_pos.size(); ++i) {
    PutObj(MakeDoor(door_colors[i], false, false), door_pos[i].first,
           door_pos[i].second);
  }
  PlaceAgent(0, 0, active_width, active_height, true);
  int door_idx = RandInt(0, static_cast<int>(door_pos.size()));
  target_pos_ = door_pos[door_idx];
  target_type_ = kDoor;
  target_color_ = door_colors[door_idx];
  SetMission(MissionGoToDoor(target_color_), static_cast<int>(target_color_));
}

void GoToDoorTask::AfterStep(Act act, const WorldObj& pre_fwd,
                             const Pos& fwd_pos, const WorldObj& pre_carrying,
                             float* reward, bool* terminated) {
  if (act == kToggle) {
    *terminated = true;
    return;
  }
  if (act == kDone) {
    if (IsAdjacent(agent_pos_, target_pos_)) {
      *reward = 1.0f - 0.9f * (static_cast<float>(step_count_) / max_steps_);
    }
    *terminated = true;
  }
}

GoToObjectTask::GoToObjectTask(int size, int num_objs, int max_steps)
    : MiniGridTask("goto_object", max_steps, 7, true, 6),
      size_(size),
      num_objs_(num_objs) {}  // NOLINT(whitespace/indent_namespace)

void GoToObjectTask::GenGrid() {
  ClearGrid(size_, size_);
  WallRect(0, 0, size_, size_);
  std::vector<std::pair<Type, Color>> objs;
  std::vector<Pos> positions;
  while (static_cast<int>(objs.size()) < num_objs_) {
    Type type =
        RandElem(std::vector<Type>(kObjectTypes.begin(), kObjectTypes.end()));
    Color color = RandColor();
    if (std::find(objs.begin(), objs.end(),
                  std::pair<Type, Color>{type, color}) != objs.end()) {
      continue;
    }
    positions.push_back(PlaceObj(WorldObj(type, color)));
    objs.emplace_back(type, color);
  }
  PlaceAgent();
  int idx = RandInt(0, static_cast<int>(objs.size()));
  target_pos_ = positions[idx];
  target_type_ = objs[idx].first;
  target_color_ = objs[idx].second;
  SetMission(
      MissionGoToObject(target_color_, target_type_),
      static_cast<int>(target_color_) * 3 + MissionObjectIndex(target_type_));
}

void GoToObjectTask::AfterStep(Act act, const WorldObj& pre_fwd,
                               const Pos& fwd_pos, const WorldObj& pre_carrying,
                               float* reward, bool* terminated) {
  if (act == kToggle) {
    *terminated = true;
    return;
  }
  if (act == kDone) {
    if (IsAdjacent(agent_pos_, target_pos_)) {
      *reward = 1.0f - 0.9f * (static_cast<float>(step_count_) / max_steps_);
    }
    *terminated = true;
  }
}

PutNearTask::PutNearTask(int size, int num_objs, int max_steps)
    : MiniGridTask("put_near", max_steps, 7, true, 6),
      size_(size),
      num_objs_(num_objs) {}  // NOLINT(whitespace/indent_namespace)

void PutNearTask::GenGrid() {
  ClearGrid(size_, size_);
  HorzWall(0, 0);
  HorzWall(0, size_ - 1);
  VertWall(0, 0);
  VertWall(size_ - 1, 0);
  std::vector<std::pair<Type, Color>> objs;
  std::vector<Pos> positions;
  auto near_existing = [&](const Pos& candidate) {
    return std::any_of(positions.begin(), positions.end(), [&](const Pos& pos) {
      return std::abs(pos.first - candidate.first) <= 1 &&
             std::abs(pos.second - candidate.second) <= 1;
    });
  };
  while (static_cast<int>(objs.size()) < num_objs_) {
    Type type =
        RandElem(std::vector<Type>(kObjectTypes.begin(), kObjectTypes.end()));
    Color color = RandColor();
    if (std::find(objs.begin(), objs.end(),
                  std::pair<Type, Color>{type, color}) != objs.end()) {
      continue;
    }
    positions.push_back(
        PlaceObj(WorldObj(type, color), 0, 0, width_, height_, near_existing));
    objs.emplace_back(type, color);
  }
  PlaceAgent();
  int move_idx = RandInt(0, static_cast<int>(objs.size()));
  move_pos_ = positions[move_idx];
  move_type_ = objs[move_idx].first;
  move_color_ = objs[move_idx].second;
  int target_idx = move_idx;
  while (target_idx == move_idx) {
    target_idx = RandInt(0, static_cast<int>(objs.size()));
  }
  target_pos_ = positions[target_idx];
  target_type_ = objs[target_idx].first;
  target_color_ = objs[target_idx].second;
  SetMission(
      MissionPutNear(move_color_, move_type_, target_color_, target_type_),
      ((static_cast<int>(move_color_) * 3 + MissionObjectIndex(move_type_)) *
       18) +
          (static_cast<int>(target_color_) * 3) +
          MissionObjectIndex(target_type_));
}

void PutNearTask::AfterStep(Act act, const WorldObj& pre_fwd,
                            const Pos& fwd_pos, const WorldObj& pre_carrying,
                            float* reward, bool* terminated) {
  if (act == kPickup && carrying_.GetType() != kEmpty &&
      (carrying_.GetType() != move_type_ ||
       carrying_.GetColor() != move_color_)) {
    *terminated = true;
    return;
  }
  if (act == kDrop && pre_carrying.GetType() != kEmpty) {
    if (GetCell(fwd_pos.first, fwd_pos.second) == pre_carrying &&
        std::abs(fwd_pos.first - target_pos_.first) <= 1 &&
        std::abs(fwd_pos.second - target_pos_.second) <= 1) {
      *reward = 1.0f - 0.9f * (static_cast<float>(step_count_) / max_steps_);
    }
    *terminated = true;
  }
}

RedBlueDoorTask::RedBlueDoorTask(int size, int max_steps)
    : MiniGridTask("red_blue_door", max_steps, 7, false, 6), size_(size) {}

void RedBlueDoorTask::GenGrid() {
  ClearGrid(2 * size_, size_);
  WallRect(0, 0, 2 * size_, size_);
  WallRect(size_ / 2, 0, size_, size_);
  PlaceAgent(size_ / 2, 0, size_, size_, true);
  red_door_pos_ = {size_ / 2, RandInt(1, size_ - 1)};
  blue_door_pos_ = {size_ / 2 + size_ - 1, RandInt(1, size_ - 1)};
  PutObj(MakeDoor(kRed, false, false), red_door_pos_.first,
         red_door_pos_.second);
  PutObj(MakeDoor(kBlue, false, false), blue_door_pos_.first,
         blue_door_pos_.second);
  SetMission("open the red door then the blue door", 0);
}

void RedBlueDoorTask::BeforeStep(Act act, const WorldObj& pre_fwd) {
  red_open_before_ =
      GetCell(red_door_pos_.first, red_door_pos_.second).GetDoorOpen();
  blue_open_before_ =
      GetCell(blue_door_pos_.first, blue_door_pos_.second).GetDoorOpen();
}

void RedBlueDoorTask::AfterStep(Act act, const WorldObj& pre_fwd,
                                const Pos& fwd_pos,
                                const WorldObj& pre_carrying, float* reward,
                                bool* terminated) {
  bool red_open_after =
      GetCell(red_door_pos_.first, red_door_pos_.second).GetDoorOpen();
  bool blue_open_after =
      GetCell(blue_door_pos_.first, blue_door_pos_.second).GetDoorOpen();
  if (blue_open_after) {
    if (red_open_before_) {
      *reward = 1.0f - 0.9f * (static_cast<float>(step_count_) / max_steps_);
    }
    *terminated = true;
  } else if (red_open_after && blue_open_before_) {
    *terminated = true;
  }
}

LockedRoomTask::LockedRoomTask(int size, int max_steps)
    : MiniGridTask("locked_room", max_steps, 7, false, 6), size_(size) {}

void LockedRoomTask::GenGrid() {
  ClearGrid(size_, size_);
  for (int i = 0; i < size_; ++i) {
    PutObj(WorldObj(kWall, kGrey), i, 0);
    PutObj(WorldObj(kWall, kGrey), i, size_ - 1);
  }
  for (int j = 0; j < size_; ++j) {
    PutObj(WorldObj(kWall, kGrey), 0, j);
    PutObj(WorldObj(kWall, kGrey), size_ - 1, j);
  }
  int left_wall = size_ / 2 - 2;
  int right_wall = size_ / 2 + 2;
  for (int j = 0; j < size_; ++j) {
    PutObj(WorldObj(kWall, kGrey), left_wall, j);
    PutObj(WorldObj(kWall, kGrey), right_wall, j);
  }
  struct LockedRoomInfo {
    Pos top;
    Pos size;
    Pos door_pos;
    Color color{kRed};
    bool locked{false};
  };
  std::vector<LockedRoomInfo> rooms;
  for (int n = 0; n < 3; ++n) {
    int j = n * (size_ / 3);
    for (int i = 0; i < left_wall; ++i) {
      PutObj(WorldObj(kWall, kGrey), i, j);
    }
    for (int i = right_wall; i < size_; ++i) {
      PutObj(WorldObj(kWall, kGrey), i, j);
    }
    int room_w = left_wall + 1;
    int room_h = size_ / 3 + 1;
    rooms.push_back({{0, j}, {room_w, room_h}, {left_wall, j + 3}});
    rooms.push_back({{right_wall, j}, {room_w, room_h}, {right_wall, j + 3}});
  }
  int locked_idx = RandInt(0, static_cast<int>(rooms.size()));
  rooms[locked_idx].locked = true;
  goal_pos_ = {
      RandInt(rooms[locked_idx].top.first + 1,
              rooms[locked_idx].top.first + rooms[locked_idx].size.first - 1),
      RandInt(
          rooms[locked_idx].top.second + 1,
          rooms[locked_idx].top.second + rooms[locked_idx].size.second - 1)};
  PutObj(WorldObj(kGoal, kGreen), goal_pos_.first, goal_pos_.second);
  std::vector<Color> colors(kColors.begin(), kColors.end());
  for (auto& room : rooms) {
    int idx = RandInt(0, static_cast<int>(colors.size()));
    room.color = colors[idx];
    colors.erase(colors.begin() + idx);
    PutObj(MakeDoor(room.color, room.locked, false), room.door_pos.first,
           room.door_pos.second);
  }
  int key_room_idx = locked_idx;
  while (key_room_idx == locked_idx) {
    key_room_idx = RandInt(0, static_cast<int>(rooms.size()));
  }
  Pos key_pos{RandInt(rooms[key_room_idx].top.first + 1,
                      rooms[key_room_idx].top.first +
                          rooms[key_room_idx].size.first - 1),
              RandInt(rooms[key_room_idx].top.second + 1,
                      rooms[key_room_idx].top.second +
                          rooms[key_room_idx].size.second - 1)};
  PutObj(WorldObj(kKey, rooms[locked_idx].color), key_pos.first,
         key_pos.second);
  PlaceAgent(left_wall, 0, right_wall - left_wall, size_, true);
  SetMission(
      MissionLockedRoom(rooms[locked_idx].color, rooms[key_room_idx].color), 0);
}

MemoryTask::MemoryTask(int size, bool random_length, int max_steps)
    : MiniGridTask("memory", max_steps, 7, false, 6),
      size_(size),
      random_length_(random_length) {}  // NOLINT(whitespace/indent_namespace)

Act MemoryTask::MapAction(Act act) const {
  return act == kPickup ? kToggle : act;
}

void MemoryTask::GenGrid() {
  ClearGrid(size_, size_);
  HorzWall(0, 0);
  HorzWall(0, size_ - 1);
  VertWall(0, 0);
  VertWall(size_ - 1, 0);
  CHECK_EQ(size_ % 2, 1);
  int upper_room_wall = size_ / 2 - 2;
  int lower_room_wall = size_ / 2 + 2;
  int hallway_end = random_length_ ? RandInt(4, size_ - 2) : size_ - 3;
  for (int i = 1; i < 5; ++i) {
    PutObj(WorldObj(kWall, kGrey), i, upper_room_wall);
    PutObj(WorldObj(kWall, kGrey), i, lower_room_wall);
  }
  PutObj(WorldObj(kWall, kGrey), 4, upper_room_wall + 1);
  PutObj(WorldObj(kWall, kGrey), 4, lower_room_wall - 1);
  for (int i = 5; i < hallway_end; ++i) {
    PutObj(WorldObj(kWall, kGrey), i, upper_room_wall + 1);
    PutObj(WorldObj(kWall, kGrey), i, lower_room_wall - 1);
  }
  for (int j = 0; j < size_; ++j) {
    if (j != size_ / 2) {
      PutObj(WorldObj(kWall, kGrey), hallway_end, j);
    }
    PutObj(WorldObj(kWall, kGrey), hallway_end + 2, j);
  }
  agent_pos_ = {RandInt(1, hallway_end + 1), size_ / 2};
  agent_dir_ = 0;
  Type start_type = RandBool() ? kKey : kBall;
  Type other_type = OtherKeyBallType(start_type);
  PutObj(WorldObj(start_type, kGreen), 1, size_ / 2 - 1);
  bool first_matches = RandBool();
  Type first_type = first_matches ? start_type : other_type;
  Type second_type = first_matches ? other_type : start_type;
  Pos pos0{hallway_end + 1, size_ / 2 - 2};
  Pos pos1{hallway_end + 1, size_ / 2 + 2};
  PutObj(WorldObj(first_type, kGreen), pos0.first, pos0.second);
  PutObj(WorldObj(second_type, kGreen), pos1.first, pos1.second);
  if (start_type == first_type) {
    success_pos_ = {pos0.first, pos0.second + 1};
    failure_pos_ = {pos1.first, pos1.second - 1};
  } else {
    success_pos_ = {pos1.first, pos1.second - 1};
    failure_pos_ = {pos0.first, pos0.second + 1};
  }
  SetMission("go to the matching object at the end of the hallway", 0);
}

void MemoryTask::AfterStep(Act act, const WorldObj& pre_fwd, const Pos& fwd_pos,
                           const WorldObj& pre_carrying, float* reward,
                           bool* terminated) {
  if (agent_pos_ == success_pos_) {
    *reward = 1.0f - 0.9f * (static_cast<float>(step_count_) / max_steps_);
    *terminated = true;
  } else if (agent_pos_ == failure_pos_) {
    *reward = 0.0f;
    *terminated = true;
  }
}

MultiRoomTask::MultiRoomTask(int min_num_rooms, int max_num_rooms,
                             int max_room_size, int max_steps)
    : MiniGridTask("multi_room", max_steps, 7, false, 6),
      min_num_rooms_(min_num_rooms),
      max_num_rooms_(max_num_rooms),
      max_room_size_(max_room_size) {}  // NOLINT(whitespace/indent_namespace)

bool MultiRoomTask::PlaceRoom(int num_left, std::vector<MultiRoomDesc>* rooms,
                              int min_size, int max_size, int entry_wall,
                              const Pos& entry_door_pos) {
  int size_x = RandInt(min_size, max_size + 1);
  int size_y = RandInt(min_size, max_size + 1);
  int top_x = 0;
  int top_y = 0;
  if (rooms->empty()) {
    top_x = entry_door_pos.first;
    top_y = entry_door_pos.second;
  } else if (entry_wall == 0) {
    top_x = entry_door_pos.first - size_x + 1;
    top_y =
        RandInt(entry_door_pos.second - size_y + 2, entry_door_pos.second + 1);
  } else if (entry_wall == 1) {
    top_x =
        RandInt(entry_door_pos.first - size_x + 2, entry_door_pos.first + 1);
    top_y = entry_door_pos.second - size_y + 1;
  } else if (entry_wall == 2) {
    top_x = entry_door_pos.first;
    top_y =
        RandInt(entry_door_pos.second - size_y + 2, entry_door_pos.second + 1);
  } else {
    top_x =
        RandInt(entry_door_pos.first - size_x + 2, entry_door_pos.first + 1);
    top_y = entry_door_pos.second;
  }
  if (top_x < 0 || top_y < 0 || top_x + size_x > width_ ||
      top_y + size_y >= height_) {
    return false;
  }
  for (std::size_t idx = 0; idx + 1 < rooms->size(); ++idx) {
    const MultiRoomDesc& room = (*rooms)[idx];
    bool non_overlap = top_x + size_x < room.top.first ||
                       room.top.first + room.size.first <= top_x ||
                       top_y + size_y < room.top.second ||
                       room.top.second + room.size.second <= top_y;
    if (!non_overlap) {
      return false;
    }
  }
  rooms->push_back(
      {{top_x, top_y}, {size_x, size_y}, entry_door_pos, {-1, -1}});
  if (num_left == 1) {
    return true;
  }
  for (int itr = 0; itr < 8; ++itr) {
    std::vector<int> wall_set = {0, 1, 2, 3};
    wall_set.erase(std::remove(wall_set.begin(), wall_set.end(), entry_wall),
                   wall_set.end());
    int exit_wall = RandElem(wall_set);
    int next_entry_wall = (exit_wall + 2) % 4;
    Pos exit_door_pos;
    if (exit_wall == 0) {
      exit_door_pos = {top_x + size_x - 1, top_y + RandInt(1, size_y - 1)};
    } else if (exit_wall == 1) {
      exit_door_pos = {top_x + RandInt(1, size_x - 1), top_y + size_y - 1};
    } else if (exit_wall == 2) {
      exit_door_pos = {top_x, top_y + RandInt(1, size_y - 1)};
    } else {
      exit_door_pos = {top_x + RandInt(1, size_x - 1), top_y};
    }
    if (PlaceRoom(num_left - 1, rooms, min_size, max_size, next_entry_wall,
                  exit_door_pos)) {
      break;
    }
  }
  return true;
}

void MultiRoomTask::GenGrid() {
  ClearGrid(25, 25);
  int num_rooms = RandInt(min_num_rooms_, max_num_rooms_ + 1);
  std::vector<MultiRoomDesc> rooms;
  while (static_cast<int>(rooms.size()) < num_rooms) {
    std::vector<MultiRoomDesc> current;
    Pos entry{RandInt(0, width_ - 2), RandInt(0, width_ - 2)};
    PlaceRoom(num_rooms, &current, 4, max_room_size_, 2, entry);
    if (current.size() > rooms.size()) {
      rooms = current;
    }
  }
  Color prev_door_color = kUnassigned;
  for (std::size_t idx = 0; idx < rooms.size(); ++idx) {
    const auto& room = rooms[idx];
    for (int i = 0; i < room.size.first; ++i) {
      PutObj(WorldObj(kWall, kGrey), room.top.first + i, room.top.second);
      PutObj(WorldObj(kWall, kGrey), room.top.first + i,
             room.top.second + room.size.second - 1);
    }
    for (int j = 0; j < room.size.second; ++j) {
      PutObj(WorldObj(kWall, kGrey), room.top.first, room.top.second + j);
      PutObj(WorldObj(kWall, kGrey), room.top.first + room.size.first - 1,
             room.top.second + j);
    }
    if (idx > 0) {
      std::vector<Color> door_colors(kColors.begin(), kColors.end());
      if (prev_door_color != kUnassigned) {
        door_colors.erase(std::remove(door_colors.begin(), door_colors.end(),
                                      prev_door_color),
                          door_colors.end());
      }
      Color color = RandElem(door_colors);
      PutObj(MakeDoor(color, false, false), room.entry_door_pos.first,
             room.entry_door_pos.second);
      prev_door_color = color;
    }
  }
  PlaceAgent(rooms.front().top.first, rooms.front().top.second,
             rooms.front().size.first, rooms.front().size.second, true);
  goal_pos_ = PlaceObj(WorldObj(kGoal, kGreen), rooms.back().top.first,
                       rooms.back().top.second, rooms.back().size.first,
                       rooms.back().size.second);
  SetMission("traverse the rooms to get to the goal", 0);
}

FourRoomsTask::FourRoomsTask(int max_steps)
    : MiniGridTask("four_rooms", max_steps, 7, false, 6) {}

void FourRoomsTask::GenGrid() {
  ClearGrid(19, 19);
  HorzWall(0, 0);
  HorzWall(0, height_ - 1);
  VertWall(0, 0);
  VertWall(width_ - 1, 0);
  int room_w = width_ / 2;
  int room_h = height_ / 2;
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 2; ++i) {
      int x_l = i * room_w;
      int y_t = j * room_h;
      int x_r = x_l + room_w;
      int y_b = y_t + room_h;
      if (i + 1 < 2) {
        VertWall(x_r, y_t, room_h);
        SetEmpty(x_r, RandInt(y_t + 1, y_b));
      }
      if (j + 1 < 2) {
        HorzWall(x_l, y_b, room_w);
        SetEmpty(RandInt(x_l + 1, x_r), y_b);
      }
    }
  }
  PlaceAgent();
  goal_pos_ = PlaceObj(WorldObj(kGoal, kGreen));
  SetMission("reach the goal", 0);
}

PlaygroundTask::PlaygroundTask(int max_steps)
    : MiniGridTask("playground", max_steps, 7, false, 6) {}

void PlaygroundTask::GenGrid() {
  ClearGrid(19, 19);
  HorzWall(0, 0);
  HorzWall(0, height_ - 1);
  VertWall(0, 0);
  VertWall(width_ - 1, 0);
  int room_w = width_ / 3;
  int room_h = height_ / 3;
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      int x_l = i * room_w;
      int y_t = j * room_h;
      int x_r = x_l + room_w;
      int y_b = y_t + room_h;
      if (i + 1 < 3) {
        VertWall(x_r, y_t, room_h);
        Pos pos{x_r, RandInt(y_t + 1, y_b - 1)};
        PutObj(MakeDoor(RandColor(), false, false), pos.first, pos.second);
      }
      if (j + 1 < 3) {
        HorzWall(x_l, y_b, room_w);
        Pos pos{RandInt(x_l + 1, x_r - 1), y_b};
        PutObj(MakeDoor(RandColor(), false, false), pos.first, pos.second);
      }
    }
  }
  PlaceAgent();
  for (int i = 0; i < 12; ++i) {
    Type type =
        RandElem(std::vector<Type>(kObjectTypes.begin(), kObjectTypes.end()));
    PlaceObj(WorldObj(type, RandColor()));
  }
  SetMission("", 0);
}

UnlockTask::UnlockTask(int max_steps)
    : RoomGridTask("unlock", 6, 1, 2, max_steps, 7) {}

void UnlockTask::GenGrid() {
  RoomGridTask::GenGrid();
  target_pos_ = AddDoor(0, 0, 0, kUnassigned, true);
  target_type_ = kDoor;
  target_color_ = GetCell(target_pos_.first, target_pos_.second).GetColor();
  AddObject(0, 0, kKey, target_color_);
  PlaceAgentInRoom(0, 0);
  SetMission("open the door", 0);
}

void UnlockTask::AfterStep(Act act, const WorldObj& pre_fwd, const Pos& fwd_pos,
                           const WorldObj& pre_carrying, float* reward,
                           bool* terminated) {
  if (act == kToggle &&
      GetCell(target_pos_.first, target_pos_.second).GetDoorOpen()) {
    *reward = 1.0f - 0.9f * (static_cast<float>(step_count_) / max_steps_);
    *terminated = true;
  }
}

UnlockPickupTask::UnlockPickupTask(int max_steps)
    : RoomGridTask("unlock_pickup", 6, 1, 2, max_steps, 7) {}

void UnlockPickupTask::GenGrid() {
  RoomGridTask::GenGrid();
  auto obj = AddObject(1, 0, kBox, kUnassigned);
  target_pos_ = obj.first;
  target_type_ = obj.second.first;
  target_color_ = obj.second.second;
  Pos door_pos = AddDoor(0, 0, 0, kUnassigned, true);
  AddObject(0, 0, kKey, GetCell(door_pos.first, door_pos.second).GetColor());
  PlaceAgentInRoom(0, 0);
  SetMission(MissionPickUp(target_color_, target_type_),
             static_cast<int>(target_color_));
}

void UnlockPickupTask::AfterStep(Act act, const WorldObj& pre_fwd,
                                 const Pos& fwd_pos,
                                 const WorldObj& pre_carrying, float* reward,
                                 bool* terminated) {
  if (act == kPickup && carrying_.GetType() == target_type_ &&
      carrying_.GetColor() == target_color_) {
    *reward = 1.0f - 0.9f * (static_cast<float>(step_count_) / max_steps_);
    *terminated = true;
  }
}

BlockedUnlockPickupTask::BlockedUnlockPickupTask(int max_steps)
    : RoomGridTask("blocked_unlock_pickup", 6, 1, 2, max_steps, 7) {}

void BlockedUnlockPickupTask::GenGrid() {
  RoomGridTask::GenGrid();
  auto obj = AddObject(1, 0, kBox, kUnassigned);
  target_pos_ = obj.first;
  target_type_ = obj.second.first;
  target_color_ = obj.second.second;
  Pos door_pos = AddDoor(0, 0, 0, kUnassigned, true);
  Pos block_pos{door_pos.first - 1, door_pos.second};
  PutObj(WorldObj(kBall, RandColor()), block_pos.first, block_pos.second);
  AddObject(0, 0, kKey, GetCell(door_pos.first, door_pos.second).GetColor());
  PlaceAgentInRoom(0, 0);
  SetMission(MissionPickUp(target_color_, target_type_), 0);
}

void BlockedUnlockPickupTask::AfterStep(Act act, const WorldObj& pre_fwd,
                                        const Pos& fwd_pos,
                                        const WorldObj& pre_carrying,
                                        float* reward, bool* terminated) {
  if (act == kPickup && carrying_.GetType() == target_type_ &&
      carrying_.GetColor() == target_color_) {
    *reward = 1.0f - 0.9f * (static_cast<float>(step_count_) / max_steps_);
    *terminated = true;
  }
}

KeyCorridorTask::KeyCorridorTask(int num_rows, int room_size, Type obj_type,
                                 int max_steps)
    : RoomGridTask("key_corridor", room_size, num_rows, 3, max_steps, 7),
      obj_type_(obj_type) {}  // NOLINT(whitespace/indent_namespace)

void KeyCorridorTask::GenGrid() {
  RoomGridTask::GenGrid();
  for (int j = 1; j < num_rows_; ++j) {
    RemoveWall(1, j, 3);
  }
  int room_idx = RandInt(0, num_rows_);
  Pos door_pos = AddDoor(2, room_idx, 2, kUnassigned, true);
  auto obj = AddObject(2, room_idx, obj_type_, kUnassigned);
  target_pos_ = obj.first;
  target_type_ = obj.second.first;
  target_color_ = obj.second.second;
  AddObject(0, RandInt(0, num_rows_), kKey,
            GetCell(door_pos.first, door_pos.second).GetColor());
  PlaceAgentInRoom(1, num_rows_ / 2);
  ConnectAll();
  SetMission(MissionPickUp(target_color_, target_type_), 0);
}

void KeyCorridorTask::AfterStep(Act act, const WorldObj& pre_fwd,
                                const Pos& fwd_pos,
                                const WorldObj& pre_carrying, float* reward,
                                bool* terminated) {
  if (act == kPickup && carrying_.GetType() == target_type_ &&
      carrying_.GetColor() == target_color_) {
    *reward = 1.0f - 0.9f * (static_cast<float>(step_count_) / max_steps_);
    *terminated = true;
  }
}

ObstructedMazeTask::ObstructedMazeTask(std::string env_name, Pos agent_room,
                                       bool key_in_box, bool blocked,
                                       int num_quarters, int max_steps, bool v1)
    : RoomGridTask(std::move(env_name), 6, 3, 3, max_steps, 7),
      agent_room_(std::move(agent_room)),
      key_in_box_(key_in_box),
      blocked_(blocked),
      num_quarters_(num_quarters),
      v1_(v1) {}  // NOLINT(whitespace/indent_namespace)

void ObstructedMazeTask::AddKeyToRoom(int i, int j, Color color,
                                      bool key_in_box) {
  WorldObj obj(kKey, color);
  if (key_in_box) {
    WorldObj box(kBox, kBlue);
    box.SetContains(std::make_unique<WorldObj>(obj));
    PlaceObj(
        box, GetRoom(i, j).top.first, GetRoom(i, j).top.second,
        GetRoom(i, j).size.first, GetRoom(i, j).size.second,
        [&](const Pos& pos_candidate) {
          return Manhattan(agent_pos_, pos_candidate) < 2;
        },
        1000);
  } else {
    AddObject(i, j, kKey, color);
  }
}

void ObstructedMazeTask::AddObstructedDoor(int i, int j, int door_idx,
                                           Color color, bool locked,
                                           bool key_in_box, bool blocked,
                                           bool add_key) {
  Pos door_pos = AddDoor(i, j, door_idx, color, locked);
  if (blocked) {
    Pos vec = kDirToVec[door_idx];
    PutObj(WorldObj(kBall, kGreen), door_pos.first - vec.first,
           door_pos.second - vec.second);
  }
  if (locked && add_key) {
    AddKeyToRoom(i, j, color, key_in_box);
  }
}

void ObstructedMazeTask::GenGrid() {
  RoomGridTask::GenGrid();
  door_colors_ = RandSubset(std::vector<Color>(kColors.begin(), kColors.end()),
                            static_cast<int>(kColors.size()));
  SetMission("pick up the blue ball", 0);
  if (env_name_ == "obstructed_maze_1dlhb") {
    AddObstructedDoor(0, 0, 0, door_colors_[0], true, key_in_box_, blocked_,
                      true);
    auto obj = AddObject(1, 0, kBall, kBlue);
    target_pos_ = obj.first;
    target_type_ = obj.second.first;
    target_color_ = obj.second.second;
    PlaceAgentInRoom(0, 0);
    return;
  }
  Pos middle_room{1, 1};
  std::vector<Pos> side_rooms = {{2, 1}, {1, 2}, {0, 1}, {1, 0}};
  side_rooms.resize(num_quarters_);
  for (int i = 0; i < static_cast<int>(side_rooms.size()); ++i) {
    Pos side = side_rooms[i];
    AddDoor(middle_room.first, middle_room.second, i, door_colors_[i], false);
    for (int sign : std::vector<int>{-1, 1}) {
      int door_idx = (i + sign + 4) % 4;
      Color color =
          door_colors_[door_idx % static_cast<int>(door_colors_.size())];
      if (v1_) {
        AddObstructedDoor(side.first, side.second, door_idx, color, true,
                          key_in_box_, blocked_, false);
      } else {
        AddObstructedDoor(side.first, side.second, door_idx, color, true,
                          key_in_box_, blocked_, true);
      }
    }
    if (v1_) {
      for (int sign : std::vector<int>{-1, 1}) {
        int door_idx = (i + sign + 4) % 4;
        Color color =
            door_colors_[door_idx % static_cast<int>(door_colors_.size())];
        AddKeyToRoom(side.first, side.second, color, key_in_box_);
      }
    }
  }
  std::vector<Pos> corners = {{2, 0}, {2, 2}, {0, 2}, {0, 0}};
  corners.resize(num_quarters_);
  Pos ball_room = RandElem(corners);
  auto obj = AddObject(ball_room.first, ball_room.second, kBall, kBlue);
  target_pos_ = obj.first;
  target_type_ = obj.second.first;
  target_color_ = obj.second.second;
  PlaceAgentInRoom(agent_room_.first, agent_room_.second);
}

void ObstructedMazeTask::AfterStep(Act act, const WorldObj& pre_fwd,
                                   const Pos& fwd_pos,
                                   const WorldObj& pre_carrying, float* reward,
                                   bool* terminated) {
  if (act == kPickup && carrying_.GetType() == target_type_ &&
      carrying_.GetColor() == target_color_) {
    *reward = 1.0f - 0.9f * (static_cast<float>(step_count_) / max_steps_);
    *terminated = true;
  }
}

MiniGridEnv::MiniGridEnv(const Spec& spec, int env_id)
    : Env<MiniGridEnvSpec>(spec, env_id) {
  const auto& conf = spec.config;
  const std::string env_name = conf["env_name"_];
  if (env_name == "empty") {
    task_ = std::make_unique<EmptyTask>(
        conf["size"_], conf["agent_start_pos"_], conf["agent_start_dir"_],
        conf["max_episode_steps"_], conf["agent_view_size"_]);
  } else if (env_name == "doorkey") {
    task_ = std::make_unique<DoorKeyTask>(conf["size"_],
                                          conf["max_episode_steps"_]);
  } else if (env_name == "distshift") {
    task_ = std::make_unique<DistShiftTask>(
        conf["width"_], conf["height"_], conf["agent_start_pos"_],
        conf["agent_start_dir"_], conf["strip2_row"_],
        conf["max_episode_steps"_]);
  } else if (env_name == "lava_gap") {
    task_ = std::make_unique<LavaGapTask>(conf["size"_],
                                          ParseType(conf["obstacle_type"_]),
                                          conf["max_episode_steps"_]);
  } else if (env_name == "crossing") {
    task_ = std::make_unique<CrossingTask>(
        conf["size"_], conf["num_crossings"_],
        ParseType(conf["obstacle_type"_]), conf["max_episode_steps"_]);
  } else if (env_name == "dynamic_obstacles") {
    task_ = std::make_unique<DynamicObstaclesTask>(
        conf["size"_], conf["agent_start_pos"_], conf["agent_start_dir"_],
        conf["n_obstacles"_], conf["max_episode_steps"_]);
  } else if (env_name == "fetch") {
    task_ = std::make_unique<FetchTask>(conf["size"_], conf["num_objs"_],
                                        conf["max_episode_steps"_]);
  } else if (env_name == "goto_door") {
    task_ = std::make_unique<GoToDoorTask>(conf["size"_],
                                           conf["max_episode_steps"_]);
  } else if (env_name == "goto_object") {
    task_ = std::make_unique<GoToObjectTask>(conf["size"_], conf["num_objs"_],
                                             conf["max_episode_steps"_]);
  } else if (env_name == "put_near") {
    task_ = std::make_unique<PutNearTask>(conf["size"_], conf["num_objs"_],
                                          conf["max_episode_steps"_]);
  } else if (env_name == "red_blue_door") {
    task_ = std::make_unique<RedBlueDoorTask>(conf["size"_],
                                              conf["max_episode_steps"_]);
  } else if (env_name == "locked_room") {
    task_ = std::make_unique<LockedRoomTask>(conf["size"_],
                                             conf["max_episode_steps"_]);
  } else if (env_name == "memory") {
    task_ = std::make_unique<MemoryTask>(conf["size"_], conf["random_length"_],
                                         conf["max_episode_steps"_]);
  } else if (env_name == "multi_room") {
    task_ = std::make_unique<MultiRoomTask>(
        conf["min_num_rooms"_], conf["max_num_rooms"_], conf["max_room_size"_],
        conf["max_episode_steps"_]);
  } else if (env_name == "four_rooms") {
    task_ = std::make_unique<FourRoomsTask>(conf["max_episode_steps"_]);
  } else if (env_name == "playground") {
    task_ = std::make_unique<PlaygroundTask>(conf["max_episode_steps"_]);
  } else if (env_name == "unlock") {
    task_ = std::make_unique<UnlockTask>(conf["max_episode_steps"_]);
  } else if (env_name == "unlock_pickup") {
    task_ = std::make_unique<UnlockPickupTask>(conf["max_episode_steps"_]);
  } else if (env_name == "blocked_unlock_pickup") {
    task_ =
        std::make_unique<BlockedUnlockPickupTask>(conf["max_episode_steps"_]);
  } else if (env_name == "key_corridor") {
    task_ = std::make_unique<KeyCorridorTask>(
        conf["num_rows"_], conf["room_size"_], ParseType(conf["obj_type"_]),
        conf["max_episode_steps"_]);
  } else {
    CHECK(env_name == "obstructed_maze_1dlhb" ||
          env_name == "obstructed_maze_full" ||
          env_name == "obstructed_maze_full_v1")
        << "Unknown MiniGrid env_name: " << env_name;
    task_ = std::make_unique<
        ObstructedMazeTask>(  // NOLINT(whitespace/indent_namespace)
        env_name, conf["agent_room"_], conf["key_in_box"_], conf["blocked"_],
        conf["num_quarters"_], conf["max_episode_steps"_],
        env_name == "obstructed_maze_full_v1");
  }
  task_->SetGenerator(&gen_);
}

bool MiniGridEnv::IsDone() { return task_->IsDone(); }

void MiniGridEnv::Reset() {
  task_->Reset();
  WriteState(0.0f);
}

void MiniGridEnv::Step(const Action& action) {
  WriteState(task_->Step(static_cast<Act>(action["action"_])));
}

std::pair<int, int> MiniGridEnv::RenderSize(int width, int height) const {
  return task_->RenderSize(width, height);
}

void MiniGridEnv::Render(int width, int height, int /*camera_id*/,
                         unsigned char* rgb) {
  task_->Render(width, height, rgb);
}

MiniGridDebugState MiniGridEnv::DebugState() const {
  return task_->DebugState();
}

void MiniGridEnv::WriteState(float reward) {
  auto state = Allocate();
  task_->GenImage(state["obs:image"_]);
  task_->WriteMission(state["obs:mission"_]);
  state["obs:direction"_] = task_->AgentDir();
  state["reward"_] = reward;
  state["info:agent_pos"_](0) = task_->AgentPos().first;
  state["info:agent_pos"_](1) = task_->AgentPos().second;
  state["info:mission_id"_] = task_->MissionId();
}

std::vector<MiniGridDebugState> MiniGridEnvPool::DebugStates(
    const std::vector<int>& env_ids) const {
  std::vector<MiniGridDebugState> states;
  states.reserve(env_ids.size());
  for (int env_id : env_ids) {
    states.push_back(envs_[env_id]->DebugState());
  }
  return states;
}

using PyMiniGridEnvSpec = PyEnvSpec<MiniGridEnvSpec>;
using PyMiniGridEnvPool = PyEnvPool<MiniGridEnvPool>;

void BindMiniGrid(py::module_& m) {
  py::class_<MiniGridDebugState>(m, "_MiniGridDebugState")
      .def_readonly("env_name", &MiniGridDebugState::env_name)
      .def_readonly("mission", &MiniGridDebugState::mission)
      .def_readonly("mission_id", &MiniGridDebugState::mission_id)
      .def_readonly("width", &MiniGridDebugState::width)
      .def_readonly("height", &MiniGridDebugState::height)
      .def_readonly("action_max", &MiniGridDebugState::action_max)
      .def_readonly("grid", &MiniGridDebugState::grid)
      .def_readonly("grid_contains", &MiniGridDebugState::grid_contains)
      .def_readonly("obstacle_positions",
                    &MiniGridDebugState::obstacle_positions)
      .def_readonly("agent_pos", &MiniGridDebugState::agent_pos)
      .def_readonly("agent_dir", &MiniGridDebugState::agent_dir)
      .def_readonly("has_carrying", &MiniGridDebugState::has_carrying)
      .def_readonly("carrying_type", &MiniGridDebugState::carrying_type)
      .def_readonly("carrying_color", &MiniGridDebugState::carrying_color)
      .def_readonly("carrying_state", &MiniGridDebugState::carrying_state)
      .def_readonly("carrying_has_contains",
                    &MiniGridDebugState::carrying_has_contains)
      .def_readonly("carrying_contains_type",
                    &MiniGridDebugState::carrying_contains_type)
      .def_readonly("carrying_contains_color",
                    &MiniGridDebugState::carrying_contains_color)
      .def_readonly("carrying_contains_state",
                    &MiniGridDebugState::carrying_contains_state)
      .def_readonly("target_pos", &MiniGridDebugState::target_pos)
      .def_readonly("target_type", &MiniGridDebugState::target_type)
      .def_readonly("target_color", &MiniGridDebugState::target_color)
      .def_readonly("move_pos", &MiniGridDebugState::move_pos)
      .def_readonly("move_type", &MiniGridDebugState::move_type)
      .def_readonly("move_color", &MiniGridDebugState::move_color)
      .def_readonly("success_pos", &MiniGridDebugState::success_pos)
      .def_readonly("failure_pos", &MiniGridDebugState::failure_pos)
      .def_readonly("goal_pos", &MiniGridDebugState::goal_pos);

  py::class_<PyMiniGridEnvSpec>(
      m, "_MiniGridEnvSpec",
      py::metaclass(py::module_::import("abc").attr("ABCMeta")))
      .def(py::init<const PyMiniGridEnvSpec::ConfigValues&>())
      .def_readonly("_config_values", &PyMiniGridEnvSpec::py_config_values)
      .def_readonly("_state_spec", &PyMiniGridEnvSpec::py_state_spec)
      .def_readonly("_action_spec", &PyMiniGridEnvSpec::py_action_spec)
      .def_readonly_static("_state_keys", &PyMiniGridEnvSpec::py_state_keys)
      .def_readonly_static("_action_keys", &PyMiniGridEnvSpec::py_action_keys)
      .def_readonly_static("_config_keys", &PyMiniGridEnvSpec::py_config_keys)
      .def_readonly_static("_default_config_values",
                           &PyMiniGridEnvSpec::py_default_config_values);

  py::class_<PyMiniGridEnvPool>(
      m, "_MiniGridEnvPool",
      py::metaclass(py::module_::import("abc").attr("ABCMeta")))
      .def(py::init<const PyMiniGridEnvSpec&>())
      .def_readonly("_spec", &PyMiniGridEnvPool::py_spec)
      .def("_recv", &PyMiniGridEnvPool::PyRecv)
      .def("_send", &PyMiniGridEnvPool::PySend)
      .def("_reset", &PyMiniGridEnvPool::PyReset)
      .def("_render", &PyMiniGridEnvPool::PyRender)
      .def_readonly_static("_state_keys", &PyMiniGridEnvPool::py_state_keys)
      .def_readonly_static("_action_keys", &PyMiniGridEnvPool::py_action_keys)
      .def("_xla", &PyMiniGridEnvPool::Xla)
      .def("_debug_states", &PyMiniGridEnvPool::DebugStates);
}

}  // namespace minigrid
