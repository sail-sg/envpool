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

#ifndef ENVPOOL_MARLGRID_MARLGRID_H_
#define ENVPOOL_MARLGRID_MARLGRID_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace marlgrid {
namespace detail {

enum Act : std::uint8_t {
  kLeft = 0,
  kRight = 1,
  kForward = 2,
  kPickup = 3,
  kDrop = 4,
  kToggle = 5,
  kDone = 6,
};

enum class CellType : std::uint8_t {
  kEmpty = 0,
  kWall = 1,
  kGoal = 2,
  kBonus = 3,
  kLava = 4,
};

enum class AgentColor : std::uint8_t {
  kRed = 0,
  kBlue = 1,
  kPurple = 2,
  kOrange = 3,
  kOlive = 4,
  kPink = 5,
};

enum class MatrixObsChannel : std::uint8_t {
  kEmpty = 0,
  kWall = 1,
  kGoal = 2,
  kBonus = 3,
  kLava = 4,
  kAgent = 5,
  kAgentRed = 6,
  kAgentGreen = 7,
  kAgentBlue = 8,
};

using Rgb = std::array<std::uint8_t, 3>;

inline constexpr int kTilePixels = 32;
inline constexpr int kTileSubdivs = 3;
inline constexpr int kMatrixObsChannels = 9;
inline constexpr double kPi = 3.14159265358979323846;
inline constexpr Rgb kShadowColor = {35, 25, 30};
inline constexpr Rgb kWorstColor = {74, 65, 42};
inline constexpr std::array<std::pair<int, int>, 4> kDirToVec = {
    std::pair<int, int>{1, 0},
    std::pair<int, int>{0, 1},
    std::pair<int, int>{-1, 0},
    std::pair<int, int>{0, -1},
};

inline Rgb ColorValue(AgentColor color) {
  static constexpr std::array<Rgb, 6> k_colors = {
      Rgb{255, 0, 0},   Rgb{0, 0, 255},   Rgb{112, 39, 195},
      Rgb{255, 165, 0}, Rgb{128, 128, 0}, Rgb{255, 0, 189},
  };
  return k_colors[static_cast<int>(color)];
}

inline Rgb PrestigeColor(float prestige, float prestige_scale) {
  double prestige_scaled = std::tanh(static_cast<double>(prestige) /
                                     static_cast<double>(prestige_scale));
  prestige_scaled = std::clamp(prestige_scaled, 0.0, 1.0);
  return Rgb{
      static_cast<std::uint8_t>((1.0 - prestige_scaled) * 255.0),
      0,
      static_cast<std::uint8_t>(prestige_scaled * 255.0),
  };
}

inline AgentColor AgentColorByIndex(int index) {
  static constexpr std::array<AgentColor, 6> k_colors = {
      AgentColor::kRed,    AgentColor::kBlue,  AgentColor::kPurple,
      AgentColor::kOrange, AgentColor::kOlive, AgentColor::kPink,
  };
  CHECK_GE(index, 0);
  CHECK_LT(index, static_cast<int>(k_colors.size()));
  return k_colors[index];
}

struct Cell {
  CellType type{CellType::kEmpty};
  int bonus_id{0};
  std::vector<int> agents;

  [[nodiscard]] bool CanOverlap() const { return type != CellType::kWall; }

  [[nodiscard]] bool CanSeeBehind() const { return type != CellType::kWall; }
};

struct Agent {
  AgentColor color{AgentColor::kRed};
  int x{-1};
  int y{-1};
  int dir{0};
  bool active{false};
  bool done{false};
  int bonus_state{-1};
  float step_reward{0.0f};
  float prestige{0.0f};
};

inline int Offset(int x, int y, int width) { return y * width + x; }

using CoordFn = std::function<bool(double, double)>;

inline CoordFn PointInRect(double xmin, double xmax, double ymin, double ymax) {
  return [=](double x, double y) {
    return x >= xmin && x <= xmax && y >= ymin && y <= ymax;
  };
}

inline CoordFn PointInCircle(double cx, double cy, double r) {
  return [=](double x, double y) {
    double dx = x - cx;
    double dy = y - cy;
    return dx * dx + dy * dy <= r * r;
  };
}

inline CoordFn PointInTriangle(const std::array<double, 2>& a,
                               const std::array<double, 2>& b,
                               const std::array<double, 2>& c) {
  return [=](double x, double y) {
    std::array<double, 2> v0 = {c[0] - a[0], c[1] - a[1]};
    std::array<double, 2> v1 = {b[0] - a[0], b[1] - a[1]};
    std::array<double, 2> v2 = {x - a[0], y - a[1]};
    double dot00 = v0[0] * v0[0] + v0[1] * v0[1];
    double dot01 = v0[0] * v1[0] + v0[1] * v1[1];
    double dot02 = v0[0] * v2[0] + v0[1] * v2[1];
    double dot11 = v1[0] * v1[0] + v1[1] * v1[1];
    double dot12 = v1[0] * v2[0] + v1[1] * v2[1];
    double inv = 1.0 / (dot00 * dot11 - dot01 * dot01);
    double u = (dot11 * dot02 - dot01 * dot12) * inv;
    double v = (dot00 * dot12 - dot01 * dot02) * inv;
    return u >= 0.0f && v >= 0.0f && u + v < 1.0f;
  };
}

inline CoordFn RotateFn(const CoordFn& fn, double cx, double cy, double theta) {
  return [=](double x, double y) {
    double centered_x = x - cx;
    double centered_y = y - cy;
    double x2 =
        cx + centered_x * std::cos(-theta) - centered_y * std::sin(-theta);
    double y2 =
        cy + centered_y * std::cos(-theta) + centered_x * std::sin(-theta);
    return fn(x2, y2);
  };
}

inline void FillCoords(std::vector<std::uint8_t>* img, int width, int height,
                       const CoordFn& fn, const Rgb& color) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      double yf = (static_cast<double>(y) + 0.5) / height;
      double xf = (static_cast<double>(x) + 0.5) / width;
      if (!fn(xf, yf)) {
        continue;
      }
      int offset = (y * width + x) * 3;
      (*img)[offset + 0] = color[0];
      (*img)[offset + 1] = color[1];
      (*img)[offset + 2] = color[2];
    }
  }
}

inline std::vector<std::uint8_t> Downsample(
    const std::vector<std::uint8_t>& src, int width, int height, int factor) {
  int out_width = width / factor;
  int out_height = height / factor;
  std::vector<std::uint8_t> out(out_width * out_height * 3, 0);
  for (int y = 0; y < out_height; ++y) {
    for (int x = 0; x < out_width; ++x) {
      for (int c = 0; c < 3; ++c) {
        int sum = 0;
        for (int dy = 0; dy < factor; ++dy) {
          for (int dx = 0; dx < factor; ++dx) {
            int sx = x * factor + dx;
            int sy = y * factor + dy;
            sum += src[(sy * width + sx) * 3 + c];
          }
        }
        out[(y * out_width + x) * 3 + c] =
            static_cast<std::uint8_t>(sum / (factor * factor));
      }
    }
  }
  return out;
}

inline std::vector<std::uint8_t> EmptyTile(int tile_size) {
  int alpha = std::max(0, std::min(20, tile_size - 10));
  std::vector<std::uint8_t> img(tile_size * tile_size * 3, alpha);
  for (int y = 1; y < tile_size; ++y) {
    for (int x = 0; x < tile_size - 1; ++x) {
      int offset = (y * tile_size + x) * 3;
      img[offset + 0] = 0;
      img[offset + 1] = 0;
      img[offset + 2] = 0;
    }
  }
  return img;
}

inline std::vector<std::uint8_t> RenderAgentTile(const Rgb& color, int dir,
                                                 int tile_size) {
  int hi_width = tile_size * kTileSubdivs;
  int hi_height = tile_size * kTileSubdivs;
  std::vector<std::uint8_t> img(hi_width * hi_height * 3, 0);
  auto tri = PointInTriangle({0.12f, 0.19f}, {0.87f, 0.50f}, {0.12f, 0.81f});
  tri = RotateFn(tri, 0.5f, 0.5f, 0.5f * kPi * dir);
  FillCoords(&img, hi_width, hi_height, tri, color);
  return Downsample(img, hi_width, hi_height, kTileSubdivs);
}

inline std::vector<std::uint8_t> RenderAgentTile(AgentColor color, int dir,
                                                 int tile_size) {
  return RenderAgentTile(ColorValue(color), dir, tile_size);
}

inline std::vector<std::uint8_t> RenderObjectTile(CellType type,
                                                  int tile_size) {
  if (type == CellType::kEmpty) {
    return EmptyTile(tile_size);
  }
  int hi_width = tile_size * kTileSubdivs;
  int hi_height = tile_size * kTileSubdivs;
  std::vector<std::uint8_t> img(hi_width * hi_height * 3, 0);
  if (type == CellType::kWall) {
    FillCoords(&img, hi_width, hi_height, PointInRect(0.0f, 1.0f, 0.0f, 1.0f),
               kWorstColor);
  } else if (type == CellType::kGoal) {
    FillCoords(&img, hi_width, hi_height, PointInRect(0.0f, 1.0f, 0.0f, 1.0f),
               Rgb{0, 255, 0});
  } else if (type == CellType::kBonus) {
    FillCoords(&img, hi_width, hi_height, PointInRect(0.0f, 1.0f, 0.0f, 1.0f),
               Rgb{255, 255, 0});
  } else if (type == CellType::kLava) {
    FillCoords(&img, hi_width, hi_height, PointInRect(0.0f, 1.0f, 0.0f, 1.0f),
               Rgb{255, 128, 0});
  }
  return Downsample(img, hi_width, hi_height, kTileSubdivs);
}

inline std::vector<std::uint8_t> BlendTiles(
    const std::vector<std::uint8_t>& base,
    const std::vector<std::uint8_t>& overlay, int tile_size) {
  int max_alpha = 0;
  std::vector<int> alpha(tile_size * tile_size, 0);
  for (int i = 0; i < tile_size * tile_size; ++i) {
    int value = overlay[i * 3 + 0] + overlay[i * 3 + 1] + overlay[i * 3 + 2];
    alpha[i] = value;
    max_alpha = std::max(max_alpha, value);
  }
  if (max_alpha == 0) {
    return base;
  }
  std::vector<std::uint8_t> out(base.size(), 0);
  for (int i = 0; i < tile_size * tile_size; ++i) {
    for (int c = 0; c < 3; ++c) {
      int offset = i * 3 + c;
      out[offset] = static_cast<std::uint8_t>(
          (base[offset] * (max_alpha - alpha[i]) + overlay[offset] * alpha[i]) /
          max_alpha);
    }
  }
  return out;
}

inline bool HasBlackCorner(const std::vector<std::uint8_t>& img,
                           int tile_size) {
  const std::array<int, 4> corners = {
      0,
      tile_size - 1,
      (tile_size - 1) * tile_size,
      tile_size * tile_size - 1,
  };
  return std::any_of(corners.begin(), corners.end(), [&](int corner) {
    int offset = corner * 3;
    return img[offset + 0] == 0 && img[offset + 1] == 0 && img[offset + 2] == 0;
  });
}

inline void AddTile(std::vector<std::uint8_t>* img,
                    const std::vector<std::uint8_t>& add) {
  for (std::size_t i = 0; i < img->size(); ++i) {
    (*img)[i] = static_cast<std::uint8_t>(
        std::min(255, static_cast<int>((*img)[i]) + static_cast<int>(add[i])));
  }
}

inline std::pair<int, int> RotateCoord(int x, int y, int size, int rot_k) {
  rot_k = ((rot_k % 4) + 4) % 4;
  if (rot_k == 1) {
    return {size - 1 - y, x};
  }
  if (rot_k == 2) {
    return {size - 1 - x, size - 1 - y};
  }
  if (rot_k == 3) {
    return {y, size - 1 - x};
  }
  return {x, y};
}

inline std::vector<std::uint8_t> RotateImage(
    const std::vector<std::uint8_t>& src, int width, int height, int rot_k) {
  rot_k = ((rot_k % 4) + 4) % 4;
  if (rot_k == 0) {
    return src;
  }
  int out_width = (rot_k % 2 == 0) ? width : height;
  int out_height = (rot_k % 2 == 0) ? height : width;
  std::vector<std::uint8_t> out(out_width * out_height * 3, 0);
  for (int y = 0; y < out_height; ++y) {
    for (int x = 0; x < out_width; ++x) {
      int sx = x;
      int sy = y;
      if (rot_k == 1) {
        sx = y;
        sy = height - 1 - x;
      } else if (rot_k == 2) {
        sx = width - 1 - x;
        sy = height - 1 - y;
      } else if (rot_k == 3) {
        sx = width - 1 - y;
        sy = x;
      }
      std::memcpy(out.data() + (y * out_width + x) * 3,
                  src.data() + (sy * width + sx) * 3, 3);
    }
  }
  return out;
}

}  // namespace detail

class MarlGridEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("env_name"_.Bind(std::string("empty")), "n_agents"_.Bind(2),
                    "grid_size"_.Bind(9), "view_size"_.Bind(7),
                    "view_tile_size"_.Bind(8), "view_offset"_.Bind(0),
                    "n_clutter"_.Bind(0), "randomize_goal"_.Bind(false),
                    "n_bonus_tiles"_.Bind(3), "bonus_reward"_.Bind(1.0f),
                    "bonus_penalty"_.Bind(0.0f), "initial_reward"_.Bind(true),
                    "reset_on_mistake"_.Bind(false), "reward_decay"_.Bind(true),
                    "respawn"_.Bind(false), "ghost_mode"_.Bind(true),
                    "prestige_coloring"_.Bind(false),
                    "prestige_beta"_.Bind(0.95f), "prestige_scale"_.Bind(2.0f),
                    "observation_format"_.Bind(std::string("pixels")));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const std::string observation_format = conf["observation_format"_];
    if (observation_format != "pixels" && observation_format != "matrix" &&
        observation_format != "full_matrix") {
      throw std::runtime_error(
          "MarlGrid observation_format must be 'pixels', 'matrix', or "
          "'full_matrix'");
    }
    const bool matrix_observation = observation_format == "matrix";
    const bool full_matrix_observation = observation_format == "full_matrix";
    int obs_size = full_matrix_observation
                       ? conf["grid_size"_]
                       : (matrix_observation
                              ? conf["view_size"_]
                              : conf["view_size"_] * conf["view_tile_size"_]);
    int obs_channels = (matrix_observation || full_matrix_observation)
                           ? detail::kMatrixObsChannels
                           : 3;
    int bound = conf["grid_size"_];
    return MakeDict(
        "obs"_.Bind(Spec<std::uint8_t>({-1, obs_size, obs_size, obs_channels},
                                       {0, 255})),
        "info:players.id"_.Bind(Spec<int>({-1}, {0, conf["max_num_players"_]})),
        "info:players.done"_.Bind(Spec<bool>({-1})),
        "info:players.active"_.Bind(Spec<bool>({-1})),
        "info:players.pos"_.Bind(Spec<int>({-1, 2}, {-1, bound})),
        "info:players.dir"_.Bind(Spec<int>({-1}, {0, 3})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& /*conf*/) {
    return MakeDict("players.action"_.Bind(Spec<int>({-1}, {0, 6})));
  }
};

using MarlGridEnvSpec = EnvSpec<MarlGridEnvFns>;

class MarlGridEnv : public Env<MarlGridEnvSpec>, public RenderableEnv {
 public:
  MarlGridEnv(const Spec& spec, int env_id)
      : Env<MarlGridEnvSpec>(spec, env_id),
        env_name_(spec.config["env_name"_]),
        n_agents_(spec.config["n_agents"_]),
        grid_size_(spec.config["grid_size"_]),
        view_size_(spec.config["view_size"_]),
        view_tile_size_(spec.config["view_tile_size"_]),
        view_offset_(spec.config["view_offset"_]),
        n_clutter_(spec.config["n_clutter"_]),
        randomize_goal_(spec.config["randomize_goal"_]),
        n_bonus_tiles_(spec.config["n_bonus_tiles"_]),
        bonus_reward_(spec.config["bonus_reward"_]),
        bonus_penalty_(spec.config["bonus_penalty"_]),
        initial_reward_(spec.config["initial_reward"_]),
        reset_on_mistake_(spec.config["reset_on_mistake"_]),
        reward_decay_(spec.config["reward_decay"_]),
        respawn_(spec.config["respawn"_]),
        ghost_mode_(spec.config["ghost_mode"_]),
        prestige_coloring_(spec.config["prestige_coloring"_]),
        prestige_beta_(spec.config["prestige_beta"_]),
        prestige_scale_(spec.config["prestige_scale"_]),
        observation_format_(spec.config["observation_format"_]),
        max_episode_steps_(spec.config["max_episode_steps"_]) {
    CHECK_GE(max_num_players_, n_agents_);
    CHECK_GE(grid_size_, 3);
    CHECK_GE(view_size_, 3);
    CHECK_LE(n_agents_, 6);
    CHECK_GE(prestige_beta_, 0.0f);
    CHECK_LE(prestige_beta_, 1.0f);
    CHECK_GT(prestige_scale_, 0.0f);
    if (observation_format_ != "pixels" && observation_format_ != "matrix" &&
        observation_format_ != "full_matrix") {
      throw std::runtime_error(
          "MarlGrid observation_format must be 'pixels', 'matrix', or "
          "'full_matrix'");
    }
    agents_.resize(n_agents_);
    for (int i = 0; i < n_agents_; ++i) {
      agents_[i].color = detail::AgentColorByIndex(i);
    }
    done_ = true;
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    step_count_ = 0;
    done_ = false;
    last_rewards_.assign(n_agents_, 0.0f);
    for (auto& agent : agents_) {
      agent.active = false;
      agent.done = false;
      agent.x = -1;
      agent.y = -1;
      agent.bonus_state = -1;
      agent.step_reward = 0.0f;
      agent.prestige = 0.0f;
    }
    GenGrid();
    for (int i = 0; i < n_agents_; ++i) {
      PlaceAgent(i);
      agents_[i].active = true;
    }
    WriteState();
  }

  void Step(const Action& action) override {
    int action_count = action["players.action"_].Shape(0);
    CHECK_EQ(action_count, n_agents_);
    last_rewards_.assign(n_agents_, 0.0f);
    for (auto& agent : agents_) {
      agent.step_reward = 0.0f;
    }
    for (auto& agent : agents_) {
      if (!agent.active && !agent.done) {
        PlaceAgent(static_cast<int>(&agent - agents_.data()));
        agent.active = true;
      }
    }
    ++step_count_;
    std::vector<int> order(n_agents_);
    for (int i = 0; i < n_agents_; ++i) {
      order[i] = i;
    }
    std::shuffle(order.begin(), order.end(), gen_);
    for (int agent_id : order) {
      DoAgentStep(agent_id, static_cast<detail::Act>(static_cast<int>(
                                action["players.action"_][agent_id])));
    }
    for (int i = 0; i < n_agents_; ++i) {
      if (agents_[i].done) {
        if (respawn_) {
          RemoveAgentFromCell(i);
          agents_[i].active = false;
          agents_[i].done = false;
          agents_[i].x = -1;
          agents_[i].y = -1;
          PlaceAgent(i);
          agents_[i].active = true;
        } else {
          agents_[i].active = false;
        }
      }
    }
    done_ = step_count_ >= max_episode_steps_ ||
            std::all_of(agents_.begin(), agents_.end(),
                        [](const detail::Agent& agent) { return agent.done; });
    WriteState();
  }

  [[nodiscard]] std::pair<int, int> RenderSize(int width,
                                               int height) const override {
    int native = grid_size_ * detail::kTilePixels;
    return {width > 0 ? width : native, height > 0 ? height : native};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    auto [render_width, render_height] = RenderSize(width, height);
    int native = grid_size_ * detail::kTilePixels;
    std::vector<std::uint8_t> native_img(native * native * 3, 0);
    RenderGrid(detail::kTilePixels, -1, 0, native_img.data());
    ResizeNearest(native_img.data(), native, native, rgb, render_width,
                  render_height);
  }

 protected:
  [[nodiscard]] int CurrentMaxEpisodeSteps() const override {
    return max_episode_steps_;
  }

 private:
  std::string env_name_;
  int n_agents_{0};
  int grid_size_{0};
  int view_size_{7};
  int view_tile_size_{8};
  int view_offset_{0};
  int n_clutter_{0};
  bool randomize_goal_{false};
  int n_bonus_tiles_{3};
  float bonus_reward_{1.0f};
  float bonus_penalty_{0.0f};
  bool initial_reward_{true};
  bool reset_on_mistake_{false};
  bool reward_decay_{true};
  bool respawn_{false};
  bool ghost_mode_{true};
  bool prestige_coloring_{false};
  float prestige_beta_{0.95f};
  float prestige_scale_{2.0f};
  std::string observation_format_{"pixels"};
  int max_episode_steps_{100};
  int step_count_{0};
  bool done_{true};
  std::vector<detail::Cell> grid_;
  std::vector<detail::Agent> agents_;
  std::vector<float> last_rewards_;

  [[nodiscard]] detail::Cell& CellAt(int x, int y) {
    return grid_[detail::Offset(x, y, grid_size_)];
  }

  [[nodiscard]] const detail::Cell& CellAt(int x, int y) const {
    return grid_[detail::Offset(x, y, grid_size_)];
  }

  [[nodiscard]] bool InBounds(int x, int y) const {
    return x >= 0 && x < grid_size_ && y >= 0 && y < grid_size_;
  }

  [[nodiscard]] int RandInt(int low, int high) {
    std::uniform_int_distribution<int> dist(low, high - 1);
    return dist(gen_);
  }

  void ClearGrid() { grid_.assign(grid_size_ * grid_size_, detail::Cell{}); }

  void WallRect() {
    for (int i = 0; i < grid_size_; ++i) {
      CellAt(i, 0).type = detail::CellType::kWall;
      CellAt(i, grid_size_ - 1).type = detail::CellType::kWall;
      CellAt(0, i).type = detail::CellType::kWall;
      CellAt(grid_size_ - 1, i).type = detail::CellType::kWall;
    }
  }

  void GenGrid() {
    ClearGrid();
    WallRect();
    if (env_name_ == "goalcycle") {
      for (int bonus_id = 0; bonus_id < n_bonus_tiles_; ++bonus_id) {
        PlaceObjectRandom(detail::CellType::kBonus, bonus_id, 100);
      }
    } else if (randomize_goal_) {
      PlaceObjectRandom(detail::CellType::kGoal, 0, 100);
    } else {
      CellAt(grid_size_ - 2, grid_size_ - 2).type = detail::CellType::kGoal;
    }
    for (int i = 0; i < n_clutter_; ++i) {
      PlaceObjectRandom(detail::CellType::kWall, 0, 100);
    }
  }

  void PlaceObjectRandom(detail::CellType type, int bonus_id, int max_tries) {
    for (int i = 0; i < max_tries; ++i) {
      int x = RandInt(0, grid_size_);
      int y = RandInt(0, grid_size_);
      detail::Cell& cell = CellAt(x, y);
      if (cell.type != detail::CellType::kEmpty || !cell.agents.empty()) {
        continue;
      }
      cell.type = type;
      cell.bonus_id = bonus_id;
      return;
    }
    throw std::runtime_error("MarlGrid object placement failed");
  }

  void PlaceAgent(int agent_id) {
    for (int i = 0; i < 100000; ++i) {
      int x = RandInt(0, grid_size_);
      int y = RandInt(0, grid_size_);
      detail::Cell& cell = CellAt(x, y);
      if (cell.type != detail::CellType::kEmpty) {
        continue;
      }
      if (!ghost_mode_ && !cell.agents.empty()) {
        continue;
      }
      cell.agents.push_back(agent_id);
      agents_[agent_id].x = x;
      agents_[agent_id].y = y;
      agents_[agent_id].dir = RandInt(0, 4);
      return;
    }
    throw std::runtime_error("MarlGrid agent placement failed");
  }

  void RemoveAgentFromCell(int agent_id) {
    detail::Agent& agent = agents_[agent_id];
    if (!InBounds(agent.x, agent.y)) {
      return;
    }
    auto& occupants = CellAt(agent.x, agent.y).agents;
    occupants.erase(std::remove(occupants.begin(), occupants.end(), agent_id),
                    occupants.end());
  }

  void MoveAgentTo(int agent_id, int x, int y) {
    RemoveAgentFromCell(agent_id);
    CellAt(x, y).agents.push_back(agent_id);
    agents_[agent_id].x = x;
    agents_[agent_id].y = y;
  }

  [[nodiscard]] float BonusReward(detail::Agent* agent,
                                  const detail::Cell& cell) const {
    bool first_bonus = false;
    int previous = agent->bonus_state;
    if (previous < 0) {
      previous = (cell.bonus_id + n_bonus_tiles_ - 1) % n_bonus_tiles_;
      agent->bonus_state = previous;
      first_bonus = true;
    }
    float reward = 0.0f;
    if (previous == cell.bonus_id) {
      reward = -std::abs(bonus_penalty_);
    } else if ((previous + 1) % n_bonus_tiles_ == cell.bonus_id) {
      agent->bonus_state = cell.bonus_id;
      reward = bonus_reward_;
    } else {
      reward = -std::abs(bonus_penalty_);
    }
    if (reset_on_mistake_) {
      agent->bonus_state = cell.bonus_id;
    }
    if (first_bonus && !initial_reward_) {
      return 0.0f;
    }
    return reward;
  }

  void ApplyPrestigeReward(detail::Agent* agent, float reward) const {
    if (!prestige_coloring_) {
      return;
    }
    if (reward >= 0.0f) {
      agent->prestige += reward;
    } else {
      agent->prestige = 0.0f;
    }
  }

  void DecayPrestige(detail::Agent* agent) const {
    if (prestige_coloring_ && agent->active) {
      agent->prestige *= prestige_beta_;
    }
  }

  void DoAgentStep(int agent_id, detail::Act act) {
    detail::Agent& agent = agents_[agent_id];
    if (!agent.active) {
      return;
    }
    auto dir_vec = detail::kDirToVec[agent.dir];
    int fwd_x = agent.x + dir_vec.first;
    int fwd_y = agent.y + dir_vec.second;
    CHECK(InBounds(fwd_x, fwd_y));
    const detail::Cell& fwd_cell = CellAt(fwd_x, fwd_y);
    if (act == detail::kLeft) {
      agent.dir = (agent.dir + 3) % 4;
    } else if (act == detail::kRight) {
      agent.dir = (agent.dir + 1) % 4;
    } else if (act == detail::kForward) {
      bool can_move = fwd_cell.CanOverlap();
      if (!ghost_mode_ && !fwd_cell.agents.empty()) {
        can_move = false;
      }
      if (can_move) {
        detail::CellType fwd_type = fwd_cell.type;
        int fwd_bonus_id = fwd_cell.bonus_id;
        MoveAgentTo(agent_id, fwd_x, fwd_y);
        float reward = 0.0f;
        if (fwd_type == detail::CellType::kGoal) {
          reward = 1.0f;
        } else if (fwd_type == detail::CellType::kBonus) {
          detail::Cell bonus_cell;
          bonus_cell.type = fwd_type;
          bonus_cell.bonus_id = fwd_bonus_id;
          reward = BonusReward(&agent, bonus_cell);
        }
        if (reward_decay_) {
          reward *= 1.0f - 0.9f * (static_cast<float>(step_count_) /
                                   static_cast<float>(max_episode_steps_));
        }
        last_rewards_[agent_id] += reward;
        agent.step_reward += reward;
        ApplyPrestigeReward(&agent, reward);
        if (fwd_type == detail::CellType::kGoal ||
            fwd_type == detail::CellType::kLava) {
          agent.done = true;
        }
      }
    } else if (act == detail::kPickup || act == detail::kDrop ||
               act == detail::kToggle || act == detail::kDone) {
      // Unsupported MarlGrid actions are accepted as no-ops.
    } else {
      throw std::runtime_error("invalid MarlGrid action");
    }
    DecayPrestige(&agent);
  }

  [[nodiscard]] std::vector<bool> VisibilityMask(
      const std::vector<bool>& transparent) const {
    int width = view_size_;
    int height = view_size_;
    int ax = view_size_ / 2;
    int ay = view_size_ - 1 - view_offset_;
    std::vector<bool> mask(width * height, false);
    mask[detail::Offset(ax, ay, width)] = true;
    for (int y = std::min(ay + 1, height - 1); y > 0; --y) {
      for (int x = ax; x < width; ++x) {
        if (mask[detail::Offset(x, y, width)] &&
            transparent[detail::Offset(x, y, width)]) {
          if (x < width - 1) {
            mask[detail::Offset(x + 1, y, width)] = true;
          }
          if (y > 0) {
            mask[detail::Offset(x, y - 1, width)] = true;
            if (x < width - 1) {
              mask[detail::Offset(x + 1, y - 1, width)] = true;
            }
          }
        }
      }
      for (int x = std::min(ax + 1, width - 1); x > 0; --x) {
        if (mask[detail::Offset(x, y, width)] &&
            transparent[detail::Offset(x, y, width)]) {
          if (x > 0) {
            mask[detail::Offset(x - 1, y, width)] = true;
          }
          if (y > 0) {
            mask[detail::Offset(x, y - 1, width)] = true;
            if (x > 0) {
              mask[detail::Offset(x - 1, y - 1, width)] = true;
            }
          }
        }
      }
    }
    for (int y = ay; y < height; ++y) {
      for (int x = ax; x < width; ++x) {
        if (mask[detail::Offset(x, y, width)] &&
            transparent[detail::Offset(x, y, width)]) {
          if (x < width - 1) {
            mask[detail::Offset(x + 1, y, width)] = true;
          }
          if (y < height - 1) {
            mask[detail::Offset(x, y + 1, width)] = true;
            if (x < width - 1) {
              mask[detail::Offset(x + 1, y + 1, width)] = true;
            }
          }
        }
      }
      for (int x = std::min(ax + 1, width - 1); x > 0; --x) {
        if (mask[detail::Offset(x, y, width)] &&
            transparent[detail::Offset(x, y, width)]) {
          if (x > 0) {
            mask[detail::Offset(x - 1, y, width)] = true;
          }
          if (y < height - 1) {
            mask[detail::Offset(x, y + 1, width)] = true;
            if (x > 0) {
              mask[detail::Offset(x - 1, y + 1, width)] = true;
            }
          }
        }
      }
    }
    return mask;
  }

  [[nodiscard]] std::pair<int, int> ViewTopLeft(
      const detail::Agent& agent) const {
    if (agent.dir == 0) {
      return {agent.x - view_offset_, agent.y - view_size_ / 2};
    }
    if (agent.dir == 1) {
      return {agent.x - view_size_ / 2, agent.y - view_offset_};
    }
    if (agent.dir == 2) {
      return {agent.x - view_size_ + 1 + view_offset_,
              agent.y - view_size_ / 2};
    }
    return {agent.x - view_size_ / 2, agent.y - view_size_ + 1 + view_offset_};
  }

  [[nodiscard]] std::vector<std::uint8_t> RenderAgentTile(int agent_id,
                                                          int tile_size) const {
    const detail::Agent& agent = agents_[agent_id];
    if (prestige_coloring_) {
      return detail::RenderAgentTile(
          detail::PrestigeColor(agent.prestige, prestige_scale_), agent.dir,
          tile_size);
    }
    return detail::RenderAgentTile(agent.color, agent.dir, tile_size);
  }

  [[nodiscard]] int TopAgentForCell(const detail::Cell* cell,
                                    int top_agent_id) const {
    if (cell == nullptr) {
      return -1;
    }
    if (top_agent_id >= 0 &&
        std::find(cell->agents.begin(), cell->agents.end(), top_agent_id) !=
            cell->agents.end() &&
        agents_[top_agent_id].active) {
      return top_agent_id;
    }
    for (int agent_id : cell->agents) {
      if (agents_[agent_id].active) {
        return agent_id;
      }
    }
    return -1;
  }

  [[nodiscard]] std::vector<std::uint8_t> RenderTile(const detail::Cell* cell,
                                                     int top_agent_id,
                                                     int tile_size) const {
    int chosen_agent = TopAgentForCell(cell, top_agent_id);
    if (cell == nullptr || cell->type == detail::CellType::kEmpty) {
      if (chosen_agent >= 0) {
        auto tile = RenderAgentTile(chosen_agent, tile_size);
        if (detail::HasBlackCorner(tile, tile_size)) {
          detail::AddTile(&tile, detail::EmptyTile(tile_size));
        }
        return tile;
      }
      return detail::EmptyTile(tile_size);
    }
    auto tile = detail::RenderObjectTile(cell->type, tile_size);
    if (chosen_agent >= 0) {
      auto agent_tile = RenderAgentTile(chosen_agent, tile_size);
      tile = detail::BlendTiles(tile, agent_tile, tile_size);
    }
    if (detail::HasBlackCorner(tile, tile_size)) {
      detail::AddTile(&tile, detail::EmptyTile(tile_size));
    }
    return tile;
  }

  void RenderGrid(int tile_size, int top_agent_id, int orientation,
                  std::uint8_t* output) const {
    int img_size = grid_size_ * tile_size;
    for (int y = 0; y < grid_size_; ++y) {
      for (int x = 0; x < grid_size_; ++x) {
        auto tile = RenderTile(&CellAt(x, y), top_agent_id, tile_size);
        tile = detail::RotateImage(tile, tile_size, tile_size, orientation);
        for (int row = 0; row < tile_size; ++row) {
          std::memcpy(
              output + ((y * tile_size + row) * img_size + x * tile_size) * 3,
              tile.data() + row * tile_size * 3, tile_size * 3);
        }
      }
    }
  }

  void RenderAgentObs(int agent_id, std::uint8_t* output) const {
    int obs_size = view_size_ * view_tile_size_;
    std::vector<std::uint8_t> image(obs_size * obs_size * 3, 0);
    for (int i = 0; i < obs_size * obs_size; ++i) {
      image[i * 3 + 0] = detail::kShadowColor[0];
      image[i * 3 + 1] = detail::kShadowColor[1];
      image[i * 3 + 2] = detail::kShadowColor[2];
    }
    const detail::Agent& agent = agents_[agent_id];
    if (!agent.active) {
      std::memcpy(output, image.data(), image.size());
      return;
    }
    int rot_k = (agent.dir + 1) % 4;
    int orientation = (4 - rot_k) % 4;
    auto [top_x, top_y] = ViewTopLeft(agent);
    std::vector<const detail::Cell*> view_cells(view_size_ * view_size_,
                                                nullptr);
    std::vector<bool> transparent(view_size_ * view_size_, true);
    for (int y = 0; y < view_size_; ++y) {
      for (int x = 0; x < view_size_; ++x) {
        auto [sx, sy] = detail::RotateCoord(x, y, view_size_, rot_k);
        int wx = top_x + sx;
        int wy = top_y + sy;
        if (InBounds(wx, wy)) {
          const detail::Cell& cell = CellAt(wx, wy);
          view_cells[detail::Offset(x, y, view_size_)] = &cell;
          transparent[detail::Offset(x, y, view_size_)] = cell.CanSeeBehind();
        }
      }
    }
    std::vector<bool> visible = VisibilityMask(transparent);
    for (int y = 0; y < view_size_; ++y) {
      for (int x = 0; x < view_size_; ++x) {
        if (!visible[detail::Offset(x, y, view_size_)]) {
          continue;
        }
        auto tile = RenderTile(view_cells[detail::Offset(x, y, view_size_)],
                               agent_id, view_tile_size_);
        tile = detail::RotateImage(tile, view_tile_size_, view_tile_size_,
                                   orientation);
        for (int row = 0; row < view_tile_size_; ++row) {
          std::memcpy(image.data() + ((y * view_tile_size_ + row) * obs_size +
                                      x * view_tile_size_) *
                                         3,
                      tile.data() + row * view_tile_size_ * 3,
                      view_tile_size_ * 3);
        }
      }
    }
    std::memcpy(output, image.data(), image.size());
  }

  static int MatrixObsOffset(int x, int y, int channel, int obs_size) {
    return (y * obs_size + x) * detail::kMatrixObsChannels + channel;
  }

  void WriteMatrixBaseTile(const detail::Cell* cell, int x, int y, int obs_size,
                           std::uint8_t* output) const {
    int base_channel = static_cast<int>(detail::MatrixObsChannel::kEmpty);
    if (cell != nullptr) {
      if (cell->type == detail::CellType::kWall) {
        base_channel = static_cast<int>(detail::MatrixObsChannel::kWall);
      } else if (cell->type == detail::CellType::kGoal) {
        base_channel = static_cast<int>(detail::MatrixObsChannel::kGoal);
      } else if (cell->type == detail::CellType::kBonus) {
        base_channel = static_cast<int>(detail::MatrixObsChannel::kBonus);
      } else if (cell->type == detail::CellType::kLava) {
        base_channel = static_cast<int>(detail::MatrixObsChannel::kLava);
      }
    }
    output[MatrixObsOffset(x, y, base_channel, obs_size)] = 255;
  }

  void WriteMatrixAgentTile(int chosen_agent, int x, int y, int obs_size,
                            std::uint8_t* output) const {
    if (chosen_agent < 0) {
      return;
    }
    detail::Rgb color =
        prestige_coloring_
            ? detail::PrestigeColor(agents_[chosen_agent].prestige,
                                    prestige_scale_)
            : detail::ColorValue(agents_[chosen_agent].color);
    output[MatrixObsOffset(
        x, y, static_cast<int>(detail::MatrixObsChannel::kAgent), obs_size)] =
        255;
    output[MatrixObsOffset(
        x, y, static_cast<int>(detail::MatrixObsChannel::kAgentRed),
        obs_size)] = color[0];
    output[MatrixObsOffset(
        x, y, static_cast<int>(detail::MatrixObsChannel::kAgentGreen),
        obs_size)] = color[1];
    output[MatrixObsOffset(
        x, y, static_cast<int>(detail::MatrixObsChannel::kAgentBlue),
        obs_size)] = color[2];
  }

  void WriteAgentMatrixObs(int agent_id, std::uint8_t* output) const {
    int obs_values = view_size_ * view_size_ * detail::kMatrixObsChannels;
    std::fill(output, output + obs_values, 0);
    for (int y = 0; y < view_size_; ++y) {
      for (int x = 0; x < view_size_; ++x) {
        output[MatrixObsOffset(
            x, y, static_cast<int>(detail::MatrixObsChannel::kEmpty),
            view_size_)] = 255;
      }
    }

    const detail::Agent& agent = agents_[agent_id];
    if (!agent.active) {
      return;
    }
    int rot_k = (agent.dir + 1) % 4;
    auto [top_x, top_y] = ViewTopLeft(agent);
    std::vector<const detail::Cell*> view_cells(view_size_ * view_size_,
                                                nullptr);
    std::vector<bool> transparent(view_size_ * view_size_, true);
    for (int y = 0; y < view_size_; ++y) {
      for (int x = 0; x < view_size_; ++x) {
        auto [sx, sy] = detail::RotateCoord(x, y, view_size_, rot_k);
        int wx = top_x + sx;
        int wy = top_y + sy;
        if (InBounds(wx, wy)) {
          const detail::Cell& cell = CellAt(wx, wy);
          view_cells[detail::Offset(x, y, view_size_)] = &cell;
          transparent[detail::Offset(x, y, view_size_)] = cell.CanSeeBehind();
        }
      }
    }
    std::vector<bool> visible = VisibilityMask(transparent);
    for (int y = 0; y < view_size_; ++y) {
      for (int x = 0; x < view_size_; ++x) {
        if (!visible[detail::Offset(x, y, view_size_)]) {
          continue;
        }
        const detail::Cell* cell = view_cells[detail::Offset(x, y, view_size_)];
        for (int channel = 0; channel < 5; ++channel) {
          output[MatrixObsOffset(x, y, channel, view_size_)] = 0;
        }
        WriteMatrixBaseTile(cell, x, y, view_size_, output);
        WriteMatrixAgentTile(TopAgentForCell(cell, agent_id), x, y, view_size_,
                             output);
      }
    }
  }

  void WriteAgentFullMatrixObs(int agent_id, std::uint8_t* output) const {
    int obs_values = grid_size_ * grid_size_ * detail::kMatrixObsChannels;
    std::fill(output, output + obs_values, 0);
    for (int y = 0; y < grid_size_; ++y) {
      for (int x = 0; x < grid_size_; ++x) {
        const detail::Cell& cell = CellAt(x, y);
        WriteMatrixBaseTile(&cell, x, y, grid_size_, output);
        WriteMatrixAgentTile(TopAgentForCell(&cell, agent_id), x, y, grid_size_,
                             output);
      }
    }
  }

  void ResizeNearest(const std::uint8_t* src, int src_width, int src_height,
                     std::uint8_t* dst, int dst_width, int dst_height) const {
    for (int y = 0; y < dst_height; ++y) {
      int sy = std::min(src_height - 1, y * src_height / dst_height);
      for (int x = 0; x < dst_width; ++x) {
        int sx = std::min(src_width - 1, x * src_width / dst_width);
        std::memcpy(dst + (y * dst_width + x) * 3,
                    src + (sy * src_width + sx) * 3, 3);
      }
    }
  }

  void WriteState() {
    auto state = Allocate(n_agents_);
    for (int i = 0; i < n_agents_; ++i) {
      state["info:players.id"_][i] = i;
      state["info:players.done"_][i] = agents_[i].done;
      state["info:players.active"_][i] = agents_[i].active;
      state["info:players.pos"_](i, 0) = agents_[i].x;
      state["info:players.pos"_](i, 1) = agents_[i].y;
      state["info:players.dir"_][i] = agents_[i].dir;
      state["reward"_][i] = last_rewards_[i];
      if (observation_format_ == "matrix") {
        WriteAgentMatrixObs(
            i, static_cast<std::uint8_t*>(state["obs"_][i].Data()));
      } else if (observation_format_ == "full_matrix") {
        WriteAgentFullMatrixObs(
            i, static_cast<std::uint8_t*>(state["obs"_][i].Data()));
      } else {
        RenderAgentObs(i, static_cast<std::uint8_t*>(state["obs"_][i].Data()));
      }
    }
  }
};

using MarlGridEnvPool = AsyncEnvPool<MarlGridEnv>;

}  // namespace marlgrid

#endif  // ENVPOOL_MARLGRID_MARLGRID_H_
