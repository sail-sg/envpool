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

#ifndef ENVPOOL_HIGHWAY_NATIVE_TASK_ENV_H_
#define ENVPOOL_HIGHWAY_NATIVE_TASK_ENV_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/highway/highway_env.h"
#include "envpool/highway/official_observation.h"
#include "envpool/highway/official_task.h"

namespace highway {
namespace native {

struct Command {
  std::array<double, 2> acceleration{0.0, 0.0};
  std::array<double, 2> steering{0.0, 0.0};
  int active_players{1};
};

struct Agent {
  double x{0.0};
  double y{0.0};
  double heading{0.0};
  double speed{0.0};
  bool crashed{false};
};

inline double Clip(double v, double low, double high) {
  return std::clamp(v, low, high);
}

inline int ClipInt(int v, int low, int high) {
  return std::clamp(v, low, high);
}

inline double Uniform(std::mt19937* gen, double low, double high) {
  std::uniform_real_distribution<double> dist(low, high);
  return dist(*gen);
}

inline highway::official::MetaAction ToMetaAction(int action) {
  using highway::official::MetaAction;
  action = ClipInt(action, 0, 4);
  if (action == 0) {
    return MetaAction::kLaneLeft;
  }
  if (action == 2) {
    return MetaAction::kLaneRight;
  }
  if (action == 3) {
    return MetaAction::kFaster;
  }
  if (action == 4) {
    return MetaAction::kSlower;
  }
  return MetaAction::kIdle;
}

inline double LMap(double value, double x0, double x1, double y0, double y1) {
  return y0 + (value - x0) * (y1 - y0) / (x1 - x0);
}

inline int Pix(double length, double scaling) {
  return static_cast<int>(length * scaling);
}

inline std::uint8_t Lighten(std::uint8_t value, double ratio = 0.68) {
  return static_cast<std::uint8_t>(
      std::min(static_cast<int>(static_cast<double>(value) / ratio), 255));
}

inline void Fill(unsigned char* rgb, int width, int height, std::uint8_t r,
                 std::uint8_t g, std::uint8_t b) {
  for (int i = 0; i < width * height; ++i) {
    rgb[3 * i + 0] = r;
    rgb[3 * i + 1] = g;
    rgb[3 * i + 2] = b;
  }
}

inline void SetPixel(unsigned char* rgb, int width, int height, int x, int y,
                     std::uint8_t r, std::uint8_t g, std::uint8_t b) {
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return;
  }
  const int offset = 3 * (y * width + x);
  rgb[offset + 0] = r;
  rgb[offset + 1] = g;
  rgb[offset + 2] = b;
}

struct SpritePixel {
  std::uint8_t r{0};
  std::uint8_t g{0};
  std::uint8_t b{0};
  bool opaque{false};
};

class Sprite {
 public:
  explicit Sprite(int size)
      : size_(std::max(1, size)), pixels_(size_ * size_) {}

  [[nodiscard]] int size() const { return size_; }

  void Set(int x, int y, std::uint8_t r, std::uint8_t g, std::uint8_t b) {
    if (x < 0 || x >= size_ || y < 0 || y >= size_) {
      return;
    }
    SpritePixel& pixel = pixels_[y * size_ + x];
    pixel.r = r;
    pixel.g = g;
    pixel.b = b;
    pixel.opaque = true;
  }

  [[nodiscard]] const SpritePixel& Get(int x, int y) const {
    return pixels_[y * size_ + x];
  }

 private:
  int size_{1};
  std::vector<SpritePixel> pixels_;
};

inline void FillRect(unsigned char* rgb, int width, int height, int x, int y,
                     int w, int h, std::uint8_t r, std::uint8_t g,
                     std::uint8_t b) {
  for (int yy = std::max(y, 0); yy < std::min(y + h, height); ++yy) {
    for (int xx = std::max(x, 0); xx < std::min(x + w, width); ++xx) {
      SetPixel(rgb, width, height, xx, yy, r, g, b);
    }
  }
}

inline void FillSpriteRect(Sprite* sprite, int x, int y, int w, int h,
                           std::uint8_t r, std::uint8_t g, std::uint8_t b) {
  for (int yy = std::max(y, 0); yy < std::min(y + h, sprite->size()); ++yy) {
    for (int xx = std::max(x, 0); xx < std::min(x + w, sprite->size()); ++xx) {
      sprite->Set(xx, yy, r, g, b);
    }
  }
}

inline void DrawSpriteRectOutline(Sprite* sprite, int x, int y, int w, int h,
                                  std::uint8_t r, std::uint8_t g,
                                  std::uint8_t b) {
  if (w <= 0 || h <= 0) {
    return;
  }
  FillSpriteRect(sprite, x, y, w, 1, r, g, b);
  FillSpriteRect(sprite, x, y + h - 1, w, 1, r, g, b);
  FillSpriteRect(sprite, x, y, 1, h, r, g, b);
  FillSpriteRect(sprite, x + w - 1, y, 1, h, r, g, b);
}

inline void BlitSprite(unsigned char* rgb, int width, int height,
                       const Sprite& sprite, int cx, int cy) {
  const int left = static_cast<int>(static_cast<double>(cx) -
                                    static_cast<double>(sprite.size()) / 2.0);
  const int top = static_cast<int>(static_cast<double>(cy) -
                                   static_cast<double>(sprite.size()) / 2.0);
  for (int y = 0; y < sprite.size(); ++y) {
    for (int x = 0; x < sprite.size(); ++x) {
      const SpritePixel& pixel = sprite.Get(x, y);
      if (pixel.opaque) {
        SetPixel(rgb, width, height, left + x, top + y, pixel.r, pixel.g,
                 pixel.b);
      }
    }
  }
}

inline void BlitRotatedSprite(unsigned char* rgb, int width, int height,
                              const Sprite& sprite, int cx, int cy,
                              double heading) {
  if (std::abs(heading) <= 2.0 * 3.14159265358979323846 / 180.0) {
    BlitSprite(rgb, width, height, sprite, cx, cy);
    return;
  }
  const double cos_h = std::cos(heading);
  const double sin_h = std::sin(heading);
  const double half = static_cast<double>(sprite.size()) / 2.0;
  const double radius = std::ceil(std::sqrt(2.0) * half);
  const int min_x = std::max(0, static_cast<int>(std::floor(cx - radius)));
  const int max_x =
      std::min(width - 1, static_cast<int>(std::ceil(cx + radius)));
  const int min_y = std::max(0, static_cast<int>(std::floor(cy - radius)));
  const int max_y =
      std::min(height - 1, static_cast<int>(std::ceil(cy + radius)));
  for (int y = min_y; y <= max_y; ++y) {
    for (int x = min_x; x <= max_x; ++x) {
      const double dx = static_cast<double>(x) + 0.5 - static_cast<double>(cx);
      const double dy = static_cast<double>(y) + 0.5 - static_cast<double>(cy);
      const double local_x = dx * cos_h + dy * sin_h + half;
      const double local_y = -dx * sin_h + dy * cos_h + half;
      const int sx = static_cast<int>(std::floor(local_x));
      const int sy = static_cast<int>(std::floor(local_y));
      if (sx < 0 || sx >= sprite.size() || sy < 0 || sy >= sprite.size()) {
        continue;
      }
      const SpritePixel& pixel = sprite.Get(sx, sy);
      if (pixel.opaque) {
        SetPixel(rgb, width, height, x, y, pixel.r, pixel.g, pixel.b);
      }
    }
  }
}

inline void DrawLine(unsigned char* rgb, int width, int height, int x0, int y0,
                     int x1, int y1, std::uint8_t r, std::uint8_t g,
                     std::uint8_t b, int thickness = 1) {
  const int dx = x1 - x0;
  const int dy = y1 - y0;
  const int steps = std::max(std::abs(dx), std::abs(dy));
  if (steps == 0) {
    const int brush_size = std::max(1, thickness);
    const int brush_offset = -(brush_size - 1) / 2;
    FillRect(rgb, width, height, x0 + brush_offset, y0 + brush_offset,
             brush_size, brush_size, r, g, b);
    return;
  }
  const int sx = dx < 0 ? -1 : 1;
  const int sy = dy < 0 ? -1 : 1;
  const int abs_dx = std::abs(dx);
  const int abs_dy = std::abs(dy);
  int err = abs_dx - abs_dy;
  int x = x0;
  int y = y0;
  const int brush_size = std::max(1, thickness);
  const int brush_offset = -(brush_size - 1) / 2;
  const int pygame_width2_dx = abs_dx >= abs_dy ? 0 : 1;
  const int pygame_width2_dy = abs_dx >= abs_dy ? 1 : 0;
  const auto draw_pixel = [&](int px, int py) {
    if (thickness == 1) {
      SetPixel(rgb, width, height, px, py, r, g, b);
    } else if (thickness == 2) {
      SetPixel(rgb, width, height, px, py, r, g, b);
      SetPixel(rgb, width, height, px + pygame_width2_dx, py + pygame_width2_dy,
               r, g, b);
    } else {
      FillRect(rgb, width, height, px + brush_offset, py + brush_offset,
               brush_size, brush_size, r, g, b);
    }
  };
  while (true) {
    draw_pixel(x, y);
    if (x == x1 && y == y1) {
      break;
    }
    const int err2 = 2 * err;
    if (err2 > -abs_dy) {
      err -= abs_dy;
      x += sx;
    }
    if (err2 < abs_dx) {
      err += abs_dx;
      y += sy;
    }
  }
}

inline void DrawCircle(unsigned char* rgb, int width, int height, int cx,
                       int cy, int radius, std::uint8_t r, std::uint8_t g,
                       std::uint8_t b, int thickness = 2) {
  const int outer2 = radius * radius;
  const int inner = std::max(0, radius - thickness);
  const int inner2 = inner * inner;
  for (int y = cy - radius; y <= cy + radius; ++y) {
    for (int x = cx - radius; x <= cx + radius; ++x) {
      const int d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      if (d2 <= outer2 && d2 >= inner2) {
        SetPixel(rgb, width, height, x, y, r, g, b);
      }
    }
  }
}

class NativeBaseFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("scenario"_.Bind(std::string("merge")),
                    "duration"_.Bind(40), "simulation_frequency"_.Bind(15),
                    "policy_frequency"_.Bind(1), "screen_width"_.Bind(600),
                    "screen_height"_.Bind(150), "render_agent"_.Bind(true));
  }
};

class NativeKinematics5Fns : public NativeBaseFns {
 public:
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const float inf = std::numeric_limits<float>::infinity();
    return MakeDict("obs"_.Bind(Spec<float>({5, 5}, {-inf, inf})),
                    "info:speed"_.Bind(Spec<float>({})),
                    "info:crashed"_.Bind(Spec<bool>({})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(MarkDiscrete(Spec<int>({-1}, {0, 4}))));
  }
};

class NativeKinematics7Action5Fns : public NativeBaseFns {
 public:
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const float inf = std::numeric_limits<float>::infinity();
    return MakeDict("obs"_.Bind(Spec<float>({15, 7}, {-inf, inf})),
                    "info:speed"_.Bind(Spec<float>({})),
                    "info:crashed"_.Bind(Spec<bool>({})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(MarkDiscrete(Spec<int>({-1}, {0, 4}))));
  }
};

class NativeKinematics7Action3Fns : public NativeBaseFns {
 public:
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const float inf = std::numeric_limits<float>::infinity();
    return MakeDict("obs"_.Bind(Spec<float>({15, 7}, {-inf, inf})),
                    "info:speed"_.Bind(Spec<float>({})),
                    "info:crashed"_.Bind(Spec<bool>({})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(MarkDiscrete(Spec<int>({-1}, {0, 2}))));
  }
};

class NativeKinematics8ContinuousFns : public NativeBaseFns {
 public:
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const float inf = std::numeric_limits<float>::infinity();
    return MakeDict("obs"_.Bind(Spec<float>({5, 8}, {-inf, inf})),
                    "info:speed"_.Bind(Spec<float>({})),
                    "info:crashed"_.Bind(Spec<bool>({})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<float>({2}, {-1.0, 1.0})));
  }
};

template <int Horizon>
class NativeTTCFns : public NativeBaseFns {
 public:
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.Bind(Spec<float>({3, 3, Horizon}, {0.0, 1.0})),
                    "info:speed"_.Bind(Spec<float>({})),
                    "info:crashed"_.Bind(Spec<bool>({})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(MarkDiscrete(Spec<int>({-1}, {0, 4}))));
  }
};

class NativeGoalFns : public NativeBaseFns {
 public:
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const double inf = std::numeric_limits<double>::infinity();
    return MakeDict("obs:observation"_.Bind(Spec<double>({6}, {-inf, inf})),
                    "obs:achieved_goal"_.Bind(Spec<double>({6}, {-inf, inf})),
                    "obs:desired_goal"_.Bind(Spec<double>({6}, {-inf, inf})),
                    "info:speed"_.Bind(Spec<float>({})),
                    "info:crashed"_.Bind(Spec<bool>({})),
                    "info:is_success"_.Bind(Spec<bool>({})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<float>({2}, {-1.0, 1.0})));
  }
};

class NativeAttributesFns : public NativeBaseFns {
 public:
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const double inf = std::numeric_limits<double>::infinity();
    return MakeDict(
        "obs:state"_.Bind(Spec<double>({4, 1}, {-inf, inf})),
        "obs:derivative"_.Bind(Spec<double>({4, 1}, {-inf, inf})),
        "obs:reference_state"_.Bind(Spec<double>({4, 1}, {-inf, inf})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<float>({1}, {-1.0, 1.0})));
  }
};

class NativeOccupancyFns : public NativeBaseFns {
 public:
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const float inf = std::numeric_limits<float>::infinity();
    return MakeDict("obs"_.Bind(Spec<float>({2, 12, 12}, {-inf, inf})),
                    "info:speed"_.Bind(Spec<float>({})),
                    "info:crashed"_.Bind(Spec<bool>({})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<float>({1}, {-1.0, 1.0})));
  }
};

class NativeMultiAgentFns : public NativeBaseFns {
 public:
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const float inf = std::numeric_limits<float>::infinity();
    // TODO(jiayi): Expose per-player termination for
    // intersection-multi-agent-v1.
    // The official wrapper can return mixed terminal tuples such as
    // (false, true), while EnvPool's top-level done is env-level scalar.
    return MakeDict(
        "obs:players.obs"_.Bind(Spec<float>({-1, 5, 5}, {-inf, inf})),
        "info:players.speed"_.Bind(Spec<float>({-1})),
        "info:players.crashed"_.Bind(Spec<bool>({-1})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "players.action"_.Bind(MarkDiscrete(Spec<int>({-1}, {0, 2}))));
  }
};

struct Discrete5Adapter {
  template <typename Action>
  static Command ReadAction(const Action& action) {
    const int a = ClipInt(static_cast<int>(action["action"_]), 0, 4);
    Command command;
    if (a == 0) {
      command.steering[0] = -0.45;
    } else if (a == 2) {
      command.steering[0] = 0.45;
    } else if (a == 3) {
      command.acceleration[0] = 4.0;
    } else if (a == 4) {
      command.acceleration[0] = -5.0;
    }
    return command;
  }
};

struct Discrete3Adapter {
  template <typename Action>
  static Command ReadAction(const Action& action) {
    const int a = ClipInt(static_cast<int>(action["action"_]), 0, 2);
    Command command;
    command.acceleration[0] = (a - 1) * 4.0;
    return command;
  }
};

struct Continuous2Adapter {
  template <typename Action>
  static Command ReadAction(const Action& action) {
    Command command;
    command.acceleration[0] = Clip(action["action"_][0], -1.0, 1.0) * 5.0;
    command.steering[0] = Clip(action["action"_][1], -1.0, 1.0);
    return command;
  }
};

struct Continuous1Adapter {
  template <typename Action>
  static Command ReadAction(const Action& action) {
    Command command;
    command.steering[0] = Clip(action["action"_][0], -1.0, 1.0);
    command.acceleration[0] = 1.2;
    return command;
  }
};

struct MultiAgentAdapter {
  template <typename Action>
  static Command ReadAction(const Action& action) {
    Command command;
    const int n =
        std::min(static_cast<int>(action["players.action"_].Shape(0)), 2);
    command.active_players = n;
    for (int i = 0; i < n; ++i) {
      const int a =
          ClipInt(static_cast<int>(action["players.action"_][i]), 0, 2);
      command.acceleration[i] = (a - 1) * 4.0;
    }
    return command;
  }
};

using NativeK5Spec = EnvSpec<NativeKinematics5Fns>;
using NativeK75Spec = EnvSpec<NativeKinematics7Action5Fns>;
using NativeK73Spec = EnvSpec<NativeKinematics7Action3Fns>;
using NativeK8CSpec = EnvSpec<NativeKinematics8ContinuousFns>;
using NativeTTC5Spec = EnvSpec<NativeTTCFns<5>>;
using NativeTTC16Spec = EnvSpec<NativeTTCFns<16>>;
using NativeGoalSpec = EnvSpec<NativeGoalFns>;
using NativeAttributesSpec = EnvSpec<NativeAttributesFns>;
using NativeOccupancySpec = EnvSpec<NativeOccupancyFns>;
using NativeMultiAgentSpec = EnvSpec<NativeMultiAgentFns>;

using NativeKinematics5EnvSpec = NativeK5Spec;
using NativeKinematics7Action5EnvSpec = NativeK75Spec;
using NativeKinematics7Action3EnvSpec = NativeK73Spec;
using NativeKinematics8ContinuousEnvSpec = NativeK8CSpec;
using NativeGoalEnvSpec = NativeGoalSpec;
using NativeAttributesEnvSpec = NativeAttributesSpec;
using NativeOccupancyEnvSpec = NativeOccupancySpec;
using NativeMultiAgentEnvSpec = NativeMultiAgentSpec;

template <typename SpecT, typename AdapterT>
class NativeTaskEnv : public Env<SpecT>, public RenderableEnv {
 public:
  using Base = Env<SpecT>;
  using Spec = typename Base::Spec;
  using State = typename Base::State;
  using Action = typename Base::Action;

  NativeTaskEnv(const Spec& spec, int env_id)
      : Base(spec, env_id),
        scenario_(spec.config["scenario"_]),
        max_episode_steps_(std::max(
            1, static_cast<int>(spec.config["duration"_]) *
                   static_cast<int>(spec.config["policy_frequency"_]))),
        simulation_frequency_(spec.config["simulation_frequency"_]),
        policy_frequency_(spec.config["policy_frequency"_]) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    elapsed_step_ = 0;
    time_ = 0.0;
    done_ = false;
    if (UseOfficialBackend()) {
      ResetOfficialBackend();
      WriteState(0.0f);
      return;
    }
    ResetAgents();
    ResetTraffic();
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    if (UseOfficialBackend()) {
      StepOfficialBackend(action);
      return;
    }
    const Command command = AdapterT::ReadAction(action);
    ++elapsed_step_;
    const int frames = std::max(1, simulation_frequency_ / policy_frequency_);
    const double dt = 1.0 / static_cast<double>(simulation_frequency_);
    for (int frame = 0; frame < frames; ++frame) {
      Simulate(command, dt);
    }
    time_ += 1.0 / static_cast<double>(policy_frequency_);
    done_ = elapsed_step_ >= max_episode_steps_ || agents_[0].crashed;
    WriteState(static_cast<float>(Reward()));
  }

  [[nodiscard]] std::pair<int, int> RenderSize(int width,
                                               int height) const override {
    const int default_width = this->spec_.config["screen_width"_];
    const int default_height = this->spec_.config["screen_height"_];
    return {width > 0 ? width : default_width,
            height > 0 ? height : default_height};
  }

  void Render(int width, int height, int camera_id,
              unsigned char* rgb) override {
    (void)camera_id;
    Fill(rgb, width, height, 100, 100, 100);
    if (UseOfficialBackend()) {
      const OfficialRenderTransform transform =
          OfficialRenderCamera(width, height);
      DrawOfficialRoad(rgb, width, height, transform);
      for (const official::RoadObject& object : official_road_->objects) {
        DrawOfficialObject(rgb, width, height, transform, object);
      }
      for (int i = 0; i < static_cast<int>(official_road_->vehicles.size());
           ++i) {
        const official::Vehicle& vehicle = official_road_->vehicles[i];
        DrawOfficialVehicle(rgb, width, height, transform, vehicle);
      }
      return;
    }
    DrawScenario(rgb, width, height);
    for (const Agent& agent : traffic_) {
      DrawAgent(rgb, width, height, agent, 90, 190, 255);
    }
    DrawAgent(rgb, width, height, agents_[0], agents_[0].crashed ? 255 : 50,
              agents_[0].crashed ? 100 : 200, 0);
    if (scenario_ == "intersection_multi") {
      DrawAgent(rgb, width, height, agents_[1], agents_[1].crashed ? 255 : 255,
                agents_[1].crashed ? 100 : 180, 20);
    }
  }

  [[nodiscard]] HighwayDebugState DebugState() const {
    HighwayDebugState state;
    state.scenario = scenario_;
    state.simulation_frequency = simulation_frequency_;
    state.policy_frequency = policy_frequency_;
    state.elapsed_step = elapsed_step_;
    state.time = time_;
    if (UseOfficialBackend()) {
      const std::vector<official::LaneIndex> lane_indexes =
          official_road_->network.LaneIndexes();
      state.road_lanes.reserve(lane_indexes.size());
      for (const official::LaneIndex& lane_index : lane_indexes) {
        const official::Lane& source =
            official_road_->network.GetLane(lane_index);
        HighwayLaneDebugState lane;
        lane.from = lane_index.from;
        lane.to = lane_index.to;
        lane.index = lane_index.id;
        lane.kind = static_cast<int>(source.Kind());
        lane.start_x = source.Start().x;
        lane.start_y = source.Start().y;
        lane.end_x = source.End().x;
        lane.end_y = source.End().y;
        lane.center_x = source.Center().x;
        lane.center_y = source.Center().y;
        lane.width = source.Width();
        const auto line_types = source.LineTypes();
        lane.line_type0 = static_cast<int>(line_types[0]);
        lane.line_type1 = static_cast<int>(line_types[1]);
        lane.forbidden = source.Forbidden();
        lane.speed_limit = source.SpeedLimit();
        lane.priority = source.Priority();
        lane.amplitude = source.Amplitude();
        lane.pulsation = source.Pulsation();
        lane.phase = source.Phase();
        lane.radius = source.Radius();
        lane.start_phase = source.StartPhase();
        lane.end_phase = source.EndPhase();
        lane.clockwise = source.Clockwise();
        state.road_lanes.push_back(lane);
      }
      state.road_objects.reserve(official_road_->objects.size());
      for (const official::RoadObject& source : official_road_->objects) {
        HighwayRoadObjectDebugState object;
        object.kind = static_cast<int>(source.kind);
        object.x = source.position.x;
        object.y = source.position.y;
        object.heading = source.heading;
        object.speed = source.speed;
        object.length = source.length;
        object.width = source.width;
        object.collidable = source.collidable;
        object.solid = source.solid;
        object.check_collisions = source.check_collisions;
        object.crashed = source.crashed;
        object.hit = source.hit;
        state.road_objects.push_back(object);
      }
      state.vehicles.reserve(official_road_->vehicles.size());
      for (const official::Vehicle& source : official_road_->vehicles) {
        HighwayVehicleDebugState vehicle;
        vehicle.kind = static_cast<int>(source.kind);
        vehicle.lane_from = source.lane_index.from;
        vehicle.lane_to = source.lane_index.to;
        vehicle.lane_index = source.lane_index.id;
        vehicle.target_lane_from = source.target_lane_index.from;
        vehicle.target_lane_to = source.target_lane_index.to;
        vehicle.target_lane_index = source.target_lane_index.id;
        vehicle.speed_index = source.speed_index;
        vehicle.x = source.position.x;
        vehicle.y = source.position.y;
        vehicle.heading = source.heading;
        vehicle.speed = source.speed;
        vehicle.target_speed = source.target_speed;
        vehicle.target_speed0 = source.target_speeds[0];
        vehicle.target_speed1 = source.target_speeds[1];
        vehicle.target_speed2 = source.target_speeds[2];
        vehicle.has_goal = source.has_goal;
        vehicle.goal_x = source.goal_position.x;
        vehicle.goal_y = source.goal_position.y;
        vehicle.goal_heading = source.goal_heading;
        vehicle.goal_speed = source.goal_speed;
        vehicle.idm_delta = source.idm_delta;
        vehicle.timer = source.timer;
        vehicle.crashed = source.crashed;
        vehicle.on_road = official_road_->network.GetLane(source.lane_index)
                              .OnLane(source.position);
        vehicle.check_collisions = source.check_collisions;
        vehicle.enable_lane_change = source.enable_lane_change;
        vehicle.route_from.reserve(source.route.size());
        vehicle.route_to.reserve(source.route.size());
        vehicle.route_id.reserve(source.route.size());
        for (const official::LaneIndex& lane_index : source.route) {
          vehicle.route_from.push_back(lane_index.from);
          vehicle.route_to.push_back(lane_index.to);
          vehicle.route_id.push_back(lane_index.id);
        }
        state.vehicles.push_back(vehicle);
      }
      return state;
    }
    state.vehicles.reserve(1 + traffic_.size());
    const Agent& ego = agents_[0];
    HighwayVehicleDebugState vehicle;
    vehicle.kind = 0;
    vehicle.x = ego.x;
    vehicle.y = ego.y;
    vehicle.heading = ego.heading;
    vehicle.speed = ego.speed;
    vehicle.target_speed = ego.speed;
    vehicle.crashed = ego.crashed;
    state.vehicles.push_back(vehicle);
    for (const Agent& source : traffic_) {
      HighwayVehicleDebugState other;
      other.kind = 1;
      other.x = source.x;
      other.y = source.y;
      other.heading = source.heading;
      other.speed = source.speed;
      other.target_speed = source.speed;
      other.crashed = source.crashed;
      state.vehicles.push_back(other);
    }
    return state;
  }

 protected:
  [[nodiscard]] int CurrentMaxEpisodeSteps() const override {
    return max_episode_steps_;
  }

 private:
  std::string scenario_;
  int max_episode_steps_;
  int simulation_frequency_;
  int policy_frequency_;
  int elapsed_step_{0};
  double time_{0.0};
  bool done_{true};
  std::array<Agent, 2> agents_;
  std::array<Agent, 2> goals_;
  std::vector<Agent> traffic_;
  std::optional<official::Road> official_road_;
  int official_ego_index_{0};
  int official_player_count_{1};
  int official_last_action_{1};
  double official_last_continuous_action_norm_{0.0};
  official::LaneIndex official_active_lane_index_{"c", "d", 0};

  [[nodiscard]] bool UseOfficialBackend() const {
    return scenario_ == "merge" || scenario_ == "roundabout" ||
           scenario_ == "two_way" || scenario_ == "u_turn" ||
           scenario_ == "exit" || scenario_ == "intersection" ||
           scenario_ == "intersection_continuous" ||
           scenario_ == "intersection_multi" || scenario_ == "lane_keeping" ||
           scenario_.find("racetrack") == 0 || IsParkingScenario();
  }

  [[nodiscard]] bool IsParkingScenario() const {
    return scenario_ == "parking" || scenario_ == "parking_action_repeat" ||
           scenario_ == "parking_parked";
  }

  [[nodiscard]] official::Vehicle& OfficialEgo() {
    return official_road_->vehicles[official_ego_index_];
  }

  [[nodiscard]] const official::Vehicle& OfficialEgo() const {
    return official_road_->vehicles[official_ego_index_];
  }

  [[nodiscard]] official::Vehicle& OfficialPlayer(int player) {
    return official_road_->vehicles[player];
  }

  [[nodiscard]] const official::Vehicle& OfficialPlayer(int player) const {
    return official_road_->vehicles[player];
  }

  void ResetOfficialBackend() {
    official_player_count_ = 1;
    if (IsParkingScenario()) {
      official_road_ = official::MakeParkingRoad();
      std::uniform_int_distribution<int> spot_dist(0, 27);
      const double heading = Uniform(&this->gen_, 0.0, 2.0 * kPi);
      official_ego_index_ = official::ResetParkingVehicles(
          &*official_road_, 0.0, heading, spot_dist(this->gen_),
          scenario_ == "parking_parked");
    } else if (scenario_ == "exit") {
      official_road_ = official::MakeExitRoad();
      official_ego_index_ = official::ResetExitVehicles(&*official_road_);
    } else if (scenario_ == "intersection" ||
               scenario_ == "intersection_continuous") {
      official_road_ = official::MakeIntersectionRoad();
      official_ego_index_ =
          official::ResetIntersectionVehicles(&*official_road_);
    } else if (scenario_ == "intersection_multi") {
      official_road_ = official::MakeIntersectionRoad();
      official_ego_index_ =
          official::ResetMultiAgentIntersectionVehicles(&*official_road_);
      official_player_count_ = 2;
    } else if (scenario_ == "lane_keeping") {
      official_road_ = official::MakeLaneKeepingRoad();
      official_ego_index_ = official::ResetLaneKeepingVehicle(&*official_road_);
      official_active_lane_index_ = {"c", "d", 0};
    } else if (scenario_.find("racetrack") == 0) {
      official_road_ = official::MakeRacetrackRoad(scenario_);
      const double longitudinal = scenario_ == "racetrack"        ? 48.0
                                  : scenario_ == "racetrack_oval" ? 50.0
                                                                  : 80.0;
      official_ego_index_ =
          official::ResetRacetrackVehicles(&*official_road_, longitudinal, 0);
      official_active_lane_index_ = {"a", "b", 0};
    } else if (scenario_ == "roundabout") {
      official_road_ = official::MakeRoundaboutRoad();
      official_ego_index_ = official::ResetRoundaboutVehicles(&*official_road_);
    } else if (scenario_ == "two_way") {
      official_road_ = official::MakeTwoWayRoad();
      official_ego_index_ = official::ResetTwoWayVehicles(&*official_road_);
    } else if (scenario_ == "u_turn") {
      official_road_ = official::MakeUTurnRoad();
      official_ego_index_ = official::ResetUTurnVehicles(&*official_road_);
    } else {
      official_road_ = official::MakeMergeRoad();
      const double p0 = Uniform(&this->gen_, -5.0, 5.0);
      const double p1 = Uniform(&this->gen_, -5.0, 5.0);
      const double p2 = Uniform(&this->gen_, -5.0, 5.0);
      const double v0 = Uniform(&this->gen_, -1.0, 1.0);
      const double v1 = Uniform(&this->gen_, -1.0, 1.0);
      const double v2 = Uniform(&this->gen_, -1.0, 1.0);
      official_ego_index_ = official::ResetMergeVehicles(&*official_road_, p0,
                                                         p1, p2, v0, v1, v2);
    }
    official_last_action_ = 1;
    official_last_continuous_action_norm_ = 0.0;
  }

  void StepOfficialBackend(const Action& action) {
    if (IsParkingScenario()) {
      if constexpr (std::is_same_v<SpecT, NativeGoalSpec>) {
        StepOfficialParking(action);
        return;
      }
    }
    if (scenario_ == "intersection_continuous") {
      if constexpr (std::is_same_v<SpecT, NativeK8CSpec>) {
        StepOfficialContinuous(action, -5.0, 5.0, -kPi / 3.0, kPi / 3.0);
        return;
      }
    }
    if (scenario_ == "intersection_multi") {
      if constexpr (std::is_same_v<AdapterT, MultiAgentAdapter>) {
        StepOfficialMultiAgentIntersection(action);
        return;
      }
    }
    if (scenario_ == "lane_keeping") {
      if constexpr (std::is_same_v<SpecT, NativeAttributesSpec>) {
        StepOfficialLaneKeeping(action);
        return;
      }
    }
    if (scenario_.find("racetrack") == 0) {
      if constexpr (std::is_same_v<SpecT, NativeOccupancySpec>) {
        StepOfficialRacetrack(action);
        return;
      }
    }
    ++elapsed_step_;
    official_last_action_ = OfficialMetaAction(action);
    const int frames = std::max(1, simulation_frequency_ / policy_frequency_);
    const double dt = 1.0 / static_cast<double>(simulation_frequency_);
    for (int frame = 0; frame < frames; ++frame) {
      if (frame == 0) {
        official::ActMDP(&OfficialEgo(), official_road_->network,
                         ToMetaAction(official_last_action_));
      }
      official_road_->Act();
      official_road_->Step(dt);
    }
    time_ += 1.0 / static_cast<double>(policy_frequency_);
    done_ = OfficialEgo().crashed || elapsed_step_ >= max_episode_steps_ ||
            (scenario_ == "merge" && OfficialEgo().position.x > 370.0) ||
            ((scenario_ == "intersection" ||
              scenario_ == "intersection_continuous") &&
             OfficialIntersectionArrived(OfficialEgo()));
    WriteState(static_cast<float>(OfficialReward()));
  }

  void StepOfficialMultiAgentIntersection(const Action& action) {
    ++elapsed_step_;
    const int frames = std::max(1, simulation_frequency_ / policy_frequency_);
    const double dt = 1.0 / static_cast<double>(simulation_frequency_);
    for (int frame = 0; frame < frames; ++frame) {
      if (frame == 0) {
        for (int player = 0; player < official_player_count_; ++player) {
          const int action_id = ClipInt(
              static_cast<int>(action["players.action"_][player]), 0, 2);
          official::ActMDP(
              &OfficialPlayer(player), official_road_->network,
              ToMetaAction(IntersectionLongitudinalAction(action_id)));
        }
      }
      official_road_->Act();
      official_road_->Step(dt);
    }
    time_ += 1.0 / static_cast<double>(policy_frequency_);
    done_ = elapsed_step_ >= max_episode_steps_;
    bool all_players_arrived = true;
    for (int player = 0; player < official_player_count_; ++player) {
      done_ = done_ || OfficialPlayer(player).crashed;
      all_players_arrived = all_players_arrived &&
                            OfficialIntersectionArrived(OfficialPlayer(player));
    }
    done_ = done_ || all_players_arrived;
    WriteOfficialMultiAgentState();
  }

  void StepOfficialLaneKeeping(const Action& action) {
    ++elapsed_step_;
    official::LowLevelAction low_level;
    const double steering = ClippedContinuousAction(action, 0);
    official_last_continuous_action_norm_ = std::abs(steering);
    low_level.steering = OfficialContinuousMap(steering, -kPi / 3.0, kPi / 3.0);
    low_level.acceleration = 0.0;
    OfficialEgo().Act(low_level);

    State state = this->Allocate();
    WriteOfficialAttributes(&state);

    const double dt = 1.0 / static_cast<double>(simulation_frequency_);
    OfficialBicycleStep(&OfficialEgo(), dt);
    time_ += 1.0 / static_cast<double>(policy_frequency_);
    done_ = false;
    state["reward"_] = static_cast<float>(OfficialLaneKeepingReward());
  }

  void StepOfficialRacetrack(const Action& action) {
    ++elapsed_step_;
    official::LowLevelAction low_level;
    const double steering = ClippedContinuousAction(action, 0);
    official_last_continuous_action_norm_ = std::abs(steering);
    low_level.steering = OfficialContinuousMap(steering, -kPi / 4.0, kPi / 4.0);
    low_level.acceleration = 0.0;
    OfficialEgo().Act(low_level);

    StepOfficialRoadFrames();
    done_ = OfficialEgo().crashed || !OfficialOnRoad(OfficialEgo()) ||
            elapsed_step_ >= max_episode_steps_;
    WriteState(static_cast<float>(OfficialRacetrackReward()));
  }

  void StepOfficialParking(const Action& action) {
    ++elapsed_step_;
    official::LowLevelAction low_level;
    low_level.acceleration =
        OfficialContinuousMap(ClippedContinuousAction(action, 0), -5.0, 5.0);
    low_level.steering = OfficialContinuousMap(
        ClippedContinuousAction(action, 1), -kPi / 4.0, kPi / 4.0);
    OfficialEgo().Act(low_level);

    StepOfficialRoadFrames();
    done_ = OfficialEgo().crashed || OfficialParkingSuccess() ||
            elapsed_step_ >= max_episode_steps_;
    WriteState(static_cast<float>(OfficialReward()));
  }

  void StepOfficialContinuous(const Action& action, double acceleration_low,
                              double acceleration_high, double steering_low,
                              double steering_high) {
    ++elapsed_step_;
    official::LowLevelAction low_level;
    low_level.acceleration =
        OfficialContinuousMap(ClippedContinuousAction(action, 0),
                              acceleration_low, acceleration_high);
    low_level.steering = OfficialContinuousMap(
        ClippedContinuousAction(action, 1), steering_low, steering_high);
    OfficialEgo().Act(low_level);

    StepOfficialRoadFrames();
    done_ = OfficialEgo().crashed || elapsed_step_ >= max_episode_steps_ ||
            (scenario_ == "intersection_continuous" &&
             OfficialIntersectionArrived(OfficialEgo()));
    WriteState(static_cast<float>(OfficialReward()));
  }

  void StepOfficialRoadFrames() {
    const int frames = std::max(1, simulation_frequency_ / policy_frequency_);
    const double dt = 1.0 / static_cast<double>(simulation_frequency_);
    for (int frame = 0; frame < frames; ++frame) {
      official_road_->Act();
      official_road_->Step(dt);
    }
    time_ += 1.0 / static_cast<double>(policy_frequency_);
  }

  [[nodiscard]] double ClippedContinuousAction(const Action& action,
                                               int index) const {
    return Clip(action["action"_][index], -1.0, 1.0);
  }

  [[nodiscard]] int OfficialMetaAction(const Action& action) const {
    if constexpr (std::is_same_v<AdapterT, MultiAgentAdapter>) {
      return ClipInt(static_cast<int>(action["players.action"_][0]), 0, 4);
    } else {
      const int action_id = ClipInt(static_cast<int>(action["action"_]), 0, 4);
      if (scenario_ == "intersection") {
        return IntersectionLongitudinalAction(action_id);
      }
      return action_id;
    }
  }

  [[nodiscard]] static int IntersectionLongitudinalAction(int action_id) {
    if (action_id == 0) {
      return 4;
    }
    if (action_id == 2) {
      return 3;
    }
    return 1;
  }

  [[nodiscard]] static double OfficialContinuousMap(double value, double low,
                                                    double high) {
    const float value32 = static_cast<float>(value);
    const float low32 = static_cast<float>(low);
    const float high32 = static_cast<float>(high);
    return static_cast<double>(low32 + (high32 - low32) * (value32 - -1.0f) /
                                           (1.0f - -1.0f));
  }

  void ResetAgents() {
    agents_[0] = {};
    agents_[1] = {};
    goals_[0] = {};
    goals_[1] = {};
    if (scenario_.find("parking") == 0) {
      agents_[0] = {-30.0 + Uniform(&this->gen_, -3.0, 3.0),
                    Uniform(&this->gen_, -2.0, 2.0), 0.0, 0.0, false};
      goals_[0] = {0.0, 14.0, 0.0, 0.0, false};
    } else if (scenario_ == "intersection" ||
               scenario_ == "intersection_continuous" ||
               scenario_ == "intersection_multi") {
      agents_[0] = {0.0, 45.0, -kPi / 2.0, 7.0, false};
      agents_[1] = {-45.0, 0.0, 0.0, 7.0, false};
      goals_[0] = {0.0, -45.0, -kPi / 2.0, 0.0, false};
      goals_[1] = {45.0, 0.0, 0.0, 0.0, false};
    } else if (scenario_.find("racetrack") == 0) {
      agents_[0] = {-35.0, 0.0, 0.0, 8.0, false};
    } else if (scenario_ == "roundabout") {
      agents_[0] = {-55.0, 0.0, 0.0, 12.0, false};
    } else if (scenario_ == "two_way") {
      agents_[0] = {0.0, 0.0, 0.0, 22.0, false};
    } else if (scenario_ == "u_turn") {
      agents_[0] = {-45.0, 4.0, 0.0, 12.0, false};
    } else if (scenario_ == "lane_keeping") {
      agents_[0] = {0.0, Uniform(&this->gen_, -0.2, 0.2),
                    Uniform(&this->gen_, -0.04, 0.04), 12.0, false};
    } else {
      agents_[0] = {0.0, 0.0, 0.0, 24.0, false};
    }
  }

  void ResetTraffic() {
    traffic_.clear();
    const int traffic_count =
        scenario_ == "exit" || scenario_ == "intersection" ? 12 : 4;
    for (int i = 0; i < traffic_count; ++i) {
      Agent agent;
      if (scenario_ == "intersection" || scenario_ == "intersection_multi") {
        agent.x = Uniform(&this->gen_, -70.0, 70.0);
        agent.y = (i % 2 == 0) ? -8.0 : 8.0;
        agent.heading = (i % 2 == 0) ? 0.0 : kPi;
        agent.speed = 5.0 + Uniform(&this->gen_, 0.0, 3.0);
      } else {
        agent.x = 30.0 + i * 28.0 + Uniform(&this->gen_, -3.0, 3.0);
        agent.y = (i % 3 - 1) * kLaneWidth;
        agent.speed = 15.0 + Uniform(&this->gen_, 0.0, 8.0);
      }
      traffic_.push_back(agent);
    }
  }

  void Simulate(const Command& command, double dt) {
    const int agents = scenario_ == "intersection_multi" ? 2 : 1;
    for (int i = 0; i < agents; ++i) {
      StepAgent(command, i, dt);
    }
    for (Agent& agent : traffic_) {
      agent.x += agent.speed * std::cos(agent.heading) * dt;
      agent.y += agent.speed * std::sin(agent.heading) * dt;
    }
    for (int i = 0; i < agents; ++i) {
      for (const Agent& traffic : traffic_) {
        if (Distance(agents_[i], traffic) < 3.0) {
          agents_[i].crashed = true;
        }
      }
    }
  }

  void StepAgent(const Command& command, int index, double dt) {
    Agent& agent = agents_[index];
    agent.speed = Clip(agent.speed + command.acceleration[index] * dt, 0.0,
                       scenario_.find("racetrack") == 0 ? 12.0 : 35.0);
    agent.heading += command.steering[index] * dt;
    agent.x += agent.speed * std::cos(agent.heading) * dt;
    agent.y += agent.speed * std::sin(agent.heading) * dt;
  }

  [[nodiscard]] static double Distance(const Agent& lhs, const Agent& rhs) {
    const double dx = lhs.x - rhs.x;
    const double dy = lhs.y - rhs.y;
    return std::sqrt(dx * dx + dy * dy);
  }

  [[nodiscard]] double Reward() const {
    if (UseOfficialBackend()) {
      return OfficialReward();
    }
    if (agents_[0].crashed) {
      return -1.0;
    }
    if (scenario_.find("parking") == 0) {
      return -0.02 * Distance(agents_[0], goals_[0]);
    }
    return 0.02 * agents_[0].speed;
  }

  [[nodiscard]] double OfficialReward() const {
    const official::Vehicle& ego = OfficialEgo();
    if (IsParkingScenario()) {
      return OfficialParkingReward();
    }
    if (scenario_ == "lane_keeping") {
      return OfficialLaneKeepingReward();
    }
    if (scenario_.find("racetrack") == 0) {
      return OfficialRacetrackReward();
    }
    if (scenario_ == "intersection" || scenario_ == "intersection_continuous") {
      if (OfficialIntersectionArrived(ego)) {
        return 1.0;
      }
      const double speed_reward =
          Clip(LMap(ego.speed, 7.0, 9.0, 0.0, 1.0), 0.0, 1.0);
      const bool on_road =
          official_road_->network.GetLane(ego.lane_index).OnLane(ego.position);
      return (-5.0 * static_cast<double>(ego.crashed) + speed_reward) *
             static_cast<double>(on_road);
    }
    if (scenario_ == "two_way") {
      const std::vector<official::LaneIndex> neighbours =
          official_road_->network.AllSideLanes(ego.lane_index);
      const double high_speed_reward =
          static_cast<double>(ego.speed_index) /
          static_cast<double>(ego.target_speeds.size() - 1);
      const double left_lane_reward =
          (static_cast<double>(neighbours.size() - 1) -
           static_cast<double>(ego.target_lane_index.id)) /
          static_cast<double>(std::max<int>(neighbours.size() - 1, 1));
      return 0.8 * high_speed_reward + 0.2 * left_lane_reward;
    }
    if (scenario_ == "u_turn") {
      const std::vector<official::LaneIndex> neighbours =
          official_road_->network.AllSideLanes(ego.lane_index);
      const double lane_reward =
          static_cast<double>(ego.lane_index.id) /
          static_cast<double>(std::max<int>(neighbours.size() - 1, 1));
      const double speed_reward =
          Clip(LMap(ego.speed, 8.0, 24.0, 0.0, 1.0), 0.0, 1.0);
      const double on_road_reward =
          official_road_->network.GetLane(ego.lane_index).OnLane(ego.position)
              ? 1.0
              : 0.0;
      const double weighted = -1.0 * static_cast<double>(ego.crashed) +
                              0.1 * lane_reward + 0.4 * speed_reward;
      return LMap(weighted, -1.0, 0.5, 0.0, 1.0) * on_road_reward;
    }
    if (scenario_ == "roundabout") {
      const double high_speed_reward =
          static_cast<double>(ego.speed_index) /
          static_cast<double>(ego.target_speeds.size() - 1);
      const double lane_change_reward = static_cast<double>(
          official_last_action_ == 0 || official_last_action_ == 2);
      const double on_road_reward =
          official_road_->network.GetLane(ego.lane_index).OnLane(ego.position)
              ? 1.0
              : 0.0;
      const double weighted = -1.0 * static_cast<double>(ego.crashed) +
                              0.2 * high_speed_reward +
                              -0.05 * lane_change_reward;
      return LMap(weighted, -1.0, 0.2, 0.0, 1.0) * on_road_reward;
    }
    if (scenario_ == "exit") {
      const bool success =
          (ego.target_lane_index.from == "1" &&
           ego.target_lane_index.to == "2" && ego.target_lane_index.id == 6) ||
          (ego.target_lane_index.from == "2" &&
           ego.target_lane_index.to == "exit" && ego.target_lane_index.id == 0);
      const double scaled_speed =
          Clip(LMap(ego.speed, 20.0, 30.0, 0.0, 1.0), 0.0, 1.0);
      const double weighted =
          1.0 * static_cast<double>(success) + 0.1 * scaled_speed;
      return Clip(LMap(weighted, 0.0, 1.0, 0.0, 1.0), 0.0, 1.0);
    }
    const double scaled_speed = LMap(ego.speed, 20.0, 30.0, 0.0, 1.0);
    double merging_speed_reward = 0.0;
    for (const official::Vehicle& vehicle : official_road_->vehicles) {
      if (vehicle.lane_index.from == "b" && vehicle.lane_index.to == "c" &&
          vehicle.lane_index.id == 2 &&
          vehicle.kind != official::VehicleKind::kVehicle) {
        merging_speed_reward +=
            (vehicle.target_speed - vehicle.speed) / vehicle.target_speed;
      }
    }
    const double weighted =
        -1.0 * static_cast<double>(ego.crashed) +
        0.1 * static_cast<double>(ego.lane_index.id) + 0.2 * scaled_speed +
        -0.05 * static_cast<double>(official_last_action_ == 0 ||
                                    official_last_action_ == 2) +
        -0.5 * merging_speed_reward;
    return LMap(weighted, -1.0 + -0.5, 0.2 + 0.1, 0.0, 1.0);
  }

  [[nodiscard]] double OfficialIntersectionReward(
      const official::Vehicle& vehicle) const {
    if (OfficialIntersectionArrived(vehicle)) {
      return 1.0;
    }
    const double speed_reward =
        Clip(LMap(vehicle.speed, 7.0, 9.0, 0.0, 1.0), 0.0, 1.0);
    const bool on_road = official_road_->network.GetLane(vehicle.lane_index)
                             .OnLane(vehicle.position);
    return (-5.0 * static_cast<double>(vehicle.crashed) + speed_reward) *
           static_cast<double>(on_road);
  }

  [[nodiscard]] bool OfficialOnRoad(const official::Vehicle& vehicle) const {
    return official_road_->network.GetLane(vehicle.lane_index)
        .OnLane(vehicle.position);
  }

  [[nodiscard]] double OfficialLaneKeepingReward() const {
    const official::Lane& lane =
        official_road_->network.GetLane(official_active_lane_index_);
    const double lateral =
        lane.LocalCoordinates(OfficialEgo().position).lateral;
    return 1.0 - std::pow(lateral / lane.Width(), 2.0);
  }

  [[nodiscard]] double OfficialRacetrackReward() const {
    const official::Vehicle& ego = OfficialEgo();
    const official::Lane& lane =
        official_road_->network.GetLane(ego.lane_index);
    const double lateral = lane.LocalCoordinates(ego.position).lateral;
    const double lane_centering_reward = 1.0 / (1.0 + 4.0 * lateral * lateral);
    const double weighted = lane_centering_reward -
                            0.3 * official_last_continuous_action_norm_ -
                            static_cast<double>(ego.crashed);
    const double normalized = LMap(weighted, -1.0, 1.0, 0.0, 1.0);
    return normalized * static_cast<double>(OfficialOnRoad(ego));
  }

  [[nodiscard]] std::array<double, 6> OfficialBicycleDerivative(
      const official::Vehicle& vehicle,
      const std::array<double, 6>& state) const {
    constexpr double mass = 1.0;
    constexpr double length_a = official::kVehicleLength / 2.0;
    constexpr double length_b = official::kVehicleLength / 2.0;
    constexpr double inertia_z =
        (official::kVehicleLength * official::kVehicleLength +
         official::kVehicleWidth * official::kVehicleWidth) /
        12.0;
    constexpr double friction_front = 15.0 * mass;
    constexpr double friction_rear = 15.0 * mass;

    const double heading = state[2];
    double speed = state[3];
    const double lateral_speed = state[4];
    const double yaw_rate = state[5];
    const double delta_f = vehicle.action.steering;
    const double theta_vf =
        std::atan2(lateral_speed + length_a * yaw_rate, speed);
    const double theta_vr =
        std::atan2(lateral_speed - length_b * yaw_rate, speed);
    double f_yf = 2.0 * friction_front * (delta_f - theta_vf);
    double f_yr = 2.0 * friction_rear * (0.0 - theta_vr);
    if (std::abs(speed) < 1.0) {
      f_yf = -mass * lateral_speed - inertia_z / length_a * yaw_rate;
      f_yr = -mass * lateral_speed + inertia_z / length_a * yaw_rate;
    }
    const double d_lateral_speed =
        1.0 / mass * (f_yf + f_yr) - yaw_rate * speed;
    const double d_yaw_rate =
        1.0 / inertia_z * (length_a * f_yf - length_b * f_yr);
    const double c = std::cos(heading);
    const double s = std::sin(heading);
    const double speed_x = c * speed - s * lateral_speed;
    const double speed_y = s * speed + c * lateral_speed;
    return {speed_x,         speed_y,   yaw_rate, vehicle.action.acceleration,
            d_lateral_speed, d_yaw_rate};
  }

  void OfficialBicycleStep(official::Vehicle* vehicle, double dt) {
    vehicle->action.steering =
        Clip(vehicle->action.steering, -kPi / 2.0, kPi / 2.0);
    vehicle->yaw_rate = Clip(vehicle->yaw_rate, -2.0 * kPi, 2.0 * kPi);
    const std::array<double, 6> state0{
        vehicle->position.x, vehicle->position.y,    vehicle->heading,
        vehicle->speed,      vehicle->lateral_speed, vehicle->yaw_rate};
    const double half_dt = dt / 2.0;
    const std::array<double, 6> k1 =
        OfficialBicycleDerivative(*vehicle, state0);
    std::array<double, 6> state1;
    for (int i = 0; i < 6; ++i) {
      state1[i] = state0[i] + k1[i] * half_dt;
    }
    const std::array<double, 6> k2 =
        OfficialBicycleDerivative(*vehicle, state1);
    std::array<double, 6> state2;
    for (int i = 0; i < 6; ++i) {
      state2[i] = state0[i] + k2[i] * half_dt;
    }
    const std::array<double, 6> k3 =
        OfficialBicycleDerivative(*vehicle, state2);
    std::array<double, 6> state3;
    for (int i = 0; i < 6; ++i) {
      state3[i] = state0[i] + k3[i] * dt;
    }
    const std::array<double, 6> k4 =
        OfficialBicycleDerivative(*vehicle, state3);
    std::array<double, 6> new_state;
    for (int i = 0; i < 6; ++i) {
      new_state[i] =
          state0[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    vehicle->position = {new_state[0], new_state[1]};
    vehicle->heading = new_state[2];
    vehicle->speed = new_state[3];
    vehicle->lateral_speed = new_state[4];
    vehicle->yaw_rate = new_state[5];
    vehicle->UpdateLane(official_road_->network);
  }

  [[nodiscard]] std::array<double, 6> OfficialParkingAchievedGoal() const {
    const official::Vehicle& ego = OfficialEgo();
    return {ego.position.x / 100.0, ego.position.y / 100.0,
            ego.Velocity().x / 5.0, ego.Velocity().y / 5.0,
            std::cos(ego.heading),  std::sin(ego.heading)};
  }

  [[nodiscard]] std::array<double, 6> OfficialParkingDesiredGoal() const {
    const official::Vehicle& ego = OfficialEgo();
    return {ego.goal_position.x / 100.0, ego.goal_position.y / 100.0, 0.0, 0.0,
            std::cos(ego.goal_heading),  std::sin(ego.goal_heading)};
  }

  [[nodiscard]] double OfficialParkingComputeReward() const {
    const std::array<double, 6> achieved = OfficialParkingAchievedGoal();
    const std::array<double, 6> desired = OfficialParkingDesiredGoal();
    constexpr std::array<double, 6> weights{1.0, 0.3, 0.0, 0.0, 0.02, 0.02};
    double weighted_distance = 0.0;
    for (int i = 0; i < 6; ++i) {
      weighted_distance += weights[i] * std::abs(achieved[i] - desired[i]);
    }
    return -std::sqrt(weighted_distance);
  }

  [[nodiscard]] double OfficialParkingReward() const {
    return OfficialParkingComputeReward() +
           -5.0 * static_cast<double>(OfficialEgo().crashed);
  }

  [[nodiscard]] bool OfficialParkingSuccess() const {
    return OfficialParkingComputeReward() > -0.12;
  }

  [[nodiscard]] bool OfficialIntersectionArrived(
      const official::Vehicle& ego) const {
    if (ego.lane_index.from.rfind("il", 0) != 0 ||
        ego.lane_index.to.rfind("o", 0) != 0) {
      return false;
    }
    const official::Lane& lane =
        official_road_->network.GetLane(ego.lane_index);
    return lane.LocalCoordinates(ego.position).longitudinal >= 25.0;
  }

  template <typename Row>
  void WriteKinematicRow(Row obs, int row, const Agent& agent, const Agent& ego,
                         int features) {
    obs(row, 0) = 1.0f;
    obs(row, 1) = static_cast<float>((agent.x - ego.x) / 100.0);
    obs(row, 2) = static_cast<float>((agent.y - ego.y) / 100.0);
    obs(row, 3) = static_cast<float>((agent.speed * std::cos(agent.heading) -
                                      ego.speed * std::cos(ego.heading)) /
                                     40.0);
    obs(row, 4) = static_cast<float>((agent.speed * std::sin(agent.heading) -
                                      ego.speed * std::sin(ego.heading)) /
                                     40.0);
    if (features >= 7) {
      obs(row, 5) = static_cast<float>(std::cos(agent.heading));
      obs(row, 6) = static_cast<float>(std::sin(agent.heading));
    }
    if (features >= 8) {
      obs(row, 7) = static_cast<float>(agent.heading / kPi);
    }
  }

  void WriteState(float reward) {
    if constexpr (std::is_same_v<AdapterT, MultiAgentAdapter>) {
      if (UseOfficialBackend()) {
        WriteOfficialMultiAgentState(reward);
        return;
      }
      State state = this->Allocate(2);
      for (int player = 0; player < 2; ++player) {
        auto obs = state["obs:players.obs"_];
        for (int r = 0; r < 5; ++r) {
          for (int c = 0; c < 5; ++c) {
            obs(player, r, c) = 0.0f;
          }
        }
        WriteKinematicRow(obs[player], 0, agents_[player], agents_[player], 5);
        state["reward"_][player] = reward;
        state["info:players.speed"_][player] =
            static_cast<float>(agents_[player].speed);
        state["info:players.crashed"_][player] = agents_[player].crashed;
      }
    } else {
      State state = this->Allocate();
      state["reward"_] = reward;
      WriteObs(&state);
    }
  }

  void WriteOfficialMultiAgentState(std::optional<float> reset_reward = {}) {
    State state = this->Allocate(official_player_count_);
    auto obs = state["obs:players.obs"_];
    official::KinematicObservationConfig config;
    config.vehicles_count = 5;
    for (int player = 0; player < official_player_count_; ++player) {
      const official::Vehicle& vehicle = OfficialPlayer(player);
      const std::vector<float> rows =
          official::ObserveKinematics(*official_road_, vehicle, config);
      for (int r = 0; r < 5; ++r) {
        for (int c = 0; c < 5; ++c) {
          obs(player, r, c) = rows[5 * r + c];
        }
      }
      state["reward"_][player] = reset_reward.value_or(
          static_cast<float>(OfficialIntersectionReward(vehicle)));
      state["info:players.speed"_][player] = static_cast<float>(vehicle.speed);
      state["info:players.crashed"_][player] = vehicle.crashed;
    }
  }

  void WriteObs(State* state) {
    if constexpr (std::is_same_v<SpecT, NativeK5Spec>) {
      if (UseOfficialBackend()) {
        WriteOfficialKinematics(state);
        return;
      }
      WriteKinematics(state, 5, 5);
      return;
    }
    if constexpr (std::is_same_v<SpecT, NativeK75Spec>) {
      if (scenario_ == "exit") {
        WriteOfficialExitKinematics(state);
        return;
      }
      WriteKinematics(state, 15, 7);
      return;
    }
    if constexpr (std::is_same_v<SpecT, NativeK73Spec>) {
      if (scenario_ == "intersection") {
        WriteOfficialIntersectionKinematics(state);
        return;
      }
      WriteKinematics(state, 15, 7);
      return;
    }
    if constexpr (std::is_same_v<SpecT, NativeK8CSpec>) {
      if (scenario_ == "intersection_continuous") {
        WriteOfficialIntersectionContinuousKinematics(state);
        return;
      }
      WriteKinematics8(state);
      return;
    }
    if constexpr (std::is_same_v<SpecT, NativeGoalSpec>) {
      if (UseOfficialBackend()) {
        WriteOfficialGoal(state);
        return;
      }
      WriteGoal(state);
      return;
    }
    if constexpr (std::is_same_v<SpecT, NativeAttributesSpec>) {
      if (scenario_ == "lane_keeping") {
        WriteOfficialAttributes(state);
        return;
      }
      WriteAttributes(state);
      return;
    }
    if constexpr (std::is_same_v<SpecT, NativeOccupancySpec>) {
      if (scenario_.find("racetrack") == 0) {
        WriteOfficialOccupancy(state);
        return;
      }
      WriteOccupancy(state);
      return;
    }
    if constexpr (std::is_same_v<SpecT, NativeTTC5Spec>) {
      if (UseOfficialBackend()) {
        WriteOfficialTTC<5>(state);
        return;
      }
      WriteTTC<5>(state);
      return;
    }
    if constexpr (std::is_same_v<SpecT, NativeTTC16Spec>) {
      if (UseOfficialBackend()) {
        WriteOfficialTTC<16>(state);
        return;
      }
      WriteTTC<16>(state);
    }
  }

  void WriteKinematics(State* state, int rows, int features) {
    auto obs = (*state)["obs"_];
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < features; ++c) {
        obs(r, c) = 0.0f;
      }
    }
    WriteKinematicRow(obs, 0, agents_[0], agents_[0], features);
    const int n = std::min(static_cast<int>(traffic_.size()), rows - 1);
    for (int i = 0; i < n; ++i) {
      WriteKinematicRow(obs, i + 1, traffic_[i], agents_[0], features);
    }
    (*state)["info:speed"_] = static_cast<float>(agents_[0].speed);
    (*state)["info:crashed"_] = agents_[0].crashed;
  }

  void WriteKinematics8(State* state) { WriteKinematics(state, 5, 8); }

  void WriteOfficialKinematics(State* state) {
    official::KinematicObservationConfig config;
    config.vehicles_count = 5;
    config.features = {
        official::KinematicFeature::kPresence, official::KinematicFeature::kX,
        official::KinematicFeature::kY,        official::KinematicFeature::kVx,
        official::KinematicFeature::kVy,
    };
    if (scenario_ == "roundabout") {
      config.absolute = true;
      config.x_min = -100.0;
      config.x_max = 100.0;
      config.y_min = -100.0;
      config.y_max = 100.0;
      config.vx_min = -15.0;
      config.vx_max = 15.0;
      config.vy_min = -15.0;
      config.vy_max = 15.0;
    } else if (scenario_ == "merge") {
      config.y_min = -2.0 * kLaneWidth;
      config.y_max = 2.0 * kLaneWidth;
    }
    const std::vector<float> rows =
        official::ObserveKinematics(*official_road_, OfficialEgo(), config);
    auto obs = (*state)["obs"_];
    for (int r = 0; r < 5; ++r) {
      for (int c = 0; c < 5; ++c) {
        obs(r, c) = rows[5 * r + c];
      }
    }
    (*state)["info:speed"_] = static_cast<float>(OfficialEgo().speed);
    (*state)["info:crashed"_] = OfficialEgo().crashed;
  }

  void WriteOfficialExitKinematics(State* state) {
    official::KinematicObservationConfig config;
    config.vehicles_count = 15;
    config.features = {
        official::KinematicFeature::kPresence,
        official::KinematicFeature::kX,
        official::KinematicFeature::kY,
        official::KinematicFeature::kVx,
        official::KinematicFeature::kVy,
        official::KinematicFeature::kCosH,
        official::KinematicFeature::kSinH,
    };
    config.clip = false;
    config.y_min = -24.0;
    config.y_max = 24.0;
    const official::Lane& exit_pre_lane =
        official_road_->network.GetLane({"1", "2", 6});
    config.ego_x_override =
        exit_pre_lane.LocalCoordinates(OfficialEgo().position).longitudinal;
    const std::vector<float> rows =
        official::ObserveKinematics(*official_road_, OfficialEgo(), config);
    auto obs = (*state)["obs"_];
    for (int r = 0; r < 15; ++r) {
      for (int c = 0; c < 7; ++c) {
        obs(r, c) = rows[7 * r + c];
      }
    }
    (*state)["info:speed"_] = static_cast<float>(OfficialEgo().speed);
    (*state)["info:crashed"_] = OfficialEgo().crashed;
  }

  void WriteOfficialIntersectionKinematics(State* state) {
    official::KinematicObservationConfig config;
    config.vehicles_count = 15;
    config.features = {
        official::KinematicFeature::kPresence,
        official::KinematicFeature::kX,
        official::KinematicFeature::kY,
        official::KinematicFeature::kVx,
        official::KinematicFeature::kVy,
        official::KinematicFeature::kCosH,
        official::KinematicFeature::kSinH,
    };
    config.absolute = true;
    config.x_min = -100.0;
    config.x_max = 100.0;
    config.y_min = -100.0;
    config.y_max = 100.0;
    config.vx_min = -20.0;
    config.vx_max = 20.0;
    config.vy_min = -20.0;
    config.vy_max = 20.0;
    config.include_obstacles = false;
    const std::vector<float> rows =
        official::ObserveKinematics(*official_road_, OfficialEgo(), config);
    auto obs = (*state)["obs"_];
    for (int r = 0; r < 15; ++r) {
      for (int c = 0; c < 7; ++c) {
        obs(r, c) = rows[7 * r + c];
      }
    }
    (*state)["info:speed"_] = static_cast<float>(OfficialEgo().speed);
    (*state)["info:crashed"_] = OfficialEgo().crashed;
  }

  void WriteOfficialIntersectionContinuousKinematics(State* state) {
    official::KinematicObservationConfig config;
    config.vehicles_count = 5;
    config.features = {
        official::KinematicFeature::kPresence,
        official::KinematicFeature::kX,
        official::KinematicFeature::kY,
        official::KinematicFeature::kVx,
        official::KinematicFeature::kVy,
        official::KinematicFeature::kLongOff,
        official::KinematicFeature::kLatOff,
        official::KinematicFeature::kAngOff,
    };
    const std::vector<float> rows =
        official::ObserveKinematics(*official_road_, OfficialEgo(), config);
    auto obs = (*state)["obs"_];
    for (int r = 0; r < 5; ++r) {
      for (int c = 0; c < 8; ++c) {
        obs(r, c) = rows[8 * r + c];
      }
    }
    (*state)["info:speed"_] = static_cast<float>(OfficialEgo().speed);
    (*state)["info:crashed"_] = OfficialEgo().crashed;
  }

  template <int Horizon>
  void WriteOfficialTTC(State* state) {
    const std::vector<float> rows = official::ObserveTimeToCollision(
        *official_road_, OfficialEgo(),
        1.0 / static_cast<double>(policy_frequency_),
        static_cast<double>(Horizon));
    auto obs = (*state)["obs"_];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < Horizon; ++k) {
          obs(i, j, k) = rows[(i * 3 + j) * Horizon + k];
        }
      }
    }
    (*state)["info:speed"_] = static_cast<float>(OfficialEgo().speed);
    (*state)["info:crashed"_] = OfficialEgo().crashed;
  }

  void WriteGoal(State* state) {
    auto observation = (*state)["obs:observation"_];
    auto achieved_goal = (*state)["obs:achieved_goal"_];
    auto desired_goal = (*state)["obs:desired_goal"_];
    const std::array<double, 6> achieved = {
        agents_[0].x / 100.0,
        agents_[0].y / 100.0,
        agents_[0].speed * std::cos(agents_[0].heading) / 5.0,
        agents_[0].speed * std::sin(agents_[0].heading) / 5.0,
        std::cos(agents_[0].heading),
        std::sin(agents_[0].heading)};
    const std::array<double, 6> desired = {
        goals_[0].x / 100.0,         goals_[0].y / 100.0,        0.0, 0.0,
        std::cos(goals_[0].heading), std::sin(goals_[0].heading)};
    for (int i = 0; i < 6; ++i) {
      observation[i] = achieved[i];
      achieved_goal[i] = achieved[i];
      desired_goal[i] = desired[i];
    }
    (*state)["info:is_success"_] = Distance(agents_[0], goals_[0]) < 3.0;
    (*state)["info:speed"_] = static_cast<float>(agents_[0].speed);
    (*state)["info:crashed"_] = agents_[0].crashed;
  }

  void WriteOfficialGoal(State* state) {
    auto observation = (*state)["obs:observation"_];
    auto achieved_goal = (*state)["obs:achieved_goal"_];
    auto desired_goal = (*state)["obs:desired_goal"_];
    const std::array<double, 6> achieved = OfficialParkingAchievedGoal();
    const std::array<double, 6> desired = OfficialParkingDesiredGoal();
    for (int i = 0; i < 6; ++i) {
      observation[i] = achieved[i];
      achieved_goal[i] = achieved[i];
      desired_goal[i] = desired[i];
    }
    (*state)["info:is_success"_] = OfficialParkingSuccess();
    (*state)["info:speed"_] = static_cast<float>(OfficialEgo().speed);
    (*state)["info:crashed"_] = OfficialEgo().crashed;
  }

  void WriteAttributes(State* state) {
    auto s = (*state)["obs:state"_];
    auto d = (*state)["obs:derivative"_];
    auto r = (*state)["obs:reference_state"_];
    const std::array<double, 4> state_values = {
        agents_[0].y, agents_[0].heading, agents_[0].speed, 0.0};
    const std::array<double, 4> deriv_values = {
        agents_[0].speed * std::sin(agents_[0].heading), agents_[0].heading,
        0.0, 0.0};
    for (int i = 0; i < 4; ++i) {
      s(i, 0) = state_values[i];
      d(i, 0) = deriv_values[i];
      r(i, 0) = 0.0;
    }
  }

  void WriteOfficialAttributes(State* state) {
    auto s = (*state)["obs:state"_];
    auto d = (*state)["obs:derivative"_];
    auto r = (*state)["obs:reference_state"_];
    const official::Vehicle& ego = OfficialEgo();
    const std::array<double, 6> bicycle_state{ego.position.x,    ego.position.y,
                                              ego.heading,       ego.speed,
                                              ego.lateral_speed, ego.yaw_rate};
    const std::array<double, 6> derivative =
        OfficialBicycleDerivative(ego, bicycle_state);
    const official::Lane& lane =
        official_road_->network.GetLane(official_active_lane_index_);
    const official::LaneCoordinates local = lane.LocalCoordinates(ego.position);
    const double reference_heading = lane.HeadingAt(local.longitudinal);
    const double reference_y = ego.position.y - local.lateral;
    const std::array<double, 4> state_values{ego.position.y, ego.heading,
                                             ego.lateral_speed, ego.yaw_rate};
    const std::array<double, 4> derivative_values{derivative[1], derivative[2],
                                                  derivative[4], derivative[5]};
    const std::array<double, 4> reference_values{reference_y, reference_heading,
                                                 0.0, 0.0};
    for (int i = 0; i < 4; ++i) {
      s(i, 0) = state_values[i];
      d(i, 0) = derivative_values[i];
      r(i, 0) = reference_values[i];
    }
  }

  void WriteOccupancy(State* state) {
    auto obs = (*state)["obs"_];
    for (int c = 0; c < 2; ++c) {
      for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 12; ++j) {
          obs(c, i, j) = c == 1 ? 1.0f : 0.0f;
        }
      }
    }
    obs(0, 6, 6) = 1.0f;
    (*state)["info:speed"_] = static_cast<float>(agents_[0].speed);
    (*state)["info:crashed"_] = agents_[0].crashed;
  }

  [[nodiscard]] std::pair<int, int> OfficialOccupancyIndex(
      official::Vec2 position) const {
    const official::Vehicle& ego = OfficialEgo();
    position = position - ego.position;
    const double c = std::cos(ego.heading);
    const double s = std::sin(ego.heading);
    const double aligned_x = c * position.x + s * position.y;
    const double aligned_y = -s * position.x + c * position.y;
    return {static_cast<int>(std::floor((aligned_x + 18.0) / 3.0)),
            static_cast<int>(std::floor((aligned_y + 18.0) / 3.0))};
  }

  template <typename Obs>
  void SetOfficialOccupancy(Obs obs, int layer, official::Vec2 position,
                            float value) const {
    const auto [i, j] = OfficialOccupancyIndex(position);
    if (0 <= i && i < 12 && 0 <= j && j < 12) {
      obs(layer, i, j) = value;
    }
  }

  void WriteOfficialOccupancy(State* state) {
    auto obs = (*state)["obs"_];
    for (int c = 0; c < 2; ++c) {
      for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 12; ++j) {
          obs(c, i, j) = 0.0f;
        }
      }
    }

    for (int index = static_cast<int>(official_road_->vehicles.size()) - 1;
         index >= 0; --index) {
      SetOfficialOccupancy(obs, 0, official_road_->vehicles[index].position,
                           1.0f);
    }

    constexpr double lane_waypoints_spacing = 3.0;
    for (const official::Lane* lane : official_road_->network.Lanes()) {
      const double origin =
          lane->LocalCoordinates(OfficialEgo().position).longitudinal;
      for (double waypoint = origin - 100.0; waypoint < origin + 100.0;
           waypoint += lane_waypoints_spacing) {
        const double clipped = Clip(waypoint, 0.0, lane->Length());
        SetOfficialOccupancy(obs, 1, lane->Position(clipped, 0.0), 1.0f);
      }
    }
    (*state)["info:speed"_] = static_cast<float>(OfficialEgo().speed);
    (*state)["info:crashed"_] = OfficialEgo().crashed;
  }

  template <int Horizon>
  void WriteTTC(State* state) {
    auto obs = (*state)["obs"_];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < Horizon; ++k) {
          obs(i, j, k) = 0.0f;
        }
      }
    }
    obs(1, 1, 0) = 1.0f;
    (*state)["info:speed"_] = static_cast<float>(agents_[0].speed);
    (*state)["info:crashed"_] = agents_[0].crashed;
  }

  [[nodiscard]] std::pair<int, int> WorldToScreen(int width, int height,
                                                  double x, double y) const {
    const double scale = scenario_ == "intersection" ||
                                 scenario_ == "intersection_multi" ||
                                 scenario_ == "intersection_continuous"
                             ? 5.0
                             : 4.0;
    return {static_cast<int>(std::lround(width * 0.5 + x * scale)),
            static_cast<int>(std::lround(height * 0.5 + y * scale))};
  }

  struct OfficialRenderTransform {
    double scaling{5.5};
    double origin_x{0.0};
    double origin_y{0.0};

    [[nodiscard]] int Pix(double length) const {
      return native::Pix(length, scaling);
    }

    [[nodiscard]] std::pair<int, int> Pos2Pix(official::Vec2 position) const {
      return {Pix(position.x - origin_x), Pix(position.y - origin_y)};
    }
  };

  [[nodiscard]] double OfficialRenderScaling() const {
    if (IsParkingScenario()) {
      return 7.0;
    }
    if (scenario_ == "exit") {
      return 5.0;
    }
    if (scenario_ == "lane_keeping") {
      return 7.0;
    }
    if (scenario_ == "intersection" || scenario_ == "intersection_multi" ||
        scenario_ == "intersection_continuous") {
      return 7.15;
    }
    return 5.5;
  }

  [[nodiscard]] std::pair<double, double> OfficialRenderCentering() const {
    if (IsParkingScenario()) {
      return {0.5, 0.5};
    }
    if (scenario_ == "intersection" || scenario_ == "intersection_multi" ||
        scenario_ == "intersection_continuous" || scenario_ == "roundabout") {
      return {0.5, 0.6};
    }
    if (scenario_.find("racetrack") == 0) {
      return {0.5, 0.5};
    }
    if (scenario_ == "lane_keeping") {
      return {0.4, 0.5};
    }
    return {0.3, 0.5};
  }

  [[nodiscard]] OfficialRenderTransform OfficialRenderCamera(int width,
                                                             int height) const {
    const double scaling = OfficialRenderScaling();
    const auto [center_x, center_y] = OfficialRenderCentering();
    const official::Vec2 focus = OfficialEgo().position;
    return {scaling, focus.x - center_x * static_cast<double>(width) / scaling,
            focus.y - center_y * static_cast<double>(height) / scaling};
  }

  void DrawOfficialStripe(unsigned char* rgb, int width, int height,
                          const OfficialRenderTransform& transform,
                          const official::Lane& lane, double start, double end,
                          double lateral) const {
    start = Clip(start, 0.0, lane.Length());
    end = Clip(end, 0.0, lane.Length());
    if (std::abs(start - end) <= 1.5) {
      return;
    }
    const auto [x0, y0] = transform.Pos2Pix(lane.Position(start, lateral));
    const auto [x1, y1] = transform.Pos2Pix(lane.Position(end, lateral));
    DrawLine(rgb, width, height, x0, y0, x1, y1, 255, 255, 255,
             std::max(1, transform.Pix(0.3)));
  }

  void DrawOfficialLane(unsigned char* rgb, int width, int height,
                        const OfficialRenderTransform& transform,
                        const official::Lane& lane) const {
    constexpr double kStripeSpacing = 4.33;
    constexpr double kStripeLength = 3.0;
    const int stripes_count =
        static_cast<int>(2.0 * static_cast<double>(height + width) /
                         (kStripeSpacing * transform.scaling));
    const official::LaneCoordinates origin_coordinates =
        lane.LocalCoordinates({transform.origin_x, transform.origin_y});
    const double s0 = (std::floor(static_cast<double>(static_cast<int>(
                                      origin_coordinates.longitudinal)) /
                                  kStripeSpacing) -
                       static_cast<double>(stripes_count / 2)) *
                      kStripeSpacing;
    const auto line_types = lane.LineTypes();
    for (int side = 0; side < 2; ++side) {
      const double lateral = (static_cast<double>(side) - 0.5) * lane.Width();
      if (line_types[side] == official::LineType::kStriped) {
        for (int k = 0; k < stripes_count; ++k) {
          const double start = s0 + static_cast<double>(k) * kStripeSpacing;
          DrawOfficialStripe(rgb, width, height, transform, lane, start,
                             start + kStripeLength, lateral);
        }
      } else if (line_types[side] == official::LineType::kContinuous) {
        for (int k = 0; k < stripes_count; ++k) {
          const double start = s0 + static_cast<double>(k) * kStripeSpacing;
          DrawOfficialStripe(rgb, width, height, transform, lane, start,
                             start + kStripeSpacing, lateral);
        }
      } else if (line_types[side] == official::LineType::kContinuousLine) {
        DrawOfficialStripe(rgb, width, height, transform, lane, s0,
                           s0 + stripes_count * kStripeSpacing + kStripeLength,
                           lateral);
      }
    }
  }

  void DrawOfficialRoad(unsigned char* rgb, int width, int height,
                        const OfficialRenderTransform& transform) const {
    for (const official::Lane* lane : official_road_->network.Lanes()) {
      DrawOfficialLane(rgb, width, height, transform, *lane);
    }
  }

  void DrawOfficialObject(unsigned char* rgb, int width, int height,
                          const OfficialRenderTransform& transform,
                          const official::RoadObject& object) const {
    const bool landmark = object.kind == official::RoadObjectKind::kLandmark;
    const bool red_obstacle =
        object.kind == official::RoadObjectKind::kObstacle && object.crashed;
    const bool green_landmark = landmark && object.hit;
    std::uint8_t r = landmark ? 100 : 200;
    std::uint8_t g = landmark ? 200 : 200;
    std::uint8_t b = landmark ? 255 : 0;
    if (red_obstacle) {
      r = 255;
      g = 100;
      b = 100;
    }
    if (green_landmark) {
      r = 50;
      g = 200;
      b = 0;
    }
    const auto object_pixel_position = transform.Pos2Pix(object.position);
    const int object_px = object_pixel_position.first;
    const int object_py = object_pixel_position.second;
    const int sprite_px = transform.Pix(object.length);
    const int rect_y = transform.Pix(object.length / 2.0 - object.width / 2.0);
    const int rect_h = transform.Pix(object.width);
    const double heading =
        std::abs(object.heading) > 2.0 * kPi / 180.0 ? object.heading : 0.0;
    Sprite object_sprite(sprite_px);
    FillSpriteRect(&object_sprite, 0, rect_y, sprite_px, rect_h, r, g, b);
    DrawSpriteRectOutline(&object_sprite, 0, rect_y, sprite_px, rect_h, 60, 60,
                          60);
    BlitRotatedSprite(rgb, width, height, object_sprite, object_px, object_py,
                      heading);
  }

  void DrawOfficialVehicle(unsigned char* rgb, int width, int height,
                           const OfficialRenderTransform& transform,
                           const official::Vehicle& vehicle) const {
    const auto [vehicle_px, vehicle_py] = transform.Pos2Pix(vehicle.position);
    const int cx = vehicle_px;
    const int cy = vehicle_py;
    if (cx < -80 || cx > width + 80 || cy < -80 || cy > height + 80) {
      return;
    }
    std::uint8_t r = 200;
    std::uint8_t g = 200;
    std::uint8_t b = 0;
    if (vehicle.crashed) {
      r = 255;
      g = 100;
      b = 100;
    } else if (vehicle.kind == official::VehicleKind::kIDM) {
      r = 100;
      g = 200;
      b = 255;
    } else if (vehicle.kind == official::VehicleKind::kMDP) {
      r = 50;
      g = 200;
      b = 0;
    }
    constexpr double tire_length = 1.0;
    constexpr double tire_width = 0.3;
    constexpr double headlight_length = 0.72;
    constexpr double headlight_width = 0.6;
    constexpr double sprite_length =
        official::kVehicleLength + 2.0 * tire_length;
    const int sprite_px = transform.Pix(sprite_length);
    const int body_x = transform.Pix(tire_length);
    const int body_y =
        transform.Pix(sprite_length / 2.0 - official::kVehicleWidth / 2.0);
    const int body_w = transform.Pix(official::kVehicleLength);
    const int body_h = transform.Pix(official::kVehicleWidth);
    const int light_x = transform.Pix(tire_length + official::kVehicleLength -
                                      headlight_length);
    const int light_left_y = transform.Pix(
        sprite_length / 2.0 - (1.4 * official::kVehicleWidth) / 3.0);
    const int light_right_y = transform.Pix(
        sprite_length / 2.0 + (0.6 * official::kVehicleWidth) / 5.0);
    const int light_w = transform.Pix(headlight_length);
    const int light_h = transform.Pix(headlight_width);
    const double heading =
        std::abs(vehicle.heading) > 2.0 * kPi / 180.0 ? vehicle.heading : 0.0;
    Sprite vehicle_sprite(sprite_px);
    FillSpriteRect(&vehicle_sprite, body_x, body_y, body_w, body_h, r, g, b);
    FillSpriteRect(&vehicle_sprite, light_x, light_left_y, light_w, light_h,
                   Lighten(r), Lighten(g), Lighten(b));
    FillSpriteRect(&vehicle_sprite, light_x, light_right_y, light_w, light_h,
                   Lighten(r), Lighten(g), Lighten(b));
    DrawSpriteRectOutline(&vehicle_sprite, body_x, body_y, body_w, body_h, 60,
                          60, 60);

    if (vehicle.kind == official::VehicleKind::kVehicle) {
      const int tire_px = transform.Pix(tire_length);
      const int tire_h = transform.Pix(tire_width);
      const int tire_local_y =
          transform.Pix(tire_length / 2.0 - tire_width / 2.0);
      const std::array<std::pair<int, int>, 4> tire_positions{{
          {transform.Pix(tire_length),
           transform.Pix(sprite_length / 2.0 - official::kVehicleWidth / 2.0)},
          {transform.Pix(tire_length),
           transform.Pix(sprite_length / 2.0 + official::kVehicleWidth / 2.0)},
          {transform.Pix(sprite_length - tire_length),
           transform.Pix(sprite_length / 2.0 - official::kVehicleWidth / 2.0)},
          {transform.Pix(sprite_length - tire_length),
           transform.Pix(sprite_length / 2.0 + official::kVehicleWidth / 2.0)},
      }};
      for (const auto& [tire_x, tire_y] : tire_positions) {
        FillSpriteRect(&vehicle_sprite, tire_x - tire_px / 2,
                       tire_y - tire_px / 2 + tire_local_y, tire_px, tire_h, 60,
                       60, 60);
      }
    }
    BlitRotatedSprite(rgb, width, height, vehicle_sprite, cx, cy, heading);
  }

  void DrawScenario(unsigned char* rgb, int width, int height) const {
    const int cx = width / 2;
    const int cy = height / 2;
    if (scenario_ == "intersection" || scenario_ == "intersection_multi" ||
        scenario_ == "intersection_continuous") {
      FillRect(rgb, width, height, cx - 28, 0, 56, height, 70, 70, 70);
      FillRect(rgb, width, height, 0, cy - 28, width, 56, 70, 70, 70);
      DrawLine(rgb, width, height, cx, 0, cx, height, 255, 255, 255, 2);
      DrawLine(rgb, width, height, 0, cy, width, cy, 255, 255, 255, 2);
      return;
    }
    if (scenario_ == "roundabout") {
      DrawCircle(rgb, width, height, cx, cy, 125, 70, 70, 70, 55);
      DrawCircle(rgb, width, height, cx, cy, 95, 255, 255, 255, 2);
      DrawCircle(rgb, width, height, cx, cy, 150, 255, 255, 255, 2);
      return;
    }
    if (scenario_.find("parking") == 0) {
      for (int i = -4; i <= 4; ++i) {
        const int x = cx + i * 36;
        DrawLine(rgb, width, height, x, cy - 90, x, cy - 35, 255, 255, 255);
        DrawLine(rgb, width, height, x, cy + 35, x, cy + 90, 255, 255, 255);
      }
      DrawLine(rgb, width, height, 0, cy, width, cy, 255, 255, 255, 2);
      return;
    }
    if (scenario_.find("racetrack") == 0) {
      DrawCircle(rgb, width, height, cx, cy, 190, 70, 70, 70, 75);
      DrawCircle(rgb, width, height, cx, cy, 150, 255, 255, 255, 2);
      DrawCircle(rgb, width, height, cx, cy, 225, 255, 255, 255, 2);
      return;
    }
    if (scenario_ == "u_turn") {
      DrawLine(rgb, width, height, 0, cy - 30, width - 170, cy - 30, 255, 255,
               255, 2);
      DrawLine(rgb, width, height, 0, cy + 30, width - 170, cy + 30, 255, 255,
               255, 2);
      DrawCircle(rgb, width, height, width - 170, cy, 60, 255, 255, 255, 2);
      return;
    }
    for (int lane = -1; lane <= 1; ++lane) {
      const int y = cy + lane * 24;
      DrawLine(rgb, width, height, 0, y, width, y, 255, 255, 255,
               lane == -1 || lane == 1 ? 2 : 1);
    }
    if (scenario_ == "merge" || scenario_ == "exit") {
      DrawLine(rgb, width, height, width / 2, cy + 70, width - 40, cy + 24, 255,
               255, 255, 2);
    }
  }

  void DrawAgent(unsigned char* rgb, int width, int height, const Agent& agent,
                 std::uint8_t r, std::uint8_t g, std::uint8_t b) const {
    auto [x, y] = WorldToScreen(width, height, agent.x, agent.y);
    FillRect(rgb, width, height, x - 12, y - 6, 24, 12, r, g, b);
    const int nose_x =
        x + static_cast<int>(std::lround(16.0 * std::cos(agent.heading)));
    const int nose_y =
        y + static_cast<int>(std::lround(16.0 * std::sin(agent.heading)));
    DrawLine(rgb, width, height, x, y, nose_x, nose_y, 30, 30, 30, 2);
  }
};

using NativeK5Env = NativeTaskEnv<NativeK5Spec, Discrete5Adapter>;
using NativeK75Env = NativeTaskEnv<NativeK75Spec, Discrete5Adapter>;
using NativeK73Env = NativeTaskEnv<NativeK73Spec, Discrete3Adapter>;
using NativeK8CEnv = NativeTaskEnv<NativeK8CSpec, Continuous2Adapter>;
using NativeTTC5Env = NativeTaskEnv<NativeTTC5Spec, Discrete5Adapter>;
using NativeTTC16Env = NativeTaskEnv<NativeTTC16Spec, Discrete5Adapter>;
using NativeGoalEnv = NativeTaskEnv<NativeGoalSpec, Continuous2Adapter>;
using NativeAttrsEnv = NativeTaskEnv<NativeAttributesSpec, Continuous1Adapter>;
using NativeOccEnv = NativeTaskEnv<NativeOccupancySpec, Continuous1Adapter>;
using NativeMultiEnv = NativeTaskEnv<NativeMultiAgentSpec, MultiAgentAdapter>;

template <typename EnvT>
class NativeTaskPool : public AsyncEnvPool<EnvT> {
 public:
  using AsyncEnvPool<EnvT>::AsyncEnvPool;

  [[nodiscard]] std::vector<HighwayDebugState> DebugStates(
      const std::vector<int>& env_ids) const {
    std::vector<HighwayDebugState> states;
    states.reserve(env_ids.size());
    for (int env_id : env_ids) {
      states.emplace_back(this->envs_[env_id]->DebugState());
    }
    return states;
  }
};

using NativeK5Pool = NativeTaskPool<NativeK5Env>;
using NativeK75Pool = NativeTaskPool<NativeK75Env>;
using NativeK73Pool = NativeTaskPool<NativeK73Env>;
using NativeK8CPool = NativeTaskPool<NativeK8CEnv>;
using NativeTTC5Pool = NativeTaskPool<NativeTTC5Env>;
using NativeTTC16Pool = NativeTaskPool<NativeTTC16Env>;
using NativeGoalPool = NativeTaskPool<NativeGoalEnv>;
using NativeAttributesPool = NativeTaskPool<NativeAttrsEnv>;
using NativeOccupancyPool = NativeTaskPool<NativeOccEnv>;
using NativeMultiAgentPool = NativeTaskPool<NativeMultiEnv>;

using NativeKinematics5Env = NativeK5Env;
using NativeKinematics7Action5Env = NativeK75Env;
using NativeKinematics7Action3Env = NativeK73Env;
using NativeKinematics8ContinuousEnv = NativeK8CEnv;
using NativeAttributesEnv = NativeAttrsEnv;
using NativeOccupancyEnv = NativeOccEnv;
using NativeMultiAgentEnv = NativeMultiEnv;
using NativeKinematics5EnvPool = NativeK5Pool;
using NativeKinematics7Action5EnvPool = NativeK75Pool;
using NativeKinematics7Action3EnvPool = NativeK73Pool;
using NativeKinematics8ContinuousEnvPool = NativeK8CPool;

}  // namespace native
}  // namespace highway

#endif  // ENVPOOL_HIGHWAY_NATIVE_TASK_ENV_H_
