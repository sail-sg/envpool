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
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/highway/highway_env.h"

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

inline void FillRect(unsigned char* rgb, int width, int height, int x, int y,
                     int w, int h, std::uint8_t r, std::uint8_t g,
                     std::uint8_t b) {
  for (int yy = std::max(y, 0); yy < std::min(y + h, height); ++yy) {
    for (int xx = std::max(x, 0); xx < std::min(x + w, width); ++xx) {
      SetPixel(rgb, width, height, xx, yy, r, g, b);
    }
  }
}

inline void DrawLine(unsigned char* rgb, int width, int height, int x0, int y0,
                     int x1, int y1, std::uint8_t r, std::uint8_t g,
                     std::uint8_t b, int thickness = 1) {
  const int dx = x1 - x0;
  const int dy = y1 - y0;
  const int steps = std::max(std::abs(dx), std::abs(dy));
  const int radius = std::max(0, thickness / 2);
  if (steps == 0) {
    SetPixel(rgb, width, height, x0, y0, r, g, b);
    return;
  }
  for (int i = 0; i <= steps; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(steps);
    const int x = static_cast<int>(std::lround(x0 + t * dx));
    const int y = static_cast<int>(std::lround(y0 + t * dy));
    FillRect(rgb, width, height, x - radius, y - radius, 2 * radius + 1,
             2 * radius + 1, r, g, b);
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
    ResetAgents();
    ResetTraffic();
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
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
    if (agents_[0].crashed) {
      return -1.0;
    }
    if (scenario_.find("parking") == 0) {
      return -0.02 * Distance(agents_[0], goals_[0]);
    }
    return 0.02 * agents_[0].speed;
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

  void WriteObs(State* state) {
    if constexpr (std::is_same_v<SpecT, NativeK5Spec>) {
      WriteKinematics(state, 5, 5);
      return;
    }
    if constexpr (std::is_same_v<SpecT, NativeK75Spec>) {
      WriteKinematics(state, 15, 7);
      return;
    }
    if constexpr (std::is_same_v<SpecT, NativeK73Spec>) {
      WriteKinematics(state, 15, 7);
      return;
    }
    if constexpr (std::is_same_v<SpecT, NativeK8CSpec>) {
      WriteKinematics8(state);
      return;
    }
    if constexpr (std::is_same_v<SpecT, NativeGoalSpec>) {
      WriteGoal(state);
      return;
    }
    if constexpr (std::is_same_v<SpecT, NativeAttributesSpec>) {
      WriteAttributes(state);
      return;
    }
    if constexpr (std::is_same_v<SpecT, NativeOccupancySpec>) {
      WriteOccupancy(state);
      return;
    }
    if constexpr (std::is_same_v<SpecT, NativeTTC5Spec>) {
      WriteTTC<5>(state);
      return;
    }
    if constexpr (std::is_same_v<SpecT, NativeTTC16Spec>) {
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

using NativeK5Pool = AsyncEnvPool<NativeK5Env>;
using NativeK75Pool = AsyncEnvPool<NativeK75Env>;
using NativeK73Pool = AsyncEnvPool<NativeK73Env>;
using NativeK8CPool = AsyncEnvPool<NativeK8CEnv>;
using NativeTTC5Pool = AsyncEnvPool<NativeTTC5Env>;
using NativeTTC16Pool = AsyncEnvPool<NativeTTC16Env>;
using NativeGoalPool = AsyncEnvPool<NativeGoalEnv>;
using NativeAttributesPool = AsyncEnvPool<NativeAttrsEnv>;
using NativeOccupancyPool = AsyncEnvPool<NativeOccEnv>;
using NativeMultiAgentPool = AsyncEnvPool<NativeMultiEnv>;

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
