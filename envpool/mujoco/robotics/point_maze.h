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

#ifndef ENVPOOL_MUJOCO_ROBOTICS_POINT_MAZE_H_
#define ENVPOOL_MUJOCO_ROBOTICS_POINT_MAZE_H_

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/robotics/mujoco_env.h"
#include "envpool/mujoco/robotics/utils.h"

namespace gymnasium_robotics {

class PointMazeEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("reward_threshold"_.Bind(0.0), "frame_skip"_.Bind(1),
                    "frame_stack"_.Bind(1),
                    "maze_map"_.Bind(std::string("U_MAZE")),
                    "reward_type"_.Bind(std::string("sparse")),
                    "continuing_task"_.Bind(true), "reset_target"_.Bind(false),
                    "maze_size_scaling"_.Bind(1.0), "maze_height"_.Bind(0.4),
                    "position_noise_range"_.Bind(0.25));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
#ifdef ENVPOOL_TEST
    return MakeDict("obs:observation"_.Bind(StackSpec(
                        Spec<mjtNum>({4}, {-inf, inf}), conf["frame_stack"_])),
                    "obs:achieved_goal"_.Bind(StackSpec(
                        Spec<mjtNum>({2}, {-inf, inf}), conf["frame_stack"_])),
                    "obs:desired_goal"_.Bind(StackSpec(
                        Spec<mjtNum>({2}, {-inf, inf}), conf["frame_stack"_])),
                    "info:success"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
                    "info:distance"_.Bind(Spec<mjtNum>({-1}, {0.0, inf})),
                    "info:qpos0"_.Bind(Spec<mjtNum>({2})),
                    "info:qvel0"_.Bind(Spec<mjtNum>({2})),
                    "info:goal0"_.Bind(Spec<mjtNum>({2})));
#else
    return MakeDict("obs:observation"_.Bind(StackSpec(
                        Spec<mjtNum>({4}, {-inf, inf}), conf["frame_stack"_])),
                    "obs:achieved_goal"_.Bind(StackSpec(
                        Spec<mjtNum>({2}, {-inf, inf}), conf["frame_stack"_])),
                    "obs:desired_goal"_.Bind(StackSpec(
                        Spec<mjtNum>({2}, {-inf, inf}), conf["frame_stack"_])),
                    "info:success"_.Bind(Spec<mjtNum>({-1}, {0.0, 1.0})),
                    "info:distance"_.Bind(Spec<mjtNum>({-1}, {0.0, inf})));
#endif
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<float>({-1, 2}, {-1.0, 1.0})));
  }
};

using PointMazeEnvSpec = EnvSpec<PointMazeEnvFns>;
using PointMazePixelEnvFns = PixelObservationEnvFns<PointMazeEnvFns>;
using PointMazePixelEnvSpec = EnvSpec<PointMazePixelEnvFns>;

template <typename EnvSpecT, bool kFromPixels>
class PointMazeEnvBase : public Env<EnvSpecT>, public MujocoRobotEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  bool sparse_reward_;
  bool continuing_task_;
  bool reset_target_;
  mjtNum maze_size_scaling_;
  mjtNum maze_height_;
  mjtNum position_noise_range_;
  mjtNum camera_distance_;
  std::array<mjtNum, 2> goal_{};
  std::array<mjtNum, 2> reset_pos_{};
  std::vector<std::array<mjtNum, 2>> goal_locations_;
  std::vector<std::array<mjtNum, 2>> reset_locations_;
  int target_site_id_;
  std::uniform_real_distribution<> unit_dist_{0.0, 1.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;
  using State = typename Base::State;

  PointMazeEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        MujocoRobotEnv(spec.config["base_path"_],
                       BuildMazeXml(spec.config["maze_map"_],
                                    spec.config["maze_size_scaling"_],
                                    spec.config["maze_height"_]),
                       spec.config["frame_skip"_],
                       spec.config["max_episode_steps"_],
                       spec.config["frame_stack"_],
                       RenderWidthOrDefault<kFromPixels>(spec.config),
                       RenderHeightOrDefault<kFromPixels>(spec.config),
                       RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        sparse_reward_(spec.config["reward_type"_] == "sparse"),
        continuing_task_(spec.config["continuing_task"_]),
        reset_target_(spec.config["reset_target"_]),
        maze_size_scaling_(spec.config["maze_size_scaling"_]),
        maze_height_(spec.config["maze_height"_]),
        position_noise_range_(spec.config["position_noise_range"_]),
        camera_distance_(MazeRows(spec.config["maze_map"_]).size() > 8 ? 12.5
                                                                       : 8.8),
        target_site_id_(SiteId(model_, "target")) {
    BuildMazeLocations(spec.config["maze_map"_]);
    InitializeRobotEnv();
    std::remove(model_path_.c_str());
  }

  ~PointMazeEnvBase() override = default;

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    goal_ = AddNoise(SampleGoal());
    UpdateTargetSitePos();
    reset_pos_ = AddNoise(SampleResetPos());
    ResetToInitialState();
    data_->qpos[0] = reset_pos_[0];
    data_->qpos[1] = reset_pos_[1];
    mj_forward(model_, data_);
    CaptureResetState();
    auto achieved_goal = AchievedGoal();
    mjtNum distance = GoalDistance(achieved_goal, goal_);
    WriteState(0.0, distance, distance <= 0.45, true);
  }

  void Step(const Action& action) override {
    data_->qvel[0] = std::clamp(data_->qvel[0], static_cast<mjtNum>(-5.0),
                                static_cast<mjtNum>(5.0));
    data_->qvel[1] = std::clamp(data_->qvel[1], static_cast<mjtNum>(-5.0),
                                static_cast<mjtNum>(5.0));
    mj_forward(model_, data_);
    const float* act = static_cast<const float*>(action["action"_].Data());
    data_->ctrl[0] =
        std::clamp(static_cast<mjtNum>(act[0]), static_cast<mjtNum>(-1.0),
                   static_cast<mjtNum>(1.0));
    data_->ctrl[1] =
        std::clamp(static_cast<mjtNum>(act[1]), static_cast<mjtNum>(-1.0),
                   static_cast<mjtNum>(1.0));
    DoSimulation();
    ++elapsed_step_;

    auto achieved_goal = AchievedGoal();
    mjtNum distance = GoalDistance(achieved_goal, goal_);
    bool success = distance <= 0.45;
    mjtNum reward = std::exp(-distance);
    if (sparse_reward_) {
      reward = success ? 1.0 : 0.0;
    }
    bool terminated = !continuing_task_ && success;
    done_ = terminated || elapsed_step_ >= max_episode_steps_;
    WriteState(reward, distance, success, false);
    UpdateGoal(achieved_goal);
  }

 protected:
  bool RenderCamera(mjvCamera* camera) override {
    camera->distance = camera_distance_;
    return true;
  }

  static const std::vector<std::string>& MazeRows(const std::string& maze_map) {
    static const std::vector<std::string> k_open = {
        "1111111", "1000001", "1000001", "1000001", "1111111",
    };
    static const std::vector<std::string> k_open_diverse_g = {
        "1111111", "1rgggg1", "1ggggg1", "1ggggg1", "1111111",
    };
    static const std::vector<std::string> k_open_diverse_gr = {
        "1111111", "1ccccc1", "1ccccc1", "1ccccc1", "1111111",
    };
    static const std::vector<std::string> k_u_maze = {
        "11111", "10001", "11101", "10001", "11111",
    };
    static const std::vector<std::string> k_medium = {
        "11111111", "10011001", "10010001", "11000111",
        "10010001", "10100101", "10001001", "11111111",
    };
    static const std::vector<std::string> k_medium_diverse_g = {
        "11111111", "1r011001", "100100g1", "11000111",
        "10010001", "1g100101", "10001g01", "11111111",
    };
    static const std::vector<std::string> k_medium_diverse_gr = {
        "11111111", "1c011001", "100100c1", "11000111",
        "10010001", "1c100101", "10001c01", "11111111",
    };
    static const std::vector<std::string> k_large = {
        "111111111111", "100001000001", "101101010101",
        "100000010001", "101111011101", "100101000001",
        "110101010111", "100100010001", "111111111111",
    };
    static const std::vector<std::string> k_large_diverse_g = {
        "111111111111", "1r0001g00001", "101101010101",
        "10000g0100g1", "101111011101", "10g101000001",
        "110101010111", "1001g0g10g01", "111111111111",
    };
    static const std::vector<std::string> k_large_diverse_gr = {
        "111111111111", "1c0001c00001", "101101010101",
        "10000c0100c1", "101111011101", "10c101000001",
        "110101010111", "1001c0c10c01", "111111111111",
    };

    if (maze_map == "OPEN") {
      return k_open;
    }
    if (maze_map == "OPEN_DIVERSE_G") {
      return k_open_diverse_g;
    }
    if (maze_map == "OPEN_DIVERSE_GR") {
      return k_open_diverse_gr;
    }
    if (maze_map == "U_MAZE") {
      return k_u_maze;
    }
    if (maze_map == "MEDIUM_MAZE") {
      return k_medium;
    }
    if (maze_map == "MEDIUM_MAZE_DIVERSE_G") {
      return k_medium_diverse_g;
    }
    if (maze_map == "MEDIUM_MAZE_DIVERSE_GR") {
      return k_medium_diverse_gr;
    }
    if (maze_map == "LARGE_MAZE") {
      return k_large;
    }
    if (maze_map == "LARGE_MAZE_DIVERSE_G") {
      return k_large_diverse_g;
    }
    if (maze_map == "LARGE_MAZE_DIVERSE_GR") {
      return k_large_diverse_gr;
    }
    throw std::runtime_error("Unknown PointMaze map: " + maze_map);
  }

  static std::string BuildMazeXml(const std::string& maze_map,
                                  mjtNum maze_size_scaling,
                                  mjtNum maze_height) {
    static std::atomic<int> counter{0};
    std::string xml_path = TemporaryDirectory();
    if (!xml_path.empty() && xml_path.back() != '/' &&
        xml_path.back() != '\\') {
#ifdef _WIN32
      xml_path.push_back('\\');
#else
      xml_path.push_back('/');
#endif
    }
    xml_path += "envpool_point_maze_" + std::to_string(ProcessId()) + "_" +
                std::to_string(counter.fetch_add(1)) + ".xml";
    std::ofstream xml(xml_path);
    if (!xml) {
      throw std::runtime_error("Failed to create temporary PointMaze XML.");
    }

    const auto& rows = MazeRows(maze_map);
    mjtNum x_center =
        static_cast<mjtNum>(rows[0].size()) * maze_size_scaling / 2.0;
    mjtNum y_center =
        static_cast<mjtNum>(rows.size()) * maze_size_scaling / 2.0;
    mjtNum z_pos = maze_height * maze_size_scaling / 2.0;
    mjtNum half_size = 0.5 * maze_size_scaling;

    xml << "<mujoco>\n"
        << "  <compiler inertiafromgeom=\"true\" angle=\"radian\" "
           "coordinate=\"local\"/>\n"
        << "  <option timestep=\"0.01\" gravity=\"0 0 0\" iterations=\"20\" "
           "integrator=\"Euler\"/>\n"
        << "  <default>\n"
        << "    <joint damping=\"1\" limited=\"false\"/>\n"
        << "    <geom friction=\".5 .1 .1\" density=\"1000\" margin=\"0.002\" "
           "condim=\"1\" contype=\"2\" conaffinity=\"1\"/>\n"
        << "  </default>\n"
        << "  <asset>\n"
        << "    <texture type=\"2d\" name=\"groundplane\" builtin=\"checker\" "
           "rgb1=\"0.2 0.3 0.4\" rgb2=\"0.1 0.2 0.3\" width=\"100\" "
           "height=\"100\"/>\n"
        << "    <texture name=\"skybox\" type=\"skybox\" "
           "builtin=\"gradient\" rgb1=\".4 .6 .8\" rgb2=\"0 0 0\" "
           "width=\"800\" height=\"800\" mark=\"random\" "
           "markrgb=\"1 1 1\"/>\n"
        << "    <material name=\"groundplane\" texture=\"groundplane\" "
           "texrepeat=\"20 20\"/>\n"
        << "    <material name=\"target\" rgba=\".6 .3 .3 1\"/>\n"
        << "  </asset>\n"
        << "  <visual>\n"
        << "    <headlight ambient=\".4 .4 .4\" diffuse=\".8 .8 .8\" "
           "specular=\"0.1 0.1 0.1\"/>\n"
        << "    <map znear=\".01\"/>\n"
        << "    <quality shadowsize=\"2048\"/>\n"
        << "  </visual>\n"
        << "  <worldbody>\n"
        << "    <geom name=\"ground\" size=\"40 40 0.25\" pos=\"0 0 -0.1\" "
           "type=\"plane\" contype=\"1\" conaffinity=\"0\" "
           "material=\"groundplane\"/>\n"
        << "    <body name=\"particle\" pos=\"0 0 0\">\n"
        << "      <geom name=\"particle_geom\" type=\"sphere\" size=\"0.1\" "
           "rgba=\"0.0 0.0 1.0 0.0\" contype=\"1\"/>\n"
        << "      <site name=\"particle_site\" pos=\"0.0 0.0 0.0\" "
           "size=\"0.2\" rgba=\"0.3 0.6 0.3 1\"/>\n"
        << "      <joint name=\"ball_x\" type=\"slide\" pos=\"0 0 0\" "
           "axis=\"1 0 0\"/>\n"
        << "      <joint name=\"ball_y\" type=\"slide\" pos=\"0 0 0\" "
           "axis=\"0 1 0\"/>\n"
        << "    </body>\n";
    for (int row = 0; row < static_cast<int>(rows.size()); ++row) {
      for (int col = 0; col < static_cast<int>(rows[row].size()); ++col) {
        if (rows[row][col] != '1') {
          continue;
        }
        mjtNum x =
            (static_cast<mjtNum>(col) + 0.5) * maze_size_scaling - x_center;
        mjtNum y =
            y_center - (static_cast<mjtNum>(row) + 0.5) * maze_size_scaling;
        xml << "    <geom name=\"block_" << row << "_" << col << "\" pos=\""
            << x << " " << y << " " << z_pos << "\" " << "size=\"" << half_size
            << " " << half_size << " " << z_pos
            << R"(" type="box" material="" contype="1" conaffinity="1" )"
            << "rgba=\"0.7 0.5 0.3 1.0\"/>\n";
      }
    }
    xml << R"(    <site name="target" pos="0 0 )" << z_pos << "\" size=\""
        << 0.2 * maze_size_scaling
        << "\" rgba=\"1 0 0 0.7\" type=\"sphere\"/>\n"
        << "  </worldbody>\n"
        << "  <actuator>\n"
        << "    <motor name=\"motor_x\" joint=\"ball_x\" gear=\"100\" "
           "ctrllimited=\"true\" ctrlrange=\"-1 1\"/>\n"
        << "    <motor name=\"motor_y\" joint=\"ball_y\" gear=\"100\" "
           "ctrllimited=\"true\" ctrlrange=\"-1 1\"/>\n"
        << "  </actuator>\n"
        << "</mujoco>\n";
    xml.close();
    return xml_path;
  }

  static int ProcessId() {
#ifdef _WIN32
    return _getpid();
#else
    return getpid();
#endif
  }

  static std::string TemporaryDirectory() {
    for (const char* env_key : {"TMPDIR", "TEMP", "TMP"}) {
      const char* env_value = std::getenv(env_key);
      if (env_value != nullptr && env_value[0] != '\0') {
        return env_value;
      }
    }
#ifdef _WIN32
    return "C:\\Windows\\Temp";
#else
    return "/tmp";
#endif
  }

  void BuildMazeLocations(const std::string& maze_map) {
    const auto& rows = MazeRows(maze_map);
    std::vector<std::array<mjtNum, 2>> empty_locations;
    std::vector<std::array<mjtNum, 2>> combined_locations;
    mjtNum x_center =
        static_cast<mjtNum>(rows[0].size()) * maze_size_scaling_ / 2.0;
    mjtNum y_center =
        static_cast<mjtNum>(rows.size()) * maze_size_scaling_ / 2.0;
    for (int row = 0; row < static_cast<int>(rows.size()); ++row) {
      for (int col = 0; col < static_cast<int>(rows[row].size()); ++col) {
        std::array<mjtNum, 2> xy{
            (static_cast<mjtNum>(col) + 0.5) * maze_size_scaling_ - x_center,
            y_center - (static_cast<mjtNum>(row) + 0.5) * maze_size_scaling_,
        };
        switch (rows[row][col]) {
          case '0':
            empty_locations.push_back(xy);
            break;
          case 'g':
            goal_locations_.push_back(xy);
            break;
          case 'r':
            reset_locations_.push_back(xy);
            break;
          case 'c':
            combined_locations.push_back(xy);
            break;
          default:
            break;
        }
      }
    }
    if (goal_locations_.empty() && reset_locations_.empty() &&
        combined_locations.empty()) {
      combined_locations = empty_locations;
    } else if (reset_locations_.empty() && combined_locations.empty()) {
      reset_locations_ = empty_locations;
    } else if (goal_locations_.empty() && combined_locations.empty()) {
      goal_locations_ = empty_locations;
    }
    goal_locations_.insert(goal_locations_.end(), combined_locations.begin(),
                           combined_locations.end());
    reset_locations_.insert(reset_locations_.end(), combined_locations.begin(),
                            combined_locations.end());
  }

  std::array<mjtNum, 2> AddNoise(std::array<mjtNum, 2> xy) {
    mjtNum scale = position_noise_range_ * maze_size_scaling_;
    xy[0] += (-scale) + 2.0 * scale * unit_dist_(gen_);
    xy[1] += (-scale) + 2.0 * scale * unit_dist_(gen_);
    return xy;
  }

  std::array<mjtNum, 2> SampleGoal() {
    if (goal_locations_.empty()) {
      throw std::runtime_error("PointMaze goal locations are empty.");
    }
    int goal_id = static_cast<int>(unit_dist_(gen_) *
                                   static_cast<double>(goal_locations_.size()));
    goal_id = std::min(goal_id, static_cast<int>(goal_locations_.size()) - 1);
    return goal_locations_[goal_id];
  }

  std::array<mjtNum, 2> SampleResetPos() {
    if (reset_locations_.empty()) {
      throw std::runtime_error("PointMaze reset locations are empty.");
    }
    std::array<mjtNum, 2> reset_pos = goal_;
    while (GoalDistance(reset_pos, goal_) <= 0.5 * maze_size_scaling_) {
      int reset_id = static_cast<int>(
          unit_dist_(gen_) * static_cast<double>(reset_locations_.size()));
      reset_id =
          std::min(reset_id, static_cast<int>(reset_locations_.size()) - 1);
      reset_pos = reset_locations_[reset_id];
    }
    return reset_pos;
  }

  void UpdateTargetSitePos() {
    model_->site_pos[3 * target_site_id_] = goal_[0];
    model_->site_pos[3 * target_site_id_ + 1] = goal_[1];
    model_->site_pos[3 * target_site_id_ + 2] =
        maze_height_ * maze_size_scaling_ / 2.0;
    mj_forward(model_, data_);
  }

  std::array<mjtNum, 2> AchievedGoal() const {
    return {data_->qpos[0], data_->qpos[1]};
  }

  static mjtNum GoalDistance(const std::array<mjtNum, 2>& achieved_goal,
                             const std::array<mjtNum, 2>& desired_goal) {
    mjtNum dx = achieved_goal[0] - desired_goal[0];
    mjtNum dy = achieved_goal[1] - desired_goal[1];
    return std::sqrt(dx * dx + dy * dy);
  }

  void UpdateGoal(const std::array<mjtNum, 2>& achieved_goal) {
    if (!continuing_task_ || !reset_target_ || goal_locations_.size() <= 1 ||
        GoalDistance(achieved_goal, goal_) > 0.45) {
      return;
    }
    while (GoalDistance(achieved_goal, goal_) <= 0.45) {
      goal_ = AddNoise(SampleGoal());
    }
    UpdateTargetSitePos();
  }

  void WriteState(mjtNum reward, mjtNum distance, bool success, bool reset) {
    State state = Allocate();
    auto achieved_goal = AchievedGoal();
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      auto obs_observation = state["obs:observation"_];
      auto* obs = PrepareObservation("obs:observation", &obs_observation);
      obs[0] = data_->qpos[0];
      obs[1] = data_->qpos[1];
      obs[2] = data_->qvel[0];
      obs[3] = data_->qvel[1];
      CommitObservation("obs:observation", &obs_observation, reset);
      auto obs_achieved_goal = state["obs:achieved_goal"_];
      AssignObservation("obs:achieved_goal", &obs_achieved_goal,
                        achieved_goal.data(), achieved_goal.size(), reset);
      auto obs_desired_goal = state["obs:desired_goal"_];
      AssignObservation("obs:desired_goal", &obs_desired_goal, goal_.data(),
                        goal_.size(), reset);
      state["info:success"_] = success ? 1.0 : 0.0;
      state["info:distance"_] = distance;
#ifdef ENVPOOL_TEST
      state["info:qpos0"_].Assign(qpos0_.data(), qpos0_.size());
      state["info:qvel0"_].Assign(qvel0_.data(), qvel0_.size());
      state["info:goal0"_].Assign(goal_.data(), goal_.size());
#endif
    }
    state["reward"_] = static_cast<float>(reward);
  }
};

using PointMazeEnv = PointMazeEnvBase<PointMazeEnvSpec, false>;
using PointMazePixelEnv = PointMazeEnvBase<PointMazePixelEnvSpec, true>;
using PointMazeEnvPool = AsyncEnvPool<PointMazeEnv>;
using PointMazePixelEnvPool = AsyncEnvPool<PointMazePixelEnv>;

}  // namespace gymnasium_robotics

#endif  // ENVPOOL_MUJOCO_ROBOTICS_POINT_MAZE_H_
