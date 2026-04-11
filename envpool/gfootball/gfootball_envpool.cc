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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include "envpool/core/py_envpool.h"
#include "envpool/gfootball/gfootball_env.h"

namespace py = pybind11;

using GfootballEnvSpec = PyEnvSpec<gfootball::GfootballEnvSpec>;
using GfootballEnvPool = PyEnvPool<gfootball::GfootballEnvPool>;

namespace gfootball {
namespace {

py::tuple ToPosition2(const Position& position) {
  return py::make_tuple(position.env_coord(0), position.env_coord(1));
}

py::tuple ToPosition3(const Position& position) {
  return py::make_tuple(position.env_coord(0), position.env_coord(1),
                        position.env_coord(2));
}

py::list ToPlayers(const std::vector<PlayerInfo>& players) {
  py::list out;
  for (const auto& player : players) {
    py::dict item;
    item["position"] = ToPosition2(player.player_position);
    item["direction"] = ToPosition2(player.player_direction);
    item["has_card"] = player.has_card;
    item["is_active"] = player.is_active;
    item["tired_factor"] = player.tired_factor;
    item["role"] = static_cast<int>(player.role);
    item["designated_player"] = player.designated_player;
    out.append(item);
  }
  return out;
}

py::list ToControllers(const std::vector<ControllerInfo>& controllers) {
  py::list out;
  for (const auto& controller : controllers) {
    out.append(controller.controlled_player);
  }
  return out;
}

class GfootballOracleEngine {
 public:
  GfootballOracleEngine(const std::string& base_path, bool render,
                        int render_resolution_x, int render_resolution_y,
                        int physics_steps_per_frame) {
    EnsureGfootballRuntimePaths(base_path);
    engine_ = std::make_unique<GameEnv>();
    engine_->game_config.render = render;
    engine_->game_config.render_resolution_x = render_resolution_x;
    engine_->game_config.render_resolution_y = render_resolution_y;
    engine_->game_config.physics_steps_per_frame = physics_steps_per_frame;
    engine_->start_game();
  }

  void Reset(const std::string& env_name, int episode_number,
             unsigned int engine_seed, int max_episode_steps) {
    auto scenario = ScenarioConfig::make();
    BuildEnvScenarioConfig(env_name, episode_number, engine_seed,
                           max_episode_steps, scenario.get());
    engine_->reset(*scenario, engine_->game_config.render);
  }

  py::dict GetInfo() {
    SetGame(engine_.get());
    SharedInfo info = engine_->get_info();
    py::dict out;
    out["ball_position"] = ToPosition3(info.ball_position);
    out["ball_direction"] = ToPosition3(info.ball_direction);
    out["ball_rotation"] = ToPosition3(info.ball_rotation);
    out["left_team"] = ToPlayers(info.left_team);
    out["right_team"] = ToPlayers(info.right_team);
    out["left_controllers"] = ToControllers(info.left_controllers);
    out["right_controllers"] = ToControllers(info.right_controllers);
    out["left_goals"] = info.left_goals;
    out["right_goals"] = info.right_goals;
    out["game_mode"] = static_cast<int>(info.game_mode);
    out["is_in_play"] = info.is_in_play;
    out["ball_owned_team"] = info.ball_owned_team;
    out["ball_owned_player"] = info.ball_owned_player;
    out["step"] = info.step;
    return out;
  }

  py::bytes GetFrame() { return py::bytes(engine_->get_frame()); }

  void Action(int action, bool left_team, int player) {
    engine_->action(action, left_team, player);
  }

  void Step() {
    SetGame(engine_.get());
    engine_->step();
  }

  int WaitingForGameCount() const { return engine_->waiting_for_game_count; }

  void SetWaitingForGameCount(int value) {
    engine_->waiting_for_game_count = value;
  }

 private:
  std::unique_ptr<GameEnv> engine_;
};

void BindOracleEngine(py::module_& m) {
  py::class_<GfootballOracleEngine>(m, "_GfootballOracleEngine")
      .def(py::init<const std::string&, bool, int, int, int>())
      .def("reset", &GfootballOracleEngine::Reset)
      .def("get_info", &GfootballOracleEngine::GetInfo)
      .def("get_frame", &GfootballOracleEngine::GetFrame)
      .def("action", &GfootballOracleEngine::Action)
      .def("step", &GfootballOracleEngine::Step)
      .def_property("waiting_for_game_count",
                    &GfootballOracleEngine::WaitingForGameCount,
                    &GfootballOracleEngine::SetWaitingForGameCount);
}

}  // namespace
}  // namespace gfootball

PYBIND11_MODULE(gfootball_envpool, m) {
  REGISTER(m, GfootballEnvSpec, GfootballEnvPool)
  gfootball::BindOracleEngine(m);
}
