/*
 * Copyright 2026 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENVPOOL_GFOOTBALL_GFOOTBALL_ENV_H_
#define ENVPOOL_GFOOTBALL_GFOOTBALL_ENV_H_

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/gfootball/gfootball_common.h"
#include "envpool/utils/image_process.h"

namespace gfootball {

using FrameSpec = Spec<uint8_t>;

class GfootballEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("env_name"_.Bind(std::string("11_vs_11_stochastic")),
                    "render"_.Bind(false), "physics_steps_per_frame"_.Bind(10),
                    "render_resolution_x"_.Bind(1280),
                    "render_resolution_y"_.Bind(ResolveRenderHeight(1280)));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config&) {
    return MakeDict("obs"_.Bind(Spec<uint8_t>(
                        {kSMMHeight, kSMMWidth, kSMMChannels}, {0, 255})),
                    "info:score"_.Bind(Spec<int>({2})),
                    "info:game_mode"_.Bind(Spec<int>({})),
                    "info:ball_owned_team"_.Bind(Spec<int>({})),
                    "info:ball_owned_player"_.Bind(Spec<int>({})),
                    "info:steps_left"_.Bind(Spec<int>({})),
                    "info:elapsed_step"_.Bind(Spec<int>({})),
                    "info:engine_seed"_.Bind(Spec<int>({})),
                    "info:episode_number"_.Bind(Spec<int>({})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config&) {
    return MakeDict(
        "action"_.Bind(MarkDiscrete(Spec<int>({-1}, {0, kActionCount - 1}))));
  }
};

using GfootballEnvSpec = EnvSpec<GfootballEnvFns>;

class GfootballEnv : public Env<GfootballEnvSpec>, public RenderableEnv {
 public:
  GfootballEnv(const Spec& spec, int env_id)
      : Env<GfootballEnvSpec>(spec, env_id),
        env_name_(spec.config["env_name"_]),
        render_enabled_(spec.config["render"_]),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        render_resolution_x_(spec.config["render_resolution_x"_]),
        render_resolution_y_(spec.config["render_resolution_y"_]) {
    EnsureGfootballRuntimePaths(spec.config["base_path"_]);
    engine_ = std::make_unique<GameEnv>();
    engine_->game_config.render = render_enabled_;
    engine_->game_config.physics_steps_per_frame =
        spec.config["physics_steps_per_frame"_];
    engine_->game_config.render_resolution_x = render_resolution_x_;
    engine_->game_config.render_resolution_y = render_resolution_y_;
    render_frame_.resize(
        FrameSizeBytes(render_resolution_x_, render_resolution_y_));
    engine_->start_game();
    episode_number_ = 1;
    ResetEpisode(/*emit_state=*/false);
    done_ = true;
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    ++episode_number_;
    ResetEpisode(/*emit_state=*/true);
  }

  void Step(const Action& action) override {
    const int requested_action =
        std::clamp(static_cast<int>(action["action"_]), 0, kActionCount - 1);
    int backend_action = ResolveBackendAction(requested_action);
    if (engine_->waiting_for_game_count == 20) {
      backend_action = game_short_pass;
    } else if (engine_->waiting_for_game_count > 20) {
      backend_action = game_idle;
      const int controlled_player = ControlledPlayer();
      if (last_info_.ball_owned_team == 0 &&
          controlled_player == last_info_.ball_owned_player) {
        backend_action =
            engine_->waiting_for_game_count < 30 ? game_right : game_left;
      }
    }
    engine_->action(backend_action, /*left_team=*/true, /*player=*/0);
    while (true) {
      SetGame(engine_.get());
      engine_->step();
      if (RetrieveObservation()) {
        break;
      }
    }

    done_ = false;
    if (last_scenario_end_episode_on_score_ &&
        (last_info_.left_goals > 0 || last_info_.right_goals > 0)) {
      done_ = true;
    }
    if (last_scenario_end_episode_on_out_of_play_ &&
        static_cast<int>(last_info_.game_mode) != kGameModeNormal &&
        previous_game_mode_ == kGameModeNormal) {
      done_ = true;
    }
    previous_game_mode_ = static_cast<int>(last_info_.game_mode);

    if (last_scenario_end_episode_on_possession_change_ &&
        last_info_.ball_owned_team != -1 && prev_ball_owned_team_ != -1 &&
        last_info_.ball_owned_team != prev_ball_owned_team_) {
      done_ = true;
    }
    if (last_info_.ball_owned_team != -1) {
      prev_ball_owned_team_ = last_info_.ball_owned_team;
    }

    const int score_diff = last_info_.left_goals - last_info_.right_goals;
    const auto reward = static_cast<float>(score_diff - previous_score_diff_);
    previous_score_diff_ = score_diff;
    if (static_cast<int>(last_info_.game_mode) != kGameModeNormal) {
      engine_->waiting_for_game_count += 1;
    } else {
      engine_->waiting_for_game_count = 0;
    }
    if (last_info_.step >= max_episode_steps_) {
      done_ = true;
    }
    WriteState(reward);
  }

  [[nodiscard]] std::pair<int, int> RenderSize(int width,
                                               int height) const override {
    if (width <= 0 && height <= 0) {
      return {render_resolution_x_, render_resolution_y_};
    }
    if (width <= 0) {
      width = static_cast<int>(
          std::lround(height * render_resolution_x_ /
                      static_cast<double>(render_resolution_y_)));
    }
    if (height <= 0) {
      height = static_cast<int>(
          std::lround(width * render_resolution_y_ /
                      static_cast<double>(render_resolution_x_)));
    }
    return {width, height};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    if (!render_enabled_) {
      throw std::runtime_error(
          "gfootball render requested without render support enabled");
    }
    Array output(FrameSpec({height, width, kRenderChannels}),
                 reinterpret_cast<char*>(rgb));
    Array input(FrameSpec({render_resolution_y_, render_resolution_x_,
                           kRenderChannels}),
                reinterpret_cast<char*>(render_frame_.data()));
    if (width == render_resolution_x_ && height == render_resolution_y_) {
      output.Assign(input);
      return;
    }
    Resize(input, &output);
  }

 protected:
  [[nodiscard]] int CurrentMaxEpisodeSteps() const override {
    return max_episode_steps_;
  }

 private:
  std::unique_ptr<GameEnv> engine_;
  std::string env_name_;
  bool render_enabled_{false};
  int max_episode_steps_{0};
  int render_resolution_x_{1280};
  int render_resolution_y_{720};
  int episode_number_{0};
  unsigned int engine_seed_{0};
  bool done_{true};
  int previous_score_diff_{0};
  int previous_game_mode_{-1};
  int prev_ball_owned_team_{-1};
  bool last_scenario_end_episode_on_score_{false};
  bool last_scenario_end_episode_on_possession_change_{false};
  bool last_scenario_end_episode_on_out_of_play_{false};
  SharedInfo last_info_;
  std::vector<unsigned char> render_frame_;

  void ResetEpisode(bool emit_state) {
    previous_score_diff_ = 0;
    previous_game_mode_ = -1;
    prev_ball_owned_team_ = -1;
    done_ = false;
    engine_seed_ = SampleEngineSeed(&gen_);
    auto scenario = ScenarioConfig::make();
    BuildEnvScenarioConfig(env_name_, episode_number_, engine_seed_,
                           max_episode_steps_, scenario.get());
    last_scenario_end_episode_on_score_ = scenario->end_episode_on_score;
    last_scenario_end_episode_on_possession_change_ =
        scenario->end_episode_on_possession_change;
    last_scenario_end_episode_on_out_of_play_ =
        scenario->end_episode_on_out_of_play;
    engine_->reset(*scenario, render_enabled_);
    while (true) {
      if (RetrieveObservation()) {
        break;
      }
      SetGame(engine_.get());
      engine_->step();
    }
    if (emit_state) {
      WriteState(0.0f);
    }
  }

  bool RetrieveObservation() {
    SetGame(engine_.get());
    last_info_ = engine_->get_info();
    if (render_enabled_) {
      TransformFrameToRgb(engine_->get_frame(), render_resolution_x_,
                          render_resolution_y_, render_frame_.data());
    }
    return last_info_.is_in_play;
  }

  int ControlledPlayer() const {
    if (last_info_.left_controllers.empty()) {
      return -1;
    }
    return last_info_.left_controllers[0].controlled_player;
  }

  static void MarkPoint(float x, float y, int layer,
                        const TArray<uint8_t>& obs) {
    obs(MinimapCoordY(y), MinimapCoordX(x), layer) = kMarkerValue;
  }

  void FillObservation(const TArray<uint8_t>& obs) const {
    std::memset(obs.Data(), 0, kSMMHeight * kSMMWidth * kSMMChannels);
    for (const auto& player : last_info_.left_team) {
      MarkPoint(player.player_position.env_coord(0),
                player.player_position.env_coord(1), 0, obs);
    }
    for (const auto& player : last_info_.right_team) {
      MarkPoint(player.player_position.env_coord(0),
                player.player_position.env_coord(1), 1, obs);
    }
    MarkPoint(last_info_.ball_position.env_coord(0),
              last_info_.ball_position.env_coord(1), 2, obs);
    const int controlled_player = ControlledPlayer();
    if (controlled_player >= 0 &&
        controlled_player < static_cast<int>(last_info_.left_team.size())) {
      const auto& player = last_info_.left_team[controlled_player];
      MarkPoint(player.player_position.env_coord(0),
                player.player_position.env_coord(1), 3, obs);
    }
  }

  void WriteState(float reward) {
    auto state = Allocate();
    FillObservation(state["obs"_]);
    state["reward"_] = reward;
    state["info:score"_](0) = last_info_.left_goals;
    state["info:score"_](1) = last_info_.right_goals;
    state["info:game_mode"_] = static_cast<int>(last_info_.game_mode);
    state["info:ball_owned_team"_] = last_info_.ball_owned_team;
    state["info:ball_owned_player"_] = last_info_.ball_owned_player;
    state["info:steps_left"_] =
        std::max(0, max_episode_steps_ - last_info_.step);
    state["info:elapsed_step"_] = last_info_.step;
    state["info:engine_seed"_] = static_cast<int>(engine_seed_);
    state["info:episode_number"_] = episode_number_;
  }
};

using GfootballEnvPool = AsyncEnvPool<GfootballEnv>;

}  // namespace gfootball

#endif  // ENVPOOL_GFOOTBALL_GFOOTBALL_ENV_H_
