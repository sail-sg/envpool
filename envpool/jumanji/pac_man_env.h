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

#ifndef ENVPOOL_JUMANJI_PAC_MAN_ENV_H_
#define ENVPOOL_JUMANJI_PAC_MAN_ENV_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/parse_utils.h"
#include "envpool/jumanji/render_utils.h"

namespace jumanji {
namespace pacman {

constexpr int kRows = 31;
constexpr int kCols = 28;
constexpr int kNumGhosts = 4;
constexpr int kNumPowerUps = 4;
constexpr int kNumPellets = 318;
constexpr int kActivePellets = 3;
constexpr int kTimeLimit = 1000;
constexpr int kReplaySteps = 32;
constexpr std::array<std::array<int, 2>, 5> kMoves = {
    {{{0, 0}}, {{-1, 0}}, {{0, 1}}, {{1, 0}}, {{0, -1}}}};  // NOLINT

inline int Offset(int row, int col) { return row * kCols + col; }
inline bool InGrid(int row, int col) {
  return 0 <= row && row < kRows && 0 <= col && col < kCols;
}

using Grid = std::array<int, kRows * kCols>;
using PelletLocations = std::array<int, kNumPellets * 2>;
using GhostLocations = std::array<int, kNumGhosts * 2>;
using PowerUpLocations = std::array<int, kNumPowerUps * 2>;

inline Grid ParseGrid(const std::string& text) {
  Grid grid{};
  if (text.empty()) {
    return grid;
  }
  std::stringstream stream(text);
  std::string token;
  int index = 0;
  while (std::getline(stream, token, ',') && index < kRows * kCols) {
    grid[index++] = std::stoi(token) != 0 ? 1 : 0;
  }
  return grid;
}

inline std::pair<int, int> ParsePosition(const std::string& text,
                                         int default_row, int default_col) {
  if (text.empty()) {
    return {default_row, default_col};
  }
  const std::size_t sep = text.find(',');
  if (sep == std::string::npos) {
    return {default_row, default_col};
  }
  return {std::clamp(std::stoi(text.substr(0, sep)), 0, kRows - 1),
          std::clamp(std::stoi(text.substr(sep + 1)), 0, kCols - 1)};
}

inline PelletLocations ParsePellets(const std::string& text) {
  PelletLocations pellets{};
  pellets.fill(-1);
  if (text.empty()) {
    return pellets;
  }
  std::stringstream stream(text);
  std::string token;
  int index = 0;
  while (std::getline(stream, token, ',') && index < kNumPellets * 2) {
    pellets[index++] = std::stoi(token);
  }
  return pellets;
}

}  // namespace pacman

class PacManEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "pacman_grid"_.Bind(std::string("")),
        "pacman_player_location"_.Bind(std::string("")),
        "pacman_pellet_locations"_.Bind(std::string("")),
        "pacman_ghost_locations"_.Bind(std::string("")),
        "pacman_power_up_locations"_.Bind(std::string("")),
        "pacman_action_mask"_.Bind(std::string("")),
        "pacman_frightened_state_time"_.Bind(0),
        "pacman_initial_score"_.Bind(0),
        "pacman_replay_pellet_locations"_.Bind(std::string("")),
        "pacman_replay_player_locations"_.Bind(std::string("")),
        "pacman_replay_ghost_locations"_.Bind(std::string("")),
        "pacman_replay_power_up_locations"_.Bind(std::string("")),
        "pacman_replay_frightened_state_time"_.Bind(std::string("")),
        "pacman_replay_action_mask"_.Bind(std::string("")),
        "pacman_replay_score"_.Bind(std::string("")),
        "pacman_replay_rewards"_.Bind(std::string("")),
        "pacman_replay_done"_.Bind(std::string("")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    (void)conf;
    return MakeDict("obs:grid"_.Bind(Spec<int>({31, 28}, {0, 1})),
                    "obs:player_locations.y"_.Bind(Spec<int>({}, {0, 30})),
                    "obs:player_locations.x"_.Bind(Spec<int>({}, {0, 27})),
                    "obs:ghost_locations"_.Bind(Spec<int>({4, 2})),
                    "obs:power_up_locations"_.Bind(Spec<int>({4, 2})),
                    "obs:frightened_state_time"_.Bind(Spec<int>({})),
                    "obs:pellet_locations"_.Bind(Spec<int>({318, 2})),
                    "obs:action_mask"_.Bind(Spec<bool>({5}, {false, true})),
                    "obs:score"_.Bind(Spec<int>({})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    (void)conf;
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 4})));
  }
};

using PacManEnvSpec = EnvSpec<PacManEnvFns>;

class PacManEnv : public Env<PacManEnvSpec>, public RenderableEnv {
 protected:
  pacman::Grid grid_{};
  pacman::Grid configured_grid_{};
  pacman::PelletLocations configured_pellet_locations_{};
  pacman::GhostLocations ghost_locations_{};
  pacman::PowerUpLocations power_up_locations_{};
  pacman::GhostLocations configured_ghost_locations_{};
  pacman::PowerUpLocations configured_power_up_locations_{};
  std::array<bool, 5> configured_action_mask_{};
  std::array<int, pacman::kReplaySteps * pacman::kNumPellets * 2>
      replay_pellet_locations_{};
  std::array<int, pacman::kReplaySteps * 2> replay_player_locations_{};
  std::array<int, pacman::kReplaySteps * pacman::kNumGhosts * 2>
      replay_ghost_locations_{};
  std::array<int, pacman::kReplaySteps * pacman::kNumPowerUps * 2>
      replay_power_up_locations_{};
  std::array<int, pacman::kReplaySteps> replay_frightened_state_time_{};
  std::array<bool, pacman::kReplaySteps * 5> replay_action_mask_{};
  std::array<int, pacman::kReplaySteps> replay_score_{};
  std::array<float, pacman::kReplaySteps> replay_rewards_{};
  std::array<bool, pacman::kReplaySteps> replay_done_{};
  std::array<int, pacman::kNumPellets> pellet_row_{};
  std::array<int, pacman::kNumPellets> pellet_col_{};
  bool use_configured_grid_;
  bool use_configured_pellets_;
  bool use_configured_action_mask_;
  bool use_replay_;
  int player_row_{1};
  int player_col_{1};
  int configured_player_row_{1};
  int configured_player_col_{1};
  int configured_frightened_state_time_{0};
  int configured_score_{0};
  int score_{0};
  int frightened_state_time_{0};
  int step_count_{0};
  bool done_{true};

 public:
  using Spec = PacManEnvSpec;
  using Action = typename Env<PacManEnvSpec>::Action;

  PacManEnv(const Spec& spec, int env_id)
      : Env<PacManEnvSpec>(spec, env_id),
        configured_grid_(pacman::ParseGrid(spec.config["pacman_grid"_])),
        configured_pellet_locations_(
            pacman::ParsePellets(spec.config["pacman_pellet_locations"_])),
        configured_ghost_locations_(
            parse::CsvArray<int, pacman::kNumGhosts * 2>(
                spec.config["pacman_ghost_locations"_])),
        configured_power_up_locations_(
            parse::CsvArray<int, pacman::kNumPowerUps * 2>(
                spec.config["pacman_power_up_locations"_])),
        configured_action_mask_(
            parse::CsvArray<bool, 5>(spec.config["pacman_action_mask"_])),
        replay_pellet_locations_(
            parse::CsvArray<int,
                            pacman::kReplaySteps * pacman::kNumPellets * 2>(
                spec.config["pacman_replay_pellet_locations"_], -1)),
        replay_player_locations_(parse::CsvArray<int, pacman::kReplaySteps * 2>(
            spec.config["pacman_replay_player_locations"_])),
        replay_ghost_locations_(
            parse::CsvArray<int, pacman::kReplaySteps * pacman::kNumGhosts * 2>(
                spec.config["pacman_replay_ghost_locations"_])),
        replay_power_up_locations_(
            parse::CsvArray<int,
                            pacman::kReplaySteps * pacman::kNumPowerUps * 2>(
                spec.config["pacman_replay_power_up_locations"_])),
        replay_frightened_state_time_(
            parse::CsvArray<int, pacman::kReplaySteps>(
                spec.config["pacman_replay_frightened_state_time"_])),
        replay_action_mask_(parse::CsvArray<bool, pacman::kReplaySteps * 5>(
            spec.config["pacman_replay_action_mask"_])),
        replay_score_(parse::CsvArray<int, pacman::kReplaySteps>(
            spec.config["pacman_replay_score"_])),
        replay_rewards_(parse::CsvArray<float, pacman::kReplaySteps>(
            spec.config["pacman_replay_rewards"_])),
        replay_done_(parse::CsvArray<bool, pacman::kReplaySteps>(
            spec.config["pacman_replay_done"_])),
        use_configured_grid_(!spec.config["pacman_grid"_].empty()),
        use_configured_pellets_(
            !spec.config["pacman_pellet_locations"_].empty()),
        use_configured_action_mask_(
            !spec.config["pacman_action_mask"_].empty()),
        use_replay_(!spec.config["pacman_replay_pellet_locations"_].empty()),
        configured_frightened_state_time_(
            spec.config["pacman_frightened_state_time"_]),
        configured_score_(spec.config["pacman_initial_score"_]) {
    std::tie(configured_player_row_, configured_player_col_) =
        pacman::ParsePosition(spec.config["pacman_player_location"_], 1, 1);
  }

  bool IsDone() override { return done_; }

  int CurrentMaxEpisodeSteps() const override { return pacman::kTimeLimit + 1; }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 256, height > 0 ? height : 256};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    render::Clear(width, height, {8, 8, 24}, rgb);
    for (int row = 0; row < pacman::kRows; ++row) {
      for (int col = 0; col < pacman::kCols; ++col) {
        if (grid_[pacman::Offset(row, col)] == 1) {
          render::FillCell(width, height, pacman::kRows, pacman::kCols, row,
                           col, {28, 40, 190}, rgb);
        } else if (HasPellet(row, col)) {
          auto [x, y] = render::CellCenter(width, height, pacman::kRows,
                                           pacman::kCols, row, col);
          render::FillCircle(width, height, x, y, 2, {245, 210, 170}, rgb);
        }
      }
    }
    auto [px, py] = render::CellCenter(width, height, pacman::kRows,
                                       pacman::kCols, player_row_, player_col_);
    render::FillCircle(width, height, px, py,
                       std::max(3, std::min(width, height) / 45),
                       {255, 230, 45}, rgb);
  }

  void Reset() override {
    if (use_configured_grid_) {
      grid_ = configured_grid_;
    } else {
      grid_.fill(0);
      for (int row = 0; row < pacman::kRows; ++row) {
        for (int col = 0; col < pacman::kCols; ++col) {
          if (row == 0 || row == pacman::kRows - 1 || col == 0 ||
              col == pacman::kCols - 1) {
            grid_[pacman::Offset(row, col)] = 1;
          }
        }
      }
    }
    pellet_row_.fill(-1);
    pellet_col_.fill(-1);
    if (use_configured_pellets_) {
      for (int pellet = 0; pellet < pacman::kNumPellets; ++pellet) {
        pellet_row_[pellet] = configured_pellet_locations_[pellet * 2];
        pellet_col_[pellet] = configured_pellet_locations_[pellet * 2 + 1];
      }
    } else {
      for (int pellet = 0; pellet < pacman::kActivePellets; ++pellet) {
        pellet_row_[pellet] = 1;
        pellet_col_[pellet] = 2 + pellet;
      }
    }
    player_row_ = configured_player_row_;
    player_col_ = configured_player_col_;
    ghost_locations_ = configured_ghost_locations_;
    power_up_locations_ = configured_power_up_locations_;
    score_ = configured_score_;
    frightened_state_time_ = configured_frightened_state_time_;
    step_count_ = 0;
    done_ = false;
    WriteState(0.0f);
  }

  void Step(const Action& action) override {
    if (use_replay_ && step_count_ < pacman::kReplaySteps) {
      ++step_count_;
      done_ = replay_done_[step_count_ - 1];
      WriteState(replay_rewards_[step_count_ - 1]);
      return;
    }
    const int action_id = std::clamp(static_cast<int>(action["action"_]), 0, 4);
    const int next_row = player_row_ + pacman::kMoves[action_id][0];
    const int next_col = player_col_ + pacman::kMoves[action_id][1];
    const bool valid = IsOpen(next_row, next_col);
    float reward = 0.0f;
    if (valid) {
      player_row_ = next_row;
      player_col_ = next_col;
      const int pellet = PelletAt(player_row_, player_col_);
      if (pellet >= 0) {
        pellet_row_[pellet] = -1;
        pellet_col_[pellet] = -1;
        score_ += 10;
        reward = 10.0f;
      }
    } else {
      reward = -1.0f;
    }
    ++step_count_;
    --frightened_state_time_;
    done_ = !valid || NoPelletsLeft() || step_count_ >= pacman::kTimeLimit;
    WriteState(reward);
  }

 private:
  bool IsOpen(int row, int col) const {
    return pacman::InGrid(row, col) && grid_[pacman::Offset(row, col)] == 0;
  }

  int PelletAt(int row, int col) const {
    for (int pellet = 0; pellet < pacman::kActivePellets; ++pellet) {
      if (pellet_row_[pellet] == row && pellet_col_[pellet] == col) {
        return pellet;
      }
    }
    return -1;
  }

  bool HasPellet(int row, int col) const { return PelletAt(row, col) >= 0; }

  bool NoPelletsLeft() const {
    for (int pellet = 0; pellet < pacman::kActivePellets; ++pellet) {
      if (pellet_row_[pellet] >= 0) {
        return false;
      }
    }
    return true;
  }

  void WriteState(float reward) {
    auto state = Allocate();
    for (int row = 0; row < pacman::kRows; ++row) {
      for (int col = 0; col < pacman::kCols; ++col) {
        state["obs:grid"_](row, col) = grid_[pacman::Offset(row, col)];
      }
    }
    state["obs:player_locations.y"_] = player_row_;
    state["obs:player_locations.x"_] = player_col_;
    if (use_replay_ && step_count_ > 0 && step_count_ <= pacman::kReplaySteps) {
      state["obs:player_locations.y"_] =
          replay_player_locations_[(step_count_ - 1) * 2];
      state["obs:player_locations.x"_] =
          replay_player_locations_[(step_count_ - 1) * 2 + 1];
    }
    for (int ghost = 0; ghost < pacman::kNumGhosts; ++ghost) {
      const int replay = ((step_count_ - 1) * pacman::kNumGhosts + ghost) * 2;
      const bool use_replay_step =
          use_replay_ && step_count_ > 0 && step_count_ <= pacman::kReplaySteps;
      state["obs:ghost_locations"_](ghost, 0) =
          use_replay_step ? replay_ghost_locations_[replay]
                          : ghost_locations_[ghost * 2];
      state["obs:ghost_locations"_](ghost, 1) =
          use_replay_step ? replay_ghost_locations_[replay + 1]
                          : ghost_locations_[ghost * 2 + 1];
    }
    for (int power_up = 0; power_up < pacman::kNumPowerUps; ++power_up) {
      const int replay =
          ((step_count_ - 1) * pacman::kNumPowerUps + power_up) * 2;
      const bool use_replay_step =
          use_replay_ && step_count_ > 0 && step_count_ <= pacman::kReplaySteps;
      state["obs:power_up_locations"_](power_up, 0) =
          use_replay_step ? replay_power_up_locations_[replay]
                          : power_up_locations_[power_up * 2];
      state["obs:power_up_locations"_](power_up, 1) =
          use_replay_step ? replay_power_up_locations_[replay + 1]
                          : power_up_locations_[power_up * 2 + 1];
    }
    state["obs:frightened_state_time"_] = frightened_state_time_;
    for (int pellet = 0; pellet < pacman::kNumPellets; ++pellet) {
      const bool use_replay_step =
          use_replay_ && step_count_ > 0 && step_count_ <= pacman::kReplaySteps;
      state["obs:pellet_locations"_](pellet, 0) =
          use_replay_step
              ? replay_pellet_locations_
                    [((step_count_ - 1) * pacman::kNumPellets + pellet) * 2]
              : pellet_row_[pellet];
      state["obs:pellet_locations"_](pellet, 1) =
          use_replay_step
              ? replay_pellet_locations_
                    [((step_count_ - 1) * pacman::kNumPellets + pellet) * 2 + 1]
              : pellet_col_[pellet];
    }
    for (int action = 0; action < 5; ++action) {
      const int row = player_row_ + pacman::kMoves[action][0];
      const int col = player_col_ + pacman::kMoves[action][1];
      if (use_replay_ && step_count_ > 0 &&
          step_count_ <= pacman::kReplaySteps) {
        state["obs:action_mask"_][action] =
            replay_action_mask_[(step_count_ - 1) * 5 + action];
      } else if (use_configured_action_mask_ && step_count_ == 0) {
        state["obs:action_mask"_][action] = configured_action_mask_[action];
      } else {
        state["obs:action_mask"_][action] = IsOpen(row, col);
      }
    }
    if (use_replay_ && step_count_ > 0 && step_count_ <= pacman::kReplaySteps) {
      state["obs:frightened_state_time"_] =
          replay_frightened_state_time_[step_count_ - 1];
      state["obs:score"_] = replay_score_[step_count_ - 1];
    } else {
      state["obs:score"_] = score_;
    }
    state["reward"_] = reward;
  }
};

using PacManEnvPool = AsyncEnvPool<PacManEnv>;

}  // namespace jumanji

#endif  // ENVPOOL_JUMANJI_PAC_MAN_ENV_H_
