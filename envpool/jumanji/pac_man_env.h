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
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/jumanji/parse_utils.h"
#include "envpool/jumanji/render_utils.h"
#include "third_party/jumanji/pacman_maze.h"

namespace jumanji {
namespace pacman {

constexpr int kRows = 31;
constexpr int kCols = 28;
constexpr int kNumGhosts = 4;
constexpr int kNumPowerUps = 4;
constexpr int kNumPellets = 318;
constexpr int kTimeLimit = 1000;
constexpr int kReplaySteps = 32;
constexpr std::array<std::array<int, 2>, 5> kMoves = {
    {{{-1, 0}}, {{0, -1}}, {{1, 0}}, {{0, 1}}, {{0, 0}}}};  // NOLINT

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
  return {std::clamp(std::stoi(text.substr(0, sep)), 0, kCols - 1),
          std::clamp(std::stoi(text.substr(sep + 1)), 0, kRows - 1)};
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
                    "obs:player_locations.y"_.Bind(Spec<int>({}, {0, 27})),
                    "obs:player_locations.x"_.Bind(Spec<int>({}, {0, 30})),
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
  pacman::GhostLocations old_ghost_locations_{};
  pacman::GhostLocations initial_ghost_locations_{};
  pacman::GhostLocations scatter_targets_{};
  std::array<int, pacman::kNumGhosts> ghost_starts_{};
  std::array<int, pacman::kNumGhosts> ghost_actions_{};
  std::array<bool, pacman::kNumGhosts> ghost_edible_{};
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
    std::tie(configured_player_col_, configured_player_row_) =
        pacman::ParsePosition(spec.config["pacman_player_location"_], 13, 23);
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
        if (grid_[pacman::Offset(row, col)] == 0) {
          render::FillCell(width, height, pacman::kRows, pacman::kCols, row,
                           col, {28, 40, 190}, rgb);
        } else if (HasPellet(row, col)) {
          auto [x, y] = render::CellCenter(width, height, pacman::kRows,
                                           pacman::kCols, row, col);
          render::FillCircle(width, height, x, y, 2, {245, 210, 170}, rgb);
        }
      }
    }
    for (int ghost = 0; ghost < pacman::kNumGhosts; ++ghost) {
      auto [x, y] = render::CellCenter(
          width, height, pacman::kRows, pacman::kCols,
          ghost_locations_[2 * ghost + 1], ghost_locations_[2 * ghost]);
      render::FillCircle(width, height, x, y, 3,
                         frightened_state_time_ > 0
                             ? render::Color{80, 100, 255}
                             : render::Palette(ghost),
                         rgb);
    }
    auto [px, py] = render::CellCenter(width, height, pacman::kRows,
                                       pacman::kCols, player_row_, player_col_);
    render::FillCircle(width, height, px, py,
                       std::max(3, std::min(width, height) / 45),
                       {255, 230, 45}, rgb);
  }

  void Reset() override {
    int pellet = 0;
    int ghost = 0;
    int power_up = 0;
    int scatter = 0;
    for (int row = 0; row < pacman::kRows; ++row) {
      for (int col = 0; col < pacman::kCols; ++col) {
        const char value = pacman::kMaze[row][col];
        grid_[pacman::Offset(row, col)] = value != 'X' ? 1 : 0;
        if (value != 'X') {
          pellet_row_[pellet] = col;
          pellet_col_[pellet++] = row;
        }
        if (value == 'G') {
          ghost_locations_[2 * ghost] = col;
          ghost_locations_[2 * ghost++ + 1] = row;
        } else if (value == 'O') {
          power_up_locations_[2 * power_up] = col;
          power_up_locations_[2 * power_up++ + 1] = row;
        } else if (value == 'S') {
          scatter_targets_[2 * scatter] = col;
          scatter_targets_[2 * scatter++ + 1] = row;
        }
      }
    }
    if (use_configured_grid_) {
      grid_ = configured_grid_;
    }
    if (use_configured_pellets_) {
      for (int index = 0; index < pacman::kNumPellets; ++index) {
        pellet_row_[index] = configured_pellet_locations_[index * 2];
        pellet_col_[index] = configured_pellet_locations_[index * 2 + 1];
      }
    }
    if (!spec_.config["pacman_ghost_locations"_].empty()) {
      ghost_locations_ = configured_ghost_locations_;
    }
    if (!spec_.config["pacman_power_up_locations"_].empty()) {
      power_up_locations_ = configured_power_up_locations_;
    }
    player_row_ = configured_player_row_;
    player_col_ = configured_player_col_;
    old_ghost_locations_ = initial_ghost_locations_ = ghost_locations_;
    ghost_starts_ = {1, 5, 10, 15};
    ghost_actions_.fill(1);
    ghost_edible_.fill(true);
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
    int next_row =
        Wrap(player_row_ + pacman::kMoves[action_id][0], pacman::kRows);
    int next_col =
        Wrap(player_col_ + pacman::kMoves[action_id][1], pacman::kCols);
    if (!IsOpen(next_row, next_col)) {
      next_row = player_row_;
      next_col = player_col_;
    }
    const auto previous_ghosts = ghost_locations_;
    MoveGhosts(action_id);
    float reward = 0.0f;
    bool dead = false;
    for (int ghost = 0; ghost < pacman::kNumGhosts; ++ghost) {
      const int col = ghost_locations_[2 * ghost];
      const int row = ghost_locations_[2 * ghost + 1];
      const bool collision = (row == next_row && col == next_col) ||
                             (row == player_row_ && col == player_col_) ||
                             (old_ghost_locations_[2 * ghost] == next_col &&
                              old_ghost_locations_[2 * ghost + 1] == next_row);
      if (collision) {
        if (frightened_state_time_ > 0) {
          reward += ghost_edible_[ghost] ? 200.0f : 0.0f;
          ghost_edible_[ghost] = false;
          ghost_locations_[2 * ghost] = initial_ghost_locations_[2 * ghost];
          ghost_locations_[2 * ghost + 1] =
              initial_ghost_locations_[2 * ghost + 1];
        } else {
          dead = true;
        }
      }
      --ghost_starts_[ghost];
    }
    old_ghost_locations_ = previous_ghosts;
    player_row_ = next_row;
    player_col_ = next_col;
    bool power_up = false;
    for (int index = 0; index < pacman::kNumPowerUps; ++index) {
      if (power_up_locations_[2 * index] == player_col_ &&
          power_up_locations_[2 * index + 1] == player_row_) {
        power_up_locations_[2 * index] = power_up_locations_[2 * index + 1] = 0;
        power_up = true;
      }
    }
    reward += power_up ? 50.0f : 0.0f;
    const int pellet = PelletAt(player_row_, player_col_);
    if (pellet >= 0) {
      pellet_row_[pellet] = pellet_col_[pellet] = 0;
      reward += 10.0f;
    }
    score_ += static_cast<int>(reward);
    ++step_count_;
    frightened_state_time_ = power_up ? 30 : frightened_state_time_ - 1;
    done_ = dead || NoPelletsLeft() || step_count_ >= pacman::kTimeLimit;
    WriteState(reward);
  }

 private:
  static int Wrap(int value, int size) { return (value % size + size) % size; }

  bool IsOpen(int row, int col) const {
    return pacman::InGrid(row, col) && grid_[pacman::Offset(row, col)] == 1;
  }

  // JAX indexing wraps negative indices and clips positive out-of-range ones.
  bool GhostCellOpen(int row, int col) const {
    row = row < 0 ? Wrap(row, pacman::kRows) : std::min(row, pacman::kRows - 1);
    col = col < 0 ? Wrap(col, pacman::kCols) : std::min(col, pacman::kCols - 1);
    return IsOpen(row, col);
  }

  void MoveGhosts(int player_action) {
    constexpr std::array<std::array<int, 2>, 5> moves = {
        {{{0, -1}}, {{-1, 0}}, {{0, 1}}, {{1, 0}}, {{0, 0}}}};  // NOLINT
    for (int ghost = 0; ghost < pacman::kNumGhosts; ++ghost) {
      const int row = ghost_locations_[2 * ghost + 1];
      const int col = ghost_locations_[2 * ghost];
      int choice = 4;
      if (ghost_starts_[ghost] < 0) {
        std::array<bool, 4> valid{};
        std::array<int, 4> distances{};
        distances.fill(std::numeric_limits<int>::max());
        for (int action = 0; action < 4; ++action) {
          const int next_row = row + moves[action][0];
          const int next_col = col + moves[action][1];
          valid[action] = GhostCellOpen(next_row, next_col);
          const bool reverse =
              next_row == old_ghost_locations_[2 * ghost + 1] &&
              next_col == old_ghost_locations_[2 * ghost];
          if (!valid[action] || reverse) {
            continue;
          }
          int dr = next_row - player_row_;
          int dc = next_col - player_col_;
          const bool scatter =
              frightened_state_time_ > 0 ||
              (ghost == 3 &&
               (row - player_row_) * (row - player_row_) +
                       (col - player_col_) * (col - player_col_) <=
                   64);
          if (scatter) {
            // Preserve the pinned oracle's coordinate convention for targets.
            dr = next_row - scatter_targets_[2 * ghost];
            dc = next_col - scatter_targets_[2 * ghost + 1];
          } else if (ghost == 1 || ghost == 2) {
            const int target_row =
                Wrap(player_row_ + 4 * pacman::kMoves[player_action][0],
                     pacman::kRows);
            const int target_col =
                Wrap(player_col_ + 4 * pacman::kMoves[player_action][1],
                     pacman::kCols);
            if (ghost == 1) {
              dr = next_row - target_col;
              dc = next_col - target_row;
            } else {
              dr += next_row - target_col;
              dc += next_col - target_row;
            }
          }
          distances[action] = dr * dr + dc * dc;
        }
        if (valid == std::array<bool, 4>{true, false, true, false} ||
            valid == std::array<bool, 4>{false, true, false, true}) {
          choice = ghost_actions_[ghost];
        } else {
          const int closest =
              *std::min_element(distances.begin(), distances.end());
          std::vector<int> candidates;
          for (int action = 0; action < 4; ++action) {
            if (distances[action] == closest) {
              candidates.push_back(action);
            }
          }
          choice = candidates[std::uniform_int_distribution<int>(
              0, static_cast<int>(candidates.size()) - 1)(gen_)];
        }
      }
      ghost_actions_[ghost] = choice;
      ghost_locations_[2 * ghost] = Wrap(col + moves[choice][1], pacman::kCols);
      ghost_locations_[2 * ghost + 1] =
          Wrap(row + moves[choice][0], pacman::kRows);
    }
  }

  int PelletAt(int row, int col) const {
    for (int pellet = 0; pellet < pacman::kNumPellets; ++pellet) {
      if (pellet_row_[pellet] == col && pellet_col_[pellet] == row) {
        return pellet;
      }
    }
    return -1;
  }

  bool HasPellet(int row, int col) const { return PelletAt(row, col) >= 0; }

  bool NoPelletsLeft() const {
    for (int pellet = 0; pellet < pacman::kNumPellets; ++pellet) {
      if (pellet_row_[pellet] > 0 || pellet_col_[pellet] > 0) {
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
    state["obs:player_locations.y"_] = player_col_;
    state["obs:player_locations.x"_] = player_row_;
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
        state["obs:action_mask"_][action] =
            action != 4 && GhostCellOpen(row, col);
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
