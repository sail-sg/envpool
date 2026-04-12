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

#ifndef ENVPOOL_GFOOTBALL_GFOOTBALL_COMMON_H_
#define ENVPOOL_GFOOTBALL_GFOOTBALL_COMMON_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>

#include "defines.hpp"
#include "game_env.hpp"
#include "gfootball_actions.h"
#include "main.hpp"

namespace gfootball {

inline constexpr int kSMMWidth = 96;
inline constexpr int kSMMHeight = 72;
inline constexpr int kSMMChannels = 4;
inline constexpr int kActionCount = 19;
inline constexpr int kGameModeNormal = 0;

inline constexpr double kMinimapXMin = -1.0;
inline constexpr double kMinimapXMax = 1.0;
inline constexpr double kMinimapYMin = -1.0 / 2.25;
inline constexpr double kMinimapYMax = 1.0 / 2.25;
inline constexpr std::uint8_t kMarkerValue = 255;

inline constexpr std::array<int, kActionCount> kDefaultActionSet = {
    game_idle,
    game_left,
    game_top_left,
    game_top,
    game_top_right,
    game_right,
    game_bottom_right,
    game_bottom,
    game_bottom_left,
    game_long_pass,
    game_high_pass,
    game_short_pass,
    game_shot,
    game_sprint,
    game_release_direction,
    game_release_sprint,
    game_sliding,
    game_dribble,
    game_release_dribble,
};

inline void SetEnvVar(const char* key, const std::string& value) {
#ifdef _WIN32
  _putenv_s(key, value.c_str());
#else
  setenv(key, value.c_str(), 1);
#endif
}

inline void SetEnvVarIfMissing(const char* key, const std::string& value) {
  if (std::getenv(key) == nullptr) {
    SetEnvVar(key, value);
  }
}

inline void EnsureGfootballRuntimePaths(const std::string& base_path) {
  SetEnvVar("GFOOTBALL_DATA_DIR", base_path + "/gfootball/assets/data");
  SetEnvVar("GFOOTBALL_FONT",
            base_path + "/gfootball/assets/fonts/AlegreyaSansSC-ExtraBold.ttf");
#if defined(__linux__)
  SetEnvVarIfMissing("MESA_GL_VERSION_OVERRIDE", "3.2");
  SetEnvVarIfMissing("MESA_GLSL_VERSION_OVERRIDE", "150");
#endif
}

inline unsigned int SampleEngineSeed(std::mt19937* gen) {
  std::uniform_int_distribution<unsigned int> dist(0, 2000000000U);
  return dist(*gen);
}

inline int ResolveBackendAction(int action) {
  return kDefaultActionSet[std::clamp(action, 0, kActionCount - 1)];
}

inline int MinimapCoordX(double value) {
  const double scaled =
      (value - kMinimapXMin) / (kMinimapXMax - kMinimapXMin) * kSMMWidth;
  return std::clamp(static_cast<int>(scaled), 0, kSMMWidth - 1);
}

inline int MinimapCoordY(double value) {
  const double scaled =
      (value - kMinimapYMin) / (kMinimapYMax - kMinimapYMin) * kSMMHeight;
  return std::clamp(static_cast<int>(scaled), 0, kSMMHeight - 1);
}

#include "envpool/gfootball/gfootball_scenarios.inc"

inline void BuildEnvScenarioConfig(const std::string& env_name,
                                   int episode_number,
                                   unsigned int game_engine_random_seed,
                                   int max_episode_steps, ScenarioConfig* cfg) {
  cfg->game_engine_random_seed = game_engine_random_seed;
  cfg->reverse_team_processing = false;
  BuildScenarioConfig(env_name, episode_number, cfg);
  if (!cfg->deterministic) {
    cfg->reverse_team_processing = (game_engine_random_seed % 2) != 0;
  }
  if (max_episode_steps > 0) {
    cfg->game_duration = max_episode_steps;
  }
}

}  // namespace gfootball

#endif  // ENVPOOL_GFOOTBALL_GFOOTBALL_COMMON_H_
