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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <stdexcept>
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
inline constexpr int kRenderChannels = 3;
inline constexpr int kGameModeNormal = 0;

inline constexpr double kMinimapXMin = -1.0;
inline constexpr double kMinimapXMax = 1.0;
inline constexpr double kMinimapYMin = -1.0 / 2.25;
inline constexpr double kMinimapYMax = 1.0 / 2.25;
inline constexpr std::uint8_t kMarkerValue = 255;

inline constexpr std::array<int, kActionCount> kDefaultActionSet = {
    game_idle,           game_left,         game_top_left,
    game_top,            game_top_right,    game_right,
    game_bottom_right,   game_bottom,       game_bottom_left,
    game_long_pass,      game_high_pass,    game_short_pass,
    game_shot,           game_sprint,       game_release_direction,
    game_release_sprint, game_sliding,      game_dribble,
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
            base_path +
                "/gfootball/assets/fonts/AlegreyaSansSC-ExtraBold.ttf");
#if defined(__linux__)
  SetEnvVarIfMissing("MESA_GL_VERSION_OVERRIDE", "3.2");
  SetEnvVarIfMissing("MESA_GLSL_VERSION_OVERRIDE", "150");
#endif
}

inline unsigned int SampleEngineSeed(std::mt19937* gen) {
  std::uniform_int_distribution<unsigned int> dist(0, 2000000000U);
  return dist(*gen);
}

inline int ResolveRenderHeight(int render_width) {
  return static_cast<int>(std::lround(0.5625 * render_width));
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

inline std::size_t FrameSizeBytes(int render_width, int render_height) {
  return static_cast<std::size_t>(render_width) * render_height *
         kRenderChannels;
}

inline void TransformFrameToRgb(const screenshoot& raw_frame, int render_width,
                                int render_height, unsigned char* dst_rgb) {
  const std::size_t expected_size = FrameSizeBytes(render_width, render_height);
  if (raw_frame.size() != expected_size) {
    throw std::runtime_error("Unexpected gfootball frame size");
  }
  const auto* src = reinterpret_cast<const unsigned char*>(raw_frame.data());
  for (int src_x = 0; src_x < render_width; ++src_x) {
    for (int src_y = 0; src_y < render_height; ++src_y) {
      const std::size_t src_index =
          (static_cast<std::size_t>(src_x) * render_height + src_y) *
          kRenderChannels;
      const int dst_y = render_height - 1 - src_y;
      const std::size_t dst_index =
          (static_cast<std::size_t>(dst_y) * render_width + src_x) *
          kRenderChannels;
      dst_rgb[dst_index + 0] = src[src_index + 2];
      dst_rgb[dst_index + 1] = src[src_index + 1];
      dst_rgb[dst_index + 2] = src[src_index + 0];
    }
  }
}

#include "envpool/gfootball/gfootball_scenarios.inc"

inline void BuildEnvScenarioConfig(const std::string& env_name,
                                   int episode_number,
                                   unsigned int game_engine_random_seed,
                                   int max_episode_steps,
                                   ScenarioConfig* cfg) {
  BuildScenarioConfig(env_name, episode_number, game_engine_random_seed, cfg);
  if (max_episode_steps > 0) {
    cfg->game_duration = max_episode_steps;
  }
}

}  // namespace gfootball

#endif  // ENVPOOL_GFOOTBALL_GFOOTBALL_COMMON_H_
