/*
 * Copyright 2023 Garena Online Private Limited
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

#ifndef ENVPOOL_PROCGEN_PROCGEN_ENV_H_
#define ENVPOOL_PROCGEN_PROCGEN_ENV_H_

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "game.h"

namespace procgen {

/*
   All the procgen's games have the same observation buffer size, 64 x 64 pixels
   x 3 colors (RGB) there are 15 possible action buttoms and observation is RGB
   32 or RGB 888,
   QT build needs: sudo apt update && sudo apt install qtdeclarative5-dev
 */
static const int kRes = 64;
static std::once_flag procgen_global_init_flag;

void ProcgenGlobalInit(std::string path) {
  if (global_resource_root.empty()) {
    global_resource_root = std::move(path);
    images_load();
  }
}

// https://github.com/openai/procgen/blob/0.10.7/procgen/src/vecgame.cpp#L156
std::size_t HashStrUint32(const std::string& str) {
  std::size_t hash = 0x811c9dc5;
  std::size_t prime = 0x1000193;
  for (uint8_t value : str) {
    hash ^= value;
    hash *= prime;
  }
  return hash;
}

class ProcgenEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "env_name"_.Bind(std::string("bigfish")), "channel_first"_.Bind(true),
        "num_levels"_.Bind(0), "start_level"_.Bind(0),
        "use_sequential_levels"_.Bind(false), "center_agent"_.Bind(true),
        "use_backgrounds"_.Bind(true), "use_monochrome_assets"_.Bind(false),
        "restrict_themes"_.Bind(false), "use_generated_assets"_.Bind(false),
        "paint_vel_info"_.Bind(false), "use_easy_jump"_.Bind(false),
        "distribution_mode"_.Bind(1));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    // The observation is RGB 64 x 64 x 3
    return MakeDict(
        "obs"_.Bind(Spec<uint8_t>(conf["channel_first"_]
                                      ? std::vector<int>{3, kRes, kRes}
                                      : std::vector<int>{kRes, kRes, 3},
                                  {0, 255})),
        "info:prev_level_seed"_.Bind(Spec<int>({-1})),
        "info:prev_level_complete"_.Bind(Spec<int>({-1})),
        "info:level_seed"_.Bind(Spec<int>({-1})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    // 15 action buttons in total, ranging from 0 to 14
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 14})));
  }
};

using ProcgenEnvSpec = EnvSpec<ProcgenEnvFns>;
using FrameSpec = Spec<uint8_t>;

class ProcgenEnv : public Env<ProcgenEnvSpec> {
 protected:
  std::shared_ptr<Game> game_;
  std::string env_name_;
  bool channel_first_;
  // buffer used by game
  FrameSpec obs_spec_;
  Array obs_;
  float reward_;
  int level_seed_, prev_level_seed_;
  uint8_t done_{1}, prev_level_complete_;

 public:
  ProcgenEnv(const Spec& spec, int env_id)
      : Env<ProcgenEnvSpec>(spec, env_id),
        env_name_(spec.config["env_name"_]),
        channel_first_(spec.config["channel_first"_]),
        obs_spec_({kRes, kRes, 3}),
        obs_(obs_spec_) {
    /* Initialize the single game we are holding in this EnvPool environment
     * It depends on some default setting along with the config map passed in
     * We mostly follow how it's done in the vector environment at Procgen and
     * translate it into single one.
     * https://github.com/openai/procgen/blob/0.10.7/procgen/src/vecgame.cpp#L312
     */
    std::call_once(procgen_global_init_flag, ProcgenGlobalInit,
                   spec.config["base_path"_] + "/procgen/assets/");
    // CHECK_NE(globalGameRegistry, nullptr);
    // game_ = globalGameRegistry->at(env_name_)();
    game_ = make_game(spec.config["env_name"_]);
    CHECK_EQ(game_->game_name, env_name_);
    game_->level_seed_rand_gen.seed(seed_);
    int num_levels = spec.config["num_levels"_];
    int start_level = spec.config["start_level"_];
    if (num_levels <= 0) {
      game_->level_seed_low = 0;
      game_->level_seed_high = std::numeric_limits<int>::max();
    } else {
      game_->level_seed_low = start_level;
      game_->level_seed_high = start_level + num_levels;
    }
    game_->game_n = env_id;
    if (game_->fixed_asset_seed == 0) {
      game_->fixed_asset_seed = static_cast<int>(HashStrUint32(env_name_));
    }

    // buffers for the game to outwrite observations each step
    game_->reward_ptr = &reward_;
    game_->first_ptr = &done_;
    game_->obs_bufs.emplace_back(static_cast<void*>(obs_.Data()));
    game_->info_bufs.emplace_back(static_cast<void*>(&level_seed_));
    game_->info_bufs.emplace_back(static_cast<void*>(&prev_level_seed_));
    game_->info_bufs.emplace_back(static_cast<void*>(&prev_level_complete_));
    game_->info_name_to_offset["level_seed"] = 0;
    game_->info_name_to_offset["prev_level_seed"] = 1;
    game_->info_name_to_offset["prev_level_complete"] = 2;
    // game options
    game_->options.use_easy_jump = spec.config["use_easy_jump"_];
    game_->options.paint_vel_info = spec.config["paint_vel_info"_];
    game_->options.use_generated_assets = spec.config["use_generated_assets"_];
    game_->options.use_monochrome_assets =
        spec.config["use_monochrome_assets"_];
    game_->options.restrict_themes = spec.config["restrict_themes"_];
    game_->options.use_backgrounds = spec.config["use_backgrounds"_];
    game_->options.center_agent = spec.config["center_agent"_];
    game_->options.use_sequential_levels =
        spec.config["use_sequential_levels"_];
    game_->options.distribution_mode =
        static_cast<DistributionMode>(spec.config["distribution_mode"_]);
    game_->game_init();
  }

  void Reset() override {
    game_->step_data.done = false;
    game_->step_data.reward = 0.0;
    game_->step_data.level_complete = false;
    game_->reset();
    game_->observe();
    WriteObs();
  }

  void Step(const Action& action) override {
    game_->action = action["action"_];
    game_->step();
    WriteObs();
  }

  bool IsDone() override { return done_ != 0; }

 private:
  void WriteObs() {
    State state = Allocate();
    if (channel_first_) {
      // convert from HWC to CHW
      auto* data = static_cast<uint8_t*>(state["obs"_].Data());
      auto* buffer = static_cast<uint8_t*>(obs_.Data());
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < kRes; ++j) {
          for (int k = 0; k < kRes; ++k) {
            data[i * kRes * kRes + j * kRes + k] =
                buffer[j * kRes * 3 + k * 3 + i];
          }
        }
      }
    } else {
      state["obs"_].Assign(obs_);
    }
    state["reward"_] = reward_;
    state["info:prev_level_seed"_] = prev_level_seed_;
    state["info:prev_level_complete"_] = prev_level_complete_;
    state["info:level_seed"_] = level_seed_;
  }
};

using ProcgenEnvPool = AsyncEnvPool<ProcgenEnv>;

}  // namespace procgen

#endif  // ENVPOOL_PROCGEN_PROCGEN_ENV_H_
