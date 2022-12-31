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

#include <cctype>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "game.h"

namespace procgen {

/*
   All the procgen's games have the same observation buffer size, 64 x 64 pixels
   x 3 colors (RGB) there are 15 possible action buttoms and observation is RGB
   32 or RGB 888,
   QT library build needs:
   sudo apt update && sudo apt install qt5-default qtdeclarative5-dev
 */
static const int kResW = 64;
static const int kResH = 64;

class ProcgenEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "env_name"_.Bind(std::string("bigfish")), "num_levels"_.Bind(0),
        "start_level"_.Bind(0), "use_sequential_levels"_.Bind(false),
        "center_agent"_.Bind(true), "use_backgrounds"_.Bind(true),
        "use_monochrome_assets"_.Bind(false), "restrict_themes"_.Bind(false),
        "use_generated_assets"_.Bind(false), "paint_vel_info"_.Bind(false),
        "distribution_mode"_.Bind(1));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    /* The observation is RGB 64 x 64 x 3 flattened out into one row plus action
     * taken and if done */
    return MakeDict("obs"_.Bind(Spec<uint8_t>({kResH, kResW, 3}, {0, 255})),
                    "info:prev_level_seed"_.Bind(Spec<int>({-1})),
                    "info:prev_level_complete"_.Bind(Spec<int>({-1})),
                    "info:level_seed"_.Bind(Spec<int>({-1})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    /* 15 action buttons in total, ranging from 0 to 14 */
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 14})));
  }
};

using ProcgenEnvSpec = EnvSpec<ProcgenEnvFns>;

class ProcgenEnv : public Env<ProcgenEnvSpec> {
 protected:
  std::shared_ptr<Game> game_;
  bool done_{true};
  RandGen game_level_seed_gen_;
  int rand_seed_;
  std::map<std::string, int> info_name_to_offset_;
  std::vector<void*> obs_bufs_;
  std::vector<void*> info_bufs_;

 public:
  ProcgenEnv(const Spec& spec, int env_id)
      : Env<ProcgenEnvSpec>(spec, env_id), rand_seed_(spec.config["seed"_]) {
    /* Initialize the single game we are holding in this EnvPool environment */
    /* It depends on some default setting along with the config map passed in */
    /* We mostly follow how it's done in the vector environment at Procgen and
     * translate it into single one */
    /* https://github.com/openai/procgen/blob/5e1dbf341d291eff40d1f9e0c0a0d5003643aebf/procgen/src/vecgame.cpp#L312
     */
    /* notice we need to allocate space for some buffer, as specificied here
       https://github.com/openai/procgen/blob/master/procgen/src/game.h#L101
    */
    game_ = globalGameRegistry->at(spec.config["env_name"_])();
    game_level_seed_gen_.seed(rand_seed_);
    game_->level_seed_rand_gen.seed(game_level_seed_gen_.randint());
    if (spec.config["num_levels"_] <= 0) {
      game_->level_seed_low = 0;
      game_->level_seed_high = INT32_MAX;
    } else {
      game_->level_seed_low = spec.config["start_level"_];
      game_->level_seed_high =
          spec.config["start_level"_] + spec.config["num_levels"_];
    }
    info_name_to_offset_["rgb"] = 0;
    info_name_to_offset_["action"] = 1;
    info_name_to_offset_["prev_level_seed"] = 2;
    info_name_to_offset_["prev_level_complete"] = 3;
    info_name_to_offset_["level_seed"] = 4;
    game_->info_name_to_offset = info_name_to_offset_;
    game_->options.distribution_mode =
        static_cast<DistributionMode>(spec.config["distribution_mode"_]);
    // allocate space for the game to outwrite observations each step
    game_->action_ptr = new int32_t(0);
    game_->reward_ptr = new float(0.0);
    game_->first_ptr = new uint8_t(0);
    obs_bufs_.resize(kResW * kResH);
    info_bufs_.resize(kResW * kResH);
    for (int i = 0; i < kResW * kResH; i++) {
      obs_bufs_[i] = new int64_t[kResW * kResH];
      info_bufs_[i] = new int64_t[kResW * kResH];
    }
    game_->obs_bufs = obs_bufs_;
    game_->info_bufs = info_bufs_;
    // if use_generated_assets is not set,
    // it will try load some pictures we don't have
    game_->options.use_generated_assets = true;
    game_->options.use_sequential_levels =
        spec.config["use_sequential_levels"_];
    game_->game_init();
    game_->reset();
    game_->initial_reset_complete = true;
  }

  void Reset() override {
    /* procgen game has itself reset method that clears out the internal state
     * of the game */
    // no need to call game_->reset()
    // in game_->step(), if dies, it will reset itself
    game_->step_data.done = false;
    game_->step_data.reward = 0.0;
    game_->step_data.level_complete = false;
    done_ = false;
    game_->observe();
    WriteObs();
  }

  void Step(const Action& action) override {
    /* Delegate the action to procgen game and let it step */
    int act = action["action"_];
    game_->action = static_cast<int32_t>(act);
    *(game_->action_ptr) = static_cast<int32_t>(act);
    game_->step();
    done_ = game_->step_data.done;
    WriteObs();
  }

  bool IsDone() override { return done_; }

 private:
  void WriteObs() {
    /* Helper function to output the information to user at current step */
    /*
       It includes:
       1. The RGB 64 x 64 frame observation
       2. Current step's reward
       https://github.com/openai/procgen/blob/5e1dbf341d291eff40d1f9e0c0a0d5003643aebf/procgen/src/game.cpp#L8
    */
    State state = Allocate();
    // state["obs"_].Assign(obs_bufs_[0]);
    state["reward"_] = static_cast<float>(*(game_->reward_ptr));
  }
};

using ProcgenEnvPool = AsyncEnvPool<ProcgenEnv>;

}  // namespace procgen

#endif  // ENVPOOL_PROCGEN_PROCGEN_ENV_H_
