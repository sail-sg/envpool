/*
 * Copyright 2021 Garena Online Private Limited
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

#ifndef ENVPOOL_PROCGEN_PROCGEN_H_
#define ENVPOOL_PROCGEN_PROCGEN_H_

#include <cctype>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/procgen/third_party_procgen.h"

namespace procgen {

/*
   All the procgen's games have the same observation buffer size, 64 x 64 pixels
   x 3 colors (RGB) there are 15 possible action buttoms and observation is RGB
   32 or RGB 888,
   https://github.com/openai/procgen/blob/5e1dbf341d291eff40d1f9e0c0a0d5003643aebf/procgen/src/game.h#L23
 */
static const int RES_W = 64;
static const int RES_H = 64;
static const int RGB_FACTOR = 3;
static const int ACTION_NUM = 15;  // 0 ~ 14 both sides included

/*
   game factory method for fetching a game instance
   Notice the inheritance hierarchy is Game > BasicAbstractGame > [detailed 15
   games]
*/
std::shared_ptr<Game> make_game(std::string name) {
  if (name == "bigfish") {
    return make_bigfish();
  } else if (name == "bossfight") {
    return make_bossfight();
  } else if (name == "caveflyer") {
    return make_caveflyer();
  } else if (name == "chaser") {
    return make_chaser();
  } else if (name == "climber") {
    return make_climber();
  } else if (name == "coinrun") {
    return make_coinrun();
  } else if (name == "dodgeball") {
    return make_dodgeball();
  } else if (name == "fruitbot") {
    return make_fruitbot();
  } else if (name == "heist") {
    return make_heist();
  } else if (name == "jumper") {
    return make_jumper();
  } else if (name == "leaper") {
    return make_leaper();
  } else if (name == "maze") {
    return make_maze();
  } else if (name == "miner") {
    return make_miner();
  } else if (name == "ninja") {
    return make_ninja();
  } else if (name == "plunder") {
    return make_plunder();
  } else if (name == "starpilot") {
    return make_starpilot();
  } else {
    // not supposed to reach here
    return make_bigfish();
  }
}

class ProcgenEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    /* necessary default parameters for procgen games */
    /* https://github.com/openai/procgen/blob/5e1dbf341d291eff40d1f9e0c0a0d5003643aebf/procgen/src/game.h#L69
     */
    return MakeDict("state_num"_.Bind(RES_W * RES_H * RGB_FACTOR),
                    "action_num"_.Bind(ACTION_NUM),
                    "game_name"_.Bind(std::string("bigfish")),
                    "use_sequential_levels"_.Bind(false), "num_levels"_.Bind(0),
                    "start_level"_.Bind(0), "distribution_mode"_.Bind(1));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    /* The observation is RGB 64 x 64 x 3 flattened out into one row plus action
     * taken and if done */
    return MakeDict("obs:obs"_.Bind(Spec<uint8_t>({-1, conf["state_num"_]})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    /* 15 action buttons in total, ranging from 0 to 14 */
    return MakeDict(
        "action"_.Bind(Spec<int>({-1}, {0, conf["action_num"_] - 1})));
  }
};

typedef class EnvSpec<ProcgenEnvFns> ProcgenEnvSpec;

class ProcgenEnv : public Env<ProcgenEnvSpec> {
 protected:
  std::shared_ptr<Game> game_;
  bool done_{false};
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
    game_ = make_game(spec.config["game_name"_]);
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
    obs_bufs_.resize(RES_W * RES_H);
    info_bufs_.resize(RES_W * RES_H);
    for (int i = 0; i < RES_W * RES_H; i++) {
      obs_bufs_[i] = new int64_t[RES_W * RES_H];
      info_bufs_[i] = new int64_t[RES_W * RES_H];
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
    State state = Allocate();
    WriteObs(state);
  }

  void Step(const Action& action) override {
    /* Delegate the action to procgen game and let it step */
    int act = action["action"_];
    game_->action = static_cast<int32_t>(act);
    *(game_->action_ptr) = static_cast<int32_t>(act);
    game_->step();
    done_ = game_->step_data.done;
    State state = Allocate();
    WriteObs(state);
  }

  bool IsDone() override { return done_; }

 private:
  void WriteObs(State& state) {  // NOLINT
    /* Helper function to output the information to user at current step */
    /*
       It includes:
       1. The RGB 64 x 64 frame observation
       2. Current step's reward
       https://github.com/openai/procgen/blob/5e1dbf341d291eff40d1f9e0c0a0d5003643aebf/procgen/src/game.cpp#L8
    */

    uint8_t* src = (uint8_t*)obs_bufs_[0];  // NOLINT
    for (int y = 0; y < RES_H; y++) {
      for (int x = 0; x < RES_W; x++) {
        for (int rgb = 0; rgb < RGB_FACTOR; rgb++) {
          int offset = rgb + x * RGB_FACTOR + y * RES_W * RGB_FACTOR;
          state["obs:obs"_][0][offset] = static_cast<uint8_t>(src[offset]);
        }
      }
    }
    state["reward"_] = static_cast<float>(*(game_->reward_ptr));
  }
};

typedef AsyncEnvPool<ProcgenEnv> ProcgenEnvPool;

}  // namespace procgen

#endif  // ENVPOOL_PROCGEN_PROCGEN_H_
