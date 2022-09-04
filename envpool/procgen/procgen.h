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

#ifndef ENVPOOL_PROCGEN_H_
#define ENVPOOL_PROCGEN_H_

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/procgen/third_party_procgen.h"

namespace procgen {

/* All the procgen's games have the same observation buffer size, 64 x 64 pixels
 */
/* there are 15 possible action buttoms and observation is RGB 32 or RGB 888,
 * both available              */
/* https://github.com/openai/procgen/blob/5e1dbf341d291eff40d1f9e0c0a0d5003643aebf/procgen/src/game.h#L23
 */
static const int RES_W = 64;
static const int RES_H = 64;
static const int RGB_FACTOR = 3;
static const int ACTION_NUM = 15;  // 0 ~ 14 both sides included

/* System independent hashing, identical to the one implemented in Procgen  */
/* https://github.com/openai/procgen/blob/HEAD/procgen/src/vecgame.cpp#L156 */
inline uint32_t hash_str_uint32(const std::string& str) {
  uint32_t hash = 0x811c9dc5;
  uint32_t prime = 0x1000193;

  for (size_t i = 0; i < str.size(); i++) {
    uint8_t value = str[i];
    hash = hash ^ value;
    hash *= prime;
  }

  return hash;
}

class ProcgenEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    /* necessary default parameters for procgen games */
    /* https://github.com/openai/procgen/blob/5e1dbf341d291eff40d1f9e0c0a0d5003643aebf/procgen/src/game.h#L69
     */
    return MakeDict(
        "state_num"_.Bind(RES_W * RES_H * RGB_FACTOR),
        "action_num"_.Bind(ACTION_NUM), "initial_reset_complete"_.Bind(false),
        "grid_step"_.Bind(false), "level_seed_low"_.Bind(0),
        "level_seed_high"_.Bind(1), "game_type"_.Bind(0), "game_n"_.Bind(0),
        "game_name"_.Bind(std::string("bigfish")), "rand_seed"_.Bind(0),
        "action"_.Bind(0), "timeout"_.Bind(1000), "cur_time"_.Bind(0),
        "episodes_remaining"_.Bind(0), "episode_done"_.Bind(false),
        "last_reward"_.Bind(-1), "last_reward_timer"_.Bind(0),
        "default_action"_.Bind(0), "fixed_asset_seed"_.Bind(0),
        "reset_count"_.Bind(0), "current_level_seed"_.Bind(0),
        "prev_level_seed"_.Bind(0), "num_levels"_.Bind(0),
        "start_level"_.Bind(0), "distribution_mode"_.Bind(1));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    /* The observation is RGB 64 x 64 x 3 flattened out into one row plus action
     * taken and if done */
    return MakeDict(
        "obs"_.Bind(Spec<int>({-1, conf["state_num"_]})),
        "action"_.Bind(Spec<int>({-1}, {0, conf["action_num"_] - 1})),
        "episode_done"_.Bind(Spec<int>({-1}, {0, 1})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    /* 15 action buttoms in total, ranging from 0 to 14 */
    return MakeDict(
        "action"_.Bind(Spec<int>({-1}, {0, conf["action_num"_] - 1})));
  }
};

typedef class EnvSpec<ProcgenEnvFns> ProcgenEnvSpec;

class ProcgenEnv : public Env<ProcgenEnvSpec> {
 protected:
  std::shared_ptr<Game> game_;
  bool done_;
  RandGen game_level_seed_gen_;
  int rand_seed_;
  std::map<std::string, int> info_name_to_offset_;
  std::vector<void*> obs_bufs_;
  std::vector<void*> info_bufs_;

 public:
  ProcgenEnv(const Spec& spec, int env_id)
      : Env<ProcgenEnvSpec>(spec, env_id),
        game_(globalGameRegistry->at(std::string(spec.config["game_name"_]))()),
        rand_seed_(spec.config["rand_seed"_]) {
    /* Initialize the single game we are holding in this EnvPool environment */
    /* It depends on some default setting along with the config map passed in */
    /* We mostly follow how it's done in the vector environment at Procgen and
     * translate it into single one */
    /* https://github.com/openai/procgen/blob/5e1dbf341d291eff40d1f9e0c0a0d5003643aebf/procgen/src/vecgame.cpp#L312
     */
    game_level_seed_gen_.seed(rand_seed_);
    game_->level_seed_rand_gen.seed(game_level_seed_gen_.randint());
    game_->level_seed_low = spec.config["level_seed_low"_];
    game_->level_seed_high = spec.config["level_seed_high"_];
    game_->timeout = spec.config["timeout"_];
    game_->game_n = spec.config["game_n"_];
    game_->is_waiting_for_step = false;
    info_name_to_offset_["rgb"] = 0;
    info_name_to_offset_["action"] = 1;
    info_name_to_offset_["prev_level_seed"] = 2;
    info_name_to_offset_["prev_level_complete"] = 3;
    info_name_to_offset_["level_seed"] = 4;
    game_->info_name_to_offset = info_name_to_offset_;
    game_->options.distribution_mode =
        static_cast<DistributionMode>(spec.config["distribution_mode"_]);
    // allocate space for the game to outwrite observations each step
    obs_bufs_.resize(RGB_FACTOR * RES_W * RES_H);
    info_bufs_.resize(info_name_to_offset_.size());
    game_->obs_bufs = obs_bufs_;
    game_->info_bufs = info_bufs_;
    game_->game_init();
  }

  void Reset() override {
    /* procgen game has itself reset method that clears out the internal state
     * of the game */
    done_ = false;
    game_->reset();
    State state = Allocate();
    WriteObs(state);
  }

  void Step(const Action& action) override {
    /* Delegate the action to procgen game and let it step */
    int act = action["action"_];
    game_->action = act;
    *(game_->action_ptr) = static_cast<int32_t>(act);
    game_->step();
    done_ = game_->step_data.done;
    State state = Allocate();
    WriteObs(state);
  }

  bool IsDone() override { return done_; }

 private:
  void WriteObs(State& state) {
    /* Helper function to output the information to user at current step */
    /* It includes:
       1. The RGB 64 x 64 frame observation
       2. The action taken this step
       3. Current step's reward
    */
    /* https://github.com/openai/procgen/blob/5e1dbf341d291eff40d1f9e0c0a0d5003643aebf/procgen/src/game.cpp#L8
     */
    uint8_t* src = (uint8_t*)(void*)obs_bufs_[0];
    for (int y = 0; y < RES_H; y++) {
      for (int x = 0; x < RES_W; x++) {
        for (int rgb = 0; rgb < RGB_FACTOR; rgb++) {
          int offset = rgb + x * RGB_FACTOR + y * RES_W * RGB_FACTOR;
          state["obs"_][offset] = static_cast<int>(src[offset]);
        }
      }
    }
    state["reward"_] = static_cast<float>(*(game_->reward_ptr));
    state["action"_] = static_cast<int>(game_->action);
    state["episode_done"_] = static_cast<int>(game_->episode_done);
  }
};

typedef AsyncEnvPool<ProcgenEnv> ProcgenEnvPool;

}  // namespace procgen

#endif