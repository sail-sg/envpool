// Copyright 2026 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ENVPOOL_CRAFTAX_CRAFTAX_H_
#define ENVPOOL_CRAFTAX_CRAFTAX_H_

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/craftax/renderer.h"
#include "envpool/craftax/state_io.h"
#include "third_party/craftax/constants.h"
#include "third_party/craftax/info.h"

namespace craftax {

template <bool Classic, bool Pixel>
struct CraftaxEnvFns {
  static auto DefaultConfig() {
    const Params params(Classic);
    return MakeDict(
        "map_size"_.Bind(std::vector<int>{params.height, params.width}),
        "num_levels"_.Bind(params.levels),
        "day_length"_.Bind(params.day_length),
        "always_diamond"_.Bind(params.always_diamond), "god_mode"_.Bind(false),
        "mob_despawn_distance"_.Bind(params.mob_despawn_distance),
        "max_attribute"_.Bind(params.max_attribute),
        "max_melee_mobs"_.Bind(params.max_melee_mobs),
        "max_passive_mobs"_.Bind(params.max_passive_mobs),
        "max_ranged_mobs"_.Bind(params.max_ranged_mobs),
        "max_mob_projectiles"_.Bind(params.max_mob_projectiles),
        "max_player_projectiles"_.Bind(params.max_player_projectiles),
        "max_growing_plants"_.Bind(params.max_growing_plants),
        "zombie_health"_.Bind(params.zombie_health),
        "cow_health"_.Bind(params.cow_health),
        "skeleton_health"_.Bind(params.skeleton_health),
        "spawn_cow_chance"_.Bind(params.spawn_cow_chance),
        "spawn_zombie_base_chance"_.Bind(params.spawn_zombie_base_chance),
        "spawn_zombie_night_chance"_.Bind(params.spawn_zombie_night_chance),
        "spawn_skeleton_chance"_.Bind(params.spawn_skeleton_chance),
        "fractal_noise_angles"_.Bind(std::vector<std::vector<float>>(4)),
        "auto_reset"_.Bind(false), "render_tile_size"_.Bind(16),
        "initial_state"_.Bind(std::vector<double>{}),
        "debug_state"_.Bind(false));
  }

  template <typename Config>
  static Params Parameters(const Config& conf) {
    Params params(Classic);
    params.symbolic = !Pixel;
    const auto& size = conf["map_size"_];
    if (size.size() != 2) {
      throw std::invalid_argument("map_size must contain height and width");
    }
    params.height = size[0];
    params.width = size[1];
    params.levels = conf["num_levels"_];
    if (params.levels != (Classic ? 1 : 9)) {
      throw std::invalid_argument(
          "Craftax requires one Classic level or nine full-game levels");
    }
    params.max_timesteps = conf["max_episode_steps"_];
#define CRAFTAX_CONFIG(field) params.field = conf[#field##_]
    CRAFTAX_CONFIG(day_length);
    CRAFTAX_CONFIG(always_diamond);
    CRAFTAX_CONFIG(god_mode);
    CRAFTAX_CONFIG(mob_despawn_distance);
    CRAFTAX_CONFIG(max_attribute);
    CRAFTAX_CONFIG(max_melee_mobs);
    CRAFTAX_CONFIG(max_passive_mobs);
    CRAFTAX_CONFIG(max_ranged_mobs);
    CRAFTAX_CONFIG(max_mob_projectiles);
    CRAFTAX_CONFIG(max_player_projectiles);
    CRAFTAX_CONFIG(max_growing_plants);
    CRAFTAX_CONFIG(zombie_health);
    CRAFTAX_CONFIG(cow_health);
    CRAFTAX_CONFIG(skeleton_health);
    CRAFTAX_CONFIG(spawn_cow_chance);
    CRAFTAX_CONFIG(spawn_zombie_base_chance);
    CRAFTAX_CONFIG(spawn_zombie_night_chance);
    CRAFTAX_CONFIG(spawn_skeleton_chance);
#undef CRAFTAX_CONFIG
    const auto& angles = conf["fractal_noise_angles"_];
    if (angles.size() != 4) {
      throw std::invalid_argument(
          "fractal_noise_angles must contain four arrays");
    }
    std::copy(angles.begin(), angles.end(),
              params.fractal_noise_angles.begin());
    const int tile = conf["render_tile_size"_];
    if (tile != 16 && tile != 64) {
      throw std::invalid_argument("render_tile_size must be 16 or 64");
    }
    return params;
  }

  template <typename Config>
  static auto StateSpec(const Config& conf) {
    Game game(Parameters(conf));
    const auto& initial = conf["initial_state"_];
    if (!initial.empty()) {
      DecodeState(&game, initial);
    }
    const std::vector<int> shape =
        Pixel ? std::vector<int>{Classic ? 63 : 130, Classic ? 63 : 110, 3}
              : std::vector<int>{Classic ? 1345 : 8268};
    const int state_size =
        conf["debug_state"_] ? static_cast<int>(EncodeState(&game).size()) : 0;
    auto spec =
        ConcatDict(MakeDict("obs"_.Bind(Spec<float>(shape, {0, 1})),
                            "info:discount"_.Bind(Spec<float>({}, {0, 1})),
                            "info:state"_.Bind(Spec<double>({state_size}))),
                   AchievementInfo<Classic>::StateSpec());
    if constexpr (Classic) {
      return ConcatDict(
          spec, MakeDict("info:score"_.Bind(Spec<float>({}, {0, 100}))));
    } else {
      return spec;
    }
  }

  template <typename Config>
  static auto ActionSpec(const Config&) {
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, Classic ? 16 : 42})));
  }
};

template <bool Classic, bool Pixel>
class CraftaxEnv : public Env<EnvSpec<CraftaxEnvFns<Classic, Pixel>>>,
                   public RenderableEnv {
 public:
  using Spec = EnvSpec<CraftaxEnvFns<Classic, Pixel>>;
  using Base = Env<Spec>;
  using Action = typename Base::Action;

  CraftaxEnv(const Spec& spec, int env_id)
      : Base(spec, env_id),
        game_(CraftaxEnvFns<Classic, Pixel>::Parameters(spec.config)),
        rng_{0, static_cast<std::uint32_t>(this->seed_)},
        auto_reset_(spec.config["auto_reset"_]),
        tile_(spec.config["render_tile_size"_]) {}

  bool IsDone() override { return needs_reset_; }

  void Reset() override {
    game_.Reset(first_reset_ ? rng_ : TakeKey(&rng_));
    // The only state injection occurs at the first reset. Subsequent episodes
    // and every step use native dynamics without further synchronization.
    const bool synchronized =
        first_reset_ && !this->spec_.config["initial_state"_].empty();
    if (synchronized) {
      DecodeState(&game_, this->spec_.config["initial_state"_]);
    }
    first_reset_ = needs_reset_ = false;
    WriteState(0.0f, false, false, true, game_.state.timestep,
               game_.state.achievements, synchronized);
  }

  void Step(const Action& action) override {
    const Key key = TakeKey(&rng_);
    const float reward =
        game_.Step(auto_reset_ ? Split(key, 0) : key, action["action"_]);
    // In the pinned AutoReset graph, the returned state's select prevents
    // contraction of the light calculation; its observation uses the fused
    // result. Preserve both observable results without changing the stream.
    if (auto_reset_) {
      game_.state.light_level = game_.Light(game_.state.timestep, false);
    }
    const bool done = game_.Done();
    const bool terminal =
        game_.state.player_health <= 0 ||
        (Classic
             ? game_.Block(game_.state.player_position) == classic::block::LAVA
             : game_.state.boss_progress >= game_.params.levels - 1);
    const bool trunc =
        done && !terminal && game_.state.timestep >= game_.params.max_timesteps;
    const int elapsed = game_.state.timestep;
    const auto achievements = game_.state.achievements;
    if (done && auto_reset_) {
      game_.Reset(Split(key, 1));
    }
    needs_reset_ = done && !auto_reset_;
    WriteState(reward, done, trunc, false, elapsed, achievements);
    if (done && auto_reset_) {
      this->ResetStepCount();
    }
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : (Classic ? 9 : 11) * tile_,
            height > 0 ? height : (Classic ? 9 : 13) * tile_};
  }

  void Render(int width, int height, int camera_id,
              unsigned char* rgb) override {
    if (camera_id != -1 && camera_id != 0) {
      throw std::invalid_argument("Craftax has one camera");
    }
    const int source_width = (Classic ? 9 : 11) * tile_;
    const int source_height = (Classic ? 9 : 13) * tile_;
    const auto pixels = Pixels(game_, tile_);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int source = (y * source_height / height * source_width +
                            x * source_width / width) *
                           3;
        for (int c = 0; c < 3; ++c) {
          rgb[(y * width + x) * 3 + c] = static_cast<unsigned char>(
              std::clamp(pixels[source + c], 0.0f, 255.0f));
        }
      }
    }
  }

 private:
  Game game_;
  Key rng_;
  bool auto_reset_, first_reset_{true}, needs_reset_{true};
  int tile_;

  void WriteState(float reward, bool done, bool trunc, bool reset, int elapsed,
                  const std::vector<std::uint8_t>& achievements,
                  bool synchronized = false) {
    auto state = this->Allocate();
    const bool reset_obs = reset || (done && auto_reset_);
    const float saved_light = game_.state.light_level;
    if (auto_reset_ && !reset_obs) {
      game_.state.light_level = game_.Light(game_.state.timestep);
    }
    std::vector<float> obs;
    if constexpr (Pixel) {
      obs = Pixels(game_, Classic ? 7 : 10);
      for (auto& value : obs) {
        value *= 1.0f / 255.0f;
      }
    } else {
      obs = game_.Symbolic();
      if constexpr (!Classic) {
        if (reset_obs && !synchronized) {
          // Reset constants are folded as division, unlike the dynamic
          // reciprocal multiplication in step's symbolic observation.
          constexpr int k_intrinsic_offset = 9 * 11 * 83 + 22;
          for (int i = 0; i < 5; ++i) {
            obs[k_intrinsic_offset + i] = 0.9f;
          }
        }
      }
    }
    game_.state.light_level = saved_light;
    std::copy(obs.begin(), obs.end(),
              static_cast<float*>(state["obs"_].Data()));
    state["reward"_] = reward;
    state["done"_] = done;
    state["trunc"_] = trunc;
    state["discount"_] = static_cast<float>(!done);
    state["info:discount"_] = static_cast<float>(!done);
    if (reset) {
      state["step_type"_] = 0;
    } else {
      state["step_type"_] = done ? 2 : 1;
    }
    state["elapsed_step"_] = elapsed;
    AchievementInfo<Classic>::Write(state, achievements, done);
    if constexpr (Classic) {
      state["info:score"_] = Game::ClassicScore(achievements, done);
    }
    if (this->spec_.config["debug_state"_]) {
      const auto encoded = EncodeState(&game_);
      std::copy(encoded.begin(), encoded.end(),
                static_cast<double*>(state["info:state"_].Data()));
    }
  }
};

}  // namespace craftax

#endif  // ENVPOOL_CRAFTAX_CRAFTAX_H_
