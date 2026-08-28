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
//
// Adapted from Craftax v1.6.1, copyright (c) 2024 Michael Matthews, MIT.

#include "envpool/craftax/game.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "third_party/craftax/constants.h"

namespace craftax {

void Game::Initialize() {
  if (params.height != params.width ||
      params.height < (params.classic ? 16 : 48) || params.height % 16 != 0 ||
      params.levels < 1 || params.levels > 9 || params.day_length < 1 ||
      params.max_timesteps < 1 || params.mob_despawn_distance < 1 ||
      params.max_melee_mobs < 1 || params.max_passive_mobs < 1 ||
      params.max_ranged_mobs < 1 || params.max_mob_projectiles < 1 ||
      params.max_player_projectiles < 1 || params.max_growing_plants < 1) {
    throw std::invalid_argument(
        "invalid Craftax map, level, day or entity capacity");
  }
  const std::array<int, 4> sizes{
      (params.height / 16 + 1) * (params.width / 16 + 1),
      (params.height / 16 + 1) * (params.width / 16 + 1),
      (params.height / 8 + 1) * (params.width / 2 + 1),
      (params.height / 4 + 1) * (params.width / 4 + 1)};
  for (int i = 0; i < 4; ++i) {
    const auto& angles = params.fractal_noise_angles[i];
    if (!angles.empty() &&
        angles.size() != static_cast<std::size_t>(sizes[i])) {
      throw std::invalid_argument(
          "fractal noise angle shape does not match map_size");
    }
  }
  state = State{};
  const int map_levels = params.classic ? 1 : 9;
  const int size = Cells() * map_levels;
  state.map.resize(size);
  state.mob_map.resize(size);
  state.item_map.resize(size);
  state.light_map.resize(size);
  state.down_ladders.resize(map_levels);
  state.up_ladders.resize(map_levels);
  state.chests_opened.resize(params.levels);
  state.monsters_killed.resize(params.levels);
  state.melee_mobs.resize(params.levels * params.max_melee_mobs);
  state.passive_mobs.resize(params.levels * params.max_passive_mobs);
  state.ranged_mobs.resize(params.levels * params.max_ranged_mobs);
  state.mob_projectiles.resize(params.levels * params.max_mob_projectiles);
  state.player_projectiles.resize(params.levels *
                                  params.max_player_projectiles);
  state.mob_projectile_directions.resize(state.mob_projectiles.size());
  state.player_projectile_directions.resize(state.player_projectiles.size());
  state.growing_plants_positions.resize(params.max_growing_plants);
  state.growing_plants_age.resize(params.max_growing_plants);
  state.growing_plants_mask.resize(params.max_growing_plants);
  state.achievements.resize(params.classic ? classic::achievement::COUNT
                                           : full::achievement::COUNT);
  state.player_direction = 3;
  state.light_level = Light(0);
}

bool Game::Near(int block) const {
  for (int y = -1; y <= 1; ++y) {
    for (int x = -1; x <= 1; ++x) {
      const Pos pos = state.player_position + Pos{y, x};
      if ((x != 0 || y != 0) && InBounds(pos) && Block(pos) == block) {
        return true;
      }
    }
  }
  return false;
}

float Game::Light(int timestep, bool fused) const {
  const float progress =
      std::fmod(static_cast<float>(timestep) / params.day_length, 1.0f) + 0.3f;
  const float cosine = std::abs(std::cos(3.141592653589793f * progress));
  // XLA fuses the final multiply/subtract in the pinned CPU oracle.
  return fused ? std::fma(-cosine * cosine, cosine, 1.0f)
               : 1.0f - (cosine * cosine) * cosine;
}

void Game::Reset(Key rng) {
  Initialize();
  if (params.classic) {
    ClassicWorld(rng);
  } else {
    FullWorld(params.symbolic ? Split(rng, 1) : rng);
  }
}

float Game::Step(Key rng, int action) {
  const int count =
      params.classic ? classic::action::COUNT : full::action::COUNT;
  if (action < 0 || action >= count) {
    throw std::invalid_argument("invalid Craftax action");
  }
  return params.classic ? ClassicStep(rng, action) : FullStep(rng, action);
}

bool Game::Done() const {
  if (state.timestep >= params.max_timesteps) {
    return true;
  }
  if (params.classic) {
    return state.player_health <= 0 ||
           Block(state.player_position) == classic::block::LAVA;
  }
  return state.player_health <= 0 || state.boss_progress >= params.levels - 1;
}

float Game::ClassicScore(const std::vector<std::uint8_t>& achievements,
                         bool done) {
  float sum = 0.0f;
  for (auto earned : achievements) {
    sum += (earned != 0u) && done ? std::log(101.0f) : 0.0f;
  }
  float x = sum * (1.0f / 22.0f);
  // The pinned scalar JAX reduction is ordered. Its exponential uses the
  // Cephes polynomial also used by LLVM's float32 math lowering, rather than
  // the platform libm expf. Only the score's finite 0..log(101) domain is
  // needed.
  const float n = std::floor(std::fma(x, 1.44269504088896341f, 0.5f));
  x = std::fma(-0.693359375f, n, x);
  x = std::fma(2.12194440e-4f, n, x);
  float z = std::fma(x, 1.9875691500e-4f, 1.3981999507e-3f);
  for (float coefficient : {8.3334519073e-3f, 4.1665795894e-2f,
                            1.6666665459e-1f, 5.0000001201e-1f}) {
    z = std::fma(z, x, coefficient);
  }
  z = std::fma(z, x * x, x);
  return std::ldexp(z + 1.0f, static_cast<int>(n)) - 1.0f;
}

// Round a finite float32 value to binary16 and back. Classic's inventory and
// intrinsics explicitly pass through float16 before joining its float32 obs.
static float Half(float value) {
  if (value == 0) {
    return value;
  }
  int exponent;
  std::frexp(value, &exponent);
  const float scale = std::ldexp(1.0f, std::max(-24, exponent - 11));
  return std::nearbyint(value / scale) * scale;
}

std::vector<float> Game::Symbolic() const {
  if (!params.classic) {
    return FullSymbolic();
  }
  constexpr int h = 7;
  constexpr int w = 9;
  constexpr int channels = 21;
  std::vector<float> obs(h * w * channels);
  const Pos top = ViewOrigin(h, w);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const Pos pos = top + Pos{y, x};
      const int block =
          InBounds(pos) ? Block(pos) : classic::block::OUT_OF_BOUNDS;
      if (block >= 0 && block < 17) {
        obs[(y * w + x) * channels + block] = 1;
      }
    }
  }
  const std::array<const std::vector<Mob>*, 4> groups{
      &state.melee_mobs, &state.passive_mobs, &state.ranged_mobs,
      &state.mob_projectiles};
  for (int group = 0; group < 4; ++group) {
    for (const auto& mob : *groups[group]) {
      Pos pos = mob.position - state.player_position + Pos{h / 2, w / 2};
      const bool on_screen = pos.y >= 0 && pos.y < h && pos.x >= 0 && pos.x < w;
      if (pos.y < 0) {
        pos.y += h;
      }
      if (pos.x < 0) {
        pos.x += w;
      }
      if (pos.y >= 0 && pos.y < h && pos.x >= 0 && pos.x < w) {
        obs[(pos.y * w + pos.x) * channels + 17 + group] =
            static_cast<float>(on_screen && mob.mask);
      }
    }
  }
  const auto& inv = state.inventory;
  for (int item :
       {inv.wood, inv.stone, inv.coal, inv.iron, inv.diamond, inv.sapling,
        inv.wood_pickaxe, inv.stone_pickaxe, inv.iron_pickaxe, inv.wood_sword,
        inv.stone_sword, inv.iron_sword}) {
    obs.push_back(Half(static_cast<float>(item) * Half(0.1f)));
  }
  for (float intrinsic :
       {state.player_health, static_cast<float>(state.player_food),
        static_cast<float>(state.player_drink),
        static_cast<float>(state.player_energy)}) {
    obs.push_back(Half(intrinsic * Half(0.1f)));
  }
  for (int i = 1; i <= 4; ++i) {
    obs.push_back(static_cast<float>(state.player_direction == i));
  }
  obs.push_back(state.light_level);
  obs.push_back(static_cast<float>(state.is_sleeping));
  return obs;
}

}  // namespace craftax
