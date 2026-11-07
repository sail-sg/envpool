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

#ifndef ENVPOOL_CRAFTAX_GAME_H_
#define ENVPOOL_CRAFTAX_GAME_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "envpool/utils/random.h"

namespace craftax {

struct Pos {
  int y{0};
  int x{0};
  Pos operator+(Pos b) const { return {y + b.y, x + b.x}; }
  Pos operator-(Pos b) const { return {y - b.y, x - b.x}; }
  bool operator==(Pos b) const { return y == b.y && x == b.x; }
  bool operator!=(Pos b) const { return !(*this == b); }
};

inline Pos Direction(int action) {
  constexpr std::array<Pos, 5> dirs{{{0, 0}, {0, -1}, {0, 1}, {-1, 0}, {1, 0}}};
  return action >= 0 && action < 5 ? dirs[action] : Pos{};
}

inline int Distance(Pos a, Pos b) {
  return std::abs(a.y - b.y) + std::abs(a.x - b.x);
}

inline float EuclideanDistance(Pos a, Pos b) {
  const Pos delta = a - b;
  return std::sqrt(static_cast<float>(delta.y * delta.y + delta.x * delta.x));
}

inline float TorchLight(Pos offset) {
  return std::clamp(1.0f - EuclideanDistance(offset, {}) * 0.2f, 0.0f, 1.0f);
}

template <typename Predicate>
int SampleCell(envpool::Random& rng, int size, Predicate valid) {
  std::vector<int> cells;
  for (int i = 0; i < size; ++i) {
    if (valid(i)) {
      cells.push_back(i);
    }
  }
  return cells.empty() ? -1 : cells[rng.Integer(cells.size())];
}

struct Mob {
  Pos position;
  float health{0};
  bool mask{false};
  int attack_cooldown{0};
  int type_id{0};
};

struct Inventory {
  int wood{0}, stone{0}, coal{0}, iron{0}, diamond{0}, sapling{0};
  int wood_pickaxe{0}, stone_pickaxe{0}, iron_pickaxe{0};
  int wood_sword{0}, stone_sword{0}, iron_sword{0};
  int pickaxe{0}, sword{0}, bow{0}, arrows{0}, torches{0};
  int ruby{0}, sapphire{0}, books{0};
  std::array<int, 4> armour{};
  std::array<int, 6> potions{};
};

struct Params {
  bool classic{false};
  int height{48}, width{48}, levels{9};
  int max_timesteps{100000}, day_length{300};
  bool always_diamond{false}, god_mode{false};
  int mob_despawn_distance{14}, max_attribute{5};
  int max_melee_mobs{3}, max_passive_mobs{3}, max_ranged_mobs{2};
  int max_mob_projectiles{3}, max_player_projectiles{3};
  int max_growing_plants{10};
  int zombie_health{5}, cow_health{3}, skeleton_health{3};
  float spawn_cow_chance{0.1f}, spawn_zombie_base_chance{0.02f};
  float spawn_zombie_night_chance{0.1f}, spawn_skeleton_chance{0.05f};
  std::array<std::vector<float>, 4> fractal_noise_angles;

  explicit Params(bool is_classic = false) : classic(is_classic) {
    if (classic) {
      height = width = 64;
      levels = 1;
      max_timesteps = 10000;
      always_diamond = true;
    }
  }
};

struct State {
  std::vector<int> map, item_map;
  std::vector<std::uint8_t> mob_map;
  std::vector<float> light_map;
  std::vector<Pos> down_ladders, up_ladders;
  std::vector<std::uint8_t> chests_opened;
  std::vector<int> monsters_killed;
  Pos player_position;
  int player_level{0}, player_direction{2};
  float player_health{9};
  int player_food{9}, player_drink{9}, player_energy{9}, player_mana{9};
  bool is_sleeping{false}, is_resting{false};
  float player_recover{0}, player_hunger{0}, player_thirst{0},
      player_fatigue{0};
  float player_recover_mana{0};
  int player_xp{0}, player_dexterity{1}, player_strength{1},
      player_intelligence{1};
  Inventory inventory;
  std::vector<Mob> melee_mobs, passive_mobs, ranged_mobs;
  std::vector<Mob> mob_projectiles, player_projectiles;
  std::vector<Pos> mob_projectile_directions, player_projectile_directions;
  std::vector<Pos> growing_plants_positions;
  std::vector<int> growing_plants_age;
  std::vector<std::uint8_t> growing_plants_mask;
  std::array<int, 6> potion_mapping{};
  std::array<std::uint8_t, 2> learned_spells{};
  int sword_enchantment{0}, bow_enchantment{0};
  std::array<int, 4> armour_enchantments{};
  int boss_progress{0}, boss_timesteps_to_spawn_this_round{0};
  float light_level{1};
  std::vector<std::uint8_t> achievements;
  std::array<std::uint32_t, 2> render_seed{};
  int timestep{0};
};

class Game {
 public:
  Params params;
  State state;

  explicit Game(Params config, std::uint32_t seed = 0)
      : params(std::move(config)), rng_(seed) {
    Initialize();
  }
  void Initialize();
  void Reset();
  float Step(int action);
  bool Done() const;
  static float ClassicScore(const std::vector<std::uint8_t>& achievements,
                            bool done);
  std::vector<float> Symbolic() const;

  int Cells() const { return params.height * params.width; }
  bool InBounds(Pos p) const {
    return p.y >= 0 && p.y < params.height && p.x >= 0 && p.x < params.width;
  }
  // JAX gathers wrap negative indices and clip the remaining out-of-range
  // indices. Scatter writes wrap negatives but drop out-of-range writes.
  int Index(Pos p, int level = -1) const {
    if (p.y < 0) {
      p.y += params.height;
    }
    if (p.x < 0) {
      p.x += params.width;
    }
    return (level < 0 ? state.player_level : level) * Cells() +
           std::clamp(p.y, 0, params.height - 1) * params.width +
           std::clamp(p.x, 0, params.width - 1);
  }
  int Block(Pos p) const { return state.map[Index(p)]; }
  Pos ViewOrigin(int rows, int cols) const {
    // Match dynamic_slice on the official padded map, including god mode
    // positions outside the world. Entity coordinates still follow the player.
    const int pad = std::max(rows, cols) + 2;
    auto origin = [pad](int position, int size, int view) {
      int start = position - view / 2 + pad;
      if (start < 0) {
        start += size + 2 * pad;
      }
      return std::clamp(start, 0, size + 2 * pad - view) - pad;
    };
    return {origin(state.player_position.y, params.height, rows),
            origin(state.player_position.x, params.width, cols)};
  }
  void SetBlock(Pos p, int value, int level = -1) {
    if (p.y < 0) {
      p.y += params.height;
    }
    if (p.x < 0) {
      p.x += params.width;
    }
    if (InBounds(p)) {
      state.map[Index(p, level)] = value;
    }
  }
  void SetMob(Pos p, bool value) {
    if (p.y < 0) {
      p.y += params.height;
    }
    if (p.x < 0) {
      p.x += params.width;
    }
    if (InBounds(p)) {
      state.mob_map[Index(p)] = static_cast<std::uint8_t>(value);
    }
  }
  bool InMob(Pos p) const {
    return (state.mob_map[Index(p)] != 0u) || p == state.player_position;
  }
  bool Near(int block) const;
  bool ClassicSolid(Pos p) const;
  bool ClassicValid(Pos p) const;
  float Light(int timestep) const;
  void MoveMob(Mob* mob, Pos proposed, int cooldown);

  void ClassicWorld();
  void ClassicCraft(int action);
  void ClassicDo(int action);
  void ClassicPlace(int action);
  void ClassicMove(int action);
  void ClassicMobs();
  void ClassicSpawn();
  void ClassicIntrinsics(int action);
  void Plants();
  float ClassicStep(int action);

  void FullWorld();
  bool FullSolid(Pos pos) const;
  bool FullValid(Pos pos, const int* collision) const;
  bool BossVulnerable() const;
  std::array<float, 3> PlayerDamage() const;
  float DamagePlayer(std::array<float, 3> damage) const;
  bool Attack(Pos pos, const std::array<float, 3>& damage, bool can_eat);
  bool Projectile(Pos pos, Pos direction, int type, bool player);
  void FullCraft(int action);
  void FullDo(int action);
  void FullPlace(int action);
  void FullChangeFloor(int action);
  void FullUse(int action);
  void FullMove(int action);
  void FullMobs();
  void FullSpawn();
  void FullIntrinsics(int action);
  void FullAchievements();
  float FullStep(int action);
  std::vector<float> FullSymbolic() const;

 private:
  envpool::Random rng_;

  Pos TowardPlayer(Pos pos);
  void SampleRenderSeed() { state.render_seed = {rng_.Bits(), rng_.Bits()}; }
};

}  // namespace craftax

#endif  // ENVPOOL_CRAFTAX_GAME_H_
