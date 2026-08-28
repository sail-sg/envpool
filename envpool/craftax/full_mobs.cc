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

#include <algorithm>
#include <vector>

#include "envpool/craftax/game.h"
#include "third_party/craftax/constants.h"

namespace craftax {
namespace b = full::block;
namespace c = full::achievement;

void Game::FullMobs(Key rng) {
  const int level = state.player_level;
  const bool boss = level == params.levels - 1;
  auto toward = [&](Pos pos, Key key) {
    const Pos delta = state.player_position - pos;
    const int y = std::abs(delta.y);
    const int x = std::abs(delta.x);
    const auto total = static_cast<float>(y + x);
    const int axis =
        std::min(1, Choice(key, {static_cast<float>(y >= x) / total,
                                 static_cast<float>(x >= y) / total}));
    return axis == 0 ? Pos{(delta.y > 0) - (delta.y < 0), 0}
                     : Pos{0, (delta.x > 0) - (delta.x < 0)};
  };
  auto move_mob = [&](Mob* mob, Pos proposed, int kind, int cooldown) {
    const Pos position =
        FullValid(proposed,
                  full::MOB_TYPE_COLLISION_MAPPING[mob->type_id][kind])
            ? proposed
            : mob->position;
    const bool remains = Distance(mob->position, state.player_position) <
                             params.mob_despawn_distance ||
                         (kind != 0 && boss);
    SetMob(mob->position, state.mob_map[Index(mob->position)] && !mob->mask);
    mob->mask &= remains;
    SetMob(position, state.mob_map[Index(position)] || mob->mask);
    mob->position = position;
    mob->attack_cooldown = cooldown;
  };
  TakeKey(&rng);
  for (int i = 0; i < params.max_melee_mobs; ++i) {
    auto& mob = state.melee_mobs[level * params.max_melee_mobs + i];
    const Pos random = mob.position + Direction(RandInt(TakeKey(&rng), 1, 5));
    const Pos chase = mob.position + toward(mob.position, TakeKey(&rng));
    const int distance = Distance(mob.position, state.player_position);
    const float choose = Uniform(TakeKey(&rng));
    Pos proposed = (distance < 10 || boss) && choose < 0.75f ? chase : random;
    const bool attack = distance == 1 && mob.attack_cooldown <= 0 && mob.mask;
    if (attack) {
      proposed = mob.position;
      std::array<float, 3> damage;
      for (int c = 0; c < 3; ++c) {
        damage[c] = full::MOB_TYPE_DAMAGE_MAPPING[mob.type_id][1][c] *
                    (state.is_sleeping ? 3.5f : 1.0f);
      }
      state.player_health -= DamagePlayer(damage);
      state.achievements[c::WAKE_UP] |= static_cast<int>(state.is_sleeping);
      state.is_sleeping = state.is_resting = false;
    }
    rng = TakeKey(&rng);
    move_mob(&mob, proposed, 1, attack ? 5 : mob.attack_cooldown - 1);
  }
  TakeKey(&rng);
  for (int i = 0; i < params.max_passive_mobs; ++i) {
    auto& mob = state.passive_mobs[level * params.max_passive_mobs + i];
    const Pos proposed = mob.position + Direction(RandInt(TakeKey(&rng), 1, 9));
    move_mob(&mob, proposed, 0, mob.attack_cooldown);
  }
  TakeKey(&rng);
  for (int i = 0; i < params.max_ranged_mobs; ++i) {
    auto& mob = state.ranged_mobs[level * params.max_ranged_mobs + i];
    const Pos random = mob.position + Direction(RandInt(TakeKey(&rng), 1, 5));
    const Pos direction = toward(mob.position, TakeKey(&rng));
    const int distance = Distance(mob.position, state.player_position);
    Pos proposed = distance >= 6 ? mob.position + direction : random;
    if (distance <= 3) {
      proposed = mob.position - direction;
    }
    if (Uniform(TakeKey(&rng)) <= 0.85f) {
      proposed = random;
    }
    const bool attack =
        ((distance >= 4 && distance <= 5) ||
         (distance <= 3 &&
          !FullValid(proposed,
                     full::MOB_TYPE_COLLISION_MAPPING[mob.type_id][2]))) &&
        mob.attack_cooldown <= 0 && mob.mask;
    if (attack) {
      Projectile(mob.position, direction,
                 full::RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_MAPPING[mob.type_id],
                 false);
      proposed = mob.position;
    }
    move_mob(&mob, proposed, 2, attack ? 4 : mob.attack_cooldown - 1);
  }
  TakeKey(&rng);
  for (int i = 0; i < params.max_mob_projectiles; ++i) {
    const int index = level * params.max_mob_projectiles + i;
    auto& mob = state.mob_projectiles[index];
    const Pos proposed = mob.position + state.mob_projectile_directions[index];
    const int block = Block(proposed);
    const bool hit = mob.mask && (proposed == state.player_position ||
                                  mob.position == state.player_position);
    const bool destroy =
        (block == b::FURNACE || block == b::CRAFTING_TABLE) && mob.mask;
    mob.mask &= InBounds(proposed) &&
                (!FullSolid(proposed) || block == b::WATER) &&
                !InMob(proposed) && !hit;
    mob.position = proposed;
    if (hit) {
      std::array<float, 3> damage;
      for (int c = 0; c < 3; ++c) {
        damage[c] = full::MOB_TYPE_DAMAGE_MAPPING[mob.type_id][3][c];
      }
      state.player_health -= DamagePlayer(damage);
      state.is_sleeping = state.is_resting = false;
    }
    SetBlock(proposed, destroy ? b::PATH : block);
  }
  TakeKey(&rng);
  for (int i = 0; i < params.max_player_projectiles; ++i) {
    const int index = level * params.max_player_projectiles + i;
    auto& mob = state.player_projectiles[index];
    const bool arrow = mob.type_id == full::projectile::ARROW ||
                       mob.type_id == full::projectile::ARROW2;
    const bool magic = mob.type_id == full::projectile::FIREBALL ||
                       mob.type_id == full::projectile::ICEBALL;
    std::array<float, 3> damage;
    for (int c = 0; c < 3; ++c) {
      damage[c] = full::MOB_TYPE_DAMAGE_MAPPING[mob.type_id][3][c] *
                  static_cast<int>(mob.mask);
    }
    if (arrow && state.bow_enchantment > 0 && state.bow_enchantment < 3) {
      damage[state.bow_enchantment] += damage[0] / 2.0f;
    }
    for (auto& component : damage) {
      if (arrow) {
        component *= 1.0f + 0.2f * (state.player_dexterity - 1);
      }
      if (magic) {
        component *= 1.0f + 0.5f * (state.player_intelligence - 1);
      }
    }
    const Pos proposed =
        mob.position + state.player_projectile_directions[index];
    const bool wall = FullSolid(proposed) && Block(proposed) != b::WATER;
    const bool hit0 = Attack(mob.position, damage, false);
    if (hit0) {
      damage = {};
    }
    const bool hit1 = Attack(proposed, damage, false);
    mob.position = proposed;
    mob.mask &= InBounds(proposed) && !wall && !hit0 && !hit1;
  }
}

void Game::FullSpawn(Key rng) {
  const int level = state.player_level;
  const bool boss = level == params.levels - 1;
  int coefficient =
      1 + 2 * static_cast<int>(state.monsters_killed[level] <
                               full::MONSTERS_KILLED_TO_CLEAR_LEVEL);
  if (boss) {
    coefficient *= state.boss_timesteps_to_spawn_this_round >= 1 ? 1000 : 0;
  }
  const std::array<std::vector<Mob>*, 3> groups{
      &state.passive_mobs, &state.melee_mobs, &state.ranged_mobs};
  const std::array<int, 3> counts{
      params.max_passive_mobs, params.max_melee_mobs, params.max_ranged_mobs};
  for (int kind = 0; kind < 3; ++kind) {
    auto& mobs = *groups[kind];
    const int base = level * counts[kind];
    const int type_level =
        boss && kind > 0 ? std::clamp(state.boss_progress, 0, 8) : level;
    const int type = full::FLOOR_MOB_MAPPING[type_level][kind];
    float chance = full::FLOOR_MOB_SPAWN_CHANCE[level][kind];
    if (kind == 1) {
      const float night = 1.0f - state.light_level;
      chance += full::FLOOR_MOB_SPAWN_CHANCE[level][3] * (night * night);
    }
    if (kind != 0) {
      chance *= coefficient;
    }
    const float draw = Uniform(TakeKey(&rng));
    std::vector<float> weights(Cells());
    int valid_count = 0;
    for (int y = 0; y < params.height; ++y) {
      for (int x = 0; x < params.width; ++x) {
        const Pos pos{y, x};
        const int block = Block(pos);
        const float distance = EuclideanDistance(pos, state.player_position);
        bool terrain = block == b::GRASS || block == b::PATH ||
                       block == b::FIRE_GRASS || block == b::ICE_GRASS;
        if (kind == 2 && type == 5) {
          terrain = block == b::WATER;
        }
        if (kind > 0 && boss) {
          terrain =
              block == b::GRAVE || block == b::GRAVE2 || block == b::GRAVE3;
        }
        const bool hostile_range = boss ? distance <= 6 : distance > 9;
        const bool range = kind == 0 ? distance > 3 : hostile_range;
        const bool valid = terrain && range &&
                           distance < params.mob_despawn_distance &&
                           (state.mob_map[Index(pos)] == 0u);
        weights[y * params.width + x] = static_cast<float>(valid);
        valid_count += static_cast<int>(valid);
      }
    }
    for (auto& weight : weights) {
      weight /= static_cast<float>(valid_count);
    }
    const int position = Choice(TakeKey(&rng), weights);
    int slot = base;
    bool has_slot = false;
    for (int i = base; i < base + counts[kind]; ++i) {
      if (!mobs[i].mask) {
        slot = i;
        has_slot = true;
        break;
      }
    }
    auto& mob = mobs[slot];
    if (has_slot && valid_count > 0 && draw < chance && (!boss || kind != 0)) {
      mob.position = {position / params.height, position % params.width};
      mob.health = full::MOB_TYPE_HEALTH_MAPPING[type][kind];
      mob.mask = true;
    }
    // Upstream writes type_id even when this draw did not spawn a mob.
    mob.type_id = type;
    SetMob(mob.position,
           (state.mob_map[Index(mob.position)] != 0u) || mob.mask);
  }
}

}  // namespace craftax
