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
// Adapted from Craftax v1.6.1, copyright (c) 2024 Michael Matthews.
// The upstream MIT notice is distributed in third_party/craftax/LICENSE.

#include <algorithm>
#include <numeric>
#include <vector>

#include "envpool/craftax/game.h"
#include "third_party/craftax/constants.h"

namespace craftax {
namespace b = classic::block;
namespace a = classic::action;
namespace c = classic::achievement;

bool Game::ClassicSolid(Pos p) const {
  const int block = Block(p);
  return block == b::WATER || block == b::STONE || block == b::TREE ||
         block == b::COAL || block == b::IRON || block == b::DIAMOND ||
         block == b::CRAFTING_TABLE || block == b::FURNACE ||
         block == b::PLANT || block == b::RIPE_PLANT;
}

bool Game::ClassicValid(Pos p) const {
  return InBounds(p) && !ClassicSolid(p) && !InMob(p) && Block(p) != b::LAVA;
}

void Game::ClassicCraft(int action) {
  if (!Near(b::CRAFTING_TABLE)) {
    return;
  }
  auto& inv = state.inventory;
  const bool furnace = Near(b::FURNACE);
  const std::array<int, 6> actions{a::MAKE_WOOD_PICKAXE, a::MAKE_STONE_PICKAXE,
                                   a::MAKE_IRON_PICKAXE, a::MAKE_WOOD_SWORD,
                                   a::MAKE_STONE_SWORD,  a::MAKE_IRON_SWORD};
  const std::array<int, 6> achievements{
      c::MAKE_WOOD_PICKAXE, c::MAKE_STONE_PICKAXE, c::MAKE_IRON_PICKAXE,
      c::MAKE_WOOD_SWORD,   c::MAKE_STONE_SWORD,   c::MAKE_IRON_SWORD};
  const std::array<int*, 6> tools{&inv.wood_pickaxe, &inv.stone_pickaxe,
                                  &inv.iron_pickaxe, &inv.wood_sword,
                                  &inv.stone_sword,  &inv.iron_sword};
  for (int i = 0; i < 6; ++i) {
    const int tier = i % 3;
    if (action != actions[i] || inv.wood < 1 || (tier > 0 && inv.stone < 1) ||
        (tier == 2 && (!furnace || inv.coal < 1 || inv.iron < 1))) {
      continue;
    }
    --inv.wood;
    inv.stone -= static_cast<int>(tier > 0);
    inv.coal -= static_cast<int>(tier == 2);
    inv.iron -= static_cast<int>(tier == 2);
    ++*tools[i];
    state.achievements[achievements[i]] = 1u;
  }
}

void Game::ClassicDo(Key rng, int action) {
  if (action != a::DO) {
    return;
  }
  const Pos pos = state.player_position + Direction(state.player_direction);
  auto& inv = state.inventory;
  const float damage = static_cast<float>(std::max(
      {1, 2 * inv.wood_sword, 3 * inv.stone_sword, 5 * inv.iron_sword}));
  bool attacked = false;
  bool killed = false;
  const std::array<std::vector<Mob>*, 3> groups{
      &state.melee_mobs, &state.passive_mobs, &state.ranged_mobs};
  const std::array<int, 3> achievements{c::DEFEAT_ZOMBIE, c::EAT_COW,
                                        c::DEFEAT_SKELETON};
  for (int group = 0; group < 3; ++group) {
    auto& mobs = *groups[group];
    int target = 0;
    bool hit = false;
    for (int i = 0; i < static_cast<int>(mobs.size()); ++i) {
      if (mobs[i].mask && mobs[i].position == pos) {
        target = i;
        hit = true;
        break;
      }
    }
    const bool old_mask = mobs[target].mask;
    mobs[target].health -= hit ? damage : 0.0f;
    // Preserve the pinned upstream health-to-mask update, including inactive
    // slots. Using `mask && health > 0` here changes official behavior.
    for (auto& mob : mobs) {
      mob.mask = mob.health > 0;
    }
    const bool kill = old_mask && !mobs[target].mask;
    attacked |= hit;
    killed |= kill;
    state.achievements[achievements[group]] |= static_cast<int>(kill);
    if (group == 1 && kill) {
      state.player_food = std::min(9, state.player_food + 6);
      state.player_hunger = 0;
    }
  }
  SetMob(pos, (state.mob_map[Index(pos)] != 0u) && !killed);
  if (!InBounds(pos) || attacked) {
    return;
  }
  const int block = Block(pos);
  const std::array<int, 5> blocks{b::TREE, b::STONE, b::COAL, b::IRON,
                                  b::DIAMOND};
  const std::array<bool, 5> allowed{
      true, inv.wood_pickaxe != 0, inv.wood_pickaxe != 0,
      inv.stone_pickaxe != 0, inv.iron_pickaxe != 0};
  const std::array<int*, 5> resources{&inv.wood, &inv.stone, &inv.coal,
                                      &inv.iron, &inv.diamond};
  const std::array<int, 5> mining_achievements{
      c::COLLECT_WOOD, c::COLLECT_STONE, c::COLLECT_COAL, c::COLLECT_IRON,
      c::COLLECT_DIAMOND};
  for (int i = 0; i < 5; ++i) {
    if (block == blocks[i] && allowed[i]) {
      SetBlock(pos, i == 0 ? b::GRASS : b::PATH);
      ++*resources[i];
      state.achievements[mining_achievements[i]] = 1u;
    }
  }
  if (block == b::GRASS && Uniform(TakeKey(&rng)) < 0.1f) {
    ++inv.sapling;
    state.achievements[c::COLLECT_SAPLING] = 1u;
  }
  if (block == b::WATER) {
    state.player_drink = std::min(9, state.player_drink + 1);
    state.player_thirst = 0;
    state.achievements[c::COLLECT_DRINK] = 1u;
  }
  if (block == b::RIPE_PLANT) {
    SetBlock(pos, b::PLANT);
    state.player_food = std::min(9, state.player_food + 4);
    state.player_hunger = 0;
    state.achievements[c::EAT_PLANT] = 1u;
    int index = 0;
    for (int i = 0; i < params.max_growing_plants; ++i) {
      if (state.growing_plants_positions[i] == pos) {
        index = i;
        break;
      }
    }
    state.growing_plants_age[index] = 0;
  }
}

void Game::ClassicPlace(int action) {
  const Pos pos = state.player_position + Direction(state.player_direction);
  if (!InBounds(pos) || InMob(pos)) {
    return;
  }
  auto& inv = state.inventory;
  if (action == a::PLACE_TABLE && !ClassicSolid(pos) && inv.wood >= 2) {
    SetBlock(pos, b::CRAFTING_TABLE);
    inv.wood -= 2;
    state.achievements[c::PLACE_TABLE] = 1u;
  } else if (action == a::PLACE_FURNACE && !ClassicSolid(pos) &&
             inv.stone > 0) {
    SetBlock(pos, b::FURNACE);
    --inv.stone;
    state.achievements[c::PLACE_FURNACE] = 1u;
  } else if (action == a::PLACE_STONE &&
             (Block(pos) == b::WATER || !ClassicSolid(pos)) && inv.stone > 0) {
    SetBlock(pos, b::STONE);
    --inv.stone;
    state.achievements[c::PLACE_STONE] = 1u;
  } else if (action == a::PLACE_PLANT && Block(pos) == b::GRASS &&
             inv.sapling > 0) {
    SetBlock(pos, b::PLANT);
    --inv.sapling;
    state.achievements[c::PLACE_PLANT] = 1u;
    for (int i = 0; i < params.max_growing_plants; ++i) {
      if (state.growing_plants_mask[i] == 0u) {
        state.growing_plants_positions[i] = pos;
        state.growing_plants_age[i] = 0;
        state.growing_plants_mask[i] = 1u;
        break;
      }
    }
  }
}

void Game::ClassicMove(int action) {
  const Pos delta = Direction(action);
  const Pos proposed = state.player_position + delta;
  if (ClassicValid(proposed) || Block(proposed) == b::LAVA) {
    state.player_position = proposed;
  }
  if (delta != Pos{}) {
    state.player_direction = action;
  }
}

void Game::MoveMob(Mob* mob, Pos proposed, int cooldown) {
  const Pos position = ClassicValid(proposed) ? proposed : mob->position;
  const bool remains = Distance(mob->position, state.player_position) <
                       params.mob_despawn_distance;
  SetMob(mob->position,
         (state.mob_map[Index(mob->position)] != 0u) && !mob->mask);
  mob->mask = mob->mask && remains;
  SetMob(position, (state.mob_map[Index(position)] != 0u) || mob->mask);
  mob->position = position;
  mob->attack_cooldown = cooldown;
}

void Game::ClassicMobs(Key rng) {
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
  TakeKey(&rng);
  for (auto& mob : state.melee_mobs) {
    const Pos random = mob.position + Direction(RandInt(TakeKey(&rng), 1, 5));
    const Pos chase = mob.position + toward(mob.position, TakeKey(&rng));
    const int distance = Distance(mob.position, state.player_position);
    const float chase_draw = Uniform(TakeKey(&rng));
    const bool close = distance < 10 && chase_draw < 0.75f;
    // Every split is evaluated by the oracle even when the condition is false.
    // Do not short-circuit key advancement with a C++ boolean expression.
    Pos proposed = close ? chase : random;
    const bool attack = distance == 1 && mob.attack_cooldown <= 0 && mob.mask;
    if (attack) {
      proposed = mob.position;
      state.player_health -= state.is_sleeping ? 7 : 2;
      state.achievements[c::WAKE_UP] |= static_cast<int>(state.is_sleeping);
      state.is_sleeping = false;
    }
    const int cooldown = attack ? 5 : mob.attack_cooldown - 1;
    rng = TakeKey(&rng);
    MoveMob(&mob, proposed, cooldown);
  }
  TakeKey(&rng);
  for (auto& mob : state.passive_mobs) {
    const Pos proposed = mob.position + Direction(RandInt(TakeKey(&rng), 1, 9));
    MoveMob(&mob, proposed, mob.attack_cooldown);
  }
  TakeKey(&rng);
  for (auto& mob : state.ranged_mobs) {
    const Pos random = mob.position + Direction(RandInt(TakeKey(&rng), 1, 5));
    const Pos direction = toward(mob.position, TakeKey(&rng));
    const int distance = Distance(mob.position, state.player_position);
    Pos proposed = distance >= 10 ? mob.position + direction : random;
    if (distance <= 3) {
      proposed = mob.position - direction;
    }
    if (Uniform(TakeKey(&rng)) <= 0.85f) {
      proposed = random;
    }
    const bool attack = ((distance >= 4 && distance <= 5) ||
                         (distance <= 3 && !ClassicValid(proposed))) &&
                        mob.attack_cooldown <= 0 && mob.mask;
    if (attack) {
      for (int i = 0; i < params.max_mob_projectiles; ++i) {
        if (!state.mob_projectiles[i].mask) {
          state.mob_projectiles[i].position = mob.position;
          state.mob_projectiles[i].mask = true;
          state.mob_projectile_directions[i] = direction;
          break;
        }
      }
      proposed = mob.position;
    }
    MoveMob(&mob, proposed, attack ? 4 : mob.attack_cooldown - 1);
  }
  TakeKey(&rng);
  for (int i = 0; i < params.max_mob_projectiles; ++i) {
    auto& mob = state.mob_projectiles[i];
    const Pos proposed = mob.position + state.mob_projectile_directions[i];
    const int block = Block(proposed);
    const bool hit = proposed == state.player_position && mob.mask;
    const bool destroy =
        (block == b::FURNACE || block == b::CRAFTING_TABLE) && mob.mask;
    mob.mask = InBounds(proposed) &&
               (!ClassicSolid(proposed) || block == b::WATER) &&
               !InMob(proposed) && mob.mask;
    mob.position = proposed;
    if (hit) {
      state.player_health -= 2;
      state.is_sleeping = false;
    }
    SetBlock(proposed, destroy ? b::PATH : block);
  }
}

void Game::ClassicSpawn(Key rng) {
  auto spawn = [&](std::vector<Mob>* mobs, float chance, int kind, int health) {
    const float draw = Uniform(TakeKey(&rng));
    std::vector<float> weights(Cells());
    int count = 0;
    for (int y = 0; y < params.height; ++y) {
      for (int x = 0; x < params.width; ++x) {
        const Pos pos{y, x};
        const int distance = Distance(pos, state.player_position);
        const int block = Block(pos);
        const bool terrain =
            (kind < 2 && block == b::GRASS) || (kind > 0 && block == b::PATH);
        const bool valid = terrain && distance > (kind == 0 ? 3 : 9) &&
                           distance < params.mob_despawn_distance &&
                           !state.mob_map[Index(pos)];
        weights[y * params.width + x] = valid;
        count += valid;
      }
    }
    for (auto& w : weights) {
      w /= static_cast<float>(count);
    }
    const int position = Choice(TakeKey(&rng), weights);
    int slot = 0;
    bool has_slot = false;
    for (int i = 0; i < static_cast<int>(mobs->size()); ++i) {
      if (!(*mobs)[i].mask) {
        slot = i;
        has_slot = true;
        break;
      }
    }
    auto& mob = (*mobs)[slot];
    if (count > 0 && draw < chance && has_slot) {
      mob.position = {position / params.height, position % params.width};
      mob.health = static_cast<float>(health);
      mob.mask = true;
    }
    SetMob(mob.position, state.mob_map[Index(mob.position)] || mob.mask);
  };
  spawn(&state.passive_mobs, params.spawn_cow_chance, 0, params.cow_health);
  const float night = 1.0f - state.light_level;
  spawn(&state.melee_mobs,
        params.spawn_zombie_base_chance +
            params.spawn_zombie_night_chance * night * night,
        1, params.zombie_health);
  spawn(&state.ranged_mobs, params.spawn_skeleton_chance, 2,
        params.skeleton_health);
}

void Game::ClassicIntrinsics(int action) {
  auto& s = state;
  s.is_sleeping |= action == a::SLEEP && s.player_energy < 9;
  const bool wake = s.player_energy >= 9 && s.is_sleeping;
  s.is_sleeping &= !wake;
  s.achievements[c::WAKE_UP] |= static_cast<int>(wake);
  const float add = s.is_sleeping ? 0.5f : 1.0f;
  s.player_hunger += add;
  if (s.player_hunger > 25) {
    s.player_food = std::max(0, s.player_food - 1);
    s.player_hunger = 0;
  }
  s.player_thirst += add;
  if (s.player_thirst > 20) {
    s.player_drink = std::max(0, s.player_drink - 1);
    s.player_thirst = 0;
  }
  s.player_fatigue = s.is_sleeping ? std::min(s.player_fatigue - 1, 0.0f)
                                   : s.player_fatigue + 1;
  if (s.player_fatigue > 30) {
    s.player_energy = std::max(0, s.player_energy - 1);
    s.player_fatigue = 0;
  }
  if (s.player_fatigue < -10) {
    s.player_energy = std::min(9, s.player_energy + 1);
    s.player_fatigue = 0;
  }
  const bool necessities = s.player_food > 0 && s.player_drink > 0 &&
                           (s.player_energy > 0 || s.is_sleeping);
  const float recovering = s.is_sleeping ? 2.0f : 1.0f;
  const float starving = s.is_sleeping ? -0.5f : -1.0f;
  s.player_recover += necessities ? recovering : starving;
  if (s.player_recover > 25) {
    s.player_health = std::min(9.0f, s.player_health + 1);
    s.player_recover = 0;
  }
  if (s.player_recover < -15) {
    --s.player_health;
    s.player_recover = 0;
  }
}

void Game::Plants() {
  for (int i = 0; i < params.max_growing_plants; ++i) {
    auto& age = state.growing_plants_age[i];
    age = (age + 1) * state.growing_plants_mask[i];
    if (age >= 600) {
      SetBlock(state.growing_plants_positions[i], b::RIPE_PLANT, 0);
    }
  }
}

float Game::ClassicStep(Key rng, int action) {
  const auto achievements = state.achievements;
  const float health = state.player_health;
  if (state.is_sleeping) {
    action = a::NOOP;
  }
  ClassicCraft(action);
  ClassicDo(TakeKey(&rng), action);
  ClassicPlace(action);
  ClassicMove(action);
  ClassicMobs(TakeKey(&rng));
  ClassicSpawn(TakeKey(&rng));
  Plants();
  ClassicIntrinsics(action);
  auto& inv = state.inventory;
  for (int* item :
       {&inv.wood, &inv.stone, &inv.coal, &inv.iron, &inv.diamond, &inv.sapling,
        &inv.wood_pickaxe, &inv.stone_pickaxe, &inv.iron_pickaxe,
        &inv.wood_sword, &inv.stone_sword, &inv.iron_sword}) {
    *item = std::min(9, *item);
  }
  if (Block(state.player_position) == b::LAVA) {
    state.player_health = 0;
  }
  state.player_health = std::max(0.0f, state.player_health);
  int earned = 0;
  for (std::size_t i = 0; i < achievements.size(); ++i) {
    earned += state.achievements[i] - achievements[i];
  }
  const float reward =
      std::fma(state.player_health - health, 0.1f, static_cast<float>(earned));
  ++state.timestep;
  state.light_level = Light(state.timestep);
  state.state_rng = TakeKey(&rng);
  return reward;
}

}  // namespace craftax
