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
#include <iterator>

#include "envpool/craftax/game.h"
#include "third_party/craftax/constants.h"

namespace craftax {
namespace b = full::block;
namespace a = full::action;
namespace c = full::achievement;

bool Game::FullSolid(Pos pos) const {
  return std::find(std::begin(full::SOLID_BLOCKS), std::end(full::SOLID_BLOCKS),
                   Block(pos)) != std::end(full::SOLID_BLOCKS);
}

bool Game::FullValid(Pos pos, const int* collision) const {
  if (!InBounds(pos) || InMob(pos) || FullSolid(pos)) {
    return false;
  }
  const int block = Block(pos);
  int kind = 0;
  if (block == b::WATER) {
    kind = 1;
  } else if (block == b::LAVA) {
    kind = 2;
  }
  return collision[kind] == 0;
}

bool Game::BossVulnerable() const {
  if (state.boss_timesteps_to_spawn_this_round > 0) {
    return false;
  }
  for (int i = 0; i < params.max_melee_mobs; ++i) {
    if (state.melee_mobs[state.player_level * params.max_melee_mobs + i].mask) {
      return false;
    }
  }
  for (int i = 0; i < params.max_ranged_mobs; ++i) {
    if (state.ranged_mobs[state.player_level * params.max_ranged_mobs + i]
            .mask) {
      return false;
    }
  }
  return true;
}

std::array<float, 3> Game::PlayerDamage() const {
  constexpr std::array<int, 5> damages{1, 2, 3, 5, 8};
  const float base =
      static_cast<float>(damages[std::clamp(state.inventory.sword, 0, 4)]);
  const float elemental =
      base * 0.5f * (1.0f + 0.05f * (state.player_intelligence - 1));
  return {base * (1.0f + 0.25f * (state.player_strength - 1)),
          state.sword_enchantment == 1 ? elemental : 0.0f,
          state.sword_enchantment == 2 ? elemental : 0.0f};
}

float Game::DamagePlayer(std::array<float, 3> damage) const {
  std::array<float, 3> defense{};
  for (int i = 0; i < 4; ++i) {
    defense[0] += state.inventory.armour[i] * 0.1f;
    defense[1] += static_cast<float>(state.armour_enchantments[i] == 1) * 0.2f;
    defense[2] += static_cast<float>(state.armour_enchantments[i] == 2) * 0.2f;
  }
  float total = 0;
  for (int i = 0; i < 3; ++i) {
    damage[i] *= state.player_level == params.levels - 1 ? 1.5f : 1.0f;
    total += (1.0f - defense[i]) * damage[i];
  }
  return total;
}

bool Game::Attack(Pos pos, const std::array<float, 3>& damage, bool can_eat) {
  const std::array<std::vector<Mob>*, 3> groups{
      &state.melee_mobs, &state.passive_mobs, &state.ranged_mobs};
  const std::array<int, 3> counts{
      params.max_melee_mobs, params.max_passive_mobs, params.max_ranged_mobs};
  const std::array<int, 3> classes{1, 0, 2};
  bool attacked = false;
  bool killed = false;
  bool killed_monster = false;
  for (int g = 0; g < 3; ++g) {
    auto& mobs = *groups[g];
    const int base = state.player_level * counts[g];
    int target = base;
    bool hit = false;
    for (int i = base; i < base + counts[g]; ++i) {
      if (mobs[i].mask && mobs[i].position == pos) {
        target = i;
        hit = true;
        break;
      }
    }
    auto& mob = mobs[target];
    const bool old_mask = mob.mask;
    float amount = 0;
    for (int i = 0; i < 3; ++i) {
      amount +=
          (1.0f - full::MOB_TYPE_DEFENSE_MAPPING[mob.type_id][classes[g]][i]) *
          damage[i];
    }
    mob.health -= hit ? amount : 0.0f;
    for (auto& m : mobs) {
      m.mask &= m.health > 0;
    }
    const bool kill = old_mask && !mob.mask;
    attacked |= hit;
    killed |= kill;
    killed_monster |= kill && g != 1;
    if (kill && (g != 1 || can_eat)) {
      state.achievements[full::MOB_ACHIEVEMENT_MAP[classes[g]][mob.type_id]] =
          1u;
    }
    if (g == 1 && kill && can_eat) {
      state.player_food =
          std::min(7 + 2 * state.player_dexterity, state.player_food + 6);
      state.player_hunger = 0;
    }
  }
  SetMob(pos, (state.mob_map[Index(pos)] != 0u) && !killed);
  state.monsters_killed[state.player_level] += static_cast<int>(killed_monster);
  return attacked;
}

bool Game::Projectile(Pos pos, Pos direction, int type, bool player) {
  auto& mobs = player ? state.player_projectiles : state.mob_projectiles;
  auto& directions = player ? state.player_projectile_directions
                            : state.mob_projectile_directions;
  const int count =
      player ? params.max_player_projectiles : params.max_mob_projectiles;
  const int base = state.player_level * count;
  for (int i = base; i < base + count; ++i) {
    if (!mobs[i].mask) {
      mobs[i].position = pos;
      mobs[i].mask = true;
      mobs[i].type_id = type;
      directions[i] = direction;
      return true;
    }
  }
  return false;
}

void Game::FullChangeFloor(int action) {
  const int item = state.item_map[Index(state.player_position)];
  if (action == a::DESCEND && state.player_level < params.levels - 1 &&
      (params.god_mode || (item == full::item::LADDER_DOWN &&
                           state.monsters_killed[state.player_level] >=
                               full::MONSTERS_KILLED_TO_CLEAR_LEVEL))) {
    ++state.player_level;
    state.player_position = state.up_ladders[state.player_level];
  } else if (action == a::ASCEND && state.player_level > 0 &&
             (params.god_mode || item == full::item::LADDER_UP)) {
    --state.player_level;
    state.player_position = state.down_ladders[state.player_level];
  }
  if (state.player_level > 0) {
    const int achievement = full::LEVEL_ACHIEVEMENT_MAP[state.player_level];
    state.player_xp += static_cast<int>(state.achievements[achievement] == 0u);
    state.achievements[achievement] = 1u;
  }
}

void Game::FullMove(int action) {
  const Pos delta = Direction(action);
  constexpr std::array<int, 3> collision{0, 1, 1};
  if (FullValid(state.player_position + delta, collision.data()) ||
      params.god_mode) {
    state.player_position = state.player_position + delta;
  }
  if (delta != Pos{}) {
    state.player_direction = action;
  }
}

void Game::FullIntrinsics(int action) {
  auto& s = state;
  const int max_health = 8 + s.player_strength;
  const int max_energy = 7 + 2 * s.player_dexterity;
  const bool not_boss = s.player_level != params.levels - 1;
  s.is_sleeping |= action == a::SLEEP && s.player_energy < max_energy;
  const bool wake = s.player_energy >= max_energy && s.is_sleeping;
  s.is_sleeping &= !wake;
  s.achievements[c::WAKE_UP] |= static_cast<int>(wake);
  s.is_resting |= action == a::REST && s.player_health < max_health;
  s.is_resting &=
      s.player_health < max_health && s.player_food > 0 && s.player_drink > 0;
  const float decay = 1.0f - 0.125f * (s.player_dexterity - 1);
  const float add = (s.is_sleeping ? 0.5f : 1.0f) * decay;
  s.player_hunger += add;
  if (s.player_hunger > 25) {
    s.player_food = std::max(0, s.player_food - static_cast<int>(not_boss));
    s.player_hunger = 0;
  }
  s.player_thirst += add;
  if (s.player_thirst > 20) {
    s.player_drink = std::max(0, s.player_drink - static_cast<int>(not_boss));
    s.player_thirst = 0;
  }
  s.player_fatigue = s.is_sleeping ? std::min(s.player_fatigue - 1, 0.0f)
                                   : s.player_fatigue + decay;
  if (s.player_fatigue > 30) {
    s.player_energy = std::max(0, s.player_energy - static_cast<int>(not_boss));
    s.player_fatigue = 0;
  }
  if (s.player_fatigue < -10) {
    s.player_energy = std::min(max_energy, s.player_energy + 1);
    s.player_fatigue = 0;
  }
  const bool necessities = s.player_food > 0 && s.player_drink > 0 &&
                           (s.player_energy > 0 || s.is_sleeping);
  const float recovering = s.is_sleeping ? 2.0f : 1.0f;
  const float starving =
      (s.is_sleeping ? -0.5f : -1.0f) * static_cast<float>(not_boss);
  s.player_recover += necessities ? recovering : starving;
  if (s.player_recover > 25) {
    s.player_health =
        std::min(static_cast<float>(max_health), s.player_health + 1);
    s.player_recover = 0;
  }
  if (s.player_recover < -15) {
    --s.player_health;
    s.player_recover = 0;
  }
  s.player_recover_mana =
      (s.player_recover_mana + (s.is_sleeping ? 2.0f : 1.0f)) *
      (1.0f + 0.25f * (s.player_intelligence - 1));
  if (s.player_recover_mana > 30) {
    ++s.player_mana;
    s.player_recover_mana = 0;
  }
}

void Game::FullAchievements() {
  const auto& inv = state.inventory;
  const std::array<int, 11> items{
      inv.wood,     inv.stone,   inv.coal, inv.iron,   inv.diamond, inv.ruby,
      inv.sapphire, inv.sapling, inv.bow,  inv.arrows, inv.torches};
  const std::array<int, 11> achievements{
      c::COLLECT_WOOD,     c::COLLECT_STONE,   c::COLLECT_COAL,
      c::COLLECT_IRON,     c::COLLECT_DIAMOND, c::COLLECT_RUBY,
      c::COLLECT_SAPPHIRE, c::COLLECT_SAPLING, c::FIND_BOW,
      c::MAKE_ARROW,       c::MAKE_TORCH};
  for (int i = 0; i < 11; ++i) {
    state.achievements[achievements[i]] |= static_cast<int>(items[i] > 0);
  }
  const std::array<int, 4> picks{c::MAKE_WOOD_PICKAXE, c::MAKE_STONE_PICKAXE,
                                 c::MAKE_IRON_PICKAXE, c::MAKE_DIAMOND_PICKAXE};
  const std::array<int, 4> swords{c::MAKE_WOOD_SWORD, c::MAKE_STONE_SWORD,
                                  c::MAKE_IRON_SWORD, c::MAKE_DIAMOND_SWORD};
  for (int i = 0; i < 4; ++i) {
    state.achievements[picks[i]] |= static_cast<int>(inv.pickaxe >= i + 1);
    state.achievements[swords[i]] |= static_cast<int>(inv.sword >= i + 1);
  }
}

float Game::FullStep(Key rng, int action) {
  const auto achievements = state.achievements;
  const float health = state.player_health;
  if (state.is_sleeping || state.is_resting) {
    action = a::NOOP;
  }
  FullChangeFloor(action);
  FullCraft(action);
  FullDo(TakeKey(&rng), action);
  FullPlace(action);
  const Key book = TakeKey(&rng);
  const Key enchant = TakeKey(&rng);
  FullUse(book, enchant, action);
  state.achievements[c::DEFEAT_NECROMANCER] |=
      static_cast<int>(state.boss_progress >= params.levels - 1);
  state.boss_timesteps_to_spawn_this_round -=
      static_cast<int>(state.player_level == params.levels - 1);
  int* attribute = nullptr;
  if (action == a::LEVEL_UP_DEXTERITY) {
    attribute = &state.player_dexterity;
  } else if (action == a::LEVEL_UP_STRENGTH) {
    attribute = &state.player_strength;
  } else if (action == a::LEVEL_UP_INTELLIGENCE) {
    attribute = &state.player_intelligence;
  }
  if ((attribute != nullptr) && state.player_xp >= 1 &&
      *attribute < params.max_attribute) {
    ++*attribute;
    --state.player_xp;
  }
  FullMove(action);
  FullMobs(TakeKey(&rng));
  FullSpawn(TakeKey(&rng));
  Plants();
  FullIntrinsics(action);
  auto& inv = state.inventory;
  for (int* item :
       {&inv.wood, &inv.stone, &inv.coal, &inv.iron, &inv.diamond, &inv.sapling,
        &inv.pickaxe, &inv.sword, &inv.bow, &inv.arrows, &inv.torches,
        &inv.ruby, &inv.sapphire, &inv.books}) {
    *item = std::min(99, *item);
  }
  for (auto& item : inv.armour) {
    item = std::min(99, item);
  }
  for (auto& item : inv.potions) {
    item = std::min(99, item);
  }
  state.player_health =
      std::clamp(state.player_health, params.god_mode ? 9.0f : 0.0f,
                 static_cast<float>(8 + state.player_strength));
  state.player_food =
      std::clamp(state.player_food, 0, 7 + 2 * state.player_dexterity);
  state.player_drink =
      std::clamp(state.player_drink, 0, 7 + 2 * state.player_dexterity);
  state.player_energy =
      std::clamp(state.player_energy, 0, 7 + 2 * state.player_dexterity);
  state.player_mana =
      std::clamp(state.player_mana, 0, 6 + 3 * state.player_intelligence);
  FullAchievements();
  int earned = 0;
  for (int i = 0; i < static_cast<int>(achievements.size()); ++i) {
    int coefficient = 5;
    if (i <= 24) {
      coefficient = 1;
    } else if (std::find(std::begin(full::INTERMEDIATE_ACHIEVEMENTS),
                         std::end(full::INTERMEDIATE_ACHIEVEMENTS),
                         i) != std::end(full::INTERMEDIATE_ACHIEVEMENTS)) {
      coefficient = 3;
    } else if (std::find(std::begin(full::VERY_ADVANCED_ACHIEVEMENTS),
                         std::end(full::VERY_ADVANCED_ACHIEVEMENTS),
                         i) != std::end(full::VERY_ADVANCED_ACHIEVEMENTS)) {
      coefficient = 8;
    }
    earned += (state.achievements[i] - achievements[i]) * coefficient;
  }
  const float reward =
      std::fma(state.player_health - health, 0.1f, static_cast<float>(earned));
  ++state.timestep;
  state.light_level = Light(state.timestep);
  state.state_rng = TakeKey(&rng);
  return reward;
}

}  // namespace craftax
