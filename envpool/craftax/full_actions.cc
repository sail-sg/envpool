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
#include <numeric>
#include <vector>

#include "envpool/craftax/game.h"
#include "third_party/craftax/constants.h"

namespace craftax {
namespace b = full::block;
namespace a = full::action;
namespace c = full::achievement;

void Game::FullCraft(int action) {
  if (!Near(b::CRAFTING_TABLE)) {
    return;
  }
  auto& inv = state.inventory;
  const bool furnace = Near(b::FURNACE);
  const std::array<int, 8> actions{
      a::MAKE_WOOD_PICKAXE,    a::MAKE_STONE_PICKAXE, a::MAKE_IRON_PICKAXE,
      a::MAKE_DIAMOND_PICKAXE, a::MAKE_WOOD_SWORD,    a::MAKE_STONE_SWORD,
      a::MAKE_IRON_SWORD,      a::MAKE_DIAMOND_SWORD};
  for (int i = 0; i < 8; ++i) {
    const int tier = i % 4 + 1;
    int& tool = i < 4 ? inv.pickaxe : inv.sword;
    const int gems = i < 4 ? 3 : 2;
    if (action != actions[i] || tool >= tier || inv.wood < 1 ||
        ((tier == 2 || tier == 3) && inv.stone < 1) ||
        (tier == 3 && (!furnace || inv.coal < 1 || inv.iron < 1)) ||
        (tier == 4 && inv.diamond < gems)) {
      continue;
    }
    --inv.wood;
    inv.stone -= static_cast<int>(tier == 2 || tier == 3);
    inv.iron -= static_cast<int>(tier == 3);
    inv.coal -= static_cast<int>(tier == 3);
    inv.diamond -= tier == 4 ? gems : 0;
    tool = tier;
  }
  if (action == a::MAKE_IRON_ARMOUR && furnace && inv.iron >= 3 &&
      inv.coal >= 3) {
    for (auto& armour : inv.armour) {
      if (armour < 1) {
        armour = 1;
        inv.iron -= 3;
        inv.coal -= 3;
        state.achievements[c::MAKE_IRON_ARMOUR] = 1u;
        break;
      }
    }
  }
  if (action == a::MAKE_DIAMOND_ARMOUR && inv.diamond >= 3) {
    for (auto& armour : inv.armour) {
      if (armour < 2) {
        armour = 2;
        inv.diamond -= 3;
        state.achievements[c::MAKE_DIAMOND_ARMOUR] = 1u;
        break;
      }
    }
  }
  if (action == a::MAKE_ARROW && inv.wood >= 1 && inv.stone >= 1 &&
      inv.arrows < 99) {
    --inv.wood;
    --inv.stone;
    inv.arrows += 2;
  }
  if (action == a::MAKE_TORCH && inv.wood >= 1 && inv.coal >= 1 &&
      inv.torches < 99) {
    --inv.wood;
    --inv.coal;
    inv.torches += 4;
  }
}

static void Loot(Key rng, int level, bool opened, Inventory* inv) {
  const bool torch = Uniform(TakeKey(&rng)) < 0.6f;
  const int torches = RandInt(TakeKey(&rng), 4, 8);
  const bool ore = Uniform(TakeKey(&rng)) < 0.6f;
  const int ore_id = Choice(TakeKey(&rng), {0.3f, 0.3f, 0.15f, 0.125f, 0.125f});
  const Key amount = TakeKey(&rng);
  const bool potion = Uniform(TakeKey(&rng)) < 0.5f;
  const int potion_id = RandInt(TakeKey(&rng), 0, 6);
  const int potions = RandInt(TakeKey(&rng), 1, 3);
  const bool arrow = Uniform(TakeKey(&rng)) < 0.25f;
  const int arrows = RandInt(TakeKey(&rng), 1, 5);
  const bool tool = Uniform(TakeKey(&rng)) < 0.2f;
  const int tool_id = RandInt(TakeKey(&rng), 0, 2);
  const int pickaxe = Choice(TakeKey(&rng), {0.4f, 0.3f, 0.2f, 0.1f}) + 1;
  const int sword = Choice(TakeKey(&rng), {0.4f, 0.3f, 0.2f, 0.1f}) + 1;
  inv->torches += static_cast<int>(torch) * torches;
  if (ore) {
    const std::array<int*, 5> ores{&inv->coal, &inv->iron, &inv->diamond,
                                   &inv->sapphire, &inv->ruby};
    constexpr std::array<int, 5> limits{4, 3, 2, 2, 2};
    const int index = std::clamp(ore_id, 0, 4);
    *ores[index] += RandInt(amount, 1, limits[index]);
  }
  inv->potions[potion_id] += static_cast<int>(potion) * potions;
  inv->arrows += static_cast<int>(arrow) * arrows;
  if (tool && tool_id == 0) {
    inv->pickaxe = std::max(inv->pickaxe, pickaxe);
  }
  if (tool && tool_id == 1) {
    inv->sword = std::max(inv->sword, sword);
  }
  if (!opened && level == 1) {
    inv->bow = 1;
  }
  if (!opened && (level == 3 || level == 4)) {
    ++inv->books;
  }
}

void Game::FullDo(Key rng, int action) {
  if (action != a::DO) {
    return;
  }
  const Pos pos = state.player_position + Direction(state.player_direction);
  const bool attacked = Attack(pos, PlayerDamage(), true);
  const int block = Block(pos);
  const bool chest = block == b::CHEST;
  const bool opened = state.chests_opened[state.player_level] != 0u;
  state.chests_opened[state.player_level] |= static_cast<int>(chest);
  const bool boss = block == b::NECROMANCER && BossVulnerable() &&
                    state.player_level == params.levels - 1;
  if (boss) {
    ++state.boss_progress;
    state.boss_timesteps_to_spawn_this_round = full::BOSS_FIGHT_SPAWN_TURNS;
  }
  if (!InBounds(pos) || attacked) {
    return;
  }
  auto& inv = state.inventory;
  if (block == b::TREE || block == b::FIRE_TREE || block == b::ICE_SHRUB) {
    ++inv.wood;
    int ground = b::GRASS;
    if (block == b::FIRE_TREE) {
      ground = b::FIRE_GRASS;
    } else if (block == b::ICE_SHRUB) {
      ground = b::ICE_GRASS;
    }
    SetBlock(pos, ground);
  }
  const std::array<int, 7> blocks{b::STONE,     b::COAL,     b::IRON,
                                  b::DIAMOND,   b::SAPPHIRE, b::RUBY,
                                  b::STALAGMITE};
  const std::array<int, 7> tier{1, 1, 2, 3, 4, 4, 1};
  const std::array<int*, 7> resources{&inv.stone,   &inv.coal,     &inv.iron,
                                      &inv.diamond, &inv.sapphire, &inv.ruby,
                                      &inv.stone};
  for (int i = 0; i < 7; ++i) {
    if (block == blocks[i] && inv.pickaxe >= tier[i]) {
      ++*resources[i];
      SetBlock(pos, b::PATH);
    }
  }
  if (block == b::FURNACE || block == b::CRAFTING_TABLE) {
    SetBlock(pos, b::PATH);
  }
  const float sapling = Uniform(TakeKey(&rng));
  if (block == b::GRASS && sapling < 0.1f) {
    ++inv.sapling;
  }
  if (block == b::WATER || block == b::FOUNTAIN) {
    state.player_drink =
        std::min(7 + 2 * state.player_dexterity, state.player_drink + 1);
    state.player_thirst = 0;
    state.achievements[c::COLLECT_DRINK] = 1u;
  }
  if (block == b::RIPE_PLANT) {
    SetBlock(pos, b::PLANT);
    state.player_food =
        std::min(7 + 2 * state.player_dexterity, state.player_food + 4);
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
  const Key loot = TakeKey(&rng);
  if (chest) {
    SetBlock(pos, b::PATH);
    Loot(loot, state.player_level, opened, &inv);
    state.achievements[c::OPEN_CHEST] = 1u;
  }
  state.achievements[c::DAMAGE_NECROMANCER] |= static_cast<int>(boss);
}

void Game::FullPlace(int action) {
  const Pos pos = state.player_position + Direction(state.player_direction);
  if (!InBounds(pos) || InMob(pos)) {
    return;
  }
  auto& inv = state.inventory;
  const int index = Index(pos);
  const int block = Block(pos);
  const bool occupied =
      FullSolid(pos) || state.item_map[index] != full::item::NONE;
  if (action == a::PLACE_TABLE && !occupied && inv.wood >= 2) {
    SetBlock(pos, b::CRAFTING_TABLE);
    inv.wood -= 2;
    state.achievements[c::PLACE_TABLE] = 1u;
  } else if (action == a::PLACE_FURNACE && !occupied && inv.stone > 0) {
    SetBlock(pos, b::FURNACE);
    --inv.stone;
    state.achievements[c::PLACE_FURNACE] = 1u;
  } else if (action == a::PLACE_STONE && (block == b::WATER || !occupied) &&
             inv.stone > 0) {
    SetBlock(pos, b::STONE);
    --inv.stone;
    state.achievements[c::PLACE_STONE] = 1u;
  } else if (action == a::PLACE_PLANT && block == b::GRASS && inv.sapling > 0 &&
             state.item_map[index] == full::item::NONE) {
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
  } else if (action == a::PLACE_TORCH && inv.torches > 0 &&
             state.item_map[index] == full::item::NONE &&
             std::find(std::begin(full::CAN_PLACE_ITEM_BLOCKS),
                       std::end(full::CAN_PLACE_ITEM_BLOCKS),
                       block) != std::end(full::CAN_PLACE_ITEM_BLOCKS)) {
    --inv.torches;
    state.item_map[index] = full::item::TORCH;
    state.achievements[c::PLACE_TORCH] = 1u;
    for (int y = -4; y <= 4; ++y) {
      for (int x = -4; x <= 4; ++x) {
        const Pos p = pos + Pos{y, x};
        if (!InBounds(p)) {
          continue;
        }
        const float light = TorchLight({y, x});
        state.light_map[Index(p)] =
            std::clamp(state.light_map[Index(p)] + light, 0.0f, 1.0f);
      }
    }
  }
}

void Game::FullUse(Key book_rng, Key enchant_rng, int action) {
  auto& s = state;
  auto& inv = s.inventory;
  if (action == a::SHOOT_ARROW && inv.bow >= 1 && inv.arrows >= 1 &&
      Projectile(s.player_position, Direction(s.player_direction),
                 full::projectile::ARROW2, true)) {
    --inv.arrows;
    s.achievements[c::FIRE_BOW] = 1u;
  }
  int spell = -1;
  if (action == a::CAST_FIREBALL) {
    spell = 0;
  } else if (action == a::CAST_ICEBALL) {
    spell = 1;
  }
  if (spell >= 0 && s.player_mana >= 2 && (s.learned_spells[spell] != 0u) &&
      Projectile(
          s.player_position, Direction(s.player_direction),
          spell == 0 ? full::projectile::FIREBALL : full::projectile::ICEBALL,
          true)) {
    s.player_mana -= 2;
    s.achievements[spell == 0 ? c::CAST_FIREBALL : c::CAST_ICEBALL] = 1u;
  }
  if (action >= a::DRINK_POTION_RED && action <= a::DRINK_POTION_YELLOW) {
    const int potion = action - a::DRINK_POTION_RED;
    if (inv.potions[potion] > 0) {
      --inv.potions[potion];
      const int effect = s.potion_mapping[potion];
      if (effect < 2) {
        s.player_health += effect == 0 ? 8 : -3;
      } else if (effect < 4) {
        s.player_mana += effect == 2 ? 8 : -3;
      } else {
        s.player_energy += effect == 4 ? 8 : -3;
      }
      s.achievements[c::DRINK_POTION] = 1u;
    }
  }
  if (action == a::READ_BOOK && inv.books > 0) {
    const auto count =
        static_cast<float>(2 - s.learned_spells[0] - s.learned_spells[1]);
    const int learn = std::clamp(
        Choice(TakeKey(&book_rng), {(1 - s.learned_spells[0]) / count,
                                    (1 - s.learned_spells[1]) / count}),
        0, 1);
    --inv.books;
    s.learned_spells[learn] = 1u;
    s.achievements[learn == 0 ? c::LEARN_FIREBALL : c::LEARN_ICEBALL] = 1u;
  }
  const int block = Block(s.player_position + Direction(s.player_direction));
  const int type = block == b::ENCHANTMENT_TABLE_FIRE ? 1 : 2;
  int& gems = type == 1 ? inv.ruby : inv.sapphire;
  if ((block != b::ENCHANTMENT_TABLE_FIRE &&
       block != b::ENCHANTMENT_TABLE_ICE) ||
      s.player_mana < 9 || gems < 1) {
    return;
  }
  bool enchanted = false;
  if (action == a::ENCHANT_SWORD && inv.sword > 0) {
    s.sword_enchantment = type;
    s.achievements[c::ENCHANT_SWORD] = 1u;
    enchanted = true;
  } else if (action == a::ENCHANT_BOW && inv.bow > 0) {
    s.bow_enchantment = type;
    enchanted = true;
  } else if (action == a::ENCHANT_ARMOUR &&
             std::accumulate(inv.armour.begin(), inv.armour.end(), 0) > 0) {
    const bool has_unenchanted =
        std::find(s.armour_enchantments.begin(), s.armour_enchantments.end(),
                  0) != s.armour_enchantments.end();
    std::vector<float> weights;
    weights.reserve(s.armour_enchantments.size());
    for (int enchantment : s.armour_enchantments) {
      weights.push_back(static_cast<float>(
          has_unenchanted ? enchantment == 0 : enchantment != type));
    }
    const int target = std::clamp(Choice(TakeKey(&enchant_rng), weights), 0, 3);
    s.armour_enchantments[target] = type;
    s.achievements[c::ENCHANT_ARMOUR] = 1u;
    enchanted = true;
  }
  if (enchanted) {
    s.player_mana -= 9;
    --gems;
  }
}

}  // namespace craftax
