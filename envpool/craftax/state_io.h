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

#ifndef ENVPOOL_CRAFTAX_STATE_IO_H_
#define ENVPOOL_CRAFTAX_STATE_IO_H_

#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "envpool/craftax/game.h"
#include "third_party/craftax/constants.h"

namespace craftax {

template <typename T>
T StateValue(double value) {
  if (!std::isfinite(value) || value < std::numeric_limits<T>::lowest() ||
      value > std::numeric_limits<T>::max() ||
      (std::is_integral_v<T> && value != std::trunc(value))) {
    throw std::invalid_argument("invalid value in Craftax state");
  }
  return static_cast<T>(value);
}

// One state traversal is shared by the reset-time oracle exchange and the
// diagnostic binding. It performs no stepping and consumes no randomness.
template <typename Visitor>
void VisitState(Game* game, Visitor visit) {
  auto& s = game->state;
  const bool classic = game->params.classic;
  auto scalar = [&](const std::string& name, auto* field) {
    std::vector<double> values{static_cast<double>(*field)};
    visit(name, &values);
    *field =
        StateValue<std::remove_reference_t<decltype(*field)>>(values.at(0));
  };
  auto array = [&](const std::string& name, auto* field) {
    std::vector<double> values(field->begin(), field->end());
    visit(name, &values);
    for (std::size_t i = 0; i < field->size(); ++i) {
      (*field)[i] = StateValue<
          typename std::remove_reference_t<decltype(*field)>::value_type>(
          values.at(i));
    }
  };
  auto position = [&](const std::string& name, Pos* field) {
    std::vector<double> values{static_cast<double>(field->y),
                               static_cast<double>(field->x)};
    visit(name, &values);
    *field = {StateValue<int>(values.at(0)), StateValue<int>(values.at(1))};
  };
  auto positions = [&](const std::string& name, std::vector<Pos>* field) {
    std::vector<double> values;
    for (const auto& pos : *field) {
      values.push_back(pos.y);
      values.push_back(pos.x);
    }
    visit(name, &values);
    for (std::size_t i = 0; i < field->size(); ++i) {
      (*field)[i] = {StateValue<int>(values.at(i * 2)),
                     StateValue<int>(values.at(i * 2 + 1))};
    }
  };
  auto mobs = [&](const std::string& name, std::vector<Mob>* field) {
    std::vector<Pos> pos;
    std::vector<float> health;
    std::vector<std::uint8_t> mask;
    std::vector<int> cooldown;
    std::vector<int> type;
    for (const auto& mob : *field) {
      pos.push_back(mob.position);
      health.push_back(mob.health);
      mask.push_back(mob.mask);
      cooldown.push_back(mob.attack_cooldown);
      type.push_back(mob.type_id);
    }
    positions(name + ".position", &pos);
    array(name + ".health", &health);
    array(name + ".mask", &mask);
    array(name + ".attack_cooldown", &cooldown);
    if (!classic) {
      array(name + ".type_id", &type);
    }
    for (std::size_t i = 0; i < field->size(); ++i) {
      (*field)[i] = {pos[i], health[i], mask[i] != 0, cooldown[i], type[i]};
    }
  };

  array("map", &s.map);
  array("mob_map", &s.mob_map);
  position("player_position", &s.player_position);
#define CRAFTAX_SCALAR(field) scalar(#field, &s.field)
#define CRAFTAX_ARRAY(field) array(#field, &s.field)
  CRAFTAX_SCALAR(player_direction);
  CRAFTAX_SCALAR(player_health);
  CRAFTAX_SCALAR(player_food);
  CRAFTAX_SCALAR(player_drink);
  CRAFTAX_SCALAR(player_energy);
  CRAFTAX_SCALAR(is_sleeping);
  CRAFTAX_SCALAR(player_recover);
  CRAFTAX_SCALAR(player_hunger);
  CRAFTAX_SCALAR(player_thirst);
  CRAFTAX_SCALAR(player_fatigue);
  CRAFTAX_SCALAR(inventory.wood);
  CRAFTAX_SCALAR(inventory.stone);
  CRAFTAX_SCALAR(inventory.coal);
  CRAFTAX_SCALAR(inventory.iron);
  CRAFTAX_SCALAR(inventory.diamond);
  CRAFTAX_SCALAR(inventory.sapling);
  if (classic) {
    CRAFTAX_SCALAR(inventory.wood_pickaxe);
    CRAFTAX_SCALAR(inventory.stone_pickaxe);
    CRAFTAX_SCALAR(inventory.iron_pickaxe);
    CRAFTAX_SCALAR(inventory.wood_sword);
    CRAFTAX_SCALAR(inventory.stone_sword);
    CRAFTAX_SCALAR(inventory.iron_sword);
  } else {
    CRAFTAX_ARRAY(item_map);
    CRAFTAX_ARRAY(light_map);
    positions("down_ladders", &s.down_ladders);
    positions("up_ladders", &s.up_ladders);
    CRAFTAX_ARRAY(chests_opened);
    CRAFTAX_ARRAY(monsters_killed);
    CRAFTAX_SCALAR(player_level);
    CRAFTAX_SCALAR(player_mana);
    CRAFTAX_SCALAR(is_resting);
    CRAFTAX_SCALAR(player_recover_mana);
    CRAFTAX_SCALAR(player_xp);
    CRAFTAX_SCALAR(player_dexterity);
    CRAFTAX_SCALAR(player_strength);
    CRAFTAX_SCALAR(player_intelligence);
    CRAFTAX_SCALAR(inventory.pickaxe);
    CRAFTAX_SCALAR(inventory.sword);
    CRAFTAX_SCALAR(inventory.bow);
    CRAFTAX_SCALAR(inventory.arrows);
    CRAFTAX_ARRAY(inventory.armour);
    CRAFTAX_SCALAR(inventory.torches);
    CRAFTAX_SCALAR(inventory.ruby);
    CRAFTAX_SCALAR(inventory.sapphire);
    CRAFTAX_ARRAY(inventory.potions);
    CRAFTAX_SCALAR(inventory.books);
    mobs("player_projectiles", &s.player_projectiles);
    positions("player_projectile_directions", &s.player_projectile_directions);
    CRAFTAX_ARRAY(potion_mapping);
    CRAFTAX_ARRAY(learned_spells);
    CRAFTAX_SCALAR(sword_enchantment);
    CRAFTAX_SCALAR(bow_enchantment);
    CRAFTAX_ARRAY(armour_enchantments);
    CRAFTAX_SCALAR(boss_progress);
    CRAFTAX_SCALAR(boss_timesteps_to_spawn_this_round);
  }
  mobs(classic ? "zombies" : "melee_mobs", &s.melee_mobs);
  mobs(classic ? "cows" : "passive_mobs", &s.passive_mobs);
  mobs(classic ? "skeletons" : "ranged_mobs", &s.ranged_mobs);
  mobs(classic ? "arrows" : "mob_projectiles", &s.mob_projectiles);
  positions(classic ? "arrow_directions" : "mob_projectile_directions",
            &s.mob_projectile_directions);
  positions("growing_plants_positions", &s.growing_plants_positions);
  CRAFTAX_ARRAY(growing_plants_age);
  CRAFTAX_ARRAY(growing_plants_mask);
  CRAFTAX_SCALAR(light_level);
  CRAFTAX_ARRAY(achievements);
  CRAFTAX_ARRAY(state_rng);
  CRAFTAX_SCALAR(timestep);
#undef CRAFTAX_SCALAR
#undef CRAFTAX_ARRAY
}

inline std::vector<double> EncodeState(Game* game) {
  std::vector<double> out;
  VisitState(game, [&](const std::string&, std::vector<double>* values) {
    out.insert(out.end(), values->begin(), values->end());
  });
  return out;
}

inline void DecodeState(Game* game, const std::vector<double>& input) {
  std::size_t offset = 0;
  VisitState(game, [&](const std::string&, std::vector<double>* values) {
    if (offset + values->size() > input.size()) {
      throw std::invalid_argument("short Craftax state");
    }
    std::copy_n(input.begin() + offset, values->size(), values->begin());
    offset += values->size();
  });
  if (offset != input.size()) {
    throw std::invalid_argument("oversized Craftax state");
  }
  const auto& s = game->state;
  auto bounded = [](const auto& values, int count) {
    return std::all_of(values.begin(), values.end(), [count](auto value) {
      return value >= 0 && value < count;
    });
  };
  const int blocks =
      game->params.classic ? classic::block::COUNT : full::block::COUNT;
  if (!bounded(s.map, blocks) || !bounded(s.item_map, full::item::COUNT) ||
      s.player_level < 0 || s.player_level >= game->params.levels ||
      s.player_direction < 1 || s.player_direction > 4 ||
      s.bow_enchantment < 0 || s.bow_enchantment > 2 ||
      s.sword_enchantment < 0 || s.sword_enchantment > 2 ||
      !bounded(s.potion_mapping, 6) || !bounded(s.armour_enchantments, 3)) {
    throw std::invalid_argument("invalid index in Craftax state");
  }
  for (const auto* group : {&s.melee_mobs, &s.passive_mobs, &s.ranged_mobs,
                            &s.mob_projectiles, &s.player_projectiles}) {
    for (const auto& mob : *group) {
      if (mob.type_id < 0 || mob.type_id >= 8) {
        throw std::invalid_argument("invalid mob type in Craftax state");
      }
    }
  }
}

}  // namespace craftax

#endif  // ENVPOOL_CRAFTAX_STATE_IO_H_
