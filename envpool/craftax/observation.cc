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

#include <vector>

#include "envpool/craftax/game.h"
#include "third_party/craftax/constants.h"

namespace craftax {

std::vector<float> Game::FullSymbolic() const {
  constexpr int h = 9;
  constexpr int w = 11;
  constexpr int channels = 83;
  std::vector<float> obs(h * w * channels);
  const Pos top = ViewOrigin(h, w);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const Pos pos = top + Pos{y, x};
      const int block = InBounds(pos) ? Block(pos) : full::block::OUT_OF_BOUNDS;
      const int item =
          InBounds(pos) ? state.item_map[Index(pos)] : full::item::NONE;
      const int offset = (y * w + x) * channels;
      if (block >= 0 && block < 37) {
        obs[offset + block] = 1;
      }
      if (item >= 0 && item < 5) {
        obs[offset + 37 + item] = 1;
      }
    }
  }
  const std::array<const std::vector<Mob>*, 5> groups{
      &state.melee_mobs, &state.passive_mobs, &state.ranged_mobs,
      &state.mob_projectiles, &state.player_projectiles};
  const std::array<int, 5> counts{
      params.max_melee_mobs, params.max_passive_mobs, params.max_ranged_mobs,
      params.max_mob_projectiles, params.max_player_projectiles};
  for (int group = 0; group < 5; ++group) {
    for (int i = 0; i < counts[group]; ++i) {
      const auto& mob =
          (*groups[group])[state.player_level * counts[group] + i];
      Pos pos = mob.position - state.player_position + Pos{h / 2, w / 2};
      const bool on_screen = pos.y >= 0 && pos.y < h && pos.x >= 0 && pos.x < w;
      if (pos.y < 0) {
        pos.y += h;
      }
      if (pos.x < 0) {
        pos.x += w;
      }
      const int identifier = group * 8 + mob.type_id;
      if (pos.y >= 0 && pos.y < h && pos.x >= 0 && pos.x < w &&
          identifier >= 0 && identifier < 40) {
        obs[(pos.y * w + pos.x) * channels + 42 + identifier] =
            static_cast<float>(on_screen && mob.mask);
      }
    }
  }
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const Pos pos = top + Pos{y, x};
      const bool visible = InBounds(pos) && state.light_map[Index(pos)] > 0.05f;
      const int offset = (y * w + x) * channels;
      if (!visible) {
        std::fill_n(obs.begin() + offset, channels - 1, 0.0f);
      }
      obs[offset + channels - 1] = static_cast<float>(visible);
    }
  }
  const auto& inv = state.inventory;
  for (int item :
       {inv.wood, inv.stone, inv.coal, inv.iron, inv.diamond, inv.sapphire,
        inv.ruby, inv.sapling, inv.torches, inv.arrows}) {
    obs.push_back(std::sqrt(static_cast<float>(item)) * 0.1f);
  }
  obs.push_back(inv.books * 0.5f);
  obs.push_back(inv.pickaxe * 0.25f);
  obs.push_back(inv.sword * 0.25f);
  obs.push_back(static_cast<float>(state.sword_enchantment));
  obs.push_back(static_cast<float>(state.bow_enchantment));
  obs.push_back(static_cast<float>(inv.bow));
  for (int potion : inv.potions) {
    obs.push_back(std::sqrt(static_cast<float>(potion)) * 0.1f);
  }
  obs.push_back(state.player_health * 0.1f);
  for (int value : {state.player_food, state.player_drink, state.player_energy,
                    state.player_mana, state.player_xp, state.player_dexterity,
                    state.player_strength, state.player_intelligence}) {
    obs.push_back(value * 0.1f);
  }
  for (int i = 1; i <= 4; ++i) {
    obs.push_back(static_cast<float>(state.player_direction == i));
  }
  for (int armour : inv.armour) {
    obs.push_back(armour * 0.5f);
  }
  for (int enchantment : state.armour_enchantments) {
    obs.push_back(static_cast<float>(enchantment));
  }
  obs.push_back(state.light_level);
  obs.push_back(static_cast<float>(state.is_sleeping));
  obs.push_back(static_cast<float>(state.is_resting));
  obs.push_back(state.learned_spells[0]);
  obs.push_back(state.learned_spells[1]);
  obs.push_back(state.player_level * 0.1f);
  obs.push_back(static_cast<float>(state.monsters_killed[state.player_level] >=
                                   full::MONSTERS_KILLED_TO_CLEAR_LEVEL));
  obs.push_back(static_cast<float>(BossVulnerable()));
  return obs;
}

}  // namespace craftax
