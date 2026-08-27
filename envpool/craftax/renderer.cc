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

#include "envpool/craftax/renderer.h"

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "lodepng.h"
#include "third_party/craftax/constants.h"

namespace craftax {
namespace {

struct Texture {
  int size;
  std::vector<unsigned char> rgba;
};

class Textures {
 public:
  std::map<std::pair<std::string, int>, Texture> images;
  std::vector<float> night;

  Textures(bool classic, int tile) {
    const std::string prefix = classic ? "classic/" : "full/";
    for (std::size_t i = 0; i < kTextureCount; ++i) {
      const auto& encoded = kTextures[i];
      const std::string name = encoded.name;
      if (name.compare(0, prefix.size(), prefix) != 0) {
        continue;
      }
      std::vector<unsigned char> pixels;
      unsigned int width = 0;
      unsigned int height = 0;
      const unsigned error =
          lodepng::decode(pixels, width, height, encoded.bytes, encoded.size);
      if ((error != 0u) || width != 16 || height != 16) {
        throw std::runtime_error("invalid Craftax texture: " + name);
      }
      for (int size : {tile, static_cast<int>(tile * 0.8),
                       static_cast<int>(tile * (classic ? 0.6 : 0.4))}) {
        Texture texture{size, std::vector<unsigned char>(size * size * 4)};
        std::vector<int> coordinates(size);
        const double step = 16.0 / size;
        double coordinate = step * 0.5;
        for (auto& value : coordinates) {
          value = std::min(15, static_cast<int>(coordinate));
          coordinate += step;
        }
        for (int y = 0; y < size; ++y) {
          for (int x = 0; x < size; ++x) {
            // Pillow's nearest-neighbour resize samples pixel centers.
            const int sy = coordinates[y];
            const int sx = coordinates[x];
            std::copy_n(pixels.begin() + (sy * 16 + sx) * 4, 4,
                        texture.rgba.begin() + (y * size + x) * 4);
          }
        }
        images.emplace(std::make_pair(name.substr(prefix.size()), size),
                       std::move(texture));
      }
    }
    const int h = (classic ? 7 : 9) * tile;
    const int w = (classic ? 9 : 11) * tile;
    night.resize(h * w);
    const float maximum = EuclideanDistance({h / 2, w / 2}, {});
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        if (classic) {
          const double dy = -1.0 + y * (2.0 / (h - 1));
          const double dx = -1.0 + x * (2.0 / (w - 1));
          night[y * w + x] =
              static_cast<float>(1.0 - std::exp(-2.0 * (dy * dy + dx * dx)));
        } else {
          night[y * w + x] =
              EuclideanDistance({y, x}, {h / 2, w / 2}) * (1.0f / maximum);
        }
      }
    }
  }

  const Texture& Get(const std::string& name, int size) const {
    return images.at({name, size});
  }
};

const Textures& GetTextures(bool classic, int tile) {
  static std::mutex mutex;
  static std::map<std::pair<bool, int>, std::unique_ptr<Textures>> textures;
  std::scoped_lock lock(mutex);
  auto& entry = textures[{classic, tile}];
  if (!entry) {
    entry = std::make_unique<Textures>(classic, tile);
  }
  return *entry;
}

// Raw RGB is used for several inventory icons. Sprites instead use either
// binary alpha or the Classic renderer's original fractional alpha channel.
enum class Blend : std::uint8_t { kRgb, kBinary, kFractional };

void Draw(std::vector<float>* pixels, int height, int width,
          const Texture& texture, int top, int left, Blend blend,
          bool flip = false, bool transpose = false) {
  const int size = texture.size;
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      if (top + y < 0 || top + y >= height || left + x < 0 ||
          left + x >= width) {
        continue;
      }
      int sy = transpose ? x : y;
      int sx = transpose ? y : x;
      if (flip) {
        sy = size - 1 - sy;
      }
      const auto* rgba = texture.rgba.data() + (sy * size + sx) * 4;
      float alpha =
          blend == Blend::kRgb ? 1.0f : static_cast<float>(rgba[3] == 255);
      if (blend == Blend::kFractional) {
        alpha = static_cast<float>(rgba[3]) * (1.0f / 255.0f);
      }
      for (int c = 0; c < 3; ++c) {
        float& pixel = (*pixels)[((top + y) * width + left + x) * 3 + c];
        pixel =
            std::fma(pixel, 1.0f - alpha, static_cast<float>(rgba[c]) * alpha);
      }
    }
  }
}

void InventoryPixels(const Game& game, const Textures& textures, int tile,
                     std::vector<float>* pixels) {
  const auto& s = game.state;
  const auto& inv = s.inventory;
  const bool classic = game.params.classic;
  const int rows = classic ? 7 : 9;
  const int cols = classic ? 9 : 11;
  const int h = (rows + (classic ? 2 : 4)) * tile;
  const int w = cols * tile;
  const int small = static_cast<int>(tile * 0.8);
  const int number = static_cast<int>(tile * (classic ? 0.6 : 0.4));
  const int inset = (tile - small) / 2 - static_cast<int>(classic);
  auto icon = [&](const std::string& name, int x, int y,
                  Blend blend = Blend::kRgb) {
    Draw(pixels, h, w, textures.Get(name, small), (rows + y) * tile + inset,
         x * tile + inset, blend);
  };
  auto digit = [&](int n, int left, int top, bool zero) {
    n = std::clamp(n, 0, 9);
    if (n == 0 && !zero) {
      return;
    }
    Draw(pixels, h, w, textures.Get(std::to_string(n) + ".png", number), top,
         left, Blend::kBinary);
  };
  auto count = [&](int n, int x, int y, bool two = true) {
    const int left = (x + 1) * tile - number - classic;
    const int top = (rows + y + 1) * tile - number - classic;
    if (classic || !two) {
      digit(n, left, top, false);
    } else {
      digit(n % 10, left, top, n > 0);
      digit(n / 10, left - number, top, false);
    }
  };
  auto item = [&](const std::string& name, int n, int x, int y,
                  Blend blend = Blend::kRgb, bool two = true) {
    if (n > 0) {
      icon(name, x, y, blend);
    }
    count(n, x, y, two);
  };
  const int health =
      classic ? static_cast<int>(s.player_health)
              : std::max(1, static_cast<int>(std::floor(s.player_health)));
  item("health.png", health, 0, 0);
  item("food.png", s.player_food, 1, 0);
  item("drink.png", s.player_drink, 2, 0);
  item("energy.png", s.player_energy, 3, 0);
  if (classic) {
    item("sapling.png", inv.sapling, 4, 0);
    item("wood.png", inv.wood, 5, 0);
    item("stone.png", inv.stone, 6, 0);
    item("coal.png", inv.coal, 7, 0);
    item("iron.png", inv.iron, 8, 0);
    item("diamond.png", inv.diamond, 0, 1);
    item("wood_pickaxe.png", inv.wood_pickaxe, 1, 1);
    item("stone_pickaxe.png", inv.stone_pickaxe, 2, 1, Blend::kBinary);
    item("iron_pickaxe.png", inv.iron_pickaxe, 3, 1, Blend::kBinary);
    item("wood_sword.png", inv.wood_sword, 4, 1, Blend::kBinary);
    item("stone_sword.png", inv.stone_sword, 5, 1, Blend::kBinary);
    item("iron_sword.png", inv.iron_sword, 6, 1, Blend::kBinary);
    return;
  }
  item("mana.png", s.player_mana, 4, 0);
  item("wood.png", inv.wood, 0, 2);
  item("stone.png", inv.stone, 1, 2);
  item("coal.png", inv.coal, 0, 1);
  item("iron.png", inv.iron, 1, 1);
  item("diamond.png", inv.diamond, 2, 1);
  item("sapphire.png", inv.sapphire, 3, 1);
  item("ruby.png", inv.ruby, 4, 1);
  item("sapling.png", inv.sapling, 5, 1);
  const std::array<const char*, 5> materials{"", "wood", "stone", "iron",
                                             "diamond"};
  if (inv.pickaxe != 0) {
    icon(std::string(materials[std::clamp(inv.pickaxe, 0, 4)]) + "_pickaxe.png",
         8, 2, Blend::kBinary);
  }
  if (inv.sword != 0) {
    icon(std::string(materials[std::clamp(inv.sword, 0, 4)]) + "_sword.png", 8,
         1, Blend::kBinary);
  }
  if (inv.bow != 0) {
    icon("bow.png", 6, 1);
  }
  item("arrow-up.png", inv.arrows, 6, 2, Blend::kBinary);
  const std::array<const char*, 4> armour{"helmet", "chestplate", "pants",
                                          "boots"};
  for (int i = 0; i < 4; ++i) {
    if (inv.armour[i] != 0) {
      icon(std::string(inv.armour[i] == 1 ? "iron_" : "diamond_") + armour[i] +
               ".png",
           7, i, Blend::kBinary);
    }
  }
  item("torch_in_inventory.png", inv.torches, 2, 2);
  const std::array<const char*, 6> colours{"red",  "green", "blue",
                                           "pink", "cyan",  "yellow"};
  for (int i = 0; i < 6; ++i) {
    item(std::string("potion_") + colours[i] + ".png", inv.potions[i], i, 3);
  }
  item("book.png", inv.books, 3, 2);
  if (s.learned_spells[0] != 0u) {
    icon("fireball.png", 4, 2);
  }
  if (s.learned_spells[1] != 0u) {
    icon("iceball.png", 5, 2);
  }
  auto enchant = [&](const std::string& name, int type, int x, int y) {
    if (type) {
      icon(
          name + (type == 1 ? "_fire_enchantment.png" : "_ice_enchantment.png"),
          x, y, Blend::kBinary);
    }
  };
  enchant("sword", s.sword_enchantment, 8, 1);
  enchant("arrow", s.bow_enchantment * static_cast<int>(inv.arrows > 0), 6, 2);
  for (int i = 0; i < 4; ++i) {
    enchant(armour[i], s.armour_enchantments[i], 7, i);
  }
  count(s.player_level, 6, 0, false);
  item("xp.png", s.player_xp, 9, 0, Blend::kRgb, false);
  item("dexterity.png", s.player_dexterity, 9, 1, Blend::kRgb, false);
  item("strength.png", s.player_strength, 9, 2, Blend::kRgb, false);
  item("intelligence.png", s.player_intelligence, 9, 3, Blend::kRgb, false);
}

}  // namespace

std::vector<float> Pixels(const Game& game, int tile) {
  if (tile != 7 && tile != 10 && tile != 16 && tile != 64) {
    throw std::invalid_argument("unsupported Craftax tile size");
  }
  const auto& s = game.state;
  const auto& p = game.params;
  const bool classic = p.classic;
  const auto& textures = GetTextures(classic, tile);
  const int rows = classic ? 7 : 9;
  const int cols = classic ? 9 : 11;
  const int mh = rows * tile;
  const int h = (rows + (classic ? 2 : 4)) * tile;
  const int w = cols * tile;
  std::vector<float> pixels(h * w * 3);
  const Pos top = game.ViewOrigin(rows, cols);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      const Pos pos = top + Pos{y, x};
      int block = game.InBounds(pos) ? game.Block(pos) : 1;
      if (!classic && block == full::block::NECROMANCER &&
          game.BossVulnerable()) {
        block = full::block::NECROMANCER_VULNERABLE;
      }
      const char* name = classic ? classic::BLOCK_TEXTURE_NAMES[block]
                                 : full::BLOCK_TEXTURE_NAMES[block];
      Draw(&pixels, h, w, textures.Get(name, tile), y * tile, x * tile,
           Blend::kRgb);
      if (block == 1 || (!classic && block == full::block::DARKNESS)) {
        for (int dy = 0; dy < tile; ++dy) {
          std::fill_n(pixels.begin() + ((y * tile + dy) * w + x * tile) * 3,
                      tile * 3, block == 1 ? 128.0f : 0.0f);
        }
      }
      if (!classic && game.InBounds(pos)) {
        int item = s.item_map[game.Index(pos)];
        if (item == full::item::LADDER_DOWN &&
            s.monsters_killed[s.player_level] <
                full::MONSTERS_KILLED_TO_CLEAR_LEVEL) {
          item = full::item::LADDER_DOWN_BLOCKED;
        }
        if (item != 0) {
          Draw(&pixels, h, w,
               textures.Get(full::ITEM_TEXTURE_NAMES[item], tile), y * tile,
               x * tile, Blend::kBinary);
        }
      }
    }
  }
  const int player =
      s.is_sleeping ? 4 : std::clamp(s.player_direction - 1, 0, 4);
  const char* player_name = classic ? classic::PLAYER_TEXTURE_NAMES[player]
                                    : full::PLAYER_TEXTURE_NAMES[player];
  Draw(&pixels, h, w, textures.Get(player_name, tile), (rows / 2) * tile,
       (cols / 2) * tile, classic ? Blend::kFractional : Blend::kBinary);
  auto mobs = [&](const std::vector<Mob>& group, int capacity,
                  const char* const* names, int num_types,
                  const std::vector<Pos>* directions = nullptr) {
    const int base = s.player_level * capacity;
    for (int i = base; i < base + capacity; ++i) {
      const auto& mob = group[i];
      const Pos local =
          mob.position - s.player_position + Pos{rows / 2, cols / 2};
      if (!mob.mask || local.y < 0 || local.x < 0 || local.y >= rows ||
          local.x >= cols) {
        continue;
      }
      const Pos dir = directions ? (*directions)[i] : Pos{};
      const Blend blend =
          classic && !directions ? Blend::kFractional : Blend::kBinary;
      Draw(&pixels, h, w,
           textures.Get(names[std::clamp(mob.type_id, 0, num_types - 1)], tile),
           local.y * tile, local.x * tile, blend, dir.y > 0 || dir.x > 0,
           dir.x != 0);
    }
  };
  constexpr std::array<const char*, 1> zombies{"zombie.png"};
  constexpr std::array<const char*, 1> cows{"cow.png"};
  constexpr std::array<const char*, 1> skeletons{"skeleton.png"};
  constexpr std::array<const char*, 1> arrows{"arrow-up.png"};
  mobs(s.melee_mobs, p.max_melee_mobs,
       classic ? zombies.data() : full::MELEE_TEXTURE_NAMES, classic ? 1 : 8);
  mobs(s.passive_mobs, p.max_passive_mobs,
       classic ? cows.data() : full::PASSIVE_TEXTURE_NAMES, classic ? 1 : 3);
  mobs(s.ranged_mobs, p.max_ranged_mobs,
       classic ? skeletons.data() : full::RANGED_TEXTURE_NAMES,
       classic ? 1 : 8);
  mobs(s.mob_projectiles, p.max_mob_projectiles,
       classic ? arrows.data() : full::PROJECTILE_TEXTURE_NAMES,
       classic ? 1 : 8, &s.mob_projectile_directions);
  if (!classic) {
    mobs(s.player_projectiles, p.max_player_projectiles,
         full::PROJECTILE_TEXTURE_NAMES, 8, &s.player_projectile_directions);
  }
  const float daylight = !classic && s.player_level != 0 ? 1.0f : s.light_level;
  const float intensity = std::max(2.0f * (0.5f - daylight), 0.0f);
  constexpr std::array<float, 3> blue{0, 16, 64};
  for (int y = 0; y < mh; ++y) {
    for (int x = 0; x < w; ++x) {
      const int index = y * w + x;
      const Pos pos = top + Pos{y / tile, x / tile};
      float light = 1.0f;
      if (!classic) {
        light = game.InBounds(pos) ? s.light_map[game.Index(pos)] : 0.0f;
      }
      float* pixel = pixels.data() + index * 3;
      std::array<float, 3> night;
      const float mask = intensity * textures.night[index];
      const float noise = std::fma(Uniform(s.state_rng, index), 95.0f, 32.0f);
      for (int c = 0; c < 3; ++c) {
        pixel[c] *= light;
        night[c] = std::fma(mask, noise, (1.0f - mask) * pixel[c]);
      }
      if (classic) {
        const float luminance = std::fma(
            0.114f, night[2], std::fma(0.299f, night[0], 0.587f * night[1]));
        for (int c = 0; c < 3; ++c) {
          night[c] = std::fma(0.4f, night[c], 0.6f * luminance);
        }
      }
      for (int c = 0; c < 3; ++c) {
        night[c] = 0.5f * night[c] + 0.5f * blue[c];
        pixel[c] = std::fma(daylight, pixel[c], (1.0f - daylight) * night[c]);
      }
      if (s.is_sleeping) {
        if (classic) {
          const float gray =
              std::fma(0.114f, pixel[2],
                       std::fma(0.299f, pixel[0], 0.587f * pixel[1])) *
              0.5f;
          pixel[0] = pixel[1] = gray;
          pixel[2] = gray + 8.0f;
        } else {
          for (int c = 0; c < 3; ++c) {
            pixel[c] *= 0.5f;
          }
        }
      }
    }
  }
  InventoryPixels(game, textures, tile, &pixels);
  return pixels;
}

}  // namespace craftax
