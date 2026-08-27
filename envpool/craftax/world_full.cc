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
#include <numeric>
#include <vector>

#include "envpool/craftax/game.h"
#include "envpool/craftax/noise.h"
#include "third_party/craftax/constants.h"

namespace craftax {
namespace b = full::block;
namespace i = full::item;

static void SmoothWorld(Game* game, Key rng,
                        const full::SmoothGenConfig& config, int level) {
  auto& s = game->state;
  const auto& p = game->params;
  const int h = p.height;
  const int w = p.width;
  const int cells = game->Cells();
  s.player_level = level;
  auto water =
      Noise(TakeKey(&rng), h, w, h / 16, w / 16, p.fractal_noise_angles[0]);
  TakeKey(&rng);
  auto mountain =
      Noise(TakeKey(&rng), h, w, h / 16, w / 16, p.fractal_noise_angles[1]);
  const auto path =
      Noise(TakeKey(&rng), h, w, h / 8, w / 2, p.fractal_noise_angles[2]);
  TakeKey(&rng);
  const auto tree =
      Noise(TakeKey(&rng), h, w, h / 4, w / 4, p.fractal_noise_angles[3]);
  std::vector<int> blocks(cells);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int i = y * w + x;
      const float distance = EuclideanDistance({y, x}, s.player_position);
      water[i] =
          water[i] +
          std::clamp(
              distance / config.player_proximity_map_water_strength, 0.0f,
              static_cast<float>(config.player_proximity_map_water_max)) -
          1.0f;
      mountain[i] = mountain[i] + 0.05f;
      mountain[i] =
          mountain[i] +
          std::clamp(distance / config.player_proximity_map_mountain_strength,
                     0.0f, config.player_proximity_map_mountain_max) -
          1.0f;
      int block = water[i] > config.water_threshold ? config.sea_block
                                                    : config.default_block;
      if (water[i] > config.sand_threshold && block != config.sea_block) {
        block = config.coast_block;
      }
      if (mountain[i] > 0.7f) {
        block = config.mountain_block;
      }
      if (mountain[i] > 0.7f && (path[i] > 0.8f || path[x * w + y] > 0.8f)) {
        block = config.path_block;
      }
      if (mountain[i] > 0.85f && water[i] > 0.4f) {
        block = config.inner_mountain_block;
      }
      if (block == config.tree_requirement_block &&
          static_cast<float>(tree[i] > config.tree_threshold_perlin) *
                  Uniform(rng, i) >
              config.tree_threshold_uniform) {
        block = config.tree;
      }
      blocks[i] = block;
    }
  }
  Key ore_rng = TakeKey(&rng);
  for (int ore = 0; ore < 5; ++ore) {
    const Key draw = TakeKey(&ore_rng);
    for (int i = 0; i < cells; ++i) {
      if (blocks[i] == config.ore_requirement_blocks[ore] &&
          Uniform(draw, i) < config.ore_chances[ore]) {
        blocks[i] = config.ores[ore];
      }
    }
  }
  std::vector<std::uint8_t> lava(cells);
  std::vector<float> weights(cells);
  int count = 0;
  for (int i = 0; i < cells; ++i) {
    lava[i] = static_cast<std::uint8_t>(mountain[i] > 0.85f && tree[i] > 0.7f);
    if (lava[i] != 0u) {
      blocks[i] = config.lava;
    }
    weights[i] = static_cast<float>(blocks[i] == b::STONE);
    count += static_cast<int>(blocks[i] == b::STONE);
  }
  for (auto& weight : weights) {
    weight /= static_cast<float>(count);
  }
  const int diamond = std::clamp(Choice(TakeKey(&rng), weights), 0, cells - 1);
  blocks[diamond] = config.default_block == b::GRASS && p.always_diamond
                        ? b::DIAMOND
                        : b::STONE;
  blocks[s.player_position.y * w + s.player_position.x] = config.player_spawn;
  for (int i = 0; i < cells; ++i) {
    weights[i] = static_cast<float>(blocks[i] == config.valid_ladder);
  }
  const int down = std::clamp(Choice(TakeKey(&rng), weights), 0, cells - 1);
  const int up = std::clamp(Choice(TakeKey(&rng), weights), 0, cells - 1);
  s.down_ladders[level] = {down / h, down % h};
  s.up_ladders[level] = {up / h, up % h};
  const int base = level * cells;
  std::copy(blocks.begin(), blocks.end(), s.map.begin() + base);
  if (config.ladder_down != 0) {
    s.item_map[base + down] = i::LADDER_DOWN;
  }
  if (config.ladder_up != 0) {
    s.item_map[base + up] = i::LADDER_UP;
  }
  std::fill_n(s.light_map.begin() + base, cells, config.default_light);
  Pos start = s.up_ladders[level] - Pos{4, 4};
  if (start.y < 0) {
    start.y += h;
  }
  if (start.x < 0) {
    start.x += w;
  }
  start.y = std::clamp(start.y, 0, h - 9);
  start.x = std::clamp(start.x, 0, w - 9);
  for (int y = 0; y < 9; ++y) {
    for (int x = 0; x < 9; ++x) {
      const float torch = TorchLight({y - 4, x - 4});
      s.light_map[base + (start.y + y) * w + start.x + x] =
          torch * (1.0f - config.default_light) + config.default_light;
    }
  }
  if (config.lava == b::LAVA) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        float light = 0;
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            const Pos q{y + dy, x + dx};
            if (game->InBounds(q) && (lava[q.y * w + q.x] != 0u)) {
              const float neighbour = dy == 0 || dx == 0 ? 0.7f : 0.2f;
              light += dy == 0 && dx == 0 ? 1.0f : neighbour;
            }
          }
        }
        s.light_map[base + y * w + x] =
            std::clamp(s.light_map[base + y * w + x] + light, 0.0f, 1.0f);
      }
    }
  }
}

static void Dungeon(Game* game, Key rng, const full::DungeonConfig& config,
                    int level) {
  auto& s = game->state;
  const int h = game->params.height;
  const int w = game->params.width;
  const int ph = h + 20;
  const int pw = w + 20;
  const int chunks = h / 16;
  s.player_level = level;
  std::vector<int> blocks(ph * pw);
  std::vector<int> items(ph * pw);
  for (int y = 10; y < h + 10; ++y) {
    std::fill_n(blocks.begin() + y * pw + 10, w, b::WALL);
  }
  const Key sizes_key = Split(rng, 2);
  rng = Split(rng, 0);
  std::array<Pos, 8> sizes;
  std::array<Pos, 8> positions;
  for (int i = 0; i < 8; ++i) {
    sizes[i] = {RandInt(sizes_key, 5, 10, i * 2),
                RandInt(sizes_key, 5, 10, i * 2 + 1)};
  }
  std::vector<float> occupancy(chunks * chunks, 1.0f);
  Key rooms_rng = TakeKey(&rng);
  for (int room = 0; room < 8; ++room) {
    const int chunk = std::clamp(Choice(TakeKey(&rooms_rng), occupancy), 0,
                                 chunks * chunks - 1);
    occupancy[chunk] = 0;
    const Key offset = TakeKey(&rooms_rng);
    const Pos pos{(chunk % chunks) * 16 + 10 + RandInt(offset, 0, 11, 0),
                  (chunk / chunks) * 16 + 10 + RandInt(offset, 0, 11, 1)};
    positions[room] = pos;
    for (int y = 0; y < sizes[room].y; ++y) {
      for (int x = 0; x < sizes[room].x; ++x) {
        blocks[(pos.y + y) * pw + pos.x + x] = b::PATH;
      }
    }
    for (int y : {0, sizes[room].y - 1}) {
      for (int x : {0, sizes[room].x - 1}) {
        items[(pos.y + y) * pw + pos.x + x] = i::TORCH;
      }
    }
    const Key chest = TakeKey(&rooms_rng);
    const Pos chest_pos{pos.y + RandInt(chest, 1, sizes[room].y - 1, 0),
                        pos.x + RandInt(chest, 1, sizes[room].x - 1, 1)};
    blocks[chest_pos.y * pw + chest_pos.x] = b::CHEST;
    const Key fountain = Split(rooms_rng, 1);
    const Key coin = Split(rooms_rng, 2);
    rooms_rng = Split(rooms_rng, 0);
    const Pos fountain_pos{pos.y + RandInt(fountain, 1, sizes[room].y - 1, 0),
                           pos.x + RandInt(fountain, 1, sizes[room].x - 1, 1)};
    if (Uniform(coin) > 0.5f) {
      blocks[fountain_pos.y * pw + fountain_pos.x] = config.fountain_block;
    }
  }
  Key paths_rng = TakeKey(&rng);
  std::vector<float> included(8);
  included.back() = 1;
  for (int path = 0; path < 8; ++path) {
    const Pos source = positions[path];
    const Pos sink =
        positions[std::clamp(Choice(TakeKey(&paths_rng), included), 0, 7)];
    if (source.x != sink.x) {
      for (int x = std::min(source.x, sink.x); x <= std::max(source.x, sink.x);
           ++x) {
        int& block = blocks[source.y * pw + x];
        if (block == b::WALL) {
          block = b::PATH;
        }
      }
    }
    if (source.y != sink.y) {
      for (int y = std::min(source.y, sink.y); y <= std::max(source.y, sink.y);
           ++y) {
        int& block = blocks[y * pw + sink.x];
        if (block == b::WALL) {
          block = b::PATH;
        }
      }
    }
    included[path] = 1;
    paths_rng = TakeKey(&paths_rng);
  }
  blocks[(positions[0].y + 2) * pw + positions[0].x + 2] = config.special_block;
  const Key rare = TakeKey(&rng);
  const int base = level * game->Cells();
  std::vector<float> weights(game->Cells());
  int path_count = 0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int i = y * w + x;
      const int padded = (y + 10) * pw + x + 10;
      bool adjacent = false;
      for (Pos delta : {Pos{}, Pos{1, 0}, Pos{-1, 0}, Pos{0, 1}, Pos{0, -1}}) {
        const Pos pos{y + delta.y, x + delta.x};
        if (game->InBounds(pos) &&
            blocks[(pos.y + 10) * pw + pos.x + 10] != b::WALL) {
          adjacent = true;
        }
      }
      const bool is_rare = 1.0f - Uniform(rare, i) > 0.9f;
      int block = blocks[padded];
      if (!adjacent) {
        block = b::DARKNESS;
      } else if (block == b::WALL) {
        block = is_rare ? b::WALL_MOSS : b::WALL;
      } else if (block == b::PATH && items[padded] == i::NONE && is_rare) {
        block = config.rare_path_replacement_block;
      }
      s.map[base + i] = block;
      s.item_map[base + i] = items[padded];
      s.light_map[base + i] = 1;
      weights[i] = static_cast<float>(block == b::PATH);
      path_count += static_cast<int>(block == b::PATH);
    }
  }
  auto normalized = weights;
  for (auto& weight : normalized) {
    weight /= static_cast<float>(path_count);
  }
  const int down =
      std::clamp(Choice(TakeKey(&rng), normalized), 0, game->Cells() - 1);
  const int up =
      std::clamp(Choice(TakeKey(&rng), weights), 0, game->Cells() - 1);
  s.down_ladders[level] = {down / h, down % h};
  s.up_ladders[level] = {up / h, up % h};
  s.item_map[base + down] = i::LADDER_DOWN;
  s.item_map[base + up] = i::LADDER_UP;
}

void Game::FullWorld(Key rng) {
  state.player_position = {params.height / 2, params.width / 2};
  const std::array<int, 6> smooth_levels{0, 2, 5, 6, 7, 8};
  for (int i = 0; i < 6; ++i) {
    SmoothWorld(this, Split(rng, i + 1), full::ALL_SMOOTHGEN_CONFIGS[i],
                smooth_levels[i]);
  }
  rng = Split(rng, 0);
  const std::array<int, 3> dungeon_levels{1, 3, 4};
  for (int i = 0; i < 3; ++i) {
    Dungeon(this, Split(rng, i + 1), full::ALL_DUNGEON_CONFIGS[i],
            dungeon_levels[i]);
  }
  rng = Split(rng, 0);
  state.player_level = 0;
  for (auto* group :
       {&state.melee_mobs, &state.passive_mobs, &state.ranged_mobs,
        &state.mob_projectiles, &state.player_projectiles}) {
    for (auto& mob : *group) {
      mob.health = 1;
    }
  }
  std::fill(state.mob_projectile_directions.begin(),
            state.mob_projectile_directions.end(), Pos{1, 1});
  std::fill(state.player_projectile_directions.begin(),
            state.player_projectile_directions.end(), Pos{1, 1});
  const Key potion = Split(TakeKey(&rng), 1);
  std::iota(state.potion_mapping.begin(), state.potion_mapping.end(), 0);
  std::stable_sort(
      state.potion_mapping.begin(), state.potion_mapping.end(),
      [&](int a, int b) { return Bits(potion, a) < Bits(potion, b); });
  if (params.god_mode) {
    auto& inv = state.inventory;
    for (int* item : {&inv.wood, &inv.stone, &inv.coal, &inv.iron, &inv.diamond,
                      &inv.sapling, &inv.arrows, &inv.torches, &inv.ruby,
                      &inv.sapphire, &inv.books}) {
      *item = 99;
    }
    inv.pickaxe = inv.sword = 4;
    inv.bow = 1;
    inv.armour.fill(2);
    inv.potions.fill(99);
  }
  state.monsters_killed[0] = 10;
  state.boss_timesteps_to_spawn_this_round = full::BOSS_FIGHT_SPAWN_TURNS;
  state.state_rng = TakeKey(&rng);
}

}  // namespace craftax
