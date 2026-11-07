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
#include "envpool/craftax/noise.h"
#include "third_party/craftax/constants.h"

namespace craftax {

void Game::ClassicWorld() {
  namespace b = classic::block;
  const int h = params.height;
  const int w = params.width;
  auto water =
      Noise(rng_, h, w, h / 16, w / 16, params.fractal_noise_angles[0]);
  auto mountain =
      Noise(rng_, h, w, h / 16, w / 16, params.fractal_noise_angles[1]);
  const auto paths =
      Noise(rng_, h, w, h / 8, w / 2, params.fractal_noise_angles[2]);
  const auto trees =
      Noise(rng_, h, w, h / 4, w / 4, params.fractal_noise_angles[3]);
  state.player_position = {h / 2, w / 2};
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int i = y * w + x;
      const float proximity =
          std::min(1.0f, Distance({y, x}, state.player_position) / 5.0f);
      water[i] = water[i] + proximity - 1.0f;
      mountain[i] = mountain[i] + 0.05f;
      mountain[i] = mountain[i] + proximity - 1.0f;
      int block = water[i] > 0.7f ? b::WATER : b::GRASS;
      if (water[i] > 0.6f && water[i] < 0.75f && block != b::WATER) {
        block = b::SAND;
      }
      if (mountain[i] > 0.7f) {
        block = b::STONE;
      }
      if (mountain[i] > 0.7f && (paths[i] > 0.8f || paths[x * w + y] > 0.8f)) {
        block = b::PATH;
      }
      if (mountain[i] > 0.85f && water[i] > 0.4f) {
        block = b::PATH;
      }
      if (block == b::STONE && rng_.Uniform() < 0.04f) {
        block = b::COAL;
      }
      if (block == b::STONE && rng_.Uniform() < 0.03f) {
        block = b::IRON;
      }
      if (block == b::STONE && mountain[i] > 0.8f && rng_.Uniform() < 0.005f) {
        block = b::DIAMOND;
      }
      if (block == b::GRASS && trees[i] > 0.5f && rng_.Uniform() > 0.8f) {
        block = b::TREE;
      }
      if (mountain[i] > 0.85f && trees[i] > 0.7f) {
        block = b::LAVA;
      }
      if ((Pos{y, x}) == state.player_position) {
        block = b::GRASS;
      }
      state.map[i] = block;
    }
  }
  const int chosen = SampleCell(
      rng_, Cells(), [&](int cell) { return state.map[cell] == b::STONE; });
  // The upstream empty-candidate fallback is the first cell.
  state.map[std::max(0, chosen)] =
      params.always_diamond ? b::DIAMOND : b::STONE;
  for (auto& mob : state.melee_mobs) {
    mob.health = 1;
  }
  for (auto& mob : state.passive_mobs) {
    mob.health = static_cast<float>(params.cow_health);
  }
  std::fill(state.mob_projectile_directions.begin(),
            state.mob_projectile_directions.end(), Pos{1, 1});
}

}  // namespace craftax
