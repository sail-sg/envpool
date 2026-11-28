// Copyright 2026 Garena Online Private Limited
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Native adaptation of RobertTLange/gymnax 1.1.0 (Apache-2.0).

#include <algorithm>
#include <bitset>
#include <memory>
#include <vector>

#include "envpool/gymnax/games.h"

namespace gymnax {
namespace {

struct Entity {
  int x, y, dir;
  int move = 0, fire = 0;

  bool At(const Entity& other) const { return x == other.x && y == other.y; }
};

using Entities = std::vector<Entity>;

void Append(Entities& list, Entity entity) {
  if (list.size() < 100) {
    list.push_back(entity);
  }
}

void Move(Entities& list, int interval) {
  for (auto& e : list) {
    if (e.move == 0) {
      e.x += e.dir;
      e.move = interval;
    } else {
      --e.move;
    }
  }
  list.erase(std::remove_if(list.begin(), list.end(),
                            [](const Entity& e) { return e.x < 0 || e.x > 9; }),
             list.end());
}

struct Seaquest : Discrete {
  using Discrete::Discrete;
  Entity player;
  Entities fish, subs, divers, bullets, enemy_bullets;
  int oxygen, rescued, spawn, diver_spawn, spawn_delay, move_delay, ramp;

  void Reset() override {
    player = {5, 0, -1};
    fish.clear();
    subs.clear();
    divers.clear();
    bullets.clear();
    enemy_bullets.clear();
    oxygen = 200;
    rescued = ramp = 0;
    spawn = spawn_delay = 20;
    diver_spawn = 30;
    move_delay = 5;
  }

  void Spawn() {
    if (spawn == 0) {
      bool right = Index(2) == 1;
      bool sub = Index(3) == 0;
      int y = 1 + Index(8);
      Entity e{right ? 0 : 9, y, right ? 1 : -1, move_delay, 10};
      Record(0, static_cast<float>(right));
      Record(1, static_cast<float>(sub));
      Record(2, y);
      auto opposing = [&](const Entity& other) {
        return other.y == e.y && other.dir != e.dir;
      };
      if (std::none_of(fish.begin(), fish.end(), opposing) &&
          std::none_of(subs.begin(), subs.end(), opposing)) {
        Append(sub ? subs : fish, e);
      }
      spawn = spawn_delay;
    }
    if (diver_spawn == 0) {
      bool right = Index(2) == 1;
      int y = 1 + Index(8);
      Record(3, static_cast<float>(right));
      Record(4, y);
      Append(divers, {right ? 0 : 9, y, right ? 1 : -1, 5});
      diver_spawn = 30;
    }
    --spawn;
    --diver_spawn;
  }

  void Hit(Entities& targets) {
    // Per cell the last bullet consumes the first target, once per pass.
    std::bitset<100> hit;
    for (int i = static_cast<int>(bullets.size()) - 1; i >= 0; --i) {
      const Entity& b = bullets[i];
      int cell = b.y * 10 + b.x;
      if (hit[cell]) {
        continue;
      }
      auto target = std::find_if(targets.begin(), targets.end(),
                                 [&](const Entity& e) { return e.At(b); });
      if (target != targets.end()) {
        hit.set(cell);
        targets.erase(target);
        bullets.erase(bullets.begin() + i);
        ++reward;
      }
    }
  }

  void PickUp() {
    auto it = divers.begin();
    while (it != divers.end() && rescued < 6) {
      if (player.At(*it)) {
        ++rescued;
        it = divers.erase(it);
      } else {
        ++it;
      }
    }
  }

  void MoveEnemies(Entities& enemies, int interval) {
    auto collision = [&](const Entity& e) { return player.At(e); };
    if (std::any_of(enemies.begin(), enemies.end(), collision)) {
      terminated = true;
    }
    Move(enemies, interval);
    if (std::any_of(enemies.begin(), enemies.end(), collision)) {
      terminated = true;
    }
  }

  void Surface() {
    if (rescued == 0) {
      terminated = true;
    }
    if (rescued == 6) {
      const int bonus = oxygen / 20;
      reward += bonus;
      // Gymnax decrements the counter even after unloading a full rescue.
      rescued = -1;
    } else {
      --rescued;
    }
    oxygen = 200;
    if (spawn_delay > 1) {
      --spawn_delay;
    }
    if (move_delay > 2 && ramp % 2 == 1) {
      --move_delay;
    }
    ++ramp;
  }

  void Step(int a) override {
    Spawn();
    bool underwater = player.y > 0;
    if (kHorizontal[a] != 0) {
      player.dir = kHorizontal[a];
      player.x = std::clamp(player.x + player.dir, 0, 9);
    }
    player.y = std::clamp(player.y + kVertical[a], 0, 8);
    if (a == 5 && player.fire == 0) {
      Append(bullets, {player.x, player.y, player.dir});
      player.fire = 5;
    }
    if (player.fire > 0) {
      --player.fire;
    }

    // Collisions are evaluated both before and after targets move.
    Move(bullets, 0);
    Hit(fish);
    Hit(subs);
    PickUp();
    Move(divers, 5);
    PickUp();
    MoveEnemies(subs, move_delay);
    Hit(subs);
    for (auto& sub : subs) {
      if (sub.fire == 0) {
        Append(enemy_bullets, {sub.x, sub.y, sub.dir});
        sub.fire = 10;
      } else {
        --sub.fire;
      }
    }
    MoveEnemies(enemy_bullets, 0);
    MoveEnemies(fish, move_delay);
    Hit(fish);

    if (oxygen <= 0) {
      terminated = true;
    }
    if (player.y > 0) {
      --oxygen;
    } else if (underwater) {
      Surface();
    }
  }

  void Observe(float* obs) const override {
    Grid grid(obs, task);
    grid.Mark(player.x, player.y, 0);
    int back = player.x - player.dir;
    if (back < 10) {
      grid.Mark((back + 10) % 10, player.y, 1);
    }
    for (int i = 0; i < oxygen / 20; ++i) {
      grid.Mark(i, 9, 7);
    }
    for (int i = 0; i < rescued; ++i) {
      grid.Mark(8 - i, 9, 8);
    }
    for (const auto& b : bullets) {
      grid.Mark(b.x, b.y, 2);
    }
    for (const auto& b : enemy_bullets) {
      grid.Mark(b.x, b.y, 4);
    }
    auto draw = [&](const Entities& list, int channel) {
      for (const auto& e : list) {
        grid.Mark(e.x, e.y, channel);
        int trail = e.x - e.dir;
        if (trail >= 0 && trail <= 9) {
          grid.Mark(trail, e.y, 3);
        }
      }
    };
    draw(fish, 5);
    draw(subs, 6);
    draw(divers, 9);
  }
};

}  // namespace

std::unique_ptr<Discrete> MakeSeaquest(const Task& task, int limit,
                                       std::mt19937& gen) {
  return std::make_unique<Seaquest>(task, limit, gen);
}

}  // namespace gymnax
