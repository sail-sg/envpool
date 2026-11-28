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
#include <array>
#include <bitset>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "envpool/gymnax/games.h"
#include "opencv2/imgproc.hpp"

namespace gymnax {
namespace {

int MoveX(int x, int a) { return std::clamp(x + kHorizontal[a], 0, 9); }

struct Breakout : Discrete {
  using Discrete::Discrete;
  int x, y, dx, dy, last_x, last_y, paddle;
  bool strike;
  std::bitset<100> bricks;

  void NewBricks() {
    for (int i = 10; i < 40; ++i) {
      bricks.set(i);
    }
  }
  void Reset() override {
    paddle = 4;
    last_x = x = 9 * Index(2);
    last_y = y = 3;
    dx = x == 0 ? 1 : -1;
    dy = 1;
    strike = false;
    NewBricks();
  }
  void Step(int a) override {
    paddle = MoveX(paddle, a);
    last_x = x;
    last_y = y;
    x += dx;
    y += dy;
    if (x < 0 || x > 9) {
      x = std::clamp(x, 0, 9);
      dx = -dx;
    }
    if (y < 0) {
      y = 0;
      dy = -dy;
    }
    bool hit = bricks[y * 10 + x];
    if (hit && !strike) {
      bricks.reset(y * 10 + x);
      reward = 1;
      dy = -dy;
      y = last_y;
    } else if (y == 9) {
      if (bricks.none()) {
        NewBricks();
      }
      if (last_x == paddle || x == paddle) {
        if (last_x != paddle) {
          dx = -dx;
        }
        dy = -dy;
        y = last_y;
      } else {
        terminated = true;
      }
    }
    strike = hit;
  }
  void Observe(float* obs) const override {
    Grid grid(obs, task);
    grid.Mark(paddle, 9, 0);
    grid.Mark(x, y, 1);
    grid.Mark(last_x, last_y, 2);
    for (int i = 10; i < 40; ++i) {
      if (bricks[i]) {
        grid.Mark(i % 10, i / 10, 3);
      }
    }
  }
};

struct Freeway : Discrete {
  using Discrete::Discrete;
  struct Car {
    int x, dir, speed, timer;
  };
  std::array<Car, 8> cars;
  int y, timer;

  void RandomizeCars() {
    for (auto& car : cars) {
      car.speed = 1 + Index(5);
      car.dir = 2 * Index(2) - 1;
      car.timer = car.speed;
    }
  }
  void Reset() override {
    y = 9;
    timer = 3;
    cars = {};
    RandomizeCars();
  }
  void Step(int a) override {
    if (timer == 0 && (a == 2 || a == 4)) {
      y = std::min(9, y + (a == 2 ? -1 : 1));
      timer = 3;
    }
    if (y == 0) {
      y = 9;
      reward = 1;
      RandomizeCars();
    }
    if (timer > 0) {
      --timer;
    }
    for (int i = 0; i < 8; ++i) {
      auto& car = cars[i];
      if (car.x == 4 && y == i + 1) {
        y = 9;
      }
      if (car.timer == 0) {
        car.x = (car.x + car.dir + 10) % 10;
        car.timer = car.speed;
        if (car.x == 4 && y == i + 1) {
          y = 9;
        }
      } else {
        --car.timer;
      }
    }
  }
  void Observe(float* obs) const override {
    Grid grid(obs, task);
    grid.Mark(4, y, 0);
    for (int i = 0; i < 8; ++i) {
      const auto& car = cars[i];
      grid.Mark(car.x, i + 1, 1);
      grid.Mark((car.x - car.dir + 10) % 10, i + 1, 1 + car.speed);
    }
  }
};

struct Asterix : Discrete {
  using Discrete::Discrete;
  struct Entity {
    int x, dir;
    bool gold;
    bool active;
  };
  std::array<Entity, 8> entities;
  int x, y, spawn, move, spawn_delay, move_delay;

  void Reset() override {
    x = y = 5;
    entities = {};
    spawn = spawn_delay = 10;
    move = move_delay = 5;
  }
  void Spawn() {
    std::array<int, 8> slots{0, 1, 2, 3, 4, 5, 6, 7};
    std::shuffle(slots.begin(), slots.end(), gen);
    // while_sample_slots tries seven lanes. When they are all occupied,
    // the last lane is overwritten by an inactive entity.
    int i = 1;
    while (i < 7 && entities[slots[i]].active) {
      ++i;
    }
    auto& e = entities[slots[i]];
    bool right = Index(2) == 1;
    bool gold = Index(3) == 0;
    e = {right ? 0 : 9, right ? 1 : -1, gold, !e.active};
    for (int j = 0; j < 8; ++j) {
      Record(j, slots[j]);
    }
    Record(8, static_cast<float>(right));
    Record(9, static_cast<float>(gold));
  }
  void Collide() {
    auto& e = entities[y - 1];
    if (e.active && e.x == x) {
      if (e.gold) {
        ++reward;
        e.active = false;
      } else {
        terminated = true;
      }
    }
  }
  void Step(int a) override {
    if (spawn == 0) {
      Spawn();
      spawn = spawn_delay;
    }
    x = MoveX(x, a);
    y = std::clamp(y + kVertical[a], 1, 8);
    Collide();
    if (move == 0) {
      for (auto& e : entities) {
        if (e.active) {
          e.x += e.dir;
          e.active = e.x >= 0 && e.x <= 9;
        }
      }
      Collide();
      move = move_delay;
    }
    --move;
    --spawn;
    // The ramp timer runs from 100 through -1, inclusive.
    if (time % 102 == 0 && spawn_delay > 1) {
      --spawn_delay;
      if (time % 204 == 0) {
        --move_delay;
      }
    }
  }
  void Observe(float* obs) const override {
    Grid grid(obs, task);
    grid.Mark(x, y, 0);
    for (int i = 0; i < 8; ++i) {
      const auto& e = entities[i];
      if (!e.active) {
        continue;
      }
      grid.Mark(e.x, i + 1, e.gold ? 3 : 1);
      int trail = e.x - e.dir;
      if (trail >= 0 && trail <= 9) {
        grid.Mark(trail, i + 1, 2);
      }
    }
  }
};

struct SpaceInvaders : Discrete {
  using Discrete::Discrete;
  std::bitset<100> aliens, bullets, enemy_bullets;
  int x, dir, interval, move, fire, enemy_fire;

  void NewWave() {
    for (int y = 0; y < 4; ++y) {
      for (int col = 2; col < 8; ++col) {
        aliens.set(y * 10 + col);
      }
    }
  }
  void Reset() override {
    x = 5;
    dir = -1;
    interval = move = 12;
    fire = 0;
    enemy_fire = 10;
    aliens.reset();
    bullets.reset();
    enemy_bullets.reset();
    NewWave();
  }
  void MoveAliens() {
    move = std::min(static_cast<int>(aliens.count()), interval);
    int edge = dir < 0 ? 0 : 9;
    for (int y = 0; y < 10; ++y) {
      if (aliens[y * 10 + edge]) {
        dir = -dir;
        if ((aliens >> 90).any()) {
          terminated = true;
        }
        aliens = (aliens << 10) | (aliens >> 90);
        return;
      }
    }
    aliens = dir < 0 ? aliens >> 1 : aliens << 1;
  }
  void Shoot() {
    // Shoot from the lowest alien in the closest column. Ties go left.
    int best = 10;
    int source = 0;
    for (int col = 0; col < 10; ++col) {
      if (std::abs(col - x) >= best) {
        continue;
      }
      for (int y = 9; y >= 0; --y) {
        if (aliens[y * 10 + col]) {
          best = std::abs(col - x);
          source = y * 10 + col;
          break;
        }
      }
    }
    enemy_bullets.set(source);
  }
  void Step(int a) override {
    x = MoveX(x, a);
    const int player = 90 + x;
    if (a == 5 && fire == 0) {
      bullets.set(player);
      fire = 5;
    }
    bullets >>= 10;
    enemy_bullets <<= 10;
    terminated = enemy_bullets[player] || aliens[player];
    if (move == 0) {
      MoveAliens();
      if (aliens[player]) {
        terminated = true;
      }
    }
    if (enemy_fire == 0) {
      Shoot();
      enemy_fire = 10;
    }
    auto hit = bullets & aliens;
    reward = static_cast<float>(hit.count());
    bullets ^= hit;
    aliens ^= hit;
    if (aliens.none()) {
      interval = std::max(6, interval - 1);
      NewWave();
    }
    --move;
    --enemy_fire;
    if (fire > 0) {
      --fire;
    }
  }
  void Observe(float* obs) const override {
    Grid grid(obs, task);
    grid.Mark(x, 9, 0);
    for (int i = 0; i < 100; ++i) {
      if (aliens[i]) {
        grid.Mark(i % 10, i / 10, 1);
        grid.Mark(i % 10, i / 10, dir < 0 ? 2 : 3);
      }
      if (bullets[i]) {
        grid.Mark(i % 10, i / 10, 4);
      }
      if (enemy_bullets[i]) {
        grid.Mark(i % 10, i / 10, 5);
      }
    }
  }
};

struct Pong : Discrete {
  using Discrete::Discrete;
  int x, y, last_x, last_y, dx, dy, left, right;

  void Reset() override {
    x = last_x = 20;
    y = last_y = left = right = 15;
    dx = 1;
    dy = 0;
  }
  void Step(int a) override {
    if (a != 0) {
      left = std::clamp(left + (a == 1 ? -1 : 1), 2, 27);
    }
    int up = std::max(2, right - 1);
    int down = std::min(27, right + 1);
    right = std::abs(y - up) < std::abs(y - down) ? up : down;
    last_x = x;
    last_y = y;
    x += dx;
    y += dy;
    // The pinned game reflects on only one vertical boundary.
    if (y >= 30) {
      y = 58 - y;
      dy = -dy;
    }
    bool hit_left = x <= 1 && std::abs(y - left) <= 2;
    bool hit_right = x > 38 && std::abs(y - right) <= 2;
    if (hit_left || hit_right) {
      x = hit_left ? 2 - x : 76 - x;
      int offset = y - (hit_left ? left : right);
      dx = -dx;
      dy = std::clamp(static_cast<int>(std::nearbyint(dy + offset / 2.0)), -3,
                      3);
    }
    terminated = x < 0 || x >= 40;
    reward = terminated ? 0 : 1;
  }
  void Observe(float* obs) const override {
    Grid grid(obs, task);
    grid.Mark(std::clamp(x, 0, 39), std::clamp(y, 0, 29), 1);
    grid.Mark(std::clamp(last_x, 0, 39), std::clamp(last_y, 0, 29), 2);
    for (int i = -2; i <= 2; ++i) {
      grid.Mark(0, left + i, 0);
      grid.Mark(39, right + i, 0);
    }
  }
};

}  // namespace

std::unique_ptr<Discrete> MakeArcade(const Task& task, int limit,
                                     std::mt19937& gen) {
  switch (task.kind) {
    case Kind::Asterix:
      return std::make_unique<Asterix>(task, limit, gen);
    case Kind::Breakout:
      return std::make_unique<Breakout>(task, limit, gen);
    case Kind::Freeway:
      return std::make_unique<Freeway>(task, limit, gen);
    case Kind::SpaceInvaders:
      return std::make_unique<SpaceInvaders>(task, limit, gen);
    case Kind::Pong:
      return std::make_unique<Pong>(task, limit, gen);
    case Kind::Seaquest:
      return MakeSeaquest(task, limit, gen);
    default:
      throw std::invalid_argument(task.id);
  }
}

std::pair<int, int> RenderSize(const Task& task) {
  if (task.kind == Kind::Pong) {
    return {400, 300};
  }
  if (task.colors != nullptr) {
    return {400, 400};
  }
  if (task.kind == Kind::MetaMaze) {
    return {440, 440};
  }
  if (task.kind == Kind::FourRooms) {
    return {520, 520};
  }
  if (task.kind == Kind::PointRobot) {
    return {500, 300};
  }
  throw std::runtime_error(std::string(task.id) + " has no RGB renderer");
}

void Discrete::Render(int width, int height, unsigned char* rgb) const {
  int rows = task.shape[0];
  int cols = task.shape[1];
  int channels = task.shape[2];
  std::vector<float> obs(rows * cols * channels);
  Observe(obs.data());
  cv::Mat image(rows, cols, CV_8UC3);
  for (int i = 0; i < rows * cols; ++i) {
    int index = 0;
    for (int c = 0; c < channels; ++c) {
      if (obs[i * channels + c] != 0) {
        index = c + 1;
      }
    }
    const auto& color = task.colors[index];
    std::copy(color.begin(), color.end(), image.data + 3 * i);
  }
  cv::Mat frame(height, width, CV_8UC3, rgb);
  cv::resize(image, frame, frame.size(), 0, 0, cv::INTER_NEAREST);
}

}  // namespace gymnax
