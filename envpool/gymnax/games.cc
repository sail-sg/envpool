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

#include "envpool/gymnax/games.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "envpool/utils/grid_render.h"
#include "zlib.h"

namespace gymnax {
namespace {

// The two files have big-endian IDX headers followed by one byte per value.
void ReadIdx(const std::string& file, std::initializer_list<int> header,
             void* dest, int size) {
  gzFile stream = gzopen(file.c_str(), "rb");
  if (stream == nullptr) {
    throw std::runtime_error("Cannot open " + file);
  }
  bool valid = true;
  for (int field : header) {
    for (int shift = 24; shift >= 0; shift -= 8) {
      int byte = gzgetc(stream);
      if (byte != ((field >> shift) & 0xff)) {
        valid = false;
      }
    }
  }
  valid = valid && gzread(stream, dest, size) == size;
  int trailing = gzgetc(stream);
  valid = valid && trailing == -1 && gzeof(stream) != 0;
  if (gzclose(stream) != Z_OK || !valid) {
    throw std::runtime_error("Invalid MNIST data in " + file);
  }
}

struct MnistBandit : Discrete {
  MnistBandit(const Task& task, int limit, std::mt19937& gen, const Mnist& data)
      : Discrete(task, limit, gen), data(data) {}
  const Mnist& data;
  int index;

  void Reset() override {
    index = Index(static_cast<int>(data.labels.size()));
    Record(0, index);
  }
  void Step(int a) override {
    reward = a == data.labels[index] ? 1 : -1;
    terminated = true;
  }
  void Observe(float* obs) const override {
    // Gymnax's loader interprets the raw pixels as signed int8.
    if (time == 0) {
      for (int i = 0; i < 784; ++i) {
        obs[i] = data.images[index * 784 + i] / 255.0f;
      }
    } else {
      std::fill_n(obs, 784, 0);
    }
  }
};

struct DeepSea : Discrete {
  using Discrete::Discrete;
  int col;
  void Reset() override { col = 0; }
  void Step(int a) override {
    const bool right = a == kDeepSea[8 * (time - 1) + col] - '0';
    if (right) {
      reward = (col == 7 ? 1 : 0) - 0.01f / 8;
    }
    col = std::clamp(col + (right ? 1 : -1), 0, 7);
    terminated = time == 8;
  }
  void Observe(float* obs) const override {
    std::fill_n(obs, 64, 0);
    if (!terminated) {
      obs[8 * time + col] = 1;
    }
  }
};

struct MemoryChain : Discrete {
  using Discrete::Discrete;
  int context;
  void Reset() override { context = Index(2); }
  void Step(int a) override {
    terminated = time == 6;
    if (terminated) {
      reward = a == context ? 1 : -1;
    }
  }
  void Observe(float* obs) const override {
    // step_env returns the observation from before the action.
    int t = std::max(time - 1, 0);
    obs[0] = 1 - t / 5.0f;
    obs[1] = 0;
    obs[2] = t == 0 ? 2 * context - 1 : 0;
  }
};

struct UmbrellaChain : Discrete {
  using Discrete::Discrete;
  int need, have;
  void Reset() override {
    need = Index(2);
    have = Index(2);
  }
  void Step(int a) override {
    if (time == 1) {
      have = a;
    }
    terminated = time == 10;
    if (terminated) {
      reward = need == have ? 1 : -1;
    } else {
      int coin = Index(2);
      Record(0, coin);
      reward = 2 * coin - 1;
    }
  }
  void Observe(float* obs) const override {
    obs[0] = need;
    obs[1] = have;
    obs[2] = 1 - time / 10.0f;
  }
};

struct DiscountingChain : Discrete {
  using Discrete::Discrete;
  int context;
  void Reset() override { context = -1; }
  void Step(int a) override {
    static constexpr std::array<int, 5> delays{1, 3, 10, 30, 100};
    if (time == 1) {
      context = a;
    }
    if (time == delays[context]) {
      reward = context == 0 ? 1.1f : 1;
    }
  }
  void Observe(float* obs) const override {
    obs[0] = context;
    obs[1] = static_cast<float>(time) / kLimit;
  }
};

struct SimpleBandit : Discrete {
  using Discrete::Discrete;
  std::array<float, 11> arms;
  void Reset() override {
    for (int i = 0; i < 11; ++i) {
      arms[i] = i / 10.0f;
    }
    std::shuffle(arms.begin(), arms.end(), gen);
    for (int i = 0; i < 11; ++i) {
      Record(i, arms[i]);
    }
  }
  void Step(int a) override {
    reward = arms[a];
    terminated = true;
  }
  void Observe(float* obs) const override { obs[0] = 1; }
};

struct TwoArmBandit : Discrete {
  using Discrete::Discrete;
  float mean;
  int last;
  std::normal_distribution<float> normal;
  void Reset() override {
    last = 0;
    if (task.kind == Kind::BernoulliBandit) {
      mean = Index(2) == 0 ? 0.9f : 0.1f;
    } else {
      mean = normal(gen) - 1;
    }
    Record(0, mean);
  }
  void Step(int a) override {
    last = a;
    if (task.kind == Kind::BernoulliBandit) {
      float draw = Uniform(0, 1);
      Record(1, draw);
      reward = static_cast<float>(draw < (a == 0 ? mean : 1 - mean));
    } else {
      float noise = normal(gen);
      Record(1, noise);
      reward = a == 0 ? 0 : mean + 0.1f * noise;
    }
  }
  void Observe(float* obs) const override {
    obs[0] = reward;
    obs[1] = 1 - last;
    obs[2] = last;
    obs[3] = time / 50.0f - 1;
  }
};

constexpr std::array<int, 4> kRow{-1, 0, 1, 0};
constexpr std::array<int, 4> kCol{0, 1, 0, -1};
constexpr int kRoomsGoal = 8 * 13 + 9;

int Move(const char* walls, int cols, int pos, int a) {
  int next = pos + kRow[a] * cols + kCol[a];
  return walls[next] == '#' ? pos : next;
}

struct MetaMaze : Discrete {
  using Discrete::Discrete;
  int pos, goal, last;
  int OpenCell() {
    int cell;
    do {
      cell = Index(11 * 11);
    } while (kMetaMaze[cell] == '#');
    return cell;
  }
  void Reset() override {
    last = 0;
    pos = 5 * 11 + 5;
    goal = OpenCell();
    Record(0, goal);
    Record(1, pos);
  }
  void Step(int a) override {
    last = a;
    pos = Move(kMetaMaze, 11, pos, a);
    if (pos == goal) {
      reward = 10;
      pos = OpenCell();
    }
    // Termination is evaluated after relocating the agent.
    terminated = pos == goal;
    Record(1, pos);
  }
  void Observe(float* obs) const override {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        *obs++ = static_cast<float>(kMetaMaze[pos + dy * 11 + dx] == '#');
      }
    }
    for (int a = 0; a < 4; ++a) {
      *obs++ = static_cast<float>(a == last);
    }
    *obs++ = reward;
    *obs = time;
  }
  void Render(int w, int h, unsigned char* rgb) const override {
    rendering::RenderBoard(11, 11, kMetaMaze, {{pos, 'A'}, {goal, 'G'}}, w, h,
                           rgb);
  }
};

}  // namespace

Mnist::Mnist(const std::string& dir) : images(60000 * 784), labels(60000) {
  ReadIdx(dir + "train-images-idx3-ubyte.gz", {2051, 60000, 28, 28},
          images.data(), 60000 * 784);
  ReadIdx(dir + "train-labels-idx1-ubyte.gz", {2049, 60000}, labels.data(),
          60000);
}

void FourRooms::Reset() { pos = 4 * 13 + 1; }

void FourRooms::Step(int action) {
  // Resampling includes the original direction, so the failure rate is 1/3.
  int alternate = Index(9) < 4 ? Index(4) : -1;
  Record(0, alternate);
  pos = Move(kFourRooms, 13, pos, alternate < 0 ? action : alternate);
  terminated = pos == kRoomsGoal;
  reward = static_cast<float>(terminated);
}

void FourRooms::Observe(int* obs) const {
  obs[0] = pos / 13;
  obs[1] = pos % 13;
  obs[2] = kRoomsGoal / 13;
  obs[3] = kRoomsGoal % 13;
}

void FourRooms::Render(int w, int h, unsigned char* rgb) const {
  rendering::RenderBoard(13, 13, kFourRooms, {{pos, 'A'}, {kRoomsGoal, 'G'}}, w,
                         h, rgb);
}

void Snake::PlaceApple() {
  int skip = Index(static_cast<int>(board.size()) - length);
  for (int i = 0; i < static_cast<int>(board.size()); ++i) {
    if (board[i] == 0 && skip-- == 0) {
      apple = i;
      return;
    }
  }
}

void Snake::Reset() {
  std::fill(board.begin(), board.end(), 0);
  head = Index(static_cast<int>(board.size()));
  direction = Index(4);
  length = 1;
  board[head] = 1;
  PlaceApple();
  Record(0, direction);
}

void Snake::Step(int a) {
  if (length == 1 || (a + 2) % 4 != direction) {
    direction = a;
  }
  int row = head / kCols + kRow[direction];
  int col = head % kCols + kCol[direction];
  bool wall = row < 0 || row >= static_cast<int>(board.size()) / kCols ||
              col < 0 || col >= kCols;
  if (!wall) {
    head = row * kCols + col;
  }
  bool ate = !wall && head == apple;
  if (ate) {
    ++length;
  } else {
    for (int& age : board) {
      age = std::max(0, age - 1);
    }
  }
  bool dead = wall || board[head] > 0;
  board[head] = length;
  reward = static_cast<int>(ate) - static_cast<int>(dead);
  terminated = dead || length == static_cast<int>(board.size());
  if (ate && !terminated) {
    PlaceApple();
  }
}

void Snake::Observe(float* obs) const {
  for (int i = 0; i < static_cast<int>(board.size()); ++i) {
    *obs++ = static_cast<float>(board[i] > 0 && i != head);
    *obs++ = static_cast<float>(i == head);
    *obs++ = static_cast<float>(i == apple);
  }
}

std::unique_ptr<Discrete> MakeDiscrete(const Task& task, int limit,
                                       std::mt19937& gen, const Mnist* data) {
  switch (task.kind) {
    case Kind::MNISTBandit:
      return std::make_unique<MnistBandit>(task, limit, gen, *data);
    case Kind::DeepSea:
      return std::make_unique<DeepSea>(task, limit, gen);
    case Kind::MemoryChain:
      return std::make_unique<MemoryChain>(task, limit, gen);
    case Kind::DiscountingChain:
      return std::make_unique<DiscountingChain>(task, limit, gen);
    case Kind::UmbrellaChain:
      return std::make_unique<UmbrellaChain>(task, limit, gen);
    case Kind::SimpleBandit:
      return std::make_unique<SimpleBandit>(task, limit, gen);
    case Kind::MetaMaze:
      return std::make_unique<MetaMaze>(task, limit, gen);
    case Kind::GaussianBandit:
    case Kind::BernoulliBandit:
      return std::make_unique<TwoArmBandit>(task, limit, gen);
    default:
      return MakeArcade(task, limit, gen);
  }
}

}  // namespace gymnax
