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

#ifndef ENVPOOL_GYMNAX_GAMES_H_
#define ENVPOOL_GYMNAX_GAMES_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace gymnax {

enum class Kind;
using Color = std::array<unsigned char, 3>;
inline constexpr float kMax = std::numeric_limits<float>::max();

struct Task {
  const char* id;
  const char* pool;
  Kind kind;
  int limit;
  std::vector<int> shape;
  int actions;
  float low, high, force;
  std::vector<int> minimal;
  const Color* colors;
};

#include "third_party/gymnax/tasks.inc"

struct Mnist {
  explicit Mnist(const std::string& dir);
  std::vector<std::int8_t> images;
  std::vector<std::uint8_t> labels;
};

struct Episode {
  Episode(const Task& task, int limit, std::mt19937& gen)
      : task(task), gen(gen), kLimit(limit) {}

  const Task& task;
  std::mt19937& gen;
  const int kLimit;
  int time = 0;
  float reward = 0;
  bool terminated = false;

  int Index(int n) { return std::uniform_int_distribution<int>(0, n - 1)(gen); }
  float Uniform(float low, float high) {
    return std::uniform_real_distribution<float>(low, high)(gen);
  }

  // Latent task parameters and exogenous events used by the step oracle.
#ifdef ENVPOOL_TEST
  std::array<float, 11> hidden{};
#endif
  void Record(int i, float value) {
#ifdef ENVPOOL_TEST
    hidden[i] = value;
#endif
  }
};

struct Discrete : Episode {
  static constexpr const char* kName = "Discrete";
  using Episode::Episode;
  virtual ~Discrete() = default;
  virtual void Reset() = 0;
  virtual void Step(int action) = 0;
  virtual void Observe(float* obs) const = 0;
  virtual void Render(int width, int height, unsigned char* rgb) const;
};

// Full MinAtar action order: stay, left, up, right, down, fire.
inline constexpr std::array<int, 6> kHorizontal{0, -1, 0, 1, 0, 0};
inline constexpr std::array<int, 6> kVertical{0, 0, -1, 0, 1, 0};

// MinAtar and Pong return channels-last binary feature maps.
struct Grid {
  Grid(float* output, const Task& task)
      : data(output), cols(task.shape[1]), channels(task.shape[2]) {
    std::fill_n(data, task.shape[0] * cols * channels, 0);
  }
  void Mark(int x, int y, int channel) {
    data[(y * cols + x) * channels + channel] = 1;
  }
  float* data;
  int cols, channels;
};

std::unique_ptr<Discrete> MakeDiscrete(const Task& task, int limit,
                                       std::mt19937& gen, const Mnist* data);
std::unique_ptr<Discrete> MakeArcade(const Task& task, int limit,
                                     std::mt19937& gen);
std::unique_ptr<Discrete> MakeSeaquest(const Task& task, int limit,
                                       std::mt19937& gen);
std::pair<int, int> RenderSize(const Task& task);

struct Robot : Episode {
  static constexpr const char* kName = "Robot";
  using Episode::Episode;
  using Vec = std::array<float, 2>;
  Vec pos, vel, goal;
  std::array<Vec, 5> urchins;
  Vec OnCircle(float radius);
  void Reset();
  void Step(const float* action);
  void Observe(float* obs) const;
  void Render(int width, int height, unsigned char* rgb) const;
};

struct FourRooms : Episode {
  static constexpr const char* kName = "FourRooms";
  using Episode::Episode;
  int pos;
  void Reset();
  void Step(int action);
  void Observe(int* obs) const;
  void Render(int width, int height, unsigned char* rgb) const;
};

struct Snake : Discrete {
  static constexpr const char* kName = "Snake";
  Snake(const Task& task, int limit, std::mt19937& gen, int height, int width)
      : Discrete(task, limit, gen), kCols(width), board(height * width) {}
  const int kCols;
  std::vector<int> board;
  int head, direction, apple, length;
  void PlaceApple();
  void Reset() override;
  void Step(int action) override;
  void Observe(float* obs) const override;
};

}  // namespace gymnax

#endif  // ENVPOOL_GYMNAX_GAMES_H_
