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
// Scenario semantics adapted from Farama MPE2 v1.1.1 (MIT; see LICENSE).

#ifndef ENVPOOL_MPE2_MODEL_H_
#define ENVPOOL_MPE2_MODEL_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "envpool/mpe2/random.h"
#include "third_party/mpe2/tasks.h"

namespace mpe2 {

using Vec = std::array<double, 2>;
using Color = std::array<double, 3>;

struct Options {
  Task task;
  int n, good, adversaries, obstacles, food, forests, collectors, deposits,
      treasures;
  int agent_neighbors{-1}, landmark_neighbors{-1};
  double local_ratio{0.5};
  bool continuous{false}, rescaling{true}, curriculum{false},
      terminate_on_success{false}, benchmark{false};

  int Players() const;
  int Landmarks() const;
  int Communication() const;
  int ObservationSize(int player) const;
  int ActionSize(int player) const;
  int MaxObservationSize() const;
  int MaxActionSize() const;
  int BenchmarkSize() const;
  void Validate() const;
};

struct Entity {
  Vec pos{}, vel{};
  Color color{0.25, 0.25, 0.25};
  std::vector<double> comm;
  std::string name;
  double size{0.05}, mass{1}, acceleration{5}, speed{0};
  bool movable{false}, collide{false}, silent{true}, alive{true};
  int holding{-1}, type{0};
};

class Model {
 public:
  const Options kOptions;
  std::vector<Entity> entities;
  std::vector<double> rewards;
  std::vector<int> goals;
  int key{0};
  double original_range{1};
  Random random, respawn_random;

  Model(const Options& opts, std::uint32_t seed);
  void Reset();
  void Step(const std::vector<std::vector<double>>& actions);
  bool Terminal() const;
  std::vector<double> Observation(int player) const;
  std::vector<double> Benchmark(int player) const;
  std::vector<double> Snapshot() const;

 private:
  std::vector<Vec> expected_;
  std::vector<double> assignment_;
  double events_{0};
  void Physics(const std::vector<std::vector<double>>& actions);
  void TreasureEvents();
  void Assignment();
  double Reward(int player) const;
  double GlobalReward() const;
  Entity& Landmark(int i) { return entities[kOptions.Players() + i]; }
  const Entity& Landmark(int i) const {
    return entities[kOptions.Players() + i];
  }
};

void Render(const Model& model, int width, int height, std::uint8_t* rgb);

}  // namespace mpe2
#endif  // ENVPOOL_MPE2_MODEL_H_
