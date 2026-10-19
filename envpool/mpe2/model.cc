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

#include "envpool/mpe2/model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mpe2 {
namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr Color TreasureColor(int type) {
  constexpr std::array colors = {
      Color{0.212, 0.408, 0.776}, Color{0.945, 0.553, 0},
      Color{0.169, 0.627, 0.173}, Color{0.839, 0.149, 0.157},
      Color{0.580, 0.404, 0.741}, Color{0.549, 0.337, 0.294}};
  return colors[type];
}

Vec Difference(const Vec& a, const Vec& b) {
  return {a[0] - b[0], a[1] - b[1]};
}
double SquaredNorm(const Vec& v) { return v[0] * v[0] + v[1] * v[1]; }
double Norm(const Vec& v) { return std::sqrt(SquaredNorm(v)); }
double Distance(const Entity& a, const Entity& b) {
  return Norm(Difference(a.pos, b.pos));
}
bool Collision(const Entity& a, const Entity& b) {
  return Distance(a, b) < a.size + b.size;
}
double Boundary(double value) {
  double x = std::abs(value);
  if (x < 0.9) {
    return 0;
  }
  if (x < 1) {
    return (x - 0.9) * 10;
  }
  return std::min(std::exp(2 * x - 2), 10.0);
}
template <typename Values>
void Append(std::vector<double>* out, const Values& values) {
  out->insert(out->end(), values.begin(), values.end());
}
std::vector<int> Indices(int begin, int end, int excluded = -1) {
  std::vector<int> result;
  for (int i = begin; i < end; ++i) {
    if (i != excluded) {
      result.push_back(i);
    }
  }
  return result;
}
}  // namespace

int Options::Players() const {
  switch (task) {
    case Task::kSimple:
      return 1;
    case Task::kSimpleAdversary:
      return n + 1;
    case Task::kSimpleCrypto:
      return 3;
    case Task::kSimplePush:
    case Task::kSimpleReference:
    case Task::kSimpleSpeakerListener:
      return 2;
    case Task::kSimpleSpread:
    case Task::kSimpleFormation:
    case Task::kSimpleLine:
      return n;
    case Task::kSimpleTag:
    case Task::kSimpleWorldComm:
      return good + adversaries;
    case Task::kCollectTreasure:
      return collectors + deposits;
  }
  throw std::invalid_argument("Unknown MPE2 task");
}
int Options::Landmarks() const {
  switch (task) {
    case Task::kSimple:
    case Task::kSimpleFormation:
      return 1;
    case Task::kSimpleAdversary:
    case Task::kSimpleSpread:
      return n;
    case Task::kSimpleCrypto:
    case Task::kSimpleLine:
    case Task::kSimplePush:
      return 2;
    case Task::kSimpleReference:
    case Task::kSimpleSpeakerListener:
      return 3;
    case Task::kSimpleTag:
      return obstacles;
    case Task::kSimpleWorldComm:
      return obstacles + food + forests;
    case Task::kCollectTreasure:
      return treasures;
  }
  throw std::invalid_argument("Unknown MPE2 task");
}
int Options::Communication() const {
  if (task == Task::kSimpleReference) {
    return 10;
  }
  if (task == Task::kSimpleSpeakerListener) {
    return 3;
  }
  if (task == Task::kSimpleCrypto || task == Task::kSimpleWorldComm) {
    return 4;
  }
  if (task == Task::kSimple || task == Task::kSimpleFormation ||
      task == Task::kSimpleLine) {
    return 0;
  }
  return 2;
}
int Options::ActionSize(int player) const {
  bool stationary = task == Task::kSimpleCrypto ||
                    (task == Task::kSimpleSpeakerListener && player == 0);
  bool speaks = stationary || task == Task::kSimpleReference ||
                (task == Task::kSimpleWorldComm && player == 0);
  if (continuous) {
    return (stationary ? 0 : 5) + (speaks ? Communication() : 0);
  }
  return (stationary ? 1 : 5) * (speaks ? Communication() : 1);
}
int Options::ObservationSize(int player) const {
  int others = agent_neighbors < 0 ? Players() - 1 : agent_neighbors;
  int landmarks = landmark_neighbors < 0 ? Landmarks() : landmark_neighbors;
  switch (task) {
    case Task::kSimple:
      return 4;
    case Task::kSimpleAdversary:
      return 2 * (landmarks + others) + (player == 0 ? 0 : 2);
    case Task::kSimpleCrypto:
      return player == 0 ? 4 : 8;
    case Task::kSimpleFormation:
      return 6;
    case Task::kSimpleLine:
      return 8;
    case Task::kSimplePush:
      return player == 0 ? 8 : 19;
    case Task::kSimpleReference:
      return 21;
    case Task::kSimpleSpeakerListener:
      return player == 0 ? 3 : 11;
    case Task::kSimpleSpread:
      return 4 + 2 * landmarks + 4 * others;
    case Task::kSimpleTag:
      return 4 + 2 * landmarks + 2 * others +
             2 * (agent_neighbors < 0
                      ? good - static_cast<int>(player >= adversaries)
                      : others);
    case Task::kSimpleWorldComm:
      return 4 + 2 * Landmarks() + 2 * (Players() - 1) +
             2 * (good - static_cast<int>(player >= adversaries)) + forests +
             (player < adversaries ? 4 : 0);
    case Task::kCollectTreasure:
      return 4 + (player < collectors ? deposits : 0) +
             (Players() - 1) * (4 + 2 * deposits) + treasures * (2 + deposits);
  }
  throw std::invalid_argument("Unknown MPE2 task");
}
int Options::MaxObservationSize() const {
  int size = 0;
  for (int p = 0; p < Players(); ++p) {
    size = std::max(size, ObservationSize(p));
  }
  return size;
}
int Options::MaxActionSize() const {
  int size = 0;
  for (int p = 0; p < Players(); ++p) {
    size = std::max(size, ActionSize(p));
  }
  return size;
}
int Options::BenchmarkSize() const {
  if (!benchmark) {
    return 0;
  }
  switch (task) {
    case Task::kSimpleAdversary:
      return Landmarks() + 1;
    case Task::kSimpleCrypto:
      return 8;
    case Task::kSimpleSpeakerListener:
    case Task::kSimpleTag:
    case Task::kSimpleWorldComm:
      return 1;
    case Task::kSimpleSpread:
      return 4;
    case Task::kCollectTreasure:
      return 2;
    default:
      return 0;
  }
}
void Options::Validate() const {
  if (Players() < 1 || Landmarks() < 0 || n < 0 || good < 0 ||
      adversaries < 0 || obstacles < 0 || food < 0 || forests < 0 ||
      collectors < 0 || deposits < 0 || treasures < 0 ||
      !std::isfinite(local_ratio) || local_ratio < 0 || local_ratio > 1 ||
      agent_neighbors == 0 || agent_neighbors < -1 || landmark_neighbors == 0 ||
      landmark_neighbors < -1) {
    throw std::invalid_argument(
        "Invalid MPE2 population, neighbor count or local_ratio");
  }
  if ((task == Task::kSimpleAdversary && n < 1) ||
      (task == Task::kCollectTreasure &&
       (collectors < 1 || deposits < 1 || deposits > 6 || treasures < 1)) ||
      (task == Task::kSimpleWorldComm &&
       (good < 1 || adversaries < 1 || food < 1))) {
    throw std::invalid_argument(
        "MPE2 task requires nonempty agent/landmark groups");
  }
}

Model::Model(const Options& opts, std::uint32_t seed)
    : kOptions(opts), random(seed), respawn_random(seed) {
  int p = kOptions.Players();
  entities.resize(p + kOptions.Landmarks());
  rewards.resize(p);
  goals.resize(p, -1);
  expected_.resize(p);
  for (int i = 0; i < p; ++i) {
    auto& e = entities[i];
    e.movable = true;
    e.name = "agent_" + std::to_string(i);
    e.comm.resize(kOptions.Communication());
    switch (kOptions.task) {
      case Task::kSimpleAdversary:
        e.size = 0.15;
        e.name = i == 0 ? "adversary_0" : "agent_" + std::to_string(i - 1);
        break;
      case Task::kSimpleSpread:
        e.size = 0.15;
        e.collide = true;
        break;
      case Task::kSimpleFormation:
        e.collide = true;
        break;
      case Task::kSimpleLine:
        e.size = 0.03;
        e.collide = true;
        break;
      case Task::kSimpleCrypto:
        e.movable = false;
        e.silent = false;
        e.name = std::array<const char*, 3>{"eve_0", "bob_0", "alice_0"}[i];
        break;
      case Task::kSimplePush:
        e.collide = true;
        e.name = i == 0 ? "adversary_0" : "agent_0";
        break;
      case Task::kSimpleReference:
        e.silent = false;
        break;
      case Task::kSimpleSpeakerListener:
        e.movable = i != 0;
        e.silent = i != 0;
        e.size = 0.075;
        e.name = i == 0 ? "speaker_0" : "listener_0";
        break;
      case Task::kSimpleTag:
      case Task::kSimpleWorldComm: {
        bool adv = i < kOptions.adversaries;
        bool world = kOptions.task == Task::kSimpleWorldComm;
        e.collide = true;
        double good_size = world ? 0.045 : 0.05;
        e.size = adv ? 0.075 : good_size;
        e.acceleration = adv ? 3 : 4;
        e.speed = adv ? 1 : 1.3;
        e.silent = !world || i != 0;
        if (!adv) {
          e.name = "agent_" + std::to_string(i - kOptions.adversaries);
        } else if (world && i == 0) {
          e.name = "leadadversary_0";
        } else {
          e.name = "adversary_" + std::to_string(i - (world ? 1 : 0));
        }
        if (!adv && !world && kOptions.curriculum) {
          // MPE2 1.1.1 inherits BaseScenario's no-op stage setter: stage is
          // always zero, including after
          // set_curriculum_stage/advance_curriculum.
          double factor = 0.5;
          e.acceleration *= factor;
          e.speed *= factor;
        }
        break;
      }
      case Task::kCollectTreasure:
        e.acceleration = 1.5;
        e.speed = 1;
        e.name = i < kOptions.collectors
                     ? "collector_" + std::to_string(i)
                     : "deposit_" + std::to_string(i - kOptions.collectors);
        if (i >= kOptions.collectors) {
          e.mass = 2.25;
          e.size = 0.075;
        }
        break;
      default:
        break;
    }
  }
  for (int i = 0; i < kOptions.Landmarks(); ++i) {
    auto& e = Landmark(i);
    if (kOptions.task == Task::kSimpleAdversary) {
      e.size = 0.08;
    }
    if (kOptions.task == Task::kSimpleSpeakerListener) {
      e.size = 0.04;
    }
    if (kOptions.task == Task::kSimpleFormation) {
      e.size = 0.03;
    }
    if (kOptions.task == Task::kSimpleLine) {
      e.size = 0.02;
    }
    if (kOptions.task == Task::kCollectTreasure) {
      e.size = 0.025;
    }
    if (kOptions.task == Task::kSimpleTag ||
        kOptions.task == Task::kSimpleWorldComm) {
      e.size = 0.2;
      e.collide = i < kOptions.obstacles;
      if (i >= kOptions.obstacles) {
        e.size = i < kOptions.obstacles + kOptions.food ? 0.03 : 0.3;
      }
    }
  }
}

void Model::Reset() {
  respawn_random =
      random;  // Upstream deep-copies before consuming reset draws.
  std::fill(rewards.begin(), rewards.end(), 0);
  std::fill(goals.begin(), goals.end(), -1);
  std::fill(expected_.begin(), expected_.end(), Vec{});
  assignment_.clear();
  events_ = 0;
  key = 0;
  int p = kOptions.Players();
  int l = kOptions.Landmarks();
  Task t = kOptions.task;
  for (auto& e : entities) {
    e.vel = {};
    e.holding = -1;
    e.type = 0;
    e.alive = true;
    std::fill(e.comm.begin(), e.comm.end(), 0);
    e.color = {0.25, 0.25, 0.25};
  }
  if (t == Task::kSimpleAdversary || t == Task::kSimplePush ||
      t == Task::kSimpleCrypto) {
    int goal = random.Integer(l);
    std::fill(goals.begin(), goals.end(), goal);
  }
  if (t == Task::kSimpleReference) {
    goals[0] = random.Integer(l);
    goals[1] = random.Integer(l);
  }
  if (t == Task::kSimpleSpeakerListener) {
    goals[0] = random.Integer(l);
  }
  if (t == Task::kSimpleCrypto) {
    key = random.Integer(l);
  }
  for (int i = 0; i < p; ++i) {
    auto& e = entities[i];
    if (t == Task::kSimpleSpread || t == Task::kSimpleFormation ||
        t == Task::kSimpleLine) {
      e.color = {0.35, 0.35, 0.85};
    }
    if (t == Task::kSimpleAdversary) {
      e.color = i == 0 ? Color{0.85, 0.35, 0.35} : Color{0.35, 0.35, 0.85};
    }
    if (t == Task::kSimplePush || t == Task::kSimpleCrypto) {
      if (i == 0) {
        e.color = {0.75, 0.25, 0.25};
      }
      if (t == Task::kSimplePush && i == 1) {
        e.color[goals[i] + 1] += 0.5;
      }
      if (t == Task::kSimpleCrypto && i == 1) {
        e.color = {};
        e.color[goals[i]] = 1;
      }
    }
    if (t == Task::kSimpleTag) {
      e.color = i < kOptions.adversaries ? Color{0.85, 0.35, 0.35}
                                         : Color{0.35, 0.85, 0.35};
    }
    if (t == Task::kSimpleWorldComm) {
      e.color = i < kOptions.adversaries ? Color{0.95, 0.45, 0.45}
                                         : Color{0.45, 0.95, 0.45};
      if (i == 0) {
        for (double& c : e.color) {
          c -= 0.3;
        }
      }
    }
    if (t == Task::kCollectTreasure) {
      e.color = {0.85, 0.85, 0.85};
      if (i >= kOptions.collectors) {
        e.color = TreasureColor(i - kOptions.collectors);
        for (double& c : e.color) {
          c *= 0.35;
        }
      }
    }
    e.pos = {random.Uniform(-1, 1), random.Uniform(-1, 1)};
  }
  for (int i = 0; i < l; ++i) {
    auto& e = Landmark(i);
    if (t == Task::kSimple) {
      e.color = {0.75, 0.25, 0.25};
    }
    if (t == Task::kSimpleAdversary) {
      e.color =
          goals[0] == i ? Color{0.15, 0.65, 0.15} : Color{0.15, 0.15, 0.15};
    }
    if (t == Task::kSimplePush) {
      e.color = {0.1, 0.1, 0.1};
      e.color[i + 1] += 0.8;
    }
    if (t == Task::kSimpleCrypto) {
      e.color = {};
      e.color[i] = 1;
    }
    if (t == Task::kSimpleReference || t == Task::kSimpleSpeakerListener) {
      e.color = t == Task::kSimpleReference ? Color{0.25, 0.25, 0.25}
                                            : Color{0.15, 0.15, 0.15};
      e.color[i] += 0.5;
    }
    double bound = 1;
    if (t == Task::kSimpleTag || t == Task::kSimpleWorldComm) {
      bound = 0.9;
    } else if (t == Task::kSimpleFormation) {
      bound = 0.5;
    } else if (t == Task::kCollectTreasure) {
      bound = 0.95;
    }
    if (t == Task::kSimpleWorldComm && i >= kOptions.obstacles) {
      e.color = i < kOptions.obstacles + kOptions.food ? Color{0.15, 0.15, 0.65}
                                                       : Color{0.6, 0.9, 0.6};
    }
    if (t == Task::kCollectTreasure) {
      e.type = random.Integer(kOptions.deposits);
      e.color = TreasureColor(e.type);
    }
    if (t != Task::kSimpleLine) {
      e.pos = {random.Uniform(-bound, bound), random.Uniform(-bound, bound)};
    }
  }
  if (t == Task::kSimpleReference) {
    entities[0].color = Landmark(goals[1]).color;
    entities[1].color = Landmark(goals[0]).color;
  }
  if (t == Task::kSimpleSpeakerListener) {
    entities[1].color = Landmark(goals[0]).color;
    for (double& c : entities[1].color) {
      c += 0.45;
    }
  }
  if (t == Task::kSimpleWorldComm) {
    for (int i = kOptions.obstacles; i < l; ++i) {
      Landmark(i).pos = {random.Uniform(-0.9, 0.9), random.Uniform(-0.9, 0.9)};
    }
  }
  if (t == Task::kSimpleLine) {
    Landmark(0).pos = {random.Uniform(-0.25, 0.25),
                       random.Uniform(-0.25, 0.25)};
    double theta = random.Uniform(0, 2 * kPi);
    Vec direction;
    do {
      direction = {std::cos(theta), std::sin(theta)};
      for (int d = 0; d < 2; ++d) {
        Landmark(1).pos[d] = Landmark(0).pos[d] + 1.25 * direction[d];
      }
      theta += 5 * (kPi / 180);
    } while (std::abs(Landmark(1).pos[0]) >= 1 ||
             std::abs(Landmark(1).pos[1]) >= 1);
    double sep = p > 1 ? 1.25 / (p - 1) : 0;
    for (int i = 0; i < p; ++i) {
      for (int d = 0; d < 2; ++d) {
        expected_[i][d] = Landmark(0).pos[d] + i * sep * direction[d];
      }
    }
  }
  original_range = 0;
  for (const auto& e : entities) {
    for (double x : e.pos) {
      original_range = std::max(original_range, std::abs(x));
    }
  }
}

void Model::Physics(const std::vector<std::vector<double>>& actions) {
  int p = kOptions.Players();
  std::vector<Vec> force(entities.size());
  for (int i = 0; i < p; ++i) {
    auto& e = entities[i];
    int action = kOptions.continuous ? 0 : static_cast<int>(actions[i][0]);
    if (e.movable) {
      if (kOptions.continuous) {
        force[i] = {static_cast<float>(actions[i][2]) -
                        static_cast<float>(actions[i][1]),
                    static_cast<float>(actions[i][4]) -
                        static_cast<float>(actions[i][3])};
      } else {
        int move = action % 5;
        action /= 5;
        if (move == 1) {
          force[i][0] = -1;
        }
        if (move == 2) {
          force[i][0] = 1;
        }
        if (move == 3) {
          force[i][1] = -1;
        }
        if (move == 4) {
          force[i][1] = 1;
        }
      }
      for (double& f : force[i]) {
        f *= e.acceleration;
      }
    }
    if (!e.silent) {
      for (int c = 0; c < kOptions.Communication(); ++c) {
        e.comm[c] = kOptions.continuous ? actions[i][(e.movable ? 5 : 0) + c]
                                        : static_cast<double>(c == action);
      }
    }
  }
  for (std::size_t i = 0; i < entities.size(); ++i) {
    for (std::size_t j = i + 1; j < entities.size(); ++j) {
      const auto& a = entities[i];
      const auto& b = entities[j];
      if (!a.collide || !b.collide) {
        continue;
      }
      Vec delta = Difference(a.pos, b.pos);
      double dist = std::sqrt(SquaredNorm(delta));
      double x = -(dist - (a.size + b.size)) / 0.001;
      double penetration =
          (std::max(0.0, x) + std::log1p(std::exp(-std::abs(x)))) * 0.001;
      for (int d = 0; d < 2; ++d) {
        double f = 100 * delta[d] / dist * penetration;
        if (a.movable) {
          force[i][d] += f;
        }
        if (b.movable) {
          force[j][d] += -f;
        }
      }
    }
  }
  for (int i = 0; i < p; ++i) {
    auto& e = entities[i];
    if (!e.movable) {
      continue;
    }
    for (int d = 0; d < 2; ++d) {
      e.pos[d] += e.vel[d] * 0.1;
      e.vel[d] = e.vel[d] * 0.75;
      e.vel[d] += (force[i][d] / e.mass) * 0.1;
    }
    double speed = std::sqrt(SquaredNorm(e.vel));
    if (e.speed > 0 && speed > e.speed) {
      for (double& v : e.vel) {
        v = v / speed * e.speed;
      }
    }
  }
}

void Model::TreasureEvents() {
  events_ = 0;
  for (int i = 0; i < kOptions.treasures; ++i) {
    auto& lm = Landmark(i);
    if (lm.alive) {
      for (int j = 0; j < kOptions.collectors; ++j) {
        auto& a = entities[j];
        if (a.holding < 0 && Collision(a, lm)) {
          events_ += 5;
          lm.alive = false;
          a.holding = lm.type;
          a.color = lm.color;
          for (double& c : a.color) {
            c *= 0.85;
          }
          lm.pos = {-999, -999};
          break;
        }
      }
    } else if (respawn_random.Uniform() <= 1) {
      lm.pos = {respawn_random.Uniform(-0.95, 0.95),
                respawn_random.Uniform(-0.95, 0.95)};
      lm.type = respawn_random.Integer(kOptions.deposits);
      lm.color = TreasureColor(lm.type);
      lm.alive = true;
    }
  }
  for (int i = 0; i < kOptions.collectors; ++i) {
    auto& a = entities[i];
    if (a.holding >= 0 &&
        Collision(a, entities[kOptions.collectors + a.holding])) {
      events_ += 5;
      a.holding = -1;
      a.color = {0.85, 0.85, 0.85};
    }
  }
}

void Model::Assignment() {
  int n = kOptions.Players();
  if (kOptions.task == Task::kSimpleFormation) {
    double theta = 2 * kPi;
    for (int i = 0; i < n; ++i) {
      Vec v = Difference(entities[i].pos, Landmark(0).pos);
      double a = std::atan2(v[1], v[0]);
      if (a < 0) {
        a += 2 * kPi;
      }
      theta = std::min(theta, a);
    }
    double sep = 2 * kPi / n;
    for (int i = 0; i < n; ++i) {
      expected_[i] = {Landmark(0).pos[0] + 0.5 * std::cos(theta + i * sep),
                      Landmark(0).pos[1] + 0.5 * std::sin(theta + i * sep)};
    }
  }
  // Shortest augmenting paths solve the same minimum-distance assignment as
  // scipy.optimize.linear_sum_assignment, without a runtime solver dependency.
  std::vector<std::vector<double>> cost(n, std::vector<double>(n));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      cost[i][j] = Norm(Difference(entities[i].pos, expected_[j]));
    }
  }
  std::vector<double> u(n + 1);
  std::vector<double> v(n + 1);
  std::vector<int> row(n + 1);
  std::vector<int> way(n + 1);
  for (int i = 1; i <= n; ++i) {
    row[0] = i;
    int col = 0;
    std::vector<double> best(n + 1, std::numeric_limits<double>::infinity());
    std::vector<bool> used(n + 1);
    do {
      used[col] = true;
      int current = row[col];
      int next = 0;
      double delta = std::numeric_limits<double>::infinity();
      for (int j = 1; j <= n; ++j) {
        if (!used[j]) {
          double candidate = cost[current - 1][j - 1] - u[current] - v[j];
          if (candidate < best[j]) {
            best[j] = candidate;
            way[j] = col;
          }
          if (best[j] < delta) {
            delta = best[j];
            next = j;
          }
        }
      }
      for (int j = 0; j <= n; ++j) {
        if (used[j]) {
          u[row[j]] += delta;
          v[j] -= delta;
        } else {
          best[j] -= delta;
        }
      }
      col = next;
    } while (row[col] != 0);
    do {
      int prev = way[col];
      row[col] = row[prev];
      col = prev;
    } while (col != 0);
  }
  assignment_.resize(n);
  for (int j = 1; j <= n; ++j) {
    assignment_[row[j] - 1] = cost[row[j] - 1][j - 1];
  }
}

double Model::GlobalReward() const {
  double reward = 0;
  if (kOptions.task == Task::kSimpleReference) {
    for (int i = 0; i < kOptions.Players(); ++i) {
      reward += Reward(i);
    }
    return reward / kOptions.Players();
  }
  if (kOptions.task == Task::kSimpleSpread) {
    for (int i = 0; i < kOptions.Landmarks(); ++i) {
      double closest = std::numeric_limits<double>::infinity();
      for (int j = 0; j < kOptions.Players(); ++j) {
        closest = std::min(closest, Distance(entities[j], Landmark(i)));
      }
      reward -= closest;
    }
  } else {
    for (double distance : assignment_) {
      reward += std::clamp(distance, 0.0, 2.0);
    }
    reward = -reward / kOptions.Players();
  }
  return reward;
}

double Model::Reward(int player) const {
  const auto& a = entities[player];
  int p = kOptions.Players();
  Task t = kOptions.task;
  if (t == Task::kSimple) {
    return -SquaredNorm(Difference(a.pos, Landmark(0).pos));
  }
  if (t == Task::kSimpleReference) {
    return -Distance(entities[1 - player], Landmark(goals[player]));
  }
  if (t == Task::kSimpleSpeakerListener) {
    return -SquaredNorm(Difference(entities[1].pos, Landmark(goals[0]).pos));
  }
  if (t == Task::kSimpleFormation || t == Task::kSimpleLine) {
    return 0;
  }
  if (t == Task::kSimplePush) {
    double d = Distance(a, Landmark(goals[player]));
    return player == 0 ? Distance(entities[1], Landmark(goals[1])) - d : -d;
  }
  if (t == Task::kSimpleAdversary) {
    double adv = Distance(entities[0], Landmark(goals[0]));
    if (player == 0) {
      return -adv;
    }
    double closest = std::numeric_limits<double>::infinity();
    for (int i = 1; i < p; ++i) {
      closest = std::min(closest, Distance(entities[i], Landmark(goals[i])));
    }
    return -closest + adv;
  }
  if (t == Task::kSimpleCrypto) {
    auto error = [&](int i) {
      double result = 0;
      if (std::all_of(entities[i].comm.begin(), entities[i].comm.end(),
                      [](double x) { return x == 0; })) {
        return result;
      }
      for (int c = 0; c < 4; ++c) {
        double d = entities[i].comm[c] - (c == goals[player]);
        result += d * d;
      }
      return result;
    };
    return player == 0 ? -error(0) : error(0) - error(1);
  }
  if (t == Task::kSimpleSpread) {
    double r = 0;
    if (!kOptions.curriculum) {
      for (int i = 0; i < p; ++i) {
        if (i != player && Collision(a, entities[i])) {
          r -= 1;
        }
      }
    }
    return r;
  }
  if (t == Task::kSimpleTag || t == Task::kSimpleWorldComm) {
    double r = 0;
    bool world = t == Task::kSimpleWorldComm;
    if (player < kOptions.adversaries) {
      if (world) {
        double closest = std::numeric_limits<double>::infinity();
        for (int i = kOptions.adversaries; i < p; ++i) {
          closest = std::min(closest, Distance(a, entities[i]));
        }
        r -= 0.1 * closest;
      }
      for (int i = kOptions.adversaries; i < p; ++i) {
        for (int j = 0; j < kOptions.adversaries; ++j) {
          if (Collision(entities[i], entities[j])) {
            r += world ? 5 : 10;
          }
        }
      }
    } else {
      for (int i = 0; i < kOptions.adversaries; ++i) {
        if (Collision(a, entities[i])) {
          r -= world ? 5 : 10;
        }
      }
      for (double x : a.pos) {
        r -= (world ? 2 : 1) * Boundary(x);
      }
      if (world) {
        double closest = std::numeric_limits<double>::infinity();
        for (int i = kOptions.obstacles; i < kOptions.obstacles + kOptions.food;
             ++i) {
          if (Collision(a, Landmark(i))) {
            r += 2;
          }
          closest = std::min(closest, Distance(a, Landmark(i)));
        }
        r -= 0.05 * closest;
      }
    }
    return r;
  }
  double r = 0;
  double closest = std::numeric_limits<double>::infinity();
  if (player < kOptions.collectors) {
    for (int i = 0; i < kOptions.collectors; ++i) {
      if (i != player && Collision(a, entities[i])) {
        r -= 5;
      }
    }
    if (a.holding < 0) {
      for (int i = 0; i < kOptions.treasures; ++i) {
        if (Landmark(i).alive) {
          closest = std::min(closest, Distance(a, Landmark(i)));
        }
      }
    } else {
      closest = Distance(a, entities[kOptions.collectors + a.holding]);
    }
  } else {
    for (int i = 0; i < kOptions.collectors; ++i) {
      if (entities[i].holding == player - kOptions.collectors) {
        closest = std::min(closest, Distance(a, entities[i]));
      }
    }
    if (!std::isfinite(closest)) {
      Vec centroid{};
      for (int i = 0; i < kOptions.collectors; ++i) {
        for (int d = 0; d < 2; ++d) {
          centroid[d] += entities[i].pos[d];
        }
      }
      for (double& x : centroid) {
        x /= kOptions.collectors;
      }
      closest = Norm(Difference(centroid, a.pos));
    }
  }
  if (std::isfinite(closest)) {
    r -= 0.1 * closest;
  }
  return r + events_;
}

void Model::Step(const std::vector<std::vector<double>>& actions) {
  Physics(actions);
  Task t = kOptions.task;
  if (t == Task::kCollectTreasure) {
    TreasureEvents();
  }
  if (t == Task::kSimpleFormation || t == Task::kSimpleLine) {
    Assignment();
  }
  bool mixed = t == Task::kSimpleSpread || t == Task::kSimpleReference ||
               t == Task::kSimpleFormation || t == Task::kSimpleLine;
  double global = mixed ? GlobalReward() : 0;
  double ratio = t == Task::kSimpleFormation || t == Task::kSimpleLine
                     ? 0
                     : kOptions.local_ratio;
  for (int i = 0; i < kOptions.Players(); ++i) {
    double r = Reward(i);
    rewards[i] = mixed ? global * (1 - ratio) + r * ratio : r;
  }
}

bool Model::Terminal() const {
  if (!kOptions.terminate_on_success) {
    return false;
  }
  if (kOptions.task == Task::kSimpleFormation ||
      kOptions.task == Task::kSimpleLine) {
    return !assignment_.empty() &&
           std::all_of(assignment_.begin(), assignment_.end(),
                       [](double d) { return d < 0.05; });
  }
  if (kOptions.task == Task::kSimpleSpread) {
    for (int l = 0; l < kOptions.Landmarks(); ++l) {
      bool covered = false;
      for (int p = 0; p < kOptions.Players(); ++p) {
        covered = covered || Distance(entities[p], Landmark(l)) < 0.1;
      }
      if (!covered) {
        return false;
      }
    }
    return true;
  }
  if (kOptions.task == Task::kSimpleTag) {
    for (int prey = kOptions.adversaries; prey < kOptions.Players(); ++prey) {
      bool caught = false;
      for (int adv = 0; adv < kOptions.adversaries; ++adv) {
        caught = caught || Collision(entities[prey], entities[adv]);
      }
      if (!caught) {
        return false;
      }
    }
    return true;
  }
  return false;
}

std::vector<double> Model::Observation(int player) const {
  std::vector<double> out;
  const auto& a = entities[player];
  int p = kOptions.Players();
  Task t = kOptions.task;
  auto pos = [&](int i) { Append(&out, Difference(entities[i].pos, a.pos)); };
  auto onehot = [&](int index, int count) {
    for (int i = 0; i < count; ++i) {
      out.push_back(i == index);
    }
  };
  auto nearest = [&](std::vector<int> ids, int count) {
    if (count >= 0) {
      std::stable_sort(ids.begin(), ids.end(), [&](int i, int j) {
        return Distance(a, entities[i]) < Distance(a, entities[j]);
      });
      ids.resize(count, -1);
    }
    return ids;
  };
  auto positions = [&](const std::vector<int>& ids) {
    for (int i : ids) {
      if (i < 0) {
        Append(&out, Vec{});
      } else {
        pos(i);
      }
    }
  };
  auto all_landmarks = Indices(p, static_cast<int>(entities.size()));
  auto others = Indices(0, p, player);
  if (t == Task::kSimpleCrypto) {
    if (player == 2) {
      onehot(goals[player], 4);
      onehot(key, 4);
    } else {
      if (player == 1) {
        onehot(key, 4);
      }
      Append(&out, entities[2].comm);
    }
    return out;
  }
  if (t == Task::kSimpleSpeakerListener) {
    if (player == 0) {
      Append(&out, Landmark(goals[0]).color);
    } else {
      Append(&out, a.vel);
      positions(all_landmarks);
      Append(&out, entities[0].comm);
    }
    return out;
  }
  if (t == Task::kSimpleAdversary) {
    if (player != 0) {
      pos(p + goals[player]);
    }
    positions(nearest(all_landmarks, kOptions.landmark_neighbors));
    positions(nearest(others, kOptions.agent_neighbors));
    return out;
  }
  if (t == Task::kCollectTreasure) {
    Append(&out, a.pos);
    Append(&out, a.vel);
    if (player < kOptions.collectors) {
      onehot(a.holding, kOptions.deposits);
    }
    for (int i : nearest(others, p - 1)) {
      pos(i);
      Append(&out, entities[i].vel);
      onehot(i < kOptions.collectors ? -1 : i - kOptions.collectors,
             kOptions.deposits);
      onehot(i < kOptions.collectors ? entities[i].holding : -1,
             kOptions.deposits);
    }
    std::stable_sort(
        all_landmarks.begin(), all_landmarks.end(), [&](int i, int j) {
          return (entities[i].alive ? Distance(a, entities[i]) : 1e6) <
                 (entities[j].alive ? Distance(a, entities[j]) : 1e6);
        });
    for (int i : all_landmarks) {
      if (entities[i].alive) {
        pos(i);
      } else {
        Append(&out, Vec{});
      }
      onehot(entities[i].alive ? entities[i].type : -1, kOptions.deposits);
    }
    return out;
  }
  Append(&out, a.vel);
  if (t == Task::kSimple) {
    positions(all_landmarks);
    return out;
  }
  if (t == Task::kSimplePush) {
    if (player != 0) {
      pos(p + goals[player]);
      Append(&out, a.color);
    }
    positions(all_landmarks);
    if (player != 0) {
      for (int i : all_landmarks) {
        Append(&out, entities[i].color);
      }
    }
    positions(others);
    return out;
  }
  if (t == Task::kSimpleReference) {
    positions(all_landmarks);
    Append(&out, Landmark(goals[player]).color);
    Append(&out, entities[1 - player].comm);
    return out;
  }
  Append(&out, a.pos);
  if (t == Task::kSimpleFormation || t == Task::kSimpleLine) {
    positions(all_landmarks);
    return out;
  }
  positions(nearest(all_landmarks, t == Task::kSimpleWorldComm
                                       ? -1
                                       : kOptions.landmark_neighbors));
  if (t == Task::kSimpleSpread || t == Task::kSimpleTag) {
    auto selected = nearest(others, kOptions.agent_neighbors);
    positions(selected);
    for (int i : selected) {
      if (t == Task::kSimpleSpread) {
        if (i < 0) {
          Append(&out, Vec{});
        } else {
          Append(&out, entities[i].comm);
        }
      } else if (i >= kOptions.adversaries) {
        Append(&out, entities[i].vel);
      } else if (kOptions.agent_neighbors >= 0) {
        Append(&out, Vec{});
      }
    }
    return out;
  }
  std::vector<double> velocities;
  std::vector<double> forest;
  for (int i : others) {
    bool mine = false;
    bool theirs = false;
    bool shared = false;
    for (int f = kOptions.obstacles + kOptions.food; f < kOptions.Landmarks();
         ++f) {
      bool x = Collision(a, Landmark(f));
      bool y = Collision(entities[i], Landmark(f));
      mine = mine || x;
      theirs = theirs || y;
      shared = shared || (x && y);
    }
    const bool visible = shared || (!mine && !theirs) || player == 0;
    if (visible) {
      pos(i);
    } else {
      Append(&out, Vec{});
    }
    if (i >= kOptions.adversaries) {
      Append(&velocities, visible ? entities[i].vel : Vec{});
    }
  }
  for (int f = kOptions.obstacles + kOptions.food; f < kOptions.Landmarks();
       ++f) {
    forest.push_back(Collision(a, Landmark(f)) ? 1 : -1);
  }
  if (player < kOptions.adversaries) {
    Append(&out, velocities);
    Append(&out, forest);
    Append(&out, entities[0].comm);
  } else {
    Append(&out, forest);
    Append(&out, velocities);
  }
  return out;
}

std::vector<double> Model::Benchmark(int player) const {
  if (!kOptions.benchmark) {
    return {};
  }
  const auto& a = entities[player];
  int p = kOptions.Players();
  switch (kOptions.task) {
    case Task::kSimpleSpeakerListener:
      return {Reward(player)};
    case Task::kSimpleAdversary: {
      std::vector<double> out;
      if (player != 0) {
        for (int i = 0; i < kOptions.Landmarks(); ++i) {
          out.push_back(SquaredNorm(Difference(a.pos, Landmark(i).pos)));
        }
      }
      out.push_back(
          SquaredNorm(Difference(a.pos, Landmark(goals[player]).pos)));
      return out;
    }
    case Task::kSimpleCrypto: {
      auto out = a.comm;
      for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<double>(i == goals[player]));
      }
      return out;
    }
    case Task::kSimpleTag:
    case Task::kSimpleWorldComm: {
      double count = 0;
      if (player < kOptions.adversaries) {
        for (int i = kOptions.adversaries; i < p; ++i) {
          count += static_cast<double>(Collision(a, entities[i]));
        }
      }
      return {count};
    }
    case Task::kSimpleSpread: {
      double collisions = 0;
      double occupied = 0;
      double distance = 0;
      double reward = 0;
      for (int i = 0; i < kOptions.Landmarks(); ++i) {
        double closest = std::numeric_limits<double>::infinity();
        for (int j = 0; j < p; ++j) {
          closest = std::min(closest, Distance(entities[j], Landmark(i)));
        }
        distance += closest;
        reward -= closest;
        occupied += static_cast<double>(closest < 0.1);
      }
      for (int i = 0; i < p; ++i) {
        if (i != player && Collision(a, entities[i])) {
          reward -= 1;
          collisions += 1;
        }
      }
      return {reward, collisions, distance, occupied};
    }
    case Task::kCollectTreasure: {
      double count = 0;
      double alive = 0;
      for (int i = 0; i < kOptions.treasures; ++i) {
        alive += static_cast<double>(Landmark(i).alive);
      }
      if (player < kOptions.collectors) {
        count = a.holding;
      } else {
        for (int i = 0; i < kOptions.collectors; ++i) {
          count += static_cast<double>(entities[i].holding ==
                                       player - kOptions.collectors);
        }
      }
      return {count, alive};
    }
    default:
      return {};
  }
}

std::vector<double> Model::Snapshot() const {
  std::vector<double> out;
  for (const auto& e : entities) {
    Append(&out, e.pos);
    Append(&out, e.vel);
    Append(&out, e.color);
    out.push_back(static_cast<double>(e.alive));
    out.push_back(e.holding);
    out.push_back(e.type);
    // Landmarks have no communication channel, but use the same record width.
    for (int i = 0; i < kOptions.Communication(); ++i) {
      out.push_back(i < static_cast<int>(e.comm.size()) ? e.comm[i] : 0);
    }
  }
  Append(&out, goals);
  out.push_back(key);
  out.push_back(original_range);
  for (const auto& v : expected_) {
    Append(&out, v);
  }
  return out;
}

}  // namespace mpe2
