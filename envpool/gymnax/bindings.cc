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

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/py_envpool.h"
#include "envpool/gymnax/games.h"

namespace gymnax {

// Several upstream space declarations exclude values produced by get_obs.
Spec<float> FloatSpec(const Task& t, int limit) {
  switch (t.kind) {
    case Kind::MemoryChain:
      return Spec<float>(t.shape, {-1, t.high});
    case Kind::MNISTBandit:
      return Spec<float>(t.shape, {-128.0f / 255, t.high});
    case Kind::MetaMaze: {
      std::vector<float> high(15, 1);
      high[13] = 10;
      high[14] = static_cast<float>(limit);
      return {t.shape, {std::vector<float>(15), high}};
    }
    case Kind::BernoulliBandit:
    case Kind::GaussianBandit: {
      const bool gaussian = t.kind == Kind::GaussianBandit;
      return Spec<float>(t.shape,
                         {{gaussian ? -kMax : 0, 0, 0, -1},
                          {gaussian ? kMax : 2, 1, 1, limit / 50.0f - 1}});
    }
    default:
      return Spec<float>(t.shape, {t.low, t.high});
  }
}

template <typename Game>
struct Functions {
  static auto DefaultConfig() {
    auto config = MakeDict("task"_.Bind(std::string()));
    if constexpr (std::is_same_v<Game, Snake>) {
      return ConcatDict(config,
                        MakeDict("width"_.Bind(10), "height"_.Bind(10)));
    } else if constexpr (std::is_same_v<Game, Discrete>) {
      return ConcatDict(config, MakeDict("use_minimal_action_set"_.Bind(true)));
    } else {
      return config;
    }
  }

  static const Task& Lookup(const std::string& name) {
    for (const Task& t : kTasks) {
      if (name == t.id && std::string_view(t.pool) == Game::kName) {
        return t;
      }
    }
    throw std::invalid_argument("Invalid Gymnax task for this pool: " + name);
  }

  template <class Config>
  static auto StateSpec(const Config& c) {
    const Task& t = Lookup(c["task"_]);
    const int limit = c["max_episode_steps"_];
    if (limit <= 0) {
      throw std::invalid_argument("max_episode_steps must be positive");
    }
    auto common = MakeDict(
#ifdef ENVPOOL_TEST
        "info:hidden"_.Bind(Spec<float>({11})),
#endif
        "terminated"_.Bind(Spec<bool>({})));
    if constexpr (std::is_same_v<Game, Snake>) {
      int rows = c["height"_];
      int cols = c["width"_];
      if (rows < 2 || cols < 2 ||
          rows > std::numeric_limits<int>::max() / 3 / cols) {
        throw std::invalid_argument("Invalid Snake board dimensions");
      }
      return ConcatDict(
          common, MakeDict("obs"_.Bind(Spec<float>({rows, cols, 3}, {0, 1})),
                           "info:length"_.Bind(Spec<int>({}))));
    } else if constexpr (std::is_same_v<Game, FourRooms>) {
      return ConcatDict(common,
                        MakeDict("obs"_.Bind(Spec<int>(t.shape, {1, 11}))));
    } else {
      return ConcatDict(common, MakeDict("obs"_.Bind(FloatSpec(t, limit))));
    }
  }

  template <class Config>
  static auto ActionSpec(const Config& c) {
    const Task& t = Lookup(c["task"_]);
    if constexpr (std::is_same_v<Game, Robot>) {
      return MakeDict(
          "action"_.Bind(Spec<float>({-1, t.actions}, {-t.force, t.force})));
    } else {
      int count = t.actions;
      if constexpr (std::is_same_v<Game, Discrete>) {
        if (!c["use_minimal_action_set"_]) {
          if (t.minimal.empty()) {
            throw std::invalid_argument("Only MinAtar has a full action set");
          }
          count = 6;
        }
      }
      return MakeDict("action"_.Bind(Spec<int>({-1}, {0, count - 1})));
    }
  }
};

template <typename Game>
struct GymnaxSpec : EnvSpec<Functions<Game>> {
  using Base = EnvSpec<Functions<Game>>;
  const Task& task;
  std::shared_ptr<const Mnist> mnist;

  explicit GymnaxSpec(const typename Base::ConfigValues& conf)
      : Base(conf), task(Functions<Game>::Lookup(this->config["task"_])) {
    if (task.kind == Kind::MNISTBandit) {
      mnist = std::make_shared<Mnist>(this->config["base_path"_] + "/gymnax/");
    }
  }
};

template <typename Game>
class GymnaxEnv : public TimeLimitEnv<GymnaxSpec<Game>>, public RenderableEnv {
  using Base = TimeLimitEnv<GymnaxSpec<Game>>;
  std::unique_ptr<Game> game_;

 public:
  GymnaxEnv(const GymnaxSpec<Game>& spec, int env_id) : Base(spec, env_id) {
    int limit = spec.config["max_episode_steps"_];
    if constexpr (std::is_same_v<Game, Discrete>) {
      game_ = MakeDiscrete(spec.task, limit, this->gen_, spec.mnist.get());
    } else if constexpr (std::is_same_v<Game, Snake>) {
      game_ = std::make_unique<Snake>(spec.task, limit, this->gen_,
                                      spec.config["height"_],
                                      spec.config["width"_]);
    } else {
      game_ = std::make_unique<Game>(spec.task, limit, this->gen_);
    }
  }

  void Reset() override {
    game_->time = 0;
    game_->terminated = false;
    game_->reward = 0;
#ifdef ENVPOOL_TEST
    game_->hidden.fill(0);
#endif
    game_->Reset();
    Write();
  }

  void Step(const typename Base::Action& action) override {
    game_->time = this->CurrentStep();
    game_->reward = 0;
    if constexpr (std::is_same_v<Game, Robot>) {
      game_->Step(static_cast<const float*>(action["action"_].Data()));
    } else {
      int a = action["action"_];
      const auto& bounds = this->spec_.action_spec["action"_].bounds;
      a = std::clamp(a, 0, std::get<1>(bounds));
      if constexpr (std::is_same_v<Game, Discrete>) {
        const auto& minimal = game_->task.minimal;
        if (!minimal.empty() && this->spec_.config["use_minimal_action_set"_]) {
          a = minimal[a];
        }
      }
      game_->Step(a);
    }
    Write();
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    auto [w, h] = gymnax::RenderSize(game_->task);
    return {width > 0 ? width : w, height > 0 ? height : h};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    if (this->CurrentStep() < 0) {
      throw std::runtime_error("reset before rendering");
    }
    game_->Render(width, height, rgb);
  }

 private:
  void Write() {
    auto state = this->Transition(game_->reward, game_->terminated);
    using Value =
        std::conditional_t<std::is_same_v<Game, FourRooms>, int, float>;
    game_->Observe(static_cast<Value*>(state["obs"_].Data()));
    if constexpr (std::is_same_v<Game, Snake>) {
      state["info:length"_] = game_->length;
    }
#ifdef ENVPOOL_TEST
    std::copy(game_->hidden.begin(), game_->hidden.end(),
              static_cast<float*>(state["info:hidden"_].Data()));
#endif
  }
};

using DiscreteEnvSpec = PyEnvSpec<GymnaxSpec<Discrete>>;
using DiscreteEnvPool = PyEnvPool<AsyncEnvPool<GymnaxEnv<Discrete>>>;
using RobotEnvSpec = PyEnvSpec<GymnaxSpec<Robot>>;
using RobotEnvPool = PyEnvPool<AsyncEnvPool<GymnaxEnv<Robot>>>;
using SnakeEnvSpec = PyEnvSpec<GymnaxSpec<Snake>>;
using SnakeEnvPool = PyEnvPool<AsyncEnvPool<GymnaxEnv<Snake>>>;
using FourRoomsEnvSpec = PyEnvSpec<GymnaxSpec<FourRooms>>;
using FourRoomsEnvPool = PyEnvPool<AsyncEnvPool<GymnaxEnv<FourRooms>>>;

PYBIND11_MODULE(gymnax_envpool, m) {
  REGISTER(m, DiscreteEnvSpec, DiscreteEnvPool)
  REGISTER(m, RobotEnvSpec, RobotEnvPool)
  REGISTER(m, SnakeEnvSpec, SnakeEnvPool)
  REGISTER(m, FourRoomsEnvSpec, FourRoomsEnvPool)
  py::list tasks;
  for (const auto& t : kShared) {
    tasks.append(py::make_tuple(t.id, t.module, t.pool, t.limit));
  }
  for (const auto& t : kTasks) {
    tasks.append(py::make_tuple(t.id, "gymnax", t.pool, t.limit));
  }
  m.attr("TASKS") = tasks;
}

}  // namespace gymnax
