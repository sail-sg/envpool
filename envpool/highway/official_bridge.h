/*
 * Copyright 2026 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// TODO: replace this official highway-env bridge with a pure C++ port. It
// intentionally lives behind the Highway module and is used only for upstream
// HighwayEnv task families that have not been ported yet.

#ifndef ENVPOOL_HIGHWAY_OFFICIAL_BRIDGE_H_
#define ENVPOOL_HIGHWAY_OFFICIAL_BRIDGE_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace highway {
namespace official {
namespace py = pybind11;

py::object MakeOfficialEnv(py::module_ gym,
                           const std::string& official_env_id) {
  // highway-env calls reset() from AbstractEnv::__init__ before EnvPool can
  // pass its deterministic seed. Keep that constructor reset cheap/reliable for
  // the intersection generator, then restore the official default config before
  // EnvPool's first seeded Reset().
  py::object wrapper;
  if (official_env_id.rfind("intersection-", 0) == 0) {
    py::dict constructor_config;
    constructor_config["initial_vehicle_count"] = 0;
    constructor_config["spawn_probability"] = 0.0;
    wrapper = gym.attr("make")(official_env_id,
                               py::arg("config") = constructor_config);
  } else {
    wrapper = gym.attr("make")(official_env_id);
  }
  py::object env = wrapper.attr("unwrapped");
  if (official_env_id.rfind("intersection-", 0) == 0) {
    env.attr("configure")(env.attr("default_config")());
  }
  return env;
}

template <typename Dtype>
void AssignArray(TArray<Dtype>* dst, py::handle src, std::size_t count) {
  py::array_t<Dtype, py::array::c_style | py::array::forcecast> arr =
      py::array::ensure(src);
  if (!arr) {
    throw std::runtime_error("Cannot convert official highway-env array");
  }
  if (static_cast<std::size_t>(arr.size()) != count) {
    throw std::runtime_error("Unexpected official highway-env array size");
  }
  dst->Assign(arr.data(), count);
}

template <typename State, typename Dtype>
void AssignState(State* state, py::handle obs, std::size_t count) {
  auto arr = (*state)["obs"_];
  AssignArray(&arr, obs, count);
}

template <typename State>
void AssignGoalState(State* state, py::handle obs) {
  py::dict dict = py::reinterpret_borrow<py::dict>(obs);
  auto observation = (*state)["obs:observation"_];
  auto achieved_goal = (*state)["obs:achieved_goal"_];
  auto desired_goal = (*state)["obs:desired_goal"_];
  AssignArray(&observation, dict["observation"], 6);
  AssignArray(&achieved_goal, dict["achieved_goal"], 6);
  AssignArray(&desired_goal, dict["desired_goal"], 6);
}

template <typename State>
void AssignAttributesState(State* state, py::handle obs) {
  py::dict dict = py::reinterpret_borrow<py::dict>(obs);
  auto state_array = (*state)["obs:state"_];
  auto derivative = (*state)["obs:derivative"_];
  auto reference_state = (*state)["obs:reference_state"_];
  AssignArray(&state_array, dict["state"], 4);
  AssignArray(&derivative, dict["derivative"], 4);
  AssignArray(&reference_state, dict["reference_state"], 4);
}

template <typename State>
void AssignMultiAgentState(State* state, py::handle obs) {
  py::tuple tuple = py::reinterpret_borrow<py::tuple>(obs);
  auto player_obs = (*state)["obs:players.obs"_];
  for (py::ssize_t i = 0; i < tuple.size(); ++i) {
    py::array_t<float, py::array::c_style | py::array::forcecast> arr =
        py::array::ensure(tuple[i]);
    if (!arr || arr.size() != 25) {
      throw std::runtime_error("Unexpected multi-agent highway observation");
    }
    std::memcpy(player_obs[i].Data(), arr.data(), 25 * sizeof(float));
  }
}

template <typename EnvSpec, typename Adapter>
class OfficialEnv : public Env<EnvSpec>, public RenderableEnv {
 public:
  using Spec = EnvSpec;
  using Action = typename Env<EnvSpec>::Action;

  OfficialEnv(const Spec& spec, int env_id)
      : Env<EnvSpec>(spec, env_id),
        official_env_id_(spec.config["official_env_id"_]),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        done_(true) {
    py::gil_scoped_acquire gil;
    py::module_::import("highway_env");
    py::module_ gym = py::module_::import("gymnasium");
    env_ = std::make_unique<py::object>(
        MakeOfficialEnv(std::move(gym), official_env_id_));
    py::dict config;
    config["offscreen_rendering"] = true;
    try {
      env_->attr("configure")(config);
    } catch (const py::error_already_set&) {
      // Older tasks still inherit AbstractEnv after unwrapping but may not need
      // an explicit configure call.
    }
    env_->attr("render_mode") = "rgb_array";
    if (spec.config["render_agent"_]) {
      py::dict render_config;
      render_config["render_agent"] = true;
      try {
        env_->attr("configure")(render_config);
      } catch (const py::error_already_set&) {
      }
    }
  }

  ~OfficialEnv() override {
    py::gil_scoped_acquire gil;
    env_.reset();
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    py::gil_scoped_acquire gil;
    py::object ret = env_->attr("reset")(py::arg("seed") = this->seed_);
    py::object obs = ret.cast<py::tuple>()[0];
    done_ = false;
    WriteState(obs, Adapter::ResetReward());
  }

  void Step(const Action& action) override {
    py::gil_scoped_acquire gil;
    py::object ret = env_->attr("step")(Adapter::PyAction(action));
    py::tuple step = ret.cast<py::tuple>();
    py::object obs = step[0];
    bool terminated = step[2].cast<bool>();
    bool truncated = step[3].cast<bool>();
    done_ = terminated || truncated;
    WriteState(obs, step[1]);
  }

  [[nodiscard]] std::pair<int, int> RenderSize(int width,
                                               int height) const override {
    if (width > 0 && height > 0) {
      return {width, height};
    }
    py::gil_scoped_acquire gil;
    py::dict config = py::reinterpret_borrow<py::dict>(env_->attr("config"));
    return {config["screen_width"].cast<int>(),
            config["screen_height"].cast<int>()};
  }

  void Render(int width, int height, int camera_id,
              unsigned char* rgb) override {
    py::gil_scoped_acquire gil;
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> frame =
        py::array::ensure(env_->attr("render")());
    auto info = frame.request();
    if (info.ndim != 3 || info.shape[2] != 3) {
      throw std::runtime_error("official highway-env render must be HxWx3");
    }
    int src_h = static_cast<int>(info.shape[0]);
    int src_w = static_cast<int>(info.shape[1]);
    const auto* src = static_cast<const std::uint8_t*>(info.ptr);
    if (src_w == width && src_h == height) {
      std::memcpy(rgb, src, static_cast<std::size_t>(width) * height * 3);
      return;
    }
    for (int y = 0; y < height; ++y) {
      int sy = std::clamp(y * src_h / height, 0, src_h - 1);
      for (int x = 0; x < width; ++x) {
        int sx = std::clamp(x * src_w / width, 0, src_w - 1);
        std::memcpy(rgb + 3 * (y * width + x), src + 3 * (sy * src_w + sx), 3);
      }
    }
  }

 protected:
  [[nodiscard]] int CurrentMaxEpisodeSteps() const override {
    return max_episode_steps_;
  }

 private:
  std::string official_env_id_;
  int max_episode_steps_;
  bool done_;
  std::unique_ptr<py::object> env_;

  void WriteState(py::handle obs, py::handle reward) {
    auto state = this->Allocate(Adapter::kPlayerNum);
    Adapter::WriteReward(&state, reward);
    Adapter::WriteObservation(&state, obs);
    Adapter::WriteInfo(&state, *env_);
  }
};

template <typename Fns, typename Adapter>
using OfficialEnvSpec = EnvSpec<Fns>;

template <typename Spec, typename Adapter>
using OfficialEnvPool = AsyncEnvPool<OfficialEnv<Spec, Adapter>>;

class DiscreteActionAdapter {
 public:
  static constexpr int kPlayerNum = 1;

  template <typename Action>
  static py::object PyAction(const Action& action) {
    return py::int_(static_cast<int>(action["action"_]));
  }

  static py::object ResetReward() { return py::float_(0.0); }

  template <typename State>
  static void WriteReward(State* state, py::handle reward) {
    (*state)["reward"_][0] = reward.cast<float>();
  }
};

class ContinuousAction2Adapter : public DiscreteActionAdapter {
 public:
  static constexpr int kPlayerNum = 1;

  template <typename Action>
  static py::object PyAction(const Action& action) {
    py::array_t<float> arr(2);
    auto data = arr.mutable_unchecked<1>();
    data(0) = action["action"_][0];
    data(1) = action["action"_][1];
    return arr;
  }
};

class ContinuousAction1Adapter : public DiscreteActionAdapter {
 public:
  static constexpr int kPlayerNum = 1;

  template <typename Action>
  static py::object PyAction(const Action& action) {
    py::array_t<float> arr(1);
    arr.mutable_at(0) = action["action"_][0];
    return arr;
  }
};

class MultiAgentActionAdapter {
 public:
  static constexpr int kPlayerNum = 2;

  template <typename Action>
  static py::object PyAction(const Action& action) {
    py::tuple tuple(2);
    tuple[0] = py::int_(static_cast<int>(action["players.action"_][0]));
    tuple[1] = py::int_(static_cast<int>(action["players.action"_][1]));
    return tuple;
  }

  static py::object ResetReward() { return py::float_(0.0); }

  template <typename State>
  static void WriteReward(State* state, py::handle reward) {
    float scalar_reward = reward.cast<float>();
    auto rewards = (*state)["reward"_];
    rewards[0] = scalar_reward;
    rewards[1] = scalar_reward;
  }
};

template <int Vehicles, int Features>
class KinematicsAdapter : public DiscreteActionAdapter {
 public:
  template <typename State>
  static void WriteObservation(State* state, py::handle obs) {
    AssignState<State, float>(state, obs, Vehicles * Features);
  }

  template <typename State>
  static void WriteInfo(State* state, py::handle env) {
    auto speed = (*state)["info:speed"_];
    auto crashed = (*state)["info:crashed"_];
    py::object vehicle = env.attr("vehicle");
    speed = vehicle.attr("speed").cast<float>();
    crashed = vehicle.attr("crashed").cast<bool>();
  }
};

template <int Horizon>
class TTCAdapter : public DiscreteActionAdapter {
 public:
  template <typename State>
  static void WriteObservation(State* state, py::handle obs) {
    AssignState<State, float>(state, obs, 3 * 3 * Horizon);
  }

  template <typename State>
  static void WriteInfo(State* state, py::handle env) {
    auto speed = (*state)["info:speed"_];
    auto crashed = (*state)["info:crashed"_];
    py::object vehicle = env.attr("vehicle");
    speed = vehicle.attr("speed").cast<float>();
    crashed = vehicle.attr("crashed").cast<bool>();
  }
};

class Kinematics8ContinuousAdapter : public ContinuousAction2Adapter {
 public:
  template <typename State>
  static void WriteObservation(State* state, py::handle obs) {
    AssignState<State, float>(state, obs, 5 * 8);
  }

  template <typename State>
  static void WriteInfo(State* state, py::handle env) {
    auto speed = (*state)["info:speed"_];
    auto crashed = (*state)["info:crashed"_];
    py::object vehicle = env.attr("vehicle");
    speed = vehicle.attr("speed").cast<float>();
    crashed = vehicle.attr("crashed").cast<bool>();
  }
};

class GoalAdapter : public ContinuousAction2Adapter {
 public:
  template <typename State>
  static void WriteObservation(State* state, py::handle obs) {
    AssignGoalState(state, obs);
  }

  template <typename State>
  static void WriteInfo(State* state, py::handle env) {
    auto is_success = (*state)["info:is_success"_];
    py::object parking_obs =
        env.attr("observation_type_parking").attr("observe")();
    py::dict dict = py::reinterpret_borrow<py::dict>(parking_obs);
    is_success =
        env.attr("_is_success")(dict["achieved_goal"], dict["desired_goal"])
            .cast<bool>();
  }
};

class AttributesAdapter : public ContinuousAction1Adapter {
 public:
  template <typename State>
  static void WriteObservation(State* state, py::handle obs) {
    AssignAttributesState(state, obs);
  }

  template <typename State>
  static void WriteInfo(State* state, py::handle env) {}
};

class OccupancyAdapter : public ContinuousAction1Adapter {
 public:
  template <typename State>
  static void WriteObservation(State* state, py::handle obs) {
    AssignState<State, float>(state, obs, 2 * 12 * 12);
  }

  template <typename State>
  static void WriteInfo(State* state, py::handle env) {
    auto speed = (*state)["info:speed"_];
    auto crashed = (*state)["info:crashed"_];
    py::object vehicle = env.attr("vehicle");
    speed = vehicle.attr("speed").cast<float>();
    crashed = vehicle.attr("crashed").cast<bool>();
  }
};

class MultiAgentAdapter : public MultiAgentActionAdapter {
 public:
  template <typename State>
  static void WriteObservation(State* state, py::handle obs) {
    AssignMultiAgentState(state, obs);
  }

  template <typename State>
  static void WriteInfo(State* state, py::handle env) {
    auto speed = (*state)["info:players.speed"_];
    auto crashed = (*state)["info:players.crashed"_];
    py::list vehicles = env.attr("controlled_vehicles");
    for (py::ssize_t i = 0; i < vehicles.size(); ++i) {
      speed[i] = vehicles[i].attr("speed").cast<float>();
      crashed[i] = vehicles[i].attr("crashed").cast<bool>();
    }
  }
};

template <typename ObsSpec>
class OfficialBaseFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("official_env_id"_.Bind(std::string("highway-v0")),
                    "render_agent"_.Bind(true));
  }
};

class OfficialKinematics5Fns : public OfficialBaseFns<void> {
 public:
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const float inf = std::numeric_limits<float>::infinity();
    return MakeDict("obs"_.Bind(Spec<float>({5, 5}, {-inf, inf})),
                    "info:speed"_.Bind(Spec<float>({})),
                    "info:crashed"_.Bind(Spec<bool>({})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(MarkDiscrete(Spec<int>({-1}, {0, 4}))));
  }
};

class OfficialKinematics7Action5Fns : public OfficialBaseFns<void> {
 public:
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const float inf = std::numeric_limits<float>::infinity();
    return MakeDict("obs"_.Bind(Spec<float>({15, 7}, {-inf, inf})),
                    "info:speed"_.Bind(Spec<float>({})),
                    "info:crashed"_.Bind(Spec<bool>({})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(MarkDiscrete(Spec<int>({-1}, {0, 4}))));
  }
};

class OfficialKinematics7Action3Fns : public OfficialBaseFns<void> {
 public:
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const float inf = std::numeric_limits<float>::infinity();
    return MakeDict("obs"_.Bind(Spec<float>({15, 7}, {-inf, inf})),
                    "info:speed"_.Bind(Spec<float>({})),
                    "info:crashed"_.Bind(Spec<bool>({})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(MarkDiscrete(Spec<int>({-1}, {0, 2}))));
  }
};

class OfficialKinematics8ContinuousFns : public OfficialBaseFns<void> {
 public:
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const float inf = std::numeric_limits<float>::infinity();
    return MakeDict("obs"_.Bind(Spec<float>({5, 8}, {-inf, inf})),
                    "info:speed"_.Bind(Spec<float>({})),
                    "info:crashed"_.Bind(Spec<bool>({})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<float>({2}, {-1.0, 1.0})));
  }
};

template <int Horizon>
class OfficialTTCFns : public OfficialBaseFns<void> {
 public:
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.Bind(Spec<float>({3, 3, Horizon}, {0.0, 1.0})),
                    "info:speed"_.Bind(Spec<float>({})),
                    "info:crashed"_.Bind(Spec<bool>({})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(MarkDiscrete(Spec<int>({-1}, {0, 4}))));
  }
};

class OfficialGoalFns : public OfficialBaseFns<void> {
 public:
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const double inf = std::numeric_limits<double>::infinity();
    return MakeDict("obs:observation"_.Bind(Spec<double>({6}, {-inf, inf})),
                    "obs:achieved_goal"_.Bind(Spec<double>({6}, {-inf, inf})),
                    "obs:desired_goal"_.Bind(Spec<double>({6}, {-inf, inf})),
                    "info:is_success"_.Bind(Spec<bool>({})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<float>({2}, {-1.0, 1.0})));
  }
};

class OfficialAttributesFns : public OfficialBaseFns<void> {
 public:
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const double inf = std::numeric_limits<double>::infinity();
    return MakeDict(
        "obs:state"_.Bind(Spec<double>({4, 1}, {-inf, inf})),
        "obs:derivative"_.Bind(Spec<double>({4, 1}, {-inf, inf})),
        "obs:reference_state"_.Bind(Spec<double>({4, 1}, {-inf, inf})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<float>({1}, {-1.0, 1.0})));
  }
};

class OfficialOccupancyFns : public OfficialBaseFns<void> {
 public:
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const float inf = std::numeric_limits<float>::infinity();
    return MakeDict("obs"_.Bind(Spec<float>({2, 12, 12}, {-inf, inf})),
                    "info:speed"_.Bind(Spec<float>({})),
                    "info:crashed"_.Bind(Spec<bool>({})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<float>({1}, {-1.0, 1.0})));
  }
};

class OfficialMultiAgentFns : public OfficialBaseFns<void> {
 public:
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const float inf = std::numeric_limits<float>::infinity();
    return MakeDict(
        "obs:players.obs"_.Bind(Spec<float>({-1, 5, 5}, {-inf, inf})),
        "info:players.speed"_.Bind(Spec<float>({-1})),
        "info:players.crashed"_.Bind(Spec<bool>({-1})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "players.action"_.Bind(MarkDiscrete(Spec<int>({-1}, {0, 2}))));
  }
};

using OfficialKinematics5EnvSpec =
    OfficialEnvSpec<OfficialKinematics5Fns, KinematicsAdapter<5, 5>>;
using OfficialKinematics5Env =
    OfficialEnv<OfficialKinematics5EnvSpec, KinematicsAdapter<5, 5>>;
using OfficialKinematics5EnvPool =
    OfficialEnvPool<OfficialKinematics5EnvSpec, KinematicsAdapter<5, 5>>;

using OfficialKinematics7Action5EnvSpec =
    OfficialEnvSpec<OfficialKinematics7Action5Fns, KinematicsAdapter<15, 7>>;
using OfficialKinematics7Action5EnvPool =
    OfficialEnvPool<OfficialKinematics7Action5EnvSpec,
                    KinematicsAdapter<15, 7>>;

using OfficialKinematics7Action3EnvSpec =
    OfficialEnvSpec<OfficialKinematics7Action3Fns, KinematicsAdapter<15, 7>>;
using OfficialKinematics7Action3EnvPool =
    OfficialEnvPool<OfficialKinematics7Action3EnvSpec,
                    KinematicsAdapter<15, 7>>;

using OfficialKinematics8ContinuousEnvSpec =
    OfficialEnvSpec<OfficialKinematics8ContinuousFns,
                    Kinematics8ContinuousAdapter>;
using OfficialKinematics8ContinuousEnvPool =
    OfficialEnvPool<OfficialKinematics8ContinuousEnvSpec,
                    Kinematics8ContinuousAdapter>;

using OfficialTTC5EnvSpec = OfficialEnvSpec<OfficialTTCFns<5>, TTCAdapter<5>>;
using OfficialTTC5EnvPool = OfficialEnvPool<OfficialTTC5EnvSpec, TTCAdapter<5>>;

using OfficialTTC16EnvSpec =
    OfficialEnvSpec<OfficialTTCFns<16>, TTCAdapter<16>>;
using OfficialTTC16EnvPool =
    OfficialEnvPool<OfficialTTC16EnvSpec, TTCAdapter<16>>;

using OfficialGoalEnvSpec = OfficialEnvSpec<OfficialGoalFns, GoalAdapter>;
using OfficialGoalEnvPool = OfficialEnvPool<OfficialGoalEnvSpec, GoalAdapter>;

using OfficialAttributesEnvSpec =
    OfficialEnvSpec<OfficialAttributesFns, AttributesAdapter>;
using OfficialAttributesEnvPool =
    OfficialEnvPool<OfficialAttributesEnvSpec, AttributesAdapter>;

using OfficialOccupancyEnvSpec =
    OfficialEnvSpec<OfficialOccupancyFns, OccupancyAdapter>;
using OfficialOccupancyEnvPool =
    OfficialEnvPool<OfficialOccupancyEnvSpec, OccupancyAdapter>;

using OfficialMultiAgentEnvSpec =
    OfficialEnvSpec<OfficialMultiAgentFns, MultiAgentAdapter>;
using OfficialMultiAgentEnvPool =
    OfficialEnvPool<OfficialMultiAgentEnvSpec, MultiAgentAdapter>;

}  // namespace official
}  // namespace highway

#endif  // ENVPOOL_HIGHWAY_OFFICIAL_BRIDGE_H_
