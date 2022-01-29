/*
 * Copyright 2021 Garena Online Private Limited
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

#ifndef ENVPOOL_GYM_PYENV_WRAPPER_H_
#define ENVPOOL_GYM_PYENV_WRAPPER_H_

#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace gym {

class GymEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("state_shape"_.bind(std::vector<int>()),
                    "action_shape"_.bind(std::vector<int>()),
                    "env_fn"_.bind(py::function()),
                    "max_episode_steps"_.bind(0));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.bind(Spec<float>(conf["state_shape"_])));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.bind(Spec<float>(conf["action_shape"_])));
  }
};

typedef class EnvSpec<GymEnvFns> GymEnvSpec;

class GymEnv : public Env<GymEnvSpec> {
 protected:
  bool done_;
  int max_episode_steps_, elapsed_step_;
  std::vector<int> action_shape_;
  py::object env_;
  py::function reset_fn_, step_fn_;

 public:
  GymEnv(const Spec& spec, int env_id)
      : Env<GymEnvSpec>(spec, env_id),
        done_(true),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        elapsed_step_(max_episode_steps_ + 1),
        action_shape_(spec.config["action_shape"_]),
        env_(spec.config["env_fn"_]()),
        reset_fn_(env_.attr("reset")),
        step_fn_(env_.attr("step")) {}

  ~GymEnv() { env_.attr("close")(); }

  bool IsDone() override { return done_; }

  void Reset() override {
    elapsed_step_ = 0;
    py::array_t<float> obs = reset_fn_();
    WriteState(obs.mutable_data(), obs.size(), 0.0f);
  }

  void Step(const Action& action) override {
    py::array_t<float> pyact(action_shape_,
                             static_cast<float*>(action["action"_].data()));
    py::tuple result = step_fn_(pyact);
    py::array_t<float> obs(result[0]);
    float reward = py::float_(result[1]);
    done_ = py::bool_(result[2]);
    if (max_episode_steps_ > 0) {
      done_ |= (++elapsed_step_ >= max_episode_steps_);
    }
    WriteState(obs.mutable_data(), obs.size(), reward);
  }

  void WriteState(float* obs, std::size_t size, float reward) {
    State state = Allocate();
    state["reward"_] = reward;
    float* tgt = static_cast<float*>(state["obs"_].data());
    memcpy(tgt, obs, size);
  }
};

typedef AsyncEnvPool<GymEnv> GymEnvPool;

}  // namespace gym

#endif  // ENVPOOL_GYM_PYENV_WRAPPER_H_
