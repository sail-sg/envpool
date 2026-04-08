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
// https://github.com/deepmind/dm_control/blob/1.0.38/dm_control/suite/lqr.py

#ifndef ENVPOOL_MUJOCO_DMC_LQR_H_
#define ENVPOOL_MUJOCO_DMC_LQR_H_

#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/dmc/mujoco_env.h"
#include "envpool/mujoco/dmc/utils.h"

namespace mujoco_dmc {

int LqrNumBodies(const std::string& task_name) {
  if (task_name == "lqr_2_1") {
    return 2;
  }
  if (task_name == "lqr_6_2") {
    return 6;
  }
  throw std::runtime_error("Unknown task_name " + task_name + " for dmc lqr.");
}

int LqrNumActuators(const std::string& task_name) {
  if (task_name == "lqr_2_1") {
    return 1;
  }
  if (task_name == "lqr_6_2") {
    return 2;
  }
  throw std::runtime_error("Unknown task_name " + task_name + " for dmc lqr.");
}

std::string GetLqrXML(const std::string& base_path,
                      const std::string& task_name, int seed) {
  std::mt19937 gen(seed);
  return XMLMakeLqr(GetFileContent(base_path, "lqr.xml"),
                    LqrNumBodies(task_name), LqrNumActuators(task_name), &gen);
}

class LqrEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("frame_skip"_.Bind(1), "frame_stack"_.Bind(1),
                    "task_name"_.Bind(std::string("lqr_2_1")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    int n_bodies = LqrNumBodies(conf["task_name"_]);
    return MakeDict("obs:position"_.Bind(StackSpec(Spec<mjtNum>({n_bodies}),
                                                   conf["frame_stack"_])),
                    "obs:velocity"_.Bind(StackSpec(Spec<mjtNum>({n_bodies}),
                                                   conf["frame_stack"_]))
#ifdef ENVPOOL_TEST
                        ,
                    "info:qpos0"_.Bind(Spec<mjtNum>({n_bodies})),
                    "info:qvel0"_.Bind(Spec<mjtNum>({n_bodies})),
                    "info:stiffness0"_.Bind(Spec<mjtNum>({n_bodies}))
#endif
    );  // NOLINT
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>(
        {-1, LqrNumActuators(conf["task_name"_])}, {-1.0e10, 1.0e10})));
  }
};

using LqrEnvSpec = EnvSpec<LqrEnvFns>;
using LqrPixelEnvFns = PixelObservationEnvFns<LqrEnvFns>;
using LqrPixelEnvSpec = EnvSpec<LqrPixelEnvFns>;

template <typename EnvSpecT, bool kFromPixels>
class LqrEnvBase : public Env<EnvSpecT>, public MujocoEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::seed_;
#ifdef ENVPOOL_TEST
  std::vector<mjtNum> qvel0_;
  std::vector<mjtNum> stiffness0_;
#endif

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  LqrEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        MujocoEnv(
            spec.config["base_path"_],
            GetLqrXML(spec.config["base_path"_], spec.config["task_name"_],
                      Env<EnvSpecT>::ResolveSeed(spec, env_id)),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_],
            spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)) {
#ifdef ENVPOOL_TEST
    qvel0_.resize(model_->nv);
    stiffness0_.resize(model_->njnt);
#endif
  }

  void TaskInitializeEpisode() override {
    mjtNum norm = 0;
    for (int i = 0; i < model_->nq; ++i) {
      data_->qpos[i] = RandNormal(0, 1)(gen_);
      norm += data_->qpos[i] * data_->qpos[i];
    }
    norm = std::sqrt(norm);
    for (int i = 0; i < model_->nq; ++i) {
      data_->qpos[i] = std::sqrt(2.0) * data_->qpos[i] / norm;
    }
#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_.get(), data_->qpos, sizeof(mjtNum) * model_->nq);
    std::memcpy(qvel0_.data(), data_->qvel, sizeof(mjtNum) * model_->nv);
    std::memcpy(stiffness0_.data(), model_->jnt_stiffness,
                sizeof(mjtNum) * model_->njnt);
#endif
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    ControlReset();
    WriteState(true);
  }

  void Step(const Action& action) override {
    auto* act = static_cast<mjtNum*>(action["action"_].Data());
    ControlStep(act);
    WriteState(false);
  }

  float TaskGetReward() override {
    mjtNum state_cost = 0.0;
    for (int i = 0; i < model_->nq; ++i) {
      state_cost += data_->qpos[i] * data_->qpos[i];
    }
    state_cost *= 0.5;

    mjtNum control_l2_norm = 0.0;
    for (int i = 0; i < model_->nu; ++i) {
      control_l2_norm += data_->ctrl[i] * data_->ctrl[i];
    }
    control_l2_norm *= 0.5;
    return static_cast<float>(1.0 - (state_cost + control_l2_norm * 0.1));
  }

  bool TaskShouldTerminateEpisode() override { return StateNorm() < 1e-6; }

 private:
  mjtNum StateNorm() {
    mjtNum norm = 0.0;
    for (int i = 0; i < model_->nq; ++i) {
      norm += data_->qpos[i] * data_->qpos[i];
    }
    for (int i = 0; i < model_->nv; ++i) {
      norm += data_->qvel[i] * data_->qvel[i];
    }
    return std::sqrt(norm);
  }

  void WriteState(bool reset) {
    auto state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      auto obs_position = state["obs:position"_];
      AssignObservation("obs:position", &obs_position, data_->qpos, model_->nq,
                        reset);
      auto obs_velocity = state["obs:velocity"_];
      AssignObservation("obs:velocity", &obs_velocity, data_->qvel, model_->nv,
                        reset);
    }
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
    state["info:qvel0"_].Assign(qvel0_.data(), model_->nv);
    state["info:stiffness0"_].Assign(stiffness0_.data(), model_->njnt);
#endif
  }
};

using LqrEnv = LqrEnvBase<LqrEnvSpec, false>;
using LqrPixelEnv = LqrEnvBase<LqrPixelEnvSpec, true>;
using LqrEnvPool = AsyncEnvPool<LqrEnv>;
using LqrPixelEnvPool = AsyncEnvPool<LqrPixelEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_LQR_H_
