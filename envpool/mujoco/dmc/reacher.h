/*
 * Copyright 2022 Garena Online Private Limited
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
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/reacher.py

#ifndef ENVPOOL_MUJOCO_DMC_REACHER_H_
#define ENVPOOL_MUJOCO_DMC_REACHER_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/dmc/mujoco_env.h"
#include "envpool/mujoco/dmc/utils.h"

namespace mujoco_dmc {

std::string GetReacherXML(const std::string& base_path,
                          const std::string& task_name) {
  return GetFileContent(base_path, "reacher.xml");
}

class ReacherEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("frame_skip"_.Bind(1), "frame_stack"_.Bind(1),
                    "task_name"_.Bind(std::string("easy")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict(
        "obs:position"_.Bind(
            StackSpec(Spec<mjtNum>({2}), conf["frame_stack"_])),
        "obs:to_target"_.Bind(
            StackSpec(Spec<mjtNum>({2}), conf["frame_stack"_])),
        "obs:velocity"_.Bind(StackSpec(Spec<mjtNum>({2}), conf["frame_stack"_]))
#ifdef ENVPOOL_TEST
            ,
        "info:qpos0"_.Bind(Spec<mjtNum>({2})),
        "info:target"_.Bind(Spec<mjtNum>({2}))
#endif
    );  // NOLINT
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 2}, {-1.0, 1.0})));
  }
};

using ReacherEnvSpec = EnvSpec<ReacherEnvFns>;
using ReacherPixelEnvFns = PixelObservationEnvFns<ReacherEnvFns>;
using ReacherPixelEnvSpec = EnvSpec<ReacherPixelEnvFns>;

template <typename EnvSpecT, bool kFromPixels>
class ReacherEnvBase : public Env<EnvSpecT>, public MujocoEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;

  const mjtNum kBigTarget = 0.05;
  const mjtNum kSmallTarget = 0.015;
  int id_target_, id_finger_;
  mjtNum target_size_;
#ifdef ENVPOOL_TEST
  std::array<mjtNum, 2> target_;
#endif

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  ReacherEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        MujocoEnv(
            spec.config["base_path"_],
            GetReacherXML(spec.config["base_path"_], spec.config["task_name"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_],
            spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        id_target_(mj_name2id(model_, mjOBJ_GEOM, "target")),
        id_finger_(mj_name2id(model_, mjOBJ_GEOM, "finger")) {
    const std::string& task_name = spec.config["task_name"_];
    if (task_name == "easy") {
      target_size_ = kBigTarget;
    } else if (task_name == "hard") {
      target_size_ = kSmallTarget;
    } else {
      throw std::runtime_error("Unknown task_name " + task_name +
                               " for dmc reacher.");
    }
  }

  void TaskInitializeEpisode() override {
    model_->geom_size[6 * 3] = target_size_;
    RandomizeLimitedAndRotationalJoints(&gen_);
    mjtNum angle = RandUniform(0, M_PI * 2)(gen_);
    mjtNum radius = RandUniform(0.05, 0.2)(gen_);
    // physics.named.model.geom_pos['target', 'x'] = radius * np.sin(angle)
    // physics.named.model.geom_pos['target', 'y'] = radius * np.cos(angle)
    model_->geom_pos[id_target_ * 3 + 0] = radius * std::sin(angle);
    model_->geom_pos[id_target_ * 3 + 1] = radius * std::cos(angle);
#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_.get(), data_->qpos, sizeof(mjtNum) * model_->nq);
    target_[0] = model_->geom_pos[id_target_ * 3 + 0];
    target_[1] = model_->geom_pos[id_target_ * 3 + 1];
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
    // radii = physics.named.model.geom_size[['target', 'finger'], 0].sum()
    mjtNum radii =
        model_->geom_size[id_target_ * 3] + model_->geom_size[id_finger_ * 3];
    return static_cast<float>(RewardTolerance(FingerToTargetDist(), 0, radii));
  }

 private:
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
      const auto& finger = FingerToTarget();
      auto obs_to_target = state["obs:to_target"_];
      AssignObservation("obs:to_target", &obs_to_target, finger.data(),
                        finger.size(), reset);
      auto obs_velocity = state["obs:velocity"_];
      AssignObservation("obs:velocity", &obs_velocity, data_->qvel, model_->nv,
                        reset);
    }
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
    state["info:target"_].Assign(target_.data(), target_.size());
#endif
  }

  std::array<mjtNum, 2> FingerToTarget() {
    // return (self.named.data.geom_xpos['target', :2] -
    //         self.named.data.geom_xpos['finger', :2])
    return {data_->geom_xpos[id_target_ * 3 + 0] -
                data_->geom_xpos[id_finger_ * 3 + 0],
            data_->geom_xpos[id_target_ * 3 + 1] -
                data_->geom_xpos[id_finger_ * 3 + 1]};
  }
  mjtNum FingerToTargetDist() {
    const auto& finger = FingerToTarget();
    return std::sqrt(finger[0] * finger[0] + finger[1] * finger[1]);
  }
};

using ReacherEnv = ReacherEnvBase<ReacherEnvSpec, false>;
using ReacherPixelEnv = ReacherEnvBase<ReacherPixelEnvSpec, true>;
using ReacherEnvPool = AsyncEnvPool<ReacherEnv>;
using ReacherPixelEnvPool = AsyncEnvPool<ReacherPixelEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_REACHER_H_
