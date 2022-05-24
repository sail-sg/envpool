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
    return MakeDict("max_episode_steps"_.Bind(1000), "frame_skip"_.Bind(1),
                    "task_name"_.Bind(std::string("easy")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs:position"_.Bind(Spec<mjtNum>({2})),
                    "obs:to_target"_.Bind(Spec<mjtNum>({2})),
                    "obs:velocity"_.Bind(Spec<mjtNum>({2})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.Bind(Spec<mjtNum>({2})),
                    "info:target"_.Bind(Spec<mjtNum>({2})),
#endif
                    "discount"_.Bind(Spec<float>({-1}, {0.0, 1.0})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 2}, {-1.0, 1.0})));
  }
};

using ReacherEnvSpec = EnvSpec<ReacherEnvFns>;

class ReacherEnv : public Env<ReacherEnvSpec>, public MujocoEnv {
 protected:
  const mjtNum kBigTarget = 0.05;
  const mjtNum kSmallTarget = 0.015;
  int id_target_, id_finger_;
  mjtNum target_size_;
#ifdef ENVPOOL_TEST
  std::array<mjtNum, 2> target_;
#endif

 public:
  ReacherEnv(const Spec& spec, int env_id)
      : Env<ReacherEnvSpec>(spec, env_id),
        MujocoEnv(
            spec.config["base_path"_],
            GetReacherXML(spec.config["base_path"_], spec.config["task_name"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_]),
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
    WriteState();
  }

  void Step(const Action& action) override {
    mjtNum* act = static_cast<mjtNum*>(action["action"_].Data());
    ControlStep(act);
    WriteState();
  }

  float TaskGetReward() override {
    // radii = physics.named.model.geom_size[['target', 'finger'], 0].sum()
    mjtNum radii =
        model_->geom_size[id_target_ * 3] + model_->geom_size[id_finger_ * 3];
    return static_cast<float>(RewardTolerance(FingerToTargetDist(), 0, radii));
  }

 private:
  void WriteState() {
    State state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    // obs
    state["obs:position"_].Assign(data_->qpos, model_->nq);
    const auto& finger = FingerToTarget();
    state["obs:to_target"_].Assign(finger.begin(), finger.size());
    state["obs:velocity"_].Assign(data_->qvel, model_->nv);
    // info for check alignment
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
    state["info:target"_].Assign(target_.begin(), target_.size());
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

using ReacherEnvPool = AsyncEnvPool<ReacherEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_REACHER_H_
