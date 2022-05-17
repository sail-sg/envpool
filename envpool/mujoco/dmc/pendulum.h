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
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/pendulum.py

#ifndef ENVPOOL_MUJOCO_DMC_PENDULUM_H_
#define ENVPOOL_MUJOCO_DMC_PENDULUM_H_

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

std::string GetPendulumXML(const std::string& base_path,
                           const std::string& task_name) {
  return GetFileContent(base_path, "pendulum.xml");
}

class PendulumEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.Bind(1000), "frame_skip"_.Bind(1),
                    "task_name"_.Bind(std::string("stand")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs:orientation"_.Bind(Spec<mjtNum>({2})),
                    "obs:velocity"_.Bind(Spec<mjtNum>({1})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.Bind(Spec<mjtNum>({1})),
#endif
                    "discount"_.Bind(Spec<float>({-1}, {0.0, 1.0})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 1}, {-1.0, 1.0})));
  }
};

using PendulumEnvSpec = EnvSpec<PendulumEnvFns>;

class PendulumEnv : public Env<PendulumEnvSpec>, public MujocoEnv {
 protected:
  const mjtNum kCosineBound = std::cos(8.0 / 180 * M_PI);
  int id_hinge_, id_pole_;
  std::uniform_real_distribution<> dist_uniform_;

 public:
  PendulumEnv(const Spec& spec, int env_id)
      : Env<PendulumEnvSpec>(spec, env_id),
        MujocoEnv(spec.config["base_path"_],
                  GetPendulumXML(spec.config["base_path"_],
                                 spec.config["task_name"_]),
                  spec.config["frame_skip"_],
                  spec.config["max_episode_steps"_]),
        id_hinge_(mj_name2id(model_, mjOBJ_JOINT, "hinge")),
        id_pole_(mj_name2id(model_, mjOBJ_XBODY, "pole")),
        dist_uniform_(-M_PI, M_PI) {}

  void TaskInitializeEpisode() override {
    data_->qpos[0] = dist_uniform_(gen_);
#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_.get(), data_->qpos, sizeof(mjtNum) * model_->nq);
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
    return static_cast<float>(RewardTolerance(PoleVertical(), kCosineBound, 1));
  }
  bool TaskShouldTerminateEpisode() override { return false; }

 private:
  void WriteState() {
    State state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    // obs
    const auto& pole_orient = PoleOrientation();
    state["obs:orientation"_].Assign(pole_orient.begin(), pole_orient.size());
    state["obs:velocity"_] = AngularVelocity();
    // info for check alignment
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
#endif
  }

  mjtNum PoleVertical() {
    // return self.named.data.xmat['pole', 'zz']
    return data_->xmat[id_pole_ * 9 + 8];
  }
  mjtNum AngularVelocity() {
    // return self.named.data.qvel['hinge'].copy()
    return data_->qvel[id_hinge_];
  }
  std::array<mjtNum, 2> PoleOrientation() {
    // return self.named.data.xmat['pole', ['zz', 'xz']]
    return {data_->xmat[id_pole_ * 9 + 8], data_->xmat[id_pole_ * 9 + 2]};
  }
};

using PendulumEnvPool = AsyncEnvPool<PendulumEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_PENDULUM_H_
