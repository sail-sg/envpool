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
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/acrobot.py

#ifndef ENVPOOL_MUJOCO_DMC_ACROBOT_H_
#define ENVPOOL_MUJOCO_DMC_ACROBOT_H_

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

std::string GetAcrobotXML(const std::string& base_path,
                          const std::string& task_name) {
  return GetFileContent(base_path, "acrobot.xml");
}

class AcrobotEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.Bind(1000), "frame_skip"_.Bind(1),
                    "task_name"_.Bind(std::string("swingup")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs:orientations"_.Bind(Spec<mjtNum>({4})),
                    "obs:velocity"_.Bind(Spec<mjtNum>({2})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.Bind(Spec<mjtNum>({2})),
#endif
                    "discount"_.Bind(Spec<float>({-1}, {0.0, 1.0})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 1}, {-1.0, 1.0})));
  }
};

using AcrobotEnvSpec = EnvSpec<AcrobotEnvFns>;

class AcrobotEnv : public Env<AcrobotEnvSpec>, public MujocoEnv {
 protected:
  int id_upper_arm_, id_lower_arm_, id_target_, id_tip_, id_shoulder_,
      id_elbow_;
  bool is_sparse_;

 public:
  AcrobotEnv(const Spec& spec, int env_id)
      : Env<AcrobotEnvSpec>(spec, env_id),
        MujocoEnv(
            spec.config["base_path"_],
            GetAcrobotXML(spec.config["base_path"_], spec.config["task_name"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_]),
        id_upper_arm_(mj_name2id(model_, mjOBJ_XBODY, "upper_arm")),
        id_lower_arm_(mj_name2id(model_, mjOBJ_XBODY, "lower_arm")),
        id_target_(mj_name2id(model_, mjOBJ_SITE, "target")),
        id_tip_(mj_name2id(model_, mjOBJ_SITE, "tip")),
        id_shoulder_(GetQposId(model_, "shoulder")),
        id_elbow_(GetQposId(model_, "elbow")),
        is_sparse_(spec.config["task_name"_] == "swingup_sparse") {
    const std::string& task_name = spec.config["task_name"_];
    if (task_name != "swingup" && task_name != "swingup_sparse") {
      throw std::runtime_error("Unknown task_name " + task_name +
                               " for dmc acrobot.");
    }
  }

  void TaskInitializeEpisode() override {
    data_->qpos[id_shoulder_] = RandUniform(-M_PI, M_PI)(gen_);
    data_->qpos[id_elbow_] = RandUniform(-M_PI, M_PI)(gen_);
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
    mjtNum target_radius = model_->site_size[id_target_];
    return static_cast<float>(RewardTolerance(ToTarget(), 0.0, target_radius,
                                              is_sparse_ ? 0.0 : 1.0));
  }
  bool TaskShouldTerminateEpisode() override { return false; }

 private:
  void WriteState() {
    State state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    // obs
    const auto& orientations = Orientations();
    state["obs:orientations"_].Assign(orientations.begin(),
                                      orientations.size());
    state["obs:velocity"_].Assign(data_->qvel, model_->nv);
    // info for check alignment
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
#endif
  }

  std::array<mjtNum, 2> Horizontal() {
    // return self.named.data.xmat[['upper_arm', 'lower_arm'], 'xz']
    return {data_->xmat[id_upper_arm_ * 9 + 2],
            data_->xmat[id_lower_arm_ * 9 + 2]};
  }
  std::array<mjtNum, 2> Vertical() {
    // return self.named.data.xmat[['upper_arm', 'lower_arm'], 'zz']
    return {data_->xmat[id_upper_arm_ * 9 + 8],
            data_->xmat[id_lower_arm_ * 9 + 8]};
  }
  mjtNum ToTarget() {
    // tip_to_target = (self.named.data.site_xpos['target'] -
    //                  self.named.data.site_xpos['tip'])
    // return np.linalg.norm(tip_to_target)
    std::array<mjtNum, 3> tip_to_target = {
        data_->site_xpos[id_target_ * 3] - data_->site_xpos[id_tip_ * 3],
        data_->site_xpos[id_target_ * 3 + 1] -
            data_->site_xpos[id_tip_ * 3 + 1],
        data_->site_xpos[id_target_ * 3 + 2] -
            data_->site_xpos[id_tip_ * 3 + 2]};
    return std::sqrt(tip_to_target[0] * tip_to_target[0] +
                     tip_to_target[1] * tip_to_target[1] +
                     tip_to_target[2] * tip_to_target[2]);
  }
  std::array<mjtNum, 4> Orientations() {
    const auto& horizontal = Horizontal();
    const auto& vertical = Vertical();
    return {horizontal[0], horizontal[1], vertical[0], vertical[1]};
  }
};

using AcrobotEnvPool = AsyncEnvPool<AcrobotEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_ACROBOT_H_
