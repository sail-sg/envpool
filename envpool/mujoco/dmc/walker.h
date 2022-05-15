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

#ifndef ENVPOOL_MUJOCO_DMC_WALKER_H_
#define ENVPOOL_MUJOCO_DMC_WALKER_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/dmc/mujoco_env.h"
#include "envpool/mujoco/dmc/utils.h"

// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/walker.py
namespace mujoco_dmc {

std::string GetWalkerXML(const std::string& base_path,
                         const std::string& task_name) {
  return GetFileContent(base_path, "walker.xml");
}

class WalkerEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.Bind(1000), "frame_skip"_.Bind(10),
                    "task_name"_.Bind(std::string("stand")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs:orientations"_.Bind(Spec<mjtNum>({14})),
                    "obs:height"_.Bind(Spec<mjtNum>({})),
                    "obs:velocity"_.Bind(Spec<mjtNum>({9})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.Bind(Spec<mjtNum>({9})),
#endif
                    "discount"_.Bind(Spec<float>({-1}, {0.0, 1.0})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 6}, {-1.0, 1.0})));
  }
};

using WalkerEnvSpec = EnvSpec<WalkerEnvFns>;

class WalkerEnv : public Env<WalkerEnvSpec>, public MujocoEnv {
 protected:
  // Minimal height of torso over foot above which stand reward is 1.
  const mjtNum kStandHeight = 1.2;
  // Horizontal speeds(meters / second) above which move reward is 1.
  const mjtNum kWalkSpeed = 1;
  const mjtNum kRunSpeed = 8;
  float move_speed_;

 public:
  WalkerEnv(const Spec& spec, int env_id)
      : Env<WalkerEnvSpec>(spec, env_id),
        MujocoEnv(
            spec.config["base_path"_],
            GetWalkerXML(spec.config["base_path"_], spec.config["task_name"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_]) {
    std::string task_name = spec.config["task_name"_];
    if (task_name == "stand") {
      move_speed_ = 0;
    } else if (task_name == "walk") {
      move_speed_ = kWalkSpeed;
    } else if (task_name == "run") {
      move_speed_ = kRunSpeed;
    } else {
      throw std::runtime_error("Unknown task_name for dmc hopper.");
    }
  }

  void TaskInitializeEpisode() override {
    // randomizers.randomize_limited_and_rotational_joints(physics,
    // self.random)
    RandomizeLimitedAndRotationalJoints(&gen_);
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
    auto standing = RewardTolerance(TorsoHeight(), kStandHeight,
                                    std::numeric_limits<double>::infinity(),
                                    kStandHeight / 2);
    auto upright = (1 + TorsoUpright()) / 2;
    auto stand_reward = (3 * standing + upright) / 4;
    if (move_speed_ == 0) {
      return static_cast<float>(stand_reward);
    }
    auto move_reward =
        RewardTolerance(HorizontalVelocity(), move_speed_,
                        std::numeric_limits<double>::infinity(),
                        move_speed_ / 2, 0.5, SigmoidType::kLinear);
    return static_cast<float>(stand_reward * (5 * move_reward + 1) / 6);
  }
  bool TaskShouldTerminateEpisode() override { return false; }

 private:
  void WriteState() {
    State state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    // obs
    auto orient = Orientations();
    state["obs:orientations"_].Assign(orient.begin(), orient.size());
    state["obs:height"_] = TorsoHeight();
    state["obs:velocity"_].Assign(data_->qvel, model_->nv);
    // info for check alignment
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
#endif
  }

  mjtNum TorsoUpright() {
    // return self.named.data.xmat['torso', 'zz']
    return data_->xmat[1 * 9 + 8];
  }
  mjtNum TorsoHeight() {
    // return self.named.data.xpos['torso', 'z']
    return data_->xpos[1 * 3 + 2];
  }
  mjtNum HorizontalVelocity() {
    // return self.named.data.sensordata['torso_subtreelinvel'][0]
    return data_->sensordata[0];
  }
  std::array<mjtNum, 14> Orientations() {
    // return self.named.data.xmat[1:, ['xx', 'xz']].ravel()
    std::array<mjtNum, 14> orient;
    for (int i = 0; i < 7; i++) {
      orient[i * 2 + 0] = data_->xmat[(1 + i) * 9 + 0];
      orient[i * 2 + 1] = data_->xmat[(1 + i) * 9 + 2];
    }
    return orient;
  }
};

using WalkerEnvPool = AsyncEnvPool<WalkerEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_WALKER_H_
