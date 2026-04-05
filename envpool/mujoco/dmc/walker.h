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
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/walker.py

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

namespace mujoco_dmc {

std::string GetWalkerXML(const std::string& base_path,
                         const std::string& task_name) {
  return GetFileContent(base_path, "walker.xml");
}

class WalkerEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("frame_skip"_.Bind(10), "frame_stack"_.Bind(1),
                    "task_name"_.Bind(std::string("stand")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict(
        "obs:orientations"_.Bind(
            StackSpec(Spec<mjtNum>({14}), conf["frame_stack"_])),
        "obs:height"_.Bind(StackSpec(Spec<mjtNum>({}), conf["frame_stack"_])),
        "obs:velocity"_.Bind(StackSpec(Spec<mjtNum>({9}), conf["frame_stack"_]))
#ifdef ENVPOOL_TEST
            ,
        "info:qpos0"_.Bind(Spec<mjtNum>({9}))
#endif
    );  // NOLINT
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 6}, {-1.0, 1.0})));
  }
};

using WalkerEnvSpec = EnvSpec<WalkerEnvFns>;
using WalkerPixelEnvFns = PixelObservationEnvFns<WalkerEnvFns>;
using WalkerPixelEnvSpec = EnvSpec<WalkerPixelEnvFns>;

template <typename EnvSpecT, bool kFromPixels>
class WalkerEnvBase : public Env<EnvSpecT>, public MujocoEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;

  // Minimal height of torso over foot above which stand reward is 1.
  const mjtNum kStandHeight = 1.2;
  // Horizontal speeds(meters / second) above which move reward is 1.
  const mjtNum kWalkSpeed = 1;
  const mjtNum kRunSpeed = 8;
  int id_torso_, id_torso_subtreelinvel_;
  mjtNum move_speed_;

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  WalkerEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        MujocoEnv(
            spec.config["base_path"_],
            GetWalkerXML(spec.config["base_path"_], spec.config["task_name"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_],
            spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        id_torso_(mj_name2id(model_, mjOBJ_XBODY, "torso")),
        id_torso_subtreelinvel_(GetSensorId(model_, "torso_subtreelinvel")) {
    const std::string& task_name = spec.config["task_name"_];
    if (task_name == "stand") {
      move_speed_ = 0;
    } else if (task_name == "walk") {
      move_speed_ = kWalkSpeed;
    } else if (task_name == "run") {
      move_speed_ = kRunSpeed;
    } else {
      throw std::runtime_error("Unknown task_name " + task_name +
                               " for dmc walker.");
    }
  }

  void TaskInitializeEpisode() override {
    RandomizeLimitedAndRotationalJoints(&gen_);
#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_.get(), data_->qpos, sizeof(mjtNum) * model_->nq);
#endif
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    ControlReset();
    WriteState(true);
  }

  void Step(const Action& action) override {
    mjtNum* act = static_cast<mjtNum*>(action["action"_].Data());
    ControlStep(act);
    WriteState(false);
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
  void WriteState(bool reset) {
    auto state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      const auto& orient = Orientations();
      auto obs_orientations = state["obs:orientations"_];
      AssignObservation("obs:orientations", &obs_orientations, orient.data(),
                        orient.size(), reset);
      auto obs_height = state["obs:height"_];
      AssignObservation("obs:height", &obs_height, TorsoHeight(), reset);
      auto obs_velocity = state["obs:velocity"_];
      AssignObservation("obs:velocity", &obs_velocity, data_->qvel, model_->nv,
                        reset);
#ifdef ENVPOOL_TEST
      state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
#endif
    }
  }

  mjtNum TorsoUpright() {
    // return self.named.data.xmat['torso', 'zz']
    return data_->xmat[id_torso_ * 9 + 8];
  }
  mjtNum TorsoHeight() {
    // return self.named.data.xpos['torso', 'z']
    return data_->xpos[id_torso_ * 3 + 2];
  }
  mjtNum HorizontalVelocity() {
    // return self.named.data.sensordata['torso_subtreelinvel'][0]
    return data_->sensordata[id_torso_subtreelinvel_];
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

using WalkerEnv = WalkerEnvBase<WalkerEnvSpec, false>;
using WalkerPixelEnv = WalkerEnvBase<WalkerPixelEnvSpec, true>;
using WalkerEnvPool = AsyncEnvPool<WalkerEnv>;
using WalkerPixelEnvPool = AsyncEnvPool<WalkerPixelEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_WALKER_H_
