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
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/hopper.py

#ifndef ENVPOOL_MUJOCO_DMC_HOPPER_H_
#define ENVPOOL_MUJOCO_DMC_HOPPER_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/dmc/mujoco_env.h"
#include "envpool/mujoco/dmc/utils.h"

namespace mujoco_dmc {

std::string GetHopperXML(const std::string& base_path,
                         const std::string& task_name) {
  return GetFileContent(base_path, "hopper.xml");
}

class HopperEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("frame_skip"_.Bind(4), "frame_stack"_.Bind(1),
                    "task_name"_.Bind(std::string("stand")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict(
        "obs:position"_.Bind(
            StackSpec(Spec<mjtNum>({6}), conf["frame_stack"_])),
        "obs:velocity"_.Bind(
            StackSpec(Spec<mjtNum>({7}), conf["frame_stack"_])),
        "obs:touch"_.Bind(StackSpec(Spec<mjtNum>({2}), conf["frame_stack"_]))
#ifdef ENVPOOL_TEST
            ,
        "info:qpos0"_.Bind(Spec<mjtNum>({7}))
#endif
    );  // NOLINT
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 4}, {-1.0, 1.0})));
  }
};

using HopperEnvSpec = EnvSpec<HopperEnvFns>;
using HopperPixelEnvFns = PixelObservationEnvFns<HopperEnvFns>;
using HopperPixelEnvSpec = EnvSpec<HopperPixelEnvFns>;

template <typename EnvSpecT, bool kFromPixels>
class HopperEnvBase : public Env<EnvSpecT>, public MujocoEnv {
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;

  const mjtNum kStandHeight = 0.6;
  const mjtNum kHopSpeed = 2;
  int id_torso_, id_foot_;
  int id_torso_subtreelinvel_, id_touch_toe_, id_touch_heel_;
  bool hopping_;

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  HopperEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        MujocoEnv(
            spec.config["base_path"_],
            GetHopperXML(spec.config["base_path"_], spec.config["task_name"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_],
            spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        id_torso_(mj_name2id(model_, mjOBJ_XBODY, "torso")),
        id_foot_(mj_name2id(model_, mjOBJ_XBODY, "foot")),
        id_torso_subtreelinvel_(GetSensorId(model_, "torso_subtreelinvel")),
        id_touch_toe_(GetSensorId(model_, "touch_toe")),
        id_touch_heel_(GetSensorId(model_, "touch_heel")) {
    const std::string& task_name = spec.config["task_name"_];
    if (task_name == "stand") {
      hopping_ = false;
    } else if (task_name == "hop") {
      hopping_ = true;
    } else {
      throw std::runtime_error("Unknown task_name " + task_name +
                               " for dmc hopper.");
    }
  }

  void TaskInitializeEpisode() override {
    // randomizers.randomize_limited_and_rotational_joints(physics, self.random)
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
    auto* act = static_cast<mjtNum*>(action["action"_].Data());
    ControlStep(act);
    WriteState(false);
  }

  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/hopper.py#L119
  float TaskGetReward() override {
    double standing = RewardTolerance(Height(), kStandHeight, 2);
    if (hopping_) {
      double hopping = RewardTolerance(
          Speed(), kHopSpeed, std::numeric_limits<double>::infinity(),
          kHopSpeed / 2, 0.5, SigmoidType::kLinear);
      return static_cast<float>(standing * hopping);
    }
    double small_control = 0.0;
    for (int i = 0; i < model_->nu; ++i) {
      small_control += RewardTolerance(data_->ctrl[i], 0.0, 0.0, 1.0, 0.0,
                                       SigmoidType::kQuadratic);
    }
    small_control = (small_control / model_->nu + 4) / 5;
    return static_cast<float>(standing * small_control);
  }

  bool TaskShouldTerminateEpisode() override { return false; }

 private:
  mjtNum Height() {
    // return (self.named.data.xipos['torso', 'z'] -
    //         self.named.data.xipos['foot', 'z'])
    return data_->xipos[id_torso_ * 3 + 2] - data_->xipos[id_foot_ * 3 + 2];
  }

  mjtNum Speed() {
    // return self.named.data.sensordata['torso_subtreelinvel'][0]
    return data_->sensordata[id_torso_subtreelinvel_];
  }

  std::array<mjtNum, 2> Touch() {
    // return np.log1p(self.named.data.sensordata[['touch_toe', 'touch_heel']])
    return {std::log1p(data_->sensordata[id_touch_toe_]),
            std::log1p(data_->sensordata[id_touch_heel_])};
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
      AssignObservation("obs:position", &obs_position, data_->qpos + 1,
                        model_->nq - 1, reset);
      auto obs_velocity = state["obs:velocity"_];
      AssignObservation("obs:velocity", &obs_velocity, data_->qvel, model_->nv,
                        reset);
      const auto& touch = Touch();
      auto obs_touch = state["obs:touch"_];
      AssignObservation("obs:touch", &obs_touch, touch.data(), 2, reset);
    }
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
#endif
  }
};

using HopperEnv = HopperEnvBase<HopperEnvSpec, false>;
using HopperPixelEnv = HopperEnvBase<HopperPixelEnvSpec, true>;
using HopperEnvPool = AsyncEnvPool<HopperEnv>;
using HopperPixelEnvPool = AsyncEnvPool<HopperPixelEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_HOPPER_H_
