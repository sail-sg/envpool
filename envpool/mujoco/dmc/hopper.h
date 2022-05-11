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
    return MakeDict("max_episode_steps"_.Bind(1000), "frame_skip"_.Bind(4),
                    "task_name"_.Bind(std::string("stand")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs:position"_.Bind(Spec<mjtNum>({6})),
                    "obs:velocity"_.Bind(Spec<mjtNum>({7})),
                    "obs:touch"_.Bind(Spec<mjtNum>({2})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.Bind(Spec<mjtNum>({7})),
#endif
                    "discount"_.Bind(Spec<float>({-1}, {0.0, 1.0})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 4}, {-1.0, 1.0})));
  }
};

using HopperEnvSpec = EnvSpec<HopperEnvFns>;

class HopperEnv : public Env<HopperEnvSpec>, public MujocoEnv {
  const mjtNum kStandHeight = 0.6;
  const mjtNum kHopSpeed = 2;
  bool hopping_;
  std::unique_ptr<mjtNum> qpos0_;

 public:
  HopperEnv(const Spec& spec, int env_id)
      : Env<HopperEnvSpec>(spec, env_id),
        MujocoEnv(
            spec.config["base_path"_],
            GetHopperXML(spec.config["base_path"_], spec.config["task_name"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_]),
        qpos0_(new mjtNum[model_->nq]) {
    std::string task_name = spec.config["task_name"_];
    if (task_name == "stand") {
      hopping_ = false;
    } else if (task_name == "hop") {
      hopping_ = true;
    } else {
      throw std::runtime_error("Unknown task_name for dmc hopper.");
    }
  }

  void TaskInitializeEpisode() override {
    // randomizers.randomize_limited_and_rotational_joints(physics, self.random)
    RandomizeLimitedAndRotationalJoints(&gen_);
    std::memcpy(qpos0_.get(), data_->qpos, sizeof(mjtNum) * model_->nq);
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    ControlReset();
    WriteState();
  }

  void Step(const Action& action) override {
    // step
    mjtNum* act = static_cast<mjtNum*>(action["action"_].Data());
    ControlStep(act);
    WriteState();
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
    return data_->xipos[5] - data_->xipos[17];
  }

  mjtNum Speed() {
    // return self.named.data.sensordata['torso_subtreelinvel'][0]
    return data_->sensordata[0];
  }

  std::array<mjtNum, 2> Touch() {
    // return np.log1p(self.named.data.sensordata[['touch_toe', 'touch_heel']])
    return std::array<mjtNum, 2>{std::log1p(data_->sensordata[3]),
                                 std::log1p(data_->sensordata[4])};
  }

  void WriteState() {
    State state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    // obs
    state["obs:position"_].Assign(data_->qpos + 1, model_->nq - 1);
    state["obs:velocity"_].Assign(data_->qvel, model_->nv);
    const auto& touch = Touch();
    state["obs:touch"_].Assign(touch.begin(), 2);
    // info for check alignment
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
#endif
  }
};

using HopperEnvPool = AsyncEnvPool<HopperEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_HOPPER_H_
