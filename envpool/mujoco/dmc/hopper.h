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

namespace mujoco {

class HopperEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.Bind(1000), "frame_skip"_.Bind(4),
                    "raw_xml"_.Bind(std::string("")),
                    "task_name"_.Bind(std::string("stand")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs:position"_.Bind(Spec<mjtNum>({6})),
                    "obs:velocity"_.Bind(Spec<mjtNum>({7})),
                    "obs:touch"_.Bind(Spec<mjtNum>({2})),
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

 public:
  HopperEnv(const Spec& spec, int env_id)
      : Env<HopperEnvSpec>(spec, env_id),
        MujocoEnv(spec.config["base_path"_], spec.config["raw_xml"_],
                  spec.config["frame_skip"_],
                  spec.config["max_episode_steps"_]) {
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

  float TaskGetReward() {
    // standing = rewards.tolerance(physics.height(), (_STAND_HEIGHT, 2))
    // if self._hopping:
    //   hopping = rewards.tolerance(physics.speed(),
    //                               bounds=(_HOP_SPEED, float('inf')),
    //                               margin=_HOP_SPEED/2,
    //                               value_at_margin=0.5,
    //                               sigmoid='linear')
    //   return standing * hopping
    // else:
    //   small_control = rewards.tolerance(physics.control(),
    //                                     margin=1, value_at_margin=0,
    //                                     sigmoid='quadratic').mean()
    //   small_control = (small_control + 4) / 5
    //   return standing * small_control
    return 0.0;
  }

  bool TaskShouldTerminateEpisode() {
    //
    return false;
  }

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

  void WriteState() {
    State state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    // obs
    state["obs:position"_].Assign(data_->qpos_ + 1, model_->nq - 1);
    state["obs:velocity"_].Assign(data_->qvel, model_->nv);
    mjtNum* obs_touch = static_cast<mjtNum*>(state["obs:touch"_].Data());
    obs_touch[0] = std::log(1 + data_->sensordata[3]);
    obs_touch[1] = std::log(1 + data_->sensordata[4]);
  }
};

using HopperEnvPool = AsyncEnvPool<HopperEnv>;

}  // namespace mujoco

#endif  // ENVPOOL_MUJOCO_DMC_HOPPER_H_
