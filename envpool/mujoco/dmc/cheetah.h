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

#ifndef ENVPOOL_MUJOCO_DMC_CHEETAH_H_
#define ENVPOOL_MUJOCO_DMC_CHEETAH_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/dmc/mujoco_env.h"
#include "envpool/mujoco/dmc/utils.h"

namespace mujoco_dmc {

std::string GetCheetahXML(const std::string& base_path,
                          const std::string& task_name) {
  return GetFileContent(base_path, "cheetah.xml");
}

class CheetahEnvFns {
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.Bind(1000), "frame_skip"_.Bind(4),
                    "task_name"_.Bind(std::string("stand")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs:position"_.Bind(Spec<mjtNum>({8})),
                    "obs:velocity"_.Bind(Spec<mjtNum>({9})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 6}, {-1.0, 1.0})));
  }
};

using CheetahEnvSpec = EnvSpec<CheetahEnvFns>;

// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/cheetah.py#L60
class CheetahEnv : public Env<CheetahEnvSpec>, public MujocoEnv {
  const mjtNum kRunSpeed = 10;

 public:
  CheetahEnv(const Spec& spec, int env_id)
      : Env<CheetahEnvSpec>(spec, env_id),
        MujocoEnv(
            spec.config["base_path"_],
            GetCheetahXML(spec.config["base_path"_], spec.config["task_name"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_]) {}

  void TaskInitializeEpisode() override {
    assert(model_->njnt == model_->nq);
    int is_limited = int(model_->jnt_limited) == 1;
    mjtNum range_min = model_->jnt_range[is_limited * 2 + 0];
    mjtNum range_max = model_->jnt_range[is_limited * 2 + 1];
    mjtNum range = range_max - range_min;
    data_->qpos[is_limited] = dist_uniform_(gen_) * range + range_min;
    for (int i = 0; i < 200; i++) PhysicsStep(200, NULL);
    data_->time = 0;
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
    return static_cast<float>(RewardTolerance(
        Speed(), kRunSpeed, std::numeric_limits<double>::infinity(), kRunSpeed,
        0, SigmoidType::kQuadratic));
  }

  bool TaskShouldTerminateEpisode() override { return false; }

 private:
  void WriteState() {
    State state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    // obs
    state["obs:position"_].Assign(data_->qpos + 1, model_->nq - 1);  // ?
    state["obs:velocity"_].Assign(data_->qvel, model_->nv);
  }

  mjtNum Speed() {
    // return self.named.data.sensordata['torso_subtreelinvel'][0]
    return data_->sensordata[0];
  }
};

using CheetahEnvPool = AsyncEnvPool<CheetahEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_CHEETAH_H_