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
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/cheetah.py

#ifndef ENVPOOL_MUJOCO_DMC_CHEETAH_H_
#define ENVPOOL_MUJOCO_DMC_CHEETAH_H_

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

std::string GetCheetahXML(const std::string& base_path,
                          const std::string& task_name) {
  return GetFileContent(base_path, "cheetah.xml");
}

class CheetahEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.Bind(1000), "frame_skip"_.Bind(1),
                    "task_name"_.Bind(std::string("run")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs:position"_.Bind(Spec<mjtNum>({8})),
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

using CheetahEnvSpec = EnvSpec<CheetahEnvFns>;

class CheetahEnv : public Env<CheetahEnvSpec>, public MujocoEnv {
 protected:
  const mjtNum kRunSpeed = 10;
  int id_torso_subtreelinvel_;
  std::uniform_real_distribution<> dist_uniform_;

 public:
  CheetahEnv(const Spec& spec, int env_id)
      : Env<CheetahEnvSpec>(spec, env_id),
        MujocoEnv(
            spec.config["base_path"_],
            GetCheetahXML(spec.config["base_path"_], spec.config["task_name"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_]),
        id_torso_subtreelinvel_(
            mj_name2id(model_, mjOBJ_SENSOR, "torso_subtreelinvel")),
        dist_uniform_(0, 1) {}

  void TaskInitializeEpisode() override {
    assert(model_->njnt == model_->nq);
    for (int i = 0; i < model_->njnt; i++) {
      bool is_limited = model_->jnt_limited[i] == 1;
      if (is_limited) {
        mjtNum range_min = model_->jnt_range[i * 2 + 0];
        mjtNum range_max = model_->jnt_range[i * 2 + 1];
        mjtNum range = range_max - range_min;
        data_->qpos[i] = dist_uniform_(gen_) * range + range_min;
      }
    }
#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_.get(), data_->qpos, sizeof(mjtNum) * model_->nq);
#endif
    PhysicsStep(200, nullptr);
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
        0, SigmoidType::kLinear));
  }

  bool TaskShouldTerminateEpisode() override { return false; }

 private:
  void WriteState() {
    State state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    // obs
    state["obs:position"_].Assign(data_->qpos + 1, model_->nq - 1);
    state["obs:velocity"_].Assign(data_->qvel, model_->nv);
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
#endif
  }

  mjtNum Speed() {
    // return self.named.data.sensordata['torso_subtreelinvel'][0]
    return data_->sensordata[id_torso_subtreelinvel_];
  }
};

using CheetahEnvPool = AsyncEnvPool<CheetahEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_CHEETAH_H_
