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
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/ball_in_cup.py

#ifndef ENVPOOL_MUJOCO_DMC_BALL_IN_CUP_H_
#define ENVPOOL_MUJOCO_DMC_BALL_IN_CUP_H_

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

std::string GetBallInCupXML(const std::string& base_path,
                            const std::string& task_name) {
  return GetFileContent(base_path, "ball_in_cup.xml");
}

class BallInCupEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.Bind(1000), "frame_skip"_.Bind(1),
                    "task_name"_.Bind(std::string("catch")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs:position"_.Bind(Spec<mjtNum>({4})),
                    "obs:velocity"_.Bind(Spec<mjtNum>({4})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.Bind(Spec<mjtNum>({4})),
#endif
                    "discount"_.Bind(Spec<float>({-1}, {0.0, 1.0})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 2}, {-1.0, 1.0})));
  }
};

using BallInCupEnvSpec = EnvSpec<BallInCupEnvFns>;

class BallInCupEnv : public Env<BallInCupEnvSpec>, public MujocoEnv {
 protected:
  std::uniform_real_distribution<> dist_uniform_;

 public:
  BallInCupEnv(const Spec& spec, int env_id)
      : Env<BallInCupEnvSpec>(spec, env_id),
        MujocoEnv(spec.config["base_path"_],
                  GetBallInCupXML(spec.config["base_path"_],
                                  spec.config["task_name"_]),
                  spec.config["frame_skip"_],
                  spec.config["max_episode_steps"_]),
        dist_uniform_(0, 1) {}

  void TaskInitializeEpisode() override {
    while (true) {
      // Assign a random ball position.
      data_->qpos[2] = dist_uniform_(gen_) * 0.4 - 0.2;  // ball_x
      data_->qpos[3] = dist_uniform_(gen_) * 0.3 + 0.2;  // ball_z
      PhysicsAfterReset();
      if (data_->ncon <= 0) {
        break
      }
    }
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

  float TaskGetReward() override { return static_cast<float>(InTarget()); }

  bool TaskShouldTerminateEpisode() override { return false; }

 private:
  void WriteState() {
    State state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    // obs
    state["obs:position"_].Assign(data_->qpos, model_->nq);
    state["obs:velocity"_].Assign(data_->qvel, model_->nv);
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
#endif
  }

  std::array<mjtNum, 2> BallToTarget() {
    std::array<mjtNum, 2> target;
    target[0] = data_->site_xpos[1 * 3];
    target[1] = data_->site_xpos[1 * 3 + 2];
    std::array<mjtNum, 2> ball;
    ball[0] = data_->site_xpos[2 * 3];
    ball[1] = data_->site_xpos[2 * 3 + 2];
    return std::array<mjtNum, 2>{target[0] - ball[0], target[1] - ball[1]};
  }

  mjtNum InTarget() {
    std::array<mjtNum, 2> ball_to_target = BallToTarget();
    for (int i = 0; i < 2; ++i) {
      if (ball_to_target[i] < 0) {
        ball_to_target[i] = -ball_to_target[i];
      }
    }
    std::array<mjtNum, 2> target_size;
    target_size[0] = model_->site_size[1 * 3];
    target_size[1] = model_->site_size[1 * 3 + 2];
    std::array<mjtNum, 2> ball_size;
    ball_size[0] = model_->geom_size[2 * 3];
    ball_size[1] = model_->geom_size[2 * 3];
    if (ball_to_target[0] < target_size[0] - ball_size[0] &&
        ball_to_target[1] < target_size[1] - ball_size[1]) {
      return true;
    } else {
      return false;
    }
  }
};

using BallInCupEnvPool = AsyncEnvPool<BallInCupEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_BALL_IN_CUP_H_
