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
    return MakeDict("max_episode_steps"_.Bind(1000), "frame_skip"_.Bind(10),
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
  int id_target_, id_ball_, id_ball_x_, id_ball_z_;
  std::uniform_real_distribution<> dist_ball_x_, dist_ball_z_;

 public:
  BallInCupEnv(const Spec& spec, int env_id)
      : Env<BallInCupEnvSpec>(spec, env_id),
        MujocoEnv(spec.config["base_path"_],
                  GetBallInCupXML(spec.config["base_path"_],
                                  spec.config["task_name"_]),
                  spec.config["frame_skip"_],
                  spec.config["max_episode_steps"_]),
        id_target_(mj_name2id(model_, mjOBJ_SITE, "target")),
        id_ball_(mj_name2id(model_, mjOBJ_XBODY, "ball")),
        id_ball_x_(mj_name2id(model_, mjOBJ_JOINT, "ball_x")),
        id_ball_z_(mj_name2id(model_, mjOBJ_JOINT, "ball_z")),
        dist_ball_x_(-0.2, 0.2),
        dist_ball_z_(0.2, 0.5) {}

  void TaskInitializeEpisode() override {
    while (true) {
      // Assign a random ball position.
      data_->qpos[id_ball_x_] = dist_ball_x_(gen_);
      data_->qpos[id_ball_z_] = dist_ball_z_(gen_);
#ifdef ENVPOOL_TEST
      std::memcpy(qpos0_.get(), data_->qpos, sizeof(mjtNum) * model_->nq);
#endif
      PhysicsAfterReset();
      if (data_->ncon <= 0) {
        break;
      }
    }
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
    // target = self.named.data.site_xpos['target', ['x', 'z']]
    // ball = self.named.data.xpos['ball', ['x', 'z']]
    // return target - ball
    std::array<mjtNum, 2> target{data_->site_xpos[id_target_ * 3],
                                 data_->site_xpos[id_target_ * 3 + 2]};
    std::array<mjtNum, 2> ball{data_->xpos[id_ball_ * 3],
                               data_->xpos[id_ball_ * 3 + 2]};
    return {target[0] - ball[0], target[1] - ball[1]};
  }

  bool InTarget() {
    // ball_to_target = abs(self.ball_to_target())
    // target_size = self.named.model.site_size['target', [0, 2]]
    // ball_size = self.named.model.geom_size['ball', 0]
    // return float(all(ball_to_target < target_size - ball_size))
    const auto& ball_to_target = BallToTarget();
    std::array<mjtNum, 2> target_size{model_->site_size[id_target_ * 3],
                                      model_->site_size[id_target_ * 3 + 2]};
    auto ball_size = model_->geom_size[id_ball_ * 3];
    return std::abs(ball_to_target[0]) < target_size[0] - ball_size &&
           std::abs(ball_to_target[1]) < target_size[1] - ball_size;
  }
};

using BallInCupEnvPool = AsyncEnvPool<BallInCupEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_BALL_IN_CUP_H_
