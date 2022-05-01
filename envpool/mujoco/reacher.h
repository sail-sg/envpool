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

#ifndef ENVPOOL_MUJOCO_REACHER_H_
#define ENVPOOL_MUJOCO_REACHER_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/mujoco_env.h"

namespace mujoco {

class ReacherEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "max_episode_steps"_.bind(50), "reward_threshold"_.bind(-3.75),
        "frame_skip"_.bind(2), "post_constraint"_.bind(true),
        "ctrl_cost_weight"_.bind(1.0), "dist_cost_weight"_.bind(1.0),
        "reset_qpos_scale"_.bind(0.1), "reset_qvel_scale"_.bind(0.005),
        "reset_goal_scale"_.bind(0.2));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict("obs"_.bind(Spec<mjtNum>({11}, {-inf, inf})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.bind(Spec<mjtNum>({4})),
                    "info:qvel0"_.bind(Spec<mjtNum>({4})),
#endif
                    "info:reward_dist"_.bind(Spec<mjtNum>({-1})),
                    "info:reward_ctrl"_.bind(Spec<mjtNum>({-1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.bind(Spec<mjtNum>({-1, 2}, {-1.0f, 1.0f})));
  }
};

typedef class EnvSpec<ReacherEnvFns> ReacherEnvSpec;

class ReacherEnv : public Env<ReacherEnvSpec>, public MujocoEnv {
 protected:
  mjtNum ctrl_cost_weight_, dist_cost_weight_;
  mjtNum reset_goal_scale_, dist_x_, dist_y_, dist_z_;
  std::uniform_real_distribution<> dist_qpos_, dist_qvel_, dist_goal_;

 public:
  ReacherEnv(const Spec& spec, int env_id)
      : Env<ReacherEnvSpec>(spec, env_id),
        MujocoEnv(spec.config["base_path"_] + "/mujoco/assets/reacher.xml",
                  spec.config["frame_skip"_], spec.config["post_constraint"_],
                  spec.config["max_episode_steps"_]),
        ctrl_cost_weight_(spec.config["ctrl_cost_weight"_]),
        dist_cost_weight_(spec.config["dist_cost_weight"_]),
        reset_goal_scale_(spec.config["reset_goal_scale"_]),
        dist_qpos_(-spec.config["reset_qpos_scale"_],
                   spec.config["reset_qpos_scale"_]),
        dist_qvel_(-spec.config["reset_qvel_scale"_],
                   spec.config["reset_qvel_scale"_]),
        dist_goal_(-spec.config["reset_goal_scale"_],
                   spec.config["reset_goal_scale"_]) {}

  void MujocoResetModel() {
    for (int i = 0; i < model_->nq - 2; ++i) {
      data_->qpos[i] = qpos0_[i] = init_qpos_[i] + dist_qpos_(gen_);
    }
    while (1) {
      mjtNum x = dist_goal_(gen_), y = dist_goal_(gen_);
      if (std::sqrt(x * x + y * y) < reset_goal_scale_) {
        data_->qpos[model_->nq - 2] = qpos0_[model_->nq - 2] = x;
        data_->qpos[model_->nq - 1] = qpos0_[model_->nq - 1] = y;
        break;
      }
    }
    for (int i = 0; i < model_->nv; ++i) {
      data_->qvel[i] = qvel0_[i] =
          i < model_->nv - 2 ? init_qvel_[i] + dist_qvel_(gen_) : 0.0;
    }
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    MujocoReset();
    WriteState(0.0f, 0, 0);
  }

  void Step(const Action& action) override {
    // step
    mjtNum* act = static_cast<mjtNum*>(action["action"_].data());
    GetDist();
    MujocoStep(act);

    // dist_cost
    mjtNum dist_cost =
        dist_cost_weight_ *
        std::sqrt(dist_x_ * dist_x_ + dist_y_ * dist_y_ + dist_z_ * dist_z_);
    // ctrl_cost
    mjtNum ctrl_cost = 0.0;
    for (int i = 0; i < model_->nu; ++i) {
      ctrl_cost += ctrl_cost_weight_ * act[i] * act[i];
    }

    // reward and done
    float reward = -dist_cost - ctrl_cost;
    done_ = (++elapsed_step_ >= max_episode_steps_);
    WriteState(reward, ctrl_cost, dist_cost);
  }

 private:
  void GetDist() {
    dist_x_ = data_->xpos[9] - data_->xpos[12];
    dist_y_ = data_->xpos[10] - data_->xpos[13];
    dist_z_ = data_->xpos[11] - data_->xpos[14];
  }

  void WriteState(float reward, mjtNum ctrl_cost, mjtNum dist_cost) {
    State state = Allocate();
    state["reward"_] = reward;
    // obs
    mjtNum* obs = static_cast<mjtNum*>(state["obs"_].data());
    *(obs++) = std::cos(data_->qpos[0]);
    *(obs++) = std::cos(data_->qpos[1]);
    *(obs++) = std::sin(data_->qpos[0]);
    *(obs++) = std::sin(data_->qpos[1]);
    for (int i = 2; i < model_->nq; ++i) {
      *(obs++) = data_->qpos[i];
    }
    for (int i = 0; i < 2; ++i) {
      *(obs++) = data_->qvel[i];
    }
    GetDist();
    *(obs++) = dist_x_;
    *(obs++) = dist_y_;
    *(obs++) = dist_z_;
    // info
    state["info:reward_dist"_] = -dist_cost;
    state["info:reward_ctrl"_] = -ctrl_cost;
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_, model_->nq);
    state["info:qvel0"_].Assign(qvel0_, model_->nv);
#endif
  }
};

typedef AsyncEnvPool<ReacherEnv> ReacherEnvPool;

}  // namespace mujoco

#endif  // ENVPOOL_MUJOCO_REACHER_H_
