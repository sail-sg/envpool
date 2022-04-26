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

#ifndef ENVPOOL_MUJOCO_PUSHER_H_
#define ENVPOOL_MUJOCO_PUSHER_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/mujoco_env.h"

namespace mujoco {

class PusherEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "max_episode_steps"_.bind(100), "reward_threshold"_.bind(0.0),
        "frame_skip"_.bind(5), "post_constraint"_.bind(true),
        "ctrl_cost_weight"_.bind(0.1), "dist_cost_weight"_.bind(1.0),
        "near_cost_weight"_.bind(0.5), "reset_qvel_scale"_.bind(0.005),
        "cylinder_x_min"_.bind(-0.3), "cylinder_x_max"_.bind(0.0),
        "cylinder_y_min"_.bind(-0.2), "cylinder_y_max"_.bind(0.2),
        "cylinder_dist_min"_.bind(0.17));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict("obs"_.bind(Spec<mjtNum>({23}, {-inf, inf})),
                    "info:reward_dist"_.bind(Spec<mjtNum>({-1})),
                    "info:reward_ctrl"_.bind(Spec<mjtNum>({-1})),
                    // TODO(jiayi): remove these two lines for speed
                    "info:qpos0"_.bind(Spec<mjtNum>({11})),
                    "info:qvel0"_.bind(Spec<mjtNum>({11})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.bind(Spec<mjtNum>({-1, 7}, {-2.0f, 2.0f})));
  }
};

typedef class EnvSpec<PusherEnvFns> PusherEnvSpec;

class PusherEnv : public Env<PusherEnvSpec>, public MujocoEnv {
 protected:
  int max_episode_steps_, elapsed_step_;
  mjtNum ctrl_cost_weight_, dist_cost_weight_, near_cost_weight_;
  mjtNum cylinder_dist_min_;
  std::unique_ptr<mjtNum> qpos0_, qvel0_;  // for align check
  std::uniform_real_distribution<> dist_qpos_x_, dist_qpos_y_, dist_qvel_;
  bool done_;

 public:
  PusherEnv(const Spec& spec, int env_id)
      : Env<PusherEnvSpec>(spec, env_id),
        MujocoEnv(spec.config["base_path"_] + "/mujoco/assets/pusher.xml",
                  spec.config["frame_skip"_], spec.config["post_constraint"_]),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        elapsed_step_(max_episode_steps_ + 1),
        ctrl_cost_weight_(spec.config["ctrl_cost_weight"_]),
        dist_cost_weight_(spec.config["dist_cost_weight"_]),
        near_cost_weight_(spec.config["near_cost_weight"_]),
        cylinder_dist_min_(spec.config["cylinder_dist_min"_]),
        qpos0_(new mjtNum[model_->nq]),
        qvel0_(new mjtNum[model_->nv]),
        dist_qpos_x_(spec.config["cylinder_x_min"_],
                     spec.config["cylinder_x_max"_]),
        dist_qpos_y_(spec.config["cylinder_y_min"_],
                     spec.config["cylinder_y_max"_]),
        dist_qvel_(-spec.config["reset_qvel_scale"_],
                   spec.config["reset_qvel_scale"_]),
        done_(true) {}

  void MujocoResetModel() {
    for (int i = 0; i < model_->nq - 4; ++i) {
      data_->qpos[i] = qpos0_.get()[i] = init_qpos_[i];
    }
    while (1) {
      mjtNum x = dist_qpos_x_(gen_), y = dist_qpos_y_(gen_);
      if (std::sqrt(x * x + y * y) > cylinder_dist_min_) {
        data_->qpos[model_->nq - 4] = qpos0_.get()[model_->nq - 4] = x;
        data_->qpos[model_->nq - 3] = qpos0_.get()[model_->nq - 3] = y;
        data_->qpos[model_->nq - 2] = qpos0_.get()[model_->nq - 2] = 0.0;
        data_->qpos[model_->nq - 1] = qpos0_.get()[model_->nq - 1] = 0.0;
        break;
      }
    }
    for (int i = 0; i < model_->nv; ++i) {
      data_->qvel[i] = qvel0_.get()[i] =
          i < model_->nv - 4 ? init_qvel_[i] + dist_qvel_(gen_) : 0.0;
    }
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    MujocoReset();
    WriteObs(0.0f, 0, 0);
  }

  void Step(const Action& action) override {
    // step
    mjtNum* act = static_cast<mjtNum*>(action["action"_].data());
    mjtNum near_cost = GetDist(30, 33);
    mjtNum dist_cost = GetDist(33, 36);
    MujocoStep(act);

    // ctrl_cost
    mjtNum ctrl_cost = 0.0;
    for (int i = 0; i < model_->nu; ++i) {
      ctrl_cost += act[i] * act[i];
    }

    // reward and done
    float reward = -ctrl_cost * ctrl_cost_weight_ -
                   dist_cost * dist_cost_weight_ -
                   near_cost * near_cost_weight_;
    done_ = (++elapsed_step_ >= max_episode_steps_);
    WriteObs(reward, ctrl_cost, dist_cost);
  }

 private:
  mjtNum GetDist(int off0, int off1) {
    mjtNum x = data_->xpos[off0 + 0] - data_->xpos[off1 + 0];
    mjtNum y = data_->xpos[off0 + 1] - data_->xpos[off1 + 1];
    mjtNum z = data_->xpos[off0 + 2] - data_->xpos[off1 + 2];
    return std::sqrt(x * x + y * y + z * z);
  }

  void WriteObs(float reward, mjtNum ctrl_cost, mjtNum dist_cost) {
    State state = Allocate();
    state["reward"_] = reward;
    // obs
    mjtNum* obs = static_cast<mjtNum*>(state["obs"_].data());
    for (int i = 0; i < 7; ++i) {
      *(obs++) = data_->qpos[i];
    }
    for (int i = 0; i < 7; ++i) {
      *(obs++) = data_->qvel[i];
    }
    for (int i = 30; i < 3 * model_->nbody; ++i) {
      *(obs++) = data_->xpos[i];
    }
    // info
    state["info:reward_dist"_] = -dist_cost;
    state["info:reward_ctrl"_] = -ctrl_cost;
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
    state["info:qvel0"_].Assign(qvel0_.get(), model_->nv);
  }
};

typedef AsyncEnvPool<PusherEnv> PusherEnvPool;

}  // namespace mujoco

#endif  // ENVPOOL_MUJOCO_PUSHER_H_