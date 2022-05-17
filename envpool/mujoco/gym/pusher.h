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

#ifndef ENVPOOL_MUJOCO_GYM_PUSHER_H_
#define ENVPOOL_MUJOCO_GYM_PUSHER_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/gym/mujoco_env.h"

namespace mujoco_gym {

class PusherEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "max_episode_steps"_.Bind(100), "reward_threshold"_.Bind(0.0),
        "frame_skip"_.Bind(5), "post_constraint"_.Bind(true),
        "ctrl_cost_weight"_.Bind(0.1), "dist_cost_weight"_.Bind(1.0),
        "near_cost_weight"_.Bind(0.5), "reset_qvel_scale"_.Bind(0.005),
        "cylinder_x_min"_.Bind(-0.3), "cylinder_x_max"_.Bind(0.0),
        "cylinder_y_min"_.Bind(-0.2), "cylinder_y_max"_.Bind(0.2),
        "cylinder_dist_min"_.Bind(0.17));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict("obs"_.Bind(Spec<mjtNum>({23}, {-inf, inf})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.Bind(Spec<mjtNum>({11})),
                    "info:qvel0"_.Bind(Spec<mjtNum>({11})),
#endif
                    "info:reward_dist"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_ctrl"_.Bind(Spec<mjtNum>({-1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 7}, {-2.0, 2.0})));
  }
};

using PusherEnvSpec = EnvSpec<PusherEnvFns>;

class PusherEnv : public Env<PusherEnvSpec>, public MujocoEnv {
 protected:
  int id_tips_arm_, id_object_, id_goal_;
  mjtNum ctrl_cost_weight_, dist_cost_weight_, near_cost_weight_;
  mjtNum cylinder_dist_min_;
  std::uniform_real_distribution<> dist_qpos_x_, dist_qpos_y_, dist_qvel_;

 public:
  PusherEnv(const Spec& spec, int env_id)
      : Env<PusherEnvSpec>(spec, env_id),
        MujocoEnv(spec.config["base_path"_] + "/mujoco/assets_gym/pusher.xml",
                  spec.config["frame_skip"_], spec.config["post_constraint"_],
                  spec.config["max_episode_steps"_]),
        id_tips_arm_(mj_name2id(model_, mjOBJ_XBODY, "tips_arm")),
        id_object_(mj_name2id(model_, mjOBJ_XBODY, "object")),
        id_goal_(mj_name2id(model_, mjOBJ_XBODY, "goal")),
        ctrl_cost_weight_(spec.config["ctrl_cost_weight"_]),
        dist_cost_weight_(spec.config["dist_cost_weight"_]),
        near_cost_weight_(spec.config["near_cost_weight"_]),
        cylinder_dist_min_(spec.config["cylinder_dist_min"_]),
        dist_qpos_x_(spec.config["cylinder_x_min"_],
                     spec.config["cylinder_x_max"_]),
        dist_qpos_y_(spec.config["cylinder_y_min"_],
                     spec.config["cylinder_y_max"_]),
        dist_qvel_(-spec.config["reset_qvel_scale"_],
                   spec.config["reset_qvel_scale"_]) {}

  void MujocoResetModel() override {
    for (int i = 0; i < model_->nq - 4; ++i) {
      data_->qpos[i] = init_qpos_[i];
    }
    while (true) {
      mjtNum x = dist_qpos_x_(gen_);
      mjtNum y = dist_qpos_y_(gen_);
      if (std::sqrt(x * x + y * y) > cylinder_dist_min_) {
        data_->qpos[model_->nq - 4] = x;
        data_->qpos[model_->nq - 3] = y;
        data_->qpos[model_->nq - 2] = 0.0;
        data_->qpos[model_->nq - 1] = 0.0;
        break;
      }
    }
    for (int i = 0; i < model_->nv; ++i) {
      data_->qvel[i] =
          i < model_->nv - 4 ? init_qvel_[i] + dist_qvel_(gen_) : 0.0;
    }
#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_, data_->qpos, sizeof(mjtNum) * model_->nq);
    std::memcpy(qvel0_, data_->qvel, sizeof(mjtNum) * model_->nv);
#endif
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    MujocoReset();
    WriteState(0.0, 0.0, 0.0);
  }

  void Step(const Action& action) override {
    // step
    mjtNum* act = static_cast<mjtNum*>(action["action"_].Data());
    mjtNum near_cost = GetDist(id_object_, id_tips_arm_);
    mjtNum dist_cost = GetDist(id_object_, id_goal_);
    MujocoStep(act);

    // ctrl_cost
    mjtNum ctrl_cost = 0.0;
    for (int i = 0; i < model_->nu; ++i) {
      ctrl_cost += act[i] * act[i];
    }

    // reward and done
    auto reward = static_cast<float>(-ctrl_cost * ctrl_cost_weight_ -
                                     dist_cost * dist_cost_weight_ -
                                     near_cost * near_cost_weight_);
    done_ = (++elapsed_step_ >= max_episode_steps_);
    WriteState(reward, ctrl_cost, dist_cost);
  }

 private:
  mjtNum GetDist(int off0, int off1) {
    mjtNum x = data_->xpos[off0 * 3 + 0] - data_->xpos[off1 * 3 + 0];
    mjtNum y = data_->xpos[off0 * 3 + 1] - data_->xpos[off1 * 3 + 1];
    mjtNum z = data_->xpos[off0 * 3 + 2] - data_->xpos[off1 * 3 + 2];
    return std::sqrt(x * x + y * y + z * z);
  }

  void WriteState(float reward, mjtNum ctrl_cost, mjtNum dist_cost) {
    State state = Allocate();
    state["reward"_] = reward;
    // obs
    mjtNum* obs = static_cast<mjtNum*>(state["obs"_].Data());
    for (int i = 0; i < 7; ++i) {
      *(obs++) = data_->qpos[i];
    }
    for (int i = 0; i < 7; ++i) {
      *(obs++) = data_->qvel[i];
    }
    for (int i = 0; i < 3; ++i) {
      *(obs++) = data_->xpos[id_tips_arm_ * 3 + i];
    }
    for (int i = 0; i < 3; ++i) {
      *(obs++) = data_->xpos[id_object_ * 3 + i];
    }
    for (int i = 0; i < 3; ++i) {
      *(obs++) = data_->xpos[id_goal_ * 3 + i];
    }
    // info
    state["info:reward_dist"_] = -dist_cost;
    state["info:reward_ctrl"_] = -ctrl_cost;
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_, model_->nq);
    state["info:qvel0"_].Assign(qvel0_, model_->nv);
#endif
  }
};

using PusherEnvPool = AsyncEnvPool<PusherEnv>;

}  // namespace mujoco_gym

#endif  // ENVPOOL_MUJOCO_GYM_PUSHER_H_
