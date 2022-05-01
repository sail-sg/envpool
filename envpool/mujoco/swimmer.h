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

#ifndef ENVPOOL_MUJOCO_SWIMMER_H_
#define ENVPOOL_MUJOCO_SWIMMER_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/mujoco_env.h"

namespace mujoco {

class SwimmerEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "max_episode_steps"_.bind(1000), "reward_threshold"_.bind(360.0),
        "frame_skip"_.bind(4), "post_constraint"_.bind(true),
        "exclude_current_positions_from_observation"_.bind(true),
        "forward_reward_weight"_.bind(1.0), "ctrl_cost_weight"_.bind(1e-4),
        "reset_noise_scale"_.bind(0.1));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    bool no_pos = conf["exclude_current_positions_from_observation"_];
    return MakeDict("obs"_.bind(Spec<mjtNum>({no_pos ? 8 : 10}, {-inf, inf})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.bind(Spec<mjtNum>({5})),
                    "info:qvel0"_.bind(Spec<mjtNum>({5})),
#endif
                    "info:reward_fwd"_.bind(Spec<mjtNum>({-1})),
                    "info:reward_ctrl"_.bind(Spec<mjtNum>({-1})),
                    "info:x_position"_.bind(Spec<mjtNum>({-1})),
                    "info:y_position"_.bind(Spec<mjtNum>({-1})),
                    "info:distance_from_origin"_.bind(Spec<mjtNum>({-1})),
                    "info:x_velocity"_.bind(Spec<mjtNum>({-1})),
                    "info:y_velocity"_.bind(Spec<mjtNum>({-1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.bind(Spec<mjtNum>({-1, 2}, {-1.0f, 1.0f})));
  }
};

typedef class EnvSpec<SwimmerEnvFns> SwimmerEnvSpec;

class SwimmerEnv : public Env<SwimmerEnvSpec>, public MujocoEnv {
 protected:
  bool no_pos_;
  mjtNum ctrl_cost_weight_, forward_reward_weight_;
  std::uniform_real_distribution<> dist_;

 public:
  SwimmerEnv(const Spec& spec, int env_id)
      : Env<SwimmerEnvSpec>(spec, env_id),
        MujocoEnv(spec.config["base_path"_] + "/mujoco/assets/swimmer.xml",
                  spec.config["frame_skip"_], spec.config["post_constraint"_],
                  spec.config["max_episode_steps"_]),
        no_pos_(spec.config["exclude_current_positions_from_observation"_]),
        ctrl_cost_weight_(spec.config["ctrl_cost_weight"_]),
        forward_reward_weight_(spec.config["forward_reward_weight"_]),
        dist_(-spec.config["reset_noise_scale"_],
              spec.config["reset_noise_scale"_]) {}

  void MujocoResetModel() {
    for (int i = 0; i < model_->nq; ++i) {
      data_->qpos[i] = qpos0_[i] = init_qpos_[i] + dist_(gen_);
    }
    for (int i = 0; i < model_->nv; ++i) {
      data_->qvel[i] = qvel0_[i] = init_qvel_[i] + dist_(gen_);
    }
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    MujocoReset();
    WriteState(0.0f, 0, 0, 0, 0, 0);
  }

  void Step(const Action& action) override {
    // step
    mjtNum* act = static_cast<mjtNum*>(action["action"_].data());
    mjtNum x_before = data_->qpos[0], y_before = data_->qpos[1];
    MujocoStep(act);
    mjtNum x_after = data_->qpos[0], y_after = data_->qpos[1];

    // ctrl_cost
    mjtNum ctrl_cost = 0.0;
    for (int i = 0; i < model_->nu; ++i) {
      ctrl_cost += ctrl_cost_weight_ * act[i] * act[i];
    }
    // xv and yv
    mjtNum dt = frame_skip_ * model_->opt.timestep;
    mjtNum xv = (x_after - x_before) / dt;
    mjtNum yv = (y_after - y_before) / dt;

    // reward and done
    float reward = xv * forward_reward_weight_ - ctrl_cost;
    done_ = (++elapsed_step_ >= max_episode_steps_);
    WriteState(reward, xv, yv, ctrl_cost, x_after, y_after);
  }

 private:
  void WriteState(float reward, mjtNum xv, mjtNum yv, mjtNum ctrl_cost,
                  mjtNum x_after, mjtNum y_after) {
    State state = Allocate();
    state["reward"_] = reward;
    // obs
    mjtNum* obs = static_cast<mjtNum*>(state["obs"_].data());
    for (int i = no_pos_ ? 2 : 0; i < model_->nq; ++i) {
      *(obs++) = data_->qpos[i];
    }
    for (int i = 0; i < model_->nv; ++i) {
      *(obs++) = data_->qvel[i];
    }
    // info
    state["info:reward_fwd"_] = xv * forward_reward_weight_;
    state["info:reward_ctrl"_] = -ctrl_cost;
    state["info:x_position"_] = x_after;
    state["info:y_position"_] = y_after;
    state["info:distance_from_origin"_] =
        std::sqrt(x_after * x_after + y_after * y_after);
    state["info:x_velocity"_] = xv;
    state["info:y_velocity"_] = yv;
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_, model_->nq);
    state["info:qvel0"_].Assign(qvel0_, model_->nv);
#endif
  }
};

typedef AsyncEnvPool<SwimmerEnv> SwimmerEnvPool;

}  // namespace mujoco

#endif  // ENVPOOL_MUJOCO_SWIMMER_H_
