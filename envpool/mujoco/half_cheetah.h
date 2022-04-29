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

#ifndef ENVPOOL_MUJOCO_HALF_CHEETAH_H_
#define ENVPOOL_MUJOCO_HALF_CHEETAH_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/mujoco_env.h"

namespace mujoco {

class HalfCheetahEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "max_episode_steps"_.bind(1000), "reward_threshold"_.bind(4800.0),
        "frame_skip"_.bind(5), "post_constraint"_.bind(true),
        "ctrl_cost_weight"_.bind(0.1), "forward_reward_weight"_.bind(1.0),
        "reset_noise_scale"_.bind(0.1));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict("obs"_.bind(Spec<mjtNum>({17}, {-inf, inf})),
                    "info:reward_run"_.bind(Spec<mjtNum>({-1})),
                    "info:reward_ctrl"_.bind(Spec<mjtNum>({-1})),
                    "info:x_position"_.bind(Spec<mjtNum>({-1})),
                    "info:x_velocity"_.bind(Spec<mjtNum>({-1})),
                    // TODO(jiayi): remove these two lines for speed
                    "info:qpos0"_.bind(Spec<mjtNum>({9})),
                    "info:qvel0"_.bind(Spec<mjtNum>({9})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.bind(Spec<mjtNum>({-1, 6}, {-1.0f, 1.0f})));
  }
};

typedef class EnvSpec<HalfCheetahEnvFns> HalfCheetahEnvSpec;

class HalfCheetahEnv : public Env<HalfCheetahEnvSpec>, public MujocoEnv {
 protected:
  mjtNum ctrl_cost_weight_, forward_reward_weight_;
  std::uniform_real_distribution<> dist_qpos_;
  std::normal_distribution<> dist_qvel_;

 public:
  HalfCheetahEnv(const Spec& spec, int env_id)
      : Env<HalfCheetahEnvSpec>(spec, env_id),
        MujocoEnv(spec.config["base_path"_] + "/mujoco/assets/half_cheetah.xml",
                  spec.config["frame_skip"_], spec.config["post_constraint"_],
                  spec.config["max_episode_steps"_]),
        ctrl_cost_weight_(spec.config["ctrl_cost_weight"_]),
        forward_reward_weight_(spec.config["forward_reward_weight"_]),
        dist_qpos_(-spec.config["reset_noise_scale"_],
                   spec.config["reset_noise_scale"_]),
        dist_qvel_(0, spec.config["reset_noise_scale"_]) {}

  void MujocoResetModel() {
    for (int i = 0; i < model_->nq; ++i) {
      data_->qpos[i] = qpos0_[i] = init_qpos_[i] + dist_qpos_(gen_);
    }
    for (int i = 0; i < model_->nv; ++i) {
      data_->qvel[i] = qvel0_[i] = init_qvel_[i] + dist_qvel_(gen_);
    }
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    current_step_ = 0;
    MujocoReset();
    WriteObs(0.0f, 0, 0, 0);
  }

  void Step(const Action& action) override {
    // step
    mjtNum* act = static_cast<mjtNum*>(action["action"_].data());
    mjtNum x_before = data_->qpos[0];
    MujocoStep(act);
    mjtNum x_after = data_->qpos[0];

    // ctrl_cost
    mjtNum ctrl_cost = 0.0;
    for (int i = 0; i < model_->nu; ++i) {
      ctrl_cost += ctrl_cost_weight_ * act[i] * act[i];
    }
    // xv
    mjtNum dt = frame_skip_ * model_->opt.timestep;
    mjtNum xv = (x_after - x_before) / dt;
    // reward and done
    float reward = xv * forward_reward_weight_ - ctrl_cost;
    done_ = (++current_step_ >= max_episode_steps_);
    WriteObs(reward, xv, ctrl_cost, x_after);
  }

 private:
  void WriteObs(float reward, mjtNum xv, mjtNum ctrl_cost, mjtNum x_after) {
    State state = Allocate();
    state["reward"_] = reward;
    // obs
    mjtNum* obs = static_cast<mjtNum*>(state["obs"_].data());
    for (int i = 1; i < model_->nq; ++i) {
      *(obs++) = data_->qpos[i];
    }
    for (int i = 0; i < model_->nv; ++i) {
      *(obs++) = data_->qvel[i];
    }
    // info
    state["info:reward_run"_] = xv * forward_reward_weight_;
    state["info:reward_ctrl"_] = -ctrl_cost;
    state["info:x_position"_] = x_after;
    state["info:x_velocity"_] = xv;
    state["info:qpos0"_].Assign(qpos0_, model_->nq);
    state["info:qvel0"_].Assign(qvel0_, model_->nv);
  }
};

typedef AsyncEnvPool<HalfCheetahEnv> HalfCheetahEnvPool;

}  // namespace mujoco

#endif  // ENVPOOL_MUJOCO_HALF_CHEETAH_H_
