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

#ifndef ENVPOOL_MUJOCO_INVERTED_PENDULUM_H_
#define ENVPOOL_MUJOCO_INVERTED_PENDULUM_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/mujoco_env.h"

namespace mujoco {

class InvertedPendulumEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.bind(1000),
                    "reward_threshold"_.bind(950.0), "frame_skip"_.bind(2),
                    "post_constraint"_.bind(true), "healthy_reward"_.bind(1.0),
                    "healthy_z_min"_.bind(-0.2), "healthy_z_max"_.bind(0.2),
                    "reset_noise_scale"_.bind(0.01));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict("obs"_.bind(Spec<mjtNum>({4}, {-inf, inf})),
                    // TODO(jiayi): remove these two lines for speed
                    "info:qpos0"_.bind(Spec<mjtNum>({2})),
                    "info:qvel0"_.bind(Spec<mjtNum>({2})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.bind(Spec<mjtNum>({-1, 1}, {-3.0f, 3.0f})));
  }
};

typedef class EnvSpec<InvertedPendulumEnvFns> InvertedPendulumEnvSpec;

class InvertedPendulumEnv : public Env<InvertedPendulumEnvSpec>,
                            public MujocoEnv {
 protected:
  mjtNum healthy_reward_, healthy_z_min_, healthy_z_max_;
  std::uniform_real_distribution<> dist_;

 public:
  InvertedPendulumEnv(const Spec& spec, int env_id)
      : Env<InvertedPendulumEnvSpec>(spec, env_id),
        MujocoEnv(
            spec.config["base_path"_] + "/mujoco/assets/inverted_pendulum.xml",
            spec.config["frame_skip"_], spec.config["post_constraint"_],
            spec.config["max_episode_steps"_]),
        healthy_reward_(spec.config["healthy_reward"_]),
        healthy_z_min_(spec.config["healthy_z_min"_]),
        healthy_z_max_(spec.config["healthy_z_max"_]),
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
    current_step_ = 0;
    MujocoReset();
    WriteObs(0.0f);
  }

  void Step(const Action& action) override {
    // step
    MujocoStep(static_cast<mjtNum*>(action["action"_].data()));

    // reward and done
    ++current_step_;
    done_ = !IsHealthy() || (current_step_ >= max_episode_steps_);
    WriteObs(1.0f);
  }

 private:
  bool IsHealthy() {
    if (data_->qpos[1] < healthy_z_min_ || data_->qpos[1] > healthy_z_max_) {
      return false;
    }
    for (int i = 0; i < model_->nq; ++i) {
      if (!std::isfinite(data_->qpos[i])) {
        return false;
      }
    }
    for (int i = 0; i < model_->nv; ++i) {
      if (!std::isfinite(data_->qvel[i])) {
        return false;
      }
    }
    return true;
  }

  void WriteObs(float reward) {
    State state = Allocate();
    state["reward"_] = reward;
    // obs
    mjtNum* obs = static_cast<mjtNum*>(state["obs"_].data());
    for (int i = 0; i < model_->nq; ++i) {
      *(obs++) = data_->qpos[i];
    }
    for (int i = 0; i < model_->nv; ++i) {
      *(obs++) = data_->qvel[i];
    }
    // info
    state["info:qpos0"_].Assign(qpos0_, model_->nq);
    state["info:qvel0"_].Assign(qvel0_, model_->nv);
  }
};

typedef AsyncEnvPool<InvertedPendulumEnv> InvertedPendulumEnvPool;

}  // namespace mujoco

#endif  // ENVPOOL_MUJOCO_INVERTED_PENDULUM_H_
