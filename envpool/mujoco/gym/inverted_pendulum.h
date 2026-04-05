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

#ifndef ENVPOOL_MUJOCO_GYM_INVERTED_PENDULUM_H_
#define ENVPOOL_MUJOCO_GYM_INVERTED_PENDULUM_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/gym/mujoco_env.h"

namespace mujoco_gym {

class InvertedPendulumEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("reward_threshold"_.Bind(950.0), "frame_skip"_.Bind(2),
                    "frame_stack"_.Bind(1), "post_constraint"_.Bind(true),
                    "healthy_reward"_.Bind(1.0),
                    "reward_if_not_terminated"_.Bind(false),
                    "xml_file"_.Bind(std::string("inverted_pendulum.xml")),
                    "healthy_z_min"_.Bind(-0.2), "healthy_z_max"_.Bind(0.2),
                    "reset_noise_scale"_.Bind(0.01));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
#ifdef ENVPOOL_TEST
    return MakeDict("obs"_.Bind(StackSpec(Spec<mjtNum>({4}, {-inf, inf}),
                                          conf["frame_stack"_])),
                    "info:qpos0"_.Bind(Spec<mjtNum>({2})),
                    "info:qvel0"_.Bind(Spec<mjtNum>({2})));
#else
    return MakeDict("obs"_.Bind(
        StackSpec(Spec<mjtNum>({4}, {-inf, inf}), conf["frame_stack"_])));
#endif
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 1}, {-3.0, 3.0})));
  }
};

using InvertedPendulumEnvSpec = EnvSpec<InvertedPendulumEnvFns>;

class InvertedPendulumEnv : public Env<InvertedPendulumEnvSpec>,
                            public MujocoEnv {
 protected:
  bool reward_if_not_terminated_;
  mjtNum healthy_reward_, healthy_z_min_, healthy_z_max_;
  std::uniform_real_distribution<> dist_;

 public:
  InvertedPendulumEnv(const Spec& spec, int env_id)
      : Env<InvertedPendulumEnvSpec>(spec, env_id),
        MujocoEnv(
            std::string(spec.config["base_path"_]) + "/mujoco/assets_gym/" +
                std::string(spec.config["xml_file"_]),
            spec.config["frame_skip"_], spec.config["post_constraint"_],
            spec.config["max_episode_steps"_], spec.config["frame_stack"_]),
        reward_if_not_terminated_(spec.config["reward_if_not_terminated"_]),
        healthy_reward_(spec.config["healthy_reward"_]),
        healthy_z_min_(spec.config["healthy_z_min"_]),
        healthy_z_max_(spec.config["healthy_z_max"_]),
        dist_(-spec.config["reset_noise_scale"_],
              spec.config["reset_noise_scale"_]) {}

  void MujocoResetModel() override {
    for (int i = 0; i < model_->nq; ++i) {
      data_->qpos[i] = init_qpos_[i] + dist_(gen_);
    }
    for (int i = 0; i < model_->nv; ++i) {
      data_->qvel[i] = init_qvel_[i] + dist_(gen_);
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
    WriteState(0.0, true);
  }

  void Step(const Action& action) override {
    // step
    MujocoStep(static_cast<mjtNum*>(action["action"_].Data()));

    // reward and done
    bool terminated = !IsHealthy();
    ++elapsed_step_;
    done_ = terminated || (elapsed_step_ >= max_episode_steps_);
    WriteState(
        reward_if_not_terminated_ ? static_cast<float>(!terminated) : 1.0,
        false);
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

  void WriteState(float reward, bool reset) {
    auto state = Allocate();
    state["reward"_] = reward;
    // obs
    auto obs_state = state["obs"_];
    mjtNum* obs = PrepareObservation(&obs_state);
    for (int i = 0; i < model_->nq; ++i) {
      *(obs++) = data_->qpos[i];
    }
    for (int i = 0; i < model_->nv; ++i) {
      *(obs++) = data_->qvel[i];
    }
    CommitObservation(&obs_state, reset);
    // info
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_, model_->nq);
    state["info:qvel0"_].Assign(qvel0_, model_->nv);
#endif
  }
};

using InvertedPendulumEnvPool = AsyncEnvPool<InvertedPendulumEnv>;

}  // namespace mujoco_gym

#endif  // ENVPOOL_MUJOCO_GYM_INVERTED_PENDULUM_H_
