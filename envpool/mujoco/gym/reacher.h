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

#ifndef ENVPOOL_MUJOCO_GYM_REACHER_H_
#define ENVPOOL_MUJOCO_GYM_REACHER_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/gym/mujoco_env.h"

namespace mujoco_gym {

class ReacherEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "max_episode_steps"_.Bind(50), "reward_threshold"_.Bind(-3.75),
        "frame_skip"_.Bind(2), "post_constraint"_.Bind(true),
        "ctrl_cost_weight"_.Bind(1.0), "dist_cost_weight"_.Bind(1.0),
        "reset_qpos_scale"_.Bind(0.1), "reset_qvel_scale"_.Bind(0.005),
        "reset_goal_scale"_.Bind(0.2));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict("obs"_.Bind(Spec<mjtNum>({11}, {-inf, inf})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.Bind(Spec<mjtNum>({4})),
                    "info:qvel0"_.Bind(Spec<mjtNum>({4})),
#endif
                    "info:reward_dist"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_ctrl"_.Bind(Spec<mjtNum>({-1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 2}, {-1.0, 1.0})));
  }
};

using ReacherEnvSpec = EnvSpec<ReacherEnvFns>;

class ReacherEnv : public Env<ReacherEnvSpec>, public MujocoEnv {
 protected:
  int id_fingertip_, id_target_;
  mjtNum ctrl_cost_weight_, dist_cost_weight_, reset_goal_scale_;
  std::uniform_real_distribution<> dist_qpos_, dist_qvel_, dist_goal_;

 public:
  ReacherEnv(const Spec& spec, int env_id)
      : Env<ReacherEnvSpec>(spec, env_id),
        MujocoEnv(spec.config["base_path"_] + "/mujoco/assets_gym/reacher.xml",
                  spec.config["frame_skip"_], spec.config["post_constraint"_],
                  spec.config["max_episode_steps"_]),
        id_fingertip_(mj_name2id(model_, mjOBJ_BODY, "fingertip")),
        id_target_(mj_name2id(model_, mjOBJ_BODY, "target")),
        ctrl_cost_weight_(spec.config["ctrl_cost_weight"_]),
        dist_cost_weight_(spec.config["dist_cost_weight"_]),
        reset_goal_scale_(spec.config["reset_goal_scale"_]),
        dist_qpos_(-spec.config["reset_qpos_scale"_],
                   spec.config["reset_qpos_scale"_]),
        dist_qvel_(-spec.config["reset_qvel_scale"_],
                   spec.config["reset_qvel_scale"_]),
        dist_goal_(-spec.config["reset_goal_scale"_],
                   spec.config["reset_goal_scale"_]) {}

  void MujocoResetModel() override {
    for (int i = 0; i < model_->nq - 2; ++i) {
      data_->qpos[i] = init_qpos_[i] + dist_qpos_(gen_);
    }
    while (true) {
      mjtNum x = dist_goal_(gen_);
      mjtNum y = dist_goal_(gen_);
      if (std::sqrt(x * x + y * y) < reset_goal_scale_) {
        data_->qpos[model_->nq - 2] = x;
        data_->qpos[model_->nq - 1] = y;
        break;
      }
    }
    for (int i = 0; i < model_->nv; ++i) {
      data_->qvel[i] =
          i < model_->nv - 2 ? init_qvel_[i] + dist_qvel_(gen_) : 0.0;
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
    const auto& dist = GetDist();
    MujocoStep(act);

    // dist_cost
    mjtNum dist_cost =
        dist_cost_weight_ *
        std::sqrt(dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2]);
    // ctrl_cost
    mjtNum ctrl_cost = 0.0;
    for (int i = 0; i < model_->nu; ++i) {
      ctrl_cost += ctrl_cost_weight_ * act[i] * act[i];
    }

    // reward and done
    auto reward = static_cast<float>(-dist_cost - ctrl_cost);
    done_ = (++elapsed_step_ >= max_episode_steps_);
    WriteState(reward, ctrl_cost, dist_cost);
  }

 private:
  std::array<mjtNum, 3> GetDist() {
    // self.get_body_com("fingertip") - self.get_body_com("target")
    return {
        data_->xpos[3 * id_fingertip_ + 0] - data_->xpos[3 * id_target_ + 0],
        data_->xpos[3 * id_fingertip_ + 1] - data_->xpos[3 * id_target_ + 1],
        data_->xpos[3 * id_fingertip_ + 2] - data_->xpos[3 * id_target_ + 2]};
  }

  void WriteState(float reward, mjtNum ctrl_cost, mjtNum dist_cost) {
    State state = Allocate();
    state["reward"_] = reward;
    // obs
    mjtNum* obs = static_cast<mjtNum*>(state["obs"_].Data());
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
    const auto& dist = GetDist();
    *(obs++) = dist[0];
    *(obs++) = dist[1];
    *(obs++) = dist[2];
    // info
    state["info:reward_dist"_] = -dist_cost;
    state["info:reward_ctrl"_] = -ctrl_cost;
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_, model_->nq);
    state["info:qvel0"_].Assign(qvel0_, model_->nv);
#endif
  }
};

using ReacherEnvPool = AsyncEnvPool<ReacherEnv>;

}  // namespace mujoco_gym

#endif  // ENVPOOL_MUJOCO_GYM_REACHER_H_
