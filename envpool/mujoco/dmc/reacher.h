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

#ifndef ENVPOOL_MUJOCO_DMC_REACHER_H_
#define ENVPOOL_MUJOCO_DMC_REACHER_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/dmc/mujoco_env.h"
#include "envpool/mujoco/dmc/utils.h"

namespace mujoco_dmc {

std::string GetReacherXML(const std::string& base_path,
                          const std::string& task_name) {
  return GetFileContent(base_path, "reacher.xml");
}

class ReacherEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.Bind(1000), "frame_skip"_.Bind(1),
                    "task_name"_.Bind(std::string("easy")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs:position"_.Bind(Spec<mjtNum>({2})),
                    "obs:to_target"_.Bind(Spec<mjtNum>({2})),
                    "obs:velocity"_.Bind(Spec<mjtNum>({2})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.Bind(Spec<mjtNum>({2})),
                    "info:geom_pos"_.Bind(Spec<mjtNum>({30})),
#endif
                    "discount"_.Bind(Spec<float>({-1}, {0.0, 1.0})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 2}, {-1.0, 1.0})));
  }
};

using ReacherEnvSpec = EnvSpec<ReacherEnvFns>;

class ReacherEnv : public Env<ReacherEnvSpec>, public MujocoEnv {
 protected:
  const mjtNum kBigTarget = 0.05;
  const mjtNum kSmallTarget = 0.015;
  float target_size_;
  std::uniform_real_distribution<> dist_uniform_;
#ifdef ENVPOOL_TEST
  //   std::unique_ptr<std::unique_ptr<mjtNum>[]> geom_pos_;
  std::unique_ptr<mjtNum> geom_pos_;
#endif

 public:
  ReacherEnv(const Spec& spec, int env_id)
      : Env<ReacherEnvSpec>(spec, env_id),
        MujocoEnv(
            spec.config["base_path"_],
            GetReacherXML(spec.config["base_path"_], spec.config["task_name"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_]),
        dist_uniform_(0, 1) {
    std::string task_name = spec.config["task_name"_];
    if (task_name == "easy") {
      target_size_ = kBigTarget;
    } else if (task_name == "hard") {
      target_size_ = kSmallTarget;
    } else {
      throw std::runtime_error("Unknown task_name for dmc reacher.");
    }
#ifdef ENVPOOL_TEST
    geom_pos_.reset(new mjtNum[model_->ngeom * 3]);
    // geom_pos_ = new std::unique_ptr<mjtNum>[model_->ngeom];
    // for(int i=0;i<model_->ngeom;i++){
    //     geom_pos_[i] = std::unique_ptr<mjtNum[]>(new mjtNum[3]);
    // }
#endif
  }

  void TaskInitializeEpisode() override {
    model_->geom_size[6 * 3] = target_size_;
    RandomizeLimitedAndRotationalJoints(&gen_);
    mjtNum angle = dist_uniform_(gen_) * 2 * M_PI;
    mjtNum radius = dist_uniform_(gen_) * 0.15 + 0.05;
    model_->geom_pos[6 * 3 + 0] = radius * sin(angle);
    model_->geom_pos[6 * 3 + 1] = radius * cos(angle);
#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_.get(), data_->qpos, sizeof(mjtNum) * model_->nq);
    std::memcpy(geom_pos_.get(), model_->geom_pos,
                sizeof(mjtNum) * model_->ngeom * 3);
    // for(int i=0;i<model_->ngeom;i++){
    //     std::memcpy(geom_pos_[i].get(), model_->geom_pos[i*3],
    //             sizeof(mjtNum) * 3);
    // }
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

  float TaskGetReward() override {
    mjtNum radii = model_->geom_size[6 * 3] + model_->geom_size[9 * 3];
    return static_cast<float>(RewardTolerance(FingerToTargetDist(), 0, radii));
  }

 private:
  void WriteState() {
    State state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    // obs
    state["obs:position"_].Assign(data_->qpos, model_->nq);
    auto finger = FingerToTarget();
    state["obs:to_target"_].Assign(finger.begin(), 2);
    state["obs:velocity"_].Assign(data_->qvel, model_->nv);
    // info for check alignment
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
    state["info:geom_pos"_].Assign(geom_pos_.get(), model_->ngeom * 3);
#endif
  }

  std::array<mjtNum, 2> FingerToTarget() {
    std::array<mjtNum, 2> finger;
    finger[0] = data_->geom_xpos[6 * 3 + 0] - data_->geom_xpos[9 * 3 + 0];
    finger[1] = data_->geom_xpos[6 * 3 + 1] - data_->geom_xpos[9 * 3 + 1];
    return finger;
  }
  mjtNum FingerToTargetDist() {
    std::array<mjtNum, 2> finger = FingerToTarget();
    return std::sqrt(finger[0] * finger[0] + finger[1] * finger[1]);
  }
};

using ReacherEnvPool = AsyncEnvPool<ReacherEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_REACHER_H_