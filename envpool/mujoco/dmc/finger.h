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
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/finger.py

#ifndef ENVPOOL_MUJOCO_DMC_FINGER_H_
#define ENVPOOL_MUJOCO_DMC_FINGER_H_

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

std::string GetFingerXML(const std::string& base_path,
                         const std::string& task_name_) {
  return GetFileContent(base_path, "finger.xml");
}

class FingerEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.Bind(1000), "frame_skip"_.Bind(2),
                    "task_name"_.Bind(std::string("spin")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs:position"_.Bind(Spec<mjtNum>({4})),
                    "obs:velocity"_.Bind(Spec<mjtNum>({3})),
                    "obs:touch"_.Bind(Spec<mjtNum>({2})),
                    "obs:target_position"_.Bind(Spec<mjtNum>({2})),
                    "obs:dist_to_target"_.Bind(Spec<mjtNum>({})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.Bind(Spec<mjtNum>({3})),
                    "info:target"_.Bind(Spec<mjtNum>({2})),
                    "info:site_size"_.Bind(Spec<mjtNum>({1})),
                    "info:rgba"_.Bind(Spec<mjtNum>({2})),
                    "info:dof_damping"_.Bind(Spec<mjtNum>({1})),
#endif
                    "discount"_.Bind(Spec<float>({-1}, {0.0, 1.0})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 2}, {-1.0, 1.0})));
  }
};

using FingerEnvSpec = EnvSpec<FingerEnvFns>;

class FingerEnv : public Env<FingerEnvSpec>, public MujocoEnv {
 protected:
  const mjtNum kEasyTargetSize = 0.07;
  const mjtNum kHardTargetSize = 0.03;
  const mjtNum kSpinVelocity = 15;
  std::uniform_real_distribution<> dist_uniform_;
  mjtNum target_radius_;
  std::string task_name_;
#ifdef ENVPOOL_TEST
  std::array<mjtNum, 2> target_;
  std::array<mjtNum, 2> rgba_;
  mjtNum dof_damping_;
  mjtNum site_size_;
#endif

 public:
  FingerEnv(const Spec& spec, int env_id)
      : Env<FingerEnvSpec>(spec, env_id),
        MujocoEnv(
            spec.config["base_path"_],
            GetFingerXML(spec.config["base_path"_], spec.config["task_name"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_]),
        dist_uniform_(0, 1) {
    task_name_ = spec.config["task_name"_];
    // if (task_name_ == "spin") {}
    if (task_name_ == "turn_easy") {
      target_radius_ = kEasyTargetSize;
    } else if (task_name_ == "turn_hard") {
      target_radius_ = kHardTargetSize;
    } else {
      throw std::runtime_error("Unknown task_name for dmc finger.");
    }
  }

  void TaskInitializeEpisode() override {
    if (task_name_ == "spin") {
      model_->site_rgba[3 * 0 + 3] = 0;
      model_->site_rgba[3 * 3 + 3] = 0;
      model_->dof_damping[2] = 0.03;
      SetRandomJointAngles();
#ifdef ENVPOOL_TEST
      rgba_[0] = 0;
      rgba_[1] = 0;
      dof_damping_ = 0.03;
#endif
    } else if (task_name_ == "turn_easy" || task_name_ == "turn_hard") {
      mjtNum target_angle = dist_uniform_(gen_) * 2 * M_PI - M_PI;
      mjtNum hinge_x = data_->xanchor[2 * 3 + 0];
      mjtNum hinge_z = data_->xanchor[2 * 3 + 2];
      mjtNum radius = model_->geom_size[5 * 3 + 0] + model_->geom_size[5 * 1] +
                      model_->geom_size[5 * 2];
      mjtNum target_x = hinge_x + radius * std::sin(target_angle);
      mjtNum target_z = hinge_z + radius * std::cos(target_angle);
      model_->site_pos[0] = target_x;
      model_->site_pos[2] = target_z;
      model_->site_size[0] = target_radius_;
#ifdef ENVPOOL_TEST
      target_[0] = target_x;
      target_[1] = target_z;
      site_size_ = target_radius_;
#endif
      SetRandomJointAngles();
    } else {
      throw std::runtime_error("Unknown task_name for dmc finger.");
    }
#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_.get(), data_->qpos, sizeof(mjtNum) * model_->nq);
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
    float reward = 0;
    if (task_name_ == "spin") {
      reward = static_cast<float>(HingeVelocity() <= -kSpinVelocity);
    } else if (task_name_ == "turn_easy" || task_name_ == "turn_hard") {
      reward = static_cast<float>(DistToTarget() <= 0);
    }
    return reward;
  }

  bool TaskShouldTerminateEpisode() override { return false; }

 private:
  void WriteState() {
    if (task_name_ != "spin" || task_name_ != "turn_easy" ||
        task_name_ != "turn_hard") {
      throw std::runtime_error("Unknown task_name_ for dmc finger.");
      return;
    }
    State state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    // obs
    std::array<mjtNum, 4> bound_pos = BoundedPosition();
    state["obs:position"_].Assign(bound_pos.begin(), bound_pos.size());
    std::array<mjtNum, 3> velocity = Velocity();
    state["obs:velocity"_].Assign(velocity.begin(), velocity.size());
    std::array<mjtNum, 2> touch = Touch();
    state["obs:touch"_].Assign(touch.begin(), touch.size());
    if (task_name_ == "turn_easy" || task_name_ == "turn_hard") {
      std::array<mjtNum, 2> target_position = TargetPosition();
      state["obs:target_position"_].Assign(target_position.begin(),
                                           target_position.size());
      state["obs:dist_to_target"_] = DistToTarget();
    }

#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
    if (task_name_ == "spin") {
      state["info:rgba"_].Assign(rgba_.begin(), rgba_.size());
      state["info:dof_damping"_] = dof_damping_;
    } else if (task_name_ == "turn_easy" || task_name_ == "turn_hard") {
      state["info:target"_].Assign(target_.begin(), target_.size());
      state["info:site_size"_] = site_size_;
    }
#endif
  }

  void SetRandomJointAngles(int max_attempts = 1000) {
    int i = 0;
    for (int i = 0; i < max_attempts; i++) {
      RandomizeLimitedAndRotationalJoints(&gen_);
      PhysicsAfterReset();
      if (data_->ncon == 0) {
        break;
      }
    }
    if (i == max_attempts) {
      throw std::runtime_error(
          "Could not find a collision-free state after max_attempts attempts");
    }
  }

  mjtNum Speed() {
    // return self.named.data.sensordata['torso_subtreelinvel'][0]
    return data_->sensordata[0];
  }

  mjtNum HingeVelocity() {
    // return self.named.data.sensordata['hinge_velocity']
    return data_->sensordata[4];
  }
  std::array<mjtNum, 4> BoundedPosition() {
    std::array<mjtNum, 4> bound_pos;
    std::array<mjtNum, 2> tip_position = TipPosition();
    bound_pos[0] = data_->sensordata[0];
    bound_pos[1] = data_->sensordata[1];
    bound_pos[2] = tip_position[0];
    bound_pos[3] = tip_position[1];
    return bound_pos;
  }

  std::array<mjtNum, 2> TipPosition() {
    std::array<mjtNum, 2> tip_position;
    tip_position[0] = data_->sensordata[5] - data_->sensordata[11];
    tip_position[1] = data_->sensordata[7] - data_->sensordata[13];
    return tip_position;
  }

  std::array<mjtNum, 2> TargetPosition() {
    std::array<mjtNum, 2> target_position;
    target_position[0] = data_->sensordata[8] - data_->sensordata[11];
    target_position[1] = data_->sensordata[10] - data_->sensordata[13];
    return target_position;
  }

  std::array<mjtNum, 2> Touch() {
    return std::array<mjtNum, 2>{std::log1p(data_->sensordata[14]),
                                 std::log1p(data_->sensordata[15])};
  }

  std::array<mjtNum, 3> Velocity() {
    return std::array<mjtNum, 3>{data_->sensordata[2], data_->sensordata[3],
                                 data_->sensordata[4]};
  }

  std::array<mjtNum, 2> ToTarget() {
    std::array<mjtNum, 2> target_position = TargetPosition();
    std::array<mjtNum, 2> tip_position = TipPosition();
    return std::array<mjtNum, 2>{target_position[0] - tip_position[0],
                                 target_position[1] - tip_position[1]};
  }

  mjtNum DistToTarget() {
    std::array<mjtNum, 2> to_target = ToTarget();
    return std::sqrt(to_target[0] * to_target[0] +
                     to_target[1] * to_target[1]) -
           model_->site_size[0];
  }
};

using FingerEnvPool = AsyncEnvPool<FingerEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_FINGER_H_
