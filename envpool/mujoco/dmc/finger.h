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
  // others
  int id_site_target_, id_site_tip_, id_hinge_, id_cap1_;
  // sensor
  int id_proximal_, id_distal_, id_proximal_velocity_;
  int id_distal_velocity_, id_hinge_velocity_;
  int id_sensor_tip_, id_sensor_target_;
  int id_spinner_, id_touchtop_, id_touchbottom_;
  std::uniform_real_distribution<> dist_uniform_;
  mjtNum target_radius_;
  bool is_spin_;
#ifdef ENVPOOL_TEST
  std::array<mjtNum, 2> target_;
#endif

 public:
  FingerEnv(const Spec& spec, int env_id)
      : Env<FingerEnvSpec>(spec, env_id),
        MujocoEnv(
            spec.config["base_path"_],
            GetFingerXML(spec.config["base_path"_], spec.config["task_name"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_]),
        id_site_target_(mj_name2id(model_, mjOBJ_SITE, "target")),
        id_site_tip_(mj_name2id(model_, mjOBJ_SITE, "tip")),
        id_hinge_(mj_name2id(model_, mjOBJ_JOINT, "hinge")),
        id_cap1_(mj_name2id(model_, mjOBJ_GEOM, "cap1")),
        id_proximal_(GetSensorId(model_, "proximal")),
        id_distal_(GetSensorId(model_, "distal")),
        id_proximal_velocity_(GetSensorId(model_, "proximal_velocity")),
        id_distal_velocity_(GetSensorId(model_, "distal_velocity")),
        id_hinge_velocity_(GetSensorId(model_, "hinge_velocity")),
        id_sensor_tip_(GetSensorId(model_, "tip")),
        id_sensor_target_(GetSensorId(model_, "target")),
        id_spinner_(GetSensorId(model_, "spinner")),
        id_touchtop_(GetSensorId(model_, "touchtop")),
        id_touchbottom_(GetSensorId(model_, "touchbottom")),
        dist_uniform_(-M_PI, M_PI),
        is_spin_(spec.config["task_name"_] == "spin") {
    const std::string& task_name = spec.config["task_name"_];
    if (task_name == "turn_easy") {
      target_radius_ = kEasyTargetSize;
    } else if (task_name == "turn_hard") {
      target_radius_ = kHardTargetSize;
    } else if (task_name != "spin") {
      throw std::runtime_error("Unknown task_name for dmc finger.");
    }
  }

  void TaskInitializeEpisode() override {
    if (is_spin_) {
      // physics.named.model.site_rgba['target', 3] = 0
      // physics.named.model.site_rgba['tip', 3] = 0
      // physics.named.model.dof_damping['hinge'] = .03
      model_->site_rgba[id_site_target_ * 3 + 3] = 0;
      model_->site_rgba[id_site_tip_ * 3 + 3] = 0;
      model_->dof_damping[id_hinge_] = 0.03;
    } else {
      // target_angle = self.random.uniform(-np.pi, np.pi)
      // hinge_x, hinge_z = physics.named.data.xanchor['hinge', ['x', 'z']]
      // radius = physics.named.model.geom_size['cap1'].sum()
      // target_x = hinge_x + radius * np.sin(target_angle)
      // target_z = hinge_z + radius * np.cos(target_angle)
      // physics.named.model.site_pos['target', ['x', 'z']] = target_x, target_z
      // physics.named.model.site_size['target', 0] = self._target_radius
      mjtNum target_angle = dist_uniform_(gen_);
      mjtNum hinge_x = data_->xanchor[id_hinge_ * 3 + 0];
      mjtNum hinge_z = data_->xanchor[id_hinge_ * 3 + 2];
      mjtNum radius = model_->geom_size[id_cap1_ * 3 + 0] +
                      model_->geom_size[id_cap1_ * 3 + 1] +
                      model_->geom_size[id_cap1_ * 3 + 2];
      mjtNum target_x = hinge_x + radius * std::sin(target_angle);
      mjtNum target_z = hinge_z + radius * std::cos(target_angle);
      model_->site_pos[id_site_target_] = target_x;
      model_->site_pos[id_site_target_ + 2] = target_z;
      model_->site_size[id_site_target_] = target_radius_;
#ifdef ENVPOOL_TEST
      target_[0] = target_x;
      target_[1] = target_z;
#endif
    }
    SetRandomJointAngles();
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
    if (is_spin_) {
      return static_cast<float>(HingeVelocity() <= -kSpinVelocity);
    }
    return static_cast<float>(DistToTarget() <= 0);
  }

  bool TaskShouldTerminateEpisode() override { return false; }

 private:
  void WriteState() {
    State state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    // obs
    const auto& bound_pos = BoundedPosition();
    const auto& velocity = Velocity();
    const auto& touch = Touch();
    state["obs:position"_].Assign(bound_pos.begin(), bound_pos.size());
    state["obs:velocity"_].Assign(velocity.begin(), velocity.size());
    state["obs:touch"_].Assign(touch.begin(), touch.size());
    if (!is_spin_) {
      const auto& target_position = TargetPosition();
      state["obs:target_position"_].Assign(target_position.begin(),
                                           target_position.size());
      state["obs:dist_to_target"_] = DistToTarget();
    }
    // info
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
    if (!is_spin_) {
      state["info:target"_].Assign(target_.begin(), target_.size());
    }
#endif
  }

  void SetRandomJointAngles(int max_attempts = 1000) {
    int i = 0;
    for (int i = 0; i < max_attempts; i++) {
      RandomizeLimitedAndRotationalJoints(&gen_);
#ifdef ENVPOOL_TEST
      std::memcpy(qpos0_.get(), data_->qpos, sizeof(mjtNum) * model_->nq);
#endif
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

  mjtNum HingeVelocity() {
    // return self.named.data.sensordata['hinge_velocity']
    return data_->sensordata[id_hinge_velocity_];
  }
  std::array<mjtNum, 4> BoundedPosition() {
    // return np.hstack((self.named.data.sensordata[['proximal', 'distal']],
    //                   self.tip_position()))
    const auto& tip_position = TipPosition();
    return {data_->sensordata[id_proximal_], data_->sensordata[id_distal_],
            tip_position[0], tip_position[1]};
  }

  std::array<mjtNum, 2> TipPosition() {
    // return (self.named.data.sensordata['tip'][[0, 2]] -
    //         self.named.data.sensordata['spinner'][[0, 2]])
    return {data_->sensordata[id_sensor_tip_] - data_->sensordata[id_spinner_],
            data_->sensordata[id_sensor_tip_ + 2] -
                data_->sensordata[id_spinner_ + 2]};
  }

  std::array<mjtNum, 2> TargetPosition() {
    // return (self.named.data.sensordata['target'][[0, 2]] -
    //         self.named.data.sensordata['spinner'][[0, 2]])
    return {
        data_->sensordata[id_sensor_target_] - data_->sensordata[id_spinner_],
        data_->sensordata[id_sensor_target_ + 2] -
            data_->sensordata[id_spinner_ + 2]};
  }

  std::array<mjtNum, 2> Touch() {
    // return np.log1p(self.named.data.sensordata[['touchtop', 'touchbottom']])
    return {std::log1p(data_->sensordata[id_touchtop_]),
            std::log1p(data_->sensordata[id_touchbottom_])};
  }

  std::array<mjtNum, 3> Velocity() {
    // return self.named.data.sensordata[['proximal_velocity',
    // 'distal_velocity', 'hinge_velocity']]
    return {data_->sensordata[id_proximal_velocity_],
            data_->sensordata[id_distal_velocity_],
            data_->sensordata[id_hinge_velocity_]};
  }

  std::array<mjtNum, 2> ToTarget() {
    // return self.target_position() - self.tip_position()
    const auto& target_position = TargetPosition();
    const auto& tip_position = TipPosition();
    return {target_position[0] - tip_position[0],
            target_position[1] - tip_position[1]};
  }

  mjtNum DistToTarget() {
    // return (np.linalg.norm(self.to_target()) -
    //         self.named.model.site_size['target', 0])
    const auto& to_target = ToTarget();
    return std::sqrt(to_target[0] * to_target[0] +
                     to_target[1] * to_target[1]) -
           model_->site_size[0];
  }
};

using FingerEnvPool = AsyncEnvPool<FingerEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_FINGER_H_
