/*
 * Copyright 2026 Garena Online Private Limited
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
// https://github.com/deepmind/dm_control/blob/1.0.38/dm_control/suite/dog.py

#ifndef ENVPOOL_MUJOCO_DMC_DOG_H_
#define ENVPOOL_MUJOCO_DMC_DOG_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/dmc/mujoco_env.h"
#include "envpool/mujoco/dmc/utils.h"

namespace mujoco_dmc {

bool DogHasBall(const std::string& task_name) { return task_name == "fetch"; }

std::string GetDogXML(const std::string& base_path,
                      const std::string& task_name) {
  return XMLMakeDog(GetFileContent(base_path, "dog.xml"), task_name);
}

class DogEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("frame_skip"_.Bind(3), "frame_stack"_.Bind(1),
                    "task_name"_.Bind(std::string("stand")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    [[maybe_unused]] int qpos_size =
        DogHasBall(conf["task_name"_]) ? 87 : 80;
    [[maybe_unused]] int qvel_size =
        DogHasBall(conf["task_name"_]) ? 85 : 79;
    return MakeDict(
        "obs:joint_angles"_.Bind(
            StackSpec(Spec<mjtNum>({73}), conf["frame_stack"_])),
        "obs:joint_velocites"_.Bind(
            StackSpec(Spec<mjtNum>({73}), conf["frame_stack"_])),
        "obs:torso_pelvis_height"_.Bind(
            StackSpec(Spec<mjtNum>({2}), conf["frame_stack"_])),
        "obs:z_projection"_.Bind(
            StackSpec(Spec<mjtNum>({9}), conf["frame_stack"_])),
        "obs:torso_com_velocity"_.Bind(
            StackSpec(Spec<mjtNum>({3}), conf["frame_stack"_])),
        "obs:inertial_sensors"_.Bind(
            StackSpec(Spec<mjtNum>({9}), conf["frame_stack"_])),
        "obs:foot_forces"_.Bind(
            StackSpec(Spec<mjtNum>({12}), conf["frame_stack"_])),
        "obs:touch_sensors"_.Bind(
            StackSpec(Spec<mjtNum>({4}), conf["frame_stack"_])),
        "obs:actuator_state"_.Bind(
            StackSpec(Spec<mjtNum>({38}), conf["frame_stack"_])),
        "obs:ball_state"_.Bind(
            StackSpec(Spec<mjtNum>({6}), conf["frame_stack"_])),
        "obs:target_position"_.Bind(
            StackSpec(Spec<mjtNum>({3}), conf["frame_stack"_]))
#ifdef ENVPOOL_TEST
            ,
        "info:qpos0"_.Bind(Spec<mjtNum>({qpos_size})),
        "info:qvel0"_.Bind(Spec<mjtNum>({qvel_size})),
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({qvel_size})),
        "info:act0"_.Bind(Spec<mjtNum>({38}))
#endif
    );  // NOLINT
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 38}, {-1.0, 1.0})));
  }
};

using DogEnvSpec = EnvSpec<DogEnvFns>;
using DogPixelEnvFns = PixelObservationEnvFns<DogEnvFns>;
using DogPixelEnvSpec = EnvSpec<DogPixelEnvFns>;

template <typename EnvSpecT, bool kFromPixels>
class DogEnvBase : public Env<EnvSpecT>, public MujocoEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;

  static constexpr mjtNum kMinUprightCosine = 0.8660254037844387;
  static constexpr mjtNum kWalkSpeed = 1.0;
  static constexpr mjtNum kTrotSpeed = 3.0;
  static constexpr mjtNum kRunSpeed = 9.0;

  bool is_fetch_;
  mjtNum move_speed_;
  std::array<mjtNum, 2> stand_height_{};
  mjtNum body_weight_{};

  int id_root_qpos_;
  int id_root_qvel_;
  int id_torso_body_;
  int id_torso_;
  int id_pelvis_;
  int id_skull_;
  int id_head_site_;
  int id_upper_bite_site_;
  int id_lower_bite_site_;
  int id_ball_geom_;
  int id_target_geom_;
  int id_floor_geom_;
  int id_ball_root_qpos_;
  int id_ball_root_qvel_;
  int id_torso_linvel_sensor_;
  int id_accelerometer_sensor_;
  int id_velocimeter_sensor_;
  int id_gyro_sensor_;
  std::array<int, 4> id_touch_sensors_{};
  std::array<int, 4> id_force_sensors_{};
  std::vector<int> hinge_qpos_;
  std::vector<int> hinge_qvel_;
#ifdef ENVPOOL_TEST
  std::unique_ptr<mjtNum[]> qvel0_;
  std::array<mjtNum, 38> act0_{};
#endif

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  DogEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        MujocoEnv(
            spec.config["base_path"_],
            GetDogXML(spec.config["base_path"_], spec.config["task_name"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_],
            spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        is_fetch_(spec.config["task_name"_] == "fetch"),
        id_root_qpos_(GetQposId(model_, "root")),
        id_root_qvel_(GetQvelId(model_, "root")),
        id_torso_body_(mj_name2id(model_, mjOBJ_BODY, "torso")),
        id_torso_(mj_name2id(model_, mjOBJ_XBODY, "torso")),
        id_pelvis_(mj_name2id(model_, mjOBJ_XBODY, "pelvis")),
        id_skull_(mj_name2id(model_, mjOBJ_XBODY, "skull")),
        id_head_site_(mj_name2id(model_, mjOBJ_SITE, "head")),
        id_upper_bite_site_(mj_name2id(model_, mjOBJ_SITE, "upper_bite")),
        id_lower_bite_site_(mj_name2id(model_, mjOBJ_SITE, "lower_bite")),
        id_ball_geom_(mj_name2id(model_, mjOBJ_GEOM, "ball")),
        id_target_geom_(mj_name2id(model_, mjOBJ_GEOM, "target")),
        id_floor_geom_(mj_name2id(model_, mjOBJ_GEOM, "floor")),
        id_ball_root_qpos_(is_fetch_ ? GetQposId(model_, "ball_root") : -1),
        id_ball_root_qvel_(is_fetch_ ? GetQvelId(model_, "ball_root") : -1),
        id_torso_linvel_sensor_(GetSensorId(model_, "torso_linvel")),
        id_accelerometer_sensor_(GetSensorId(model_, "accelerometer")),
        id_velocimeter_sensor_(GetSensorId(model_, "velocimeter")),
        id_gyro_sensor_(GetSensorId(model_, "gyro")) {
#ifdef ENVPOOL_TEST
    qvel0_.reset(new mjtNum[model_->nv]);
#endif
    const std::string& task_name = spec.config["task_name"_];
    if (task_name == "stand" || task_name == "fetch") {
      move_speed_ = 0.0;
    } else if (task_name == "walk") {
      move_speed_ = kWalkSpeed;
    } else if (task_name == "trot") {
      move_speed_ = kTrotSpeed;
    } else if (task_name == "run") {
      move_speed_ = kRunSpeed;
    } else {
      throw std::runtime_error("Unknown task_name " + task_name +
                               " for dmc dog.");
    }

    id_touch_sensors_ = {GetSensorId(model_, "palm_L"),
                         GetSensorId(model_, "palm_R"),
                         GetSensorId(model_, "sole_L"),
                         GetSensorId(model_, "sole_R")};
    id_force_sensors_ = {GetSensorId(model_, "foot_L"),
                         GetSensorId(model_, "foot_R"),
                         GetSensorId(model_, "hand_L"),
                         GetSensorId(model_, "hand_R")};
    for (int joint_id = 0; joint_id < model_->njnt; ++joint_id) {
      if (model_->jnt_type[joint_id] == mjJNT_HINGE) {
        hinge_qpos_.push_back(model_->jnt_qposadr[joint_id]);
        hinge_qvel_.push_back(model_->jnt_dofadr[joint_id]);
      }
    }
  }

  void TaskInitializeEpisode() override {
    stand_height_ = {data_->xpos[id_torso_ * 3 + 2] * 0.9,
                     data_->xpos[id_pelvis_ * 3 + 2] * 0.9};
    body_weight_ = -model_->opt.gravity[2] *
                   model_->body_subtreemass[id_torso_body_];

    mjtNum azimuth = RandUniform(0, 2 * M_PI)(gen_);
    data_->qpos[id_root_qpos_ + 3] = std::cos(azimuth / 2);
    data_->qpos[id_root_qpos_ + 4] = 0.0;
    data_->qpos[id_root_qpos_ + 5] = 0.0;
    data_->qpos[id_root_qpos_ + 6] = std::sin(azimuth / 2);

    data_->qvel[id_root_qvel_ + 0] = 2 * RandNormal(0, 1)(gen_);
    data_->qvel[id_root_qvel_ + 1] = 2 * RandNormal(0, 1)(gen_);
    data_->qvel[id_root_qvel_ + 5] = 2 * RandNormal(0, 1)(gen_);

    for (int actuator_id = 0; actuator_id < model_->nu; ++actuator_id) {
      mjtNum lower = model_->actuator_ctrlrange[actuator_id * 2 + 0];
      mjtNum upper = model_->actuator_ctrlrange[actuator_id * 2 + 1];
      data_->act[actuator_id] = RandUniform(lower, upper)(gen_);
    }

    if (is_fetch_) {
      mjtNum radius = 0.75 * model_->geom_size[id_floor_geom_ * 3 + 0];
      mjtNum ball_azimuth = RandUniform(0, 2 * M_PI)(gen_);
      data_->qpos[id_ball_root_qpos_ + 0] = radius * std::sin(ball_azimuth);
      data_->qpos[id_ball_root_qpos_ + 1] = radius * std::cos(ball_azimuth);
      data_->qpos[id_ball_root_qpos_ + 2] = 0.05;
      mjtNum vertical_height = RandUniform(0, 3)(gen_);
      mjtNum vertical_velocity =
          std::sqrt(2 * (-model_->opt.gravity[2]) * vertical_height);
      mjtNum horizontal_speed = RandUniform(0, 5)(gen_);
      mjtNum dir_x = -std::sin(ball_azimuth) + 0.05 * RandNormal(0, 1)(gen_);
      mjtNum dir_y = -std::cos(ball_azimuth) + 0.05 * RandNormal(0, 1)(gen_);
      data_->qvel[id_ball_root_qvel_ + 0] = horizontal_speed * dir_x;
      data_->qvel[id_ball_root_qvel_ + 1] = horizontal_speed * dir_y;
      data_->qvel[id_ball_root_qvel_ + 2] = vertical_velocity;
    }
#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_.get(), data_->qpos, sizeof(mjtNum) * model_->nq);
    std::memcpy(qvel0_.get(), data_->qvel, sizeof(mjtNum) * model_->nv);
    for (int i = 0; i < model_->na; ++i) {
      act0_[i] = data_->act[i];
    }
#endif
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    ControlReset();
    WriteState(true);
  }

  void Step(const Action& action) override {
    auto* act = static_cast<mjtNum*>(action["action"_].Data());
    ControlStep(act);
    WriteState(false);
  }

  float TaskGetReward() override {
    mjtNum reward = StandReward();
    if (is_fetch_) {
      mjtNum bite_radius = model_->site_size[id_upper_bite_site_ * 3 + 0];
      mjtNum reach_ball =
          RewardTolerance(BallToMouthDistance(), 0.0, bite_radius, 2.0, 0.1,
                          SigmoidType::kReciprocal);
      reach_ball = (6 * reach_ball + 1) / 7;

      mjtNum target_radius = model_->geom_size[id_target_geom_ * 3 + 0];
      mjtNum bring_margin = model_->geom_size[id_floor_geom_ * 3 + 0];
      mjtNum ball_near_target =
          RewardTolerance(BallToTargetDistance(), 0.0, target_radius,
                          bring_margin, 0.1, SigmoidType::kReciprocal);
      mjtNum fetch_ball = (ball_near_target + 1) / 2;
      if (BallToTargetDistance() < 2 * target_radius) {
        reach_ball = 1.0;
      }
      return static_cast<float>(reward * reach_ball * fetch_ball);
    }
    if (move_speed_ == 0) {
      return static_cast<float>(reward);
    }
    mjtNum speed_margin = std::max<mjtNum>(1.0, move_speed_);
    mjtNum forward =
        RewardTolerance(ComForwardVelocity(), move_speed_, 2 * move_speed_,
                        speed_margin, 0.0, SigmoidType::kLinear);
    forward = (4 * forward + 1) / 5;
    return static_cast<float>(reward * forward);
  }

  bool TaskShouldTerminateEpisode() override { return false; }

 private:
  std::array<mjtNum, 3> TransformToFrame(const std::array<mjtNum, 3>& vec,
                                         const mjtNum* frame) const {
    return {vec[0] * frame[0] + vec[1] * frame[3] + vec[2] * frame[6],
            vec[0] * frame[1] + vec[1] * frame[4] + vec[2] * frame[7],
            vec[0] * frame[2] + vec[1] * frame[5] + vec[2] * frame[8]};
  }

  mjtNum StandReward() {
    mjtNum torso = RewardTolerance(data_->xpos[id_torso_ * 3 + 2],
                                   stand_height_[0],
                                   std::numeric_limits<double>::infinity(),
                                   stand_height_[0]);
    mjtNum pelvis = RewardTolerance(data_->xpos[id_pelvis_ * 3 + 2],
                                    stand_height_[1],
                                    std::numeric_limits<double>::infinity(),
                                    stand_height_[1]);
    const auto& upright = Upright();
    mjtNum upright_reward = 1.0;
    for (mjtNum value : upright) {
      upright_reward *= RewardTolerance(
          value, kMinUprightCosine, std::numeric_limits<double>::infinity(),
          kMinUprightCosine + 1, 0.0, SigmoidType::kLinear);
    }
    mjtNum touch = RewardTolerance(TouchSensorSum(), body_weight_,
                                   std::numeric_limits<double>::infinity(),
                                   body_weight_, 0.9, SigmoidType::kLinear);
    return torso * pelvis * upright_reward * touch;
  }

  std::array<mjtNum, 3> Upright() {
    return {data_->xmat[id_skull_ * 9 + 8], data_->xmat[id_torso_ * 9 + 8],
            data_->xmat[id_pelvis_ * 9 + 8]};
  }

  mjtNum TouchSensorSum() {
    mjtNum sum = 0.0;
    for (int id : id_touch_sensors_) {
      sum += data_->sensordata[id];
    }
    return sum;
  }

  std::array<mjtNum, 3> CenterOfMassVelocity() {
    return {data_->sensordata[id_torso_linvel_sensor_ + 0],
            data_->sensordata[id_torso_linvel_sensor_ + 1],
            data_->sensordata[id_torso_linvel_sensor_ + 2]};
  }

  std::array<mjtNum, 3> TorsoComVelocity() {
    return TransformToFrame(CenterOfMassVelocity(),
                            data_->xmat + id_torso_ * 9);
  }

  mjtNum ComForwardVelocity() { return TorsoComVelocity()[0]; }

  std::array<mjtNum, 2> TorsoPelvisHeight() {
    return {data_->xpos[id_torso_ * 3 + 2],
            data_->xpos[id_pelvis_ * 3 + 2]};
  }

  std::array<mjtNum, 9> ZProjection() {
    return {data_->xmat[id_skull_ * 9 + 6],  data_->xmat[id_skull_ * 9 + 7],
            data_->xmat[id_skull_ * 9 + 8],  data_->xmat[id_torso_ * 9 + 6],
            data_->xmat[id_torso_ * 9 + 7],  data_->xmat[id_torso_ * 9 + 8],
            data_->xmat[id_pelvis_ * 9 + 6], data_->xmat[id_pelvis_ * 9 + 7],
            data_->xmat[id_pelvis_ * 9 + 8]};
  }

  std::array<mjtNum, 9> InertialSensors() {
    std::array<mjtNum, 9> result;
    for (int i = 0; i < 3; ++i) {
      result[i] = data_->sensordata[id_accelerometer_sensor_ + i];
      result[3 + i] = data_->sensordata[id_velocimeter_sensor_ + i];
      result[6 + i] = data_->sensordata[id_gyro_sensor_ + i];
    }
    return result;
  }

  std::array<mjtNum, 12> FootForces() {
    std::array<mjtNum, 12> result;
    for (int sensor = 0; sensor < 4; ++sensor) {
      for (int i = 0; i < 3; ++i) {
        result[sensor * 3 + i] =
            data_->sensordata[id_force_sensors_[sensor] + i];
      }
    }
    return result;
  }

  std::array<mjtNum, 4> TouchSensors() {
    return {data_->sensordata[id_touch_sensors_[0]],
            data_->sensordata[id_touch_sensors_[1]],
            data_->sensordata[id_touch_sensors_[2]],
            data_->sensordata[id_touch_sensors_[3]]};
  }

  std::array<mjtNum, 73> JointAngles() {
    std::array<mjtNum, 73> result;
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = data_->qpos[hinge_qpos_[i]];
    }
    return result;
  }

  std::array<mjtNum, 73> JointVelocities() {
    std::array<mjtNum, 73> result;
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = data_->qvel[hinge_qvel_[i]];
    }
    return result;
  }

  std::array<mjtNum, 38> ActuatorState() {
    std::array<mjtNum, 38> result;
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = data_->act[i];
    }
    return result;
  }

  std::array<mjtNum, 6> BallInHeadFrame() {
    if (!is_fetch_) {
      return {};
    }
    const mjtNum* head_frame = data_->site_xmat + id_head_site_ * 9;
    std::array<mjtNum, 3> head_to_ball = {
        data_->geom_xpos[id_ball_geom_ * 3 + 0] -
            data_->site_xpos[id_head_site_ * 3 + 0],
        data_->geom_xpos[id_ball_geom_ * 3 + 1] -
            data_->site_xpos[id_head_site_ * 3 + 1],
        data_->geom_xpos[id_ball_geom_ * 3 + 2] -
            data_->site_xpos[id_head_site_ * 3 + 2],
    };
    mjtNum head_velocity[6];
    mjtNum ball_velocity[6];
    mj_objectVelocity(model_, data_, mjOBJ_SITE, id_head_site_, head_velocity,
                      0);
    mj_objectVelocity(model_, data_, mjOBJ_GEOM, id_ball_geom_, ball_velocity,
                      0);
    std::array<mjtNum, 3> head_to_ball_velocity = {
        ball_velocity[3] - head_velocity[3],
        ball_velocity[4] - head_velocity[4],
        ball_velocity[5] - head_velocity[5],
    };
    const auto& pos = TransformToFrame(head_to_ball, head_frame);
    const auto& vel = TransformToFrame(head_to_ball_velocity, head_frame);
    return {pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]};
  }

  std::array<mjtNum, 3> TargetInHeadFrame() {
    if (!is_fetch_) {
      return {};
    }
    std::array<mjtNum, 3> head_to_target = {
        data_->geom_xpos[id_target_geom_ * 3 + 0] -
            data_->site_xpos[id_head_site_ * 3 + 0],
        data_->geom_xpos[id_target_geom_ * 3 + 1] -
            data_->site_xpos[id_head_site_ * 3 + 1],
        data_->geom_xpos[id_target_geom_ * 3 + 2] -
            data_->site_xpos[id_head_site_ * 3 + 2],
    };
    return TransformToFrame(head_to_target,
                            data_->site_xmat + id_head_site_ * 9);
  }

  mjtNum BallToMouthDistance() {
    std::array<mjtNum, 3> ball_pos = {
        data_->geom_xpos[id_ball_geom_ * 3 + 0],
        data_->geom_xpos[id_ball_geom_ * 3 + 1],
        data_->geom_xpos[id_ball_geom_ * 3 + 2],
    };
    mjtNum upper =
        Distance(ball_pos, data_->site_xpos + id_upper_bite_site_ * 3);
    mjtNum lower =
        Distance(ball_pos, data_->site_xpos + id_lower_bite_site_ * 3);
    return 0.5 * (upper + lower);
  }

  mjtNum BallToTargetDistance() {
    return Distance(data_->geom_xpos + id_ball_geom_ * 3,
                    data_->geom_xpos + id_target_geom_ * 3);
  }

  mjtNum Distance(const mjtNum* a, const mjtNum* b) const {
    mjtNum dx = a[0] - b[0];
    mjtNum dy = a[1] - b[1];
    mjtNum dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  mjtNum Distance(const std::array<mjtNum, 3>& a, const mjtNum* b) const {
    mjtNum dx = a[0] - b[0];
    mjtNum dy = a[1] - b[1];
    mjtNum dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  void WriteState(bool reset) {
    auto state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      const auto& joint_angles = JointAngles();
      const auto& joint_velocities = JointVelocities();
      const auto& torso_pelvis_height = TorsoPelvisHeight();
      const auto& z_projection = ZProjection();
      const auto& torso_com_velocity = TorsoComVelocity();
      const auto& inertial_sensors = InertialSensors();
      const auto& foot_forces = FootForces();
      const auto& touch_sensors = TouchSensors();
      const auto& actuator_state = ActuatorState();
      const auto& ball_state = BallInHeadFrame();
      const auto& target_position = TargetInHeadFrame();

      auto obs_joint_angles = state["obs:joint_angles"_];
      AssignObservation("obs:joint_angles", &obs_joint_angles,
                        joint_angles.data(), joint_angles.size(), reset);
      auto obs_joint_velocites = state["obs:joint_velocites"_];
      AssignObservation("obs:joint_velocites", &obs_joint_velocites,
                        joint_velocities.data(), joint_velocities.size(),
                        reset);
      auto obs_torso_pelvis_height = state["obs:torso_pelvis_height"_];
      AssignObservation("obs:torso_pelvis_height", &obs_torso_pelvis_height,
                        torso_pelvis_height.data(),
                        torso_pelvis_height.size(), reset);
      auto obs_z_projection = state["obs:z_projection"_];
      AssignObservation("obs:z_projection", &obs_z_projection,
                        z_projection.data(), z_projection.size(), reset);
      auto obs_torso_com_velocity = state["obs:torso_com_velocity"_];
      AssignObservation("obs:torso_com_velocity", &obs_torso_com_velocity,
                        torso_com_velocity.data(), torso_com_velocity.size(),
                        reset);
      auto obs_inertial_sensors = state["obs:inertial_sensors"_];
      AssignObservation("obs:inertial_sensors", &obs_inertial_sensors,
                        inertial_sensors.data(), inertial_sensors.size(),
                        reset);
      auto obs_foot_forces = state["obs:foot_forces"_];
      AssignObservation("obs:foot_forces", &obs_foot_forces,
                        foot_forces.data(), foot_forces.size(), reset);
      auto obs_touch_sensors = state["obs:touch_sensors"_];
      AssignObservation("obs:touch_sensors", &obs_touch_sensors,
                        touch_sensors.data(), touch_sensors.size(), reset);
      auto obs_actuator_state = state["obs:actuator_state"_];
      AssignObservation("obs:actuator_state", &obs_actuator_state,
                        actuator_state.data(), actuator_state.size(), reset);
      auto obs_ball_state = state["obs:ball_state"_];
      AssignObservation("obs:ball_state", &obs_ball_state, ball_state.data(),
                        ball_state.size(), reset);
      auto obs_target_position = state["obs:target_position"_];
      AssignObservation("obs:target_position", &obs_target_position,
                        target_position.data(), target_position.size(), reset);
    }
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
    state["info:qvel0"_].Assign(qvel0_.get(), model_->nv);
    state["info:qacc_warmstart0"_].Assign(data_->qacc_warmstart, model_->nv);
    state["info:act0"_].Assign(act0_.data(), model_->na);
#endif
  }
};

using DogEnv = DogEnvBase<DogEnvSpec, false>;
using DogPixelEnv = DogEnvBase<DogPixelEnvSpec, true>;
using DogEnvPool = AsyncEnvPool<DogEnv>;
using DogPixelEnvPool = AsyncEnvPool<DogPixelEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_DOG_H_
