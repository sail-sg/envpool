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
// https://github.com/deepmind/dm_control/blob/1.0.38/dm_control/suite/quadruped.py

#ifndef ENVPOOL_MUJOCO_DMC_QUADRUPED_H_
#define ENVPOOL_MUJOCO_DMC_QUADRUPED_H_

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

bool QuadrupedHasBall(const std::string& task_name) {
  return task_name == "fetch";
}

bool QuadrupedHasRangefinder(const std::string& task_name) {
  return task_name == "escape";
}

std::string GetQuadrupedXML(const std::string& base_path,
                            const std::string& task_name) {
  return XMLMakeQuadruped(GetFileContent(base_path, "quadruped.xml"),
                          task_name);
}

class QuadrupedEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("frame_skip"_.Bind(4), "frame_stack"_.Bind(1),
                    "task_name"_.Bind(std::string("walk")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    [[maybe_unused]] int qpos_size =
        QuadrupedHasBall(conf["task_name"_]) ? 30 : 23;
    [[maybe_unused]] int qvel_size =
        QuadrupedHasBall(conf["task_name"_]) ? 28 : 22;
    [[maybe_unused]] int hfield_size =
        QuadrupedHasRangefinder(conf["task_name"_]) ? 201 * 201 : 1;
    return MakeDict(
        "obs:egocentric_state"_.Bind(
            StackSpec(Spec<mjtNum>({44}), conf["frame_stack"_])),
        "obs:torso_velocity"_.Bind(
            StackSpec(Spec<mjtNum>({3}), conf["frame_stack"_])),
        "obs:torso_upright"_.Bind(
            StackSpec(Spec<mjtNum>({}), conf["frame_stack"_])),
        "obs:imu"_.Bind(StackSpec(Spec<mjtNum>({6}), conf["frame_stack"_])),
        "obs:force_torque"_.Bind(
            StackSpec(Spec<mjtNum>({24}), conf["frame_stack"_])),
        "obs:origin"_.Bind(StackSpec(Spec<mjtNum>({3}), conf["frame_stack"_])),
        "obs:rangefinder"_.Bind(
            StackSpec(Spec<mjtNum>({20}), conf["frame_stack"_])),
        "obs:ball_state"_.Bind(
            StackSpec(Spec<mjtNum>({9}), conf["frame_stack"_])),
        "obs:target_position"_.Bind(
            StackSpec(Spec<mjtNum>({3}), conf["frame_stack"_]))
#ifdef ENVPOOL_TEST
            ,
        "info:qpos0"_.Bind(Spec<mjtNum>({qpos_size})),
        "info:qvel0"_.Bind(Spec<mjtNum>({qvel_size})),
        "info:qacc_warmstart0"_.Bind(Spec<mjtNum>({qvel_size})),
        "info:act0"_.Bind(Spec<mjtNum>({12})),
        "info:hfield0"_.Bind(Spec<mjtNum>({hfield_size}))
#endif
    );  // NOLINT
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>(
        {-1, 12},
        std::make_tuple(std::vector<mjtNum>{-1.0, -1.0, -0.8, -1.0, -1.0, -0.8,
                                            -1.0, -1.0, -0.8, -1.0, -1.0, -0.8},
                        std::vector<mjtNum>{1.0, 1.1, 0.8, 1.0, 1.1, 0.8, 1.0,
                                            1.1, 0.8, 1.0, 1.1, 0.8}))));
  }
};

using QuadrupedEnvSpec = EnvSpec<QuadrupedEnvFns>;
using QuadrupedPixelEnvFns = PixelObservationEnvFns<QuadrupedEnvFns>;
using QuadrupedPixelEnvSpec = EnvSpec<QuadrupedPixelEnvFns>;

template <typename EnvSpecT, bool kFromPixels>
class QuadrupedEnvBase : public Env<EnvSpecT>, public MujocoEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;

  enum class TaskKind : std::uint8_t { kMove, kEscape, kFetch };

  static constexpr mjtNum kWalkSpeed = 0.5;
  static constexpr mjtNum kRunSpeed = 5.0;

  TaskKind task_kind_;
  mjtNum desired_speed_{};
  bool has_rangefinder_;
  bool has_ball_;

  int id_root_qpos_;
  int id_root_qvel_;
  int id_ball_root_qpos_;
  int id_ball_root_qvel_;
  int id_torso_;
  int id_workspace_site_;
  int id_target_site_;
  int id_floor_geom_;
  int id_ball_geom_;
  int id_ball_body_;
  int id_velocimeter_sensor_;
  int id_imu_accel_sensor_;
  int id_imu_gyro_sensor_;
  std::array<int, 8> id_force_torque_sensors_{};
  std::array<int, 20> id_rangefinder_sensors_{};
  std::vector<int> hinge_qpos_;
  std::vector<int> hinge_qvel_;
#ifdef ENVPOOL_TEST
  std::unique_ptr<mjtNum[]> qvel0_;
  std::array<mjtNum, 12> act0_{};
  std::vector<mjtNum> hfield0_;
#endif

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  QuadrupedEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        MujocoEnv(spec.config["base_path"_],
                  GetQuadrupedXML(spec.config["base_path"_],
                                  spec.config["task_name"_]),
                  spec.config["frame_skip"_], spec.config["max_episode_steps"_],
                  spec.config["frame_stack"_],
                  RenderWidthOrDefault<kFromPixels>(spec.config),
                  RenderHeightOrDefault<kFromPixels>(spec.config),
                  RenderCameraIdOrDefault<kFromPixels>(spec.config)),
        has_rangefinder_(QuadrupedHasRangefinder(spec.config["task_name"_])),
        has_ball_(QuadrupedHasBall(spec.config["task_name"_])),
        id_root_qpos_(GetQposId(model_, "root")),
        id_root_qvel_(GetQvelId(model_, "root")),
        id_ball_root_qpos_(has_ball_ ? GetQposId(model_, "ball_root") : -1),
        id_ball_root_qvel_(has_ball_ ? GetQvelId(model_, "ball_root") : -1),
        id_torso_(mj_name2id(model_, mjOBJ_XBODY, "torso")),
        id_workspace_site_(mj_name2id(model_, mjOBJ_SITE, "workspace")),
        id_target_site_(mj_name2id(model_, mjOBJ_SITE, "target")),
        id_floor_geom_(mj_name2id(model_, mjOBJ_GEOM, "floor")),
        id_ball_geom_(mj_name2id(model_, mjOBJ_GEOM, "ball")),
        id_ball_body_(mj_name2id(model_, mjOBJ_XBODY, "ball")),
        id_velocimeter_sensor_(GetSensorId(model_, "velocimeter")),
        id_imu_accel_sensor_(GetSensorId(model_, "imu_accel")),
        id_imu_gyro_sensor_(GetSensorId(model_, "imu_gyro")) {
    const std::string& task_name = spec.config["task_name"_];
    if (task_name == "walk") {
      task_kind_ = TaskKind::kMove;
      desired_speed_ = kWalkSpeed;
    } else if (task_name == "run") {
      task_kind_ = TaskKind::kMove;
      desired_speed_ = kRunSpeed;
    } else if (task_name == "escape") {
      task_kind_ = TaskKind::kEscape;
    } else if (task_name == "fetch") {
      task_kind_ = TaskKind::kFetch;
    } else {
      throw std::runtime_error("Unknown task_name " + task_name +
                               " for dmc quadruped.");
    }

#ifdef ENVPOOL_TEST
    qvel0_.reset(new mjtNum[model_->nv]);
    int hfield_size = has_rangefinder_ && model_->nhfield > 0
                          ? model_->hfield_nrow[0] * model_->hfield_ncol[0]
                          : 1;
    hfield0_.assign(hfield_size, 0.0);
#endif

    id_force_torque_sensors_ = {
        GetSensorId(model_, "force_toe_front_left"),
        GetSensorId(model_, "force_toe_front_right"),
        GetSensorId(model_, "force_toe_back_right"),
        GetSensorId(model_, "force_toe_back_left"),
        GetSensorId(model_, "torque_toe_front_left"),
        GetSensorId(model_, "torque_toe_front_right"),
        GetSensorId(model_, "torque_toe_back_right"),
        GetSensorId(model_, "torque_toe_back_left"),
    };
    if (has_rangefinder_) {
      for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 5; ++col) {
          int idx = row * 5 + col;
          id_rangefinder_sensors_[idx] = GetSensorId(
              model_, "rf_" + std::to_string(row) + std::to_string(col));
        }
      }
    } else {
      id_rangefinder_sensors_.fill(-1);
    }
    for (int joint_id = 0; joint_id < model_->njnt; ++joint_id) {
      if (model_->jnt_type[joint_id] == mjJNT_HINGE) {
        hinge_qpos_.push_back(model_->jnt_qposadr[joint_id]);
        hinge_qvel_.push_back(model_->jnt_dofadr[joint_id]);
      }
    }
  }

  void TaskInitializeEpisode() override {
    if (task_kind_ == TaskKind::kEscape) {
      GenerateTerrain();
      std::array<mjtNum, 4> orientation = RandomUnitQuaternion();
      FindNonContactingHeight(orientation, 0.0, 0.0);
    } else if (task_kind_ == TaskKind::kFetch) {
      mjtNum azimuth = RandUniform(0, 2 * M_PI)(gen_);
      std::array<mjtNum, 4> orientation = {std::cos(azimuth / 2), 0.0, 0.0,
                                           std::sin(azimuth / 2)};
      mjtNum spawn_radius = 0.9 * model_->geom_size[id_floor_geom_ * 3 + 0];
      mjtNum x_pos = RandUniform(-spawn_radius, spawn_radius)(gen_);
      mjtNum y_pos = RandUniform(-spawn_radius, spawn_radius)(gen_);
      FindNonContactingHeight(orientation, x_pos, y_pos);
      data_->qpos[id_ball_root_qpos_ + 0] =
          RandUniform(-spawn_radius, spawn_radius)(gen_);
      data_->qpos[id_ball_root_qpos_ + 1] =
          RandUniform(-spawn_radius, spawn_radius)(gen_);
      data_->qpos[id_ball_root_qpos_ + 2] = 2.0;
      data_->qvel[id_ball_root_qvel_ + 0] = 5 * RandNormal(0, 1)(gen_);
      data_->qvel[id_ball_root_qvel_ + 1] = 5 * RandNormal(0, 1)(gen_);
    } else {
      std::array<mjtNum, 4> orientation = RandomUnitQuaternion();
      FindNonContactingHeight(orientation, 0.0, 0.0);
    }
#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_.get(), data_->qpos, sizeof(mjtNum) * model_->nq);
    std::memcpy(qvel0_.get(), data_->qvel, sizeof(mjtNum) * model_->nv);
    for (int i = 0; i < model_->na; ++i) {
      act0_[i] = data_->act[i];
    }
    if (has_rangefinder_ && model_->nhfield > 0) {
      int hfield_size = model_->hfield_nrow[0] * model_->hfield_ncol[0];
      int start = model_->hfield_adr[0];
      for (int i = 0; i < hfield_size; ++i) {
        hfield0_[i] = model_->hfield_data[start + i];
      }
    } else {
      hfield0_[0] = 0.0;
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
    if (task_kind_ == TaskKind::kMove) {
      mjtNum move_reward =
          RewardTolerance(TorsoVelocity()[0], desired_speed_,
                          std::numeric_limits<double>::infinity(),
                          desired_speed_, 0.5, SigmoidType::kLinear);
      return static_cast<float>(UprightReward(0.0) * move_reward);
    }
    if (task_kind_ == TaskKind::kEscape) {
      mjtNum terrain_size = model_->hfield_size[0];
      mjtNum escape_reward =
          RewardTolerance(OriginDistance(), terrain_size,
                          std::numeric_limits<double>::infinity(), terrain_size,
                          0.0, SigmoidType::kLinear);
      return static_cast<float>(UprightReward(20.0) * escape_reward);
    }
    mjtNum arena_radius =
        model_->geom_size[id_floor_geom_ * 3 + 0] * std::sqrt(2.0);
    mjtNum workspace_radius = model_->site_size[id_workspace_site_ * 3 + 0];
    mjtNum ball_radius = model_->geom_size[id_ball_geom_ * 3 + 0];
    mjtNum reach_reward = RewardTolerance(
        SelfToBallDistance(), 0.0, workspace_radius + ball_radius, arena_radius,
        0.0, SigmoidType::kLinear);
    mjtNum target_radius = model_->site_size[id_target_site_ * 3 + 0];
    mjtNum fetch_reward =
        RewardTolerance(BallToTargetDistance(), 0.0, target_radius,
                        arena_radius, 0.0, SigmoidType::kLinear);
    mjtNum reach_then_fetch = reach_reward * (0.5 + 0.5 * fetch_reward);
    return static_cast<float>(UprightReward(0.0) * reach_then_fetch);
  }

  bool TaskShouldTerminateEpisode() override { return false; }

 private:
  std::array<mjtNum, 4> RandomUnitQuaternion() {
    std::array<mjtNum, 4> q = {RandNormal(0, 1)(gen_), RandNormal(0, 1)(gen_),
                               RandNormal(0, 1)(gen_), RandNormal(0, 1)(gen_)};
    mjtNum norm =
        std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    return {q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm};
  }

  void FindNonContactingHeight(const std::array<mjtNum, 4>& orientation,
                               mjtNum x_pos, mjtNum y_pos) {
    mjtNum z_pos = 0.0;
    for (int attempts = 0; attempts < 10000; ++attempts) {
      data_->qpos[id_root_qpos_ + 0] = x_pos;
      data_->qpos[id_root_qpos_ + 1] = y_pos;
      data_->qpos[id_root_qpos_ + 2] = z_pos;
      for (int i = 0; i < 4; ++i) {
        data_->qpos[id_root_qpos_ + 3 + i] = orientation[i];
      }
      PhysicsAfterReset();
      if (data_->ncon <= 0) {
        return;
      }
      z_pos += 0.01;
    }
    throw std::runtime_error(
        "Failed to find a non-contacting quadruped configuration.");
  }

  void GenerateTerrain() {
    if (model_->nhfield == 0) {
      return;
    }
    int res = model_->hfield_nrow[0];
    int start = model_->hfield_adr[0];
    for (int row = 0; row < res; ++row) {
      mjtNum y = -1.0 + 2.0 * row / (res - 1);
      for (int col = 0; col < res; ++col) {
        mjtNum x = -1.0 + 2.0 * col / (res - 1);
        mjtNum radius = std::clamp(std::sqrt(x * x + y * y), 0.04, 1.0);
        mjtNum bowl = 0.5 - std::cos(2 * M_PI * radius) / 2.0;
        mjtNum bump = RandUniform(0.15, 1.0)(gen_);
        model_->hfield_data[start + row * res + col] = bowl * bump;
      }
    }
  }

  std::array<mjtNum, 3> TransformToFrame(const std::array<mjtNum, 3>& vec,
                                         const mjtNum* frame) const {
    return {vec[0] * frame[0] + vec[1] * frame[3] + vec[2] * frame[6],
            vec[0] * frame[1] + vec[1] * frame[4] + vec[2] * frame[7],
            vec[0] * frame[2] + vec[1] * frame[5] + vec[2] * frame[8]};
  }

  mjtNum UprightReward(mjtNum deviation_angle) {
    mjtNum deviation = std::cos(deviation_angle * M_PI / 180.0);
    return RewardTolerance(TorsoUpright(), deviation,
                           std::numeric_limits<double>::infinity(),
                           1 + deviation, 0.0, SigmoidType::kLinear);
  }

  mjtNum TorsoUpright() { return data_->xmat[id_torso_ * 9 + 8]; }

  std::array<mjtNum, 3> TorsoVelocity() {
    return {data_->sensordata[id_velocimeter_sensor_ + 0],
            data_->sensordata[id_velocimeter_sensor_ + 1],
            data_->sensordata[id_velocimeter_sensor_ + 2]};
  }

  std::array<mjtNum, 44> EgocentricState() {
    std::array<mjtNum, 44> result;
    for (int i = 0; i < 16; ++i) {
      result[i] = data_->qpos[hinge_qpos_[i]];
      result[16 + i] = data_->qvel[hinge_qvel_[i]];
    }
    for (int i = 0; i < 12; ++i) {
      result[32 + i] = data_->act[i];
    }
    return result;
  }

  std::array<mjtNum, 6> Imu() {
    return {data_->sensordata[id_imu_accel_sensor_ + 0],
            data_->sensordata[id_imu_accel_sensor_ + 1],
            data_->sensordata[id_imu_accel_sensor_ + 2],
            data_->sensordata[id_imu_gyro_sensor_ + 0],
            data_->sensordata[id_imu_gyro_sensor_ + 1],
            data_->sensordata[id_imu_gyro_sensor_ + 2]};
  }

  std::array<mjtNum, 24> ForceTorque() {
    std::array<mjtNum, 24> result;
    for (int sensor = 0; sensor < 8; ++sensor) {
      for (int i = 0; i < 3; ++i) {
        result[sensor * 3 + i] =
            std::asinh(data_->sensordata[id_force_torque_sensors_[sensor] + i]);
      }
    }
    return result;
  }

  std::array<mjtNum, 20> Rangefinder() {
    std::array<mjtNum, 20> result{};
    if (!has_rangefinder_) {
      return result;
    }
    for (int i = 0; i < 20; ++i) {
      mjtNum reading = data_->sensordata[id_rangefinder_sensors_[i]];
      result[i] = reading == -1.0 ? 1.0 : std::tanh(reading);
    }
    return result;
  }

  std::array<mjtNum, 3> Origin() {
    std::array<mjtNum, 3> minus_torso_pos = {-data_->xpos[id_torso_ * 3 + 0],
                                             -data_->xpos[id_torso_ * 3 + 1],
                                             -data_->xpos[id_torso_ * 3 + 2]};
    return TransformToFrame(minus_torso_pos, data_->xmat + id_torso_ * 9);
  }

  mjtNum OriginDistance() {
    mjtNum x = data_->site_xpos[id_workspace_site_ * 3 + 0];
    mjtNum y = data_->site_xpos[id_workspace_site_ * 3 + 1];
    mjtNum z = data_->site_xpos[id_workspace_site_ * 3 + 2];
    return std::sqrt(x * x + y * y + z * z);
  }

  std::array<mjtNum, 9> BallState() {
    std::array<mjtNum, 9> result{};
    if (!has_ball_) {
      return result;
    }
    const mjtNum* torso_frame = data_->xmat + id_torso_ * 9;
    std::array<mjtNum, 3> ball_rel_pos = {
        data_->xpos[id_ball_body_ * 3 + 0] - data_->xpos[id_torso_ * 3 + 0],
        data_->xpos[id_ball_body_ * 3 + 1] - data_->xpos[id_torso_ * 3 + 1],
        data_->xpos[id_ball_body_ * 3 + 2] - data_->xpos[id_torso_ * 3 + 2],
    };
    std::array<mjtNum, 3> ball_rel_vel = {
        data_->qvel[id_ball_root_qvel_ + 0] - data_->qvel[id_root_qvel_ + 0],
        data_->qvel[id_ball_root_qvel_ + 1] - data_->qvel[id_root_qvel_ + 1],
        data_->qvel[id_ball_root_qvel_ + 2] - data_->qvel[id_root_qvel_ + 2],
    };
    std::array<mjtNum, 3> ball_rot_vel = {
        data_->qvel[id_ball_root_qvel_ + 3],
        data_->qvel[id_ball_root_qvel_ + 4],
        data_->qvel[id_ball_root_qvel_ + 5],
    };
    const auto& pos = TransformToFrame(ball_rel_pos, torso_frame);
    const auto& vel = TransformToFrame(ball_rel_vel, torso_frame);
    const auto& rot = TransformToFrame(ball_rot_vel, torso_frame);
    return {pos[0], pos[1], pos[2], vel[0], vel[1],
            vel[2], rot[0], rot[1], rot[2]};
  }

  std::array<mjtNum, 3> TargetPosition() {
    std::array<mjtNum, 3> result{};
    if (!has_ball_) {
      return result;
    }
    std::array<mjtNum, 3> torso_to_target = {
        data_->site_xpos[id_target_site_ * 3 + 0] -
            data_->xpos[id_torso_ * 3 + 0],
        data_->site_xpos[id_target_site_ * 3 + 1] -
            data_->xpos[id_torso_ * 3 + 1],
        data_->site_xpos[id_target_site_ * 3 + 2] -
            data_->xpos[id_torso_ * 3 + 2],
    };
    return TransformToFrame(torso_to_target, data_->xmat + id_torso_ * 9);
  }

  mjtNum BallToTargetDistance() {
    mjtNum dx = data_->site_xpos[id_target_site_ * 3 + 0] -
                data_->xpos[id_ball_body_ * 3 + 0];
    mjtNum dy = data_->site_xpos[id_target_site_ * 3 + 1] -
                data_->xpos[id_ball_body_ * 3 + 1];
    return std::sqrt(dx * dx + dy * dy);
  }

  mjtNum SelfToBallDistance() {
    mjtNum dx = data_->site_xpos[id_workspace_site_ * 3 + 0] -
                data_->xpos[id_ball_body_ * 3 + 0];
    mjtNum dy = data_->site_xpos[id_workspace_site_ * 3 + 1] -
                data_->xpos[id_ball_body_ * 3 + 1];
    return std::sqrt(dx * dx + dy * dy);
  }

  void WriteState(bool reset) {
    auto state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      const auto& egocentric_state = EgocentricState();
      const auto& torso_velocity = TorsoVelocity();
      const auto& imu = Imu();
      const auto& force_torque = ForceTorque();
      const auto& origin = Origin();
      const auto& rangefinder = Rangefinder();
      const auto& ball_state = BallState();
      const auto& target_position = TargetPosition();

      auto obs_egocentric_state = state["obs:egocentric_state"_];
      AssignObservation("obs:egocentric_state", &obs_egocentric_state,
                        egocentric_state.data(), egocentric_state.size(),
                        reset);
      auto obs_torso_velocity = state["obs:torso_velocity"_];
      AssignObservation("obs:torso_velocity", &obs_torso_velocity,
                        torso_velocity.data(), torso_velocity.size(), reset);
      auto obs_torso_upright = state["obs:torso_upright"_];
      AssignObservation("obs:torso_upright", &obs_torso_upright, TorsoUpright(),
                        reset);
      auto obs_imu = state["obs:imu"_];
      AssignObservation("obs:imu", &obs_imu, imu.data(), imu.size(), reset);
      auto obs_force_torque = state["obs:force_torque"_];
      AssignObservation("obs:force_torque", &obs_force_torque,
                        force_torque.data(), force_torque.size(), reset);
      auto obs_origin = state["obs:origin"_];
      AssignObservation("obs:origin", &obs_origin, origin.data(), origin.size(),
                        reset);
      auto obs_rangefinder = state["obs:rangefinder"_];
      AssignObservation("obs:rangefinder", &obs_rangefinder, rangefinder.data(),
                        rangefinder.size(), reset);
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
    state["info:hfield0"_].Assign(hfield0_.data(), hfield0_.size());
#endif
  }
};

using QuadrupedEnv = QuadrupedEnvBase<QuadrupedEnvSpec, false>;
using QuadrupedPixelEnv = QuadrupedEnvBase<QuadrupedPixelEnvSpec, true>;
using QuadrupedEnvPool = AsyncEnvPool<QuadrupedEnv>;
using QuadrupedPixelEnvPool = AsyncEnvPool<QuadrupedPixelEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_QUADRUPED_H_
