// Copyright 2026 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ENVPOOL_MUJOCO_PLAYGROUND_OP3_H_
#define ENVPOOL_MUJOCO_PLAYGROUND_OP3_H_

#include <mujoco.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/frame_stack.h"
#include "envpool/mujoco/playground/mujoco_env.h"

namespace mujoco_playground {

using envpool::mujoco::PixelObservationEnvFns;
using envpool::mujoco::RenderCameraIdOrDefault;
using envpool::mujoco::RenderHeightOrDefault;
using envpool::mujoco::RenderWidthOrDefault;
using envpool::mujoco::StackSpec;

constexpr int kOp3ActionDim = 20;
constexpr int kOp3ObsDim = 49;
constexpr int kOp3HistoryLen = 3;
constexpr int kOp3StateDim = kOp3ObsDim * kOp3HistoryLen;
constexpr int kOp3Feet = 2;
constexpr int kOp3FootSensors = 2;
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
constexpr const char* kOp3FeetSites[kOp3Feet] = {"left_foot", "right_foot"};
inline const std::array<const char*, kOp3FootSensors>& Op3LeftFootSensors() {
  static constexpr std::array<const char*, kOp3FootSensors> k_names = {
      "l_foot1_floor_found", "l_foot2_floor_found"};
  return k_names;
}

inline const std::array<const char*, kOp3FootSensors>& Op3RightFootSensors() {
  static constexpr std::array<const char*, kOp3FootSensors> k_names = {
      "r_foot1_floor_found", "r_foot2_floor_found"};
  return k_names;
}

class PlaygroundOp3EnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "frame_stack"_.Bind(1), "task_name"_.Bind(std::string("Op3Joystick")),
        "ctrl_dt"_.Bind(0.02), "sim_dt"_.Bind(0.004), "kp"_.Bind(21.1),
        "kd"_.Bind(1.084), "early_termination"_.Bind(true),
        "action_scale"_.Bind(0.3), "obs_noise"_.Bind(0.05),
        "max_foot_height"_.Bind(0.07), "lin_vel_x_min"_.Bind(-0.6),
        "lin_vel_x_max"_.Bind(1.5), "lin_vel_y_min"_.Bind(-0.8),
        "lin_vel_y_max"_.Bind(0.8), "ang_vel_yaw_min"_.Bind(-0.7),
        "ang_vel_yaw_max"_.Bind(0.7), "tracking_lin_vel_scale"_.Bind(1.5),
        "tracking_ang_vel_scale"_.Bind(0.8), "lin_vel_z_scale"_.Bind(-2.0),
        "ang_vel_xy_scale"_.Bind(-0.05), "orientation_scale"_.Bind(-5.0),
        "torques_scale"_.Bind(-0.0002), "action_rate_scale"_.Bind(-0.01),
        "zero_cmd_scale"_.Bind(-0.5), "termination_scale"_.Bind(-1.0),
        "feet_slip_scale"_.Bind(-0.1), "feet_clearance_scale"_.Bind(0.0),
        "energy_scale"_.Bind(-0.0001), "tracking_sigma"_.Bind(0.25));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs:state"_.Bind(StackSpec(Spec<mjtNum>({kOp3StateDim}, {-inf, inf}),
                                    conf["frame_stack"_])),
        "info:terminated"_.Bind(Spec<bool>({-1})),
        "info:command"_.Bind(Spec<mjtNum>({-1, 3})),
        "info:reward_tracking_lin_vel"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_tracking_ang_vel"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_lin_vel_z"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_ang_vel_xy"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_orientation"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_torques"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_action_rate"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_zero_cmd"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_termination"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_feet_slip"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_feet_clearance"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_energy"_.Bind(Spec<mjtNum>({-1}))
#ifdef ENVPOOL_TEST
            ,
        "info:qpos0"_.Bind(Spec<mjtNum>({64})),
        "info:qvel0"_.Bind(Spec<mjtNum>({64})),
        "info:ctrl0"_.Bind(Spec<mjtNum>({64})),
        "info:qpos"_.Bind(Spec<mjtNum>({64})),
        "info:qvel"_.Bind(Spec<mjtNum>({64})),
        "info:ctrl"_.Bind(Spec<mjtNum>({64})),
        "info:qacc"_.Bind(Spec<mjtNum>({64})),
        "info:qacc_warmstart"_.Bind(Spec<mjtNum>({64})),
        "info:xfrc_applied"_.Bind(Spec<mjtNum>({256})),
        "info:last_act"_.Bind(Spec<mjtNum>({kOp3ActionDim})),
        "info:last_last_act"_.Bind(Spec<mjtNum>({kOp3ActionDim})),
        "info:last_vel"_.Bind(Spec<mjtNum>({kOp3ActionDim})),
        "info:step"_.Bind(Spec<int>({-1})),
        "info:obs_history"_.Bind(Spec<mjtNum>({kOp3StateDim})),
        "info:sensordata"_.Bind(Spec<mjtNum>({256}))
#endif
    );  // NOLINT
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<mjtNum>({-1, kOp3ActionDim}, {-1.0, 1.0})));
  }
};

using Op3Aliases = PlaygroundEnvAliases<PlaygroundOp3EnvFns>;
using PlaygroundOp3EnvSpec = Op3Aliases::Spec;
using PlaygroundOp3PixelEnvFns = Op3Aliases::PixelFns;
using PlaygroundOp3PixelEnvSpec = Op3Aliases::PixelSpec;

template <typename EnvSpecT, bool kFromPixels>
class PlaygroundOp3EnvBase : public Env<EnvSpecT>, public PlaygroundMujocoEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  int n_substeps_{5};
  std::vector<mjtNum> init_qpos_;
  std::vector<mjtNum> reset_qpos_;
  std::vector<mjtNum> reset_qvel_;
  std::vector<mjtNum> reset_ctrl_;
  std::array<mjtNum, kOp3ActionDim> default_pose_{};
  std::array<mjtNum, kOp3ActionDim> lowers_{};
  std::array<mjtNum, kOp3ActionDim> uppers_{};
  std::array<mjtNum, kOp3ActionDim> last_act_{};
  std::array<mjtNum, kOp3ActionDim> last_last_act_{};
  std::array<mjtNum, kOp3ActionDim> last_vel_{};
  std::array<mjtNum, 3> command_{};
  std::array<mjtNum, kOp3StateDim> obs_history_{};
  std::array<int, kOp3Feet> feet_site_ids_{};
  std::array<int, kOp3Feet> foot_linvel_sensor_adrs_{};
  std::array<int, kOp3FootSensors> left_floor_sensor_adrs_{};
  std::array<int, kOp3FootSensors> right_floor_sensor_adrs_{};
  int torso_body_id_{-1};
  int gyro_adr_{-1};
  int upvector_adr_{-1};
  int global_linvel_adr_{-1};
  int global_angvel_adr_{-1};
  int local_linvel_adr_{-1};
  int step_{0};
  mjtNum reward_tracking_lin_vel_{0.0};
  mjtNum reward_tracking_ang_vel_{0.0};
  mjtNum reward_lin_vel_z_{0.0};
  mjtNum reward_ang_vel_xy_{0.0};
  mjtNum reward_orientation_{0.0};
  mjtNum reward_torques_{0.0};
  mjtNum reward_action_rate_{0.0};
  mjtNum reward_zero_cmd_{0.0};
  mjtNum reward_termination_{0.0};
  mjtNum reward_feet_slip_{0.0};
  mjtNum reward_feet_clearance_{0.0};
  mjtNum reward_energy_{0.0};
  std::uniform_real_distribution<mjtNum> unit_uniform_{0.0, 1.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  PlaygroundOp3EnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        PlaygroundMujocoEnv(Op3XmlPath(spec.config["base_path"_]),
                            spec.config["max_episode_steps"_],
                            spec.config["frame_stack"_],
                            RenderWidthOrDefault<kFromPixels>(spec.config),
                            RenderHeightOrDefault<kFromPixels>(spec.config),
                            RenderCameraIdOrDefault<kFromPixels>(spec.config)) {
    const std::string task_name = spec.config["task_name"_];
    if (task_name != "Op3Joystick") {
      throw std::runtime_error("Unsupported playground OP3 task_name " +
                               task_name);
    }
    if (model_->nq < 7 + kOp3ActionDim || model_->nv < 6 + kOp3ActionDim ||
        model_->nu != kOp3ActionDim) {
      throw std::runtime_error("Unexpected OP3 model dimensions.");
    }
    n_substeps_ = static_cast<int>(
        std::llround(spec.config["ctrl_dt"_] / spec.config["sim_dt"_]));
    model_->opt.timestep = spec.config["sim_dt"_];
    model_->opt.ccd_iterations = 10;
    for (int i = 6; i < model_->nv; ++i) {
      model_->dof_damping[i] = spec.config["kd"_];
    }
    for (int i = 0; i < model_->nu; ++i) {
      model_->actuator_gainprm[i * mjNGAIN] = spec.config["kp"_];
      model_->actuator_biasprm[i * mjNBIAS + 1] = -spec.config["kp"_];
    }
    const int home_id = mj_name2id(model_, mjOBJ_KEY, "stand_bent_knees");
    if (home_id < 0) {
      throw std::runtime_error("OP3 model is missing stand_bent_knees key.");
    }
    init_qpos_.assign(model_->key_qpos + home_id * model_->nq,
                      model_->key_qpos + (home_id + 1) * model_->nq);
    std::copy(init_qpos_.begin() + 7, init_qpos_.begin() + 7 + kOp3ActionDim,
              default_pose_.begin());
    for (int i = 0; i < kOp3ActionDim; ++i) {
      lowers_[i] = model_->actuator_ctrlrange[i * 2];
      uppers_[i] = model_->actuator_ctrlrange[i * 2 + 1];
    }
    for (int i = 0; i < kOp3Feet; ++i) {
      feet_site_ids_[i] = RequireId(mjOBJ_SITE, kOp3FeetSites[i]);
      foot_linvel_sensor_adrs_[i] =
          SensorAdr(std::string(kOp3FeetSites[i]) + "_global_linvel");
    }
    for (int i = 0; i < kOp3FootSensors; ++i) {
      left_floor_sensor_adrs_[i] = SensorAdr(Op3LeftFootSensors()[i]);
      right_floor_sensor_adrs_[i] = SensorAdr(Op3RightFootSensors()[i]);
    }
    torso_body_id_ = RequireId(mjOBJ_BODY, "body_link");
    gyro_adr_ = SensorAdr("gyro");
    upvector_adr_ = SensorAdr("upvector");
    global_linvel_adr_ = SensorAdr("global_linvel");
    global_angvel_adr_ = SensorAdr("global_angvel");
    local_linvel_adr_ = SensorAdr("local_linvel");
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    has_cached_render_ = false;
    done_ = false;
    terminated_ = false;
    elapsed_step_ = 0;
    step_ = 0;
    ResetRewards();
    std::fill(last_act_.begin(), last_act_.end(), 0.0);
    std::fill(last_last_act_.begin(), last_last_act_.end(), 0.0);
    std::fill(last_vel_.begin(), last_vel_.end(), 0.0);
    std::fill(obs_history_.begin(), obs_history_.end(), 0.0);

    mj_resetData(model_, data_);
    std::copy(init_qpos_.begin(), init_qpos_.end(), data_->qpos);
    std::fill(data_->qvel, data_->qvel + model_->nv, 0.0);
    std::fill(data_->ctrl, data_->ctrl + model_->nu, 0.0);
    mj_forward(model_, data_);
    reset_qpos_.assign(data_->qpos, data_->qpos + model_->nq);
    reset_qvel_.assign(data_->qvel, data_->qvel + model_->nv);
    reset_ctrl_.assign(data_->ctrl, data_->ctrl + model_->nu);

    SampleCommand();
    UpdateObs(/*add_noise=*/true);
    WriteState(0.0, true);
  }

  void Step(const Action& action) override {
    has_cached_render_ = false;
    const auto* act = static_cast<const mjtNum*>(action["action"_].Data());

    for (int i = 0; i < kOp3ActionDim; ++i) {
      const mjtNum target =
          default_pose_[i] + act[i] * spec_.config["action_scale"_];
      data_->ctrl[i] = std::clamp(target, lowers_[i], uppers_[i]);
    }
    for (int i = 0; i < n_substeps_; ++i) {
      mj_step(model_, data_);
    }

    UpdateObs(/*add_noise=*/true);
    terminated_ = GetTermination();
    ComputeRewards(act);
    mjtNum reward =
        (reward_tracking_lin_vel_ + reward_tracking_ang_vel_ +
         reward_lin_vel_z_ + reward_ang_vel_xy_ + reward_orientation_ +
         reward_torques_ + reward_action_rate_ + reward_zero_cmd_ +
         reward_termination_ + reward_feet_slip_ + reward_feet_clearance_ +
         reward_energy_) *
        Dt();
    reward = std::clamp(reward, static_cast<mjtNum>(0.0),
                        static_cast<mjtNum>(10000.0));

    std::copy(last_act_.begin(), last_act_.end(), last_last_act_.begin());
    std::copy(act, act + kOp3ActionDim, last_act_.begin());
    std::copy(data_->qvel + 6, data_->qvel + 6 + kOp3ActionDim,
              last_vel_.begin());
    ++step_;
    if (step_ > 500) {
      SampleCommand();
    }
    if (terminated_ || step_ > 500) {
      step_ = 0;
    }
    ++elapsed_step_;
    done_ = terminated_ || elapsed_step_ >= max_episode_steps_;
    WriteState(static_cast<float>(reward), false);
  }

 private:
  static std::string Op3XmlPath(const std::string& base_path) {
    return base_path +
           "/mujoco/playground/assets/mujoco_playground/_src/locomotion/op3/"
           "xmls/scene_mjx_feetonly.xml";
  }

  mjtNum Dt() const { return spec_.config["ctrl_dt"_]; }

  int RequireId(mjtObj obj_type, const std::string& name) const {
    const int id = mj_name2id(model_, obj_type, name.c_str());
    if (id < 0) {
      throw std::runtime_error("Missing MuJoCo object: " + name);
    }
    return id;
  }

  int SensorAdr(const std::string& name) const {
    const int id = RequireId(mjOBJ_SENSOR, name);
    return model_->sensor_adr[id];
  }

  mjtNum Uniform(mjtNum low, mjtNum high) {
    return low + (high - low) * unit_uniform_(gen_);
  }

  void ResetRewards() {
    reward_tracking_lin_vel_ = 0.0;
    reward_tracking_ang_vel_ = 0.0;
    reward_lin_vel_z_ = 0.0;
    reward_ang_vel_xy_ = 0.0;
    reward_orientation_ = 0.0;
    reward_torques_ = 0.0;
    reward_action_rate_ = 0.0;
    reward_zero_cmd_ = 0.0;
    reward_termination_ = 0.0;
    reward_feet_slip_ = 0.0;
    reward_feet_clearance_ = 0.0;
    reward_energy_ = 0.0;
  }

  void SampleCommand() {
    command_[0] =
        Uniform(spec_.config["lin_vel_x_min"_], spec_.config["lin_vel_x_max"_]);
    command_[1] =
        Uniform(spec_.config["lin_vel_y_min"_], spec_.config["lin_vel_y_max"_]);
    command_[2] = Uniform(spec_.config["ang_vel_yaw_min"_],
                          spec_.config["ang_vel_yaw_max"_]);
    if (unit_uniform_(gen_) < 0.1) {
      std::fill(command_.begin(), command_.end(), 0.0);
    }
  }

  bool GetTermination() const {
    bool joint_limit = false;
    for (int i = 0; i < kOp3ActionDim; ++i) {
      const mjtNum q = data_->qpos[7 + i];
      joint_limit = joint_limit || q < lowers_[i] || q > uppers_[i];
    }
    const bool fall = data_->sensordata[upvector_adr_ + 2] < 0.85 ||
                      data_->xpos[torso_body_id_ * 3 + 2] < 0.21;
    if (spec_.config["early_termination"_]) {
      return joint_limit || fall;
    }
    return joint_limit;
  }

  bool LeftFootContact() const {
    return std::any_of(
        left_floor_sensor_adrs_.begin(), left_floor_sensor_adrs_.end(),
        [this](int sensor_adr) { return data_->sensordata[sensor_adr] > 0.0; });
  }

  bool RightFootContact() const {
    return std::any_of(
        right_floor_sensor_adrs_.begin(), right_floor_sensor_adrs_.end(),
        [this](int sensor_adr) { return data_->sensordata[sensor_adr] > 0.0; });
  }

  void UpdateObs(bool add_noise) {
    std::array<mjtNum, kOp3ObsDim> obs{};
    int index = 0;
    for (int i = 0; i < 3; ++i) {
      obs[index++] = data_->sensordata[gyro_adr_ + i];
    }
    for (int i = 0; i < 3; ++i) {
      obs[index++] = data_->sensordata[upvector_adr_ + i];
    }
    for (int i = 0; i < 3; ++i) {
      obs[index++] = command_[i];
    }
    for (int i = 0; i < kOp3ActionDim; ++i) {
      obs[index++] = data_->qpos[7 + i] - default_pose_[i];
    }
    for (int i = 0; i < kOp3ActionDim; ++i) {
      obs[index++] = last_act_[i];
    }
    if (add_noise && spec_.config["obs_noise"_] >= 0.0) {
      for (int i = 0; i < kOp3ObsDim; ++i) {
        obs[i] = std::clamp(obs[i], static_cast<mjtNum>(-100.0),
                            static_cast<mjtNum>(100.0));
        obs[i] += spec_.config["obs_noise"_] * Uniform(-1.0, 1.0);
      }
    }
    for (int i = kOp3StateDim - 1; i >= kOp3ObsDim; --i) {
      obs_history_[i] = obs_history_[i - kOp3ObsDim];
    }
    std::copy(obs.begin(), obs.end(), obs_history_.begin());
  }

  void ComputeRewards(const mjtNum* act) {
    const mjtNum lin_x_error =
        (command_[0] - data_->sensordata[local_linvel_adr_ + 0]) *
        (command_[0] - data_->sensordata[local_linvel_adr_ + 0]);
    const mjtNum lin_y_error =
        (command_[1] - data_->sensordata[local_linvel_adr_ + 1]) *
        (command_[1] - data_->sensordata[local_linvel_adr_ + 1]);
    reward_tracking_lin_vel_ = std::exp(-(lin_x_error + lin_y_error) /
                                        spec_.config["tracking_sigma"_]) *
                               spec_.config["tracking_lin_vel_scale"_];

    const mjtNum yaw_error = command_[2] - data_->sensordata[gyro_adr_ + 2];
    reward_tracking_ang_vel_ =
        std::exp(-(yaw_error * yaw_error) / spec_.config["tracking_sigma"_]) *
        spec_.config["tracking_ang_vel_scale"_];

    const mjtNum z_vel = data_->sensordata[global_linvel_adr_ + 2];
    reward_lin_vel_z_ = z_vel * z_vel * spec_.config["lin_vel_z_scale"_];
    const mjtNum ang_x = data_->sensordata[global_angvel_adr_ + 0];
    const mjtNum ang_y = data_->sensordata[global_angvel_adr_ + 1];
    reward_ang_vel_xy_ =
        (ang_x * ang_x + ang_y * ang_y) * spec_.config["ang_vel_xy_scale"_];
    const mjtNum up_x = data_->sensordata[upvector_adr_ + 0];
    const mjtNum up_y = data_->sensordata[upvector_adr_ + 1];
    reward_orientation_ =
        (up_x * up_x + up_y * up_y) * spec_.config["orientation_scale"_];

    mjtNum torque_sq = 0.0;
    mjtNum torque_abs = 0.0;
    for (int i = 0; i < model_->nu; ++i) {
      torque_sq += data_->actuator_force[i] * data_->actuator_force[i];
      torque_abs += std::abs(data_->actuator_force[i]);
    }
    reward_torques_ =
        (std::sqrt(torque_sq) + torque_abs) * spec_.config["torques_scale"_];

    mjtNum action_rate = 0.0;
    mjtNum zero_cmd = 0.0;
    for (int i = 0; i < kOp3ActionDim; ++i) {
      const mjtNum delta = act[i] - last_act_[i];
      const mjtNum accel = act[i] - 2.0 * last_act_[i] + last_last_act_[i];
      action_rate += delta * delta + accel * accel;
      zero_cmd += act[i] * act[i];
    }
    reward_action_rate_ = action_rate * spec_.config["action_rate_scale"_];
    const mjtNum cmd_norm =
        std::sqrt(command_[0] * command_[0] + command_[1] * command_[1] +
                  command_[2] * command_[2]);
    reward_zero_cmd_ =
        (cmd_norm < 0.1 ? zero_cmd : 0.0) * spec_.config["zero_cmd_scale"_];
    reward_termination_ = (terminated_ && step_ < 500 ? 1.0 : 0.0) *
                          spec_.config["termination_scale"_];

    const std::array<bool, kOp3Feet> foot_contacts = {LeftFootContact(),
                                                      RightFootContact()};
    mjtNum feet_slip = 0.0;
    mjtNum feet_clearance = 0.0;
    for (int foot = 0; foot < kOp3Feet; ++foot) {
      const int adr = foot_linvel_sensor_adrs_[foot];
      const mjtNum vx = data_->sensordata[adr + 0];
      const mjtNum vy = data_->sensordata[adr + 1];
      const mjtNum vel_xy_sq = vx * vx + vy * vy;
      if (foot_contacts[foot]) {
        feet_slip += vel_xy_sq;
      }
      const mjtNum vel_norm = std::sqrt(std::sqrt(vel_xy_sq));
      const mjtNum foot_z = data_->site_xpos[feet_site_ids_[foot] * 3 + 2];
      const mjtNum delta = foot_z - spec_.config["max_foot_height"_];
      feet_clearance += delta * delta * vel_norm;
    }
    reward_feet_slip_ = feet_slip * spec_.config["feet_slip_scale"_];
    reward_feet_clearance_ =
        feet_clearance * spec_.config["feet_clearance_scale"_];

    mjtNum energy = 0.0;
    for (int i = 0; i < kOp3ActionDim; ++i) {
      energy +=
          std::abs(data_->qvel[6 + i]) * std::abs(data_->actuator_force[i]);
    }
    reward_energy_ = energy * spec_.config["energy_scale"_];
  }

  void WriteState(float reward, bool reset) {
    auto state = Allocate();
    state["reward"_] = reward;
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      auto obs_state = state["obs:state"_];
      mjtNum* obs = PrepareObservation("obs:state", &obs_state);
      std::copy(obs_history_.begin(), obs_history_.end(), obs);
      CommitObservation("obs:state", &obs_state, reset);
    }
    state["info:terminated"_] = terminated_;
    state["info:command"_].Assign(command_.data(), command_.size());
    state["info:reward_tracking_lin_vel"_] = reward_tracking_lin_vel_;
    state["info:reward_tracking_ang_vel"_] = reward_tracking_ang_vel_;
    state["info:reward_lin_vel_z"_] = reward_lin_vel_z_;
    state["info:reward_ang_vel_xy"_] = reward_ang_vel_xy_;
    state["info:reward_orientation"_] = reward_orientation_;
    state["info:reward_torques"_] = reward_torques_;
    state["info:reward_action_rate"_] = reward_action_rate_;
    state["info:reward_zero_cmd"_] = reward_zero_cmd_;
    state["info:reward_termination"_] = reward_termination_;
    state["info:reward_feet_slip"_] = reward_feet_slip_;
    state["info:reward_feet_clearance"_] = reward_feet_clearance_;
    state["info:reward_energy"_] = reward_energy_;
#ifdef ENVPOOL_TEST
    std::array<mjtNum, 64> pad{};
    std::copy(data_->qpos, data_->qpos + model_->nq, pad.begin());
    state["info:qpos"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(reset_qpos_.begin(), reset_qpos_.end(), pad.begin());
    state["info:qpos0"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(data_->qvel, data_->qvel + model_->nv, pad.begin());
    state["info:qvel"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(reset_qvel_.begin(), reset_qvel_.end(), pad.begin());
    state["info:qvel0"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(data_->ctrl, data_->ctrl + model_->nu, pad.begin());
    state["info:ctrl"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(reset_ctrl_.begin(), reset_ctrl_.end(), pad.begin());
    state["info:ctrl0"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(data_->qacc, data_->qacc + model_->nv, pad.begin());
    state["info:qacc"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(data_->qacc_warmstart, data_->qacc_warmstart + model_->nv,
              pad.begin());
    state["info:qacc_warmstart"_].Assign(pad.data(), pad.size());
    std::array<mjtNum, 256> force_pad{};
    std::copy(data_->xfrc_applied, data_->xfrc_applied + model_->nbody * 6,
              force_pad.begin());
    state["info:xfrc_applied"_].Assign(force_pad.data(), force_pad.size());
    state["info:last_act"_].Assign(last_act_.data(), last_act_.size());
    state["info:last_last_act"_].Assign(last_last_act_.data(),
                                        last_last_act_.size());
    state["info:last_vel"_].Assign(last_vel_.data(), last_vel_.size());
    state["info:step"_] = step_;
    state["info:obs_history"_].Assign(obs_history_.data(), obs_history_.size());
    std::array<mjtNum, 256> sensor_pad{};
    std::copy(data_->sensordata, data_->sensordata + model_->nsensordata,
              sensor_pad.begin());
    state["info:sensordata"_].Assign(sensor_pad.data(), sensor_pad.size());
#endif
  }
};

template <typename Spec, bool kFromPixels>
using Op3Base = PlaygroundOp3EnvBase<Spec, kFromPixels>;
using Op3Env = Op3Base<PlaygroundOp3EnvSpec, false>;
using Op3PixelEnv = Op3Base<PlaygroundOp3PixelEnvSpec, true>;
using PlaygroundOp3Env = Op3Env;
using PlaygroundOp3PixelEnv = Op3PixelEnv;
using PlaygroundOp3EnvPool = PlaygroundEnvPoolT<PlaygroundOp3Env>;
using PlaygroundOp3PixelEnvPool = PlaygroundEnvPoolT<PlaygroundOp3PixelEnv>;

}  // namespace mujoco_playground

#endif  // ENVPOOL_MUJOCO_PLAYGROUND_OP3_H_
