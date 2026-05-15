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

#ifndef ENVPOOL_MUJOCO_PLAYGROUND_BARKOUR_H_
#define ENVPOOL_MUJOCO_PLAYGROUND_BARKOUR_H_

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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mujoco_playground {

constexpr int kBarkourActionDim = 12;
constexpr int kBarkourObsDim = 31;
constexpr int kBarkourHistoryLen = 15;
constexpr int kBarkourStateDim = kBarkourObsDim * kBarkourHistoryLen;
constexpr int kBarkourFeet = 4;
inline const std::array<const char*, kBarkourFeet>& BarkourFeetGeoms() {
  static constexpr std::array<const char*, kBarkourFeet> k_names = {
      "foot_front_left", "foot_hind_left", "foot_front_right",
      "foot_hind_right"};
  return k_names;
}

class PlaygroundBarkourEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "frame_stack"_.Bind(1),
        "task_name"_.Bind(std::string("BarkourJoystick")),
        "ctrl_dt"_.Bind(0.02), "sim_dt"_.Bind(0.004), "action_scale"_.Bind(0.3),
        "obs_noise"_.Bind(0.05), "lin_vel_x_min"_.Bind(-0.6),
        "lin_vel_x_max"_.Bind(1.5), "lin_vel_y_min"_.Bind(-0.8),
        "lin_vel_y_max"_.Bind(0.8), "ang_vel_yaw_min"_.Bind(-0.7),
        "ang_vel_yaw_max"_.Bind(0.7), "tracking_lin_vel_scale"_.Bind(1.5),
        "tracking_ang_vel_scale"_.Bind(0.8), "lin_vel_z_scale"_.Bind(-2.0),
        "ang_vel_xy_scale"_.Bind(-0.05), "orientation_scale"_.Bind(-5.0),
        "torques_scale"_.Bind(-0.0002), "action_rate_scale"_.Bind(-0.1),
        "stand_still_scale"_.Bind(-0.5), "termination_scale"_.Bind(-1.0),
        "feet_air_time_scale"_.Bind(0.2), "tracking_sigma"_.Bind(0.25),
        "velocity_kick_min"_.Bind(0.1), "velocity_kick_max"_.Bind(1.0),
        "kick_duration_steps_min"_.Bind(1), "kick_duration_steps_max"_.Bind(10),
        "kick_wait_steps_min"_.Bind(50), "kick_wait_steps_max"_.Bind(150));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict("obs:state"_.Bind(
                        StackSpec(Spec<mjtNum>({kBarkourStateDim}, {-inf, inf}),
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
                    "info:reward_stand_still"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_termination"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_feet_air_time"_.Bind(Spec<mjtNum>({-1}))
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
                    "info:last_act"_.Bind(Spec<mjtNum>({kBarkourActionDim})),
                    "info:last_vel"_.Bind(Spec<mjtNum>({kBarkourActionDim})),
                    "info:last_contact"_.Bind(Spec<bool>({kBarkourFeet})),
                    "info:feet_air_time"_.Bind(Spec<mjtNum>({kBarkourFeet})),
                    "info:kick_dir"_.Bind(Spec<mjtNum>({3})),
                    "info:kick_wait_steps"_.Bind(Spec<int>({-1})),
                    "info:kick_duration_steps"_.Bind(Spec<int>({-1})),
                    "info:vel_kick"_.Bind(Spec<mjtNum>({-1})),
                    "info:last_kick_step"_.Bind(Spec<mjtNum>({-1})),
                    "info:step"_.Bind(Spec<int>({-1})),
                    "info:obs_history"_.Bind(Spec<mjtNum>({kBarkourStateDim})),
                    "info:sensordata"_.Bind(Spec<mjtNum>({256}))
#endif
    );  // NOLINT
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<mjtNum>({-1, kBarkourActionDim}, {-1.0, 1.0})));
  }
};

using BarkourAliases = PlaygroundEnvAliases<PlaygroundBarkourEnvFns>;
using PlaygroundBarkourEnvSpec = BarkourAliases::Spec;
using PlaygroundBarkourPixelEnvFns = BarkourAliases::PixelFns;
using PlaygroundBarkourPixelEnvSpec = BarkourAliases::PixelSpec;

template <typename EnvSpecT, bool kFromPixels>
class PlaygroundBarkourEnvBase : public Env<EnvSpecT>,
                                 public PlaygroundMujocoEnv {
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
  std::array<mjtNum, kBarkourActionDim> default_pose_{};
  std::array<mjtNum, kBarkourActionDim> lowers_{};
  std::array<mjtNum, kBarkourActionDim> uppers_{};
  std::array<mjtNum, kBarkourActionDim> last_act_{};
  std::array<mjtNum, kBarkourActionDim> last_vel_{};
  std::array<mjtNum, 3> command_{};
  std::array<bool, kBarkourFeet> last_contact_{};
  std::array<mjtNum, kBarkourFeet> feet_air_time_{};
  std::array<mjtNum, kBarkourStateDim> obs_history_{};
  std::array<mjtNum, 3> kick_dir_{};
  std::array<int, kBarkourFeet> foot_geom_ids_{};
  int floor_geom_id_{-1};
  int torso_body_id_{-1};
  int gyro_adr_{-1};
  int upvector_adr_{-1};
  int global_linvel_adr_{-1};
  int global_angvel_adr_{-1};
  int local_linvel_adr_{-1};
  int step_{0};
  int kick_wait_steps_{50};
  int kick_duration_steps_{1};
  mjtNum vel_kick_{0.0};
  mjtNum last_kick_step_{-std::numeric_limits<mjtNum>::infinity()};
  mjtNum torso_mass_{0.0};
  mjtNum reward_tracking_lin_vel_{0.0};
  mjtNum reward_tracking_ang_vel_{0.0};
  mjtNum reward_lin_vel_z_{0.0};
  mjtNum reward_ang_vel_xy_{0.0};
  mjtNum reward_orientation_{0.0};
  mjtNum reward_torques_{0.0};
  mjtNum reward_action_rate_{0.0};
  mjtNum reward_stand_still_{0.0};
  mjtNum reward_termination_{0.0};
  mjtNum reward_feet_air_time_{0.0};
  std::uniform_real_distribution<mjtNum> unit_uniform_{0.0, 1.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  PlaygroundBarkourEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        PlaygroundMujocoEnv(BarkourXmlPath(spec.config["base_path"_]),
                            spec.config["max_episode_steps"_],
                            spec.config["frame_stack"_],
                            RenderWidthOrDefault<kFromPixels>(spec.config),
                            RenderHeightOrDefault<kFromPixels>(spec.config),
                            RenderCameraIdOrDefault<kFromPixels>(spec.config)) {
    const std::string task_name = spec.config["task_name"_];
    if (task_name != "BarkourJoystick") {
      throw std::runtime_error("Unsupported playground Barkour task_name " +
                               task_name);
    }
    if (model_->nq < 7 + kBarkourActionDim ||
        model_->nv < 6 + kBarkourActionDim || model_->nu != kBarkourActionDim) {
      throw std::runtime_error("Unexpected Barkour model dimensions.");
    }
    n_substeps_ = static_cast<int>(
        std::llround(spec.config["ctrl_dt"_] / spec.config["sim_dt"_]));
    model_->opt.timestep = spec.config["sim_dt"_];
    for (int i = 6; i < model_->nv; ++i) {
      model_->dof_damping[i] = 0.5239;
    }
    for (int i = 0; i < model_->nu; ++i) {
      model_->actuator_gainprm[i * mjNGAIN] = 35.0;
      model_->actuator_biasprm[i * mjNBIAS + 1] = -35.0;
    }

    const int home_id = mj_name2id(model_, mjOBJ_KEY, "home");
    if (home_id < 0) {
      throw std::runtime_error("Barkour model is missing home keyframe.");
    }
    init_qpos_.assign(model_->key_qpos + home_id * model_->nq,
                      model_->key_qpos + (home_id + 1) * model_->nq);
    std::copy(init_qpos_.begin() + 7,
              init_qpos_.begin() + 7 + kBarkourActionDim,
              default_pose_.begin());
    lowers_ = {-0.7, -1.0, 0.05, -0.7, -1.0, 0.05,
               -0.7, -1.0, 0.05, -0.7, -1.0, 0.05};
    uppers_ = {0.52, 2.1, 2.1, 0.52, 2.1, 2.1, 0.52, 2.1, 2.1, 0.52, 2.1, 2.1};
    for (int i = 0; i < kBarkourFeet; ++i) {
      foot_geom_ids_[i] = RequireId(mjOBJ_GEOM, BarkourFeetGeoms()[i]);
    }
    floor_geom_id_ = RequireId(mjOBJ_GEOM, "floor");
    torso_body_id_ = RequireId(mjOBJ_BODY, "torso");
    torso_mass_ = model_->body_subtreemass[torso_body_id_];
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
    last_kick_step_ = -std::numeric_limits<mjtNum>::infinity();
    ResetRewards();
    std::fill(last_act_.begin(), last_act_.end(), 0.0);
    std::fill(last_vel_.begin(), last_vel_.end(), 0.0);
    std::fill(last_contact_.begin(), last_contact_.end(), false);
    std::fill(feet_air_time_.begin(), feet_air_time_.end(), 0.0);
    std::fill(obs_history_.begin(), obs_history_.end(), 0.0);
    std::fill(kick_dir_.begin(), kick_dir_.end(), 0.0);

    mj_resetData(model_, data_);
    std::copy(init_qpos_.begin(), init_qpos_.end(), data_->qpos);
    std::fill(data_->qvel, data_->qvel + model_->nv, 0.0);
    std::copy(default_pose_.begin(), default_pose_.end(), data_->ctrl);
    mj_forward(model_, data_);
    reset_qpos_.assign(data_->qpos, data_->qpos + model_->nq);
    reset_qvel_.assign(data_->qvel, data_->qvel + model_->nv);
    reset_ctrl_.assign(data_->ctrl, data_->ctrl + model_->nu);

    SampleCommand();
    kick_wait_steps_ = UniformInt(spec_.config["kick_wait_steps_min"_],
                                  spec_.config["kick_wait_steps_max"_]);
    kick_duration_steps_ = UniformInt(spec_.config["kick_duration_steps_min"_],
                                      spec_.config["kick_duration_steps_max"_]);
    vel_kick_ = Uniform(spec_.config["velocity_kick_min"_],
                        spec_.config["velocity_kick_max"_]);
    UpdateObs(/*add_noise=*/true);
    WriteState(0.0, true);
  }

  void Step(const Action& action) override {
    has_cached_render_ = false;
    const auto* act = static_cast<const mjtNum*>(action["action"_].Data());
    MaybeApplyPerturbation();

    for (int i = 0; i < kBarkourActionDim; ++i) {
      const mjtNum target =
          default_pose_[i] + act[i] * spec_.config["action_scale"_];
      data_->ctrl[i] = std::clamp(target, lowers_[i], uppers_[i]);
    }
    for (int i = 0; i < n_substeps_; ++i) {
      mj_step(model_, data_);
    }

    UpdateObs(/*add_noise=*/true);
    std::array<bool, kBarkourFeet> contact{};
    for (int i = 0; i < kBarkourFeet; ++i) {
      contact[i] = FootContact(i);
    }
    std::array<bool, kBarkourFeet> first_contact{};
    for (int i = 0; i < kBarkourFeet; ++i) {
      const bool contact_filt = contact[i] || last_contact_[i];
      first_contact[i] = feet_air_time_[i] > 0.0 && contact_filt;
      feet_air_time_[i] += Dt();
    }

    terminated_ = data_->sensordata[upvector_adr_ + 2] < 0.0;
    for (int i = 0; i < kBarkourActionDim; ++i) {
      const mjtNum q = data_->qpos[7 + i];
      terminated_ = terminated_ || q < lowers_[i] || q > uppers_[i];
    }
    terminated_ = terminated_ || data_->xpos[torso_body_id_ * 3 + 2] < 0.18;
    ComputeRewards(act, first_contact);
    mjtNum reward =
        (reward_tracking_lin_vel_ + reward_tracking_ang_vel_ +
         reward_lin_vel_z_ + reward_ang_vel_xy_ + reward_orientation_ +
         reward_torques_ + reward_action_rate_ + reward_stand_still_ +
         reward_termination_ + reward_feet_air_time_) *
        Dt();
    reward = std::clamp(reward, static_cast<mjtNum>(0.0),
                        static_cast<mjtNum>(10000.0));

    std::copy(act, act + kBarkourActionDim, last_act_.begin());
    std::copy(data_->qvel + 6, data_->qvel + 6 + kBarkourActionDim,
              last_vel_.begin());
    ++step_;
    for (int i = 0; i < kBarkourFeet; ++i) {
      feet_air_time_[i] = contact[i] ? 0.0 : feet_air_time_[i];
      last_contact_[i] = contact[i];
    }
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
  static std::string BarkourXmlPath(const std::string& base_path) {
    return base_path +
           "/mujoco/playground/assets/mujoco_menagerie/"
           "google_barkour_vb/scene_mjx.xml";
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

  int UniformInt(int low, int high_exclusive) {
    if (high_exclusive <= low) {
      return low;
    }
    std::uniform_int_distribution<int> dist(low, high_exclusive - 1);
    return dist(gen_);
  }

  void ResetRewards() {
    reward_tracking_lin_vel_ = 0.0;
    reward_tracking_ang_vel_ = 0.0;
    reward_lin_vel_z_ = 0.0;
    reward_ang_vel_xy_ = 0.0;
    reward_orientation_ = 0.0;
    reward_torques_ = 0.0;
    reward_action_rate_ = 0.0;
    reward_stand_still_ = 0.0;
    reward_termination_ = 0.0;
    reward_feet_air_time_ = 0.0;
  }

  void SampleCommand() {
    command_[0] =
        Uniform(spec_.config["lin_vel_x_min"_], spec_.config["lin_vel_x_max"_]);
    command_[1] =
        Uniform(spec_.config["lin_vel_y_min"_], spec_.config["lin_vel_y_max"_]);
    command_[2] = Uniform(spec_.config["ang_vel_yaw_min"_],
                          spec_.config["ang_vel_yaw_max"_]);
  }

  bool FootContact(int foot_index) const {
    const int foot_geom = foot_geom_ids_[foot_index];
    return std::any_of(data_->contact, data_->contact + data_->ncon,
                       [this, foot_geom](const mjContact& contact) {
                         return (contact.geom1 == foot_geom &&
                                 contact.geom2 == floor_geom_id_) ||
                                (contact.geom2 == foot_geom &&
                                 contact.geom1 == floor_geom_id_);
                       });
  }

  void MaybeApplyPerturbation() {
    std::fill(data_->xfrc_applied, data_->xfrc_applied + model_->nbody * 6,
              0.0);
    if (kick_wait_steps_ <= 0 || kick_duration_steps_ <= 0) {
      return;
    }
    const bool start_kick = step_ != 0 && step_ % kick_wait_steps_ == 0;
    if (start_kick) {
      const mjtNum angle = Uniform(0.0, 2.0 * M_PI);
      kick_dir_[0] = std::cos(angle);
      kick_dir_[1] = std::sin(angle);
      kick_dir_[2] = 0.0;
      last_kick_step_ = step_;
    }
    if (!std::isfinite(last_kick_step_)) {
      return;
    }
    const mjtNum duration =
        std::clamp(static_cast<mjtNum>(step_) - last_kick_step_,
                   static_cast<mjtNum>(0.0), static_cast<mjtNum>(100000.0));
    if (duration >= kick_duration_steps_) {
      return;
    }
    const mjtNum u_t = 0.5 * std::sin(M_PI * duration / kick_duration_steps_);
    const mjtNum force =
        u_t * torso_mass_ * vel_kick_ / (kick_duration_steps_ * Dt());
    mjtNum* xfrc = data_->xfrc_applied + torso_body_id_ * 6;
    xfrc[0] = force * kick_dir_[0];
    xfrc[1] = force * kick_dir_[1];
    xfrc[2] = force * kick_dir_[2];
  }

  void UpdateObs(bool add_noise) {
    std::array<mjtNum, kBarkourObsDim> obs{};
    int index = 0;
    obs[index++] = data_->sensordata[gyro_adr_ + 2] * 0.25;
    for (int i = 0; i < 3; ++i) {
      obs[index++] = data_->sensordata[upvector_adr_ + i];
    }
    obs[index++] = command_[0] * 2.0;
    obs[index++] = command_[1] * 2.0;
    obs[index++] = command_[2] * 0.25;
    for (int i = 0; i < kBarkourActionDim; ++i) {
      obs[index++] = data_->qpos[7 + i] - default_pose_[i];
    }
    for (int i = 0; i < kBarkourActionDim; ++i) {
      obs[index++] = last_act_[i];
    }
    for (int i = 0; i < kBarkourObsDim; ++i) {
      obs[i] = std::clamp(obs[i], static_cast<mjtNum>(-100.0),
                          static_cast<mjtNum>(100.0));
      if (add_noise && spec_.config["obs_noise"_] >= 0.0) {
        obs[i] += spec_.config["obs_noise"_] * Uniform(-1.0, 1.0);
      }
    }
    for (int i = kBarkourStateDim - 1; i >= kBarkourObsDim; --i) {
      obs_history_[i] = obs_history_[i - kBarkourObsDim];
    }
    std::copy(obs.begin(), obs.end(), obs_history_.begin());
  }

  void ComputeRewards(const mjtNum* act,
                      const std::array<bool, kBarkourFeet>& first_contact) {
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
    for (int i = 0; i < model_->nv; ++i) {
      torque_sq += data_->qfrc_actuator[i] * data_->qfrc_actuator[i];
      torque_abs += std::abs(data_->qfrc_actuator[i]);
    }
    reward_torques_ =
        (std::sqrt(torque_sq) + torque_abs) * spec_.config["torques_scale"_];

    mjtNum action_rate = 0.0;
    for (int i = 0; i < kBarkourActionDim; ++i) {
      const mjtNum delta = act[i] - last_act_[i];
      action_rate += delta * delta;
    }
    reward_action_rate_ = action_rate * spec_.config["action_rate_scale"_];

    const mjtNum cmd_norm =
        std::sqrt(command_[0] * command_[0] + command_[1] * command_[1]);
    mjtNum stand_still = 0.0;
    if (cmd_norm > 0.0 && command_[1] / cmd_norm < 0.1) {
      for (int i = 0; i < kBarkourActionDim; ++i) {
        stand_still += std::abs(data_->qpos[7 + i] - default_pose_[i]);
      }
    }
    reward_stand_still_ = stand_still * spec_.config["stand_still_scale"_];

    reward_termination_ = (terminated_ && step_ < 500 ? 1.0 : 0.0) *
                          spec_.config["termination_scale"_];

    mjtNum air_time = 0.0;
    for (int i = 0; i < kBarkourFeet; ++i) {
      if (first_contact[i]) {
        air_time += feet_air_time_[i] - 0.1;
      }
    }
    if (cmd_norm <= 0.05) {
      air_time = 0.0;
    }
    reward_feet_air_time_ = air_time * spec_.config["feet_air_time_scale"_];
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
    state["info:reward_stand_still"_] = reward_stand_still_;
    state["info:reward_termination"_] = reward_termination_;
    state["info:reward_feet_air_time"_] = reward_feet_air_time_;
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
    state["info:last_vel"_].Assign(last_vel_.data(), last_vel_.size());
    state["info:last_contact"_].Assign(last_contact_.data(),
                                       last_contact_.size());
    state["info:feet_air_time"_].Assign(feet_air_time_.data(),
                                        feet_air_time_.size());
    state["info:kick_dir"_].Assign(kick_dir_.data(), kick_dir_.size());
    state["info:kick_wait_steps"_] = kick_wait_steps_;
    state["info:kick_duration_steps"_] = kick_duration_steps_;
    state["info:vel_kick"_] = vel_kick_;
    state["info:last_kick_step"_] = last_kick_step_;
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
using BarkourBase = PlaygroundBarkourEnvBase<Spec, kFromPixels>;
using BarkourEnv = BarkourBase<PlaygroundBarkourEnvSpec, false>;
using BarkourPixelEnv = BarkourBase<PlaygroundBarkourPixelEnvSpec, true>;
using PlaygroundBarkourEnv = BarkourEnv;
using PlaygroundBarkourPixelEnv = BarkourPixelEnv;
using PlaygroundBarkourEnvPool = PlaygroundEnvPoolT<PlaygroundBarkourEnv>;
using BarkourPixelEnvPool = PlaygroundEnvPoolT<PlaygroundBarkourPixelEnv>;
using PlaygroundBarkourPixelEnvPool = BarkourPixelEnvPool;

}  // namespace mujoco_playground

#endif  // ENVPOOL_MUJOCO_PLAYGROUND_BARKOUR_H_
