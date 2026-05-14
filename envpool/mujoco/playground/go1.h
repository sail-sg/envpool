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

#ifndef ENVPOOL_MUJOCO_PLAYGROUND_GO1_H_
#define ENVPOOL_MUJOCO_PLAYGROUND_GO1_H_

#include <mujoco.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
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

using envpool::mujoco::PixelObservationEnvFns;
using envpool::mujoco::RenderCameraIdOrDefault;
using envpool::mujoco::RenderHeightOrDefault;
using envpool::mujoco::RenderWidthOrDefault;
using envpool::mujoco::StackSpec;

constexpr int kGo1ActionDim = 12;
constexpr int kGo1StateDim = 48;
constexpr int kGo1PrivilegedStateDim = 123;
constexpr int kGo1GetupStateDim = 42;
constexpr int kGo1GetupPrivilegedStateDim = 91;
constexpr int kGo1HandstandStateDim = 45;
constexpr int kGo1HandstandPrivilegedStateDim = 94;
constexpr int kGo1Feet = 4;
constexpr int kGo1HandstandContactGeoms = 12;
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
constexpr const char* kGo1FeetSites[kGo1Feet] = {"FR", "FL", "RR", "RL"};

class PlaygroundGo1EnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "frame_stack"_.Bind(1),
        "task_name"_.Bind(std::string("Go1JoystickFlatTerrain")),
        "ctrl_dt"_.Bind(0.02), "sim_dt"_.Bind(0.004), "kp"_.Bind(35.0),
        "kd"_.Bind(0.5), "action_scale"_.Bind(0.5),
        "soft_joint_pos_limit_factor"_.Bind(0.95), "noise_level"_.Bind(1.0),
        "noise_joint_pos"_.Bind(0.03), "noise_joint_vel"_.Bind(1.5),
        "noise_gyro"_.Bind(0.2), "noise_gravity"_.Bind(0.05),
        "noise_linvel"_.Bind(0.1), "tracking_lin_vel_scale"_.Bind(1.0),
        "tracking_ang_vel_scale"_.Bind(0.5), "lin_vel_z_scale"_.Bind(-0.5),
        "ang_vel_xy_scale"_.Bind(-0.05), "orientation_scale"_.Bind(-5.0),
        "dof_pos_limits_scale"_.Bind(-1.0), "pose_scale"_.Bind(0.5),
        "termination_scale"_.Bind(-1.0), "stand_still_scale"_.Bind(-1.0),
        "torques_scale"_.Bind(-0.0002), "action_rate_scale"_.Bind(-0.01),
        "energy_scale"_.Bind(-0.001), "feet_clearance_scale"_.Bind(-2.0),
        "feet_height_scale"_.Bind(-0.2), "feet_slip_scale"_.Bind(-0.1),
        "feet_air_time_scale"_.Bind(0.1), "tracking_sigma"_.Bind(0.25),
        "max_foot_height"_.Bind(0.1), "command_a0"_.Bind(1.5),
        "command_a1"_.Bind(0.8), "command_a2"_.Bind(1.2),
        "command_b0"_.Bind(0.9), "command_b1"_.Bind(0.25),
        "command_b2"_.Bind(0.5));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs:state"_.Bind(StackSpec(Spec<mjtNum>({kGo1StateDim}, {-inf, inf}),
                                    conf["frame_stack"_])),
        "obs:privileged_state"_.Bind(
            StackSpec(Spec<mjtNum>({kGo1PrivilegedStateDim}, {-inf, inf}),
                      conf["frame_stack"_])),
        "info:terminated"_.Bind(Spec<bool>({-1})),
        "info:command"_.Bind(Spec<mjtNum>({-1, 3})),
        "info:steps_until_next_cmd"_.Bind(Spec<int>({-1})),
        "info:reward_tracking_lin_vel"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_tracking_ang_vel"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_lin_vel_z"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_ang_vel_xy"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_orientation"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_dof_pos_limits"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_pose"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_termination"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_stand_still"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_torques"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_action_rate"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_energy"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_feet_clearance"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_feet_height"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_feet_slip"_.Bind(Spec<mjtNum>({-1})),
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
        "info:last_act"_.Bind(Spec<mjtNum>({kGo1ActionDim})),
        "info:last_last_act"_.Bind(Spec<mjtNum>({kGo1ActionDim})),
        "info:feet_air_time"_.Bind(Spec<mjtNum>({kGo1Feet})),
        "info:last_contact"_.Bind(Spec<bool>({kGo1Feet})),
        "info:swing_peak"_.Bind(Spec<mjtNum>({kGo1Feet})),
        "info:sensordata"_.Bind(Spec<mjtNum>({256}))
#endif
    );  // NOLINT
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<mjtNum>({-1, kGo1ActionDim}, {-1.0, 1.0})));
  }
};

using PlaygroundGo1EnvSpec = EnvSpec<PlaygroundGo1EnvFns>;
using PlaygroundGo1PixelEnvFns = PixelObservationEnvFns<PlaygroundGo1EnvFns>;
using PlaygroundGo1PixelEnvSpec = EnvSpec<PlaygroundGo1PixelEnvFns>;

template <typename EnvSpecT, bool kFromPixels>
class PlaygroundGo1EnvBase : public Env<EnvSpecT>, public PlaygroundMujocoEnv {
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
  std::vector<mjtNum> default_pose_;
  std::array<mjtNum, kGo1ActionDim> last_act_{};
  std::array<mjtNum, kGo1ActionDim> last_last_act_{};
  std::array<mjtNum, 3> command_{};
  std::array<mjtNum, kGo1Feet> feet_air_time_{};
  std::array<mjtNum, kGo1Feet> swing_peak_{};
  std::array<bool, kGo1Feet> last_contact_{};
  std::array<mjtNum, kGo1ActionDim> soft_lowers_{};
  std::array<mjtNum, kGo1ActionDim> soft_uppers_{};
  std::array<int, kGo1Feet> feet_site_ids_{};
  std::array<int, kGo1Feet> feet_floor_sensor_ids_{};
  std::array<int, kGo1Feet> feet_floor_sensor_adrs_{};
  std::array<int, kGo1Feet> feet_linvel_sensor_adrs_{};
  int imu_site_id_{-1};
  int torso_body_id_{-1};
  int gyro_adr_{-1};
  int local_linvel_adr_{-1};
  int accelerometer_adr_{-1};
  int upvector_adr_{-1};
  int global_linvel_adr_{-1};
  int global_angvel_adr_{-1};
  int steps_until_next_cmd_{0};
  std::exponential_distribution<mjtNum> exponential_{1.0};
  std::uniform_real_distribution<mjtNum> unit_uniform_{0.0, 1.0};
  std::uniform_real_distribution<mjtNum> noise_uniform_{-1.0, 1.0};
  mjtNum reward_tracking_lin_vel_{0.0};
  mjtNum reward_tracking_ang_vel_{0.0};
  mjtNum reward_lin_vel_z_{0.0};
  mjtNum reward_ang_vel_xy_{0.0};
  mjtNum reward_orientation_{0.0};
  mjtNum reward_dof_pos_limits_{0.0};
  mjtNum reward_pose_{0.0};
  mjtNum reward_termination_{0.0};
  mjtNum reward_stand_still_{0.0};
  mjtNum reward_torques_{0.0};
  mjtNum reward_action_rate_{0.0};
  mjtNum reward_energy_{0.0};
  mjtNum reward_feet_clearance_{0.0};
  mjtNum reward_feet_height_{0.0};
  mjtNum reward_feet_slip_{0.0};
  mjtNum reward_feet_air_time_{0.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  PlaygroundGo1EnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        PlaygroundMujocoEnv(
            Go1XmlPath(spec.config["base_path"_], spec.config["task_name"_]),
            spec.config["max_episode_steps"_], spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config)) {
    (void)env_id;
    const std::string task_name = spec.config["task_name"_];
    if (task_name != "Go1JoystickFlatTerrain" &&
        task_name != "Go1JoystickRoughTerrain") {
      throw std::runtime_error("Unsupported playground Go1 task_name " +
                               task_name);
    }
    if (model_->nq < 7 + kGo1ActionDim || model_->nv < 6 + kGo1ActionDim ||
        model_->nu != kGo1ActionDim) {
      throw std::runtime_error("Unexpected Go1 model dimensions.");
    }
    model_->opt.timestep = spec.config["sim_dt"_];
    model_->opt.ccd_iterations = 20;
    for (int i = 6; i < model_->nv; ++i) {
      model_->dof_damping[i] = spec.config["kd"_];
    }
    for (int i = 0; i < model_->nu; ++i) {
      model_->actuator_gainprm[i * mjNGAIN + 0] = spec.config["kp"_];
      model_->actuator_biasprm[i * mjNBIAS + 1] = -spec.config["kp"_];
    }
    n_substeps_ = static_cast<int>(
        std::llround(spec.config["ctrl_dt"_] / spec.config["sim_dt"_]));

    const int home_key = mj_name2id(model_, mjOBJ_KEY, "home");
    if (home_key < 0) {
      throw std::runtime_error("Go1 model is missing home keyframe.");
    }
    init_qpos_.assign(model_->key_qpos + home_key * model_->nq,
                      model_->key_qpos + (home_key + 1) * model_->nq);
    default_pose_.assign(init_qpos_.begin() + 7,
                         init_qpos_.begin() + 7 + kGo1ActionDim);
    for (int i = 0; i < kGo1ActionDim; ++i) {
      soft_lowers_[i] = model_->jnt_range[(i + 1) * 2] *
                        spec.config["soft_joint_pos_limit_factor"_];
      soft_uppers_[i] = model_->jnt_range[(i + 1) * 2 + 1] *
                        spec.config["soft_joint_pos_limit_factor"_];
    }
    imu_site_id_ = RequireId(mjOBJ_SITE, "imu");
    torso_body_id_ = RequireId(mjOBJ_BODY, "trunk");
    gyro_adr_ = SensorAdr("gyro");
    local_linvel_adr_ = SensorAdr("local_linvel");
    accelerometer_adr_ = SensorAdr("accelerometer");
    upvector_adr_ = SensorAdr("upvector");
    global_linvel_adr_ = SensorAdr("global_linvel");
    global_angvel_adr_ = SensorAdr("global_angvel");
    for (int i = 0; i < kGo1Feet; ++i) {
      feet_site_ids_[i] = RequireId(mjOBJ_SITE, kGo1FeetSites[i]);
      feet_floor_sensor_ids_[i] = RequireId(
          mjOBJ_SENSOR, std::string(kGo1FeetSites[i]) + "_floor_found");
      feet_floor_sensor_adrs_[i] =
          model_->sensor_adr[feet_floor_sensor_ids_[i]];
      feet_linvel_sensor_adrs_[i] =
          SensorAdr(std::string(kGo1FeetSites[i]) + "_global_linvel");
    }
  }

  static std::string Go1XmlPath(const std::string& base_path,
                                const std::string& task_name) {
    const char* xml_name = task_name == "Go1JoystickRoughTerrain"
                               ? "scene_mjx_feetonly_rough_terrain.xml"
                               : "scene_mjx_feetonly_flat_terrain.xml";
    return base_path +
           "/mujoco/playground/assets/mujoco_playground/_src/locomotion/go1/"
           "xmls/" +
           xml_name;
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    has_cached_render_ = false;
    done_ = false;
    terminated_ = false;
    elapsed_step_ = 0;
    ResetRewards();
    std::fill(last_act_.begin(), last_act_.end(), 0.0);
    std::fill(last_last_act_.begin(), last_last_act_.end(), 0.0);
    std::fill(feet_air_time_.begin(), feet_air_time_.end(), 0.0);
    std::fill(swing_peak_.begin(), swing_peak_.end(), 0.0);
    std::fill(last_contact_.begin(), last_contact_.end(), false);

    mj_resetData(model_, data_);
    std::copy(init_qpos_.begin(), init_qpos_.end(), data_->qpos);
    std::fill(data_->qvel, data_->qvel + model_->nv, 0.0);
    std::copy(default_pose_.begin(), default_pose_.end(), data_->ctrl);

    std::uniform_real_distribution<mjtNum> dxy_dist(-0.5, 0.5);
    std::uniform_real_distribution<mjtNum> yaw_dist(-3.14, 3.14);
    data_->qpos[0] += dxy_dist(gen_);
    data_->qpos[1] += dxy_dist(gen_);
    const mjtNum yaw = yaw_dist(gen_);
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const mjtNum yaw_quat[4] = {std::cos(yaw / 2.0), 0.0, 0.0,
                                std::sin(yaw / 2.0)};
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum new_quat[4];
    mju_mulQuat(new_quat, data_->qpos + 3, yaw_quat);
    std::memcpy(data_->qpos + 3, new_quat, sizeof(mjtNum) * 4);
    for (int i = 0; i < 6; ++i) {
      data_->qvel[i] = dxy_dist(gen_);
    }
    mj_forward(model_, data_);
    reset_qpos_.assign(data_->qpos, data_->qpos + model_->nq);
    reset_qvel_.assign(data_->qvel, data_->qvel + model_->nv);
    reset_ctrl_.assign(data_->ctrl, data_->ctrl + model_->nu);

    steps_until_next_cmd_ =
        static_cast<int>(std::llround(exponential_(gen_) * 5.0 / Dt()));
    command_[0] = UniformSymmetric(spec_.config["command_a0"_]);
    command_[1] = UniformSymmetric(spec_.config["command_a1"_]);
    command_[2] = UniformSymmetric(spec_.config["command_a2"_]);
    WriteState(0.0, true);
  }

  void Step(const Action& action) override {
    has_cached_render_ = false;
    const auto* act = static_cast<const mjtNum*>(action["action"_].Data());
    for (int i = 0; i < kGo1ActionDim; ++i) {
      data_->ctrl[i] =
          default_pose_[i] + act[i] * spec_.config["action_scale"_];
    }
    for (int i = 0; i < n_substeps_; ++i) {
      mj_step(model_, data_);
    }

    std::array<bool, kGo1Feet> contact{};
    std::array<bool, kGo1Feet> contact_filt{};
    std::array<bool, kGo1Feet> first_contact{};
    for (int i = 0; i < kGo1Feet; ++i) {
      contact[i] = data_->sensordata[feet_floor_sensor_adrs_[i]] > 0.0;
      contact_filt[i] = contact[i] || last_contact_[i];
      first_contact[i] = (feet_air_time_[i] > 0.0) && contact_filt[i];
      feet_air_time_[i] += Dt();
      swing_peak_[i] =
          std::max(swing_peak_[i], data_->site_xpos[feet_site_ids_[i] * 3 + 2]);
    }

    terminated_ = data_->sensordata[upvector_adr_ + 2] < 0.0;
    ComputeRewards(act, first_contact, contact);
    mjtNum reward =
        (reward_tracking_lin_vel_ + reward_tracking_ang_vel_ +
         reward_lin_vel_z_ + reward_ang_vel_xy_ + reward_orientation_ +
         reward_dof_pos_limits_ + reward_pose_ + reward_termination_ +
         reward_stand_still_ + reward_torques_ + reward_action_rate_ +
         reward_energy_ + reward_feet_clearance_ + reward_feet_height_ +
         reward_feet_slip_ + reward_feet_air_time_) *
        Dt();
    reward = std::clamp(reward, static_cast<mjtNum>(0.0),
                        static_cast<mjtNum>(10000.0));

    --steps_until_next_cmd_;
    if (steps_until_next_cmd_ <= 0 || terminated_) {
      SampleCommand();
      steps_until_next_cmd_ =
          static_cast<int>(std::llround(exponential_(gen_) * 5.0 / Dt()));
    }
    ++elapsed_step_;
    done_ = terminated_ || elapsed_step_ >= max_episode_steps_;
    WriteState(static_cast<float>(reward), false);

    for (int i = 0; i < kGo1ActionDim; ++i) {
      last_last_act_[i] = last_act_[i];
      last_act_[i] = act[i];
    }
    for (int i = 0; i < kGo1Feet; ++i) {
      if (contact[i]) {
        feet_air_time_[i] = 0.0;
        swing_peak_[i] = 0.0;
      }
      last_contact_[i] = contact[i];
    }
  }

 private:
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

  mjtNum Dt() const { return spec_.config["ctrl_dt"_]; }

  mjtNum UniformSymmetric(mjtNum bound) {
    return (unit_uniform_(gen_) * 2.0 - 1.0) * bound;
  }

  mjtNum MaybeNoisy(mjtNum value, mjtNum scale) {
    return value + noise_uniform_(gen_) * spec_.config["noise_level"_] * scale;
  }

  void Sensor3(int adr, mjtNum* out) const {
    out[0] = data_->sensordata[adr];
    out[1] = data_->sensordata[adr + 1];
    out[2] = data_->sensordata[adr + 2];
  }

  void Gravity(mjtNum* out) const {
    const mjtNum* mat = data_->site_xmat + imu_site_id_ * 9;
    out[0] = -mat[6];
    out[1] = -mat[7];
    out[2] = -mat[8];
  }

  mjtNum Norm3(const std::array<mjtNum, 3>& x) const {
    return std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
  }

  void SampleCommand() {
    const std::array<mjtNum, 3> a = {
        spec_.config["command_a0"_],
        spec_.config["command_a1"_],
        spec_.config["command_a2"_],
    };
    const std::array<mjtNum, 3> b = {
        spec_.config["command_b0"_],
        spec_.config["command_b1"_],
        spec_.config["command_b2"_],
    };
    for (int i = 0; i < 3; ++i) {
      const mjtNum y = UniformSymmetric(a[i]);
      const mjtNum z = unit_uniform_(gen_) < b[i] ? 1.0 : 0.0;
      const mjtNum w = unit_uniform_(gen_) < 0.5 ? 1.0 : 0.0;
      command_[i] = command_[i] - w * (command_[i] - y * z);
    }
  }

  void ResetRewards() {
    reward_tracking_lin_vel_ = 0.0;
    reward_tracking_ang_vel_ = 0.0;
    reward_lin_vel_z_ = 0.0;
    reward_ang_vel_xy_ = 0.0;
    reward_orientation_ = 0.0;
    reward_dof_pos_limits_ = 0.0;
    reward_pose_ = 0.0;
    reward_termination_ = 0.0;
    reward_stand_still_ = 0.0;
    reward_torques_ = 0.0;
    reward_action_rate_ = 0.0;
    reward_energy_ = 0.0;
    reward_feet_clearance_ = 0.0;
    reward_feet_height_ = 0.0;
    reward_feet_slip_ = 0.0;
    reward_feet_air_time_ = 0.0;
  }

  void ComputeRewards(const mjtNum* action,
                      const std::array<bool, kGo1Feet>& first_contact,
                      const std::array<bool, kGo1Feet>& contact) {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum local_linvel[3];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum gyro[3];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum global_linvel[3];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum global_angvel[3];
    Sensor3(local_linvel_adr_, local_linvel);
    Sensor3(gyro_adr_, gyro);
    Sensor3(global_linvel_adr_, global_linvel);
    Sensor3(global_angvel_adr_, global_angvel);
    const mjtNum lin_vel_error =
        (command_[0] - local_linvel[0]) * (command_[0] - local_linvel[0]) +
        (command_[1] - local_linvel[1]) * (command_[1] - local_linvel[1]);
    const mjtNum ang_vel_error =
        (command_[2] - gyro[2]) * (command_[2] - gyro[2]);
    reward_tracking_lin_vel_ =
        std::exp(-lin_vel_error / spec_.config["tracking_sigma"_]) *
        spec_.config["tracking_lin_vel_scale"_];
    reward_tracking_ang_vel_ =
        std::exp(-ang_vel_error / spec_.config["tracking_sigma"_]) *
        spec_.config["tracking_ang_vel_scale"_];
    reward_lin_vel_z_ =
        global_linvel[2] * global_linvel[2] * spec_.config["lin_vel_z_scale"_];
    reward_ang_vel_xy_ = (global_angvel[0] * global_angvel[0] +
                          global_angvel[1] * global_angvel[1]) *
                         spec_.config["ang_vel_xy_scale"_];
    reward_orientation_ =
        (data_->sensordata[upvector_adr_] * data_->sensordata[upvector_adr_] +
         data_->sensordata[upvector_adr_ + 1] *
             data_->sensordata[upvector_adr_ + 1]) *
        spec_.config["orientation_scale"_];

    mjtNum pose_weighted_error = 0.0;
    mjtNum stand_still = 0.0;
    mjtNum joint_limit = 0.0;
    const std::array<mjtNum, 3> command_arr = {command_[0], command_[1],
                                               command_[2]};
    const bool zero_command = Norm3(command_arr) < 0.01;
    for (int i = 0; i < kGo1ActionDim; ++i) {
      const mjtNum q = data_->qpos[7 + i];
      const mjtNum delta = q - default_pose_[i];
      const mjtNum weight = (i % 3 == 2) ? 0.1 : 1.0;
      pose_weighted_error += delta * delta * weight;
      stand_still += std::abs(delta);
      joint_limit += -std::min(q - soft_lowers_[i], static_cast<mjtNum>(0.0));
      joint_limit += std::max(q - soft_uppers_[i], static_cast<mjtNum>(0.0));
    }
    reward_pose_ = std::exp(-pose_weighted_error) * spec_.config["pose_scale"_];
    reward_stand_still_ = stand_still * (zero_command ? 1.0 : 0.0) *
                          spec_.config["stand_still_scale"_];
    reward_dof_pos_limits_ =
        joint_limit * spec_.config["dof_pos_limits_scale"_];
    reward_termination_ =
        (terminated_ ? 1.0 : 0.0) * spec_.config["termination_scale"_];

    mjtNum torque_l2 = 0.0;
    mjtNum torque_l1 = 0.0;
    mjtNum energy = 0.0;
    mjtNum action_rate = 0.0;
    for (int i = 0; i < kGo1ActionDim; ++i) {
      const mjtNum torque = data_->actuator_force[i];
      torque_l2 += torque * torque;
      torque_l1 += std::abs(torque);
      energy += std::abs(data_->qvel[6 + i]) * std::abs(torque);
      const mjtNum da = action[i] - last_act_[i];
      action_rate += da * da;
    }
    reward_torques_ =
        (std::sqrt(torque_l2) + torque_l1) * spec_.config["torques_scale"_];
    reward_energy_ = energy * spec_.config["energy_scale"_];
    reward_action_rate_ = action_rate * spec_.config["action_rate_scale"_];

    mjtNum feet_slip = 0.0;
    mjtNum feet_clearance = 0.0;
    mjtNum feet_height = 0.0;
    mjtNum feet_air_time = 0.0;
    const bool nonzero_command = !zero_command;
    for (int i = 0; i < kGo1Feet; ++i) {
      const mjtNum vx = data_->sensordata[feet_linvel_sensor_adrs_[i]];
      const mjtNum vy = data_->sensordata[feet_linvel_sensor_adrs_[i] + 1];
      const mjtNum vel_xy_sq = vx * vx + vy * vy;
      feet_slip +=
          vel_xy_sq * (contact[i] ? 1.0 : 0.0) * (nonzero_command ? 1.0 : 0.0);
      const mjtNum vel_norm = std::sqrt(std::sqrt(vel_xy_sq));
      const mjtNum foot_z = data_->site_xpos[feet_site_ids_[i] * 3 + 2];
      feet_clearance +=
          std::abs(foot_z - spec_.config["max_foot_height"_]) * vel_norm;
      const mjtNum error =
          swing_peak_[i] / spec_.config["max_foot_height"_] - 1.0;
      feet_height += error * error * (first_contact[i] ? 1.0 : 0.0) *
                     (nonzero_command ? 1.0 : 0.0);
      feet_air_time += (feet_air_time_[i] - 0.1) *
                       (first_contact[i] ? 1.0 : 0.0) *
                       (nonzero_command ? 1.0 : 0.0);
    }
    reward_feet_slip_ = feet_slip * spec_.config["feet_slip_scale"_];
    reward_feet_clearance_ =
        feet_clearance * spec_.config["feet_clearance_scale"_];
    reward_feet_height_ = feet_height * spec_.config["feet_height_scale"_];
    reward_feet_air_time_ =
        feet_air_time * spec_.config["feet_air_time_scale"_];
  }

  void FillObs(mjtNum* state, mjtNum* privileged_state) {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum gyro[3];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum gravity[3];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum local_linvel[3];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum accelerometer[3];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum global_angvel[3];
    Sensor3(gyro_adr_, gyro);
    Gravity(gravity);
    Sensor3(local_linvel_adr_, local_linvel);
    Sensor3(accelerometer_adr_, accelerometer);
    Sensor3(global_angvel_adr_, global_angvel);

    int n = 0;
    for (mjtNum value : local_linvel) {
      state[n++] = MaybeNoisy(value, spec_.config["noise_linvel"_]);
    }
    for (mjtNum value : gyro) {
      state[n++] = MaybeNoisy(value, spec_.config["noise_gyro"_]);
    }
    for (mjtNum value : gravity) {
      state[n++] = MaybeNoisy(value, spec_.config["noise_gravity"_]);
    }
    for (int i = 0; i < kGo1ActionDim; ++i) {
      state[n++] =
          MaybeNoisy(data_->qpos[7 + i], spec_.config["noise_joint_pos"_]) -
          default_pose_[i];
    }
    for (int i = 0; i < kGo1ActionDim; ++i) {
      state[n++] =
          MaybeNoisy(data_->qvel[6 + i], spec_.config["noise_joint_vel"_]);
    }
    for (int i = 0; i < kGo1ActionDim; ++i) {
      state[n++] = last_act_[i];
    }
    for (int i = 0; i < 3; ++i) {
      state[n++] = command_[i];
    }

    int p = 0;
    for (int i = 0; i < kGo1StateDim; ++i) {
      privileged_state[p++] = state[i];
    }
    for (mjtNum value : gyro) {
      privileged_state[p++] = value;
    }
    for (mjtNum value : accelerometer) {
      privileged_state[p++] = value;
    }
    for (mjtNum value : gravity) {
      privileged_state[p++] = value;
    }
    for (mjtNum value : local_linvel) {
      privileged_state[p++] = value;
    }
    for (mjtNum value : global_angvel) {
      privileged_state[p++] = value;
    }
    for (int i = 0; i < kGo1ActionDim; ++i) {
      privileged_state[p++] = data_->qpos[7 + i] - default_pose_[i];
    }
    for (int i = 0; i < kGo1ActionDim; ++i) {
      privileged_state[p++] = data_->qvel[6 + i];
    }
    for (int i = 0; i < kGo1ActionDim; ++i) {
      privileged_state[p++] = data_->actuator_force[i];
    }
    for (int i = 0; i < kGo1Feet; ++i) {
      privileged_state[p++] = last_contact_[i] ? 1.0 : 0.0;
    }
    for (int foot = 0; foot < kGo1Feet; ++foot) {
      for (int axis = 0; axis < 3; ++axis) {
        privileged_state[p++] =
            data_->sensordata[feet_linvel_sensor_adrs_[foot] + axis];
      }
    }
    for (int i = 0; i < kGo1Feet; ++i) {
      privileged_state[p++] = feet_air_time_[i];
    }
    for (int i = 0; i < 3; ++i) {
      privileged_state[p++] = data_->xfrc_applied[torso_body_id_ * 6 + i];
    }
    privileged_state[p++] = 0.0;
  }

  void WriteState(float reward, bool reset) {
    auto state = Allocate();
    state["reward"_] = reward;
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      auto obs_state = state["obs:state"_];
      auto obs_privileged = state["obs:privileged_state"_];
      mjtNum* obs = PrepareObservation("obs:state", &obs_state);
      mjtNum* priv =
          PrepareObservation("obs:privileged_state", &obs_privileged);
      FillObs(obs, priv);
      CommitObservation("obs:state", &obs_state, reset);
      CommitObservation("obs:privileged_state", &obs_privileged, reset);
    }
    state["info:terminated"_] = terminated_;
    state["info:command"_].Assign(command_.data(), command_.size());
    state["info:steps_until_next_cmd"_] = steps_until_next_cmd_;
    state["info:reward_tracking_lin_vel"_] = reward_tracking_lin_vel_;
    state["info:reward_tracking_ang_vel"_] = reward_tracking_ang_vel_;
    state["info:reward_lin_vel_z"_] = reward_lin_vel_z_;
    state["info:reward_ang_vel_xy"_] = reward_ang_vel_xy_;
    state["info:reward_orientation"_] = reward_orientation_;
    state["info:reward_dof_pos_limits"_] = reward_dof_pos_limits_;
    state["info:reward_pose"_] = reward_pose_;
    state["info:reward_termination"_] = reward_termination_;
    state["info:reward_stand_still"_] = reward_stand_still_;
    state["info:reward_torques"_] = reward_torques_;
    state["info:reward_action_rate"_] = reward_action_rate_;
    state["info:reward_energy"_] = reward_energy_;
    state["info:reward_feet_clearance"_] = reward_feet_clearance_;
    state["info:reward_feet_height"_] = reward_feet_height_;
    state["info:reward_feet_slip"_] = reward_feet_slip_;
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
    std::copy(data_->qacc, data_->qacc + model_->nv, pad.begin());
    state["info:qacc"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(data_->qacc_warmstart, data_->qacc_warmstart + model_->nv,
              pad.begin());
    state["info:qacc_warmstart"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(reset_ctrl_.begin(), reset_ctrl_.end(), pad.begin());
    state["info:ctrl0"_].Assign(pad.data(), pad.size());
    state["info:last_act"_].Assign(last_act_.data(), last_act_.size());
    state["info:last_last_act"_].Assign(last_last_act_.data(),
                                        last_last_act_.size());
    state["info:feet_air_time"_].Assign(feet_air_time_.data(),
                                        feet_air_time_.size());
    state["info:last_contact"_].Assign(last_contact_.data(),
                                       last_contact_.size());
    state["info:swing_peak"_].Assign(swing_peak_.data(), swing_peak_.size());
    std::array<mjtNum, 256> sensor_pad{};
    std::copy(data_->sensordata, data_->sensordata + model_->nsensordata,
              sensor_pad.begin());
    state["info:sensordata"_].Assign(sensor_pad.data(), sensor_pad.size());
#endif
  }
};

using PlaygroundGo1Env = PlaygroundGo1EnvBase<PlaygroundGo1EnvSpec, false>;
using PlaygroundGo1PixelEnv =
    PlaygroundGo1EnvBase<PlaygroundGo1PixelEnvSpec, true>;
using PlaygroundGo1EnvPool = AsyncEnvPool<PlaygroundGo1Env>;
using PlaygroundGo1PixelEnvPool = AsyncEnvPool<PlaygroundGo1PixelEnv>;

class PlaygroundGo1GetupEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "frame_stack"_.Bind(1), "task_name"_.Bind(std::string("Go1Getup")),
        "ctrl_dt"_.Bind(0.02), "sim_dt"_.Bind(0.004), "kp"_.Bind(35.0),
        "kd"_.Bind(0.5), "action_scale"_.Bind(0.5),
        "soft_joint_pos_limit_factor"_.Bind(0.95), "noise_level"_.Bind(1.0),
        "noise_joint_pos"_.Bind(0.03), "noise_joint_vel"_.Bind(1.5),
        "noise_gyro"_.Bind(0.2), "noise_gravity"_.Bind(0.05),
        "drop_from_height_prob"_.Bind(0.6), "settle_time"_.Bind(0.5),
        "energy_termination_threshold"_.Bind(
            std::numeric_limits<mjtNum>::infinity()),
        "orientation_scale"_.Bind(1.0), "torso_height_scale"_.Bind(1.0),
        "posture_scale"_.Bind(1.0), "stand_still_scale"_.Bind(1.0),
        "action_rate_scale"_.Bind(-0.001), "dof_pos_limits_scale"_.Bind(-0.1),
        "torques_scale"_.Bind(-1e-5), "dof_acc_scale"_.Bind(-2.5e-7),
        "dof_vel_scale"_.Bind(-0.1));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs:state"_.Bind(
            StackSpec(Spec<mjtNum>({kGo1GetupStateDim}, {-inf, inf}),
                      conf["frame_stack"_])),
        "obs:privileged_state"_.Bind(
            StackSpec(Spec<mjtNum>({kGo1GetupPrivilegedStateDim}, {-inf, inf}),
                      conf["frame_stack"_])),
        "info:terminated"_.Bind(Spec<bool>({-1})),
        "info:reward_orientation"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_torso_height"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_posture"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_stand_still"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_action_rate"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_dof_pos_limits"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_torques"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_dof_acc"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_dof_vel"_.Bind(Spec<mjtNum>({-1}))
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
        "info:last_act"_.Bind(Spec<mjtNum>({kGo1ActionDim})),
        "info:last_last_act"_.Bind(Spec<mjtNum>({kGo1ActionDim})),
        "info:sensordata"_.Bind(Spec<mjtNum>({256}))
#endif
    );  // NOLINT
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<mjtNum>({-1, kGo1ActionDim}, {-1.0, 1.0})));
  }
};

using PlaygroundGo1GetupEnvSpec = EnvSpec<PlaygroundGo1GetupEnvFns>;
using PlaygroundGo1GetupPixelEnvFns =
    PixelObservationEnvFns<PlaygroundGo1GetupEnvFns>;
using PlaygroundGo1GetupPixelEnvSpec = EnvSpec<PlaygroundGo1GetupPixelEnvFns>;

template <typename EnvSpecT, bool kFromPixels>
class PlaygroundGo1GetupEnvBase : public Env<EnvSpecT>,
                                  public PlaygroundMujocoEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  int n_substeps_{5};
  int settle_steps_{125};
  std::vector<mjtNum> init_qpos_;
  std::vector<mjtNum> reset_qpos_;
  std::vector<mjtNum> reset_qvel_;
  std::vector<mjtNum> reset_ctrl_;
  std::vector<mjtNum> default_pose_;
  std::array<mjtNum, kGo1ActionDim> last_act_{};
  std::array<mjtNum, kGo1ActionDim> last_last_act_{};
  std::array<mjtNum, kGo1ActionDim> soft_lowers_{};
  std::array<mjtNum, kGo1ActionDim> soft_uppers_{};
  int imu_site_id_{-1};
  int gyro_adr_{-1};
  int local_linvel_adr_{-1};
  int accelerometer_adr_{-1};
  int global_angvel_adr_{-1};
  std::uniform_real_distribution<mjtNum> unit_uniform_{0.0, 1.0};
  std::uniform_real_distribution<mjtNum> half_uniform_{-0.5, 0.5};
  std::uniform_real_distribution<mjtNum> noise_uniform_{-1.0, 1.0};
  std::normal_distribution<mjtNum> normal_{0.0, 1.0};
  mjtNum reward_orientation_{0.0};
  mjtNum reward_torso_height_{0.0};
  mjtNum reward_posture_{0.0};
  mjtNum reward_stand_still_{0.0};
  mjtNum reward_action_rate_{0.0};
  mjtNum reward_dof_pos_limits_{0.0};
  mjtNum reward_torques_{0.0};
  mjtNum reward_dof_acc_{0.0};
  mjtNum reward_dof_vel_{0.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  PlaygroundGo1GetupEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        PlaygroundMujocoEnv(Go1GetupXmlPath(spec.config["base_path"_]),
                            spec.config["max_episode_steps"_],
                            spec.config["frame_stack"_],
                            RenderWidthOrDefault<kFromPixels>(spec.config),
                            RenderHeightOrDefault<kFromPixels>(spec.config),
                            RenderCameraIdOrDefault<kFromPixels>(spec.config)) {
    (void)env_id;
    const std::string task_name = spec.config["task_name"_];
    if (task_name != "Go1Getup") {
      throw std::runtime_error("Unsupported playground Go1 getup task_name " +
                               task_name);
    }
    if (model_->nq < 7 + kGo1ActionDim || model_->nv < 6 + kGo1ActionDim ||
        model_->nu != kGo1ActionDim) {
      throw std::runtime_error("Unexpected Go1 getup model dimensions.");
    }
    model_->opt.timestep = spec.config["sim_dt"_];
    model_->opt.ccd_iterations = 20;
    for (int i = 6; i < model_->nv; ++i) {
      model_->dof_damping[i] = spec.config["kd"_];
    }
    for (int i = 0; i < model_->nu; ++i) {
      model_->actuator_gainprm[i * mjNGAIN + 0] = spec.config["kp"_];
      model_->actuator_biasprm[i * mjNBIAS + 1] = -spec.config["kp"_];
    }
    n_substeps_ = static_cast<int>(
        std::llround(spec.config["ctrl_dt"_] / spec.config["sim_dt"_]));
    settle_steps_ =
        static_cast<int>(spec.config["settle_time"_] / spec.config["sim_dt"_]);

    const int home_key = mj_name2id(model_, mjOBJ_KEY, "home");
    if (home_key < 0) {
      throw std::runtime_error("Go1 getup model is missing home keyframe.");
    }
    init_qpos_.assign(model_->key_qpos + home_key * model_->nq,
                      model_->key_qpos + (home_key + 1) * model_->nq);
    default_pose_.assign(init_qpos_.begin() + 7,
                         init_qpos_.begin() + 7 + kGo1ActionDim);
    for (int i = 0; i < kGo1ActionDim; ++i) {
      const mjtNum low = model_->jnt_range[(i + 1) * 2];
      const mjtNum high = model_->jnt_range[(i + 1) * 2 + 1];
      const mjtNum center = (low + high) * 0.5;
      const mjtNum range = high - low;
      soft_lowers_[i] =
          center - 0.5 * range * spec.config["soft_joint_pos_limit_factor"_];
      soft_uppers_[i] =
          center + 0.5 * range * spec.config["soft_joint_pos_limit_factor"_];
    }
    imu_site_id_ = RequireId(mjOBJ_SITE, "imu");
    gyro_adr_ = SensorAdr("gyro");
    local_linvel_adr_ = SensorAdr("local_linvel");
    accelerometer_adr_ = SensorAdr("accelerometer");
    global_angvel_adr_ = SensorAdr("global_angvel");
  }

  static std::string Go1GetupXmlPath(const std::string& base_path) {
    return base_path +
           "/mujoco/playground/assets/mujoco_playground/_src/locomotion/go1/"
           "xmls/scene_mjx_fullcollisions_flat_terrain.xml";
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    has_cached_render_ = false;
    done_ = false;
    terminated_ = false;
    elapsed_step_ = 0;
    ResetRewards();
    std::fill(last_act_.begin(), last_act_.end(), 0.0);
    std::fill(last_last_act_.begin(), last_last_act_.end(), 0.0);

    mj_resetData(model_, data_);
    const bool drop =
        unit_uniform_(gen_) < spec_.config["drop_from_height_prob"_];
    if (drop) {
      std::fill(data_->qpos, data_->qpos + model_->nq, 0.0);
      data_->qpos[2] = 0.5;
      // NOLINTNEXTLINE(modernize-avoid-c-arrays)
      mjtNum quat[4] = {normal_(gen_), normal_(gen_), normal_(gen_),
                        normal_(gen_)};
      const mjtNum norm = std::sqrt(quat[0] * quat[0] + quat[1] * quat[1] +
                                    quat[2] * quat[2] + quat[3] * quat[3]) +
                          1e-6;
      for (int i = 0; i < 4; ++i) {
        data_->qpos[3 + i] = quat[i] / norm;
      }
      for (int i = 0; i < kGo1ActionDim; ++i) {
        std::uniform_real_distribution<mjtNum> joint_dist(
            model_->jnt_range[(i + 1) * 2], model_->jnt_range[(i + 1) * 2 + 1]);
        data_->qpos[7 + i] = joint_dist(gen_);
      }
    } else {
      std::copy(init_qpos_.begin(), init_qpos_.end(), data_->qpos);
    }
    std::fill(data_->qvel, data_->qvel + model_->nv, 0.0);
    for (int i = 0; i < 6; ++i) {
      data_->qvel[i] = half_uniform_(gen_);
    }
    for (int i = 0; i < kGo1ActionDim; ++i) {
      data_->ctrl[i] = data_->qpos[7 + i];
    }
    mj_forward(model_, data_);
    for (int i = 0; i < settle_steps_; ++i) {
      mj_step(model_, data_);
    }
    mj_forward(model_, data_);
    data_->time = 0.0;
    reset_qpos_.assign(data_->qpos, data_->qpos + model_->nq);
    reset_qvel_.assign(data_->qvel, data_->qvel + model_->nv);
    reset_ctrl_.assign(data_->ctrl, data_->ctrl + model_->nu);
    WriteState(0.0, true);
  }

  void Step(const Action& action) override {
    has_cached_render_ = false;
    const auto* act = static_cast<const mjtNum*>(action["action"_].Data());
    for (int i = 0; i < kGo1ActionDim; ++i) {
      data_->ctrl[i] =
          data_->qpos[7 + i] + act[i] * spec_.config["action_scale"_];
    }
    for (int i = 0; i < n_substeps_; ++i) {
      mj_step(model_, data_);
    }
    mj_forward(model_, data_);

    mjtNum energy = 0.0;
    for (int i = 0; i < kGo1ActionDim; ++i) {
      energy += std::abs(data_->actuator_force[i] * data_->qvel[6 + i]);
    }
    terminated_ = energy > spec_.config["energy_termination_threshold"_];
    ComputeRewards(act);
    mjtNum reward =
        (reward_orientation_ + reward_torso_height_ + reward_posture_ +
         reward_stand_still_ + reward_action_rate_ + reward_dof_pos_limits_ +
         reward_torques_ + reward_dof_acc_ + reward_dof_vel_) *
        Dt();
    reward = std::clamp(reward, static_cast<mjtNum>(0.0),
                        static_cast<mjtNum>(10000.0));

    ++elapsed_step_;
    done_ = terminated_ || elapsed_step_ >= max_episode_steps_;
    WriteState(static_cast<float>(reward), false);

    for (int i = 0; i < kGo1ActionDim; ++i) {
      last_last_act_[i] = last_act_[i];
      last_act_[i] = act[i];
    }
  }

 private:
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

  mjtNum Dt() const { return spec_.config["ctrl_dt"_]; }

  mjtNum MaybeNoisy(mjtNum value, mjtNum scale) {
    return value + noise_uniform_(gen_) * spec_.config["noise_level"_] * scale;
  }

  void Sensor3(int adr, mjtNum* out) const {
    out[0] = data_->sensordata[adr];
    out[1] = data_->sensordata[adr + 1];
    out[2] = data_->sensordata[adr + 2];
  }

  void Gravity(mjtNum* out) const {
    const mjtNum* mat = data_->site_xmat + imu_site_id_ * 9;
    out[0] = -mat[6];
    out[1] = -mat[7];
    out[2] = -mat[8];
  }

  void ResetRewards() {
    reward_orientation_ = 0.0;
    reward_torso_height_ = 0.0;
    reward_posture_ = 0.0;
    reward_stand_still_ = 0.0;
    reward_action_rate_ = 0.0;
    reward_dof_pos_limits_ = 0.0;
    reward_torques_ = 0.0;
    reward_dof_acc_ = 0.0;
    reward_dof_vel_ = 0.0;
  }

  void ComputeRewards(const mjtNum* action) {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum gravity[3];
    Gravity(gravity);
    const mjtNum z_des = 0.275;
    const mjtNum torso_height = data_->site_xpos[imu_site_id_ * 3 + 2];
    const mjtNum orientation_error = gravity[0] * gravity[0] +
                                     gravity[1] * gravity[1] +
                                     (-1.0 - gravity[2]) * (-1.0 - gravity[2]);
    const bool is_upright = orientation_error < 0.01;
    const mjtNum height = std::min(torso_height, z_des);
    const bool is_at_desired_height = (z_des - height) < 0.005;
    const bool gate = is_upright && is_at_desired_height;

    reward_orientation_ =
        std::exp(-2.0 * orientation_error) * spec_.config["orientation_scale"_];
    reward_torso_height_ =
        (std::exp(height) - 1.0) * spec_.config["torso_height_scale"_];

    mjtNum posture_cost = 0.0;
    mjtNum stand_still_cost = 0.0;
    mjtNum action_rate = 0.0;
    mjtNum joint_limit = 0.0;
    mjtNum torque_l2 = 0.0;
    mjtNum torque_l1 = 0.0;
    mjtNum dof_acc = 0.0;
    mjtNum dof_vel = 0.0;
    const mjtNum max_velocity = 2.0 * M_PI;
    for (int i = 0; i < kGo1ActionDim; ++i) {
      const mjtNum q = data_->qpos[7 + i];
      const mjtNum qvel = data_->qvel[6 + i];
      const mjtNum torque = data_->actuator_force[i];
      posture_cost += (q - default_pose_[i]) * (q - default_pose_[i]);
      stand_still_cost += action[i] * action[i];
      const mjtNum da = action[i] - last_act_[i];
      const mjtNum dda = action[i] - 2.0 * last_act_[i] + last_last_act_[i];
      action_rate += da * da + dda * dda;
      joint_limit += -std::min(q - soft_lowers_[i], static_cast<mjtNum>(0.0));
      joint_limit += std::max(q - soft_uppers_[i], static_cast<mjtNum>(0.0));
      torque_l2 += torque * torque;
      torque_l1 += std::abs(torque);
      dof_acc += data_->qacc[6 + i] * data_->qacc[6 + i];
      const mjtNum vel_cost =
          std::max(std::abs(qvel) - max_velocity, static_cast<mjtNum>(0.0));
      dof_vel += vel_cost * vel_cost;
    }

    reward_posture_ = (is_upright ? 1.0 : 0.0) * std::exp(-0.5 * posture_cost) *
                      spec_.config["posture_scale"_];
    reward_stand_still_ = (gate ? 1.0 : 0.0) *
                          std::exp(-0.5 * stand_still_cost) *
                          spec_.config["stand_still_scale"_];
    reward_action_rate_ = action_rate * spec_.config["action_rate_scale"_];
    reward_dof_pos_limits_ =
        joint_limit * spec_.config["dof_pos_limits_scale"_];
    reward_torques_ =
        (std::sqrt(torque_l2) + torque_l1) * spec_.config["torques_scale"_];
    reward_dof_acc_ = dof_acc * spec_.config["dof_acc_scale"_];
    reward_dof_vel_ = dof_vel * spec_.config["dof_vel_scale"_];
  }

  void FillObs(mjtNum* state, mjtNum* privileged_state) {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum gyro[3];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum gravity[3];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum local_linvel[3];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum accelerometer[3];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum global_angvel[3];
    Sensor3(gyro_adr_, gyro);
    Gravity(gravity);
    Sensor3(local_linvel_adr_, local_linvel);
    Sensor3(accelerometer_adr_, accelerometer);
    Sensor3(global_angvel_adr_, global_angvel);

    int n = 0;
    for (mjtNum value : gyro) {
      state[n++] = MaybeNoisy(value, spec_.config["noise_gyro"_]);
    }
    for (mjtNum value : gravity) {
      state[n++] = MaybeNoisy(value, spec_.config["noise_gravity"_]);
    }
    for (int i = 0; i < kGo1ActionDim; ++i) {
      state[n++] =
          MaybeNoisy(data_->qpos[7 + i], spec_.config["noise_joint_pos"_]) -
          default_pose_[i];
    }
    for (int i = 0; i < kGo1ActionDim; ++i) {
      state[n++] =
          MaybeNoisy(data_->qvel[6 + i], spec_.config["noise_joint_vel"_]);
    }
    for (int i = 0; i < kGo1ActionDim; ++i) {
      state[n++] = last_act_[i];
    }

    int p = 0;
    for (int i = 0; i < kGo1GetupStateDim; ++i) {
      privileged_state[p++] = state[i];
    }
    for (mjtNum value : gyro) {
      privileged_state[p++] = value;
    }
    for (mjtNum value : accelerometer) {
      privileged_state[p++] = value;
    }
    for (mjtNum value : local_linvel) {
      privileged_state[p++] = value;
    }
    for (mjtNum value : global_angvel) {
      privileged_state[p++] = value;
    }
    for (int i = 0; i < kGo1ActionDim; ++i) {
      privileged_state[p++] = data_->qpos[7 + i];
    }
    for (int i = 0; i < kGo1ActionDim; ++i) {
      privileged_state[p++] = data_->qvel[6 + i];
    }
    for (int i = 0; i < kGo1ActionDim; ++i) {
      privileged_state[p++] = data_->actuator_force[i];
    }
    privileged_state[p++] = data_->site_xpos[imu_site_id_ * 3 + 2];
  }

  void WriteState(float reward, bool reset) {
    auto state = Allocate();
    state["reward"_] = reward;
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      auto obs_state = state["obs:state"_];
      auto obs_privileged = state["obs:privileged_state"_];
      mjtNum* obs = PrepareObservation("obs:state", &obs_state);
      mjtNum* priv =
          PrepareObservation("obs:privileged_state", &obs_privileged);
      FillObs(obs, priv);
      CommitObservation("obs:state", &obs_state, reset);
      CommitObservation("obs:privileged_state", &obs_privileged, reset);
    }
    state["info:terminated"_] = terminated_;
    state["info:reward_orientation"_] = reward_orientation_;
    state["info:reward_torso_height"_] = reward_torso_height_;
    state["info:reward_posture"_] = reward_posture_;
    state["info:reward_stand_still"_] = reward_stand_still_;
    state["info:reward_action_rate"_] = reward_action_rate_;
    state["info:reward_dof_pos_limits"_] = reward_dof_pos_limits_;
    state["info:reward_torques"_] = reward_torques_;
    state["info:reward_dof_acc"_] = reward_dof_acc_;
    state["info:reward_dof_vel"_] = reward_dof_vel_;
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
    std::copy(data_->qacc, data_->qacc + model_->nv, pad.begin());
    state["info:qacc"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(data_->qacc_warmstart, data_->qacc_warmstart + model_->nv,
              pad.begin());
    state["info:qacc_warmstart"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(reset_ctrl_.begin(), reset_ctrl_.end(), pad.begin());
    state["info:ctrl0"_].Assign(pad.data(), pad.size());
    state["info:last_act"_].Assign(last_act_.data(), last_act_.size());
    state["info:last_last_act"_].Assign(last_last_act_.data(),
                                        last_last_act_.size());
    std::array<mjtNum, 256> sensor_pad{};
    std::copy(data_->sensordata, data_->sensordata + model_->nsensordata,
              sensor_pad.begin());
    state["info:sensordata"_].Assign(sensor_pad.data(), sensor_pad.size());
#endif
  }
};

using PlaygroundGo1GetupEnv =
    PlaygroundGo1GetupEnvBase<PlaygroundGo1GetupEnvSpec, false>;
using PlaygroundGo1GetupPixelEnv =
    PlaygroundGo1GetupEnvBase<PlaygroundGo1GetupPixelEnvSpec, true>;
using PlaygroundGo1GetupEnvPool = AsyncEnvPool<PlaygroundGo1GetupEnv>;
using PlaygroundGo1GetupPixelEnvPool = AsyncEnvPool<PlaygroundGo1GetupPixelEnv>;

class PlaygroundGo1HandstandEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "frame_stack"_.Bind(1), "task_name"_.Bind(std::string("Go1Handstand")),
        "ctrl_dt"_.Bind(0.02), "sim_dt"_.Bind(0.004), "kp"_.Bind(35.0),
        "kd"_.Bind(0.5), "action_scale"_.Bind(0.3),
        "soft_joint_pos_limit_factor"_.Bind(0.9), "init_from_crouch"_.Bind(0.0),
        "energy_termination_threshold"_.Bind(inf), "noise_level"_.Bind(1.0),
        "noise_joint_pos"_.Bind(0.01), "noise_joint_vel"_.Bind(1.5),
        "noise_gyro"_.Bind(0.2), "noise_gravity"_.Bind(0.05),
        "noise_linvel"_.Bind(0.1), "height_scale"_.Bind(1.0),
        "orientation_scale"_.Bind(1.0), "contact_scale"_.Bind(-0.1),
        "action_rate_scale"_.Bind(0.0), "termination_scale"_.Bind(0.0),
        "dof_pos_limits_scale"_.Bind(-0.5), "torques_scale"_.Bind(0.0),
        "pose_scale"_.Bind(-0.1), "stay_still_scale"_.Bind(0.0),
        "energy_scale"_.Bind(0.0), "dof_acc_scale"_.Bind(0.0));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict(
        "obs:state"_.Bind(
            StackSpec(Spec<mjtNum>({kGo1HandstandStateDim}, {-inf, inf}),
                      conf["frame_stack"_])),
        "obs:privileged_state"_.Bind(StackSpec(
            Spec<mjtNum>({kGo1HandstandPrivilegedStateDim}, {-inf, inf}),
            conf["frame_stack"_])),
        "info:terminated"_.Bind(Spec<bool>({-1})),
        "info:reward_height"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_orientation"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_contact"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_action_rate"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_termination"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_dof_pos_limits"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_torques"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_pose"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_stay_still"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_energy"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_dof_acc"_.Bind(Spec<mjtNum>({-1}))
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
        "info:last_act"_.Bind(Spec<mjtNum>({kGo1ActionDim})),
        "info:sensordata"_.Bind(Spec<mjtNum>({256}))
#endif
    );  // NOLINT
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<mjtNum>({-1, kGo1ActionDim}, {-1.0, 1.0})));
  }
};

using PlaygroundGo1HandstandEnvSpec = EnvSpec<PlaygroundGo1HandstandEnvFns>;
using PlaygroundGo1HandstandPixelEnvFns =
    PixelObservationEnvFns<PlaygroundGo1HandstandEnvFns>;
using PlaygroundGo1HandstandPixelEnvSpec =
    EnvSpec<PlaygroundGo1HandstandPixelEnvFns>;

template <typename EnvSpecT, bool kFromPixels>
class PlaygroundGo1HandstandEnvBase : public Env<EnvSpecT>,
                                      public PlaygroundMujocoEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  int n_substeps_{5};
  std::vector<mjtNum> init_qpos_;
  std::vector<mjtNum> crouch_qpos_;
  std::vector<mjtNum> reset_qpos_;
  std::vector<mjtNum> reset_qvel_;
  std::vector<mjtNum> reset_ctrl_;
  std::vector<mjtNum> default_pose_;
  std::array<mjtNum, kGo1ActionDim> last_act_{};
  std::array<mjtNum, kGo1ActionDim> soft_lowers_{};
  std::array<mjtNum, kGo1ActionDim> soft_uppers_{};
  std::array<int, kGo1ActionDim> joint_ids_{};
  std::array<mjtNum, kGo1ActionDim> joint_pose_{};
  int joint_pose_count_{6};
  std::array<int, kGo1Feet> feet_floor_sensor_adrs_{};
  std::array<int, kGo1HandstandContactGeoms>
      full_collision_floor_sensor_adrs_{};
  std::array<mjtNum, 3> desired_forward_vec_{0.0, 0.0, -1.0};
  int imu_site_id_{-1};
  int gyro_adr_{-1};
  int local_linvel_adr_{-1};
  int accelerometer_adr_{-1};
  int upvector_adr_{-1};
  int global_angvel_adr_{-1};
  mjtNum z_des_{0.55};
  std::uniform_real_distribution<mjtNum> unit_uniform_{0.0, 1.0};
  std::uniform_real_distribution<mjtNum> half_uniform_{-0.5, 0.5};
  std::uniform_real_distribution<mjtNum> noise_uniform_{-1.0, 1.0};
  mjtNum reward_height_{0.0};
  mjtNum reward_orientation_{0.0};
  mjtNum reward_contact_{0.0};
  mjtNum reward_action_rate_{0.0};
  mjtNum reward_termination_{0.0};
  mjtNum reward_dof_pos_limits_{0.0};
  mjtNum reward_torques_{0.0};
  mjtNum reward_pose_{0.0};
  mjtNum reward_stay_still_{0.0};
  mjtNum reward_energy_{0.0};
  mjtNum reward_dof_acc_{0.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  PlaygroundGo1HandstandEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        PlaygroundMujocoEnv(Go1HandstandXmlPath(spec.config["base_path"_]),
                            spec.config["max_episode_steps"_],
                            spec.config["frame_stack"_],
                            RenderWidthOrDefault<kFromPixels>(spec.config),
                            RenderHeightOrDefault<kFromPixels>(spec.config),
                            RenderCameraIdOrDefault<kFromPixels>(spec.config)) {
    (void)env_id;
    const std::string task_name = spec.config["task_name"_];
    if (task_name != "Go1Handstand" && task_name != "Go1Footstand") {
      throw std::runtime_error(
          "Unsupported playground Go1 handstand task_name " + task_name);
    }
    if (model_->nq < 7 + kGo1ActionDim || model_->nv < 6 + kGo1ActionDim ||
        model_->nu != kGo1ActionDim) {
      throw std::runtime_error("Unexpected Go1 handstand model dimensions.");
    }
    model_->opt.timestep = spec.config["sim_dt"_];
    model_->opt.ccd_iterations = 20;
    for (int i = 6; i < model_->nv; ++i) {
      model_->dof_damping[i] = spec.config["kd"_];
    }
    for (int i = 0; i < model_->nu; ++i) {
      model_->actuator_gainprm[i * mjNGAIN + 0] = spec.config["kp"_];
      model_->actuator_biasprm[i * mjNBIAS + 1] = -spec.config["kp"_];
    }
    n_substeps_ = static_cast<int>(
        std::llround(spec.config["ctrl_dt"_] / spec.config["sim_dt"_]));

    const int home_key = RequireId(mjOBJ_KEY, "home");
    const int crouch_key = RequireId(mjOBJ_KEY, "pre_recovery");
    init_qpos_.assign(model_->key_qpos + home_key * model_->nq,
                      model_->key_qpos + (home_key + 1) * model_->nq);
    crouch_qpos_.assign(model_->key_qpos + crouch_key * model_->nq,
                        model_->key_qpos + (crouch_key + 1) * model_->nq);
    default_pose_.assign(init_qpos_.begin() + 7,
                         init_qpos_.begin() + 7 + kGo1ActionDim);
    for (int i = 0; i < kGo1ActionDim; ++i) {
      const mjtNum low = model_->jnt_range[(i + 1) * 2];
      const mjtNum high = model_->jnt_range[(i + 1) * 2 + 1];
      const mjtNum center = (low + high) * 0.5;
      const mjtNum range = high - low;
      soft_lowers_[i] =
          center - 0.5 * range * spec.config["soft_joint_pos_limit_factor"_];
      soft_uppers_[i] =
          center + 0.5 * range * spec.config["soft_joint_pos_limit_factor"_];
    }
    ConfigureTask(task_name);

    imu_site_id_ = RequireId(mjOBJ_SITE, "imu");
    gyro_adr_ = SensorAdr("gyro");
    local_linvel_adr_ = SensorAdr("local_linvel");
    accelerometer_adr_ = SensorAdr("accelerometer");
    upvector_adr_ = SensorAdr("upvector");
    global_angvel_adr_ = SensorAdr("global_angvel");
    for (int i = 0; i < kGo1Feet; ++i) {
      feet_floor_sensor_adrs_[i] =
          SensorAdr(std::string(kGo1FeetSites[i]) + "_floor_found");
    }
  }

  static std::string Go1HandstandXmlPath(const std::string& base_path) {
    return base_path +
           "/mujoco/playground/assets/mujoco_playground/_src/locomotion/go1/"
           "xmls/scene_mjx_flat_terrain.xml";
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    has_cached_render_ = false;
    done_ = false;
    terminated_ = false;
    elapsed_step_ = 0;
    ResetRewards();
    std::fill(last_act_.begin(), last_act_.end(), 0.0);

    mj_resetData(model_, data_);
    const bool from_crouch =
        unit_uniform_(gen_) < spec_.config["init_from_crouch"_];
    const auto& qpos_source = from_crouch ? crouch_qpos_ : init_qpos_;
    std::copy(qpos_source.begin(), qpos_source.end(), data_->qpos);
    data_->qpos[0] += half_uniform_(gen_);
    data_->qpos[1] += half_uniform_(gen_);
    std::uniform_real_distribution<mjtNum> yaw_dist(-3.14, 3.14);
    const mjtNum yaw = yaw_dist(gen_);
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const mjtNum yaw_quat[4] = {std::cos(yaw / 2.0), 0.0, 0.0,
                                std::sin(yaw / 2.0)};
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum new_quat[4];
    mju_mulQuat(new_quat, data_->qpos + 3, yaw_quat);
    std::memcpy(data_->qpos + 3, new_quat, sizeof(mjtNum) * 4);

    std::fill(data_->qvel, data_->qvel + model_->nv, 0.0);
    if (!from_crouch) {
      for (int i = 0; i < 6; ++i) {
        data_->qvel[i] = half_uniform_(gen_);
      }
    }
    for (int i = 0; i < kGo1ActionDim; ++i) {
      data_->ctrl[i] = data_->qpos[7 + i];
    }
    mj_forward(model_, data_);
    reset_qpos_.assign(data_->qpos, data_->qpos + model_->nq);
    reset_qvel_.assign(data_->qvel, data_->qvel + model_->nv);
    reset_ctrl_.assign(data_->ctrl, data_->ctrl + model_->nu);
    WriteState(0.0, true);
  }

  void Step(const Action& action) override {
    has_cached_render_ = false;
    const auto* act = static_cast<const mjtNum*>(action["action"_].Data());
    for (int i = 0; i < kGo1ActionDim; ++i) {
      data_->ctrl[i] += act[i] * spec_.config["action_scale"_];
    }
    for (int i = 0; i < n_substeps_; ++i) {
      mj_step(model_, data_);
    }
    mj_forward(model_, data_);

    mjtNum energy = 0.0;
    for (int i = 0; i < kGo1ActionDim; ++i) {
      energy += std::abs(data_->actuator_force[i] * data_->qvel[6 + i]);
    }
    terminated_ = data_->sensordata[upvector_adr_ + 2] < -0.25 ||
                  HasFullCollisionFloorContact() ||
                  energy > spec_.config["energy_termination_threshold"_];
    ComputeRewards(act);
    mjtNum reward = (reward_height_ + reward_orientation_ + reward_contact_ +
                     reward_action_rate_ + reward_termination_ +
                     reward_dof_pos_limits_ + reward_torques_ + reward_pose_ +
                     reward_stay_still_ + reward_energy_ + reward_dof_acc_) *
                    Dt();
    reward = std::clamp(reward, static_cast<mjtNum>(0.0),
                        static_cast<mjtNum>(10000.0));

    ++elapsed_step_;
    done_ = terminated_ || elapsed_step_ >= max_episode_steps_;
    WriteState(static_cast<float>(reward), false);

    for (int i = 0; i < kGo1ActionDim; ++i) {
      last_act_[i] = act[i];
    }
  }

 private:
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

  mjtNum Dt() const { return spec_.config["ctrl_dt"_]; }

  mjtNum MaybeNoisy(mjtNum value, mjtNum scale) {
    return value + noise_uniform_(gen_) * spec_.config["noise_level"_] * scale;
  }

  void Sensor3(int adr, mjtNum* out) const {
    out[0] = data_->sensordata[adr];
    out[1] = data_->sensordata[adr + 1];
    out[2] = data_->sensordata[adr + 2];
  }

  void Gravity(mjtNum* out) const {
    const mjtNum* mat = data_->site_xmat + imu_site_id_ * 9;
    out[0] = -mat[6];
    out[1] = -mat[7];
    out[2] = -mat[8];
  }

  void ForwardVector(mjtNum* out) const {
    const mjtNum* mat = data_->site_xmat + imu_site_id_ * 9;
    out[0] = mat[0];
    out[1] = mat[3];
    out[2] = mat[6];
  }

  void ConfigureTask(const std::string& task_name) {
    if (task_name == "Go1Footstand") {
      // NOLINTNEXTLINE(modernize-avoid-c-arrays)
      const int ids[6] = {0, 1, 2, 3, 4, 5};
      std::copy(ids, ids + 6, joint_ids_.begin());
      desired_forward_vec_ = {0.0, 0.0, 1.0};
      z_des_ = 0.53;
    } else {
      // NOLINTNEXTLINE(modernize-avoid-c-arrays)
      const int ids[6] = {6, 7, 8, 9, 10, 11};
      std::copy(ids, ids + 6, joint_ids_.begin());
      desired_forward_vec_ = {0.0, 0.0, -1.0};
      z_des_ = 0.55;
    }
    // The pinned v0.2.0 Footstand subclass keeps Handstand's front
    // full-collision floor sensors for termination, even though it updates
    // separate rear geom metadata that is not used by step().
    const std::array<std::string, kGo1HandstandContactGeoms> geom_names = {
        "fl_calf1",  "fl_calf2",  "fr_calf1",  "fr_calf2",
        "fl_thigh1", "fl_thigh2", "fl_thigh3", "fr_thigh1",
        "fr_thigh2", "fr_thigh3", "fl_hip",    "fr_hip"};
    joint_pose_count_ = 6;
    for (int i = 0; i < joint_pose_count_; ++i) {
      joint_pose_[i] = default_pose_[joint_ids_[i]];
    }
    for (int i = 0; i < kGo1HandstandContactGeoms; ++i) {
      full_collision_floor_sensor_adrs_[i] =
          SensorAdr(geom_names[i] + "_floor_found");
    }
  }

  bool HasFullCollisionFloorContact() const {
    return std::any_of(
        full_collision_floor_sensor_adrs_.begin(),
        full_collision_floor_sensor_adrs_.end(),
        [this](int adr) { return data_->sensordata[adr] > 0.0; });
  }

  bool HasFootContact() const {
    return std::any_of(
        feet_floor_sensor_adrs_.begin(), feet_floor_sensor_adrs_.end(),
        [this](int adr) { return data_->sensordata[adr] > 0.0; });
  }

  void ResetRewards() {
    reward_height_ = 0.0;
    reward_orientation_ = 0.0;
    reward_contact_ = 0.0;
    reward_action_rate_ = 0.0;
    reward_termination_ = 0.0;
    reward_dof_pos_limits_ = 0.0;
    reward_torques_ = 0.0;
    reward_pose_ = 0.0;
    reward_stay_still_ = 0.0;
    reward_energy_ = 0.0;
    reward_dof_acc_ = 0.0;
  }

  void ComputeRewards(const mjtNum* action) {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum forward[3];
    ForwardVector(forward);
    const mjtNum torso_height = data_->site_xpos[imu_site_id_ * 3 + 2];
    const mjtNum height = std::min(torso_height, z_des_);
    const mjtNum height_error = z_des_ - height;
    const mjtNum cos_dist = forward[0] * desired_forward_vec_[0] +
                            forward[1] * desired_forward_vec_[1] +
                            forward[2] * desired_forward_vec_[2];
    const mjtNum normalized = 0.5 * cos_dist + 0.5;
    reward_height_ = std::exp(-height_error) * spec_.config["height_scale"_];
    reward_orientation_ =
        normalized * normalized * spec_.config["orientation_scale"_];
    reward_contact_ =
        (HasFootContact() ? 1.0 : 0.0) * spec_.config["contact_scale"_];
    reward_termination_ =
        (terminated_ ? 1.0 : 0.0) * spec_.config["termination_scale"_];

    mjtNum action_rate = 0.0;
    mjtNum joint_limit = 0.0;
    mjtNum torque_cost = 0.0;
    mjtNum pose_cost = 0.0;
    mjtNum energy = 0.0;
    mjtNum dof_acc = 0.0;
    for (int i = 0; i < kGo1ActionDim; ++i) {
      const mjtNum q = data_->qpos[7 + i];
      const mjtNum qvel = data_->qvel[6 + i];
      const mjtNum torque = data_->actuator_force[i];
      const mjtNum da = action[i] - last_act_[i];
      action_rate += da * da;
      joint_limit += -std::min(q - soft_lowers_[i], static_cast<mjtNum>(0.0));
      joint_limit += std::max(q - soft_uppers_[i], static_cast<mjtNum>(0.0));
      torque_cost += torque * torque;
      energy += std::abs(qvel * torque);
      dof_acc += data_->qacc[6 + i] * data_->qacc[6 + i];
    }
    for (int i = 0; i < joint_pose_count_; ++i) {
      const int joint_id = joint_ids_[i];
      const mjtNum pose_err = data_->qpos[7 + joint_id] - joint_pose_[i];
      pose_cost += pose_err * pose_err;
    }
    const mjtNum stay_still = data_->qvel[0] * data_->qvel[0] +
                              data_->qvel[1] * data_->qvel[1] +
                              data_->qvel[5] * data_->qvel[5];
    reward_action_rate_ = action_rate * spec_.config["action_rate_scale"_];
    reward_dof_pos_limits_ =
        joint_limit * spec_.config["dof_pos_limits_scale"_];
    reward_torques_ = torque_cost * spec_.config["torques_scale"_];
    reward_pose_ = pose_cost * spec_.config["pose_scale"_];
    reward_stay_still_ = stay_still * spec_.config["stay_still_scale"_];
    reward_energy_ = energy * spec_.config["energy_scale"_];
    reward_dof_acc_ = dof_acc * spec_.config["dof_acc_scale"_];
  }

  void FillObs(mjtNum* state, mjtNum* privileged_state) {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum gyro[3];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum gravity[3];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum local_linvel[3];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum accelerometer[3];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum global_angvel[3];
    Sensor3(gyro_adr_, gyro);
    Gravity(gravity);
    Sensor3(local_linvel_adr_, local_linvel);
    Sensor3(accelerometer_adr_, accelerometer);
    Sensor3(global_angvel_adr_, global_angvel);

    int n = 0;
    for (mjtNum value : local_linvel) {
      state[n++] = MaybeNoisy(value, spec_.config["noise_linvel"_]);
    }
    for (mjtNum value : gyro) {
      state[n++] = MaybeNoisy(value, spec_.config["noise_gyro"_]);
    }
    for (mjtNum value : gravity) {
      state[n++] = MaybeNoisy(value, spec_.config["noise_gravity"_]);
    }
    for (int i = 0; i < kGo1ActionDim; ++i) {
      state[n++] =
          MaybeNoisy(data_->qpos[7 + i], spec_.config["noise_joint_pos"_]) -
          default_pose_[i];
    }
    for (int i = 0; i < kGo1ActionDim; ++i) {
      state[n++] =
          MaybeNoisy(data_->qvel[6 + i], spec_.config["noise_joint_vel"_]);
    }
    for (int i = 0; i < kGo1ActionDim; ++i) {
      state[n++] = last_act_[i];
    }

    int p = 0;
    for (int i = 0; i < kGo1HandstandStateDim; ++i) {
      privileged_state[p++] = state[i];
    }
    for (mjtNum value : gyro) {
      privileged_state[p++] = value;
    }
    for (mjtNum value : accelerometer) {
      privileged_state[p++] = value;
    }
    for (mjtNum value : local_linvel) {
      privileged_state[p++] = value;
    }
    for (mjtNum value : global_angvel) {
      privileged_state[p++] = value;
    }
    for (int i = 0; i < kGo1ActionDim; ++i) {
      privileged_state[p++] = data_->qpos[7 + i];
    }
    for (int i = 0; i < kGo1ActionDim; ++i) {
      privileged_state[p++] = data_->qvel[6 + i];
    }
    for (int i = 0; i < kGo1ActionDim; ++i) {
      privileged_state[p++] = data_->actuator_force[i];
    }
    privileged_state[p++] = data_->site_xpos[imu_site_id_ * 3 + 2];
  }

  void WriteState(float reward, bool reset) {
    auto state = Allocate();
    state["reward"_] = reward;
    if constexpr (kFromPixels) {
      auto obs_pixels = state["obs:pixels"_];
      AssignPixelObservation("obs:pixels", &obs_pixels, reset);
    } else {
      auto obs_state = state["obs:state"_];
      auto obs_privileged = state["obs:privileged_state"_];
      mjtNum* obs = PrepareObservation("obs:state", &obs_state);
      mjtNum* priv =
          PrepareObservation("obs:privileged_state", &obs_privileged);
      FillObs(obs, priv);
      CommitObservation("obs:state", &obs_state, reset);
      CommitObservation("obs:privileged_state", &obs_privileged, reset);
    }
    state["info:terminated"_] = terminated_;
    state["info:reward_height"_] = reward_height_;
    state["info:reward_orientation"_] = reward_orientation_;
    state["info:reward_contact"_] = reward_contact_;
    state["info:reward_action_rate"_] = reward_action_rate_;
    state["info:reward_termination"_] = reward_termination_;
    state["info:reward_dof_pos_limits"_] = reward_dof_pos_limits_;
    state["info:reward_torques"_] = reward_torques_;
    state["info:reward_pose"_] = reward_pose_;
    state["info:reward_stay_still"_] = reward_stay_still_;
    state["info:reward_energy"_] = reward_energy_;
    state["info:reward_dof_acc"_] = reward_dof_acc_;
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
    std::copy(data_->qacc, data_->qacc + model_->nv, pad.begin());
    state["info:qacc"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(data_->qacc_warmstart, data_->qacc_warmstart + model_->nv,
              pad.begin());
    state["info:qacc_warmstart"_].Assign(pad.data(), pad.size());
    pad.fill(0.0);
    std::copy(reset_ctrl_.begin(), reset_ctrl_.end(), pad.begin());
    state["info:ctrl0"_].Assign(pad.data(), pad.size());
    state["info:last_act"_].Assign(last_act_.data(), last_act_.size());
    std::array<mjtNum, 256> sensor_pad{};
    std::copy(data_->sensordata, data_->sensordata + model_->nsensordata,
              sensor_pad.begin());
    state["info:sensordata"_].Assign(sensor_pad.data(), sensor_pad.size());
#endif
  }
};

using PlaygroundGo1HandstandEnv =
    PlaygroundGo1HandstandEnvBase<PlaygroundGo1HandstandEnvSpec, false>;
using PlaygroundGo1HandstandPixelEnv =
    PlaygroundGo1HandstandEnvBase<PlaygroundGo1HandstandPixelEnvSpec, true>;
using PlaygroundGo1HandstandEnvPool = AsyncEnvPool<PlaygroundGo1HandstandEnv>;
using PlaygroundGo1HandstandPixelEnvPool =
    AsyncEnvPool<PlaygroundGo1HandstandPixelEnv>;

}  // namespace mujoco_playground

#endif  // ENVPOOL_MUJOCO_PLAYGROUND_GO1_H_
