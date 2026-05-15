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

#ifndef ENVPOOL_MUJOCO_PLAYGROUND_APOLLO_H_
#define ENVPOOL_MUJOCO_PLAYGROUND_APOLLO_H_

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

constexpr int kApolloActionDim = 32;
constexpr int kApolloStateDim = 112;
constexpr int kApolloPrivilegedStateDim = 224;
constexpr int kApolloFeet = 2;
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
constexpr const char* kApolloFeetSites[kApolloFeet] = {"l_foot", "r_foot"};
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
constexpr const char* kApolloFeetFloorSensors[kApolloFeet] = {
    "collision_l_sole_floor_found",
    "collision_r_sole_floor_found",
};
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
constexpr const char* kApolloCollisionSensors[] = {
    "collision_l_hand_plate_collision_capsule_body_l_thigh_found",
    "collision_r_hand_plate_collision_capsule_body_r_thigh_found",
    "collision_l_sole_collision_r_sole_found",
    "collision_capsule_body_l_shin_collision_capsule_body_r_shin_found",
    "collision_capsule_body_l_thigh_collision_capsule_body_r_thigh_found",
};
constexpr int kApolloCollisionSensorCount = 5;
constexpr std::array<mjtNum, kApolloActionDim> kApolloPoseWeights = {
    5.0,  5.0,  5.0, 1.0, 1.0, 1.0, 1.0,  1.0,  0.1, 1.0, 1.0,
    1.0,  1.0,  1.0, 1.0, 0.1, 1.0, 1.0,  1.0,  1.0, 1.0, 1.0,
    0.01, 0.01, 1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 1.0, 1.0,
};

class PlaygroundApolloEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "frame_stack"_.Bind(1),
        "task_name"_.Bind(std::string("ApolloJoystickFlatTerrain")),
        "ctrl_dt"_.Bind(0.02), "sim_dt"_.Bind(0.005), "action_scale"_.Bind(0.5),
        "noise_level"_.Bind(1.0), "noise_joint_pos"_.Bind(0.03),
        "noise_joint_vel"_.Bind(1.5), "noise_gravity"_.Bind(0.05),
        "noise_linvel"_.Bind(0.1), "noise_gyro"_.Bind(0.2),
        "tracking_scale"_.Bind(1.0), "lin_vel_z_scale"_.Bind(0.0),
        "ang_vel_xy_scale"_.Bind(-0.15), "orientation_scale"_.Bind(-1.0),
        "torques_scale"_.Bind(0.0), "action_rate_scale"_.Bind(0.0),
        "energy_scale"_.Bind(0.0), "feet_phase_scale"_.Bind(1.0),
        "alive_scale"_.Bind(0.0), "termination_scale"_.Bind(0.0),
        "pose_scale"_.Bind(-1.0), "collision_scale"_.Bind(-1.0),
        "tracking_sigma"_.Bind(0.25), "max_foot_height"_.Bind(0.12),
        "push_enable"_.Bind(1.0), "push_interval_min"_.Bind(5.0),
        "push_interval_max"_.Bind(10.0), "push_magnitude_min"_.Bind(0.1),
        "push_magnitude_max"_.Bind(2.0), "command_min0"_.Bind(-1.5),
        "command_min1"_.Bind(-0.8), "command_min2"_.Bind(-1.5),
        "command_max0"_.Bind(1.5), "command_max1"_.Bind(0.8),
        "command_max2"_.Bind(1.5), "command_zero_prob0"_.Bind(0.9),
        "command_zero_prob1"_.Bind(0.25), "command_zero_prob2"_.Bind(0.5));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict("obs:state"_.Bind(
                        StackSpec(Spec<mjtNum>({kApolloStateDim}, {-inf, inf}),
                                  conf["frame_stack"_])),
                    "obs:privileged_state"_.Bind(StackSpec(
                        Spec<mjtNum>({kApolloPrivilegedStateDim}, {-inf, inf}),
                        conf["frame_stack"_])),
                    "info:terminated"_.Bind(Spec<bool>({-1})),
                    "info:command"_.Bind(Spec<mjtNum>({-1, 3})),
                    "info:steps_until_next_cmd"_.Bind(Spec<int>({-1})),
                    "info:reward_termination"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_alive"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_tracking"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_lin_vel_z"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_ang_vel_xy"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_orientation"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_feet_phase"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_torques"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_action_rate"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_energy"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_collision"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_pose"_.Bind(Spec<mjtNum>({-1}))
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
                    "info:sensordata"_.Bind(Spec<mjtNum>({256})),
                    "info:last_act"_.Bind(Spec<mjtNum>({kApolloActionDim})),
                    "info:phase"_.Bind(Spec<mjtNum>({2})),
                    "info:phase_dt"_.Bind(Spec<mjtNum>({1})),
                    "info:push"_.Bind(Spec<mjtNum>({2})),
                    "info:push_step"_.Bind(Spec<int>({-1})),
                    "info:push_interval_steps"_.Bind(Spec<int>({-1})),
                    "info:filtered_linvel"_.Bind(Spec<mjtNum>({3})),
                    "info:filtered_angvel"_.Bind(Spec<mjtNum>({3}))
#endif
    );  // NOLINT
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<mjtNum>({-1, kApolloActionDim}, {-1.0, 1.0})));
  }
};

using ApolloAliases = PlaygroundEnvAliases<PlaygroundApolloEnvFns>;
using PlaygroundApolloEnvSpec = ApolloAliases::Spec;
using PlaygroundApolloPixelEnvFns = ApolloAliases::PixelFns;
using PlaygroundApolloPixelEnvSpec = ApolloAliases::PixelSpec;

template <typename EnvSpecT, bool kFromPixels>
class PlaygroundApolloEnvBase : public Env<EnvSpecT>,
                                public PlaygroundMujocoEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  int n_substeps_{4};
  int step_{0};
  int steps_until_next_cmd_{0};
  int push_step_{0};
  int push_interval_steps_{0};
  std::vector<mjtNum> init_qpos_;
  std::vector<mjtNum> reset_qpos_;
  std::vector<mjtNum> reset_qvel_;
  std::vector<mjtNum> reset_ctrl_;
  std::array<mjtNum, kApolloActionDim> default_ctrl_{};
  std::array<mjtNum, kApolloActionDim> last_act_{};
  std::array<mjtNum, kApolloActionDim> obs_last_act_{};
  std::array<mjtNum, kApolloActionDim> actuator_torque_limits_{};
  std::array<mjtNum, 3> command_{};
  std::array<mjtNum, 2> phase_{};
  std::array<mjtNum, 2> obs_phase_{};
  std::array<mjtNum, 2> push_{};
  std::array<mjtNum, 3> filtered_linvel_{};
  std::array<mjtNum, 3> filtered_angvel_{};
  mjtNum phase_dt_{0.0};
  int imu_site_id_{-1};
  int torso_body_id_{-1};
  int gyro_adr_{-1};
  int local_linvel_adr_{-1};
  int upvector_adr_{-1};
  int global_linvel_adr_{-1};
  int global_angvel_adr_{-1};
  std::array<int, kApolloFeet> feet_site_ids_{};
  std::array<int, kApolloFeet> feet_floor_sensor_adrs_{};
  std::array<int, kApolloCollisionSensorCount> collision_sensor_adrs_{};
  std::exponential_distribution<mjtNum> exponential_{1.0};
  std::uniform_real_distribution<mjtNum> unit_uniform_{0.0, 1.0};
  std::uniform_real_distribution<mjtNum> noise_uniform_{-1.0, 1.0};
  mjtNum reward_termination_{0.0};
  mjtNum reward_alive_{0.0};
  mjtNum reward_tracking_{0.0};
  mjtNum reward_lin_vel_z_{0.0};
  mjtNum reward_ang_vel_xy_{0.0};
  mjtNum reward_orientation_{0.0};
  mjtNum reward_feet_phase_{0.0};
  mjtNum reward_torques_{0.0};
  mjtNum reward_action_rate_{0.0};
  mjtNum reward_energy_{0.0};
  mjtNum reward_collision_{0.0};
  mjtNum reward_pose_{0.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  PlaygroundApolloEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        PlaygroundMujocoEnv(ApolloXmlPath(spec.config["base_path"_]),
                            spec.config["max_episode_steps"_],
                            spec.config["frame_stack"_],
                            RenderWidthOrDefault<kFromPixels>(spec.config),
                            RenderHeightOrDefault<kFromPixels>(spec.config),
                            RenderCameraIdOrDefault<kFromPixels>(spec.config)) {
    (void)env_id;
    const std::string task_name = spec.config["task_name"_];
    if (task_name != "ApolloJoystickFlatTerrain") {
      throw std::runtime_error("Unsupported playground Apollo task_name " +
                               task_name);
    }
    if (model_->nq < 7 + kApolloActionDim ||
        model_->nv < 6 + kApolloActionDim || model_->nu != kApolloActionDim) {
      throw std::runtime_error("Unexpected Apollo model dimensions.");
    }
    model_->opt.timestep = spec.config["sim_dt"_];
    n_substeps_ = static_cast<int>(
        std::llround(spec.config["ctrl_dt"_] / spec.config["sim_dt"_]));

    const int key_id = mj_name2id(model_, mjOBJ_KEY, "knees_bent");
    if (key_id < 0) {
      throw std::runtime_error("Apollo model is missing knees_bent keyframe.");
    }
    init_qpos_.assign(model_->key_qpos + key_id * model_->nq,
                      model_->key_qpos + (key_id + 1) * model_->nq);
    std::copy(model_->key_ctrl + key_id * model_->nu,
              model_->key_ctrl + (key_id + 1) * model_->nu,
              default_ctrl_.begin());
    for (int i = 0; i < kApolloActionDim; ++i) {
      actuator_torque_limits_[i] = model_->jnt_actfrcrange[(i + 1) * 2 + 1];
    }

    imu_site_id_ = RequireId(mjOBJ_SITE, "imu");
    torso_body_id_ = RequireId(mjOBJ_BODY, "torso_link");
    gyro_adr_ = SensorAdr("gyro");
    local_linvel_adr_ = SensorAdr("local_linvel");
    upvector_adr_ = SensorAdr("upvector");
    global_linvel_adr_ = SensorAdr("global_linvel");
    global_angvel_adr_ = SensorAdr("global_angvel");
    for (int i = 0; i < kApolloFeet; ++i) {
      feet_site_ids_[i] = RequireId(mjOBJ_SITE, kApolloFeetSites[i]);
      feet_floor_sensor_adrs_[i] = SensorAdr(kApolloFeetFloorSensors[i]);
    }
    for (int i = 0; i < kApolloCollisionSensorCount; ++i) {
      collision_sensor_adrs_[i] = SensorAdr(kApolloCollisionSensors[i]);
    }
  }

  static std::string ApolloXmlPath(const std::string& base_path) {
    return base_path +
           "/mujoco/playground/assets/mujoco_playground/_src/locomotion/"
           "apollo/xmls/scene_mjx_feetonly_flat_terrain.xml";
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    has_cached_render_ = false;
    done_ = false;
    terminated_ = false;
    elapsed_step_ = 0;
    step_ = 0;
    push_step_ = 0;
    ResetRewards();
    std::fill(last_act_.begin(), last_act_.end(), 0.0);
    std::fill(obs_last_act_.begin(), obs_last_act_.end(), 0.0);
    std::fill(filtered_linvel_.begin(), filtered_linvel_.end(), 0.0);
    std::fill(filtered_angvel_.begin(), filtered_angvel_.end(), 0.0);
    std::fill(push_.begin(), push_.end(), 0.0);

    mj_resetData(model_, data_);
    std::copy(init_qpos_.begin(), init_qpos_.end(), data_->qpos);
    std::fill(data_->qvel, data_->qvel + model_->nv, 0.0);
    std::copy(default_ctrl_.begin(), default_ctrl_.end(), data_->ctrl);

    std::uniform_real_distribution<mjtNum> dxy_dist(-0.5, 0.5);
    std::uniform_real_distribution<mjtNum> yaw_dist(-3.14, 3.14);
    std::uniform_real_distribution<mjtNum> joint_scale_dist(0.5, 1.5);
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
    for (int i = 0; i < kApolloActionDim; ++i) {
      data_->qpos[7 + i] *= joint_scale_dist(gen_);
    }
    for (int i = 0; i < 6; ++i) {
      data_->qvel[i] = dxy_dist(gen_);
    }
    mj_forward(model_, data_);
    reset_qpos_.assign(data_->qpos, data_->qpos + model_->nq);
    reset_qvel_.assign(data_->qvel, data_->qvel + model_->nv);
    reset_ctrl_.assign(data_->ctrl, data_->ctrl + model_->nu);

    phase_[0] = 0.0;
    phase_[1] = M_PI;
    const mjtNum gait_freq = Uniform(1.25, 1.75);
    phase_dt_ = 2.0 * M_PI * Dt() * gait_freq;
    obs_phase_ = phase_;

    const mjtNum push_interval = Uniform(spec_.config["push_interval_min"_],
                                         spec_.config["push_interval_max"_]);
    push_interval_steps_ = static_cast<int>(std::llround(push_interval / Dt()));

    steps_until_next_cmd_ =
        static_cast<int>(std::llround(exponential_(gen_) * 5.0 / Dt()));
    for (int i = 0; i < 3; ++i) {
      command_[i] = Uniform(CommandMin(i), CommandMax(i));
    }
    WriteState(0.0, true);
  }

  void Step(const Action& action) override {
    has_cached_render_ = false;
    const auto* act = static_cast<const mjtNum*>(action["action"_].Data());

    ApplyPush();
    for (int i = 0; i < kApolloActionDim; ++i) {
      data_->ctrl[i] =
          default_ctrl_[i] + act[i] * spec_.config["action_scale"_];
    }
    for (int i = 0; i < n_substeps_; ++i) {
      mj_step(model_, data_);
    }

    Sensor3(local_linvel_adr_, filtered_linvel_.data());
    Sensor3(gyro_adr_, filtered_angvel_.data());
    obs_phase_ = phase_;
    obs_last_act_ = last_act_;
    terminated_ = data_->sensordata[upvector_adr_ + 2] < 0.0;
    ComputeRewards(act);
    const mjtNum reward =
        (reward_termination_ + reward_alive_ + reward_tracking_ +
         reward_lin_vel_z_ + reward_ang_vel_xy_ + reward_orientation_ +
         reward_feet_phase_ + reward_torques_ + reward_action_rate_ +
         reward_energy_ + reward_collision_ + reward_pose_) *
        Dt();

    ++step_;
    for (int i = 0; i < kApolloFeet; ++i) {
      phase_[i] = WrapPhase(phase_[i] + phase_dt_);
    }
    const bool nonzero_command = Norm3(command_) > 0.01;
    if (!nonzero_command) {
      phase_[0] = M_PI;
      phase_[1] = M_PI;
    }
    --steps_until_next_cmd_;
    if (steps_until_next_cmd_ <= 0 || terminated_) {
      SampleCommand();
      steps_until_next_cmd_ =
          static_cast<int>(std::llround(exponential_(gen_) * 5.0 / Dt()));
    }
    ++elapsed_step_;
    done_ = terminated_ || elapsed_step_ >= max_episode_steps_;
    WriteState(static_cast<float>(reward), false);

    std::copy(act, act + kApolloActionDim, last_act_.begin());
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

  mjtNum Uniform(mjtNum min, mjtNum max) {
    return min + unit_uniform_(gen_) * (max - min);
  }

  mjtNum CommandMin(int i) const {
    if (i == 0) {
      return spec_.config["command_min0"_];
    }
    if (i == 1) {
      return spec_.config["command_min1"_];
    }
    return spec_.config["command_min2"_];
  }

  mjtNum CommandMax(int i) const {
    if (i == 0) {
      return spec_.config["command_max0"_];
    }
    if (i == 1) {
      return spec_.config["command_max1"_];
    }
    return spec_.config["command_max2"_];
  }

  mjtNum CommandZeroProb(int i) const {
    if (i == 0) {
      return spec_.config["command_zero_prob0"_];
    }
    if (i == 1) {
      return spec_.config["command_zero_prob1"_];
    }
    return spec_.config["command_zero_prob2"_];
  }

  mjtNum MaybeNoisy(mjtNum value, mjtNum scale) {
    return value + noise_uniform_(gen_) * spec_.config["noise_level"_] * scale;
  }

  void Sensor3(int adr, mjtNum* out) const {
    out[0] = data_->sensordata[adr];
    out[1] = data_->sensordata[adr + 1];
    out[2] = data_->sensordata[adr + 2];
  }

  void GravityFromImu(mjtNum* out) const {
    const mjtNum* mat = data_->site_xmat + imu_site_id_ * 9;
    out[0] = -mat[6];
    out[1] = -mat[7];
    out[2] = -mat[8];
  }

  mjtNum Norm3(const std::array<mjtNum, 3>& x) const {
    return std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
  }

  mjtNum WrapPhase(mjtNum x) const {
    return std::fmod(x + M_PI, 2.0 * M_PI) - M_PI;
  }

  void SampleCommand() {
    for (int i = 0; i < 3; ++i) {
      const mjtNum y = Uniform(CommandMin(i), CommandMax(i));
      const mjtNum z = unit_uniform_(gen_) < CommandZeroProb(i) ? 1.0 : 0.0;
      const mjtNum w = unit_uniform_(gen_) < 0.5 ? 1.0 : 0.0;
      command_[i] = command_[i] - w * (command_[i] - y * z);
    }
  }

  void ApplyPush() {
    push_[0] = 0.0;
    push_[1] = 0.0;
    const bool start_push = push_interval_steps_ > 0 &&
                            ((push_step_ + 1) % push_interval_steps_ == 0) &&
                            spec_.config["push_enable"_] != 0.0;
    if (start_push) {
      const mjtNum theta = Uniform(0.0, 2.0 * M_PI);
      const mjtNum magnitude = Uniform(spec_.config["push_magnitude_min"_],
                                       spec_.config["push_magnitude_max"_]);
      push_[0] = std::cos(theta);
      push_[1] = std::sin(theta);
      data_->qvel[0] += push_[0] * magnitude;
      data_->qvel[1] += push_[1] * magnitude;
    }
    ++push_step_;
  }

  mjtNum GaitRz(mjtNum phi) const {
    const mjtNum x = (phi + M_PI) / (2.0 * M_PI);
    auto bezier = [](mjtNum t) {
      return t * t * t + 3.0 * (t * t * (1.0 - t));
    };
    const mjtNum swing_height = spec_.config["max_foot_height"_];
    if (x <= 0.5) {
      return swing_height * bezier(2.0 * x);
    }
    return swing_height * (1.0 - bezier(2.0 * x - 1.0));
  }

  void ResetRewards() {
    reward_termination_ = 0.0;
    reward_alive_ = 0.0;
    reward_tracking_ = 0.0;
    reward_lin_vel_z_ = 0.0;
    reward_ang_vel_xy_ = 0.0;
    reward_orientation_ = 0.0;
    reward_feet_phase_ = 0.0;
    reward_torques_ = 0.0;
    reward_action_rate_ = 0.0;
    reward_energy_ = 0.0;
    reward_collision_ = 0.0;
    reward_pose_ = 0.0;
  }

  void ComputeRewards(const mjtNum* action) {
    const mjtNum lin_vel_error = (command_[0] - filtered_linvel_[0]) *
                                     (command_[0] - filtered_linvel_[0]) +
                                 (command_[1] - filtered_linvel_[1]) *
                                     (command_[1] - filtered_linvel_[1]);
    const mjtNum r_linvel =
        std::exp(-lin_vel_error / spec_.config["tracking_sigma"_]);
    const mjtNum ang_vel_error = (command_[2] - filtered_angvel_[2]) *
                                 (command_[2] - filtered_angvel_[2]);
    const mjtNum r_angvel =
        std::exp(-ang_vel_error / spec_.config["tracking_sigma"_]);
    reward_tracking_ =
        (r_linvel + 0.5 * r_angvel) * spec_.config["tracking_scale"_];
    reward_lin_vel_z_ = filtered_linvel_[2] * filtered_linvel_[2] *
                        spec_.config["lin_vel_z_scale"_];
    reward_ang_vel_xy_ = (filtered_angvel_[0] * filtered_angvel_[0] +
                          filtered_angvel_[1] * filtered_angvel_[1]) *
                         spec_.config["ang_vel_xy_scale"_];
    reward_orientation_ =
        (data_->sensordata[upvector_adr_] * data_->sensordata[upvector_adr_] +
         data_->sensordata[upvector_adr_ + 1] *
             data_->sensordata[upvector_adr_ + 1]) *
        spec_.config["orientation_scale"_];
    reward_termination_ =
        (terminated_ ? 1.0 : 0.0) * spec_.config["termination_scale"_];
    reward_alive_ = (terminated_ ? 0.0 : 1.0) * spec_.config["alive_scale"_];

    mjtNum feet_phase_error = 0.0;
    for (int i = 0; i < kApolloFeet; ++i) {
      const mjtNum foot_z = data_->site_xpos[feet_site_ids_[i] * 3 + 2];
      const mjtNum err = foot_z - GaitRz(phase_[i]);
      feet_phase_error += err * err;
    }
    reward_feet_phase_ =
        std::exp(-feet_phase_error / 0.01) * spec_.config["feet_phase_scale"_];

    mjtNum torques = 0.0;
    mjtNum action_rate = 0.0;
    mjtNum energy = 0.0;
    mjtNum pose = 0.0;
    const bool zero_command = Norm3(command_) < 0.01;
    const mjtNum lateral_cmd = std::abs(command_[1]);
    for (int i = 0; i < kApolloActionDim; ++i) {
      torques += std::abs(data_->actuator_force[i]);
      const mjtNum da = action[i] - last_act_[i];
      action_rate += da * da;
      const mjtNum torque_limit =
          actuator_torque_limits_[i] == 0.0 ? 1.0 : actuator_torque_limits_[i];
      energy += std::abs(data_->qvel[6 + i] *
                         (data_->actuator_force[i] / torque_limit));
      mjtNum weight = zero_command ? 1.0 : kApolloPoseWeights[i];
      if ((i == 21 || i == 27) && lateral_cmd > 0.3) {
        weight = 0.01;
      }
      const mjtNum delta = data_->qpos[7 + i] - init_qpos_[7 + i];
      pose += delta * delta * weight;
    }
    reward_torques_ = torques * spec_.config["torques_scale"_];
    reward_action_rate_ = action_rate * spec_.config["action_rate_scale"_];
    reward_energy_ = energy * spec_.config["energy_scale"_];
    reward_pose_ = pose * spec_.config["pose_scale"_];

    bool collision = false;
    for (int adr : collision_sensor_adrs_) {
      collision = collision || data_->sensordata[adr] > 0.0;
    }
    reward_collision_ =
        (collision ? 1.0 : 0.0) * spec_.config["collision_scale"_];
  }

  void FillObs(mjtNum* state, mjtNum* privileged_state) {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum gyro[3];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum gravity[3];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum linvel[3];
    Sensor3(gyro_adr_, gyro);
    GravityFromImu(gravity);
    Sensor3(local_linvel_adr_, linvel);

    int n = 0;
    for (mjtNum value : linvel) {
      state[n++] = MaybeNoisy(value, spec_.config["noise_linvel"_]);
    }
    for (mjtNum value : gyro) {
      state[n++] = MaybeNoisy(value, spec_.config["noise_gyro"_]);
    }
    for (mjtNum value : gravity) {
      state[n++] = MaybeNoisy(value, spec_.config["noise_gravity"_]);
    }
    for (int i = 0; i < 3; ++i) {
      state[n++] = command_[i];
    }
    for (int i = 0; i < kApolloActionDim; ++i) {
      state[n++] =
          MaybeNoisy(data_->qpos[7 + i], spec_.config["noise_joint_pos"_]) -
          init_qpos_[7 + i];
    }
    for (int i = 0; i < kApolloActionDim; ++i) {
      state[n++] =
          MaybeNoisy(data_->qvel[6 + i], spec_.config["noise_joint_vel"_]);
    }
    for (int i = 0; i < kApolloActionDim; ++i) {
      state[n++] = obs_last_act_[i];
    }
    for (int i = 0; i < kApolloFeet; ++i) {
      state[n++] = std::cos(obs_phase_[i]);
    }
    for (int i = 0; i < kApolloFeet; ++i) {
      state[n++] = std::sin(obs_phase_[i]);
    }

    int p = 0;
    for (int i = 0; i < kApolloStateDim; ++i) {
      privileged_state[p++] = state[i];
    }
    for (mjtNum value : gyro) {
      privileged_state[p++] = value;
    }
    for (mjtNum value : gravity) {
      privileged_state[p++] = value;
    }
    for (mjtNum value : linvel) {
      privileged_state[p++] = value;
    }
    for (int i = 0; i < kApolloActionDim; ++i) {
      privileged_state[p++] = data_->qpos[7 + i] - init_qpos_[7 + i];
    }
    for (int i = 0; i < kApolloActionDim; ++i) {
      privileged_state[p++] = data_->qvel[6 + i];
    }
    for (int i = 0; i < kApolloActionDim; ++i) {
      privileged_state[p++] = data_->actuator_force[i];
    }
    for (int i = 0; i < 3; ++i) {
      privileged_state[p++] = data_->qpos[i];
    }
    for (int i = 0; i < 4; ++i) {
      privileged_state[p++] = data_->qpos[3 + i];
    }
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
    state["info:reward_termination"_] = reward_termination_;
    state["info:reward_alive"_] = reward_alive_;
    state["info:reward_tracking"_] = reward_tracking_;
    state["info:reward_lin_vel_z"_] = reward_lin_vel_z_;
    state["info:reward_ang_vel_xy"_] = reward_ang_vel_xy_;
    state["info:reward_orientation"_] = reward_orientation_;
    state["info:reward_feet_phase"_] = reward_feet_phase_;
    state["info:reward_torques"_] = reward_torques_;
    state["info:reward_action_rate"_] = reward_action_rate_;
    state["info:reward_energy"_] = reward_energy_;
    state["info:reward_collision"_] = reward_collision_;
    state["info:reward_pose"_] = reward_pose_;
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
    std::array<mjtNum, 256> sensor_pad{};
    std::copy(data_->sensordata, data_->sensordata + model_->nsensordata,
              sensor_pad.begin());
    state["info:sensordata"_].Assign(sensor_pad.data(), sensor_pad.size());
    state["info:last_act"_].Assign(last_act_.data(), last_act_.size());
    state["info:phase"_].Assign(phase_.data(), phase_.size());
    state["info:phase_dt"_].Assign(&phase_dt_, 1);
    state["info:push"_].Assign(push_.data(), push_.size());
    state["info:push_step"_] = push_step_;
    state["info:push_interval_steps"_] = push_interval_steps_;
    state["info:filtered_linvel"_].Assign(filtered_linvel_.data(),
                                          filtered_linvel_.size());
    state["info:filtered_angvel"_].Assign(filtered_angvel_.data(),
                                          filtered_angvel_.size());
#endif
  }
};

template <typename Spec, bool kFromPixels>
using ApolloBase = PlaygroundApolloEnvBase<Spec, kFromPixels>;
using ApolloEnv = ApolloBase<PlaygroundApolloEnvSpec, false>;
using ApolloPixelEnv = ApolloBase<PlaygroundApolloPixelEnvSpec, true>;
using PlaygroundApolloEnv = ApolloEnv;
using PlaygroundApolloPixelEnv = ApolloPixelEnv;
using PlaygroundApolloEnvPool = PlaygroundEnvPoolT<PlaygroundApolloEnv>;
using ApolloPixelEnvPool = PlaygroundEnvPoolT<PlaygroundApolloPixelEnv>;
using PlaygroundApolloPixelEnvPool = ApolloPixelEnvPool;

}  // namespace mujoco_playground

#endif  // ENVPOOL_MUJOCO_PLAYGROUND_APOLLO_H_
