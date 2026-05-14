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

#ifndef ENVPOOL_MUJOCO_PLAYGROUND_PANDA_H_
#define ENVPOOL_MUJOCO_PLAYGROUND_PANDA_H_

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

using envpool::mujoco::PixelObservationEnvFns;
using envpool::mujoco::RenderCameraIdOrDefault;
using envpool::mujoco::RenderHeightOrDefault;
using envpool::mujoco::RenderWidthOrDefault;
using envpool::mujoco::StackSpec;

constexpr int kPandaActionDim = 8;
constexpr int kPandaCartesianActionDim = 3;
constexpr int kPandaPickStateDim = 66;
constexpr int kPandaCartesianStateDim = 70;
constexpr int kPandaOpenCabinetStateDim = 55;
constexpr int kPandaMaxStateDim = kPandaCartesianStateDim;
constexpr int kPandaArmJoints = 7;
constexpr int kPandaRobotJoints = 9;
constexpr int kPandaFloorSensors = 3;

class PlaygroundPandaEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "frame_stack"_.Bind(1), "task_name"_.Bind(std::string("PandaPickCube")),
        "ctrl_dt"_.Bind(0.02), "sim_dt"_.Bind(0.005),
        "action_scale"_.Bind(0.04), "gripper_box_scale"_.Bind(4.0),
        "box_target_scale"_.Bind(8.0), "no_floor_collision_scale"_.Bind(0.25),
        "no_box_collision_scale"_.Bind(0.05),
        "no_barrier_collision_scale"_.Bind(0.25),
        "robot_target_qpos_scale"_.Bind(0.3), "action_rate"_.Bind(-0.0005),
        "no_soln_reward"_.Bind(-0.01), "lifted_reward"_.Bind(0.5),
        "success_reward"_.Bind(2.0), "success_threshold"_.Bind(0.05),
        "box_init_range"_.Bind(0.05), "guide_sample_prob"_.Bind(0.05));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    const std::string task_name = conf["task_name"_];
    int state_dim = kPandaPickStateDim;
    if (task_name == "PandaOpenCabinet") {
      state_dim = kPandaOpenCabinetStateDim;
    } else if (task_name == "PandaPickCubeCartesian") {
      state_dim = kPandaCartesianStateDim;
    }
    return MakeDict(
        "obs:state"_.Bind(StackSpec(Spec<mjtNum>({state_dim}, {-inf, inf}),
                                    conf["frame_stack"_])),
        "info:terminated"_.Bind(Spec<bool>({-1})),
        "info:out_of_bounds"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_gripper_box"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_box_target"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_no_floor_collision"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_no_box_collision"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_no_barrier_collision"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_robot_target_qpos"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_lifted"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_success"_.Bind(Spec<mjtNum>({-1}))
#ifdef ENVPOOL_TEST
            ,
        "info:qpos"_.Bind(Spec<mjtNum>({64})),
        "info:qvel"_.Bind(Spec<mjtNum>({64})),
        "info:ctrl"_.Bind(Spec<mjtNum>({64})),
        "info:qacc"_.Bind(Spec<mjtNum>({64})),
        "info:qacc_warmstart"_.Bind(Spec<mjtNum>({64})),
        "info:xfrc_applied"_.Bind(Spec<mjtNum>({512})),
        "info:mocap_pos"_.Bind(Spec<mjtNum>({64})),
        "info:mocap_quat"_.Bind(Spec<mjtNum>({64})),
        "info:xpos"_.Bind(Spec<mjtNum>({512})),
        "info:xmat"_.Bind(Spec<mjtNum>({2048})),
        "info:site_xpos"_.Bind(Spec<mjtNum>({512})),
        "info:site_xmat"_.Bind(Spec<mjtNum>({2048})),
        "info:sensordata"_.Bind(Spec<mjtNum>({256})),
        "info:target_pos"_.Bind(Spec<mjtNum>({3})),
        "info:reached_box"_.Bind(Spec<mjtNum>({-1})),
        "info:previously_gripped"_.Bind(Spec<mjtNum>({-1})),
        "info:prev_reward"_.Bind(Spec<mjtNum>({-1})),
        "info:current_pos"_.Bind(Spec<mjtNum>({3})),
        "info:prev_action"_.Bind(Spec<mjtNum>({3})),
        "info:no_soln"_.Bind(Spec<mjtNum>({-1})),
        "info:steps"_.Bind(Spec<int>({-1}))
#endif
    );  // NOLINT
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    const int action_dim = conf["task_name"_] == "PandaPickCubeCartesian"
                               ? kPandaCartesianActionDim
                               : kPandaActionDim;
    return MakeDict(
        "action"_.Bind(Spec<mjtNum>({-1, action_dim}, {-1.0, 1.0})));
  }
};

using PandaAliases = PlaygroundEnvAliases<PlaygroundPandaEnvFns>;
using PlaygroundPandaEnvSpec = PandaAliases::Spec;
using PlaygroundPandaPixelEnvFns = PandaAliases::PixelFns;
using PlaygroundPandaPixelEnvSpec = PandaAliases::PixelSpec;

template <typename EnvSpecT, bool kFromPixels>
class PlaygroundPandaEnvBase : public Env<EnvSpecT>,
                               public PlaygroundMujocoEnv {
 protected:
  using Base = Env<EnvSpecT>;
  using Base::Allocate;
  using Base::gen_;
  using Base::spec_;

  int n_substeps_{4};
  bool is_open_cabinet_{false};
  bool is_cartesian_{false};
  bool sample_orientation_{false};
  int state_dim_{kPandaPickStateDim};
  std::vector<mjtNum> init_qpos_;
  std::vector<mjtNum> guide_qpos_;
  std::vector<mjtNum> init_ctrl_;
  std::vector<mjtNum> guide_ctrl_;
  std::array<mjtNum, kPandaArmJoints> init_arm_qpos_{};
  std::array<int, kPandaArmJoints> robot_arm_qposadr_{};
  std::array<int, kPandaRobotJoints> robot_qposadr_{};
  std::array<mjtNum, kPandaActionDim> lowers_{};
  std::array<mjtNum, kPandaActionDim> uppers_{};
  std::array<mjtNum, 3> target_pos_{};
  std::array<mjtNum, kPandaMaxStateDim> obs_{};
  std::array<mjtNum, 3> cartesian_start_pos_{};
  std::array<mjtNum, 3> cartesian_current_pos_{};
  std::array<mjtNum, 9> cartesian_start_rot_{};
  std::array<mjtNum, kPandaCartesianActionDim> prev_cartesian_action_{};
  std::array<mjtNum, kPandaCartesianActionDim> last_cartesian_action_{};
  int gripper_site_id_{-1};
  int obj_body_id_{-1};
  int obj_qposadr_{-1};
  int mocap_target_id_{-1};
  int hand_geom_id_{-1};
  int box_hand_sensor_adr_{-1};
  std::array<int, kPandaFloorSensors> floor_sensor_adrs_{};
  std::array<int, kPandaFloorSensors> barrier_sensor_adrs_{};
  mjtNum reached_box_{0.0};
  mjtNum previously_gripped_{0.0};
  mjtNum prev_cartesian_reward_{0.0};
  mjtNum no_soln_{0.0};
  mjtNum out_of_bounds_{0.0};
  mjtNum reward_gripper_box_{0.0};
  mjtNum reward_box_target_{0.0};
  mjtNum reward_no_floor_collision_{0.0};
  mjtNum reward_no_box_collision_{0.0};
  mjtNum reward_no_barrier_collision_{0.0};
  mjtNum reward_robot_target_qpos_{0.0};
  mjtNum reward_lifted_{0.0};
  mjtNum reward_success_{0.0};
  std::uniform_real_distribution<mjtNum> unit_uniform_{0.0, 1.0};

 public:
  using Spec = EnvSpecT;
  using Action = typename Base::Action;

  PlaygroundPandaEnvBase(const Spec& spec, int env_id)
      : Env<EnvSpecT>(spec, env_id),
        PlaygroundMujocoEnv(
            PandaXmlPath(spec.config["base_path"_], spec.config["task_name"_]),
            spec.config["max_episode_steps"_], spec.config["frame_stack"_],
            RenderWidthOrDefault<kFromPixels>(spec.config),
            RenderHeightOrDefault<kFromPixels>(spec.config),
            RenderCameraIdOrDefault<kFromPixels>(spec.config),
            envpool::mujoco::CameraPolicy::kDmControl) {
    const std::string task_name = spec.config["task_name"_];
    is_open_cabinet_ = task_name == "PandaOpenCabinet";
    is_cartesian_ = task_name == "PandaPickCubeCartesian";
    sample_orientation_ = task_name == "PandaPickCubeOrientation";
    if (!is_open_cabinet_ && task_name != "PandaPickCube" &&
        !sample_orientation_ && !is_cartesian_) {
      throw std::runtime_error("Unsupported Panda task_name " + task_name);
    }
    state_dim_ = kPandaPickStateDim;
    if (is_open_cabinet_) {
      state_dim_ = kPandaOpenCabinetStateDim;
    } else if (is_cartesian_) {
      state_dim_ = kPandaCartesianStateDim;
    }
    if (model_->nu != kPandaActionDim) {
      throw std::runtime_error("Unexpected Panda action dimension.");
    }
    n_substeps_ = static_cast<int>(
        std::llround(spec.config["ctrl_dt"_] / spec.config["sim_dt"_]));
    model_->opt.timestep = spec.config["sim_dt"_];
    if (is_open_cabinet_) {
      model_->geom_conaffinity[RequireId(mjOBJ_GEOM, "hand_capsule")] = 3;
    }
    if (is_cartesian_) {
      SetCartesianFingerMaterial();
    }
    std::string robot_home_key = "home";
    if (is_open_cabinet_) {
      robot_home_key = "upright";
    } else if (is_cartesian_) {
      robot_home_key = "low_home";
    }
    InitModelIds(is_open_cabinet_ ? "handle" : "box", robot_home_key);
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    has_cached_render_ = false;
    done_ = false;
    terminated_ = false;
    elapsed_step_ = 0;
    reached_box_ = 0.0;
    previously_gripped_ = 0.0;
    ResetRewards();

    mj_resetData(model_, data_);
    std::copy(init_qpos_.begin(), init_qpos_.end(), data_->qpos);
    std::fill(data_->qvel, data_->qvel + model_->nv, 0.0);
    std::copy(init_ctrl_.begin(), init_ctrl_.end(), data_->ctrl);
    for (int i = 0; i < model_->nmocap; ++i) {
      data_->mocap_pos[3 * i + 0] = 0.0;
      data_->mocap_pos[3 * i + 1] = 0.0;
      data_->mocap_pos[3 * i + 2] = 0.0;
      data_->mocap_quat[4 * i + 0] = 1.0;
      data_->mocap_quat[4 * i + 1] = 0.0;
      data_->mocap_quat[4 * i + 2] = 0.0;
      data_->mocap_quat[4 * i + 3] = 0.0;
    }

    if (is_open_cabinet_) {
      ResetOpenCabinet();
    } else if (is_cartesian_) {
      ResetCartesian();
    } else {
      ResetPickCube();
    }
    mj_forward(model_, data_);
    UpdateObs();
    WriteState(0.0, true);
  }

  void Step(const Action& action) override {
    has_cached_render_ = false;
    if (is_cartesian_) {
      StepCartesian(action);
      return;
    }
    const auto* act = static_cast<const mjtNum*>(action["action"_].Data());
    for (int i = 0; i < kPandaActionDim; ++i) {
      data_->ctrl[i] =
          std::clamp(data_->ctrl[i] + act[i] * spec_.config["action_scale"_],
                     lowers_[i], uppers_[i]);
    }
    for (int i = 0; i < n_substeps_; ++i) {
      mj_step(model_, data_);
    }

    const mjtNum reward =
        is_open_cabinet_ ? ComputeOpenCabinetReward() : ComputePickReward();
    UpdateObs();
    out_of_bounds_ = OutOfBounds() ? 1.0 : 0.0;
    terminated_ = out_of_bounds_ > 0.0 || HasNaN();
    ++elapsed_step_;
    done_ = terminated_ || elapsed_step_ >= max_episode_steps_;
    WriteState(static_cast<float>(reward), false);
  }

 private:
  static std::string PandaXmlPath(const std::string& base_path,
                                  const std::string& task_name) {
    std::string xml_name = "mjx_single_cube.xml";
    if (task_name == "PandaOpenCabinet") {
      xml_name = "mjx_cabinet.xml";
    } else if (task_name == "PandaPickCubeCartesian") {
      xml_name = "mjx_single_cube_camera.xml";
    }
    return base_path +
           "/mujoco/playground/assets/mujoco_playground/_src/"
           "manipulation/franka_emika_panda/xmls/" +
           xml_name;
  }

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

  void InitModelIds(const std::string& obj_name,
                    const std::string& keyframe_name) {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const char* arm_joints[kPandaArmJoints] = {
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"};
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const char* robot_joints[kPandaRobotJoints] = {
        "joint1", "joint2", "joint3",        "joint4",       "joint5",
        "joint6", "joint7", "finger_joint1", "finger_joint2"};
    for (int i = 0; i < kPandaArmJoints; ++i) {
      const int joint_id = RequireId(mjOBJ_JOINT, arm_joints[i]);
      robot_arm_qposadr_[i] = model_->jnt_qposadr[joint_id];
    }
    for (int i = 0; i < kPandaRobotJoints; ++i) {
      const int joint_id = RequireId(mjOBJ_JOINT, robot_joints[i]);
      robot_qposadr_[i] = model_->jnt_qposadr[joint_id];
    }
    gripper_site_id_ = RequireId(mjOBJ_SITE, "gripper");
    hand_geom_id_ = RequireId(mjOBJ_GEOM, "hand_capsule");
    obj_body_id_ = RequireId(mjOBJ_BODY, obj_name);
    if (model_->body_jntnum[obj_body_id_] <= 0) {
      throw std::runtime_error("Panda object body has no joint: " + obj_name);
    }
    obj_qposadr_ = model_->jnt_qposadr[model_->body_jntadr[obj_body_id_]];
    const int mocap_body = RequireId(mjOBJ_BODY, "mocap_target");
    mocap_target_id_ = model_->body_mocapid[mocap_body];
    if (mocap_target_id_ < 0) {
      throw std::runtime_error("Panda mocap_target is not a mocap body.");
    }
    const int key_id = RequireId(mjOBJ_KEY, keyframe_name);
    init_qpos_.assign(model_->key_qpos + key_id * model_->nq,
                      model_->key_qpos + (key_id + 1) * model_->nq);
    init_ctrl_.assign(model_->key_ctrl + key_id * model_->nu,
                      model_->key_ctrl + (key_id + 1) * model_->nu);
    for (int i = 0; i < kPandaArmJoints; ++i) {
      init_arm_qpos_[i] = init_qpos_[robot_arm_qposadr_[i]];
    }
    for (int i = 0; i < kPandaActionDim; ++i) {
      lowers_[i] = model_->actuator_ctrlrange[2 * i];
      uppers_[i] = model_->actuator_ctrlrange[2 * i + 1];
    }
    if (is_open_cabinet_) {
      barrier_sensor_adrs_ = {
          SensorAdr("barrier_left_finger_pad_found"),
          SensorAdr("barrier_right_finger_pad_found"),
          SensorAdr("barrier_hand_capsule_found"),
      };
    } else {
      floor_sensor_adrs_ = {
          SensorAdr("left_finger_pad_floor_found"),
          SensorAdr("right_finger_pad_floor_found"),
          SensorAdr("hand_capsule_floor_found"),
      };
      if (is_cartesian_) {
        box_hand_sensor_adr_ = SensorAdr("box_hand_found");
        const int guide_id = RequireId(mjOBJ_KEY, "picked");
        guide_qpos_.assign(model_->key_qpos + guide_id * model_->nq,
                           model_->key_qpos + (guide_id + 1) * model_->nq);
        guide_ctrl_.assign(model_->key_ctrl + guide_id * model_->nu,
                           model_->key_ctrl + (guide_id + 1) * model_->nu);
        const auto start_tip = ComputeFrankaFk(init_ctrl_.data());
        for (int i = 0; i < 3; ++i) {
          cartesian_start_pos_[i] = start_tip[4 * i + 3];
          for (int j = 0; j < 3; ++j) {
            cartesian_start_rot_[3 * i + j] = start_tip[4 * i + j];
          }
        }
      }
    }
  }

  void SetCartesianFingerMaterial() {
    const int mesh_id = RequireId(mjOBJ_MESH, "finger_1");
    const int mat_id = RequireId(mjOBJ_MATERIAL, "off_white");
    for (int i = 0; i < model_->ngeom; ++i) {
      if (model_->geom_dataid[i] == mesh_id) {
        model_->geom_matid[i] = mat_id;
      }
    }
  }

  void ResetPickCube() {
    const std::array<mjtNum, 3> init_obj_pos = {init_qpos_[obj_qposadr_ + 0],
                                                init_qpos_[obj_qposadr_ + 1],
                                                init_qpos_[obj_qposadr_ + 2]};
    for (int i = 0; i < 2; ++i) {
      data_->qpos[obj_qposadr_ + i] = init_obj_pos[i] + Uniform(-0.2, 0.2);
      target_pos_[i] = init_obj_pos[i] + Uniform(-0.2, 0.2);
    }
    data_->qpos[obj_qposadr_ + 2] = init_obj_pos[2];
    target_pos_[2] = init_obj_pos[2] + Uniform(0.2, 0.4);
    data_->mocap_pos[3 * mocap_target_id_ + 0] = target_pos_[0];
    data_->mocap_pos[3 * mocap_target_id_ + 1] = target_pos_[1];
    data_->mocap_pos[3 * mocap_target_id_ + 2] = target_pos_[2];

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum quat[4] = {1.0, 0.0, 0.0, 0.0};
    if (sample_orientation_) {
      // NOLINTNEXTLINE(modernize-avoid-c-arrays)
      mjtNum axis[3] = {Uniform(-1.0, 1.0), Uniform(-1.0, 1.0),
                        Uniform(-1.0, 1.0)};
      const mjtNum axis_norm =
          std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
      if (axis_norm > 0.0) {
        axis[0] /= axis_norm;
        axis[1] /= axis_norm;
        axis[2] /= axis_norm;
      } else {
        axis[0] = 1.0;
      }
      const mjtNum theta = Uniform(0.0, M_PI / 4.0);
      const mjtNum half = 0.5 * theta;
      quat[0] = std::cos(half);
      const mjtNum scale = std::sin(half);
      quat[1] = axis[0] * scale;
      quat[2] = axis[1] * scale;
      quat[3] = axis[2] * scale;
    }
    std::copy(quat, quat + 4, data_->mocap_quat + 4 * mocap_target_id_);
  }

  void ResetCartesian() {
    const mjtNum x_plane = cartesian_start_pos_[0] - 0.03;
    data_->qpos[obj_qposadr_ + 0] = x_plane;
    data_->qpos[obj_qposadr_ + 1] = Uniform(-spec_.config["box_init_range"_],
                                            spec_.config["box_init_range"_]);
    data_->qpos[obj_qposadr_ + 2] = 0.0;
    target_pos_ = {x_plane, 0.0, 0.20};
    data_->mocap_pos[3 * mocap_target_id_ + 0] = target_pos_[0];
    data_->mocap_pos[3 * mocap_target_id_ + 1] = target_pos_[1];
    data_->mocap_pos[3 * mocap_target_id_ + 2] = target_pos_[2];
    data_->mocap_quat[4 * mocap_target_id_ + 0] = 1.0;
    data_->mocap_quat[4 * mocap_target_id_ + 1] = 0.0;
    data_->mocap_quat[4 * mocap_target_id_ + 2] = 0.0;
    data_->mocap_quat[4 * mocap_target_id_ + 3] = 0.0;
    cartesian_current_pos_ = cartesian_start_pos_;
    prev_cartesian_action_.fill(0.0);
    last_cartesian_action_.fill(0.0);
    prev_cartesian_reward_ = 0.0;
    no_soln_ = 0.0;
  }

  void ResetOpenCabinet() {
    target_pos_ = {0.3 + Uniform(-0.1, 0.1), 0.0, 0.5};
    const mjtNum eps = M_PI / 6.0;
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const mjtNum mins[kPandaArmJoints] = {-eps, -eps, -eps, -2 * eps,
                                          -eps, 0.0,  -eps};
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const mjtNum maxs[kPandaArmJoints] = {eps, eps,     eps, 0.0,
                                          eps, 2 * eps, eps};
    for (int i = 0; i < kPandaArmJoints; ++i) {
      const mjtNum perturb = Uniform(mins[i], maxs[i]);
      data_->qpos[robot_arm_qposadr_[i]] =
          init_qpos_[robot_arm_qposadr_[i]] + perturb;
      data_->ctrl[i] = init_ctrl_[i] + perturb;
    }
  }

  void ResetRewards() {
    out_of_bounds_ = 0.0;
    reward_gripper_box_ = 0.0;
    reward_box_target_ = 0.0;
    reward_no_floor_collision_ = 0.0;
    reward_no_box_collision_ = 0.0;
    reward_no_barrier_collision_ = 0.0;
    reward_robot_target_qpos_ = 0.0;
    reward_lifted_ = 0.0;
    reward_success_ = 0.0;
  }

  static std::array<mjtNum, 16> Identity4() {
    return {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  }

  static std::array<mjtNum, 16> Mat4Mul(const std::array<mjtNum, 16>& lhs,
                                        const std::array<mjtNum, 16>& rhs) {
    std::array<mjtNum, 16> out{};
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        mjtNum value = 0.0;
        for (int k = 0; k < 4; ++k) {
          value += lhs[4 * i + k] * rhs[4 * k + j];
        }
        out[4 * i + j] = value;
      }
    }
    return out;
  }

  static std::array<mjtNum, 16> Dh(mjtNum theta, mjtNum alpha, mjtNum a,
                                   mjtNum d, mjtNum offset) {
    const mjtNum ca = std::cos(alpha);
    const mjtNum sa = std::sin(alpha);
    const mjtNum th = theta + offset;
    const mjtNum ct = std::cos(th);
    const mjtNum st = std::sin(th);
    return {ct,      -st,     0.0, a,      st * ca, ct * ca, -sa, -d * sa,
            st * sa, ct * sa, ca,  d * ca, 0.0,     0.0,     0.0, 1.0};
  }

  static std::array<mjtNum, 16> ComputeFrankaFk(const mjtNum* joint_pos) {
    constexpr mjtNum k_pi2 = M_PI / 2.0;
    auto transform = Identity4();
    transform = Mat4Mul(transform, Dh(joint_pos[0], 0.0, 0.0, 0.333, 0.0));
    transform = Mat4Mul(transform, Dh(joint_pos[1], -k_pi2, 0.0, 0.0, 0.0));
    transform = Mat4Mul(transform, Dh(joint_pos[2], k_pi2, 0.0, 0.316, 0.0));
    transform = Mat4Mul(transform, Dh(joint_pos[3], k_pi2, 0.0825, 0.0, 0.0));
    transform =
        Mat4Mul(transform, Dh(joint_pos[4], -k_pi2, -0.0825, 0.384, 0.0));
    transform = Mat4Mul(transform, Dh(joint_pos[5], k_pi2, 0.0, 0.0, 0.0));
    transform = Mat4Mul(transform, Dh(joint_pos[6], k_pi2, 0.088, 0.0, 0.0));
    transform = Mat4Mul(transform, Dh(-M_PI / 4.0, 0.0, 0.0, 0.2104, 0.0));
    return transform;
  }

  static std::array<mjtNum, 3> Vec3(mjtNum x, mjtNum y, mjtNum z) {
    return {x, y, z};
  }

  static std::array<mjtNum, 3> VecAdd(const std::array<mjtNum, 3>& lhs,
                                      const std::array<mjtNum, 3>& rhs) {
    return {lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
  }

  static std::array<mjtNum, 3> VecSub(const std::array<mjtNum, 3>& lhs,
                                      const std::array<mjtNum, 3>& rhs) {
    return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
  }

  static std::array<mjtNum, 3> VecScale(const std::array<mjtNum, 3>& value,
                                        mjtNum scale) {
    return {value[0] * scale, value[1] * scale, value[2] * scale};
  }

  static mjtNum Dot3(const std::array<mjtNum, 3>& lhs,
                     const std::array<mjtNum, 3>& rhs) {
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
  }

  static std::array<mjtNum, 3> Cross3(const std::array<mjtNum, 3>& lhs,
                                      const std::array<mjtNum, 3>& rhs) {
    return {lhs[1] * rhs[2] - lhs[2] * rhs[1],
            lhs[2] * rhs[0] - lhs[0] * rhs[2],
            lhs[0] * rhs[1] - lhs[1] * rhs[0]};
  }

  static mjtNum Norm3Array(const std::array<mjtNum, 3>& value) {
    return std::sqrt(Dot3(value, value));
  }

  static std::array<mjtNum, 3> Normalize3(std::array<mjtNum, 3> value) {
    const mjtNum norm = Norm3Array(value);
    if (norm > 0.0) {
      value[0] /= norm;
      value[1] /= norm;
      value[2] /= norm;
    }
    return value;
  }

  static std::array<mjtNum, 3> Mat3MulVec(const std::array<mjtNum, 9>& mat,
                                          const std::array<mjtNum, 3>& vec) {
    return {mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2],
            mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2],
            mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2]};
  }

  static std::array<mjtNum, 3> Mat3TransposeMulVec(
      const std::array<mjtNum, 9>& mat, const std::array<mjtNum, 3>& vec) {
    return {mat[0] * vec[0] + mat[3] * vec[1] + mat[6] * vec[2],
            mat[1] * vec[0] + mat[4] * vec[1] + mat[7] * vec[2],
            mat[2] * vec[0] + mat[5] * vec[1] + mat[8] * vec[2]};
  }

  static std::array<mjtNum, 9> Mat3Mul(const std::array<mjtNum, 9>& lhs,
                                       const std::array<mjtNum, 9>& rhs) {
    std::array<mjtNum, 9> out{};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        mjtNum value = 0.0;
        for (int k = 0; k < 3; ++k) {
          value += lhs[3 * i + k] * rhs[3 * k + j];
        }
        out[3 * i + j] = value;
      }
    }
    return out;
  }

  static std::array<mjtNum, 9> Mat3Transpose(const std::array<mjtNum, 9>& mat) {
    return {mat[0], mat[3], mat[6], mat[1], mat[4],
            mat[7], mat[2], mat[5], mat[8]};
  }

  static bool Invalid(mjtNum value) { return !std::isfinite(value); }

  static bool AnyInvalid(const std::array<mjtNum, kPandaArmJoints>& value) {
    return std::any_of(value.begin(), value.end(),
                       [](mjtNum item) { return Invalid(item); });
  }

  static std::array<mjtNum, 3> Mat4Pos(const std::array<mjtNum, 16>& mat) {
    return {mat[3], mat[7], mat[11]};
  }

  static std::array<mjtNum, 3> Mat4ZAxis(const std::array<mjtNum, 16>& mat) {
    return {mat[2], mat[6], mat[10]};
  }

  static std::array<mjtNum, kPandaArmJoints> ComputeFrankaIk(
      const std::array<mjtNum, 16>& t_ee_0, mjtNum q7, const mjtNum* q_actual) {
    constexpr mjtNum k_pi4 = M_PI / 4.0;
    constexpr mjtNum d1 = 0.3330;
    constexpr mjtNum d3 = 0.3160;
    constexpr mjtNum d5 = 0.3840;
    constexpr mjtNum d7e = 0.2104;
    constexpr mjtNum a4 = 0.0825;
    constexpr mjtNum a7 = 0.0880;
    constexpr mjtNum ll24 = 0.10666225;
    constexpr mjtNum ll46 = 0.15426225;
    constexpr mjtNum l24 = 0.326591870689;
    constexpr mjtNum l46 = 0.392762332715;
    constexpr mjtNum theta_h46 = 1.35916951803;
    constexpr mjtNum theta_342 = 1.31542071191;
    constexpr mjtNum theta_46h = 0.211626808766;
    constexpr mjtNum q6_min = -0.0175;
    constexpr mjtNum q6_max = 3.7525;

    auto cast_q = [](mjtNum value) -> mjtNum {
      return static_cast<float>(value);
    };
    std::array<mjtNum, kPandaArmJoints> q{};
    q.fill(0.0);
    q[6] = cast_q(q7);

    const mjtNum c1_a = std::cos(q_actual[0]);
    const mjtNum s1_a = std::sin(q_actual[0]);
    const mjtNum c2_a = std::cos(q_actual[1]);
    const mjtNum s2_a = std::sin(q_actual[1]);
    const mjtNum c3_a = std::cos(q_actual[2]);
    const mjtNum s3_a = std::sin(q_actual[2]);
    const mjtNum c4_a = std::cos(q_actual[3]);
    const mjtNum s4_a = std::sin(q_actual[3]);
    const mjtNum c5_a = std::cos(q_actual[4]);
    const mjtNum s5_a = std::sin(q_actual[4]);
    const mjtNum c6_a = std::cos(q_actual[5]);
    const mjtNum s6_a = std::sin(q_actual[5]);

    const std::array<std::array<mjtNum, 16>, 7> a_mats = {{
        {c1_a, -s1_a, 0.0, 0.0, s1_a, c1_a, 0.0, 0.0, 0.0, 0.0, 1.0, d1, 0.0,
         0.0, 0.0, 1.0},
        {c2_a, -s2_a, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -s2_a, -c2_a, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0},
        {c3_a, -s3_a, 0.0, 0.0, 0.0, 0.0, -1.0, -d3, s3_a, c3_a, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0},
        {c4_a, -s4_a, 0.0, a4, 0.0, 0.0, -1.0, 0.0, s4_a, c4_a, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0},
        {1.0, 0.0, 0.0, -a4, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 1.0},
        {c5_a, -s5_a, 0.0, 0.0, 0.0, 0.0, 1.0, d5, -s5_a, -c5_a, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0},
        {c6_a, -s6_a, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, s6_a, c6_a, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0},
    }};
    std::array<std::array<mjtNum, 16>, 7> t_mats{};
    t_mats[0] = a_mats[0];
    for (int i = 1; i < 7; ++i) {
      t_mats[i] = Mat4Mul(t_mats[i - 1], a_mats[i]);
    }

    const auto v62_a = VecSub(Mat4Pos(t_mats[1]), Mat4Pos(t_mats[6]));
    const auto v6h_a = VecSub(Mat4Pos(t_mats[4]), Mat4Pos(t_mats[6]));
    const auto z6_a = Mat4ZAxis(t_mats[6]);
    const bool is_case6_0 = Dot3(Cross3(v6h_a, v62_a), z6_a) <= 0.0;
    const bool is_case1_1 = q_actual[1] < 0.0;

    std::array<mjtNum, 9> r_ee = {t_ee_0[0], t_ee_0[1], t_ee_0[2],
                                  t_ee_0[4], t_ee_0[5], t_ee_0[6],
                                  t_ee_0[8], t_ee_0[9], t_ee_0[10]};
    const auto z_ee = Vec3(t_ee_0[2], t_ee_0[6], t_ee_0[10]);
    const auto p_ee = Vec3(t_ee_0[3], t_ee_0[7], t_ee_0[11]);
    const auto p_7 = VecSub(p_ee, VecScale(z_ee, d7e));

    const auto x_ee_6 =
        Vec3(std::cos(q[6] - k_pi4), -std::sin(q[6] - k_pi4), 0.0);
    auto x_6 = Normalize3(Mat3MulVec(r_ee, x_ee_6));
    const auto p_6 = VecSub(p_7, VecScale(x_6, a7));

    const auto p_2 = Vec3(0.0, 0.0, d1);
    const auto v26 = VecSub(p_6, p_2);
    const mjtNum ll26 = Dot3(v26, v26);
    const mjtNum l26 = std::sqrt(ll26);
    const mjtNum theta246 = std::acos((ll24 + ll46 - ll26) / 2.0 / l24 / l46);
    q[3] = cast_q(theta246 + theta_h46 + theta_342 - 2.0 * M_PI);

    const mjtNum theta462 = std::acos((ll26 + ll46 - ll24) / 2.0 / l26 / l46);
    const mjtNum theta26h = theta_46h + theta462;
    const mjtNum d26 = -l26 * std::cos(theta26h);

    const auto z_6 = Cross3(z_ee, x_6);
    const auto y_6 = Cross3(z_6, x_6);
    const auto y_6_norm = Normalize3(y_6);
    const auto z_6_norm = Normalize3(z_6);
    const std::array<mjtNum, 9> r_6 = {x_6[0], y_6_norm[0], z_6_norm[0],
                                       x_6[1], y_6_norm[1], z_6_norm[1],
                                       x_6[2], y_6_norm[2], z_6_norm[2]};
    const auto v_6_62 = Mat3TransposeMulVec(r_6, VecScale(v26, -1.0));

    const mjtNum phi6 = std::atan2(v_6_62[1], v_6_62[0]);
    const mjtNum theta6 = std::asin(
        d26 / std::sqrt(v_6_62[0] * v_6_62[0] + v_6_62[1] * v_6_62[1]));
    q[5] = cast_q(is_case6_0 ? M_PI - theta6 - phi6 : theta6 - phi6);
    if (q[5] <= q6_min) {
      q[5] = cast_q(q[5] + 2.0 * M_PI);
    }
    if (q[5] >= q6_max) {
      q[5] = cast_q(q[5] - 2.0 * M_PI);
    }

    const mjtNum theta_p26 = 3.0 * M_PI / 2.0 - theta462 - theta246 - theta_342;
    const mjtNum theta_p = M_PI - theta_p26 - theta26h;
    const mjtNum lp6 = l26 * std::sin(theta_p26) / std::sin(theta_p);

    const auto z_6_5 = Vec3(std::sin(q[5]), std::cos(q[5]), 0.0);
    const auto z_5 = Mat3MulVec(r_6, z_6_5);
    const auto v2p = VecSub(VecSub(p_6, VecScale(z_5, lp6)), p_2);
    const mjtNum l2p = Norm3Array(v2p);
    const bool greater_than_1 = std::abs(v2p[2] / l2p) > 0.999;
    q[0] = cast_q(greater_than_1 ? q_actual[0] : std::atan2(v2p[1], v2p[0]));
    q[1] = cast_q(greater_than_1 ? 0.0 : std::acos(v2p[2] / l2p));
    if (is_case1_1) {
      q[0] = cast_q(q[0] < 0.0 ? q[0] + M_PI : q[0] - M_PI);
      if (!greater_than_1) {
        q[1] = cast_q(-q[1]);
      }
    }

    const auto z_3 = VecScale(v2p, 1.0 / l2p);
    const auto y_3 = Normalize3(VecScale(Cross3(v26, v2p), -1.0));
    const auto x_3 = Cross3(y_3, z_3);
    const mjtNum c1 = std::cos(q[0]);
    const mjtNum s1 = std::sin(q[0]);
    const std::array<mjtNum, 9> r_1 = {c1,  -s1, 0.0, s1, c1,
                                       0.0, 0.0, 0.0, 1.0};
    const mjtNum c2 = std::cos(q[1]);
    const mjtNum s2 = std::sin(q[1]);
    const std::array<mjtNum, 9> r_1_2 = {c2,  -s2, 0.0, 0.0, 0.0,
                                         1.0, -s2, -c2, 0.0};
    const auto r_2 = Mat3Mul(r_1, r_1_2);
    const auto x_2_3 = Mat3TransposeMulVec(r_2, x_3);
    q[2] = cast_q(std::atan2(x_2_3[2], x_2_3[0]));

    const auto vh4 = VecAdd(
        VecSub(VecAdd(p_2, VecAdd(VecScale(z_3, d3), VecScale(x_3, a4))), p_6),
        VecScale(z_5, d5));
    const mjtNum c6 = std::cos(q[5]);
    const mjtNum s6 = std::sin(q[5]);
    const std::array<mjtNum, 9> r_5_6 = {c6,   -s6, 0.0, 0.0, 0.0,
                                         -1.0, s6,  c6,  0.0};
    const auto r_5 = Mat3Mul(r_6, Mat3Transpose(r_5_6));
    const auto v_5_h4 = Mat3TransposeMulVec(r_5, vh4);
    q[4] = cast_q(-std::atan2(v_5_h4[1], v_5_h4[0]));
    return q;
  }

  static mjtNum Norm3(const mjtNum* values) {
    return std::sqrt(values[0] * values[0] + values[1] * values[1] +
                     values[2] * values[2]);
  }

  mjtNum ArmPoseError() const {
    mjtNum err = 0.0;
    for (int i = 0; i < kPandaArmJoints; ++i) {
      const mjtNum delta =
          data_->qpos[robot_arm_qposadr_[i]] - init_arm_qpos_[i];
      err += delta * delta;
    }
    return std::sqrt(err);
  }

  bool SensorContact(const std::array<int, kPandaFloorSensors>& adrs) const {
    return std::any_of(adrs.begin(), adrs.end(), [this](int adr) {
      return data_->sensordata[adr] > 0.0;
    });
  }

  mjtNum ComputePickReward() {
    const mjtNum* box_pos = data_->xpos + 3 * obj_body_id_;
    const mjtNum* gripper_pos = data_->site_xpos + 3 * gripper_site_id_;
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum target_delta[3] = {target_pos_[0] - box_pos[0],
                              target_pos_[1] - box_pos[1],
                              target_pos_[2] - box_pos[2]};
    const mjtNum pos_err = Norm3(target_delta);
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum target_mat[9];
    mju_quat2Mat(target_mat, data_->mocap_quat + 4 * mocap_target_id_);
    const mjtNum* box_mat = data_->xmat + 9 * obj_body_id_;
    mjtNum rot_err_sq = 0.0;
    for (int i = 0; i < 6; ++i) {
      const mjtNum delta = target_mat[i] - box_mat[i];
      rot_err_sq += delta * delta;
    }
    const mjtNum rot_err = std::sqrt(rot_err_sq);
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum gripper_delta[3] = {box_pos[0] - gripper_pos[0],
                               box_pos[1] - gripper_pos[1],
                               box_pos[2] - gripper_pos[2]};
    const mjtNum gripper_err = Norm3(gripper_delta);
    reached_box_ = std::max(reached_box_, gripper_err < 0.012 ? 1.0 : 0.0);
    reward_gripper_box_ = 1.0 - std::tanh(5.0 * gripper_err);
    reward_box_target_ =
        (1.0 - std::tanh(5.0 * (0.9 * pos_err + 0.1 * rot_err))) * reached_box_;
    reward_no_floor_collision_ = SensorContact(floor_sensor_adrs_) ? 0.0 : 1.0;
    reward_no_barrier_collision_ = 0.0;
    reward_robot_target_qpos_ = 1.0 - std::tanh(ArmPoseError());
    const mjtNum reward =
        reward_gripper_box_ * spec_.config["gripper_box_scale"_] +
        reward_box_target_ * spec_.config["box_target_scale"_] +
        reward_no_floor_collision_ * spec_.config["no_floor_collision_scale"_] +
        reward_robot_target_qpos_ * spec_.config["robot_target_qpos_scale"_];
    return std::clamp(reward, static_cast<mjtNum>(-10000.0),
                      static_cast<mjtNum>(10000.0));
  }

  void StepCartesian(const Action& action) {
    const auto* act = static_cast<const mjtNum*>(action["action"_].Data());
    std::array<mjtNum, kPandaCartesianActionDim> applied_action = {
        act[0], act[1], act[2]};
    const bool newly_reset = elapsed_step_ == 0;
    if (newly_reset) {
      prev_cartesian_reward_ = 0.0;
      cartesian_current_pos_ = cartesian_start_pos_;
      reached_box_ = 0.0;
      prev_cartesian_action_.fill(0.0);
    }

    const mjtNum guide_sample_prob = spec_.config["guide_sample_prob"_];
    if (newly_reset && guide_sample_prob > 0.0 &&
        Uniform(0.0, 1.0) < guide_sample_prob) {
      std::copy(guide_qpos_.begin(), guide_qpos_.end(), data_->qpos);
      std::copy(guide_ctrl_.begin(), guide_ctrl_.end(), data_->ctrl);
    }

    MoveCartesianTip(applied_action);
    for (int i = 0; i < n_substeps_; ++i) {
      mj_step(model_, data_);
    }

    mjtNum total_reward = ComputePickReward();
    reward_no_box_collision_ =
        data_->sensordata[box_hand_sensor_adr_] > 0.0 ? 0.0 : 1.0;
    const mjtNum da =
        std::sqrt((applied_action[0] - prev_cartesian_action_[0]) *
                      (applied_action[0] - prev_cartesian_action_[0]) +
                  (applied_action[1] - prev_cartesian_action_[1]) *
                      (applied_action[1] - prev_cartesian_action_[1]) +
                  (applied_action[2] - prev_cartesian_action_[2]) *
                      (applied_action[2] - prev_cartesian_action_[2]));
    prev_cartesian_action_ = applied_action;
    total_reward += spec_.config["action_rate"_] * da;
    total_reward += no_soln_ * spec_.config["no_soln_reward"_];

    const mjtNum* box_pos = data_->xpos + 3 * obj_body_id_;
    reward_lifted_ = box_pos[2] > 0.05 ? spec_.config["lifted_reward"_] : 0.0;
    total_reward += reward_lifted_;
    reward_success_ = CartesianSuccess() ? 1.0 : 0.0;
    total_reward += reward_success_ * spec_.config["success_reward"_];

    mjtNum reward = std::max(total_reward - prev_cartesian_reward_, 0.0);
    prev_cartesian_reward_ = std::max(total_reward, prev_cartesian_reward_);
    if (newly_reset) {
      reward = 0.0;
    }

    last_cartesian_action_ = applied_action;
    UpdateObs();
    out_of_bounds_ = OutOfBounds() ? 1.0 : 0.0;
    terminated_ = out_of_bounds_ > 0.0 || HasNaN() || reward_success_ > 0.0;
    ++elapsed_step_;
    done_ = terminated_ || elapsed_step_ >= max_episode_steps_;
    WriteState(static_cast<float>(reward), false);
  }

  void MoveCartesianTip(
      const std::array<mjtNum, kPandaCartesianActionDim>& action) {
    const mjtNum action_scale = spec_.config["action_scale"_];
    std::array<mjtNum, 3> new_tip_pos = cartesian_current_pos_;
    new_tip_pos[1] += action[0] * action_scale;
    new_tip_pos[2] += action[1] * action_scale;
    new_tip_pos[0] = std::clamp(new_tip_pos[0], 0.25, 0.77);
    new_tip_pos[1] = std::clamp(new_tip_pos[1], -0.32, 0.32);
    new_tip_pos[2] = std::clamp(new_tip_pos[2], 0.02, 0.5);

    auto tip_transform = Identity4();
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        tip_transform[4 * i + j] = cartesian_start_rot_[3 * i + j];
      }
      tip_transform[4 * i + 3] = new_tip_pos[i];
    }
    auto ik = ComputeFrankaIk(tip_transform, data_->ctrl[6], data_->ctrl);
    bool no_soln = AnyInvalid(ik);
    if (no_soln) {
      for (int i = 0; i < kPandaArmJoints; ++i) {
        ik[i] = data_->ctrl[i];
      }
    }
    const bool invalid_after_fallback = AnyInvalid(ik);
    no_soln = no_soln || invalid_after_fallback;
    if (!invalid_after_fallback) {
      cartesian_current_pos_ = new_tip_pos;
    }
    for (int i = 0; i < kPandaArmJoints; ++i) {
      data_->ctrl[i] = ik[i];
    }
    const mjtNum jaw_action = action[2] < 0.0 ? -1.0 : 1.0;
    data_->ctrl[7] += jaw_action * 0.02;
    for (int i = 0; i < kPandaActionDim; ++i) {
      data_->ctrl[i] = std::clamp(data_->ctrl[i], lowers_[i], uppers_[i]);
    }
    no_soln_ = no_soln ? 1.0 : 0.0;
  }

  bool CartesianSuccess() const {
    const mjtNum* box_pos = data_->xpos + 3 * obj_body_id_;
    const std::array<mjtNum, 3> delta = {box_pos[0] - target_pos_[0],
                                         box_pos[1] - target_pos_[1],
                                         box_pos[2] - target_pos_[2]};
    return Norm3Array(delta) < spec_.config["success_threshold"_];
  }

  mjtNum ComputeOpenCabinetReward() {
    const mjtNum* box_pos = data_->xpos + 3 * obj_body_id_;
    const mjtNum* gripper_pos = data_->site_xpos + 3 * gripper_site_id_;
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum target_delta[3] = {target_pos_[0] - box_pos[0],
                              target_pos_[1] - box_pos[1],
                              target_pos_[2] - box_pos[2]};
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum gripper_delta[3] = {box_pos[0] - gripper_pos[0],
                               box_pos[1] - gripper_pos[1],
                               box_pos[2] - gripper_pos[2]};
    const mjtNum gripper_err = Norm3(gripper_delta);
    reached_box_ = std::max(reached_box_, gripper_err < 0.012 ? 1.0 : 0.0);
    reward_gripper_box_ = 1.0 - std::tanh(5.0 * gripper_err);
    reward_box_target_ =
        (1.0 - std::tanh(5.0 * Norm3(target_delta))) * reached_box_;
    reward_no_floor_collision_ = 0.0;
    reward_no_barrier_collision_ =
        SensorContact(barrier_sensor_adrs_) ? 0.0 : 1.0;
    reward_robot_target_qpos_ = 1.0 - std::tanh(ArmPoseError());
    const mjtNum reward =
        reward_gripper_box_ * spec_.config["gripper_box_scale"_] +
        reward_box_target_ * spec_.config["box_target_scale"_] +
        reward_no_barrier_collision_ *
            spec_.config["no_barrier_collision_scale"_] +
        reward_robot_target_qpos_ * spec_.config["robot_target_qpos_scale"_];
    return std::clamp(reward, static_cast<mjtNum>(-10000.0),
                      static_cast<mjtNum>(10000.0));
  }

  bool OutOfBounds() const {
    const mjtNum* pos = data_->xpos + 3 * obj_body_id_;
    return std::abs(pos[0]) > 1.0 || std::abs(pos[1]) > 1.0 ||
           std::abs(pos[2]) > 1.0 || pos[2] < 0.0;
  }

  bool HasNaN() const {
    return std::any_of(data_->qpos, data_->qpos + model_->nq,
                       [](mjtNum q) { return std::isnan(q); }) ||
           std::any_of(data_->qvel, data_->qvel + model_->nv,
                       [](mjtNum qvel) { return std::isnan(qvel); });
  }

  void UpdateObs() {
    obs_.fill(0.0);
    int index = 0;
    std::copy(data_->qpos, data_->qpos + model_->nq, obs_.begin() + index);
    index += model_->nq;
    std::copy(data_->qvel, data_->qvel + model_->nv, obs_.begin() + index);
    index += model_->nv;
    const mjtNum* gripper_pos = data_->site_xpos + 3 * gripper_site_id_;
    std::copy(gripper_pos, gripper_pos + 3, obs_.begin() + index);
    index += 3;
    const mjtNum* gripper_mat = data_->site_xmat + 9 * gripper_site_id_;
    std::copy(gripper_mat + 3, gripper_mat + 9, obs_.begin() + index);
    index += 6;
    const mjtNum* obj_mat = data_->xmat + 9 * obj_body_id_;
    std::copy(obj_mat + 3, obj_mat + 9, obs_.begin() + index);
    index += 6;
    const mjtNum* obj_pos = data_->xpos + 3 * obj_body_id_;
    for (int i = 0; i < 3; ++i) {
      obs_[index++] = obj_pos[i] - gripper_pos[i];
    }
    for (int i = 0; i < 3; ++i) {
      obs_[index++] = target_pos_[i] - obj_pos[i];
    }
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    mjtNum target_mat[9];
    mju_quat2Mat(target_mat, data_->mocap_quat + 4 * mocap_target_id_);
    for (int i = 0; i < 6; ++i) {
      obs_[index++] = target_mat[i] - obj_mat[i];
    }
    for (int i = 0; i < kPandaActionDim; ++i) {
      obs_[index++] = data_->ctrl[i] - data_->qpos[robot_qposadr_[i]];
    }
    if (is_cartesian_) {
      obs_[index++] = no_soln_;
      for (mjtNum value : last_cartesian_action_) {
        obs_[index++] = value;
      }
    }
  }

  template <std::size_t N>
  static void CopyPadded(const mjtNum* src, int count,
                         std::array<mjtNum, N>* dst) {
    dst->fill(0.0);
    const int n = std::min<int>(count, static_cast<int>(N));
    std::copy(src, src + n, dst->begin());
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
      std::copy(obs_.begin(), obs_.begin() + state_dim_, obs);
      CommitObservation("obs:state", &obs_state, reset);
    }
    state["info:terminated"_] = terminated_;
    state["info:out_of_bounds"_] = out_of_bounds_;
    state["info:reward_gripper_box"_] = reward_gripper_box_;
    state["info:reward_box_target"_] = reward_box_target_;
    state["info:reward_no_floor_collision"_] = reward_no_floor_collision_;
    state["info:reward_no_box_collision"_] = reward_no_box_collision_;
    state["info:reward_no_barrier_collision"_] = reward_no_barrier_collision_;
    state["info:reward_robot_target_qpos"_] = reward_robot_target_qpos_;
    state["info:reward_lifted"_] = reward_lifted_;
    state["info:reward_success"_] = reward_success_;
#ifdef ENVPOOL_TEST
    std::array<mjtNum, 64> pad64{};
    CopyPadded(data_->qpos, model_->nq, &pad64);
    state["info:qpos"_].Assign(pad64.data(), pad64.size());
    CopyPadded(data_->qvel, model_->nv, &pad64);
    state["info:qvel"_].Assign(pad64.data(), pad64.size());
    CopyPadded(data_->ctrl, model_->nu, &pad64);
    state["info:ctrl"_].Assign(pad64.data(), pad64.size());
    CopyPadded(data_->qacc, model_->nv, &pad64);
    state["info:qacc"_].Assign(pad64.data(), pad64.size());
    CopyPadded(data_->qacc_warmstart, model_->nv, &pad64);
    state["info:qacc_warmstart"_].Assign(pad64.data(), pad64.size());
    CopyPadded(data_->mocap_pos, model_->nmocap * 3, &pad64);
    state["info:mocap_pos"_].Assign(pad64.data(), pad64.size());
    CopyPadded(data_->mocap_quat, model_->nmocap * 4, &pad64);
    state["info:mocap_quat"_].Assign(pad64.data(), pad64.size());
    std::array<mjtNum, 512> pad512{};
    CopyPadded(data_->xfrc_applied, model_->nbody * 6, &pad512);
    state["info:xfrc_applied"_].Assign(pad512.data(), pad512.size());
    CopyPadded(data_->xpos, model_->nbody * 3, &pad512);
    state["info:xpos"_].Assign(pad512.data(), pad512.size());
    CopyPadded(data_->site_xpos, model_->nsite * 3, &pad512);
    state["info:site_xpos"_].Assign(pad512.data(), pad512.size());
    std::array<mjtNum, 2048> pad2048{};
    CopyPadded(data_->xmat, model_->nbody * 9, &pad2048);
    state["info:xmat"_].Assign(pad2048.data(), pad2048.size());
    CopyPadded(data_->site_xmat, model_->nsite * 9, &pad2048);
    state["info:site_xmat"_].Assign(pad2048.data(), pad2048.size());
    std::array<mjtNum, 256> sensor_pad{};
    CopyPadded(data_->sensordata, model_->nsensordata, &sensor_pad);
    state["info:sensordata"_].Assign(sensor_pad.data(), sensor_pad.size());
    state["info:target_pos"_].Assign(target_pos_.data(), target_pos_.size());
    state["info:reached_box"_] = reached_box_;
    state["info:previously_gripped"_] = previously_gripped_;
    state["info:prev_reward"_] = prev_cartesian_reward_;
    state["info:current_pos"_].Assign(cartesian_current_pos_.data(),
                                      cartesian_current_pos_.size());
    state["info:prev_action"_].Assign(prev_cartesian_action_.data(),
                                      prev_cartesian_action_.size());
    state["info:no_soln"_] = no_soln_;
    state["info:steps"_] = done_ ? 0 : elapsed_step_;
#endif
  }
};

template <typename Spec, bool kFromPixels>
using PandaBase = PlaygroundPandaEnvBase<Spec, kFromPixels>;
using PandaEnv = PandaBase<PlaygroundPandaEnvSpec, false>;
using PandaPixelEnv = PandaBase<PlaygroundPandaPixelEnvSpec, true>;
using PlaygroundPandaEnv = PandaEnv;
using PlaygroundPandaPixelEnv = PandaPixelEnv;
using PlaygroundPandaEnvPool = PlaygroundEnvPoolT<PlaygroundPandaEnv>;
using PlaygroundPandaPixelEnvPool = PlaygroundEnvPoolT<PlaygroundPandaPixelEnv>;

}  // namespace mujoco_playground

#endif  // ENVPOOL_MUJOCO_PLAYGROUND_PANDA_H_
