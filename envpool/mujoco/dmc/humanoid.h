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
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/humanoid.py

#ifndef ENVPOOL_MUJOCO_DMC_HUMANOID_H_
#define ENVPOOL_MUJOCO_DMC_HUMANOID_H_

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

std::string GetHumanoidXML(const std::string& base_path,
                           const std::string& task_name) {
  return GetFileContent(base_path, "humanoid.xml");
}

class HumanoidEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("frame_skip"_.Bind(5),
                    "task_name"_.Bind(std::string("stand")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs:joint_angles"_.Bind(Spec<mjtNum>({21})),
                    "obs:head_height"_.Bind(Spec<mjtNum>({})),
                    "obs:extremities"_.Bind(Spec<mjtNum>({12})),
                    "obs:torso_vertical"_.Bind(Spec<mjtNum>({3})),
                    "obs:com_velocity"_.Bind(Spec<mjtNum>({3})),
                    "obs:position"_.Bind(Spec<mjtNum>({28})),
                    "obs:velocity"_.Bind(Spec<mjtNum>({27}))
#ifdef ENVPOOL_TEST
                        ,
                    "info:qpos0"_.Bind(Spec<mjtNum>({28}))
#endif
    );  // NOLINT
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 21}, {-1.0, 1.0})));
  }
};

using HumanoidEnvSpec = EnvSpec<HumanoidEnvFns>;

class HumanoidEnv : public Env<HumanoidEnvSpec>, public MujocoEnv {
 protected:
  // Height of head above which stand reward is 1.
  const mjtNum kStandHeight = 1.4;
  // Horizontal speeds above which move reward is 1.
  const mjtNum kWalkSpeed = 1;
  const mjtNum kRunSpeed = 10;
  int id_head_;
  int id_left_hand_;
  int id_left_foot_;
  int id_right_hand_;
  int id_right_foot_;
  int id_torso_;
  int id_torso_subtreelinvel_;
  mjtNum move_speed_;
  bool is_pure_state_;

 public:
  HumanoidEnv(const Spec& spec, int env_id)
      : Env<HumanoidEnvSpec>(spec, env_id),
        MujocoEnv(spec.config["base_path"_],
                  GetHumanoidXML(spec.config["base_path"_],
                                 spec.config["task_name"_]),
                  spec.config["frame_skip"_],
                  spec.config["max_episode_steps"_]),
        id_head_(mj_name2id(model_, mjOBJ_XBODY, "head")),
        id_left_hand_(mj_name2id(model_, mjOBJ_XBODY, "left_hand")),
        id_left_foot_(mj_name2id(model_, mjOBJ_XBODY, "left_foot")),
        id_right_hand_(mj_name2id(model_, mjOBJ_XBODY, "right_hand")),
        id_right_foot_(mj_name2id(model_, mjOBJ_XBODY, "right_foot")),
        id_torso_(mj_name2id(model_, mjOBJ_XBODY, "torso")),
        id_torso_subtreelinvel_(GetSensorId(model_, "torso_subtreelinvel")),
        is_pure_state_(spec.config["task_name"_] == "run_pure_state") {
    const std::string& task_name = spec.config["task_name"_];
    if (task_name == "stand") {
      move_speed_ = 0;
    } else if (task_name == "walk") {
      move_speed_ = kWalkSpeed;
    } else if (task_name == "run" || task_name == "run_pure_state") {
      move_speed_ = kRunSpeed;
    } else {
      throw std::runtime_error("Unknown task_name " + task_name +
                               " for dmc humanoid.");
    }
  }

  void TaskInitializeEpisode() override {
    while (true) {
      // Find a collision-free random initial configuration.
      // randomizers.randomize_limited_and_rotational_joints(physics,
      // self.random)
      RandomizeLimitedAndRotationalJoints(&gen_);
#ifdef ENVPOOL_TEST
      std::memcpy(qpos0_.get(), data_->qpos, sizeof(mjtNum) * model_->nq);
#endif
      PhysicsAfterReset();
      if (data_->ncon <= 0) {
        break;
      }
    }
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
    auto standing = RewardTolerance(HeadHeight(), kStandHeight,
                                    std::numeric_limits<double>::infinity(),
                                    kStandHeight / 4);
    auto upright = RewardTolerance(TorsoUpright(), 0.9,
                                   std::numeric_limits<double>::infinity(), 1.9,
                                   0.0, SigmoidType::kLinear);
    auto stand_reward = standing * upright;
    mjtNum small_control = 0.0;
    for (int i = 0; i < model_->nu; ++i) {
      small_control += RewardTolerance(data_->ctrl[i], 0.0, 0.0, 1.0, 0.0,
                                       SigmoidType::kQuadratic);
    }
    small_control = (small_control / model_->nu + 4.0) / 5.0;
    auto center_of_mass_velocity = CenterOfMassVelocity();
    if (move_speed_ == 0) {
      mjtNum dont_move = 0.0;
      for (int i = 0; i < 2; ++i) {
        dont_move +=
            0.5 * RewardTolerance(center_of_mass_velocity[i], 0.0, 0.0, 2.0);
      }
      return static_cast<float>(small_control * stand_reward * dont_move);
    }
    auto com_velocity =
        std::sqrt(center_of_mass_velocity[0] * center_of_mass_velocity[0] +
                  center_of_mass_velocity[1] * center_of_mass_velocity[1]);
    auto move = RewardTolerance(com_velocity, move_speed_,
                                std::numeric_limits<double>::infinity(),
                                move_speed_, 0.0, SigmoidType::kLinear);
    move = (5.0 * move + 1.0) / 6.0;
    return static_cast<float>(small_control * stand_reward * move);
  }

  bool TaskShouldTerminateEpisode() override { return false; }

 private:
  void WriteState() {
    State state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    // obs
    const auto& joint_angles = JointAngles();
    const auto& extremities = Extremities();
    const auto& com_velocity = CenterOfMassVelocity();
    const auto& torso_vertical = TorsoVerticalOrientation();
    state["obs:velocity"_].Assign(data_->qvel, model_->nv);
    if (is_pure_state_) {
      state["obs:position"_].Assign(data_->qpos, model_->nq);
    } else {
      state["obs:joint_angles"_].Assign(joint_angles.begin(),
                                        joint_angles.size());
      state["obs:head_height"_] = HeadHeight();
      state["obs:extremities"_].Assign(extremities.begin(), extremities.size());
      state["obs:torso_vertical"_].Assign(torso_vertical.begin(),
                                          torso_vertical.size());
      state["obs:com_velocity"_].Assign(com_velocity.begin(),
                                        com_velocity.size());
    }
    // info
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
#endif
  }

  mjtNum TorsoUpright() {
    // return self.named.data.xmat['torso', 'zz']
    return data_->xmat[id_torso_ * 9 + 8];
  }
  mjtNum HeadHeight() {
    // return self.named.data.xpos['head', 'z']
    return data_->xpos[id_head_ * 3 + 2];
  }
  std::array<mjtNum, 3> CenterOfMassPosition() {
    // return self.named.data.subtree_com['torso'].copy()
    return {
        data_->subtree_com[id_torso_ * 3],
        data_->subtree_com[id_torso_ * 3 + 1],
        data_->subtree_com[id_torso_ * 3 + 2],
    };
  }
  std::array<mjtNum, 3> CenterOfMassVelocity() {
    // return self.named.data.sensordata['torso_subtreelinvel'].copy()
    return {
        data_->sensordata[id_torso_subtreelinvel_ * 3],
        data_->sensordata[id_torso_subtreelinvel_ * 3 + 1],
        data_->sensordata[id_torso_subtreelinvel_ * 3 + 2],
    };
  }
  std::array<mjtNum, 3> TorsoVerticalOrientation() {
    // return self.named.data.xmat['torso', ['zx', 'zy', 'zz']]
    return {
        data_->xmat[id_torso_ * 9 + 6],
        data_->xmat[id_torso_ * 9 + 7],
        data_->xmat[id_torso_ * 9 + 8],
    };
  }
  std::array<mjtNum, 12> Extremities() {
    // returns end effector positions in egocentric frame.
    // torso_frame = self.named.data.xmat['torso'].reshape(3, 3)
    // torso_pos = self.named.data.xpos['torso']
    // positions = []
    // for side in ('left_', 'right_'):
    //   for limb in ('hand', 'foot'):
    //     torso_to_limb = self.named.data.xpos[side + limb] - torso_pos
    //     positions.append(torso_to_limb.dot(torso_frame))
    // return np.hstack(positions)
    std::array<mjtNum, 9> torso_frame;
    for (int i = 0; i < 9; i++) {
      torso_frame[i] = data_->xmat[id_torso_ * 9 + i];
    }
    std::array<mjtNum, 3> torso_pos;
    for (int i = 0; i < 3; i++) {
      torso_pos[i] = data_->xpos[id_torso_ * 3 + i];
    }
    // left hand
    std::array<mjtNum, 3> torso_to_limb_lh;
    for (int i = 0; i < 3; i++) {
      torso_to_limb_lh[i] =
          (data_->xpos[id_left_hand_ * 3] - torso_pos[0]) * torso_frame[i] +
          (data_->xpos[id_left_hand_ * 3 + 1] - torso_pos[1]) *
              torso_frame[i + 3] +
          (data_->xpos[id_left_hand_ * 3 + 2] - torso_pos[2]) *
              torso_frame[i + 6];
    }
    // left foot
    std::array<mjtNum, 3> torso_to_limb_lf;
    for (int i = 0; i < 3; i++) {
      torso_to_limb_lf[i] =
          (data_->xpos[id_left_foot_ * 3] - torso_pos[0]) * torso_frame[i] +
          (data_->xpos[id_left_foot_ * 3 + 1] - torso_pos[1]) *
              torso_frame[i + 3] +
          (data_->xpos[id_left_foot_ * 3 + 2] - torso_pos[2]) *
              torso_frame[i + 6];
    }
    // right hand
    std::array<mjtNum, 3> torso_to_limb_rh;
    for (int i = 0; i < 3; i++) {
      torso_to_limb_rh[i] =
          (data_->xpos[id_right_hand_ * 3] - torso_pos[0]) * torso_frame[i] +
          (data_->xpos[id_right_hand_ * 3 + 1] - torso_pos[1]) *
              torso_frame[i + 3] +
          (data_->xpos[id_right_hand_ * 3 + 2] - torso_pos[2]) *
              torso_frame[i + 6];
    }
    // right foot
    std::array<mjtNum, 3> torso_to_limb_rf;
    for (int i = 0; i < 3; i++) {
      torso_to_limb_rf[i] =
          (data_->xpos[id_right_foot_ * 3] - torso_pos[0]) * torso_frame[i] +
          (data_->xpos[id_right_foot_ * 3 + 1] - torso_pos[1]) *
              torso_frame[i + 3] +
          (data_->xpos[id_right_foot_ * 3 + 2] - torso_pos[2]) *
              torso_frame[i + 6];
    }
    return {
        torso_to_limb_lh[0], torso_to_limb_lh[1], torso_to_limb_lh[2],
        torso_to_limb_lf[0], torso_to_limb_lf[1], torso_to_limb_lf[2],
        torso_to_limb_rh[0], torso_to_limb_rh[1], torso_to_limb_rh[2],
        torso_to_limb_rf[0], torso_to_limb_rf[1], torso_to_limb_rf[2],
    };
  }
  std::array<mjtNum, 21> JointAngles() {
    // return self.data.qpos[7:].copy()
    std::array<mjtNum, 21> joint_angles;
    for (int i = 0; i < 21; i++) {
      joint_angles[i] = data_->qpos[7 + i];
    }
    return joint_angles;
  }
};

using HumanoidEnvPool = AsyncEnvPool<HumanoidEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_HUMANOID_H_
