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
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/fish.py

#ifndef ENVPOOL_MUJOCO_DMC_FISH_H_
#define ENVPOOL_MUJOCO_DMC_FISH_H_

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

std::string GetFishXML(const std::string& base_path,
                       const std::string& task_name_) {
  return GetFileContent(base_path, "fish.xml");
}

class FishEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.Bind(1000), "frame_skip"_.Bind(10),
                    "task_name"_.Bind(std::string("upright")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs:joint_angles"_.Bind(Spec<mjtNum>({7})),
                    "obs:upright"_.Bind(Spec<mjtNum>({})),
                    "obs:velocity"_.Bind(Spec<mjtNum>({13})),
                    "obs:target"_.Bind(Spec<mjtNum>({3})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.Bind(Spec<mjtNum>({14})),
                    "info:target0"_.Bind(Spec<mjtNum>({3})),
#endif
                    "discount"_.Bind(Spec<float>({-1}, {0.0, 1.0})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 5}, {-1.0, 1.0})));
  }
};

using FishEnvSpec = EnvSpec<FishEnvFns>;

class FishEnv : public Env<FishEnvSpec>, public MujocoEnv {
 protected:
  int id_mouth_;
  int id_qpos_finleft_pitch_;
  int id_qpos_finleft_roll_;
  int id_qpos_finright_pitch_;
  int id_qpos_finright_roll_;
  int id_qpos_root_;
  int id_qpos_tail1_;
  int id_qpos_tail2_;
  int id_qpos_tail_twist_;
  int id_qvel_finleft_pitch_;
  int id_qvel_finleft_roll_;
  int id_qvel_finright_pitch_;
  int id_qvel_finright_roll_;
  int id_qvel_tail1_;
  int id_qvel_tail2_;
  int id_qvel_tail_twist_;
  int id_torso_;
  int id_target_;
  std::normal_distribution<> dist_normal_;
  std::uniform_real_distribution<> dist_uniform_;
  bool is_swim_;
#ifdef ENVPOOL_TEST
  std::array<mjtNum, 3> target0_;
#endif

 public:
  FishEnv(const Spec& spec, int env_id)
      : Env<FishEnvSpec>(spec, env_id),
        MujocoEnv(
            spec.config["base_path"_],
            GetFishXML(spec.config["base_path"_], spec.config["task_name"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_]),
        id_torso_(mj_name2id(model_, mjOBJ_XBODY, "torso")),
        id_target_(mj_name2id(model_, mjOBJ_GEOM, "target")),
        id_mouth_(mj_name2id(model_, mjOBJ_GEOM, "mouth")),
        // qpos
        id_qpos_root_(GetQposId(model_, "root")),
        id_qpos_tail1_(GetQposId(model_, "tail1")),
        id_qpos_tail_twist_(GetQposId(model_, "tail_twist")),
        id_qpos_tail2_(GetQposId(model_, "tail2")),
        id_qpos_finright_roll_(GetQposId(model_, "finright_roll")),
        id_qpos_finright_pitch_(GetQposId(model_, "finright_pitch")),
        id_qpos_finleft_roll_(GetQposId(model_, "finleft_roll")),
        id_qpos_finleft_pitch_(GetQposId(model_, "finleft_pitch")),
        // qvel
        id_qvel_tail1_(GetQvelId(model_, "tail1")),
        id_qvel_tail_twist_(GetQvelId(model_, "tail_twist")),
        id_qvel_tail2_(GetQvelId(model_, "tail2")),
        id_qvel_finright_roll_(GetQvelId(model_, "finright_roll")),
        id_qvel_finright_pitch_(GetQvelId(model_, "finright_pitch")),
        id_qvel_finleft_roll_(GetQvelId(model_, "finleft_roll")),
        id_qvel_finleft_pitch_(GetQvelId(model_, "finleft_pitch")),
        dist_normal_(0, 1),
        dist_uniform_(0, 1),
        is_swim_(spec.config["task_name"_] == "swim") {
    const std::string& task_name = spec.config["task_name"_];
    if (task_name != "upright" && task_name != "swim") {
      throw std::runtime_error("Unknown task_name " + task_name +
                               " for dmc fish.");
    }
  }

  void TaskInitializeEpisode() override {
    // quat = self.random.randn(4)
    // physics.named.data.qpos['root'][3:7] = quat / np.linalg.norm(quat)
    std::array<mjtNum, 4> quat = {dist_normal_(gen_), dist_normal_(gen_),
                                  dist_normal_(gen_), dist_normal_(gen_)};
    mjtNum quat_norm = std::sqrt(quat[0] * quat[0] + quat[1] * quat[1] +
                                 quat[2] * quat[2] + quat[3] * quat[3]);
    for (int i = 0; i < 4; ++i) {
      data_->qpos[id_qpos_root_ + 3] = quat[i] / quat_norm;
    }
    // for joint in _JOINTS:
    //   physics.named.data.qpos[joint] = self.random.uniform(-.2, .2)
    data_->qpos[id_qpos_tail1_] = dist_uniform_(gen_) * 0.4 - 0.2;
    data_->qpos[id_qpos_tail_twist_] = dist_uniform_(gen_) * 0.4 - 0.2;
    data_->qpos[id_qpos_tail2_] = dist_uniform_(gen_) * 0.4 - 0.2;
    data_->qpos[id_qpos_finright_roll_] = dist_uniform_(gen_) * 0.4 - 0.2;
    data_->qpos[id_qpos_finright_pitch_] = dist_uniform_(gen_) * 0.4 - 0.2;
    data_->qpos[id_qpos_finleft_roll_] = dist_uniform_(gen_) * 0.4 - 0.2;
    data_->qpos[id_qpos_finleft_pitch_] = dist_uniform_(gen_) * 0.4 - 0.2;
    if (is_swim_) {
      // Randomize target position.
      // physics.named.model.geom_pos['target', 'x'] = self.random.uniform(-.4,
      // .4) physics.named.model.geom_pos['target', 'y'] =
      // self.random.uniform(-.4, .4) physics.named.model.geom_pos['target',
      // 'z'] = self.random.uniform(.1, .3)
      mjtNum target_x = dist_uniform_(gen_) * 0.8 - 0.4;
      mjtNum target_y = dist_uniform_(gen_) * 0.8 - 0.4;
      mjtNum target_z = dist_uniform_(gen_) * 0.2 + 0.1;
      model_->geom_pos[id_target_ * 3] = target_x;
      model_->geom_pos[id_target_ * 3 + 1] = target_y;
      model_->geom_pos[id_target_ * 3 + 2] = target_z;
    } else {
      // Hide the target. It's irrelevant for this task.
      // physics.named.model.geom_rgba['target', 3] = 0
      model_->geom_rgba[id_target_ * 3 + 3] = 0;
    }
#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_.get(), data_->qpos, sizeof(mjtNum) * model_->nq);
    target0_[0] = model_->geom_pos[id_target_ * 3 + 0];
    target0_[1] = model_->geom_pos[id_target_ * 3 + 1];
    target0_[2] = model_->geom_pos[id_target_ * 3 + 2];
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
    if (is_swim_) {
      return static_cast<float>(RewardTolerance(UpRight(), 1.0, 1.0, 1.0));
    }
    mjtNum radii = model_->geom_size[id_mouth_] + model_->geom_size[id_target_];
    const auto& target = MouthToTarget();
    auto target_norm = std::sqrt(target[0] * target[0] + target[1] * target[1] +
                                 target[2] * target[2]);
    auto in_target = RewardTolerance(target_norm, 0.0, radii, 2 * radii);
    auto is_upright = 0.5 * (UpRight() + 1.0);
    return static_cast<float>((7 * in_target + is_upright) / 8);
  }

  bool TaskShouldTerminateEpisode() override { return false; }

 private:
  void WriteState() {
    State state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    // obs
    const auto& joint_angles = JointAngles();
    state["obs:joint_angles"_].Assign(joint_angles.begin(),
                                      joint_angles.size());
    state["obs:upright"_] = UpRight();
    state["obs:velocity"_].Assign(data_->qvel, model_->nv);
    if (is_swim_) {
      const auto& target = MouthToTarget();
      state["obs:target"_].Assign(target.begin(), target.size());
    }
    // info
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
    state["info:target0"_].Assign(target0_.begin(), target0_.size());
#endif
  }
  mjtNum UpRight() {
    // return self.named.data.xmat['torso', 'zz']
    return data_->xmat[id_torso_ * 9 + 8];
  }
  std::array<mjtNum, 6> TorsoVelocity() {
    // return self.data.sensordata
    return {data_->sensordata[0], data_->sensordata[1], data_->sensordata[2],
            data_->sensordata[3], data_->sensordata[4], data_->sensordata[5]};
  }
  std::array<mjtNum, 7> JointVelocities() {
    // return self.named.data.qvel[_JOINTS]
    return {data_->qvel[id_qvel_tail1_],
            data_->qvel[id_qvel_tail_twist_],
            data_->qvel[id_qvel_tail2_],
            data_->qvel[id_qvel_finright_roll_],
            data_->qvel[id_qvel_finright_pitch_],
            data_->qvel[id_qvel_finleft_roll_],
            data_->qvel[id_qvel_finleft_pitch_]};
  }
  std::array<mjtNum, 7> JointAngles() {
    // return self.named.data.qpos[_JOINTS]
    return {data_->qpos[id_qpos_tail1_],
            data_->qpos[id_qpos_tail_twist_],
            data_->qpos[id_qpos_tail2_],
            data_->qpos[id_qpos_finright_roll_],
            data_->qpos[id_qpos_finright_pitch_],
            data_->qpos[id_qpos_finleft_roll_],
            data_->qpos[id_qpos_finleft_pitch_]};
  }
  std::array<mjtNum, 3> MouthToTarget() {
    // returns a vector, from mouth to target in local coordinate of mouth.
    // mouth_to_target_global = data.geom_xpos['target'] -
    // data.geom_xpos['mouth'] return
    // mouth_to_target_global.dot(data.geom_xmat['mouth'].reshape(3, 3))
    std::array<mjtNum, 3> mouth_to_target_global;
    for (int i = 0; i < 3; i++) {
      mouth_to_target_global[i] = (data_->geom_xpos[id_target_ * 3 + i] -
                                   data_->geom_xpos[id_mouth_ * 3 + i]);
    }
    std::array<mjtNum, 3> mouth_to_target;
    for (int i = 0; i < 3; i++) {
      mouth_to_target[i] =
          mouth_to_target_global[0] * data_->geom_xpos[id_mouth_ * 9 + i] +
          mouth_to_target_global[1] * data_->geom_xpos[id_mouth_ * 9 + i + 3] +
          mouth_to_target_global[2] * data_->geom_xpos[id_mouth_ * 9 + i + 6];
    }
    return mouth_to_target;
  }
};

using FishEnvPool = AsyncEnvPool<FishEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_FISH_H_
