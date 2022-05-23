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
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/swimmer.py

#ifndef ENVPOOL_MUJOCO_DMC_SWIMMER_H_
#define ENVPOOL_MUJOCO_DMC_SWIMMER_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <regex>
#include <set>
#include <string>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/dmc/mujoco_env.h"
#include "envpool/mujoco/dmc/utils.h"

namespace mujoco_dmc {

std::string GetSwimmerXML(const std::string& base_path,
                          const std::string& task_name) {
  auto content = GetFileContent(base_path, "swimmer.xml");
  if (task_name == "swimmer6") {
    return XMLMakeSwimmer(content, 6);
  }
  if (task_name == "swimmer15") {
    return XMLMakeSwimmer(content, 15);
  }
}

class SwimmerEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.Bind(1000), "frame_skip"_.Bind(15),
                    "task_name"_.Bind(std::string("swimmer6")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    const std::string task_name = conf["task_name"_];
    int n_joints;
    if (task_name == "swimmer6") {
      n_joints = 6;
    } else if (task_name == "swimmer15") {
      n_joints = 15;
    } else {
      throw std::runtime_error("Unknown task_name " + task_name +
                               " for dmc cartpole.");
    }
    return MakeDict("obs:joints"_.Bind(Spec<mjtNum>({n_joints - 1})),
                    "obs:to_target"_.Bind(Spec<mjtNum>({2})),
                    "obs:body_velocities"_.Bind(Spec<mjtNum>({3 * n_joints})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.Bind(Spec<mjtNum>({n_joints + 2})),
                    "info:target0"_.Bind(Spec<mjtNum>({2})),
#endif
                    "discount"_.Bind(Spec<float>({-1}, {0.0, 1.0})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    const std::string task_name = conf["task_name"_];
    int n_joints;
    if (task_name == "swimmer6") {
      n_joints = 6;
    } else if (task_name == "swimmer15") {
      n_joints = 15;
    } else {
      throw std::runtime_error("Unknown task_name " + task_name +
                               " for dmc cartpole.");
    }
    return MakeDict(
        "action"_.Bind(Spec<mjtNum>({-1, n_joints - 1}, {-1.0, 1.0})));
  }
};

using SwimmerEnvSpec = EnvSpec<SwimmerEnvFns>;

class SwimmerEnv : public Env<SwimmerEnvSpec>, public MujocoEnv {
 protected:
  int id_nose_, id_head_;
  int id_target_, id_target_light_;
#ifdef ENVPOOL_TEST
  std::array<mjtNum, 2> target0_;
#endif

 public:
  SwimmerEnv(const Spec& spec, int env_id)
      : Env<SwimmerEnvSpec>(spec, env_id),
        MujocoEnv(
            spec.config["base_path"_],
            GetSwimmerXML(spec.config["base_path"_], spec.config["task_name"_]),
            spec.config["frame_skip"_], spec.config["max_episode_steps"_]),
        id_nose_(mj_name2id(model_, mjOBJ_GEOM, "nose")),
        id_head_(mj_name2id(model_, mjOBJ_XBODY, "head")),
        id_target_(mj_name2id(model_, mjOBJ_GEOM, "target")),
        id_target_(mj_name2id(model_, mjOBJ_GEOM, "target_light")) {
    const std::string& task_name = spec.config["task_name"_];
    if (task_name != "swimmer6" && task_name != "swimmer15") {
      throw std::runtime_error("Unknown task_name " + task_name +
                               " for dmc swimmer.");
    }
  }

  void TaskInitializeEpisode() override {
    RandomizeLimitedAndRotationalJoints(&gen_);
    bool close_target = RandNormal(0, 1) < 0.2;
    mjtNum target_box = 2;
    if (close_target) {
      target_box = 0.3;
    }
    mjtNum x_pos = RandUniform(-target_box, target_box)(gen_);
    mjtNum y_pos = RandUniform(-target_box, target_box)(gen_);
    // physics.named.model.geom_pos['target', 'x'] = xpos
    // physics.named.model.geom_pos['target', 'y'] = ypos
    model_->geom_pos[id_target_ * 3 + 0] = x_pos;
    model_->geom_pos[id_target_ * 3 + 1] = y_pos;
    // physics.named.model.light_pos['target_light', 'x'] = xpos
    // physics.named.model.light_pos['target_light', 'y'] = ypos
    model_->light_pos[id_target_light_ * 3 + 0] = x_pos;
    model_->light_pos[id_target_light_ * 3 + 1] = y_pos;
#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_.get(), data_->qpos, sizeof(mjtNum) * model_->nq);
    target0_[0] = model_->geom_pos[id_target_ * 3 + 0];
    target0_[1] = model_->geom_pos[id_target_ * 3 + 1];
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
    mjtNum target_size = model_->geom_size[id_target_ * 3];
    // const auto& target_size = NoseToTargetDist();
    return static_cast<float>(RewardTolerance(NoseToTargetDist(), 0.0,
                                              target_size, 5 * target_size, 0.1,
                                              SigmoidType::kLongTail));
  }

  bool TaskShouldTerminateEpisode() override { return false; }

 private:
  void WriteState() {
    State state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    // obs
    const auto& joints = Joints();
    state["obs:joints"_].Assign(joints.data(), joints.size());
    const auto& to_target = NoseToTarget();
    state["obs:to_target"_].Assign(to_target.begin(), to_target.size());
    const auto& body_velocities = BodyVelocities();
    state["obs:body_velocities"_].Assign(body_velocities.data(),
                                         body_velocities.size());
    // info
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
    state["info:target0"_].Assign(target0_.begin(), target0_.size());
#endif
  }

  std::array<mjtNum, 2> NoseToTarget() {
    // nose_to_target = (self.named.data.geom_xpos['target'] -
    //                   self.named.data.geom_xpos['nose'])
    std::array<mjtNum, 3> nose_to_target_global;
    for (int i = 0; i < 3; i++) {
      nose_to_target_global[i] = (data_->geom_xpos[id_target_ * 3 + i] -
                                  data_->geom_xpos[id_nose_ * 3 + i]);
    }
    // head_orientation = self.named.data.xmat['head'].reshape(3, 3)
    // return nose_to_target.dot(head_orientation)[:2]
    std::array<mjtNum, 3> nose_to_target;
    for (int i = 0; i < 3; i++) {
      nose_to_target[i] =
          nose_to_target_global[0] * data_->geom_xmat[id_head_ * 9 + i + 0] +
          nose_to_target_global[1] * data_->geom_xmat[id_head_ * 9 + i + 3] +
          nose_to_target_global[2] * data_->geom_xmat[id_head_ * 9 + i + 6];
    }
    return {nose_to_target[0], nose_to_target[1]};
  }
  mjtNum NoseToTargetDist() {
    // return np.linalg.norm(self.nose_to_target())
    const auto& nose_to_target = NoseToTarget();
    return std::sqrt(nose_to_target[0] * nose_to_target[0] +
                     nose_to_target[1] * nose_to_target[1] +
                     nose_to_target[2] * nose_to_target[2]);
  }
  std::vector<mjtNum> BodyVelocities() {
    // returns local body velocities: x,y linear, z rotational.
    std::vector<mjtNum> result;
    for (int i = 0; i < model_->nbody - 1; ++i) {
      result.emplace_back(data_->sensordata[i * 6 + 12 + 0]);
      result.emplace_back(data_->sensordata[i * 6 + 12 + 1]);
      result.emplace_back(data_->sensordata[i * 6 + 12 + 5]);
    }
    return result;
  }

  std::vector<mjtNum> Joints() {
    // return self.data.qpos[3:].copy()
    std::vector<mjtNum> result;
    for (int i = 3; i < model_->njnt; ++i) {
      result.emplace_back(data_->qpos[i]);
    }
    return result;
  }
};

using SwimmerEnvPool = AsyncEnvPool<SwimmerEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_SWIMMER_H_
