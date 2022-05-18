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
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/point_mass.py

#ifndef ENVPOOL_MUJOCO_DMC_POINT_MASS_H_
#define ENVPOOL_MUJOCO_DMC_POINT_MASS_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/dmc/mujoco_env.h"
#include "envpool/mujoco/dmc/utils.h"

namespace mujoco_dmc {

std::string GetPointMassXML(const std::string& base_path,
                            const std::string& task_name) {
  return GetFileContent(base_path, "point_mass.xml");
}

class PointMassEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("max_episode_steps"_.Bind(1000), "frame_skip"_.Bind(1),
                    "task_name"_.Bind(std::string("easy")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs:position"_.Bind(Spec<mjtNum>({2})),
                    "obs:velocity"_.Bind(Spec<mjtNum>({2})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.Bind(Spec<mjtNum>({2})),
                    "info:wrap_prm"_.Bind(Spec<mjtNum>({4})),
#endif
                    "discount"_.Bind(Spec<float>({-1}, {0.0, 1.0})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 2}, {-1.0, 1.0})));
  }
};

using PointMassEnvSpec = EnvSpec<PointMassEnvFns>;

class PointMassEnv : public Env<PointMassEnvSpec>, public MujocoEnv {
 protected:
  bool randomize_gains_;
  int id_geom_target_, id_geom_pointmass_;
  std::normal_distribution<> dist_normal_;
#ifdef ENVPOOL_TEST
  std::unique_ptr<mjtNum> wrap_prm_;
#endif

 public:
  PointMassEnv(const Spec& spec, int env_id)
      : Env<PointMassEnvSpec>(spec, env_id),
        MujocoEnv(spec.config["base_path"_],
                  GetPointMassXML(spec.config["base_path"_],
                                  spec.config["task_name"_]),
                  spec.config["frame_skip"_],
                  spec.config["max_episode_steps"_]),

        id_geom_target_(mj_name2id(model_, mjOBJ_GEOM, "target")),
        id_geom_pointmass_(mj_name2id(model_, mjOBJ_GEOM, "pointmass")),
        dist_normal_(0, 1) {
    const std::string& task_name = spec.config["task_name"_];
    if (task_name == "easy") {
      randomize_gains_ = false;
    } else if (task_name == "hard") {
      randomize_gains_ = true;
    } else {
      throw std::runtime_error("Unknown task_name " + task_name +
                               " for dmc point_mass.");
    }
#ifdef ENVPOOL_TEST
    wrap_prm_.reset(new mjtNum[model_->nwrap]);
#endif
  }

  void TaskInitializeEpisode() override {
    RandomizeLimitedAndRotationalJoints(&gen_);
    if (randomize_gains_) {
      const auto& dir1 = GetDir();
      bool parallel = true;
      std::array<mjtNum, 2> dir2;
      while (parallel) {
        dir2 = GetDir();
        parallel = std::abs(dir1[0] * dir2[0] + dir1[1] * dir2[1]) > 0.9;
      }
      model_->wrap_prm[0] = dir1[0];
      model_->wrap_prm[1] = dir1[1];
      model_->wrap_prm[2] = dir2[0];
      model_->wrap_prm[3] = dir2[1];
    }
#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_.get(), data_->qpos, sizeof(mjtNum) * model_->nq);
    std::memcpy(wrap_prm_.get(), model_->wrap_prm,
                sizeof(mjtNum) * model_->nwrap);
#endif
  }

  std::array<mjtNum, 2> GetDir() {
    std::array<mjtNum, 2> dir = {dist_normal_(gen_), dist_normal_(gen_)};
    mjtNum norm_of_dir = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1]);
    return {dir[0] / norm_of_dir, dir[1] / norm_of_dir};
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
    mjtNum target_size = model_->geom_size[id_geom_target_ * 3 + 0];
    mjtNum near_target =
        RewardTolerance(MassToTargetDist(), 0, target_size, target_size);
    mjtNum control_reward =
        (RewardTolerance(data_->ctrl[0], 0, 0, 1, 0, SigmoidType::kQuadratic) +
         RewardTolerance(data_->ctrl[1], 0, 0, 1, 0, SigmoidType::kQuadratic)) /
        2;
    mjtNum small_control = (control_reward + 4) / 5;
    return static_cast<float>(near_target * small_control);
  }

  bool TaskShouldTerminateEpisode() override { return false; }

 private:
  mjtNum MassToTargetDist() {
    // return np.linalg.norm(self.mass_to_target())
    const auto& mass_to_target = MassToTarget();
    return std::sqrt(mass_to_target[0] * mass_to_target[0] +
                     mass_to_target[1] * mass_to_target[1] +
                     mass_to_target[2] * mass_to_target[2]);
  }

  std::array<mjtNum, 3> MassToTarget() {
    // return (self.named.data.geom_xpos['target'] -
    //         self.named.data.geom_xpos['pointmass'])
    return {data_->geom_xpos[id_geom_target_ * 3 + 0] -
                data_->geom_xpos[id_geom_pointmass_ * 3 + 0],
            data_->geom_xpos[id_geom_target_ * 3 + 1] -
                data_->geom_xpos[id_geom_pointmass_ * 3 + 1],
            data_->geom_xpos[id_geom_target_ * 3 + 2] -
                data_->geom_xpos[id_geom_pointmass_ * 3 + 2]};
  }

  void WriteState() {
    State state = Allocate();
    state["reward"_] = reward_;
    state["discount"_] = discount_;
    // obs
    state["obs:position"_].Assign(data_->qpos, model_->nq);
    state["obs:velocity"_].Assign(data_->qvel, model_->nv);
    // info for check alignment
#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_.get(), model_->nq);
    state["info:wrap_prm"_].Assign(wrap_prm_.get(), model_->nwrap);
#endif
  }
};

using PointMassEnvPool = AsyncEnvPool<PointMassEnv>;

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_POINT_MASS_H_
