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

#ifndef ENVPOOL_MUJOCO_MUJOCO_ENV_H_
#define ENVPOOL_MUJOCO_MUJOCO_ENV_H_

#include <mjxmacro.h>
#include <mujoco.h>

#include <string>

class MujocoEnv {
 private:
  std::array<char, 1000> error_;

 protected:
  mjModel* model_;
  mjData* data_;
  mjtNum *init_qpos_, *init_qvel_;
  mjtNum *qpos0_, *qvel0_;  // for align check
  int frame_skip_;
  bool post_constraint_;
  int max_episode_steps_, elapsed_step_;
  bool done_;

 public:
  MujocoEnv(const std::string& xml, int frame_skip, bool post_constraint,
            int max_episode_steps)
      : model_(mj_loadXML(xml.c_str(), nullptr, error_.begin(), 1000)),
        data_(mj_makeData(model_)),
        init_qpos_(new mjtNum[model_->nq]),
        init_qvel_(new mjtNum[model_->nv]),
        qpos0_(new mjtNum[model_->nq]),
        qvel0_(new mjtNum[model_->nv]),
        frame_skip_(frame_skip),
        post_constraint_(post_constraint),
        max_episode_steps_(max_episode_steps),
        elapsed_step_(max_episode_steps + 1),
        done_(true) {
    memcpy(init_qpos_, data_->qpos, sizeof(mjtNum) * model_->nq);
    memcpy(init_qvel_, data_->qvel, sizeof(mjtNum) * model_->nv);
  }

  ~MujocoEnv() {
    mj_deleteData(data_);
    mj_deleteModel(model_);
    delete[] init_qpos_;
    delete[] init_qvel_;
    delete[] qpos0_;
    delete[] qvel0_;
  }

  void MujocoReset() {
    mj_resetData(model_, data_);
    MujocoResetModel();
    mj_forward(model_, data_);
  }

  virtual void MujocoResetModel() {
    throw std::runtime_error("reset_model not implemented");
  }

  void MujocoStep(const mjtNum* action) {
    for (int i = 0; i < model_->nu; ++i) {
      data_->ctrl[i] = action[i];
    }
    for (int i = 0; i < frame_skip_; ++i) {
      mj_step(model_, data_);
    }
    if (post_constraint_) {
      mj_rnePostConstraint(model_, data_);
    }
  }
};

#endif  // ENVPOOL_MUJOCO_MUJOCO_ENV_H_
