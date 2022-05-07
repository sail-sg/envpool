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

#ifndef ENVPOOL_MUJOCO_MUJOCO_TASK_H_
#define ENVPOOL_MUJOCO_MUJOCO_TASK_H_

#include <mjxmacro.h>
#include <mujoco.h>

#include <string>

class MujocoTask {
 private:
  std::array<char, 1000> error_;

 protected:
  mjModel* model_;
  mjData* data_;

 public:
  MujocoTask(const std::string& xml, int frame_skip, bool post_constraint,
             int max_episode_steps)
      : model_(mj_loadXML(xml.c_str(), nullptr, error_.begin(), 1000)),
        data_(mj_makeData(model_)),
  {
    memcpy(init_qpos_, data_->qpos, sizeof(mjtNum) * model_->nq);
    memcpy(init_qvel_, data_->qvel, sizeof(mjtNum) * model_->nv);
  }

  ~MujocoTask() {
    mj_deleteData(data_);
    mj_deleteModel(model_);
    delete[] init_qpos_;
    delete[] init_qvel_;
    delete[] qpos0_;
    delete[] qvel0_;
  }

  void MujocoInitializeEpisodeMjcf() {}

  void MujocoInitializeEpisode() {}

  void MujocoBeforeStep(const mjtNum* action) {}

  void MujocoBeforeSubStep(const mjtNum* action) {}

  void MujocoAfterStep() {}

  void MujocoAftersSubStep() {}

  float MujocoGetReward() { return 0.0; }

  float MujocoGetDiscount() { return 1.0; }

  bool MujocoShouldTerminateEpisode() { return false; }
};

#endif  // ENVPOOL_MUJOCO_MUJOCO_TASK_H_
