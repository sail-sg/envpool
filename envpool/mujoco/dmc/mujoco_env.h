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

#ifndef ENVPOOL_MUJOCO_DMC_MUJOCO_ENV_H_
#define ENVPOOL_MUJOCO_DMC_MUJOCO_ENV_H_

#include <mjxmacro.h>
#include <mujoco.h>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace mujoco {

/*
 * This class combines with dmc Task and Physics API.
 *
 * https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/rl/control.py
 * https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/base.py
 * https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py
 * https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/composer/task.py
 */
class MujocoEnv {
 private:
  std::array<char, 1000> error_;

 protected:
  std::unique_ptr<mjModel> model_;
  std::unique_ptr<mjData> data_;
  int n_sub_steps_, max_episode_steps_, elapsed_step_;
  float reward_, discount_;
  bool done_;

 public:
  MujocoEnv(const std::string& base_path, const std::string& raw_xml,
            int n_sub_steps, int max_episode_steps)
      : n_sub_steps_(n_sub_steps),
        max_episode_steps_(max_episode_steps),
        elapsed_step_(max_episode_steps + 1),
        done_(true) {
    // initialize vfs from common assets and raw xml
    // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/wrapper/core.py#L158
    // https://github.com/deepmind/mujoco/blob/main/python/mujoco/structs.cc
    // MjModelWrapper::LoadXML
    std::unique_ptr<mjVFS, void (*)(mjVFS*)> vfs(new mjVFS, [](mjVFS* vfs) {
      mj_deleteVFS(vfs);
      delete vfs;
    });
    mj_defaultVFS(vfs.get());
    // save raw_xml into vfs
    std::string model_filename("model_.xml");
    mj_makeEmptyFileVFS(vfs.get(), model_filename.c_str(), raw_xml.size());
    std::memcpy(vfs->filedata[vfs->nfile - 1], raw_xml.c_str(), raw_xml.size());
    // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/common/__init__.py#L28
    std::vector<std::string> common_assets_name({"./common/materials.xml",
                                                 "./common/skybox.xml",
                                                 "./common/visual.xml"});
    for (const auto& i : common_assets_name) {
      std::string filename = base_path + "/" + i;
      std::ifstream ifs(filename);
      std::stringstream buffer;
      buffer << ifs.rdbuf();
      std::string content = buffer.str();
      mj_makeEmptyFileVFS(vfs.get(), i.c_str(), content.size());
      std::memcpy(vfs->filedata[vfs->nfile - 1], content.c_str(),
                  content.size());
    }
    // create model and data
    model_ = std::make_unique<mjModel>(
        mj_loadXML(model_filename.c_str(), vfs.get(), error_.begin(), 1000),
        [](mjModel* model) {
          mj_deleteModel(model);
          delete mode;
        });
    data_ = std::make_unique<mjData>(mj_makeData(model_), [](mjData* data) {
      mj_deleteData(data);
      delete data;
    });
  }

  // rl control Environment
  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/rl/control.py#L77
  void ControlReset() {
    elapsed_step_ = 0;
    discount_ = 1.0;
    done_ = false;
    // attention: no keyframe_id
    PhysicsReset();  // first mj_forward
    TaskInitializeEpisode();
    PhysicsAfterReset();  // second mj_forward
  }

  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/rl/control.py#L94
  void ControlStep(const mjtNum* action) {
    TaskBeforeStep(action);
    PhysicsStep(n_sub_steps_);
    TaskAfterStep();
    reward_ = TaskGetReward();
    if (++elapsed_step_ >= max_episode_steps_) {
      discount_ = 1.0;
      done_ = true;
    } else if (TaskShouldTerminateEpisode()) {
      // copy from composer task
      // this is different because we don't use None in C++
      discount_ = 0.0;
      done_ = true;
    } else {
      discount_ = TaskGetDiscount();
      done_ = false;
    }
  }

  // Task
  void TaskInitializeEpisodeMjcf() {}
  void TaskInitializeEpisode() {}
  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/base.py#L73
  void TaskBeforeStep(const mjtNum* action) { PhysicsSetControl(action); }
  void TaskBeforeSubStep(const mjtNum* action) {}
  void TaskAfterStep() {}
  void TaskAftersSubStep() {}
  float TaskGetReward() {
    throw std::runtime_error("GetReward not implemented");
  }
  float TaskGetDiscount() { return 1.0; }
  bool TaskShouldTerminateEpisode() { return false; }

  // Physics
  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L263
  void PhysicsReset(int keyframe_id = -1) {
    if (keyframe_id < 0) {
      mj_resetData(model_.get(), data_.get());
    } else {
      // actually no one steps to this line
      assert(keyframe_id < model_->nkey);
      mj_resetDataKeyframe(model_.get(), data_.get(), keyframe_id);
    }

    // PhysicsAfterReset may be overwritten?
    int old_flags = model_->opt.disableflags;
    model_->opt.disableflags |= mjDSBL_ACTUATION;
    PhysicsForward();
    model_->opt.disableflags = old_flags;
  }

  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L286
  void PhysicsAfterReset() {
    int old_flags = model_->opt.disableflags;
    model_->opt.disableflags |= mjDSBL_ACTUATION;
    PhysicsForward();
    model_->opt.disableflags = old_flags;
  }

  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L138
  void PhysicsSetControl(const mjtNum* control) {
    std::memcpy(data_->ctrl, control, sizeof(mjtNum) * model_->nu);
  }

  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L292
  void PhysicsForward() { mj_forward(model_.get(), data_.get()); }

  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L146
  void PhysicsStep(int nstep) {
    if (model_->opt.integrator != mjINT_RK4) {
      mj_step2(model_.get(), data_.get());
      if (nstep > 1) {
        for (int i = 0; i < nstep - 1; ++i) {
          mj_step(model_.get(), data_.get());
        }
      }
    } else {
      for (int i = 0; i < nstep; ++i) {
        mj_step(model_.get(), data_.get());
      }
    }
    mj_step1(model_.get(), data_.get());
  }
};

}  // namespace mujoco

#endif  // ENVPOOL_MUJOCO_DMC_MUJOCO_ENV_H_
