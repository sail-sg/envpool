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

#ifndef ENVPOOL_GYMNASIUM_ROBOTICS_MUJOCO_ENV_H_
#define ENVPOOL_GYMNASIUM_ROBOTICS_MUJOCO_ENV_H_

#include <mujoco.h>

#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/env.h"
#include "envpool/gymnasium_robotics/utils.h"
#include "envpool/mujoco/offscreen_renderer.h"

namespace gymnasium_robotics {

class MujocoRobotEnv : public RenderableEnv {
 protected:
  std::string model_path_;
  mjModel* model_;
  mjData* data_{nullptr};
  int frame_skip_;
  int max_episode_steps_;
  int elapsed_step_;
  bool done_{true};
  mjtNum initial_time_{0.0};
  std::vector<mjtNum> initial_qpos_;
  std::vector<mjtNum> initial_qvel_;
#ifdef ENVPOOL_TEST
  std::vector<mjtNum> qpos0_;
  std::vector<mjtNum> qvel0_;
#endif
  std::unique_ptr<envpool::mujoco::OffscreenRenderer> renderer_;

 public:
  MujocoRobotEnv(const std::string& base_path, const std::string& model_path,
                 int frame_skip, int max_episode_steps)
      : model_path_(ResolveModelPath(base_path, model_path)),
        model_(LoadModel(model_path_)),
        data_(mj_makeData(model_)),
        frame_skip_(frame_skip),
        max_episode_steps_(max_episode_steps),
        elapsed_step_(max_episode_steps + 1),
        initial_qpos_(model_->nq),
        initial_qvel_(model_->nv)
#ifdef ENVPOOL_TEST
        ,
        qpos0_(model_->nq),
        qvel0_(model_->nv)
#endif
  {}

  ~MujocoRobotEnv() override {
    mj_deleteData(data_);
    mj_deleteModel(model_);
  }

  void Render(int width, int height, int camera_id,
              unsigned char* rgb) override {
    RenderCallback();
    if (renderer_ == nullptr) {
      renderer_ = std::make_unique<envpool::mujoco::OffscreenRenderer>(
          envpool::mujoco::CameraPolicy::kGymLike);
    }
    renderer_->Render(model_, data_, width, height, camera_id, rgb);
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 480, height > 0 ? height : 480};
  }

 protected:
  static std::string ResolveModelPath(const std::string& base_path,
                                      const std::string& model_path) {
    if (!model_path.empty() && model_path[0] == '/') {
      return model_path;
    }
    return base_path + "/gymnasium_robotics/assets/" + model_path;
  }

  static mjModel* LoadModel(const std::string& xml_path) {
    std::array<char, 1000> error{};
    mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error.data(), 1000);
    if (model == nullptr) {
      throw std::runtime_error(error.data());
    }
    return model;
  }

  virtual void EnvSetup() {}
  virtual void StepCallback() {}
  virtual void RenderCallback() {}

  void InitializeRobotEnv() {
    initial_time_ = data_->time;
    std::memcpy(initial_qpos_.data(), data_->qpos, sizeof(mjtNum) * model_->nq);
    std::memcpy(initial_qvel_.data(), data_->qvel, sizeof(mjtNum) * model_->nv);
  }

  void ResetToInitialState() {
    mj_resetData(model_, data_);
    data_->time = initial_time_;
    std::memcpy(data_->qpos, initial_qpos_.data(), sizeof(mjtNum) * model_->nq);
    std::memcpy(data_->qvel, initial_qvel_.data(), sizeof(mjtNum) * model_->nv);
    if (model_->na != 0) {
      mju_zero(data_->act, model_->na);
    }
    mj_forward(model_, data_);
  }

  double Dt() const { return model_->opt.timestep * frame_skip_; }

  void DoSimulation() {
    for (int i = 0; i < frame_skip_; ++i) {
      mj_step(model_, data_);
    }
  }

  void CaptureResetState() {
#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_.data(), data_->qpos, sizeof(mjtNum) * model_->nq);
    std::memcpy(qvel0_.data(), data_->qvel, sizeof(mjtNum) * model_->nv);
#endif
  }
};

}  // namespace gymnasium_robotics

#endif  // ENVPOOL_GYMNASIUM_ROBOTICS_MUJOCO_ENV_H_
