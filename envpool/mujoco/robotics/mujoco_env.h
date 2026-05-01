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

#ifndef ENVPOOL_MUJOCO_ROBOTICS_MUJOCO_ENV_H_
#define ENVPOOL_MUJOCO_ROBOTICS_MUJOCO_ENV_H_

#include <mujoco.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "envpool/core/env.h"
#include "envpool/mujoco/frame_stack.h"
#include "envpool/mujoco/offscreen_renderer.h"
#include "envpool/mujoco/robotics/utils.h"

namespace gymnasium_robotics {

using envpool::mujoco::PixelObservationEnvFns;
using envpool::mujoco::RenderCameraIdOrDefault;
using envpool::mujoco::RenderHeightOrDefault;
using envpool::mujoco::RenderWidthOrDefault;
using envpool::mujoco::StackSpec;

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
  std::vector<mjtNum> qacc0_;
  std::vector<mjtNum> qacc_warmstart0_;
#endif
  std::unique_ptr<envpool::mujoco::OffscreenRenderer> renderer_;
  envpool::mujoco::FrameStackBuffer frame_stack_buffer_;
  envpool::mujoco::PixelFrameStackBuffer pixel_frame_stack_buffer_;
  int render_width_;
  int render_height_;
  int render_camera_id_;
  // Robotics render callbacks update visualization state, so cache the exact
  // frame emitted for native pixel observations and matching render() calls.
  std::vector<uint8_t> cached_render_;
  bool has_cached_render_{false};
  int cached_render_width_{0};
  int cached_render_height_{0};
  int cached_render_camera_id_{0};

 public:
  MujocoRobotEnv(const std::string& base_path, const std::string& model_path,
                 int frame_skip, int max_episode_steps, int frame_stack,
                 int render_width = 84, int render_height = 84,
                 int render_camera_id = -1)
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
        qvel0_(model_->nv),
        qacc0_(model_->nv),
        qacc_warmstart0_(model_->nv)
#endif
        ,
        frame_stack_buffer_(frame_stack),
        pixel_frame_stack_buffer_(frame_stack),
        render_width_(render_width),
        render_height_(render_height),
        render_camera_id_(render_camera_id) {
  }

  ~MujocoRobotEnv() override {
    mj_deleteData(data_);
    mj_deleteModel(model_);
  }

  void RenderFresh(int width, int height, int camera_id, unsigned char* rgb) {
    RenderCallback();
#ifdef _WIN32
    // Native pixel observations are rendered on worker threads, while env
    // teardown happens on the Python thread. Recreating the renderer on
    // Windows avoids cross-thread WGL resource lifetime issues.
    envpool::mujoco::OffscreenRenderer renderer(
        envpool::mujoco::CameraPolicy::kGymLike,
        DisableAuxiliaryRenderVisuals(), ShareRenderContext());
#else
    if (renderer_ == nullptr) {
      renderer_ = std::make_unique<envpool::mujoco::OffscreenRenderer>(
          envpool::mujoco::CameraPolicy::kGymLike,
          DisableAuxiliaryRenderVisuals(), ShareRenderContext());
    }
#endif
    mjvCamera camera_override;
    InitializeRenderCamera(&camera_override);
    if (RenderCamera(&camera_override)) {
#ifdef _WIN32
      renderer.Render(model_, data_, width, height, camera_id, rgb,
                      &camera_override);
#else
      renderer_->Render(model_, data_, width, height, camera_id, rgb,
                        &camera_override);
#endif
    } else {
#ifdef _WIN32
      renderer.Render(model_, data_, width, height, camera_id, rgb);
#else
      renderer_->Render(model_, data_, width, height, camera_id, rgb);
#endif
    }
  }

  bool CopyCachedRender(int width, int height, int camera_id,
                        unsigned char* rgb) const {
    if (!has_cached_render_ || cached_render_width_ != width ||
        cached_render_height_ != height ||
        cached_render_camera_id_ != camera_id) {
      return false;
    }
    std::memcpy(rgb, cached_render_.data(), cached_render_.size());
    return true;
  }

  void UpdateCachedRender(int width, int height, int camera_id,
                          const unsigned char* rgb) {
    std::size_t render_size = static_cast<std::size_t>(width) * height * 3;
    cached_render_.assign(rgb, rgb + render_size);
    has_cached_render_ = true;
    cached_render_width_ = width;
    cached_render_height_ = height;
    cached_render_camera_id_ = camera_id;
  }

  void Render(int width, int height, int camera_id,
              unsigned char* rgb) override {
    if (CopyCachedRender(width, height, camera_id, rgb)) {
      return;
    }
    RenderFresh(width, height, camera_id, rgb);
  }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : 480, height > 0 ? height : 480};
  }

 protected:
  static std::string ResolveModelPath(const std::string& base_path,
                                      const std::string& model_path) {
    if (IsAbsolutePath(model_path)) {
      return model_path;
    }
    return base_path + "/mujoco/robotics/assets/" + model_path;
  }

  static bool IsAbsolutePath(const std::string& path) {
    if (path.empty()) {
      return false;
    }
    if (path[0] == '/') {
      return true;
    }
#ifdef _WIN32
    if (path[0] == '\\') {
      return true;
    }
    return path.size() >= 3 &&
           std::isalpha(static_cast<unsigned char>(path[0])) &&
           path[1] == ':' && (path[2] == '/' || path[2] == '\\');
#else
    return false;
#endif
  }

  static mjModel* LoadModel(const std::string& xml_path) {
    std::array<char, 1000> error{};
    mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error.data(), 1000);
    if (model == nullptr) {
      throw std::runtime_error(error.data());
    }
    return model;
  }

  static mjtNum MedianGeomPosition(const mjData* data, int ngeom, int axis) {
    std::vector<mjtNum> positions(ngeom);
    for (int geom_id = 0; geom_id < ngeom; ++geom_id) {
      positions[geom_id] = data->geom_xpos[geom_id * 3 + axis];
    }
    std::sort(positions.begin(), positions.end());
    int mid = ngeom / 2;
    if (ngeom % 2 == 0) {
      return (positions[mid - 1] + positions[mid]) * static_cast<mjtNum>(0.5);
    }
    return positions[mid];
  }

  virtual void EnvSetup() {}
  virtual void StepCallback() {}
  virtual void RenderCallback() {}
  virtual bool RenderCamera(mjvCamera* camera) {
    (void)camera;
    return false;
  }
  virtual bool DisableAuxiliaryRenderVisuals() const { return true; }
  virtual bool ShareRenderContext() const { return false; }

  void InitializeRenderCamera(mjvCamera* camera) const {
    mjv_defaultCamera(camera);
    camera->type = mjCAMERA_FREE;
    camera->fixedcamid = -1;
    camera->distance = model_->stat.extent;
    if (model_->ngeom == 0) {
      return;
    }
    for (int axis = 0; axis < 3; ++axis) {
      camera->lookat[axis] = MedianGeomPosition(data_, model_->ngeom, axis);
    }
  }

  void InitializeRobotEnv() {
    initial_time_ = data_->time;
    std::memcpy(initial_qpos_.data(), data_->qpos, sizeof(mjtNum) * model_->nq);
    std::memcpy(initial_qvel_.data(), data_->qvel, sizeof(mjtNum) * model_->nv);
  }

  void ResetToInitialState() {
    has_cached_render_ = false;
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
    has_cached_render_ = false;
    for (int i = 0; i < frame_skip_; ++i) {
      mj_step(model_, data_);
    }
  }

  void CaptureResetState() {
#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_.data(), data_->qpos, sizeof(mjtNum) * model_->nq);
    std::memcpy(qvel0_.data(), data_->qvel, sizeof(mjtNum) * model_->nv);
    std::memcpy(qacc0_.data(), data_->qacc, sizeof(mjtNum) * model_->nv);
    std::memcpy(qacc_warmstart0_.data(), data_->qacc_warmstart,
                sizeof(mjtNum) * model_->nv);
#endif
  }

  mjtNum* PrepareObservation(std::string_view key, Array* target) {
    return frame_stack_buffer_.Prepare(key, target);
  }

  void CommitObservation(std::string_view key, Array* target, bool reset) {
    frame_stack_buffer_.Commit(key, target, reset);
  }

  void AssignObservation(std::string_view key, Array* target,
                         const mjtNum* data, std::size_t size, bool reset) {
    frame_stack_buffer_.Assign(key, target, data, size, reset);
  }

  void AssignObservation(std::string_view key, Array* target, mjtNum value,
                         bool reset) {
    frame_stack_buffer_.AssignScalar(key, target, value, reset);
  }

  void AssignPixelObservation(std::string_view key, Array* target, bool reset) {
    uint8_t* scratch =
        pixel_frame_stack_buffer_.Prepare(key, render_width_, render_height_);
    RenderFresh(render_width_, render_height_, render_camera_id_, scratch);
    UpdateCachedRender(render_width_, render_height_, render_camera_id_,
                       scratch);
    pixel_frame_stack_buffer_.Commit(key, target, render_width_, render_height_,
                                     reset);
  }
};

}  // namespace gymnasium_robotics

#endif  // ENVPOOL_MUJOCO_ROBOTICS_MUJOCO_ENV_H_
