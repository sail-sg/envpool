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

#ifndef ENVPOOL_MUJOCO_GYM_MUJOCO_ENV_H_
#define ENVPOOL_MUJOCO_GYM_MUJOCO_ENV_H_

#include <mjxmacro.h>
#include <mujoco.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "envpool/core/env.h"
#include "envpool/mujoco/frame_stack.h"
#include "envpool/mujoco/offscreen_renderer.h"

namespace mujoco_gym {

using envpool::mujoco::PixelObservationEnvFns;
using envpool::mujoco::RenderCameraIdOrDefault;
using envpool::mujoco::RenderHeightOrDefault;
using envpool::mujoco::RenderWidthOrDefault;
using envpool::mujoco::StackSpec;

class MujocoEnv : public RenderableEnv {
 private:
  std::array<char, 1000> error_;
  std::string xml_path_;

  static std::string ResolveXMLPath(const std::string& xml) {
    auto ext = xml.rfind(".xml");
    if (ext == std::string::npos) {
      return xml;
    }
    std::string patched = xml.substr(0, ext) + "_envpool.xml";
    std::ifstream stream(patched);
    return stream.good() ? patched : xml;
  }

 protected:
  mjModel* model_;
  mjData* data_{nullptr};
  mjtNum *init_qpos_{nullptr}, *init_qvel_{nullptr};
#ifdef ENVPOOL_TEST
  mjtNum *qpos0_{nullptr}, *qvel0_{nullptr};  // for align check
#endif
  int frame_skip_;
  int frame_stack_;
  bool post_constraint_;
  int max_episode_steps_, elapsed_step_;
  bool done_{true};
  std::unique_ptr<envpool::mujoco::OffscreenRenderer> renderer_;
  envpool::mujoco::FrameStackBuffer frame_stack_buffer_;
  envpool::mujoco::PixelFrameStackBuffer pixel_frame_stack_buffer_;
  int render_width_, render_height_, render_camera_id_;
  std::vector<uint8_t> cached_render_;
  bool has_cached_render_{false};
  int cached_render_width_{0};
  int cached_render_height_{0};
  int cached_render_camera_id_{0};

 public:
  MujocoEnv(const std::string& xml, int frame_skip, bool post_constraint,
            int max_episode_steps, int frame_stack, int render_width = 84,
            int render_height = 84, int render_camera_id = -1)
      : xml_path_(ResolveXMLPath(xml)),
        model_(mj_loadXML(xml_path_.c_str(), nullptr, error_.data(), 1000)),
        frame_skip_(frame_skip),
        frame_stack_(frame_stack),
        post_constraint_(post_constraint),
        max_episode_steps_(max_episode_steps),
        elapsed_step_(max_episode_steps + 1),
        frame_stack_buffer_(frame_stack),
        pixel_frame_stack_buffer_(frame_stack),
        render_width_(render_width),
        render_height_(render_height),
        render_camera_id_(render_camera_id) {
    if (model_ == nullptr) {
      throw std::runtime_error(error_.data());
    }
    if (frame_stack_ < 1) {
      throw std::invalid_argument("frame_stack must be greater than 0");
    }
    data_ = mj_makeData(model_);
    init_qpos_ = new mjtNum[model_->nq];
    init_qvel_ = new mjtNum[model_->nv];
#ifdef ENVPOOL_TEST
    qpos0_ = new mjtNum[model_->nq];
    qvel0_ = new mjtNum[model_->nv];
#endif
    std::memcpy(init_qpos_, data_->qpos, sizeof(mjtNum) * model_->nq);
    std::memcpy(init_qvel_, data_->qvel, sizeof(mjtNum) * model_->nv);
  }

  ~MujocoEnv() override {
    mj_deleteData(data_);
    mj_deleteModel(model_);
    delete[] init_qpos_;
    delete[] init_qvel_;
#ifdef ENVPOOL_TEST
    delete[] qpos0_;
    delete[] qvel0_;
#endif
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

  void RenderFresh(int width, int height, int camera_id, unsigned char* rgb) {
    mjvCamera camera_override;
    InitializeRenderCamera(&camera_override);
    mjvCamera* camera =
        RenderCamera(&camera_override) ? &camera_override : nullptr;
#ifdef _WIN32
    // Native pixel observations are rendered on worker threads, while env
    // teardown happens on the Python thread. Recreating the renderer on
    // Windows avoids cross-thread WGL resource lifetime issues.
    envpool::mujoco::OffscreenRenderer renderer(
        envpool::mujoco::CameraPolicy::kGymLike);
    renderer.Render(model_, data_, width, height, camera_id, rgb, camera);
#else
    if (renderer_ == nullptr) {
      renderer_ = std::make_unique<envpool::mujoco::OffscreenRenderer>(
          envpool::mujoco::CameraPolicy::kGymLike);
    }
    renderer_->Render(model_, data_, width, height, camera_id, rgb, camera);
#endif
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

  virtual bool RenderCamera(mjvCamera* camera) {
    (void)camera;
    return false;
  }

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

  mjtNum* PrepareObservation(Array* target) {
    return frame_stack_buffer_.Prepare("obs", target);
  }

  void CommitObservation(Array* target, bool reset) {
    frame_stack_buffer_.Commit("obs", target, reset);
  }

  void AssignPixelObservation(Array* target, bool reset) {
    uint8_t* scratch = pixel_frame_stack_buffer_.Prepare(
        "obs:pixels", render_width_, render_height_);
    RenderFresh(render_width_, render_height_, render_camera_id_, scratch);
    UpdateCachedRender(render_width_, render_height_, render_camera_id_,
                       scratch);
    pixel_frame_stack_buffer_.Commit("obs:pixels", target, render_width_,
                                     render_height_, reset);
  }
};

}  // namespace mujoco_gym

#endif  // ENVPOOL_MUJOCO_GYM_MUJOCO_ENV_H_
