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

#ifndef ENVPOOL_MUJOCO_PLAYGROUND_MUJOCO_ENV_H_
#define ENVPOOL_MUJOCO_PLAYGROUND_MUJOCO_ENV_H_

#include <mujoco.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/frame_stack.h"
#include "envpool/mujoco/offscreen_renderer.h"

namespace mujoco_playground {

template <typename EnvFns>
using PlaygroundEnvSpecT = EnvSpec<EnvFns>;

template <typename EnvFns>
using PlaygroundPixelEnvFns = envpool::mujoco::PixelObservationEnvFns<EnvFns>;

template <typename Env>
using PlaygroundEnvPoolT = AsyncEnvPool<Env>;

template <typename EnvFns>
struct PlaygroundEnvAliases {
  using Spec = PlaygroundEnvSpecT<EnvFns>;
  using PixelFns = PlaygroundPixelEnvFns<EnvFns>;
  using PixelSpec = PlaygroundEnvSpecT<PixelFns>;
};

class PlaygroundMujocoEnv : public RenderableEnv {
 private:
  std::array<char, 1000> error_{};

 protected:
  mjModel* model_{nullptr};
  mjData* data_{nullptr};
  int max_episode_steps_{0};
  int elapsed_step_{0};
  bool done_{true};
  bool terminated_{false};
  std::unique_ptr<envpool::mujoco::OffscreenRenderer> renderer_;
  envpool::mujoco::FrameStackBuffer frame_stack_buffer_;
  envpool::mujoco::PixelFrameStackBuffer pixel_frame_stack_buffer_;
  int render_width_{84};
  int render_height_{84};
  int render_camera_id_{-1};
  envpool::mujoco::CameraPolicy camera_policy_{
      envpool::mujoco::CameraPolicy::kGymLike};
  std::vector<uint8_t> cached_render_;
  bool has_cached_render_{false};
  int cached_render_width_{0};
  int cached_render_height_{0};
  int cached_render_camera_id_{0};

 public:
  PlaygroundMujocoEnv(const std::string& xml_path, int max_episode_steps,
                      int frame_stack, int render_width = 84,
                      int render_height = 84, int render_camera_id = -1,
                      envpool::mujoco::CameraPolicy camera_policy =
                          envpool::mujoco::CameraPolicy::kGymLike)
      : max_episode_steps_(max_episode_steps),
        frame_stack_buffer_(frame_stack),
        pixel_frame_stack_buffer_(frame_stack),
        render_width_(render_width),
        render_height_(render_height),
        render_camera_id_(render_camera_id),
        camera_policy_(camera_policy) {
    model_ = mj_loadXML(xml_path.c_str(), nullptr, error_.data(), 1000);
    if (model_ == nullptr) {
      throw std::runtime_error(error_.data());
    }
    data_ = mj_makeData(model_);
  }

  ~PlaygroundMujocoEnv() override {
    mj_deleteData(data_);
    mj_deleteModel(model_);
  }

  void RenderFresh(int width, int height, int camera_id, unsigned char* rgb) {
#ifdef _WIN32
    envpool::mujoco::OffscreenRenderer renderer(camera_policy_);
    renderer.Render(model_, data_, width, height, camera_id, rgb);
#else
    if (renderer_ == nullptr) {
      renderer_ = std::make_unique<envpool::mujoco::OffscreenRenderer>(
          camera_policy_, /*disable_auxiliary_visuals=*/false,
          /*share_cgl_context=*/false,
          /*prefer_offline_cgl_context=*/false, /*resize_offscreen=*/true);
    }
    renderer_->Render(model_, data_, width, height, camera_id, rgb);
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

  mjtNum* PrepareObservation(std::string_view key, Array* target) {
    return frame_stack_buffer_.Prepare(key, target);
  }

  void CommitObservation(std::string_view key, Array* target, bool reset) {
    frame_stack_buffer_.Commit(key, target, reset);
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

}  // namespace mujoco_playground

#endif  // ENVPOOL_MUJOCO_PLAYGROUND_MUJOCO_ENV_H_
