/*
 * Copyright 2026 Garena Online Private Limited
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

#ifndef ENVPOOL_MUJOCO_OFFSCREEN_RENDERER_H_
#define ENVPOOL_MUJOCO_OFFSCREEN_RENDERER_H_

#include <mujoco.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace envpool::mujoco {

enum class CameraPolicy : std::uint8_t {
  kGymLike,
  kDmControl,
};

class GlContext {
 public:
  virtual ~GlContext() = default;

  virtual void MakeCurrent() = 0;
};

std::unique_ptr<GlContext> CreateGlContext();

class OffscreenRenderer {
 public:
  explicit OffscreenRenderer(
      CameraPolicy camera_policy = CameraPolicy::kGymLike);
  ~OffscreenRenderer();

  void Render(const mjModel* model, mjData* data, int width, int height,
              int camera_id, unsigned char* rgb);

 private:
  void Initialize(const mjModel* model);
  void UpdateCamera(const mjModel* model, const mjData* data, int camera_id);

  std::unique_ptr<GlContext> gl_context_;
  mjvScene scene_;
  mjvCamera camera_;
  mjvOption option_;
  mjrContext context_;
  std::vector<unsigned char> scratch_;
  CameraPolicy camera_policy_;
  bool initialized_{false};
  bool free_camera_initialized_{false};
};

}  // namespace envpool::mujoco

#endif  // ENVPOOL_MUJOCO_OFFSCREEN_RENDERER_H_
