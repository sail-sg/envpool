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

#include "envpool/mujoco/offscreen_renderer.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(__APPLE__) && __has_include(<OpenGL/OpenGL.h>)
#include <OpenGL/OpenGL.h>
#define ENVPOOL_HAS_CGL 1
#elif defined(__linux__) && __has_include(<EGL/egl.h>)
#include <EGL/egl.h>
#define ENVPOOL_HAS_EGL 1
#endif

namespace envpool::mujoco {

#if defined(ENVPOOL_HAS_CGL)

class CglContext final : public GlContext {
 public:
  CglContext() {
    CGLPixelFormatAttribute attribs[] = {
        kCGLPFAOpenGLProfile,
        static_cast<CGLPixelFormatAttribute>(kCGLOGLPVersion_Legacy),
        kCGLPFAColorSize,
        static_cast<CGLPixelFormatAttribute>(24),
        kCGLPFAAlphaSize,
        static_cast<CGLPixelFormatAttribute>(8),
        kCGLPFADepthSize,
        static_cast<CGLPixelFormatAttribute>(24),
        kCGLPFAStencilSize,
        static_cast<CGLPixelFormatAttribute>(8),
        kCGLPFAAccelerated,
        static_cast<CGLPixelFormatAttribute>(0),
    };
    GLint npix = 0;
    CGLError err =
        CGLChoosePixelFormat(attribs, &pixel_format_, &npix);
    if (err != kCGLNoError || pixel_format_ == nullptr || npix == 0) {
      throw std::runtime_error("failed to create CGL pixel format");
    }
    err = CGLCreateContext(pixel_format_, nullptr, &context_);
    if (err != kCGLNoError || context_ == nullptr) {
      CGLReleasePixelFormat(pixel_format_);
      pixel_format_ = nullptr;
      throw std::runtime_error("failed to create CGL context");
    }
  }

  ~CglContext() override {
    if (context_ != nullptr) {
      CGLSetCurrentContext(nullptr);
      CGLReleaseContext(context_);
    }
    if (pixel_format_ != nullptr) {
      CGLReleasePixelFormat(pixel_format_);
    }
  }

  void MakeCurrent() override {
    CGLError err = CGLSetCurrentContext(context_);
    if (err != kCGLNoError) {
      throw std::runtime_error("failed to make CGL context current");
    }
  }

 private:
  CGLPixelFormatObj pixel_format_{nullptr};
  CGLContextObj context_{nullptr};
};

#elif defined(ENVPOOL_HAS_EGL)

class EglContext final : public GlContext {
 public:
  EglContext() {
    const EGLint config_attribs[] = {
        EGL_RED_SIZE,          8, EGL_GREEN_SIZE,        8,
        EGL_BLUE_SIZE,         8, EGL_ALPHA_SIZE,        8,
        EGL_DEPTH_SIZE,       24, EGL_STENCIL_SIZE,      8,
        EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER, EGL_SURFACE_TYPE,
        EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE,
    };

    display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display_ == EGL_NO_DISPLAY) {
      throw std::runtime_error("failed to get EGL display");
    }
    EGLint major = 0;
    EGLint minor = 0;
    if (eglInitialize(display_, &major, &minor) != EGL_TRUE) {
      display_ = EGL_NO_DISPLAY;
      throw std::runtime_error("failed to initialize EGL");
    }
    EGLConfig config = nullptr;
    EGLint num_configs = 0;
    if (eglChooseConfig(display_, config_attribs, &config, 1, &num_configs) !=
            EGL_TRUE ||
        num_configs < 1) {
      eglTerminate(display_);
      display_ = EGL_NO_DISPLAY;
      throw std::runtime_error("failed to choose EGL config");
    }
    if (eglBindAPI(EGL_OPENGL_API) != EGL_TRUE) {
      eglTerminate(display_);
      display_ = EGL_NO_DISPLAY;
      throw std::runtime_error("failed to bind EGL OpenGL API");
    }
    context_ = eglCreateContext(display_, config, EGL_NO_CONTEXT, nullptr);
    if (context_ == EGL_NO_CONTEXT) {
      eglTerminate(display_);
      display_ = EGL_NO_DISPLAY;
      throw std::runtime_error("failed to create EGL context");
    }
  }

  ~EglContext() override {
    if (display_ != EGL_NO_DISPLAY) {
      eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
      if (context_ != EGL_NO_CONTEXT) {
        eglDestroyContext(display_, context_);
      }
      eglTerminate(display_);
      eglReleaseThread();
    }
  }

  void MakeCurrent() override {
    if (eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, context_) !=
        EGL_TRUE) {
      throw std::runtime_error("failed to make EGL context current");
    }
  }

 private:
  EGLDisplay display_{EGL_NO_DISPLAY};
  EGLContext context_{EGL_NO_CONTEXT};
};

#endif

std::unique_ptr<GlContext> CreateGlContext() {
#if defined(ENVPOOL_HAS_CGL)
  return std::make_unique<CglContext>();
#elif defined(ENVPOOL_HAS_EGL)
  return std::make_unique<EglContext>();
#else
  throw std::runtime_error(
      "MuJoCo rendering is unsupported on this platform/build");
#endif
}

OffscreenRenderer::OffscreenRenderer() {
  mjv_defaultScene(&scene_);
  mjv_defaultCamera(&camera_);
  mjv_defaultOption(&option_);
  mjr_defaultContext(&context_);
  camera_.fixedcamid = -1;
}

OffscreenRenderer::~OffscreenRenderer() {
  if (!initialized_) {
    return;
  }
  gl_context_->MakeCurrent();
  mjr_freeContext(&context_);
  mjv_freeScene(&scene_);
}

void OffscreenRenderer::Initialize(const mjModel* model) {
  gl_context_ = CreateGlContext();
  gl_context_->MakeCurrent();
  mjv_makeScene(model, &scene_, 10000);
  mjr_makeContext(model, &context_, mjFONTSCALE_150);
  mjr_setBuffer(mjFB_OFFSCREEN, &context_);
  initialized_ = true;
}

void OffscreenRenderer::UpdateCamera(const mjModel* model, const mjData* data,
                                     int camera_id) {
  if (camera_id < -1 || camera_id >= model->ncam) {
    throw std::out_of_range("camera_id is out of range");
  }
  if (camera_id == -1) {
    int track_camera_id = mj_name2id(model, mjOBJ_CAMERA, "track");
    if (track_camera_id >= 0) {
      camera_.type = mjCAMERA_FIXED;
      camera_.fixedcamid = track_camera_id;
      return;
    }
    camera_.type = mjCAMERA_FREE;
    camera_.fixedcamid = -1;
    if (!free_camera_initialized_ && model->ngeom > 0) {
      for (int axis = 0; axis < 3; ++axis) {
        std::vector<mjtNum> positions(model->ngeom);
        for (int geom_id = 0; geom_id < model->ngeom; ++geom_id) {
          positions[geom_id] = data->geom_xpos[geom_id * 3 + axis];
        }
        auto mid = positions.begin() + positions.size() / 2;
        std::nth_element(positions.begin(), mid, positions.end());
        camera_.lookat[axis] = *mid;
      }
      camera_.distance = model->stat.extent;
      free_camera_initialized_ = true;
    }
  } else {
    camera_.type = mjCAMERA_FIXED;
    camera_.fixedcamid = camera_id;
  }
}

void OffscreenRenderer::Render(const mjModel* model, mjData* data, int width,
                               int height, int camera_id, unsigned char* rgb) {
  if (!initialized_) {
    Initialize(model);
  }
  gl_context_->MakeCurrent();
  if (context_.offWidth < width || context_.offHeight < height) {
    mjr_resizeOffscreen(width, height, &context_);
  }
  mjr_setBuffer(mjFB_OFFSCREEN, &context_);
  UpdateCamera(model, data, camera_id);

  mjrRect viewport = {0, 0, width, height};
  mjv_updateScene(model, data, &option_, nullptr, &camera_, mjCAT_ALL, &scene_);
  mjr_render(viewport, &scene_, &context_);

  std::size_t frame_bytes =
      static_cast<std::size_t>(width) * height * 3 * sizeof(unsigned char);
  if (scratch_.size() != frame_bytes) {
    scratch_.resize(frame_bytes);
  }
  mjr_readPixels(scratch_.data(), nullptr, viewport, &context_);

  std::size_t row_bytes =
      static_cast<std::size_t>(width) * 3 * sizeof(unsigned char);
  for (int y = 0; y < height; ++y) {
    const auto* src =
        scratch_.data() + static_cast<std::size_t>(height - 1 - y) * row_bytes;
    std::memcpy(rgb + static_cast<std::size_t>(y) * row_bytes, src, row_bytes);
  }
}

}  // namespace envpool::mujoco
