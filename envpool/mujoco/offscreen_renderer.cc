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
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(__APPLE__) && __has_include(<OpenGL/OpenGL.h>)
#include <OpenGL/OpenGL.h>
#define ENVPOOL_HAS_CGL 1
#elif defined(__linux__) && __has_include(<EGL/egl.h>)
#include <EGL/egl.h>
#if __has_include(<EGL/eglext.h>)
#include <EGL/eglext.h>
#define ENVPOOL_HAS_EGL_DEVICE_EXT 1
#endif
#define ENVPOOL_HAS_EGL 1
#endif

namespace envpool::mujoco {

#if defined(ENVPOOL_HAS_CGL)

class CglContext final : public GlContext {
 public:
  CglContext() {
    const std::array<CGLPixelFormatAttribute, 12> attribs = {
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
        kCGLPFAAllowOfflineRenderers,
        static_cast<CGLPixelFormatAttribute>(0),
    };
    GLint npix = 0;
    CGLError err = CGLChoosePixelFormat(attribs.data(), &pixel_format_, &npix);
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

  void ClearCurrent() override {
    CGLError err = CGLSetCurrentContext(nullptr);
    if (err != kCGLNoError) {
      throw std::runtime_error("failed to clear CGL context");
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
    const std::array<EGLint, 19> config_attribs = {
        EGL_RED_SIZE,
        8,
        EGL_GREEN_SIZE,
        8,
        EGL_BLUE_SIZE,
        8,
        EGL_ALPHA_SIZE,
        8,
        EGL_DEPTH_SIZE,
        24,
        EGL_STENCIL_SIZE,
        8,
        EGL_COLOR_BUFFER_TYPE,
        EGL_RGB_BUFFER,
        EGL_SURFACE_TYPE,
        EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE,
        EGL_OPENGL_BIT,
        EGL_NONE,
    };
    const std::array<EGLint, 5> pbuffer_attribs = {
        EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE,
    };

    display_ = CreateDisplay();
    if (display_ == EGL_NO_DISPLAY) {
      throw std::runtime_error("failed to initialize EGL");
    }
    eglReleaseThread();
    EGLConfig config = nullptr;
    EGLint num_configs = 0;
    if (eglChooseConfig(display_, config_attribs.data(), &config, 1,
                        &num_configs) != EGL_TRUE ||
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
    surface_ =
        eglCreatePbufferSurface(display_, config, pbuffer_attribs.data());
    if (surface_ == EGL_NO_SURFACE) {
      eglTerminate(display_);
      display_ = EGL_NO_DISPLAY;
      throw std::runtime_error("failed to create EGL pbuffer surface");
    }
    context_ = eglCreateContext(display_, config, EGL_NO_CONTEXT, nullptr);
    if (context_ == EGL_NO_CONTEXT) {
      eglDestroySurface(display_, surface_);
      surface_ = EGL_NO_SURFACE;
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
      if (surface_ != EGL_NO_SURFACE) {
        eglDestroySurface(display_, surface_);
      }
      eglTerminate(display_);
      eglReleaseThread();
    }
  }

  void MakeCurrent() override {
    if (eglMakeCurrent(display_, surface_, surface_, context_) != EGL_TRUE) {
      throw std::runtime_error("failed to make EGL context current");
    }
  }

  void ClearCurrent() override {
    if (eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE,
                       EGL_NO_CONTEXT) != EGL_TRUE) {
      throw std::runtime_error("failed to clear EGL context");
    }
    eglReleaseThread();
  }

 private:
  static EGLDisplay TryInitializeDisplay(EGLDisplay display) {
    if (display == EGL_NO_DISPLAY) {
      return EGL_NO_DISPLAY;
    }
    EGLint major = 0;
    EGLint minor = 0;
    if (eglInitialize(display, &major, &minor) == EGL_TRUE) {
      return display;
    }
    return EGL_NO_DISPLAY;
  }

#if defined(ENVPOOL_HAS_EGL_DEVICE_EXT)
  static int ParseSelectedDevice() {
    const char* selected_device = std::getenv("MUJOCO_EGL_DEVICE_ID");
    if (selected_device == nullptr) {
      return -1;
    }
    char* end = nullptr;
    errno = 0;
    int64_t device_idx = std::strtoll(selected_device, &end, 10);
    if (errno != 0 || end == selected_device || *end != '\0' ||
        device_idx < 0 || device_idx > std::numeric_limits<int>::max()) {
      throw std::runtime_error(
          "MUJOCO_EGL_DEVICE_ID must be a non-negative integer");
    }
    return static_cast<int>(device_idx);
  }

  static EGLDisplay TryInitializeDeviceDisplay() {
    auto* query_devices = reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(
        eglGetProcAddress("eglQueryDevicesEXT"));
    auto* get_platform_display =
        reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
            eglGetProcAddress("eglGetPlatformDisplayEXT"));
    if (query_devices == nullptr || get_platform_display == nullptr) {
      return EGL_NO_DISPLAY;
    }
    constexpr EGLint k_max_devices = 16;
    std::array<EGLDeviceEXT, k_max_devices> devices = {};
    EGLint num_devices = 0;
    if (query_devices(
            k_max_devices, devices.data(), &num_devices) != EGL_TRUE ||
        num_devices < 1) {
      return EGL_NO_DISPLAY;
    }

    int begin = 0;
    int end = num_devices;
    int selected_device = ParseSelectedDevice();
    if (selected_device >= 0) {
      if (selected_device >= num_devices) {
        throw std::runtime_error("MUJOCO_EGL_DEVICE_ID is out of range");
      }
      begin = selected_device;
      end = selected_device + 1;
    }

    for (int device_idx = begin; device_idx < end; ++device_idx) {
      EGLDisplay display = get_platform_display(EGL_PLATFORM_DEVICE_EXT,
                                                devices[device_idx], nullptr);
      if (display == EGL_NO_DISPLAY || eglGetError() != EGL_SUCCESS) {
        continue;
      }
      display = TryInitializeDisplay(display);
      if (display != EGL_NO_DISPLAY && eglGetError() == EGL_SUCCESS) {
        return display;
      }
    }
    return EGL_NO_DISPLAY;
  }
#endif

  static EGLDisplay CreateDisplay() {
#if defined(ENVPOOL_HAS_EGL_DEVICE_EXT)
    EGLDisplay display = TryInitializeDeviceDisplay();
    if (display != EGL_NO_DISPLAY) {
      return display;
    }
#endif
    return TryInitializeDisplay(eglGetDisplay(EGL_DEFAULT_DISPLAY));
  }

  EGLDisplay display_{EGL_NO_DISPLAY};
  EGLContext context_{EGL_NO_CONTEXT};
  EGLSurface surface_{EGL_NO_SURFACE};
};

#endif

std::shared_ptr<GlContext> CreateGlContext() {
#if defined(ENVPOOL_HAS_CGL)
  thread_local std::shared_ptr<GlContext> context =
      std::make_shared<CglContext>();
  return context;
#elif defined(ENVPOOL_HAS_EGL)
  thread_local std::shared_ptr<GlContext> context =
      std::make_shared<EglContext>();
  return context;
#else
  throw std::runtime_error(
      "MuJoCo rendering is unsupported on this platform/build");
#endif
}

OffscreenRenderer::OffscreenRenderer(CameraPolicy camera_policy)
    : camera_policy_(camera_policy) {
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
  if (camera_id == -1 && camera_policy_ == CameraPolicy::kGymLike) {
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
  } else if (camera_id == -1) {
    camera_.type = mjCAMERA_FREE;
    camera_.fixedcamid = -1;
    if (!free_camera_initialized_) {
      mjv_defaultFreeCamera(model, &camera_);
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
  gl_context_->ClearCurrent();
}

}  // namespace envpool::mujoco
