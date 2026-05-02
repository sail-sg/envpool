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
#include <mutex>
#include <stdexcept>
#include <vector>

#if defined(__APPLE__) && __has_include(<OpenGL/OpenGL.h>)
#include <OpenGL/OpenGL.h>
#define ENVPOOL_HAS_CGL 1
#elif defined(_WIN32) && __has_include(<windows.h>)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define ENVPOOL_HAS_WGL 1
#elif defined(__linux__) && __has_include(<EGL/egl.h>)
#include <EGL/egl.h>
#if __has_include(<EGL/eglext.h>)
#include <EGL/eglext.h>
#define ENVPOOL_HAS_EGL_DEVICE_EXT 1
#endif
#define ENVPOOL_HAS_EGL 1
#endif

namespace envpool::mujoco {

namespace {

mjtNum MedianGeomPosition(const mjData* data, int ngeom, int axis) {
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

}  // namespace

#if defined(ENVPOOL_HAS_CGL)

class CglContext final : public GlContext {
 public:
  CglContext() {
    const std::array<CGLPixelFormatAttribute, 17> attribs = {
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
        kCGLPFAMultisample,
        kCGLPFASampleBuffers,
        static_cast<CGLPixelFormatAttribute>(1),
        kCGLPFASamples,
        static_cast<CGLPixelFormatAttribute>(4),
        kCGLPFAAccelerated,
        static_cast<CGLPixelFormatAttribute>(0),  // terminator
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
      if (locked_) {
        CGLUnlockContext(context_);
      }
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
    // Match mujoco.cgl.GLContext: software/offline CGL renderers rely on the
    // context being locked while it is current.
    if (!locked_) {
      err = CGLLockContext(context_);
      if (err != kCGLNoError) {
        throw std::runtime_error("failed to lock CGL context");
      }
      locked_ = true;
    }
  }

  void ClearCurrent() override {
    if (locked_) {
      CGLError err = CGLUnlockContext(context_);
      if (err != kCGLNoError) {
        throw std::runtime_error("failed to unlock CGL context");
      }
      locked_ = false;
    }
    CGLError err = CGLSetCurrentContext(nullptr);
    if (err != kCGLNoError) {
      throw std::runtime_error("failed to clear CGL context");
    }
  }

 private:
  CGLPixelFormatObj pixel_format_{nullptr};
  CGLContextObj context_{nullptr};
  bool locked_{false};
};

#elif defined(ENVPOOL_HAS_WGL)

namespace {

constexpr char kWindowClassName[] = "EnvPoolMuJoCoOffscreenWindow";

void EnsureWindowClassRegistered() {
  static std::once_flag registered;
  std::call_once(registered, [] {
    WNDCLASSA window_class = {};
    window_class.style = CS_OWNDC;
    window_class.lpfnWndProc = DefWindowProcA;
    window_class.hInstance = GetModuleHandleA(nullptr);
    window_class.lpszClassName = kWindowClassName;
    if (RegisterClassA(&window_class) == 0 &&
        GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
      throw std::runtime_error("failed to register WGL window class");
    }
  });
}

}  // namespace

class WglContext final : public GlContext {
 public:
  WglContext() {
    EnsureWindowClassRegistered();
    window_ = CreateWindowExA(0, kWindowClassName, "EnvPoolMuJoCoOffscreen",
                              WS_OVERLAPPEDWINDOW, 0, 0, 1, 1, nullptr, nullptr,
                              GetModuleHandleA(nullptr), nullptr);
    if (window_ == nullptr) {
      throw std::runtime_error("failed to create WGL window");
    }
    device_context_ = GetDC(window_);
    if (device_context_ == nullptr) {
      DestroyWindow(window_);
      window_ = nullptr;
      throw std::runtime_error("failed to acquire WGL device context");
    }

    PIXELFORMATDESCRIPTOR pixel_format = {};
    pixel_format.nSize = sizeof(pixel_format);
    pixel_format.nVersion = 1;
    pixel_format.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL;
    pixel_format.iPixelType = PFD_TYPE_RGBA;
    pixel_format.cColorBits = 24;
    pixel_format.cAlphaBits = 8;
    pixel_format.cDepthBits = 24;
    pixel_format.cStencilBits = 8;
    pixel_format.iLayerType = PFD_MAIN_PLANE;
    int format_id = ChoosePixelFormat(device_context_, &pixel_format);
    if (format_id == 0 ||
        SetPixelFormat(device_context_, format_id, &pixel_format) == FALSE) {
      ReleaseDC(window_, device_context_);
      device_context_ = nullptr;
      DestroyWindow(window_);
      window_ = nullptr;
      throw std::runtime_error("failed to configure WGL pixel format");
    }

    context_ = wglCreateContext(device_context_);
    if (context_ == nullptr) {
      ReleaseDC(window_, device_context_);
      device_context_ = nullptr;
      DestroyWindow(window_);
      window_ = nullptr;
      throw std::runtime_error("failed to create WGL context");
    }
  }

  ~WglContext() override {
    if (context_ != nullptr) {
      wglMakeCurrent(nullptr, nullptr);
      wglDeleteContext(context_);
    }
    if (window_ != nullptr && device_context_ != nullptr) {
      ReleaseDC(window_, device_context_);
    }
    if (window_ != nullptr) {
      DestroyWindow(window_);
    }
  }

  void MakeCurrent() override {
    if (wglMakeCurrent(device_context_, context_) == FALSE) {
      throw std::runtime_error("failed to make WGL context current");
    }
  }

  void ClearCurrent() override {
    if (wglMakeCurrent(nullptr, nullptr) == FALSE) {
      throw std::runtime_error("failed to clear WGL context");
    }
  }

 private:
  HWND window_{nullptr};
  HDC device_context_{nullptr};
  HGLRC context_{nullptr};
};

class BorrowedWglContext final : public GlContext {
 public:
  BorrowedWglContext()
      : device_context_(wglGetCurrentDC()), context_(wglGetCurrentContext()) {
    if (device_context_ == nullptr || context_ == nullptr) {
      throw std::runtime_error("failed to capture current WGL context");
    }
  }

  void MakeCurrent() override {
    if (wglMakeCurrent(device_context_, context_) == FALSE) {
      throw std::runtime_error("failed to make borrowed WGL context current");
    }
  }

  void ClearCurrent() override {
    if (wglMakeCurrent(nullptr, nullptr) == FALSE) {
      throw std::runtime_error("failed to clear borrowed WGL context");
    }
  }

 private:
  HDC device_context_{nullptr};
  HGLRC context_{nullptr};
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
    const EGLBoolean query_ok =
        query_devices(k_max_devices, devices.data(), &num_devices);
    if (query_ok != EGL_TRUE || num_devices < 1) {
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

std::shared_ptr<GlContext> CreateGlContext(bool share_cgl_context) {
#if defined(ENVPOOL_HAS_CGL)
  if (share_cgl_context) {
    thread_local std::shared_ptr<GlContext> context =
        std::make_shared<CglContext>();
    return context;
  }
  // Match Gymnasium's CGL lifecycle: create a context per renderer/viewer.
  // Reusing one CGL context across different MuJoCo models can leave renderer
  // state behind on macOS software/offline renderers.
  return std::make_shared<CglContext>();
#elif defined(ENVPOOL_HAS_WGL)
  (void)share_cgl_context;
  if (wglGetCurrentContext() != nullptr && wglGetCurrentDC() != nullptr) {
    // Borrowed WGL handles become invalid if another library later calls
    // `glfw.terminate()`, so do not cache them across renderer instances.
    return std::make_shared<BorrowedWglContext>();
  }
  thread_local std::shared_ptr<GlContext> context =
      std::make_shared<WglContext>();
  return context;
#elif defined(ENVPOOL_HAS_EGL)
  (void)share_cgl_context;
  thread_local std::shared_ptr<GlContext> context =
      std::make_shared<EglContext>();
  return context;
#else
  (void)share_cgl_context;
  throw std::runtime_error(
      "MuJoCo rendering is unsupported on this platform/build");
#endif
}

OffscreenRenderer::OffscreenRenderer(CameraPolicy camera_policy,
                                     bool disable_auxiliary_visuals,
                                     bool share_cgl_context)
    : camera_policy_(camera_policy), share_cgl_context_(share_cgl_context) {
  mjv_defaultScene(&scene_);
  mjv_defaultCamera(&camera_);
  mjv_defaultOption(&option_);
  mjv_defaultPerturb(&perturb_);
  mjr_defaultContext(&context_);
  if (disable_auxiliary_visuals) {
    option_.flags[mjVIS_TENDON] = 0;
    option_.flags[mjVIS_ACTUATOR] = 0;
    option_.flags[mjVIS_ACTIVATION] = 0;
  }
  camera_.fixedcamid = -1;
}

OffscreenRenderer::~OffscreenRenderer() {
  if (!initialized_) {
    return;
  }
  gl_context_->MakeCurrent();
  mjr_freeContext(&context_);
  mjv_freeScene(&scene_);
#if !defined(ENVPOOL_HAS_CGL)
  gl_context_->ClearCurrent();
#endif
}

void OffscreenRenderer::Initialize(const mjModel* model) {
  gl_context_ = CreateGlContext(share_cgl_context_);
  gl_context_->MakeCurrent();
  mjv_makeScene(model, &scene_, 10000);
  mjr_makeContext(model, &context_, mjFONTSCALE_150);
  mjr_setBuffer(mjFB_OFFSCREEN, &context_);
  context_.readDepthMap = mjDEPTH_ZEROFAR;
  initialized_ = true;
}

void OffscreenRenderer::UpdateCamera(const mjModel* model, const mjData* data,
                                     int camera_id,
                                     const mjvCamera* camera_override) {
  if (camera_id < -1 || camera_id >= model->ncam) {
    throw std::out_of_range("camera_id is out of range");
  }
  if (camera_id == -1 && camera_override != nullptr) {
    camera_ = *camera_override;
    return;
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
        camera_.lookat[axis] = MedianGeomPosition(data, model->ngeom, axis);
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
                               int height, int camera_id, unsigned char* rgb,
                               const mjvCamera* camera_override,
                               const mjvOption* option_override) {
  if (!initialized_) {
    Initialize(model);
  }
  gl_context_->MakeCurrent();
  mjr_setBuffer(mjFB_OFFSCREEN, &context_);
  UpdateCamera(model, data, camera_id, camera_override);

  mjrRect viewport = {0, 0, width, height};
  auto render_scene = [&] {
    mjv_updateScene(model, data,
                    option_override != nullptr ? option_override : &option_,
                    nullptr, &camera_, mjCAT_ALL, &scene_);
    mjr_render(viewport, &scene_, &context_);
  };
  render_scene();
#if defined(ENVPOOL_HAS_CGL)
  // Match the first-frame CGL warmup needed by MuJoCo's Python renderer on
  // macOS; otherwise a few offscreen tasks can differ on their first frame.
  render_scene();
#endif

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
