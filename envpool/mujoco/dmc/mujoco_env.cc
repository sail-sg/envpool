// Copyright 2022 Garena Online Private Limited
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

#include "envpool/mujoco/dmc/mujoco_env.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace mujoco_dmc {

MujocoEnv::MujocoEnv(const std::string& base_path, const std::string& raw_xml,
                     int n_sub_steps, int max_episode_steps, int height,
                     int width, const std::string& camera_id, bool depth,
                     bool segmentation, )
    : n_sub_steps_(n_sub_steps),
      max_episode_steps_(max_episode_steps),
      elapsed_step_(max_episode_steps + 1),
      done_(true),
      height_(240),
      width_(320),
      camera_id_("-1"),
      depth_(false),
      segmentation_(false) {
  initOpenGL();
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
  std::vector<std::string> common_assets_name(
      {"./common/materials.xml", "./common/skybox.xml", "./common/visual.xml"});
  for (const auto& asset_name : common_assets_name) {
    std::string content = GetFileContent(base_path, asset_name);
    mj_makeEmptyFileVFS(vfs.get(), asset_name.c_str(), content.size());
    std::memcpy(vfs->filedata[vfs->nfile - 1], content.c_str(), content.size());
  }
  // create model and data
  model_ = mj_loadXML(model_filename.c_str(), vfs.get(), error_.begin(), 1000);
  data_ = mj_makeData(model_);
  // MuJoCo visualization
  mjvScene scene_;
  mjvCamera camera_;
  mjvOption option_;
  mjrContext context_;
  // init visualization
  mjv_defaultCamera(&camera_);
  mjv_defaultOption(&option_);
  mjv_defaultScene(&scene_);
  mjr_defaultContext(&context_);

  // create scene and context
  // void mjv_makeScene(const mjModel* m, mjvScene* scn, int maxgeom);
  mjv_makeScene(model_, &scene_, 2000);
  // void mjr_makeContext(const mjModel* m, mjrContext* con, int fontscale);
  mjr_makeContext(model_, &context_, 200);
  // set rendering to offscreen buffer
  mjr_setBuffer(mjFB_OFFSCREEN, &context_);
  // allocate rgb and depth buffers
  unsigned char* rgb_array_ = (unsigned char*)std::malloc(3 * width_ * height_);
  auto* depth_array_ =
      (reinterpret_cast<float*>)std::malloc(sizeof(float) * width_ * height_);
  // camera configuration
  // cam.lookat[0] = m->stat.center[0];
  // cam.lookat[1] = m->stat.center[1];
  // cam.lookat[2] = m->stat.center[2];
  // cam.distance = 1.5 * m->stat.extent;

#ifdef ENVPOOL_TEST
  qpos0_.reset(new mjtNum[model_->nq]);
#endif
}

MujocoEnv::~MujocoEnv() {
  mj_deleteModel(model_);
  mj_deleteData(data_);
  mjr_freeContext(&context_);
  mjv_freeScene(&scene_);
  closeOpenGL();
}

// rl control Environment
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/rl/control.py#L77
void MujocoEnv::ControlReset() {
  elapsed_step_ = 0;
  discount_ = 1.0;
  done_ = false;
  TaskInitializeEpisodeMjcf();
  // attention: no keyframe_id
  PhysicsReset();  // first mj_forward
  TaskInitializeEpisode();
  PhysicsAfterReset();  // second mj_forward
}

// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/rl/control.py#L94
void MujocoEnv::ControlStep(const mjtNum* action) {
  TaskBeforeStep(action);
  PhysicsStep(n_sub_steps_, action);
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
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/base.py#L73
void MujocoEnv::TaskBeforeStep(const mjtNum* action) {
  PhysicsSetControl(action);
}
float MujocoEnv::TaskGetReward() {
  throw std::runtime_error("GetReward not implemented");
}
float MujocoEnv::TaskGetDiscount() { return 1.0; }
bool MujocoEnv::TaskShouldTerminateEpisode() { return false; }

// Physics
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L263
void MujocoEnv::PhysicsReset(int keyframe_id) {
  if (keyframe_id < 0) {
    mj_resetData(model_, data_);
  } else {
    // actually no one steps to this line
    assert(keyframe_id < model_->nkey);
    mj_resetDataKeyframe(model_, data_, keyframe_id);
  }

  // PhysicsAfterReset may be overwritten?
  int old_flags = model_->opt.disableflags;
  model_->opt.disableflags |= mjDSBL_ACTUATION;
  PhysicsForward();
  model_->opt.disableflags = old_flags;
}

// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L286
void MujocoEnv::PhysicsAfterReset() {
  int old_flags = model_->opt.disableflags;
  model_->opt.disableflags |= mjDSBL_ACTUATION;
  PhysicsForward();
  model_->opt.disableflags = old_flags;
}

// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L138
void MujocoEnv::PhysicsSetControl(const mjtNum* control) {
  std::memcpy(data_->ctrl, control, sizeof(mjtNum) * model_->nu);
}

// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L292
void MujocoEnv::PhysicsForward() { mj_forward(model_, data_); }

// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L146
void MujocoEnv::PhysicsStep(int nstep, const mjtNum* action) {
  TaskBeforeSubStep(action);
  if (model_->opt.integrator != mjINT_RK4) {
    mj_step2(model_, data_);
  } else {
    mj_step(model_, data_);
  }
  TaskAftersSubStep();
  if (nstep > 1) {
    for (int i = 0; i < nstep - 1; ++i) {
      TaskBeforeSubStep(action);
      mj_step(model_, data_);
      TaskAftersSubStep();
    }
  }
  mj_step1(model_, data_);
}

// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L165
void MujocoEnv::PhysicsRender(int height, int width,
                              const std::string& camera_id, bool depth,
                              bool segmentation) {
  // update abstract scene
  mjv_updateScene(model_, data_, &option_, NULL, &camera_, mjCAT_ALL, &scene_);
  mjrRect viewport = {0, 0, width_, height_};
  // render scene in offscreen buffer
  mjr_render(viewport, &scene_, &context_);

  // read rgb and depth buffers
  mjr_readPixels(rgb_array_, depth_array_, viewport, &context_);

  // segmentation results not implemented

  return { rgb_array_, depth_array_, segmentation_array_ }
}

// randomizer
// https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/utils/randomizers.py#L35
void MujocoEnv::RandomizeLimitedAndRotationalJoints(std::mt19937* gen) {
  for (int joint_id = 0; joint_id < model_->njnt; ++joint_id) {
    int joint_type = model_->jnt_type[joint_id];
    mjtByte is_limited = model_->jnt_limited[joint_id];
    mjtNum range_min = model_->jnt_range[joint_id * 2 + 0];
    mjtNum range_max = model_->jnt_range[joint_id * 2 + 1];
    int qpos_offset = model_->jnt_qposadr[joint_id];
    if (is_limited != 0) {
      if (joint_type == mjJNT_HINGE || joint_type == mjJNT_SLIDE) {
        data_->qpos[qpos_offset] = RandUniform(range_min, range_max)(*gen);
      } else if (joint_type == mjJNT_BALL) {
        // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/utils/randomizers.py#L23
        std::array<mjtNum, 3> axis = {RandNormal(0, 1)(*gen),
                                      RandNormal(0, 1)(*gen),
                                      RandNormal(0, 1)(*gen)};
        auto norm = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] +
                              axis[2] * axis[2]);
        axis = {axis[0] / norm, axis[1] / norm, axis[2] / norm};
        auto angle = RandUniform(0, range_max)(*gen);
        mju_axisAngle2Quat(data_->qpos + qpos_offset, axis.begin(), angle);
      }
    } else if (joint_type == mjJNT_HINGE) {
      data_->qpos[qpos_offset] = RandUniform(-M_PI, M_PI)(*gen);
    } else if (joint_type == mjJNT_BALL || joint_type == mjJNT_FREE) {
      std::array<mjtNum, 4> quat;
      if (joint_type == mjJNT_BALL) {
        quat = {RandNormal(0, 1)(*gen), RandNormal(0, 1)(*gen),
                RandNormal(0, 1)(*gen), RandNormal(0, 1)(*gen)};
      } else {
        quat = {RandUniform(0, 1)(*gen), RandUniform(0, 1)(*gen),
                RandUniform(0, 1)(*gen), RandUniform(0, 1)(*gen)};
      }
      auto norm = std::sqrt(quat[0] * quat[0] + quat[1] * quat[1] +
                            quat[2] * quat[2] + quat[3] * quat[3]);
      int extra_offset = joint_type == mjJNT_BALL ? 0 : 3;
      data_->qpos[qpos_offset + extra_offset + 0] = quat[0] / norm;
      data_->qpos[qpos_offset + extra_offset + 1] = quat[1] / norm;
      data_->qpos[qpos_offset + extra_offset + 2] = quat[2] / norm;
      data_->qpos[qpos_offset + extra_offset + 3] = quat[3] / norm;
    }
  }
}
// create OpenGL context/window
void initOpenGL(void) {
  //------------------------ EGL
#if defined(MJ_EGL)
  // desired config
  const EGLint configAttribs[] = {EGL_RED_SIZE,
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
                                  EGL_NONE};

  // get default display
  EGLDisplay eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (eglDpy == EGL_NO_DISPLAY) {
    mju_error_i("Could not get EGL display, error 0x%x\n", eglGetError());
  }

  // initialize
  EGLint major, minor;
  if (eglInitialize(eglDpy, &major, &minor) != EGL_TRUE) {
    mju_error_i("Could not initialize EGL, error 0x%x\n", eglGetError());
  }

  // choose config
  EGLint numConfigs;
  EGLConfig eglCfg;
  if (eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs) !=
      EGL_TRUE) {
    mju_error_i("Could not choose EGL config, error 0x%x\n", eglGetError());
  }

  // bind OpenGL API
  if (eglBindAPI(EGL_OPENGL_API) != EGL_TRUE) {
    mju_error_i("Could not bind EGL OpenGL API, error 0x%x\n", eglGetError());
  }

  // create context
  EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, NULL);
  if (eglCtx == EGL_NO_CONTEXT) {
    mju_error_i("Could not create EGL context, error 0x%x\n", eglGetError());
  }

  // make context current, no surface (let OpenGL handle FBO)
  if (eglMakeCurrent(eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, eglCtx) !=
      EGL_TRUE) {
    mju_error_i("Could not make EGL context current, error 0x%x\n",
                eglGetError());
  }

  //------------------------ OSMESA
#elif defined(MJ_OSMESA)
  // create context
  ctx = OSMesaCreateContextExt(GL_RGBA, 24, 8, 8, 0);
  if (!ctx) {
    mju_error("OSMesa context creation failed");
  }

  // make current
  if (!OSMesaMakeCurrent(ctx, buffer, GL_UNSIGNED_BYTE, 800, 800)) {
    mju_error("OSMesa make current failed");
  }

  //------------------------ GLFW
#else
  // init GLFW
  if (!glfwInit()) {
    mju_error("Could not initialize GLFW");
  }

  // create invisible window, single-buffered
  glfwWindowHint(GLFW_VISIBLE, 0);
  glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
  GLFWwindow* window =
      glfwCreateWindow(800, 800, "Invisible window", NULL, NULL);
  if (!window) {
    mju_error("Could not create GLFW window");
  }

  // make context current
  glfwMakeContextCurrent(window);
#endif
}

// close OpenGL context/window
void closeOpenGL(void) {
  //------------------------ EGL
#if defined(MJ_EGL)
  // get current display
  EGLDisplay eglDpy = eglGetCurrentDisplay();
  if (eglDpy == EGL_NO_DISPLAY) {
    return;
  }

  // get current context
  EGLContext eglCtx = eglGetCurrentContext();

  // release context
  eglMakeCurrent(eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

  // destroy context if valid
  if (eglCtx != EGL_NO_CONTEXT) {
    eglDestroyContext(eglDpy, eglCtx);
  }

  // terminate display
  eglTerminate(eglDpy);

  //------------------------ OSMESA
#elif defined(MJ_OSMESA)
  OSMesaDestroyContext(ctx);

  //------------------------ GLFW
#else
// terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
  glfwTerminate();
#endif
#endif
}
}  // namespace mujoco_dmc
