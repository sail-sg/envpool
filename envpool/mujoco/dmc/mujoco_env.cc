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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mujoco_dmc {

MujocoEnv::MujocoEnv(const std::string& base_path, const std::string& raw_xml,
                     int n_sub_steps, int max_episode_steps, const int height,
                     const int, const std::string& camera_id, bool depth,
                     bool segmentation)
    : n_sub_steps_(n_sub_steps),
      max_episode_steps_(max_episode_steps),
      elapsed_step_(max_episode_steps + 1) {
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

  // default free camera
  // mjv_defaultFreeCamera(model_, &camera_);
  // set rendering to offscreen buffer
  mjr_setBuffer(mjFB_OFFSCREEN, &context_);
  // allocate rgb and depth buffers
  // std::vector<unsigned char> rgb_array_(3 * width_ * height_);
  // std::vector<float> depth_array_(width_ * height_);
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
  // std::free(rgb_array_);
  // std::free(depth_array_);
  mj_deleteData(data_);
  mj_deleteModel(model_);
  // mjr_freeContext(&context_);
  // mjv_freeScene(&scene_);
  // closeOpenGL();
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

  // return {rgb_array_, depth_array_, segmentation_array_};
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

/**
 * FrameStack env wrapper implementation.
 *
 * The original gray scale image are saved inside maxpool_buf_.
 * The stacked result is in stack_buf_ where len(stack_buf_) == stack_num_.
 *
 * At reset time, we need to clear all data in stack_buf_ with push_all =
 * true and maxpool = false (there is only one observation); at step time,
 * we push max(maxpool_buf_[0], maxpool_buf_[1]) at the end of
 * stack_buf_, and pop the first item in stack_buf_, with push_all = false
 * and maxpool = true.
 *
 * @param push_all whether to use the most recent observation to write all
 *   of the data in stack_buf_.
 * @param maxpool whether to perform maxpool operation on the last two
 *   observation. Maybe there is only one?
 */
// void MujocoEnv::PushStack(bool push_all, bool maxpool) {
//   auto* ptr = static_cast<uint8_t*>(maxpool_buf_[0].Data());
//   if (maxpool) {
//     auto* ptr1 = static_cast<uint8_t*>(maxpool_buf_[1].Data());
//     for (std::size_t i = 0; i < maxpool_buf_[0].size; ++i) {
//       ptr[i] = std::max(ptr[i], ptr1[i]);
//     }
//   }
//   Resize(maxpool_buf_[0], &resize_img_, use_inter_area_resize_);
//   Array tgt = std::move(*stack_buf_.begin());
//   ptr = static_cast<uint8_t*>(tgt.Data());
//   stack_buf_.pop_front();
//   if (gray_scale_) {
//     tgt.Assign(resize_img_);
//   } else {
//     auto* ptr1 = static_cast<uint8_t*>(resize_img_.Data());
//     // tgt = resize_img_.transpose(1, 2, 0)
//     // tgt[i, j, k] = resize_img_[j, k, i]
//     std::size_t h = resize_img_.Shape(0);
//     std::size_t w = resize_img_.Shape(1);
//     for (std::size_t j = 0; j < h; ++j) {
//       for (std::size_t k = 0; k < w; ++k) {
//         for (std::size_t i = 0; i < 3; ++i) {
//           ptr[i * h * w + j * w + k] = ptr1[j * w * 3 + k * 3 + i];
//         }
//       }
//     }
//   }
//   std::size_t size = tgt.size;
//   stack_buf_.push_back(std::move(tgt));
//   if (push_all) {
//     for (auto& s : stack_buf_) {
//       auto* ptr_s = static_cast<uint8_t*>(s.Data());
//       if (ptr != ptr_s) {
//         std::memcpy(ptr_s, ptr, size);
//       }
//     }
//   }
// }

// create OpenGL context/window
void MujocoEnv::initOpenGL(void) {
  //------------------------ GLFW

#if defined(MJ_GLFW)
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
  //------------------------ OSMESA
#elif defined(MJ_OSMESA)
  // create context
  ctx_ = OSMesaCreateContextExt(GL_RGBA, 24, 8, 8, 0);
  if (!ctx_) {
    mju_error("OSMesa context creation failed");
  }

  // make current
  if (!OSMesaMakeCurrent(ctx_, buffer_, GL_UNSIGNED_BYTE, 800, 800)) {
    mju_error("OSMesa make current failed");
  }

  //------------------------ EGL

#else
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
  eglInitialize(eglDpy, &major, &minor);

  // if (eglInitialize(eglDpy, &major, &minor) != EGL_TRUE) {
  //   mju_error_i("Could not initialize EGL, error 0x%x\n", eglGetError());
  // }

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
#endif
}

// close OpenGL context/window
void MujocoEnv::closeOpenGL(void) {
  //------------------------ GLFW

#if defined(MJ_GLFW)
// terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
  glfwTerminate();
#endif

  //------------------------ OSMESA
#elif defined(MJ_OSMESA)
  OSMesaDestroyContext(ctx_);
  // std::free(buffer_);
  //------------------------ EGL
#else
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
#endif
}
}  // namespace mujoco_dmc
