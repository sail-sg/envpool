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

#ifndef ENVPOOL_MUJOCO_DMC_MUJOCO_ENV_H_
#define ENVPOOL_MUJOCO_DMC_MUJOCO_ENV_H_

#include <mjxmacro.h>
#include <mujoco.h>
// select EGL, OSMESA or GLFW
#if defined(MJ_EGL)
#include <EGL/egl.h>
#elif defined(MJ_OSMESA)
#include <GL/osmesa.h>
OSMesaContext ctx;
unsigned char buffer[10000000];
#else
#include <GLFW/glfw3.h>
#endif

#include <memory>
#include <random>
#include <string>

#include "array_safety.h"
#include "envpool/mujoco/dmc/utils.h"

namespace mujoco_dmc {

/*
 * This class combines with dmc Task and Physics API.
 *
 * https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/rl/control.py
 * https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/base.py
 * https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py
 * https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/composer/task.py
 * https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/composer/environment.py
 */
class MujocoEnv {
 private:
  std::array<char, 1000> error_;

 protected:
  mjModel* model_;
  mjData* data_;
  mjvScene scene_;
  mjvCamera camera_;
  mjvOption option_;
  mjrContext context_;
  int n_sub_steps_, max_episode_steps_, elapsed_step_;
  float reward_, discount_;
  bool done_;
  int height_, width_;
  bool depth_, segmentation_;
  const std::string& camera_id_;
  unsigned char* rgb_array_;
  auto* depth_array_;
#ifdef ENVPOOL_TEST
  std::unique_ptr<mjtNum> qpos0_;
#endif

 public:
  MujocoEnv(const std::string& base_path, const std::string& raw_xml,
            int n_sub_steps, int max_episode_steps);
  ~MujocoEnv();

  // rl control Environment
  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/rl/control.py#L77
  void ControlReset();

  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/rl/control.py#L94
  void ControlStep(const mjtNum* action);

  // Task
  virtual void TaskInitializeEpisodeMjcf() {}
  virtual void TaskInitializeEpisode() {}
  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/base.py#L73
  virtual void TaskBeforeStep(const mjtNum* action);
  virtual void TaskBeforeSubStep(const mjtNum* action) {}
  virtual void TaskAfterStep() {}
  virtual void TaskAftersSubStep() {}
  virtual float TaskGetReward();
  virtual float TaskGetDiscount();
  virtual bool TaskShouldTerminateEpisode();

  // Physics
  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L263
  void PhysicsReset(int keyframe_id = -1);

  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L286
  void PhysicsAfterReset();

  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L138
  void PhysicsSetControl(const mjtNum* control);

  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L292
  void PhysicsForward();

  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L146
  void PhysicsStep(int nstep, const mjtNum* action);

  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py#L165
  void PhysicsRender(int height, int width, const std::string& camera_id,
                     bool depth, bool segmentation);
  // randomizer
  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/utils/randomizers.py#L35
  void RandomizeLimitedAndRotationalJoints(std::mt19937* gen);
  // create OpenGL context/window
  void initOpenGL(void);
  // close OpenGL context/window
  void closeOpenGL(void);
};

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_MUJOCO_ENV_H_
