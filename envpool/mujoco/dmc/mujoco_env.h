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

#include <array>
#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <utility>

#include "envpool/core/env.h"
#include "envpool/mujoco/dmc/utils.h"
#include "envpool/mujoco/frame_stack.h"
#include "envpool/mujoco/offscreen_renderer.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mujoco_dmc {

using envpool::mujoco::StackSpec;

/*
 * This class combines with dmc Task and Physics API.
 *
 * https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/rl/control.py
 * https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/base.py
 * https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/mujoco/engine.py
 * https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/composer/task.py
 * https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/composer/environment.py
 */
class MujocoEnv : public RenderableEnv {
 private:
  std::array<char, 1000> error_;

 protected:
  mjModel* model_;
  mjData* data_;
  int n_sub_steps_, max_episode_steps_, elapsed_step_;
  float reward_, discount_;
  bool done_{true};
#ifdef ENVPOOL_TEST
  std::unique_ptr<mjtNum> qpos0_;
#endif
  std::unique_ptr<envpool::mujoco::OffscreenRenderer> renderer_;
  envpool::mujoco::FrameStackBuffer frame_stack_buffer_;

 public:
  MujocoEnv(const std::string& base_path, const std::string& raw_xml,
            int n_sub_steps, int max_episode_steps, int frame_stack);
  ~MujocoEnv() override;

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

  // randomizer
  // https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/utils/randomizers.py#L35
  void RandomizeLimitedAndRotationalJoints(std::mt19937* gen);

  void Render(int width, int height, int camera_id,
              unsigned char* rgb) override {
    if (renderer_ == nullptr) {
      renderer_ = std::make_unique<envpool::mujoco::OffscreenRenderer>(
          envpool::mujoco::CameraPolicy::kDmControl);
    }
    renderer_->Render(model_, data_, width, height, camera_id, rgb);
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

  void AssignObservation(std::string_view key, Array* target,
                         const mjtNum* data, std::size_t size, bool reset) {
    frame_stack_buffer_.Assign(key, target, data, size, reset);
  }

  void AssignObservation(std::string_view key, Array* target, mjtNum value,
                         bool reset) {
    frame_stack_buffer_.AssignScalar(key, target, value, reset);
  }
};

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_MUJOCO_ENV_H_
