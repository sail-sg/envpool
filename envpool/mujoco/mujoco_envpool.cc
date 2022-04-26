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

#include "envpool/core/py_envpool.h"
#include "envpool/mujoco/ant.h"
#include "envpool/mujoco/half_cheetah.h"
#include "envpool/mujoco/hopper.h"
#include "envpool/mujoco/humanoid.h"
#include "envpool/mujoco/humanoid_standup.h"
#include "envpool/mujoco/inverted_pendulum.h"
#include "envpool/mujoco/pusher.h"
#include "envpool/mujoco/reacher.h"
#include "envpool/mujoco/swimmer.h"

typedef PyEnvSpec<mujoco::AntEnvSpec> AntEnvSpec;
typedef PyEnvPool<mujoco::AntEnvPool> AntEnvPool;

typedef PyEnvSpec<mujoco::HalfCheetahEnvSpec> HalfCheetahEnvSpec;
typedef PyEnvPool<mujoco::HalfCheetahEnvPool> HalfCheetahEnvPool;

typedef PyEnvSpec<mujoco::HopperEnvSpec> HopperEnvSpec;
typedef PyEnvPool<mujoco::HopperEnvPool> HopperEnvPool;

typedef PyEnvSpec<mujoco::HumanoidEnvSpec> HumanoidEnvSpec;
typedef PyEnvPool<mujoco::HumanoidEnvPool> HumanoidEnvPool;

typedef PyEnvSpec<mujoco::HumanoidStandupEnvSpec> HumanoidStandupEnvSpec;
typedef PyEnvPool<mujoco::HumanoidStandupEnvPool> HumanoidStandupEnvPool;

typedef PyEnvSpec<mujoco::InvertedPendulumEnvSpec> InvertedPendulumEnvSpec;
typedef PyEnvPool<mujoco::InvertedPendulumEnvPool> InvertedPendulumEnvPool;

typedef PyEnvSpec<mujoco::PusherEnvSpec> PusherEnvSpec;
typedef PyEnvPool<mujoco::PusherEnvPool> PusherEnvPool;

typedef PyEnvSpec<mujoco::ReacherEnvSpec> ReacherEnvSpec;
typedef PyEnvPool<mujoco::ReacherEnvPool> ReacherEnvPool;

typedef PyEnvSpec<mujoco::SwimmerEnvSpec> SwimmerEnvSpec;
typedef PyEnvPool<mujoco::SwimmerEnvPool> SwimmerEnvPool;

PYBIND11_MODULE(mujoco_envpool, m) {
  REGISTER(m, AntEnvSpec, AntEnvPool)
  REGISTER(m, HalfCheetahEnvSpec, HalfCheetahEnvPool)
  REGISTER(m, HopperEnvSpec, HopperEnvPool)
  REGISTER(m, HumanoidEnvSpec, HumanoidEnvPool)
  REGISTER(m, HumanoidStandupEnvSpec, HumanoidStandupEnvPool)
  REGISTER(m, InvertedPendulumEnvSpec, InvertedPendulumEnvPool)
  REGISTER(m, PusherEnvSpec, PusherEnvPool)
  REGISTER(m, ReacherEnvSpec, ReacherEnvPool)
  REGISTER(m, SwimmerEnvSpec, SwimmerEnvPool)
}
