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
#include "envpool/mujoco/inverted_double_pendulum.h"
#include "envpool/mujoco/inverted_pendulum.h"
#include "envpool/mujoco/pusher.h"
#include "envpool/mujoco/reacher.h"
#include "envpool/mujoco/swimmer.h"
#include "envpool/mujoco/walker2d.h"

using AntEnvSpec = PyEnvSpec<mujoco::AntEnvSpec>;
using AntEnvPool = PyEnvPool<mujoco::AntEnvPool>;

using HalfCheetahEnvSpec = PyEnvSpec<mujoco::HalfCheetahEnvSpec>;
using HalfCheetahEnvPool = PyEnvPool<mujoco::HalfCheetahEnvPool>;

using HopperEnvSpec = PyEnvSpec<mujoco::HopperEnvSpec>;
using HopperEnvPool = PyEnvPool<mujoco::HopperEnvPool>;

using HumanoidEnvSpec = PyEnvSpec<mujoco::HumanoidEnvSpec>;
using HumanoidEnvPool = PyEnvPool<mujoco::HumanoidEnvPool>;

using HumanoidStandupEnvSpec = PyEnvSpec<mujoco::HumanoidStandupEnvSpec>;
using HumanoidStandupEnvPool = PyEnvPool<mujoco::HumanoidStandupEnvPool>;

using InvertedDoublePendulumEnvSpec =
    PyEnvSpec<mujoco::InvertedDoublePendulumEnvSpec>;
using InvertedDoublePendulumEnvPool =
    PyEnvPool<mujoco::InvertedDoublePendulumEnvPool>;

using InvertedPendulumEnvSpec = PyEnvSpec<mujoco::InvertedPendulumEnvSpec>;
using InvertedPendulumEnvPool = PyEnvPool<mujoco::InvertedPendulumEnvPool>;

using PusherEnvSpec = PyEnvSpec<mujoco::PusherEnvSpec>;
using PusherEnvPool = PyEnvPool<mujoco::PusherEnvPool>;

using ReacherEnvSpec = PyEnvSpec<mujoco::ReacherEnvSpec>;
using ReacherEnvPool = PyEnvPool<mujoco::ReacherEnvPool>;

using SwimmerEnvSpec = PyEnvSpec<mujoco::SwimmerEnvSpec>;
using SwimmerEnvPool = PyEnvPool<mujoco::SwimmerEnvPool>;

using Walker2dEnvSpec = PyEnvSpec<mujoco::Walker2dEnvSpec>;
using Walker2dEnvPool = PyEnvPool<mujoco::Walker2dEnvPool>;

PYBIND11_MODULE(mujoco_envpool, m) {
  REGISTER(m, AntEnvSpec, AntEnvPool)
  REGISTER(m, HalfCheetahEnvSpec, HalfCheetahEnvPool)
  REGISTER(m, HopperEnvSpec, HopperEnvPool)
  REGISTER(m, HumanoidEnvSpec, HumanoidEnvPool)
  REGISTER(m, HumanoidStandupEnvSpec, HumanoidStandupEnvPool)
  REGISTER(m, InvertedDoublePendulumEnvSpec, InvertedDoublePendulumEnvPool)
  REGISTER(m, InvertedPendulumEnvSpec, InvertedPendulumEnvPool)
  REGISTER(m, PusherEnvSpec, PusherEnvPool)
  REGISTER(m, ReacherEnvSpec, ReacherEnvPool)
  REGISTER(m, SwimmerEnvSpec, SwimmerEnvPool)
  REGISTER(m, Walker2dEnvSpec, Walker2dEnvPool)
}
