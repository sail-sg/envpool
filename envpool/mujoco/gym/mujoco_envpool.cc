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
#include "envpool/mujoco/gym/ant.h"
#include "envpool/mujoco/gym/half_cheetah.h"
#include "envpool/mujoco/gym/hopper.h"
#include "envpool/mujoco/gym/humanoid.h"
#include "envpool/mujoco/gym/humanoid_standup.h"
#include "envpool/mujoco/gym/inverted_double_pendulum.h"
#include "envpool/mujoco/gym/inverted_pendulum.h"
#include "envpool/mujoco/gym/pusher.h"
#include "envpool/mujoco/gym/reacher.h"
#include "envpool/mujoco/gym/swimmer.h"
#include "envpool/mujoco/gym/walker2d.h"

using GymAntEnvSpec = PyEnvSpec<mujoco_gym::AntEnvSpec>;
using GymAntEnvPool = PyEnvPool<mujoco_gym::AntEnvPool>;

using GymAntPixelEnvSpec = PyEnvSpec<mujoco_gym::AntPixelEnvSpec>;
using GymAntPixelEnvPool = PyEnvPool<mujoco_gym::AntPixelEnvPool>;

using GymHalfCheetahEnvSpec = PyEnvSpec<mujoco_gym::HalfCheetahEnvSpec>;
using GymHalfCheetahEnvPool = PyEnvPool<mujoco_gym::HalfCheetahEnvPool>;

using GymHalfCheetahPixelEnvSpec =
    PyEnvSpec<mujoco_gym::HalfCheetahPixelEnvSpec>;
using GymHalfCheetahPixelEnvPool =
    PyEnvPool<mujoco_gym::HalfCheetahPixelEnvPool>;

using GymHopperEnvSpec = PyEnvSpec<mujoco_gym::HopperEnvSpec>;
using GymHopperEnvPool = PyEnvPool<mujoco_gym::HopperEnvPool>;

using GymHopperPixelEnvSpec = PyEnvSpec<mujoco_gym::HopperPixelEnvSpec>;
using GymHopperPixelEnvPool = PyEnvPool<mujoco_gym::HopperPixelEnvPool>;

using GymHumanoidEnvSpec = PyEnvSpec<mujoco_gym::HumanoidEnvSpec>;
using GymHumanoidEnvPool = PyEnvPool<mujoco_gym::HumanoidEnvPool>;

using GymHumanoidPixelEnvSpec = PyEnvSpec<mujoco_gym::HumanoidPixelEnvSpec>;
using GymHumanoidPixelEnvPool = PyEnvPool<mujoco_gym::HumanoidPixelEnvPool>;

using GymHumanoidStandupEnvSpec = PyEnvSpec<mujoco_gym::HumanoidStandupEnvSpec>;
using GymHumanoidStandupEnvPool = PyEnvPool<mujoco_gym::HumanoidStandupEnvPool>;

using GymHumanoidStandupPixelEnvSpec =
    PyEnvSpec<mujoco_gym::HumanoidStandupPixelEnvSpec>;
using GymHumanoidStandupPixelEnvPool =
    PyEnvPool<mujoco_gym::HumanoidStandupPixelEnvPool>;

using GymInvertedDoublePendulumEnvSpec =
    PyEnvSpec<mujoco_gym::InvertedDoublePendulumEnvSpec>;
using GymInvertedDoublePendulumEnvPool =
    PyEnvPool<mujoco_gym::InvertedDoublePendulumEnvPool>;

using GymInvertedDoublePendulumPixelEnvSpec =
    PyEnvSpec<mujoco_gym::InvertedDoublePendulumPixelEnvSpec>;
using GymInvertedDoublePendulumPixelEnvPool =
    PyEnvPool<mujoco_gym::InvertedDoublePendulumPixelEnvPool>;

using GymInvertedPendulumEnvSpec =
    PyEnvSpec<mujoco_gym::InvertedPendulumEnvSpec>;
using GymInvertedPendulumEnvPool =
    PyEnvPool<mujoco_gym::InvertedPendulumEnvPool>;

using GymInvertedPendulumPixelEnvSpec =
    PyEnvSpec<mujoco_gym::InvertedPendulumPixelEnvSpec>;
using GymInvertedPendulumPixelEnvPool =
    PyEnvPool<mujoco_gym::InvertedPendulumPixelEnvPool>;

using GymPusherEnvSpec = PyEnvSpec<mujoco_gym::PusherEnvSpec>;
using GymPusherEnvPool = PyEnvPool<mujoco_gym::PusherEnvPool>;

using GymPusherPixelEnvSpec = PyEnvSpec<mujoco_gym::PusherPixelEnvSpec>;
using GymPusherPixelEnvPool = PyEnvPool<mujoco_gym::PusherPixelEnvPool>;

using GymReacherEnvSpec = PyEnvSpec<mujoco_gym::ReacherEnvSpec>;
using GymReacherEnvPool = PyEnvPool<mujoco_gym::ReacherEnvPool>;

using GymReacherPixelEnvSpec = PyEnvSpec<mujoco_gym::ReacherPixelEnvSpec>;
using GymReacherPixelEnvPool = PyEnvPool<mujoco_gym::ReacherPixelEnvPool>;

using GymSwimmerEnvSpec = PyEnvSpec<mujoco_gym::SwimmerEnvSpec>;
using GymSwimmerEnvPool = PyEnvPool<mujoco_gym::SwimmerEnvPool>;

using GymSwimmerPixelEnvSpec = PyEnvSpec<mujoco_gym::SwimmerPixelEnvSpec>;
using GymSwimmerPixelEnvPool = PyEnvPool<mujoco_gym::SwimmerPixelEnvPool>;

using GymWalker2dEnvSpec = PyEnvSpec<mujoco_gym::Walker2dEnvSpec>;
using GymWalker2dEnvPool = PyEnvPool<mujoco_gym::Walker2dEnvPool>;

using GymWalker2dPixelEnvSpec = PyEnvSpec<mujoco_gym::Walker2dPixelEnvSpec>;
using GymWalker2dPixelEnvPool = PyEnvPool<mujoco_gym::Walker2dPixelEnvPool>;

PYBIND11_MODULE(mujoco_gym_envpool, m) {
  REGISTER(m, GymAntEnvSpec, GymAntEnvPool)
  REGISTER(m, GymAntPixelEnvSpec, GymAntPixelEnvPool)
  REGISTER(m, GymHalfCheetahEnvSpec, GymHalfCheetahEnvPool)
  REGISTER(m, GymHalfCheetahPixelEnvSpec, GymHalfCheetahPixelEnvPool)
  REGISTER(m, GymHopperEnvSpec, GymHopperEnvPool)
  REGISTER(m, GymHopperPixelEnvSpec, GymHopperPixelEnvPool)
  REGISTER(m, GymHumanoidEnvSpec, GymHumanoidEnvPool)
  REGISTER(m, GymHumanoidPixelEnvSpec, GymHumanoidPixelEnvPool)
  REGISTER(m, GymHumanoidStandupEnvSpec, GymHumanoidStandupEnvPool)
  REGISTER(m, GymHumanoidStandupPixelEnvSpec, GymHumanoidStandupPixelEnvPool)
  REGISTER(m, GymInvertedDoublePendulumEnvSpec,
           GymInvertedDoublePendulumEnvPool)
  REGISTER(m, GymInvertedDoublePendulumPixelEnvSpec,
           GymInvertedDoublePendulumPixelEnvPool)
  REGISTER(m, GymInvertedPendulumEnvSpec, GymInvertedPendulumEnvPool)
  REGISTER(m, GymInvertedPendulumPixelEnvSpec, GymInvertedPendulumPixelEnvPool)
  REGISTER(m, GymPusherEnvSpec, GymPusherEnvPool)
  REGISTER(m, GymPusherPixelEnvSpec, GymPusherPixelEnvPool)
  REGISTER(m, GymReacherEnvSpec, GymReacherEnvPool)
  REGISTER(m, GymReacherPixelEnvSpec, GymReacherPixelEnvPool)
  REGISTER(m, GymSwimmerEnvSpec, GymSwimmerEnvPool)
  REGISTER(m, GymSwimmerPixelEnvSpec, GymSwimmerPixelEnvPool)
  REGISTER(m, GymWalker2dEnvSpec, GymWalker2dEnvPool)
  REGISTER(m, GymWalker2dPixelEnvSpec, GymWalker2dPixelEnvPool)
}
