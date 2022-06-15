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
#include "envpool/mujoco/dmc/acrobot.h"
#include "envpool/mujoco/dmc/ball_in_cup.h"
#include "envpool/mujoco/dmc/cartpole.h"
#include "envpool/mujoco/dmc/cheetah.h"
#include "envpool/mujoco/dmc/finger.h"
#include "envpool/mujoco/dmc/fish.h"
#include "envpool/mujoco/dmc/hopper.h"
#include "envpool/mujoco/dmc/humanoid.h"
#include "envpool/mujoco/dmc/humanoid_CMU.h"
#include "envpool/mujoco/dmc/manipulator.h"
#include "envpool/mujoco/dmc/pendulum.h"
#include "envpool/mujoco/dmc/point_mass.h"
#include "envpool/mujoco/dmc/reacher.h"
#include "envpool/mujoco/dmc/swimmer.h"
#include "envpool/mujoco/dmc/walker.h"

using DmcAcrobotEnvSpec = PyEnvSpec<mujoco_dmc::AcrobotEnvSpec>;
using DmcAcrobotEnvPool = PyEnvPool<mujoco_dmc::AcrobotEnvPool>;

using DmcBallInCupEnvSpec = PyEnvSpec<mujoco_dmc::BallInCupEnvSpec>;
using DmcBallInCupEnvPool = PyEnvPool<mujoco_dmc::BallInCupEnvPool>;

using DmcCartpoleEnvSpec = PyEnvSpec<mujoco_dmc::CartpoleEnvSpec>;
using DmcCartpoleEnvPool = PyEnvPool<mujoco_dmc::CartpoleEnvPool>;

using DmcCheetahEnvSpec = PyEnvSpec<mujoco_dmc::CheetahEnvSpec>;
using DmcCheetahEnvPool = PyEnvPool<mujoco_dmc::CheetahEnvPool>;

using DmcFingerEnvSpec = PyEnvSpec<mujoco_dmc::FingerEnvSpec>;
using DmcFingerEnvPool = PyEnvPool<mujoco_dmc::FingerEnvPool>;

using DmcFishEnvSpec = PyEnvSpec<mujoco_dmc::FishEnvSpec>;
using DmcFishEnvPool = PyEnvPool<mujoco_dmc::FishEnvPool>;

using DmcHopperEnvSpec = PyEnvSpec<mujoco_dmc::HopperEnvSpec>;
using DmcHopperEnvPool = PyEnvPool<mujoco_dmc::HopperEnvPool>;

using DmcHumanoidEnvSpec = PyEnvSpec<mujoco_dmc::HumanoidEnvSpec>;
using DmcHumanoidEnvPool = PyEnvPool<mujoco_dmc::HumanoidEnvPool>;

using DmcHumanoidCMUEnvSpec = PyEnvSpec<mujoco_dmc::HumanoidCMUEnvSpec>;
using DmcHumanoidCMUEnvPool = PyEnvPool<mujoco_dmc::HumanoidCMUEnvPool>;

using DmcManipulatorEnvSpec = PyEnvSpec<mujoco_dmc::ManipulatorEnvSpec>;
using DmcManipulatorEnvPool = PyEnvPool<mujoco_dmc::ManipulatorEnvPool>;

using DmcPendulumEnvSpec = PyEnvSpec<mujoco_dmc::PendulumEnvSpec>;
using DmcPendulumEnvPool = PyEnvPool<mujoco_dmc::PendulumEnvPool>;

using DmcPointMassEnvSpec = PyEnvSpec<mujoco_dmc::PointMassEnvSpec>;
using DmcPointMassEnvPool = PyEnvPool<mujoco_dmc::PointMassEnvPool>;

using DmcReacherEnvSpec = PyEnvSpec<mujoco_dmc::ReacherEnvSpec>;
using DmcReacherEnvPool = PyEnvPool<mujoco_dmc::ReacherEnvPool>;

using DmcSwimmerEnvSpec = PyEnvSpec<mujoco_dmc::SwimmerEnvSpec>;
using DmcSwimmerEnvPool = PyEnvPool<mujoco_dmc::SwimmerEnvPool>;

using DmcWalkerEnvSpec = PyEnvSpec<mujoco_dmc::WalkerEnvSpec>;
using DmcWalkerEnvPool = PyEnvPool<mujoco_dmc::WalkerEnvPool>;

PYBIND11_MODULE(mujoco_dmc_envpool, m) {
  REGISTER(m, DmcAcrobotEnvSpec, DmcAcrobotEnvPool)
  REGISTER(m, DmcBallInCupEnvSpec, DmcBallInCupEnvPool)
  REGISTER(m, DmcCartpoleEnvSpec, DmcCartpoleEnvPool)
  REGISTER(m, DmcCheetahEnvSpec, DmcCheetahEnvPool)
  REGISTER(m, DmcFingerEnvSpec, DmcFingerEnvPool)
  REGISTER(m, DmcFishEnvSpec, DmcFishEnvPool)
  REGISTER(m, DmcHopperEnvSpec, DmcHopperEnvPool)
  REGISTER(m, DmcHumanoidEnvSpec, DmcHumanoidEnvPool)
  REGISTER(m, DmcHumanoidCMUEnvSpec, DmcHumanoidCMUEnvPool)
  REGISTER(m, DmcManipulatorEnvSpec, DmcManipulatorEnvPool)
  REGISTER(m, DmcPendulumEnvSpec, DmcPendulumEnvPool)
  REGISTER(m, DmcPointMassEnvSpec, DmcPointMassEnvPool)
  REGISTER(m, DmcReacherEnvSpec, DmcReacherEnvPool)
  REGISTER(m, DmcSwimmerEnvSpec, DmcSwimmerEnvPool)
  REGISTER(m, DmcWalkerEnvSpec, DmcWalkerEnvPool)
}
