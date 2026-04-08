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
#include "envpool/mujoco/dmc/dog.h"
#include "envpool/mujoco/dmc/finger.h"
#include "envpool/mujoco/dmc/fish.h"
#include "envpool/mujoco/dmc/hopper.h"
#include "envpool/mujoco/dmc/humanoid.h"
#include "envpool/mujoco/dmc/humanoid_CMU.h"
#include "envpool/mujoco/dmc/lqr.h"
#include "envpool/mujoco/dmc/manipulator.h"
#include "envpool/mujoco/dmc/pendulum.h"
#include "envpool/mujoco/dmc/point_mass.h"
#include "envpool/mujoco/dmc/quadruped.h"
#include "envpool/mujoco/dmc/reacher.h"
#include "envpool/mujoco/dmc/stacker.h"
#include "envpool/mujoco/dmc/swimmer.h"
#include "envpool/mujoco/dmc/walker.h"

using DmcAcrobotEnvSpec = PyEnvSpec<mujoco_dmc::AcrobotEnvSpec>;
using DmcAcrobotEnvPool = PyEnvPool<mujoco_dmc::AcrobotEnvPool>;
using DmcAcrobotPixelEnvSpec = PyEnvSpec<mujoco_dmc::AcrobotPixelEnvSpec>;
using DmcAcrobotPixelEnvPool = PyEnvPool<mujoco_dmc::AcrobotPixelEnvPool>;

using DmcBallInCupEnvSpec = PyEnvSpec<mujoco_dmc::BallInCupEnvSpec>;
using DmcBallInCupEnvPool = PyEnvPool<mujoco_dmc::BallInCupEnvPool>;
using DmcBallInCupPixelEnvSpec = PyEnvSpec<mujoco_dmc::BallInCupPixelEnvSpec>;
using DmcBallInCupPixelEnvPool = PyEnvPool<mujoco_dmc::BallInCupPixelEnvPool>;

using DmcCartpoleEnvSpec = PyEnvSpec<mujoco_dmc::CartpoleEnvSpec>;
using DmcCartpoleEnvPool = PyEnvPool<mujoco_dmc::CartpoleEnvPool>;
using DmcCartpolePixelEnvSpec = PyEnvSpec<mujoco_dmc::CartpolePixelEnvSpec>;
using DmcCartpolePixelEnvPool = PyEnvPool<mujoco_dmc::CartpolePixelEnvPool>;

using DmcCheetahEnvSpec = PyEnvSpec<mujoco_dmc::CheetahEnvSpec>;
using DmcCheetahEnvPool = PyEnvPool<mujoco_dmc::CheetahEnvPool>;
using DmcCheetahPixelEnvSpec = PyEnvSpec<mujoco_dmc::CheetahPixelEnvSpec>;
using DmcCheetahPixelEnvPool = PyEnvPool<mujoco_dmc::CheetahPixelEnvPool>;

using DmcDogEnvSpec = PyEnvSpec<mujoco_dmc::DogEnvSpec>;
using DmcDogEnvPool = PyEnvPool<mujoco_dmc::DogEnvPool>;
using DmcDogPixelEnvSpec = PyEnvSpec<mujoco_dmc::DogPixelEnvSpec>;
using DmcDogPixelEnvPool = PyEnvPool<mujoco_dmc::DogPixelEnvPool>;

using DmcFingerEnvSpec = PyEnvSpec<mujoco_dmc::FingerEnvSpec>;
using DmcFingerEnvPool = PyEnvPool<mujoco_dmc::FingerEnvPool>;
using DmcFingerPixelEnvSpec = PyEnvSpec<mujoco_dmc::FingerPixelEnvSpec>;
using DmcFingerPixelEnvPool = PyEnvPool<mujoco_dmc::FingerPixelEnvPool>;

using DmcFishEnvSpec = PyEnvSpec<mujoco_dmc::FishEnvSpec>;
using DmcFishEnvPool = PyEnvPool<mujoco_dmc::FishEnvPool>;
using DmcFishPixelEnvSpec = PyEnvSpec<mujoco_dmc::FishPixelEnvSpec>;
using DmcFishPixelEnvPool = PyEnvPool<mujoco_dmc::FishPixelEnvPool>;

using DmcHopperEnvSpec = PyEnvSpec<mujoco_dmc::HopperEnvSpec>;
using DmcHopperEnvPool = PyEnvPool<mujoco_dmc::HopperEnvPool>;
using DmcHopperPixelEnvSpec = PyEnvSpec<mujoco_dmc::HopperPixelEnvSpec>;
using DmcHopperPixelEnvPool = PyEnvPool<mujoco_dmc::HopperPixelEnvPool>;

using DmcHumanoidEnvSpec = PyEnvSpec<mujoco_dmc::HumanoidEnvSpec>;
using DmcHumanoidEnvPool = PyEnvPool<mujoco_dmc::HumanoidEnvPool>;
using DmcHumanoidPixelEnvSpec = PyEnvSpec<mujoco_dmc::HumanoidPixelEnvSpec>;
using DmcHumanoidPixelEnvPool = PyEnvPool<mujoco_dmc::HumanoidPixelEnvPool>;

using DmcHumanoidCMUEnvSpec = PyEnvSpec<mujoco_dmc::HumanoidCMUEnvSpec>;
using DmcHumanoidCMUEnvPool = PyEnvPool<mujoco_dmc::HumanoidCMUEnvPool>;
using DmcHumanoidCMUPixelEnvSpec =
    PyEnvSpec<mujoco_dmc::HumanoidCMUPixelEnvSpec>;
using DmcHumanoidCMUPixelEnvPool =
    PyEnvPool<mujoco_dmc::HumanoidCMUPixelEnvPool>;

using DmcLqrEnvSpec = PyEnvSpec<mujoco_dmc::LqrEnvSpec>;
using DmcLqrEnvPool = PyEnvPool<mujoco_dmc::LqrEnvPool>;
using DmcLqrPixelEnvSpec = PyEnvSpec<mujoco_dmc::LqrPixelEnvSpec>;
using DmcLqrPixelEnvPool = PyEnvPool<mujoco_dmc::LqrPixelEnvPool>;

using DmcManipulatorEnvSpec = PyEnvSpec<mujoco_dmc::ManipulatorEnvSpec>;
using DmcManipulatorEnvPool = PyEnvPool<mujoco_dmc::ManipulatorEnvPool>;
using DmcManipulatorPixelEnvSpec =
    PyEnvSpec<mujoco_dmc::ManipulatorPixelEnvSpec>;
using DmcManipulatorPixelEnvPool =
    PyEnvPool<mujoco_dmc::ManipulatorPixelEnvPool>;

using DmcPendulumEnvSpec = PyEnvSpec<mujoco_dmc::PendulumEnvSpec>;
using DmcPendulumEnvPool = PyEnvPool<mujoco_dmc::PendulumEnvPool>;
using DmcPendulumPixelEnvSpec = PyEnvSpec<mujoco_dmc::PendulumPixelEnvSpec>;
using DmcPendulumPixelEnvPool = PyEnvPool<mujoco_dmc::PendulumPixelEnvPool>;

using DmcPointMassEnvSpec = PyEnvSpec<mujoco_dmc::PointMassEnvSpec>;
using DmcPointMassEnvPool = PyEnvPool<mujoco_dmc::PointMassEnvPool>;
using DmcPointMassPixelEnvSpec = PyEnvSpec<mujoco_dmc::PointMassPixelEnvSpec>;
using DmcPointMassPixelEnvPool = PyEnvPool<mujoco_dmc::PointMassPixelEnvPool>;

using DmcQuadrupedEnvSpec = PyEnvSpec<mujoco_dmc::QuadrupedEnvSpec>;
using DmcQuadrupedEnvPool = PyEnvPool<mujoco_dmc::QuadrupedEnvPool>;
using DmcQuadrupedPixelEnvSpec = PyEnvSpec<mujoco_dmc::QuadrupedPixelEnvSpec>;
using DmcQuadrupedPixelEnvPool = PyEnvPool<mujoco_dmc::QuadrupedPixelEnvPool>;

using DmcReacherEnvSpec = PyEnvSpec<mujoco_dmc::ReacherEnvSpec>;
using DmcReacherEnvPool = PyEnvPool<mujoco_dmc::ReacherEnvPool>;
using DmcReacherPixelEnvSpec = PyEnvSpec<mujoco_dmc::ReacherPixelEnvSpec>;
using DmcReacherPixelEnvPool = PyEnvPool<mujoco_dmc::ReacherPixelEnvPool>;

using DmcStackerEnvSpec = PyEnvSpec<mujoco_dmc::StackerEnvSpec>;
using DmcStackerEnvPool = PyEnvPool<mujoco_dmc::StackerEnvPool>;
using DmcStackerPixelEnvSpec = PyEnvSpec<mujoco_dmc::StackerPixelEnvSpec>;
using DmcStackerPixelEnvPool = PyEnvPool<mujoco_dmc::StackerPixelEnvPool>;

using DmcSwimmerEnvSpec = PyEnvSpec<mujoco_dmc::SwimmerEnvSpec>;
using DmcSwimmerEnvPool = PyEnvPool<mujoco_dmc::SwimmerEnvPool>;
using DmcSwimmerPixelEnvSpec = PyEnvSpec<mujoco_dmc::SwimmerPixelEnvSpec>;
using DmcSwimmerPixelEnvPool = PyEnvPool<mujoco_dmc::SwimmerPixelEnvPool>;

using DmcWalkerEnvSpec = PyEnvSpec<mujoco_dmc::WalkerEnvSpec>;
using DmcWalkerEnvPool = PyEnvPool<mujoco_dmc::WalkerEnvPool>;
using DmcWalkerPixelEnvSpec = PyEnvSpec<mujoco_dmc::WalkerPixelEnvSpec>;
using DmcWalkerPixelEnvPool = PyEnvPool<mujoco_dmc::WalkerPixelEnvPool>;

PYBIND11_MODULE(mujoco_dmc_envpool, m) {
  REGISTER(m, DmcAcrobotEnvSpec, DmcAcrobotEnvPool)
  REGISTER(m, DmcAcrobotPixelEnvSpec, DmcAcrobotPixelEnvPool)
  REGISTER(m, DmcBallInCupEnvSpec, DmcBallInCupEnvPool)
  REGISTER(m, DmcBallInCupPixelEnvSpec, DmcBallInCupPixelEnvPool)
  REGISTER(m, DmcCartpoleEnvSpec, DmcCartpoleEnvPool)
  REGISTER(m, DmcCartpolePixelEnvSpec, DmcCartpolePixelEnvPool)
  REGISTER(m, DmcCheetahEnvSpec, DmcCheetahEnvPool)
  REGISTER(m, DmcCheetahPixelEnvSpec, DmcCheetahPixelEnvPool)
  REGISTER(m, DmcDogEnvSpec, DmcDogEnvPool)
  REGISTER(m, DmcDogPixelEnvSpec, DmcDogPixelEnvPool)
  REGISTER(m, DmcFingerEnvSpec, DmcFingerEnvPool)
  REGISTER(m, DmcFingerPixelEnvSpec, DmcFingerPixelEnvPool)
  REGISTER(m, DmcFishEnvSpec, DmcFishEnvPool)
  REGISTER(m, DmcFishPixelEnvSpec, DmcFishPixelEnvPool)
  REGISTER(m, DmcHopperEnvSpec, DmcHopperEnvPool)
  REGISTER(m, DmcHopperPixelEnvSpec, DmcHopperPixelEnvPool)
  REGISTER(m, DmcHumanoidEnvSpec, DmcHumanoidEnvPool)
  REGISTER(m, DmcHumanoidPixelEnvSpec, DmcHumanoidPixelEnvPool)
  REGISTER(m, DmcHumanoidCMUEnvSpec, DmcHumanoidCMUEnvPool)
  REGISTER(m, DmcHumanoidCMUPixelEnvSpec, DmcHumanoidCMUPixelEnvPool)
  REGISTER(m, DmcLqrEnvSpec, DmcLqrEnvPool)
  REGISTER(m, DmcLqrPixelEnvSpec, DmcLqrPixelEnvPool)
  REGISTER(m, DmcManipulatorEnvSpec, DmcManipulatorEnvPool)
  REGISTER(m, DmcManipulatorPixelEnvSpec, DmcManipulatorPixelEnvPool)
  REGISTER(m, DmcPendulumEnvSpec, DmcPendulumEnvPool)
  REGISTER(m, DmcPendulumPixelEnvSpec, DmcPendulumPixelEnvPool)
  REGISTER(m, DmcPointMassEnvSpec, DmcPointMassEnvPool)
  REGISTER(m, DmcPointMassPixelEnvSpec, DmcPointMassPixelEnvPool)
  REGISTER(m, DmcQuadrupedEnvSpec, DmcQuadrupedEnvPool)
  REGISTER(m, DmcQuadrupedPixelEnvSpec, DmcQuadrupedPixelEnvPool)
  REGISTER(m, DmcReacherEnvSpec, DmcReacherEnvPool)
  REGISTER(m, DmcReacherPixelEnvSpec, DmcReacherPixelEnvPool)
  REGISTER(m, DmcStackerEnvSpec, DmcStackerEnvPool)
  REGISTER(m, DmcStackerPixelEnvSpec, DmcStackerPixelEnvPool)
  REGISTER(m, DmcSwimmerEnvSpec, DmcSwimmerEnvPool)
  REGISTER(m, DmcSwimmerPixelEnvSpec, DmcSwimmerPixelEnvPool)
  REGISTER(m, DmcWalkerEnvSpec, DmcWalkerEnvPool)
  REGISTER(m, DmcWalkerPixelEnvSpec, DmcWalkerPixelEnvPool)
}
