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

#include "envpool/core/py_envpool.h"
#include "envpool/mujoco/playground/aloha.h"
#include "envpool/mujoco/playground/apollo.h"
#include "envpool/mujoco/playground/barkour.h"
#include "envpool/mujoco/playground/berkeley_humanoid.h"
#include "envpool/mujoco/playground/g1.h"
#include "envpool/mujoco/playground/go1.h"
#include "envpool/mujoco/playground/h1.h"
#include "envpool/mujoco/playground/hand.h"
#include "envpool/mujoco/playground/op3.h"
#include "envpool/mujoco/playground/panda.h"
#include "envpool/mujoco/playground/panda_robotiq.h"
#include "envpool/mujoco/playground/spot.h"
#include "envpool/mujoco/playground/t1.h"

#define REGISTER_PLAYGROUND_ENV(MODULE, NAME)                        \
  using NAME##EnvSpec = PyEnvSpec<mujoco_playground::NAME##EnvSpec>; \
  using NAME##EnvPool = PyEnvPool<mujoco_playground::NAME##EnvPool>; \
  REGISTER(MODULE, NAME##EnvSpec, NAME##EnvPool)

PYBIND11_MODULE(playground_envpool, m) {
  REGISTER_PLAYGROUND_ENV(m, PlaygroundAloha)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundAlohaPixel)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundApollo)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundApolloPixel)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundBarkour)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundBarkourPixel)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundBerkeleyHumanoid)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundBerkeleyHumanoidPixel)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundG1)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundG1Pixel)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundGo1)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundGo1Pixel)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundGo1Getup)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundGo1GetupPixel)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundGo1Handstand)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundGo1HandstandPixel)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundH1)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundH1Pixel)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundHand)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundHandPixel)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundOp3)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundOp3Pixel)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundPanda)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundPandaPixel)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundPandaRobotiq)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundPandaRobotiqPixel)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundSpotJoystick)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundSpotJoystickPixel)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundSpotGetup)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundSpotGetupPixel)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundSpotGait)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundSpotGaitPixel)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundT1)
  REGISTER_PLAYGROUND_ENV(m, PlaygroundT1Pixel)
}

#undef REGISTER_PLAYGROUND_ENV
