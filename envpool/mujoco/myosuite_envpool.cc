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
#include "envpool/mujoco/myosuite/myobase.h"
#include "envpool/mujoco/myosuite/myobase_extended.h"
#include "envpool/mujoco/myosuite/myochallenge.h"
#include "envpool/mujoco/myosuite/myochallenge_extended.h"
#include "envpool/mujoco/myosuite/myodm.h"

using MyoSuitePoseEnvSpec = PyEnvSpec<myosuite_envpool::MyoSuitePoseEnvSpec>;
using MyoSuitePoseEnvPool = PyEnvPool<myosuite_envpool::MyoSuitePoseEnvPool>;
using MyoSuitePosePixelEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoSuitePosePixelEnvSpec>;
using MyoSuitePosePixelEnvPool =
    PyEnvPool<myosuite_envpool::MyoSuitePosePixelEnvPool>;

using MyoSuiteReachEnvSpec = PyEnvSpec<myosuite_envpool::MyoSuiteReachEnvSpec>;
using MyoSuiteReachEnvPool = PyEnvPool<myosuite_envpool::MyoSuiteReachEnvPool>;
using MyoSuiteReachPixelEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoSuiteReachPixelEnvSpec>;
using MyoSuiteReachPixelEnvPool =
    PyEnvPool<myosuite_envpool::MyoSuiteReachPixelEnvPool>;

using MyoSuiteKeyTurnEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoSuiteKeyTurnEnvSpec>;
using MyoSuiteKeyTurnEnvPool =
    PyEnvPool<myosuite_envpool::MyoSuiteKeyTurnEnvPool>;
using MyoSuiteKeyTurnPixelEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoSuiteKeyTurnPixelEnvSpec>;
using MyoSuiteKeyTurnPixelEnvPool =
    PyEnvPool<myosuite_envpool::MyoSuiteKeyTurnPixelEnvPool>;

using MyoSuiteObjHoldEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoSuiteObjHoldEnvSpec>;
using MyoSuiteObjHoldEnvPool =
    PyEnvPool<myosuite_envpool::MyoSuiteObjHoldEnvPool>;
using MyoSuiteObjHoldPixelEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoSuiteObjHoldPixelEnvSpec>;
using MyoSuiteObjHoldPixelEnvPool =
    PyEnvPool<myosuite_envpool::MyoSuiteObjHoldPixelEnvPool>;

using MyoSuiteTorsoEnvSpec = PyEnvSpec<myosuite_envpool::MyoSuiteTorsoEnvSpec>;
using MyoSuiteTorsoEnvPool = PyEnvPool<myosuite_envpool::MyoSuiteTorsoEnvPool>;
using MyoSuiteTorsoPixelEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoSuiteTorsoPixelEnvSpec>;
using MyoSuiteTorsoPixelEnvPool =
    PyEnvPool<myosuite_envpool::MyoSuiteTorsoPixelEnvPool>;

using MyoSuitePenTwirlEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoSuitePenTwirlEnvSpec>;
using MyoSuitePenTwirlEnvPool =
    PyEnvPool<myosuite_envpool::MyoSuitePenTwirlEnvPool>;
using MyoSuitePenTwirlPixelEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoSuitePenTwirlPixelEnvSpec>;
using MyoSuitePenTwirlPixelEnvPool =
    PyEnvPool<myosuite_envpool::MyoSuitePenTwirlPixelEnvPool>;

using MyoSuiteReorientEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoSuiteReorientEnvSpec>;
using MyoSuiteReorientEnvPool =
    PyEnvPool<myosuite_envpool::MyoSuiteReorientEnvPool>;
using MyoSuiteReorientPixelEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoSuiteReorientPixelEnvSpec>;
using MyoSuiteReorientPixelEnvPool =
    PyEnvPool<myosuite_envpool::MyoSuiteReorientPixelEnvPool>;

using MyoSuiteWalkEnvSpec = PyEnvSpec<myosuite_envpool::MyoSuiteWalkEnvSpec>;
using MyoSuiteWalkEnvPool = PyEnvPool<myosuite_envpool::MyoSuiteWalkEnvPool>;
using MyoSuiteWalkPixelEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoSuiteWalkPixelEnvSpec>;
using MyoSuiteWalkPixelEnvPool =
    PyEnvPool<myosuite_envpool::MyoSuiteWalkPixelEnvPool>;

using MyoSuiteTerrainEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoSuiteTerrainEnvSpec>;
using MyoSuiteTerrainEnvPool =
    PyEnvPool<myosuite_envpool::MyoSuiteTerrainEnvPool>;
using MyoSuiteTerrainPixelEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoSuiteTerrainPixelEnvSpec>;
using MyoSuiteTerrainPixelEnvPool =
    PyEnvPool<myosuite_envpool::MyoSuiteTerrainPixelEnvPool>;

using MyoChallengeReorientEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoChallengeReorientEnvSpec>;
using MyoChallengeReorientEnvPool =
    PyEnvPool<myosuite_envpool::MyoChallengeReorientEnvPool>;
using MyoChallengeReorientPixelEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoChallengeReorientPixelEnvSpec>;
using MyoChallengeReorientPixelEnvPool =
    PyEnvPool<myosuite_envpool::MyoChallengeReorientPixelEnvPool>;

using MyoChallengeRelocateEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoChallengeRelocateEnvSpec>;
using MyoChallengeRelocateEnvPool =
    PyEnvPool<myosuite_envpool::MyoChallengeRelocateEnvPool>;
using MyoChallengeRelocatePixelEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoChallengeRelocatePixelEnvSpec>;
using MyoChallengeRelocatePixelEnvPool =
    PyEnvPool<myosuite_envpool::MyoChallengeRelocatePixelEnvPool>;

using MyoChallengeBaodingEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoChallengeBaodingEnvSpec>;
using MyoChallengeBaodingEnvPool =
    PyEnvPool<myosuite_envpool::MyoChallengeBaodingEnvPool>;
using MyoChallengeBaodingPixelEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoChallengeBaodingPixelEnvSpec>;
using MyoChallengeBaodingPixelEnvPool =
    PyEnvPool<myosuite_envpool::MyoChallengeBaodingPixelEnvPool>;

using MyoChallengeBimanualEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoChallengeBimanualEnvSpec>;
using MyoChallengeBimanualEnvPool =
    PyEnvPool<myosuite_envpool::MyoChallengeBimanualEnvPool>;
using MyoChallengeBimanualPixelEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoChallengeBimanualPixelEnvSpec>;
using MyoChallengeBimanualPixelEnvPool =
    PyEnvPool<myosuite_envpool::MyoChallengeBimanualPixelEnvPool>;

using MyoChallengeRunTrackEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoChallengeRunTrackEnvSpec>;
using MyoChallengeRunTrackEnvPool =
    PyEnvPool<myosuite_envpool::MyoChallengeRunTrackEnvPool>;
using MyoChallengeRunTrackPixelEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoChallengeRunTrackPixelEnvSpec>;
using MyoChallengeRunTrackPixelEnvPool =
    PyEnvPool<myosuite_envpool::MyoChallengeRunTrackPixelEnvPool>;

using MyoChallengeSoccerEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoChallengeSoccerEnvSpec>;
using MyoChallengeSoccerEnvPool =
    PyEnvPool<myosuite_envpool::MyoChallengeSoccerEnvPool>;
using MyoChallengeSoccerPixelEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoChallengeSoccerPixelEnvSpec>;
using MyoChallengeSoccerPixelEnvPool =
    PyEnvPool<myosuite_envpool::MyoChallengeSoccerPixelEnvPool>;

using MyoChallengeChaseTagEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoChallengeChaseTagEnvSpec>;
using MyoChallengeChaseTagEnvPool =
    PyEnvPool<myosuite_envpool::MyoChallengeChaseTagEnvPool>;
using MyoChallengeChaseTagPixelEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoChallengeChaseTagPixelEnvSpec>;
using MyoChallengeChaseTagPixelEnvPool =
    PyEnvPool<myosuite_envpool::MyoChallengeChaseTagPixelEnvPool>;

using MyoChallengeTableTennisEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoChallengeTableTennisEnvSpec>;
using MyoChallengeTableTennisEnvPool =
    PyEnvPool<myosuite_envpool::MyoChallengeTableTennisEnvPool>;
using MyoChallengeTableTennisPixelEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoChallengeTableTennisPixelEnvSpec>;
using MyoChallengeTableTennisPixelEnvPool =
    PyEnvPool<myosuite_envpool::MyoChallengeTableTennisPixelEnvPool>;

using MyoDMTrackEnvSpec = PyEnvSpec<myosuite_envpool::MyoDMTrackEnvSpec>;
using MyoDMTrackEnvPool = PyEnvPool<myosuite_envpool::MyoDMTrackEnvPool>;
using MyoDMTrackPixelEnvSpec =
    PyEnvSpec<myosuite_envpool::MyoDMTrackPixelEnvSpec>;
using MyoDMTrackPixelEnvPool =
    PyEnvPool<myosuite_envpool::MyoDMTrackPixelEnvPool>;

PYBIND11_MODULE(myosuite_envpool, m) {
  REGISTER(m, MyoSuitePoseEnvSpec, MyoSuitePoseEnvPool)
  REGISTER(m, MyoSuitePosePixelEnvSpec, MyoSuitePosePixelEnvPool)
  REGISTER(m, MyoSuiteReachEnvSpec, MyoSuiteReachEnvPool)
  REGISTER(m, MyoSuiteReachPixelEnvSpec, MyoSuiteReachPixelEnvPool)
  REGISTER(m, MyoSuiteKeyTurnEnvSpec, MyoSuiteKeyTurnEnvPool)
  REGISTER(m, MyoSuiteKeyTurnPixelEnvSpec, MyoSuiteKeyTurnPixelEnvPool)
  REGISTER(m, MyoSuiteObjHoldEnvSpec, MyoSuiteObjHoldEnvPool)
  REGISTER(m, MyoSuiteObjHoldPixelEnvSpec, MyoSuiteObjHoldPixelEnvPool)
  REGISTER(m, MyoSuiteTorsoEnvSpec, MyoSuiteTorsoEnvPool)
  REGISTER(m, MyoSuiteTorsoPixelEnvSpec, MyoSuiteTorsoPixelEnvPool)
  REGISTER(m, MyoSuitePenTwirlEnvSpec, MyoSuitePenTwirlEnvPool)
  REGISTER(m, MyoSuitePenTwirlPixelEnvSpec, MyoSuitePenTwirlPixelEnvPool)
  REGISTER(m, MyoSuiteReorientEnvSpec, MyoSuiteReorientEnvPool)
  REGISTER(m, MyoSuiteReorientPixelEnvSpec, MyoSuiteReorientPixelEnvPool)
  REGISTER(m, MyoSuiteWalkEnvSpec, MyoSuiteWalkEnvPool)
  REGISTER(m, MyoSuiteWalkPixelEnvSpec, MyoSuiteWalkPixelEnvPool)
  REGISTER(m, MyoSuiteTerrainEnvSpec, MyoSuiteTerrainEnvPool)
  REGISTER(m, MyoSuiteTerrainPixelEnvSpec, MyoSuiteTerrainPixelEnvPool)
  REGISTER(m, MyoChallengeReorientEnvSpec, MyoChallengeReorientEnvPool)
  REGISTER(m, MyoChallengeReorientPixelEnvSpec,
           MyoChallengeReorientPixelEnvPool)
  REGISTER(m, MyoChallengeRelocateEnvSpec, MyoChallengeRelocateEnvPool)
  REGISTER(m, MyoChallengeRelocatePixelEnvSpec,
           MyoChallengeRelocatePixelEnvPool)
  REGISTER(m, MyoChallengeBaodingEnvSpec, MyoChallengeBaodingEnvPool)
  REGISTER(m, MyoChallengeBaodingPixelEnvSpec,
           MyoChallengeBaodingPixelEnvPool)
  REGISTER(m, MyoChallengeBimanualEnvSpec, MyoChallengeBimanualEnvPool)
  REGISTER(m, MyoChallengeBimanualPixelEnvSpec,
           MyoChallengeBimanualPixelEnvPool)
  REGISTER(m, MyoChallengeRunTrackEnvSpec, MyoChallengeRunTrackEnvPool)
  REGISTER(m, MyoChallengeRunTrackPixelEnvSpec,
           MyoChallengeRunTrackPixelEnvPool)
  REGISTER(m, MyoChallengeSoccerEnvSpec, MyoChallengeSoccerEnvPool)
  REGISTER(m, MyoChallengeSoccerPixelEnvSpec,
           MyoChallengeSoccerPixelEnvPool)
  REGISTER(m, MyoChallengeChaseTagEnvSpec, MyoChallengeChaseTagEnvPool)
  REGISTER(m, MyoChallengeChaseTagPixelEnvSpec,
           MyoChallengeChaseTagPixelEnvPool)
  REGISTER(m, MyoChallengeTableTennisEnvSpec, MyoChallengeTableTennisEnvPool)
  REGISTER(m, MyoChallengeTableTennisPixelEnvSpec,
           MyoChallengeTableTennisPixelEnvPool)
  REGISTER(m, MyoDMTrackEnvSpec, MyoDMTrackEnvPool)
  REGISTER(m, MyoDMTrackPixelEnvSpec, MyoDMTrackPixelEnvPool)
}
