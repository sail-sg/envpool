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
#include "envpool/mujoco/myosuite/myochallenge.h"
#include "envpool/mujoco/myosuite/myochallenge_extended.h"

namespace myosuite_envpool {

void RegisterMyoSuiteChallengeBindings(py::module_& module) {
  using MyoChallengeReorientEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoChallengeReorientEnvSpec>;
  using MyoChallengeReorientEnvPool =
      PyEnvPool<::myosuite_envpool::MyoChallengeReorientEnvPool>;
  using MyoChallengeReorientPixelEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoChallengeReorientPixelEnvSpec>;
  using MyoChallengeReorientPixelEnvPool =
      PyEnvPool<::myosuite_envpool::MyoChallengeReorientPixelEnvPool>;

  using MyoChallengeRelocateEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoChallengeRelocateEnvSpec>;
  using MyoChallengeRelocateEnvPool =
      PyEnvPool<::myosuite_envpool::MyoChallengeRelocateEnvPool>;
  using MyoChallengeRelocatePixelEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoChallengeRelocatePixelEnvSpec>;
  using MyoChallengeRelocatePixelEnvPool =
      PyEnvPool<::myosuite_envpool::MyoChallengeRelocatePixelEnvPool>;

  using MyoChallengeBaodingEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoChallengeBaodingEnvSpec>;
  using MyoChallengeBaodingEnvPool =
      PyEnvPool<::myosuite_envpool::MyoChallengeBaodingEnvPool>;
  using MyoChallengeBaodingPixelEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoChallengeBaodingPixelEnvSpec>;
  using MyoChallengeBaodingPixelEnvPool =
      PyEnvPool<::myosuite_envpool::MyoChallengeBaodingPixelEnvPool>;

  using MyoChallengeBimanualEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoChallengeBimanualEnvSpec>;
  using MyoChallengeBimanualEnvPool =
      PyEnvPool<::myosuite_envpool::MyoChallengeBimanualEnvPool>;
  using MyoChallengeBimanualPixelEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoChallengeBimanualPixelEnvSpec>;
  using MyoChallengeBimanualPixelEnvPool =
      PyEnvPool<::myosuite_envpool::MyoChallengeBimanualPixelEnvPool>;

  using MyoChallengeRunTrackEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoChallengeRunTrackEnvSpec>;
  using MyoChallengeRunTrackEnvPool =
      PyEnvPool<::myosuite_envpool::MyoChallengeRunTrackEnvPool>;
  using MyoChallengeRunTrackPixelEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoChallengeRunTrackPixelEnvSpec>;
  using MyoChallengeRunTrackPixelEnvPool =
      PyEnvPool<::myosuite_envpool::MyoChallengeRunTrackPixelEnvPool>;

  using MyoChallengeSoccerEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoChallengeSoccerEnvSpec>;
  using MyoChallengeSoccerEnvPool =
      PyEnvPool<::myosuite_envpool::MyoChallengeSoccerEnvPool>;
  using MyoChallengeSoccerPixelEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoChallengeSoccerPixelEnvSpec>;
  using MyoChallengeSoccerPixelEnvPool =
      PyEnvPool<::myosuite_envpool::MyoChallengeSoccerPixelEnvPool>;

  using MyoChallengeChaseTagEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoChallengeChaseTagEnvSpec>;
  using MyoChallengeChaseTagEnvPool =
      PyEnvPool<::myosuite_envpool::MyoChallengeChaseTagEnvPool>;
  using MyoChallengeChaseTagPixelEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoChallengeChaseTagPixelEnvSpec>;
  using MyoChallengeChaseTagPixelEnvPool =
      PyEnvPool<::myosuite_envpool::MyoChallengeChaseTagPixelEnvPool>;

  using MyoChallengeTableTennisEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoChallengeTableTennisEnvSpec>;
  using MyoChallengeTableTennisEnvPool =
      PyEnvPool<::myosuite_envpool::MyoChallengeTableTennisEnvPool>;
  using MyoChallengeTableTennisPixelEnvSpec =
      PyEnvSpec<::myosuite_envpool::MyoChallengeTableTennisPixelEnvSpec>;
  using MyoChallengeTableTennisPixelEnvPool =
      PyEnvPool<::myosuite_envpool::MyoChallengeTableTennisPixelEnvPool>;

  REGISTER(module, MyoChallengeReorientEnvSpec, MyoChallengeReorientEnvPool)
  REGISTER(module, MyoChallengeReorientPixelEnvSpec,
           MyoChallengeReorientPixelEnvPool)
  REGISTER(module, MyoChallengeRelocateEnvSpec, MyoChallengeRelocateEnvPool)
  REGISTER(module, MyoChallengeRelocatePixelEnvSpec,
           MyoChallengeRelocatePixelEnvPool)
  REGISTER(module, MyoChallengeBaodingEnvSpec, MyoChallengeBaodingEnvPool)
  REGISTER(module, MyoChallengeBaodingPixelEnvSpec,
           MyoChallengeBaodingPixelEnvPool)
  REGISTER(module, MyoChallengeBimanualEnvSpec, MyoChallengeBimanualEnvPool)
  REGISTER(module, MyoChallengeBimanualPixelEnvSpec,
           MyoChallengeBimanualPixelEnvPool)
  REGISTER(module, MyoChallengeRunTrackEnvSpec, MyoChallengeRunTrackEnvPool)
  REGISTER(module, MyoChallengeRunTrackPixelEnvSpec,
           MyoChallengeRunTrackPixelEnvPool)
  REGISTER(module, MyoChallengeSoccerEnvSpec, MyoChallengeSoccerEnvPool)
  REGISTER(module, MyoChallengeSoccerPixelEnvSpec,
           MyoChallengeSoccerPixelEnvPool)
  REGISTER(module, MyoChallengeChaseTagEnvSpec, MyoChallengeChaseTagEnvPool)
  REGISTER(module, MyoChallengeChaseTagPixelEnvSpec,
           MyoChallengeChaseTagPixelEnvPool)
  REGISTER(module, MyoChallengeTableTennisEnvSpec,
           MyoChallengeTableTennisEnvPool)
  REGISTER(module, MyoChallengeTableTennisPixelEnvSpec,
           MyoChallengeTableTennisPixelEnvPool)
}

}  // namespace myosuite_envpool
