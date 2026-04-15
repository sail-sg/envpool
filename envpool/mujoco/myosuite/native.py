# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Internal native MyoSuite wrappers used before public registration."""

from envpool.mujoco.myosuite_envpool import (
    _MyoChallengeBaodingEnvPool,
    _MyoChallengeBaodingEnvSpec,
    _MyoChallengeBaodingPixelEnvPool,
    _MyoChallengeBaodingPixelEnvSpec,
    _MyoChallengeBimanualEnvPool,
    _MyoChallengeBimanualEnvSpec,
    _MyoChallengeBimanualPixelEnvPool,
    _MyoChallengeBimanualPixelEnvSpec,
    _MyoChallengeChaseTagEnvPool,
    _MyoChallengeChaseTagEnvSpec,
    _MyoChallengeChaseTagPixelEnvPool,
    _MyoChallengeChaseTagPixelEnvSpec,
    _MyoChallengeRelocateEnvPool,
    _MyoChallengeRelocateEnvSpec,
    _MyoChallengeRelocatePixelEnvPool,
    _MyoChallengeRelocatePixelEnvSpec,
    _MyoChallengeReorientEnvPool,
    _MyoChallengeReorientEnvSpec,
    _MyoChallengeReorientPixelEnvPool,
    _MyoChallengeReorientPixelEnvSpec,
    _MyoChallengeRunTrackEnvPool,
    _MyoChallengeRunTrackEnvSpec,
    _MyoChallengeRunTrackPixelEnvPool,
    _MyoChallengeRunTrackPixelEnvSpec,
    _MyoChallengeSoccerEnvPool,
    _MyoChallengeSoccerEnvSpec,
    _MyoChallengeSoccerPixelEnvPool,
    _MyoChallengeSoccerPixelEnvSpec,
    _MyoChallengeTableTennisEnvPool,
    _MyoChallengeTableTennisEnvSpec,
    _MyoChallengeTableTennisPixelEnvPool,
    _MyoChallengeTableTennisPixelEnvSpec,
    _MyoDMTrackEnvPool,
    _MyoDMTrackEnvSpec,
    _MyoDMTrackPixelEnvPool,
    _MyoDMTrackPixelEnvSpec,
    _MyoSuiteKeyTurnEnvPool,
    _MyoSuiteKeyTurnEnvSpec,
    _MyoSuiteKeyTurnPixelEnvPool,
    _MyoSuiteKeyTurnPixelEnvSpec,
    _MyoSuiteObjHoldEnvPool,
    _MyoSuiteObjHoldEnvSpec,
    _MyoSuiteObjHoldPixelEnvPool,
    _MyoSuiteObjHoldPixelEnvSpec,
    _MyoSuitePenTwirlEnvPool,
    _MyoSuitePenTwirlEnvSpec,
    _MyoSuitePenTwirlPixelEnvPool,
    _MyoSuitePenTwirlPixelEnvSpec,
    _MyoSuitePoseEnvPool,
    _MyoSuitePoseEnvSpec,
    _MyoSuitePosePixelEnvPool,
    _MyoSuitePosePixelEnvSpec,
    _MyoSuiteReachEnvPool,
    _MyoSuiteReachEnvSpec,
    _MyoSuiteReachPixelEnvPool,
    _MyoSuiteReachPixelEnvSpec,
    _MyoSuiteReorientEnvPool,
    _MyoSuiteReorientEnvSpec,
    _MyoSuiteReorientPixelEnvPool,
    _MyoSuiteReorientPixelEnvSpec,
    _MyoSuiteTerrainEnvPool,
    _MyoSuiteTerrainEnvSpec,
    _MyoSuiteTerrainPixelEnvPool,
    _MyoSuiteTerrainPixelEnvSpec,
    _MyoSuiteTorsoEnvPool,
    _MyoSuiteTorsoEnvSpec,
    _MyoSuiteTorsoPixelEnvPool,
    _MyoSuiteTorsoPixelEnvSpec,
    _MyoSuiteWalkEnvPool,
    _MyoSuiteWalkEnvSpec,
    _MyoSuiteWalkPixelEnvPool,
    _MyoSuiteWalkPixelEnvSpec,
)

from envpool.python.api import py_env

(
    MyoSuitePoseEnvSpec,
    MyoSuitePoseDMEnvPool,
    MyoSuitePoseGymnasiumEnvPool,
) = py_env(_MyoSuitePoseEnvSpec, _MyoSuitePoseEnvPool)
(
    MyoSuitePosePixelEnvSpec,
    MyoSuitePosePixelDMEnvPool,
    MyoSuitePosePixelGymnasiumEnvPool,
) = py_env(_MyoSuitePosePixelEnvSpec, _MyoSuitePosePixelEnvPool)
(
    MyoSuiteReachEnvSpec,
    MyoSuiteReachDMEnvPool,
    MyoSuiteReachGymnasiumEnvPool,
) = py_env(_MyoSuiteReachEnvSpec, _MyoSuiteReachEnvPool)
(
    MyoSuiteReachPixelEnvSpec,
    MyoSuiteReachPixelDMEnvPool,
    MyoSuiteReachPixelGymnasiumEnvPool,
) = py_env(_MyoSuiteReachPixelEnvSpec, _MyoSuiteReachPixelEnvPool)
(
    MyoSuiteReorientEnvSpec,
    MyoSuiteReorientDMEnvPool,
    MyoSuiteReorientGymnasiumEnvPool,
) = py_env(_MyoSuiteReorientEnvSpec, _MyoSuiteReorientEnvPool)
(
    MyoSuiteReorientPixelEnvSpec,
    MyoSuiteReorientPixelDMEnvPool,
    MyoSuiteReorientPixelGymnasiumEnvPool,
) = py_env(_MyoSuiteReorientPixelEnvSpec, _MyoSuiteReorientPixelEnvPool)
(
    MyoSuiteWalkEnvSpec,
    MyoSuiteWalkDMEnvPool,
    MyoSuiteWalkGymnasiumEnvPool,
) = py_env(_MyoSuiteWalkEnvSpec, _MyoSuiteWalkEnvPool)
(
    MyoSuiteWalkPixelEnvSpec,
    MyoSuiteWalkPixelDMEnvPool,
    MyoSuiteWalkPixelGymnasiumEnvPool,
) = py_env(_MyoSuiteWalkPixelEnvSpec, _MyoSuiteWalkPixelEnvPool)
(
    MyoSuiteTerrainEnvSpec,
    MyoSuiteTerrainDMEnvPool,
    MyoSuiteTerrainGymnasiumEnvPool,
) = py_env(_MyoSuiteTerrainEnvSpec, _MyoSuiteTerrainEnvPool)
(
    MyoSuiteTerrainPixelEnvSpec,
    MyoSuiteTerrainPixelDMEnvPool,
    MyoSuiteTerrainPixelGymnasiumEnvPool,
) = py_env(_MyoSuiteTerrainPixelEnvSpec, _MyoSuiteTerrainPixelEnvPool)
(
    MyoSuiteKeyTurnEnvSpec,
    MyoSuiteKeyTurnDMEnvPool,
    MyoSuiteKeyTurnGymnasiumEnvPool,
) = py_env(_MyoSuiteKeyTurnEnvSpec, _MyoSuiteKeyTurnEnvPool)
(
    MyoSuiteKeyTurnPixelEnvSpec,
    MyoSuiteKeyTurnPixelDMEnvPool,
    MyoSuiteKeyTurnPixelGymnasiumEnvPool,
) = py_env(_MyoSuiteKeyTurnPixelEnvSpec, _MyoSuiteKeyTurnPixelEnvPool)
(
    MyoSuiteObjHoldEnvSpec,
    MyoSuiteObjHoldDMEnvPool,
    MyoSuiteObjHoldGymnasiumEnvPool,
) = py_env(_MyoSuiteObjHoldEnvSpec, _MyoSuiteObjHoldEnvPool)
(
    MyoSuiteObjHoldPixelEnvSpec,
    MyoSuiteObjHoldPixelDMEnvPool,
    MyoSuiteObjHoldPixelGymnasiumEnvPool,
) = py_env(_MyoSuiteObjHoldPixelEnvSpec, _MyoSuiteObjHoldPixelEnvPool)
(
    MyoSuiteTorsoEnvSpec,
    MyoSuiteTorsoDMEnvPool,
    MyoSuiteTorsoGymnasiumEnvPool,
) = py_env(_MyoSuiteTorsoEnvSpec, _MyoSuiteTorsoEnvPool)
(
    MyoSuiteTorsoPixelEnvSpec,
    MyoSuiteTorsoPixelDMEnvPool,
    MyoSuiteTorsoPixelGymnasiumEnvPool,
) = py_env(_MyoSuiteTorsoPixelEnvSpec, _MyoSuiteTorsoPixelEnvPool)
(
    MyoSuitePenTwirlEnvSpec,
    MyoSuitePenTwirlDMEnvPool,
    MyoSuitePenTwirlGymnasiumEnvPool,
) = py_env(_MyoSuitePenTwirlEnvSpec, _MyoSuitePenTwirlEnvPool)
(
    MyoSuitePenTwirlPixelEnvSpec,
    MyoSuitePenTwirlPixelDMEnvPool,
    MyoSuitePenTwirlPixelGymnasiumEnvPool,
) = py_env(_MyoSuitePenTwirlPixelEnvSpec, _MyoSuitePenTwirlPixelEnvPool)
(
    MyoChallengeReorientEnvSpec,
    MyoChallengeReorientDMEnvPool,
    MyoChallengeReorientGymnasiumEnvPool,
) = py_env(_MyoChallengeReorientEnvSpec, _MyoChallengeReorientEnvPool)
(
    MyoChallengeReorientPixelEnvSpec,
    MyoChallengeReorientPixelDMEnvPool,
    MyoChallengeReorientPixelGymnasiumEnvPool,
) = py_env(_MyoChallengeReorientPixelEnvSpec, _MyoChallengeReorientPixelEnvPool)
(
    MyoChallengeRelocateEnvSpec,
    MyoChallengeRelocateDMEnvPool,
    MyoChallengeRelocateGymnasiumEnvPool,
) = py_env(_MyoChallengeRelocateEnvSpec, _MyoChallengeRelocateEnvPool)
(
    MyoChallengeRelocatePixelEnvSpec,
    MyoChallengeRelocatePixelDMEnvPool,
    MyoChallengeRelocatePixelGymnasiumEnvPool,
) = py_env(_MyoChallengeRelocatePixelEnvSpec, _MyoChallengeRelocatePixelEnvPool)
(
    MyoChallengeBaodingEnvSpec,
    MyoChallengeBaodingDMEnvPool,
    MyoChallengeBaodingGymnasiumEnvPool,
) = py_env(_MyoChallengeBaodingEnvSpec, _MyoChallengeBaodingEnvPool)
(
    MyoChallengeBaodingPixelEnvSpec,
    MyoChallengeBaodingPixelDMEnvPool,
    MyoChallengeBaodingPixelGymnasiumEnvPool,
) = py_env(_MyoChallengeBaodingPixelEnvSpec, _MyoChallengeBaodingPixelEnvPool)
(
    MyoChallengeBimanualEnvSpec,
    MyoChallengeBimanualDMEnvPool,
    MyoChallengeBimanualGymnasiumEnvPool,
) = py_env(_MyoChallengeBimanualEnvSpec, _MyoChallengeBimanualEnvPool)
(
    MyoChallengeBimanualPixelEnvSpec,
    MyoChallengeBimanualPixelDMEnvPool,
    MyoChallengeBimanualPixelGymnasiumEnvPool,
) = py_env(_MyoChallengeBimanualPixelEnvSpec, _MyoChallengeBimanualPixelEnvPool)
(
    MyoChallengeRunTrackEnvSpec,
    MyoChallengeRunTrackDMEnvPool,
    MyoChallengeRunTrackGymnasiumEnvPool,
) = py_env(_MyoChallengeRunTrackEnvSpec, _MyoChallengeRunTrackEnvPool)
(
    MyoChallengeRunTrackPixelEnvSpec,
    MyoChallengeRunTrackPixelDMEnvPool,
    MyoChallengeRunTrackPixelGymnasiumEnvPool,
) = py_env(_MyoChallengeRunTrackPixelEnvSpec, _MyoChallengeRunTrackPixelEnvPool)
(
    MyoChallengeSoccerEnvSpec,
    MyoChallengeSoccerDMEnvPool,
    MyoChallengeSoccerGymnasiumEnvPool,
) = py_env(_MyoChallengeSoccerEnvSpec, _MyoChallengeSoccerEnvPool)
(
    MyoChallengeSoccerPixelEnvSpec,
    MyoChallengeSoccerPixelDMEnvPool,
    MyoChallengeSoccerPixelGymnasiumEnvPool,
) = py_env(_MyoChallengeSoccerPixelEnvSpec, _MyoChallengeSoccerPixelEnvPool)
(
    MyoChallengeChaseTagEnvSpec,
    MyoChallengeChaseTagDMEnvPool,
    MyoChallengeChaseTagGymnasiumEnvPool,
) = py_env(_MyoChallengeChaseTagEnvSpec, _MyoChallengeChaseTagEnvPool)
(
    MyoChallengeChaseTagPixelEnvSpec,
    MyoChallengeChaseTagPixelDMEnvPool,
    MyoChallengeChaseTagPixelGymnasiumEnvPool,
) = py_env(_MyoChallengeChaseTagPixelEnvSpec, _MyoChallengeChaseTagPixelEnvPool)
(
    MyoChallengeTableTennisEnvSpec,
    MyoChallengeTableTennisDMEnvPool,
    MyoChallengeTableTennisGymnasiumEnvPool,
) = py_env(_MyoChallengeTableTennisEnvSpec, _MyoChallengeTableTennisEnvPool)
(
    MyoChallengeTableTennisPixelEnvSpec,
    MyoChallengeTableTennisPixelDMEnvPool,
    MyoChallengeTableTennisPixelGymnasiumEnvPool,
) = py_env(
    _MyoChallengeTableTennisPixelEnvSpec,
    _MyoChallengeTableTennisPixelEnvPool,
)
(
    MyoDMTrackEnvSpec,
    MyoDMTrackDMEnvPool,
    MyoDMTrackGymnasiumEnvPool,
) = py_env(_MyoDMTrackEnvSpec, _MyoDMTrackEnvPool)
(
    MyoDMTrackPixelEnvSpec,
    MyoDMTrackPixelDMEnvPool,
    MyoDMTrackPixelGymnasiumEnvPool,
) = py_env(_MyoDMTrackPixelEnvSpec, _MyoDMTrackPixelEnvPool)

__all__ = [
    "MyoSuitePoseEnvSpec",
    "MyoSuitePoseDMEnvPool",
    "MyoSuitePoseGymnasiumEnvPool",
    "MyoSuitePosePixelEnvSpec",
    "MyoSuitePosePixelDMEnvPool",
    "MyoSuitePosePixelGymnasiumEnvPool",
    "MyoSuiteReachEnvSpec",
    "MyoSuiteReachDMEnvPool",
    "MyoSuiteReachGymnasiumEnvPool",
    "MyoSuiteReachPixelEnvSpec",
    "MyoSuiteReachPixelDMEnvPool",
    "MyoSuiteReachPixelGymnasiumEnvPool",
    "MyoSuiteReorientEnvSpec",
    "MyoSuiteReorientDMEnvPool",
    "MyoSuiteReorientGymnasiumEnvPool",
    "MyoSuiteReorientPixelEnvSpec",
    "MyoSuiteReorientPixelDMEnvPool",
    "MyoSuiteReorientPixelGymnasiumEnvPool",
    "MyoSuiteWalkEnvSpec",
    "MyoSuiteWalkDMEnvPool",
    "MyoSuiteWalkGymnasiumEnvPool",
    "MyoSuiteWalkPixelEnvSpec",
    "MyoSuiteWalkPixelDMEnvPool",
    "MyoSuiteWalkPixelGymnasiumEnvPool",
    "MyoSuiteTerrainEnvSpec",
    "MyoSuiteTerrainDMEnvPool",
    "MyoSuiteTerrainGymnasiumEnvPool",
    "MyoSuiteTerrainPixelEnvSpec",
    "MyoSuiteTerrainPixelDMEnvPool",
    "MyoSuiteTerrainPixelGymnasiumEnvPool",
    "MyoSuiteKeyTurnEnvSpec",
    "MyoSuiteKeyTurnDMEnvPool",
    "MyoSuiteKeyTurnGymnasiumEnvPool",
    "MyoSuiteKeyTurnPixelEnvSpec",
    "MyoSuiteKeyTurnPixelDMEnvPool",
    "MyoSuiteKeyTurnPixelGymnasiumEnvPool",
    "MyoSuiteObjHoldEnvSpec",
    "MyoSuiteObjHoldDMEnvPool",
    "MyoSuiteObjHoldGymnasiumEnvPool",
    "MyoSuiteObjHoldPixelEnvSpec",
    "MyoSuiteObjHoldPixelDMEnvPool",
    "MyoSuiteObjHoldPixelGymnasiumEnvPool",
    "MyoSuiteTorsoEnvSpec",
    "MyoSuiteTorsoDMEnvPool",
    "MyoSuiteTorsoGymnasiumEnvPool",
    "MyoSuiteTorsoPixelEnvSpec",
    "MyoSuiteTorsoPixelDMEnvPool",
    "MyoSuiteTorsoPixelGymnasiumEnvPool",
    "MyoSuitePenTwirlEnvSpec",
    "MyoSuitePenTwirlDMEnvPool",
    "MyoSuitePenTwirlGymnasiumEnvPool",
    "MyoSuitePenTwirlPixelEnvSpec",
    "MyoSuitePenTwirlPixelDMEnvPool",
    "MyoSuitePenTwirlPixelGymnasiumEnvPool",
    "MyoChallengeReorientEnvSpec",
    "MyoChallengeReorientDMEnvPool",
    "MyoChallengeReorientGymnasiumEnvPool",
    "MyoChallengeReorientPixelEnvSpec",
    "MyoChallengeReorientPixelDMEnvPool",
    "MyoChallengeReorientPixelGymnasiumEnvPool",
    "MyoChallengeRelocateEnvSpec",
    "MyoChallengeRelocateDMEnvPool",
    "MyoChallengeRelocateGymnasiumEnvPool",
    "MyoChallengeRelocatePixelEnvSpec",
    "MyoChallengeRelocatePixelDMEnvPool",
    "MyoChallengeRelocatePixelGymnasiumEnvPool",
    "MyoChallengeBaodingEnvSpec",
    "MyoChallengeBaodingDMEnvPool",
    "MyoChallengeBaodingGymnasiumEnvPool",
    "MyoChallengeBaodingPixelEnvSpec",
    "MyoChallengeBaodingPixelDMEnvPool",
    "MyoChallengeBaodingPixelGymnasiumEnvPool",
    "MyoChallengeBimanualEnvSpec",
    "MyoChallengeBimanualDMEnvPool",
    "MyoChallengeBimanualGymnasiumEnvPool",
    "MyoChallengeBimanualPixelEnvSpec",
    "MyoChallengeBimanualPixelDMEnvPool",
    "MyoChallengeBimanualPixelGymnasiumEnvPool",
    "MyoChallengeRunTrackEnvSpec",
    "MyoChallengeRunTrackDMEnvPool",
    "MyoChallengeRunTrackGymnasiumEnvPool",
    "MyoChallengeRunTrackPixelEnvSpec",
    "MyoChallengeRunTrackPixelDMEnvPool",
    "MyoChallengeRunTrackPixelGymnasiumEnvPool",
    "MyoChallengeSoccerEnvSpec",
    "MyoChallengeSoccerDMEnvPool",
    "MyoChallengeSoccerGymnasiumEnvPool",
    "MyoChallengeSoccerPixelEnvSpec",
    "MyoChallengeSoccerPixelDMEnvPool",
    "MyoChallengeSoccerPixelGymnasiumEnvPool",
    "MyoChallengeChaseTagEnvSpec",
    "MyoChallengeChaseTagDMEnvPool",
    "MyoChallengeChaseTagGymnasiumEnvPool",
    "MyoChallengeChaseTagPixelEnvSpec",
    "MyoChallengeChaseTagPixelDMEnvPool",
    "MyoChallengeChaseTagPixelGymnasiumEnvPool",
    "MyoChallengeTableTennisEnvSpec",
    "MyoChallengeTableTennisDMEnvPool",
    "MyoChallengeTableTennisGymnasiumEnvPool",
    "MyoChallengeTableTennisPixelEnvSpec",
    "MyoChallengeTableTennisPixelDMEnvPool",
    "MyoChallengeTableTennisPixelGymnasiumEnvPool",
    "MyoDMTrackEnvSpec",
    "MyoDMTrackDMEnvPool",
    "MyoDMTrackGymnasiumEnvPool",
    "MyoDMTrackPixelEnvSpec",
    "MyoDMTrackPixelDMEnvPool",
    "MyoDMTrackPixelGymnasiumEnvPool",
]
