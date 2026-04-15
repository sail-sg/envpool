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
    _MyoSuiteTorsoEnvPool,
    _MyoSuiteTorsoEnvSpec,
    _MyoSuiteTorsoPixelEnvPool,
    _MyoSuiteTorsoPixelEnvSpec,
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
]
