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
"""MuJoCo Playground native MuJoCo envs."""

from envpool.mujoco.playground_envpool import (
    _PlaygroundAlohaEnvPool,
    _PlaygroundAlohaEnvSpec,
    _PlaygroundAlohaPixelEnvPool,
    _PlaygroundAlohaPixelEnvSpec,
    _PlaygroundApolloEnvPool,
    _PlaygroundApolloEnvSpec,
    _PlaygroundApolloPixelEnvPool,
    _PlaygroundApolloPixelEnvSpec,
    _PlaygroundBarkourEnvPool,
    _PlaygroundBarkourEnvSpec,
    _PlaygroundBarkourPixelEnvPool,
    _PlaygroundBarkourPixelEnvSpec,
    _PlaygroundBerkeleyHumanoidEnvPool,
    _PlaygroundBerkeleyHumanoidEnvSpec,
    _PlaygroundBerkeleyHumanoidPixelEnvPool,
    _PlaygroundBerkeleyHumanoidPixelEnvSpec,
    _PlaygroundG1EnvPool,
    _PlaygroundG1EnvSpec,
    _PlaygroundG1PixelEnvPool,
    _PlaygroundG1PixelEnvSpec,
    _PlaygroundGo1EnvPool,
    _PlaygroundGo1EnvSpec,
    _PlaygroundGo1GetupEnvPool,
    _PlaygroundGo1GetupEnvSpec,
    _PlaygroundGo1GetupPixelEnvPool,
    _PlaygroundGo1GetupPixelEnvSpec,
    _PlaygroundGo1HandstandEnvPool,
    _PlaygroundGo1HandstandEnvSpec,
    _PlaygroundGo1HandstandPixelEnvPool,
    _PlaygroundGo1HandstandPixelEnvSpec,
    _PlaygroundGo1PixelEnvPool,
    _PlaygroundGo1PixelEnvSpec,
    _PlaygroundH1EnvPool,
    _PlaygroundH1EnvSpec,
    _PlaygroundH1PixelEnvPool,
    _PlaygroundH1PixelEnvSpec,
    _PlaygroundHandEnvPool,
    _PlaygroundHandEnvSpec,
    _PlaygroundHandPixelEnvPool,
    _PlaygroundHandPixelEnvSpec,
    _PlaygroundOp3EnvPool,
    _PlaygroundOp3EnvSpec,
    _PlaygroundOp3PixelEnvPool,
    _PlaygroundOp3PixelEnvSpec,
    _PlaygroundPandaEnvPool,
    _PlaygroundPandaEnvSpec,
    _PlaygroundPandaPixelEnvPool,
    _PlaygroundPandaPixelEnvSpec,
    _PlaygroundPandaRobotiqEnvPool,
    _PlaygroundPandaRobotiqEnvSpec,
    _PlaygroundPandaRobotiqPixelEnvPool,
    _PlaygroundPandaRobotiqPixelEnvSpec,
    _PlaygroundSpotGaitEnvPool,
    _PlaygroundSpotGaitEnvSpec,
    _PlaygroundSpotGaitPixelEnvPool,
    _PlaygroundSpotGaitPixelEnvSpec,
    _PlaygroundSpotGetupEnvPool,
    _PlaygroundSpotGetupEnvSpec,
    _PlaygroundSpotGetupPixelEnvPool,
    _PlaygroundSpotGetupPixelEnvSpec,
    _PlaygroundSpotJoystickEnvPool,
    _PlaygroundSpotJoystickEnvSpec,
    _PlaygroundSpotJoystickPixelEnvPool,
    _PlaygroundSpotJoystickPixelEnvSpec,
    _PlaygroundT1EnvPool,
    _PlaygroundT1EnvSpec,
    _PlaygroundT1PixelEnvPool,
    _PlaygroundT1PixelEnvSpec,
)

from envpool.python.api import py_env

(
    PlaygroundAlohaEnvSpec,
    PlaygroundAlohaDMEnvPool,
    PlaygroundAlohaGymnasiumEnvPool,
) = py_env(_PlaygroundAlohaEnvSpec, _PlaygroundAlohaEnvPool)
(
    PlaygroundAlohaPixelEnvSpec,
    PlaygroundAlohaPixelDMEnvPool,
    PlaygroundAlohaPixelGymnasiumEnvPool,
) = py_env(_PlaygroundAlohaPixelEnvSpec, _PlaygroundAlohaPixelEnvPool)

(
    PlaygroundApolloEnvSpec,
    PlaygroundApolloDMEnvPool,
    PlaygroundApolloGymnasiumEnvPool,
) = py_env(_PlaygroundApolloEnvSpec, _PlaygroundApolloEnvPool)
(
    PlaygroundApolloPixelEnvSpec,
    PlaygroundApolloPixelDMEnvPool,
    PlaygroundApolloPixelGymnasiumEnvPool,
) = py_env(_PlaygroundApolloPixelEnvSpec, _PlaygroundApolloPixelEnvPool)

(
    PlaygroundBarkourEnvSpec,
    PlaygroundBarkourDMEnvPool,
    PlaygroundBarkourGymnasiumEnvPool,
) = py_env(_PlaygroundBarkourEnvSpec, _PlaygroundBarkourEnvPool)
(
    PlaygroundBarkourPixelEnvSpec,
    PlaygroundBarkourPixelDMEnvPool,
    PlaygroundBarkourPixelGymnasiumEnvPool,
) = py_env(_PlaygroundBarkourPixelEnvSpec, _PlaygroundBarkourPixelEnvPool)

(
    PlaygroundBerkeleyHumanoidEnvSpec,
    PlaygroundBerkeleyHumanoidDMEnvPool,
    PlaygroundBerkeleyHumanoidGymnasiumEnvPool,
) = py_env(
    _PlaygroundBerkeleyHumanoidEnvSpec, _PlaygroundBerkeleyHumanoidEnvPool
)

(
    PlaygroundBerkeleyHumanoidPixelEnvSpec,
    PlaygroundBerkeleyHumanoidPixelDMEnvPool,
    PlaygroundBerkeleyHumanoidPixelGymnasiumEnvPool,
) = py_env(
    _PlaygroundBerkeleyHumanoidPixelEnvSpec,
    _PlaygroundBerkeleyHumanoidPixelEnvPool,
)

(
    PlaygroundG1EnvSpec,
    PlaygroundG1DMEnvPool,
    PlaygroundG1GymnasiumEnvPool,
) = py_env(_PlaygroundG1EnvSpec, _PlaygroundG1EnvPool)
(
    PlaygroundG1PixelEnvSpec,
    PlaygroundG1PixelDMEnvPool,
    PlaygroundG1PixelGymnasiumEnvPool,
) = py_env(_PlaygroundG1PixelEnvSpec, _PlaygroundG1PixelEnvPool)

(
    PlaygroundGo1EnvSpec,
    PlaygroundGo1DMEnvPool,
    PlaygroundGo1GymnasiumEnvPool,
) = py_env(_PlaygroundGo1EnvSpec, _PlaygroundGo1EnvPool)
(
    PlaygroundGo1PixelEnvSpec,
    PlaygroundGo1PixelDMEnvPool,
    PlaygroundGo1PixelGymnasiumEnvPool,
) = py_env(_PlaygroundGo1PixelEnvSpec, _PlaygroundGo1PixelEnvPool)
(
    PlaygroundH1EnvSpec,
    PlaygroundH1DMEnvPool,
    PlaygroundH1GymnasiumEnvPool,
) = py_env(_PlaygroundH1EnvSpec, _PlaygroundH1EnvPool)
(
    PlaygroundH1PixelEnvSpec,
    PlaygroundH1PixelDMEnvPool,
    PlaygroundH1PixelGymnasiumEnvPool,
) = py_env(_PlaygroundH1PixelEnvSpec, _PlaygroundH1PixelEnvPool)
(
    PlaygroundHandEnvSpec,
    PlaygroundHandDMEnvPool,
    PlaygroundHandGymnasiumEnvPool,
) = py_env(_PlaygroundHandEnvSpec, _PlaygroundHandEnvPool)
(
    PlaygroundHandPixelEnvSpec,
    PlaygroundHandPixelDMEnvPool,
    PlaygroundHandPixelGymnasiumEnvPool,
) = py_env(_PlaygroundHandPixelEnvSpec, _PlaygroundHandPixelEnvPool)
(
    PlaygroundGo1GetupEnvSpec,
    PlaygroundGo1GetupDMEnvPool,
    PlaygroundGo1GetupGymnasiumEnvPool,
) = py_env(_PlaygroundGo1GetupEnvSpec, _PlaygroundGo1GetupEnvPool)
(
    PlaygroundGo1GetupPixelEnvSpec,
    PlaygroundGo1GetupPixelDMEnvPool,
    PlaygroundGo1GetupPixelGymnasiumEnvPool,
) = py_env(_PlaygroundGo1GetupPixelEnvSpec, _PlaygroundGo1GetupPixelEnvPool)
(
    PlaygroundGo1HandstandEnvSpec,
    PlaygroundGo1HandstandDMEnvPool,
    PlaygroundGo1HandstandGymnasiumEnvPool,
) = py_env(_PlaygroundGo1HandstandEnvSpec, _PlaygroundGo1HandstandEnvPool)
(
    PlaygroundGo1HandstandPixelEnvSpec,
    PlaygroundGo1HandstandPixelDMEnvPool,
    PlaygroundGo1HandstandPixelGymnasiumEnvPool,
) = py_env(
    _PlaygroundGo1HandstandPixelEnvSpec,
    _PlaygroundGo1HandstandPixelEnvPool,
)
(
    PlaygroundOp3EnvSpec,
    PlaygroundOp3DMEnvPool,
    PlaygroundOp3GymnasiumEnvPool,
) = py_env(_PlaygroundOp3EnvSpec, _PlaygroundOp3EnvPool)
(
    PlaygroundOp3PixelEnvSpec,
    PlaygroundOp3PixelDMEnvPool,
    PlaygroundOp3PixelGymnasiumEnvPool,
) = py_env(_PlaygroundOp3PixelEnvSpec, _PlaygroundOp3PixelEnvPool)
(
    PlaygroundPandaEnvSpec,
    PlaygroundPandaDMEnvPool,
    PlaygroundPandaGymnasiumEnvPool,
) = py_env(_PlaygroundPandaEnvSpec, _PlaygroundPandaEnvPool)
(
    PlaygroundPandaPixelEnvSpec,
    PlaygroundPandaPixelDMEnvPool,
    PlaygroundPandaPixelGymnasiumEnvPool,
) = py_env(_PlaygroundPandaPixelEnvSpec, _PlaygroundPandaPixelEnvPool)
(
    PlaygroundPandaRobotiqEnvSpec,
    PlaygroundPandaRobotiqDMEnvPool,
    PlaygroundPandaRobotiqGymnasiumEnvPool,
) = py_env(_PlaygroundPandaRobotiqEnvSpec, _PlaygroundPandaRobotiqEnvPool)
(
    PlaygroundPandaRobotiqPixelEnvSpec,
    PlaygroundPandaRobotiqPixelDMEnvPool,
    PlaygroundPandaRobotiqPixelGymnasiumEnvPool,
) = py_env(
    _PlaygroundPandaRobotiqPixelEnvSpec,
    _PlaygroundPandaRobotiqPixelEnvPool,
)
(
    PlaygroundSpotJoystickEnvSpec,
    PlaygroundSpotJoystickDMEnvPool,
    PlaygroundSpotJoystickGymnasiumEnvPool,
) = py_env(_PlaygroundSpotJoystickEnvSpec, _PlaygroundSpotJoystickEnvPool)
(
    PlaygroundSpotJoystickPixelEnvSpec,
    PlaygroundSpotJoystickPixelDMEnvPool,
    PlaygroundSpotJoystickPixelGymnasiumEnvPool,
) = py_env(
    _PlaygroundSpotJoystickPixelEnvSpec, _PlaygroundSpotJoystickPixelEnvPool
)
(
    PlaygroundSpotGetupEnvSpec,
    PlaygroundSpotGetupDMEnvPool,
    PlaygroundSpotGetupGymnasiumEnvPool,
) = py_env(_PlaygroundSpotGetupEnvSpec, _PlaygroundSpotGetupEnvPool)
(
    PlaygroundSpotGetupPixelEnvSpec,
    PlaygroundSpotGetupPixelDMEnvPool,
    PlaygroundSpotGetupPixelGymnasiumEnvPool,
) = py_env(_PlaygroundSpotGetupPixelEnvSpec, _PlaygroundSpotGetupPixelEnvPool)
(
    PlaygroundSpotGaitEnvSpec,
    PlaygroundSpotGaitDMEnvPool,
    PlaygroundSpotGaitGymnasiumEnvPool,
) = py_env(_PlaygroundSpotGaitEnvSpec, _PlaygroundSpotGaitEnvPool)
(
    PlaygroundSpotGaitPixelEnvSpec,
    PlaygroundSpotGaitPixelDMEnvPool,
    PlaygroundSpotGaitPixelGymnasiumEnvPool,
) = py_env(_PlaygroundSpotGaitPixelEnvSpec, _PlaygroundSpotGaitPixelEnvPool)
(
    PlaygroundT1EnvSpec,
    PlaygroundT1DMEnvPool,
    PlaygroundT1GymnasiumEnvPool,
) = py_env(_PlaygroundT1EnvSpec, _PlaygroundT1EnvPool)
(
    PlaygroundT1PixelEnvSpec,
    PlaygroundT1PixelDMEnvPool,
    PlaygroundT1PixelGymnasiumEnvPool,
) = py_env(_PlaygroundT1PixelEnvSpec, _PlaygroundT1PixelEnvPool)

__all__ = [
    "PlaygroundAlohaDMEnvPool",
    "PlaygroundAlohaEnvSpec",
    "PlaygroundAlohaGymnasiumEnvPool",
    "PlaygroundAlohaPixelDMEnvPool",
    "PlaygroundAlohaPixelEnvSpec",
    "PlaygroundAlohaPixelGymnasiumEnvPool",
    "PlaygroundApolloDMEnvPool",
    "PlaygroundApolloEnvSpec",
    "PlaygroundApolloGymnasiumEnvPool",
    "PlaygroundApolloPixelDMEnvPool",
    "PlaygroundApolloPixelEnvSpec",
    "PlaygroundApolloPixelGymnasiumEnvPool",
    "PlaygroundBarkourDMEnvPool",
    "PlaygroundBarkourEnvSpec",
    "PlaygroundBarkourGymnasiumEnvPool",
    "PlaygroundBarkourPixelDMEnvPool",
    "PlaygroundBarkourPixelEnvSpec",
    "PlaygroundBarkourPixelGymnasiumEnvPool",
    "PlaygroundBerkeleyHumanoidDMEnvPool",
    "PlaygroundBerkeleyHumanoidEnvSpec",
    "PlaygroundBerkeleyHumanoidGymnasiumEnvPool",
    "PlaygroundBerkeleyHumanoidPixelDMEnvPool",
    "PlaygroundBerkeleyHumanoidPixelEnvSpec",
    "PlaygroundBerkeleyHumanoidPixelGymnasiumEnvPool",
    "PlaygroundG1DMEnvPool",
    "PlaygroundG1EnvSpec",
    "PlaygroundG1GymnasiumEnvPool",
    "PlaygroundG1PixelDMEnvPool",
    "PlaygroundG1PixelEnvSpec",
    "PlaygroundG1PixelGymnasiumEnvPool",
    "PlaygroundGo1DMEnvPool",
    "PlaygroundGo1EnvSpec",
    "PlaygroundGo1GetupDMEnvPool",
    "PlaygroundGo1GetupEnvSpec",
    "PlaygroundGo1GetupGymnasiumEnvPool",
    "PlaygroundGo1GetupPixelDMEnvPool",
    "PlaygroundGo1GetupPixelEnvSpec",
    "PlaygroundGo1GetupPixelGymnasiumEnvPool",
    "PlaygroundGo1GymnasiumEnvPool",
    "PlaygroundGo1HandstandDMEnvPool",
    "PlaygroundGo1HandstandEnvSpec",
    "PlaygroundGo1HandstandGymnasiumEnvPool",
    "PlaygroundGo1HandstandPixelDMEnvPool",
    "PlaygroundGo1HandstandPixelEnvSpec",
    "PlaygroundGo1HandstandPixelGymnasiumEnvPool",
    "PlaygroundGo1PixelDMEnvPool",
    "PlaygroundGo1PixelEnvSpec",
    "PlaygroundGo1PixelGymnasiumEnvPool",
    "PlaygroundHandDMEnvPool",
    "PlaygroundHandEnvSpec",
    "PlaygroundHandGymnasiumEnvPool",
    "PlaygroundHandPixelDMEnvPool",
    "PlaygroundHandPixelEnvSpec",
    "PlaygroundHandPixelGymnasiumEnvPool",
    "PlaygroundH1DMEnvPool",
    "PlaygroundH1EnvSpec",
    "PlaygroundH1GymnasiumEnvPool",
    "PlaygroundH1PixelDMEnvPool",
    "PlaygroundH1PixelEnvSpec",
    "PlaygroundH1PixelGymnasiumEnvPool",
    "PlaygroundOp3DMEnvPool",
    "PlaygroundOp3EnvSpec",
    "PlaygroundOp3GymnasiumEnvPool",
    "PlaygroundOp3PixelDMEnvPool",
    "PlaygroundOp3PixelEnvSpec",
    "PlaygroundOp3PixelGymnasiumEnvPool",
    "PlaygroundPandaDMEnvPool",
    "PlaygroundPandaEnvSpec",
    "PlaygroundPandaGymnasiumEnvPool",
    "PlaygroundPandaPixelDMEnvPool",
    "PlaygroundPandaPixelEnvSpec",
    "PlaygroundPandaPixelGymnasiumEnvPool",
    "PlaygroundPandaRobotiqDMEnvPool",
    "PlaygroundPandaRobotiqEnvSpec",
    "PlaygroundPandaRobotiqGymnasiumEnvPool",
    "PlaygroundPandaRobotiqPixelDMEnvPool",
    "PlaygroundPandaRobotiqPixelEnvSpec",
    "PlaygroundPandaRobotiqPixelGymnasiumEnvPool",
    "PlaygroundSpotGaitDMEnvPool",
    "PlaygroundSpotGaitEnvSpec",
    "PlaygroundSpotGaitGymnasiumEnvPool",
    "PlaygroundSpotGaitPixelDMEnvPool",
    "PlaygroundSpotGaitPixelEnvSpec",
    "PlaygroundSpotGaitPixelGymnasiumEnvPool",
    "PlaygroundSpotGetupDMEnvPool",
    "PlaygroundSpotGetupEnvSpec",
    "PlaygroundSpotGetupGymnasiumEnvPool",
    "PlaygroundSpotGetupPixelDMEnvPool",
    "PlaygroundSpotGetupPixelEnvSpec",
    "PlaygroundSpotGetupPixelGymnasiumEnvPool",
    "PlaygroundSpotJoystickDMEnvPool",
    "PlaygroundSpotJoystickEnvSpec",
    "PlaygroundSpotJoystickGymnasiumEnvPool",
    "PlaygroundSpotJoystickPixelDMEnvPool",
    "PlaygroundSpotJoystickPixelEnvSpec",
    "PlaygroundSpotJoystickPixelGymnasiumEnvPool",
    "PlaygroundT1DMEnvPool",
    "PlaygroundT1EnvSpec",
    "PlaygroundT1GymnasiumEnvPool",
    "PlaygroundT1PixelDMEnvPool",
    "PlaygroundT1PixelEnvSpec",
    "PlaygroundT1PixelGymnasiumEnvPool",
]
