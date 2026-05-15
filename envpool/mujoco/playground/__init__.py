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

from envpool.mujoco import playground_envpool as _bindings
from envpool.python.api import py_env

_PY_ENV_PREFIXES = (
    "PlaygroundAloha",
    "PlaygroundAlohaPixel",
    "PlaygroundApollo",
    "PlaygroundApolloPixel",
    "PlaygroundBarkour",
    "PlaygroundBarkourPixel",
    "PlaygroundBerkeleyHumanoid",
    "PlaygroundBerkeleyHumanoidPixel",
    "PlaygroundG1",
    "PlaygroundG1Pixel",
    "PlaygroundGo1",
    "PlaygroundGo1Pixel",
    "PlaygroundH1",
    "PlaygroundH1Pixel",
    "PlaygroundHand",
    "PlaygroundHandPixel",
    "PlaygroundGo1Getup",
    "PlaygroundGo1GetupPixel",
    "PlaygroundGo1Handstand",
    "PlaygroundGo1HandstandPixel",
    "PlaygroundOp3",
    "PlaygroundOp3Pixel",
    "PlaygroundPanda",
    "PlaygroundPandaPixel",
    "PlaygroundPandaRobotiq",
    "PlaygroundPandaRobotiqPixel",
    "PlaygroundSpotJoystick",
    "PlaygroundSpotJoystickPixel",
    "PlaygroundSpotGetup",
    "PlaygroundSpotGetupPixel",
    "PlaygroundSpotGait",
    "PlaygroundSpotGaitPixel",
    "PlaygroundT1",
    "PlaygroundT1Pixel",
)

__all__: list[str] = []


def _install_py_env(prefix: str) -> None:
    env_spec, dm_env_pool, gymnasium_env_pool = py_env(
        getattr(_bindings, f"_{prefix}EnvSpec"),
        getattr(_bindings, f"_{prefix}EnvPool"),
    )
    globals()[f"{prefix}EnvSpec"] = env_spec
    globals()[f"{prefix}DMEnvPool"] = dm_env_pool
    globals()[f"{prefix}GymnasiumEnvPool"] = gymnasium_env_pool
    __all__.extend((
        f"{prefix}EnvSpec",
        f"{prefix}DMEnvPool",
        f"{prefix}GymnasiumEnvPool",
    ))


for _prefix in _PY_ENV_PREFIXES:
    _install_py_env(_prefix)

del _prefix
