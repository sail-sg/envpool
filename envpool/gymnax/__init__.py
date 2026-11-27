# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native Gymnax tasks."""

from envpool.python.api import py_env

from .gymnax_envpool import (
    _DiscreteEnvPool,
    _DiscreteEnvSpec,
    _FourRoomsEnvPool,
    _FourRoomsEnvSpec,
    _RobotEnvPool,
    _RobotEnvSpec,
    _SnakeEnvPool,
    _SnakeEnvSpec,
)

DiscreteEnvSpec, DiscreteDMEnvPool, DiscreteGymnasiumEnvPool = py_env(
    _DiscreteEnvSpec, _DiscreteEnvPool
)
FourRoomsEnvSpec, FourRoomsDMEnvPool, FourRoomsGymnasiumEnvPool = py_env(
    _FourRoomsEnvSpec, _FourRoomsEnvPool
)
RobotEnvSpec, RobotDMEnvPool, RobotGymnasiumEnvPool = py_env(
    _RobotEnvSpec, _RobotEnvPool
)
SnakeEnvSpec, SnakeDMEnvPool, SnakeGymnasiumEnvPool = py_env(
    _SnakeEnvSpec, _SnakeEnvPool
)
