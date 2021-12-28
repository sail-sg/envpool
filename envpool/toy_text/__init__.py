# Copyright 2021 Garena Online Private Limited
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
"""Toy text env in EnvPool."""

from envpool.python.api import py_env

from .toy_text_envpool import (
  _BlackjackEnvPool,
  _BlackjackEnvSpec,
  _CatchEnvPool,
  _CatchEnvSpec,
  _CliffWalkingEnvPool,
  _CliffWalkingEnvSpec,
  _FrozenLakeEnvPool,
  _FrozenLakeEnvSpec,
  _NChainEnvPool,
  _NChainEnvSpec,
  _TaxiEnvPool,
  _TaxiEnvSpec,
)

CatchEnvSpec, CatchDMEnvPool, CatchGymEnvPool = py_env(
  _CatchEnvSpec, _CatchEnvPool
)

FrozenLakeEnvSpec, FrozenLakeDMEnvPool, FrozenLakeGymEnvPool = py_env(
  _FrozenLakeEnvSpec, _FrozenLakeEnvPool
)

TaxiEnvSpec, TaxiDMEnvPool, TaxiGymEnvPool = py_env(_TaxiEnvSpec, _TaxiEnvPool)

NChainEnvSpec, NChainDMEnvPool, NChainGymEnvPool = py_env(
  _NChainEnvSpec, _NChainEnvPool
)

CliffWalkingEnvSpec, CliffWalkingDMEnvPool, CliffWalkingGymEnvPool = py_env(
  _CliffWalkingEnvSpec, _CliffWalkingEnvPool
)

BlackjackEnvSpec, BlackjackDMEnvPool, BlackjackGymEnvPool = py_env(
  _BlackjackEnvSpec, _BlackjackEnvPool
)

__all__ = [
  "CatchEnvSpec",
  "CatchDMEnvPool",
  "CatchGymEnvPool",
  "FrozenLakeEnvSpec",
  "FrozenLakeDMEnvPool",
  "FrozenLakeGymEnvPool",
  "TaxiEnvSpec",
  "TaxiDMEnvPool",
  "TaxiGymEnvPool",
  "NChainEnvSpec",
  "NChainDMEnvPool",
  "NChainGymEnvPool",
  "CliffWalkingEnvSpec",
  "CliffWalkingDMEnvPool",
  "CliffWalkingGymEnvPool",
  "BlackjackEnvSpec",
  "BlackjackDMEnvPool",
  "BlackjackGymEnvPool",
]
