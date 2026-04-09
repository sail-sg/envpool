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
"""MiniGrid env in EnvPool."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from envpool.python.api import py_env

from .minigrid_envpool import (
    _MiniGridDebugState,
    _MiniGridEnvPool,
    _MiniGridEnvSpec,
)


def _decode_mission_row(row: np.ndarray) -> str:
    mission = np.asarray(row, dtype=np.uint8).reshape(-1)
    zero = np.flatnonzero(mission == 0)
    end = int(zero[0]) if zero.size else int(mission.shape[0])
    return mission[:end].tobytes().decode("utf-8")


def decode_mission(mission: np.ndarray) -> str | np.ndarray:
    """Decode the fixed-size mission byte buffer returned by the C++ backend."""
    arr = np.asarray(mission, dtype=np.uint8)
    if arr.ndim == 1:
        return _decode_mission_row(arr)
    return np.asarray([_decode_mission_row(row) for row in arr], dtype=object)


def _normalize_env_ids(
    env_ids: np.ndarray | list[int] | None, num_envs: int
) -> np.ndarray:
    if env_ids is None:
        return np.arange(num_envs, dtype=np.int32)
    return np.asarray(env_ids, dtype=np.int32)


(
    MiniGridEnvSpec,
    MiniGridDMEnvPool,
    MiniGridGymnasiumEnvPool,
) = py_env(_MiniGridEnvSpec, _MiniGridEnvPool)

cast(Any, MiniGridEnvSpec).decode_mission = staticmethod(decode_mission)


def _debug_states(
    self: Any, env_ids: np.ndarray | list[int] | None = None
) -> list[Any]:
    env_ids = _normalize_env_ids(env_ids, self.config["num_envs"])
    return self._debug_states(env_ids)


for _env_cls in (
    MiniGridDMEnvPool,
    MiniGridGymnasiumEnvPool,
):
    cast(Any, _env_cls).decode_mission = staticmethod(decode_mission)
    cast(Any, _env_cls).debug_states = _debug_states


__all__ = [
    "MiniGridEnvSpec",
    "MiniGridDMEnvPool",
    "MiniGridGymnasiumEnvPool",
    "_MiniGridDebugState",
    "decode_mission",
]
