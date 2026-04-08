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
"""Highway driving environments in EnvPool."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from envpool.python.api import py_env

from .highway_envpool import (
    _HighwayDebugState,
    _HighwayEnvPool,
    _HighwayEnvSpec,
    _HighwayVehicleDebugState,
    _PyOfficialAttributesEnvPool,
    _PyOfficialAttributesEnvSpec,
    _PyOfficialGoalEnvPool,
    _PyOfficialGoalEnvSpec,
    _PyOfficialKinematics5EnvPool,
    _PyOfficialKinematics5EnvSpec,
    _PyOfficialKinematics7Action3EnvPool,
    _PyOfficialKinematics7Action3EnvSpec,
    _PyOfficialKinematics7Action5EnvPool,
    _PyOfficialKinematics7Action5EnvSpec,
    _PyOfficialKinematics8ContinuousEnvPool,
    _PyOfficialKinematics8ContinuousEnvSpec,
    _PyOfficialMultiAgentEnvPool,
    _PyOfficialMultiAgentEnvSpec,
    _PyOfficialOccupancyEnvPool,
    _PyOfficialOccupancyEnvSpec,
    _PyOfficialTTC5EnvPool,
    _PyOfficialTTC5EnvSpec,
    _PyOfficialTTC16EnvPool,
    _PyOfficialTTC16EnvSpec,
)

(
    HighwayEnvSpec,
    HighwayDMEnvPool,
    HighwayGymEnvPool,
    HighwayGymnasiumEnvPool,
) = py_env(_HighwayEnvSpec, _HighwayEnvPool)

(
    PyOfficialKinematics5EnvSpec,
    PyOfficialKinematics5DMEnvPool,
    PyOfficialKinematics5GymEnvPool,
    PyOfficialKinematics5GymnasiumEnvPool,
) = py_env(_PyOfficialKinematics5EnvSpec, _PyOfficialKinematics5EnvPool)

(
    PyOfficialKinematics7Action5EnvSpec,
    PyOfficialKinematics7Action5DMEnvPool,
    PyOfficialKinematics7Action5GymEnvPool,
    PyOfficialKinematics7Action5GymnasiumEnvPool,
) = py_env(
    _PyOfficialKinematics7Action5EnvSpec,
    _PyOfficialKinematics7Action5EnvPool,
)

(
    PyOfficialKinematics7Action3EnvSpec,
    PyOfficialKinematics7Action3DMEnvPool,
    PyOfficialKinematics7Action3GymEnvPool,
    PyOfficialKinematics7Action3GymnasiumEnvPool,
) = py_env(
    _PyOfficialKinematics7Action3EnvSpec,
    _PyOfficialKinematics7Action3EnvPool,
)

(
    PyOfficialKinematics8ContinuousEnvSpec,
    PyOfficialKinematics8ContinuousDMEnvPool,
    PyOfficialKinematics8ContinuousGymEnvPool,
    PyOfficialKinematics8ContinuousGymnasiumEnvPool,
) = py_env(
    _PyOfficialKinematics8ContinuousEnvSpec,
    _PyOfficialKinematics8ContinuousEnvPool,
)

(
    PyOfficialTTC5EnvSpec,
    PyOfficialTTC5DMEnvPool,
    PyOfficialTTC5GymEnvPool,
    PyOfficialTTC5GymnasiumEnvPool,
) = py_env(_PyOfficialTTC5EnvSpec, _PyOfficialTTC5EnvPool)

(
    PyOfficialTTC16EnvSpec,
    PyOfficialTTC16DMEnvPool,
    PyOfficialTTC16GymEnvPool,
    PyOfficialTTC16GymnasiumEnvPool,
) = py_env(_PyOfficialTTC16EnvSpec, _PyOfficialTTC16EnvPool)

(
    PyOfficialGoalEnvSpec,
    PyOfficialGoalDMEnvPool,
    PyOfficialGoalGymEnvPool,
    PyOfficialGoalGymnasiumEnvPool,
) = py_env(_PyOfficialGoalEnvSpec, _PyOfficialGoalEnvPool)

(
    PyOfficialAttributesEnvSpec,
    PyOfficialAttributesDMEnvPool,
    PyOfficialAttributesGymEnvPool,
    PyOfficialAttributesGymnasiumEnvPool,
) = py_env(_PyOfficialAttributesEnvSpec, _PyOfficialAttributesEnvPool)

(
    PyOfficialOccupancyEnvSpec,
    PyOfficialOccupancyDMEnvPool,
    PyOfficialOccupancyGymEnvPool,
    PyOfficialOccupancyGymnasiumEnvPool,
) = py_env(_PyOfficialOccupancyEnvSpec, _PyOfficialOccupancyEnvPool)

(
    PyOfficialMultiAgentEnvSpec,
    PyOfficialMultiAgentDMEnvPool,
    PyOfficialMultiAgentGymEnvPool,
    PyOfficialMultiAgentGymnasiumEnvPool,
) = py_env(_PyOfficialMultiAgentEnvSpec, _PyOfficialMultiAgentEnvPool)


def _normalize_env_ids(
    env_ids: np.ndarray | list[int] | None, num_envs: int
) -> np.ndarray:
    if env_ids is None:
        return np.arange(num_envs, dtype=np.int32)
    return np.asarray(env_ids, dtype=np.int32)


def _debug_states(
    self: Any, env_ids: np.ndarray | list[int] | None = None
) -> list[Any]:
    env_ids = _normalize_env_ids(env_ids, self.config["num_envs"])
    return self._debug_states(env_ids)


for _env_cls in (
    HighwayDMEnvPool,
    HighwayGymEnvPool,
    HighwayGymnasiumEnvPool,
):
    cast(Any, _env_cls).debug_states = _debug_states


__all__ = [
    "HighwayEnvSpec",
    "HighwayDMEnvPool",
    "HighwayGymEnvPool",
    "HighwayGymnasiumEnvPool",
    "PyOfficialAttributesEnvSpec",
    "PyOfficialAttributesDMEnvPool",
    "PyOfficialAttributesGymEnvPool",
    "PyOfficialAttributesGymnasiumEnvPool",
    "PyOfficialGoalEnvSpec",
    "PyOfficialGoalDMEnvPool",
    "PyOfficialGoalGymEnvPool",
    "PyOfficialGoalGymnasiumEnvPool",
    "PyOfficialKinematics5EnvSpec",
    "PyOfficialKinematics5DMEnvPool",
    "PyOfficialKinematics5GymEnvPool",
    "PyOfficialKinematics5GymnasiumEnvPool",
    "PyOfficialKinematics7Action3EnvSpec",
    "PyOfficialKinematics7Action3DMEnvPool",
    "PyOfficialKinematics7Action3GymEnvPool",
    "PyOfficialKinematics7Action3GymnasiumEnvPool",
    "PyOfficialKinematics7Action5EnvSpec",
    "PyOfficialKinematics7Action5DMEnvPool",
    "PyOfficialKinematics7Action5GymEnvPool",
    "PyOfficialKinematics7Action5GymnasiumEnvPool",
    "PyOfficialKinematics8ContinuousEnvSpec",
    "PyOfficialKinematics8ContinuousDMEnvPool",
    "PyOfficialKinematics8ContinuousGymEnvPool",
    "PyOfficialKinematics8ContinuousGymnasiumEnvPool",
    "PyOfficialMultiAgentEnvSpec",
    "PyOfficialMultiAgentDMEnvPool",
    "PyOfficialMultiAgentGymEnvPool",
    "PyOfficialMultiAgentGymnasiumEnvPool",
    "PyOfficialOccupancyEnvSpec",
    "PyOfficialOccupancyDMEnvPool",
    "PyOfficialOccupancyGymEnvPool",
    "PyOfficialOccupancyGymnasiumEnvPool",
    "PyOfficialTTC5EnvSpec",
    "PyOfficialTTC5DMEnvPool",
    "PyOfficialTTC5GymEnvPool",
    "PyOfficialTTC5GymnasiumEnvPool",
    "PyOfficialTTC16EnvSpec",
    "PyOfficialTTC16DMEnvPool",
    "PyOfficialTTC16GymEnvPool",
    "PyOfficialTTC16GymnasiumEnvPool",
    "_HighwayDebugState",
    "_HighwayVehicleDebugState",
]
