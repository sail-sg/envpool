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
    _PyNativeAttributesEnvPool,
    _PyNativeAttributesEnvSpec,
    _PyNativeGoalEnvPool,
    _PyNativeGoalEnvSpec,
    _PyNativeKinematics5EnvPool,
    _PyNativeKinematics5EnvSpec,
    _PyNativeKinematics7Action3EnvPool,
    _PyNativeKinematics7Action3EnvSpec,
    _PyNativeKinematics7Action5EnvPool,
    _PyNativeKinematics7Action5EnvSpec,
    _PyNativeKinematics8ContinuousEnvPool,
    _PyNativeKinematics8ContinuousEnvSpec,
    _PyNativeMultiAgentEnvPool,
    _PyNativeMultiAgentEnvSpec,
    _PyNativeOccupancyEnvPool,
    _PyNativeOccupancyEnvSpec,
    _PyNativeTTC5EnvPool,
    _PyNativeTTC5EnvSpec,
    _PyNativeTTC16EnvPool,
    _PyNativeTTC16EnvSpec,
)

(
    HighwayEnvSpec,
    HighwayDMEnvPool,
    HighwayGymnasiumEnvPool,
) = py_env(_HighwayEnvSpec, _HighwayEnvPool)

(
    NativeKinematics5EnvSpec,
    NativeKinematics5DMEnvPool,
    NativeKinematics5GymnasiumEnvPool,
) = py_env(_PyNativeKinematics5EnvSpec, _PyNativeKinematics5EnvPool)

(
    NativeKinematics7Action5EnvSpec,
    NativeKinematics7Action5DMEnvPool,
    NativeKinematics7Action5GymnasiumEnvPool,
) = py_env(
    _PyNativeKinematics7Action5EnvSpec,
    _PyNativeKinematics7Action5EnvPool,
)

(
    NativeKinematics7Action3EnvSpec,
    NativeKinematics7Action3DMEnvPool,
    NativeKinematics7Action3GymnasiumEnvPool,
) = py_env(
    _PyNativeKinematics7Action3EnvSpec,
    _PyNativeKinematics7Action3EnvPool,
)

(
    NativeKinematics8ContinuousEnvSpec,
    NativeKinematics8ContinuousDMEnvPool,
    NativeKinematics8ContinuousGymnasiumEnvPool,
) = py_env(
    _PyNativeKinematics8ContinuousEnvSpec,
    _PyNativeKinematics8ContinuousEnvPool,
)

(
    NativeTTC5EnvSpec,
    NativeTTC5DMEnvPool,
    NativeTTC5GymnasiumEnvPool,
) = py_env(_PyNativeTTC5EnvSpec, _PyNativeTTC5EnvPool)

(
    NativeTTC16EnvSpec,
    NativeTTC16DMEnvPool,
    NativeTTC16GymnasiumEnvPool,
) = py_env(_PyNativeTTC16EnvSpec, _PyNativeTTC16EnvPool)

(
    NativeGoalEnvSpec,
    NativeGoalDMEnvPool,
    NativeGoalGymnasiumEnvPool,
) = py_env(_PyNativeGoalEnvSpec, _PyNativeGoalEnvPool)

(
    NativeAttributesEnvSpec,
    NativeAttributesDMEnvPool,
    NativeAttributesGymnasiumEnvPool,
) = py_env(_PyNativeAttributesEnvSpec, _PyNativeAttributesEnvPool)

(
    NativeOccupancyEnvSpec,
    NativeOccupancyDMEnvPool,
    NativeOccupancyGymnasiumEnvPool,
) = py_env(_PyNativeOccupancyEnvSpec, _PyNativeOccupancyEnvPool)

(
    NativeMultiAgentEnvSpec,
    NativeMultiAgentDMEnvPool,
    NativeMultiAgentGymnasiumEnvPool,
) = py_env(_PyNativeMultiAgentEnvSpec, _PyNativeMultiAgentEnvPool)


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
    HighwayGymnasiumEnvPool,
    NativeAttributesDMEnvPool,
    NativeAttributesGymnasiumEnvPool,
    NativeGoalDMEnvPool,
    NativeGoalGymnasiumEnvPool,
    NativeKinematics5DMEnvPool,
    NativeKinematics5GymnasiumEnvPool,
    NativeKinematics7Action3DMEnvPool,
    NativeKinematics7Action3GymnasiumEnvPool,
    NativeKinematics7Action5DMEnvPool,
    NativeKinematics7Action5GymnasiumEnvPool,
    NativeKinematics8ContinuousDMEnvPool,
    NativeKinematics8ContinuousGymnasiumEnvPool,
    NativeMultiAgentDMEnvPool,
    NativeMultiAgentGymnasiumEnvPool,
    NativeOccupancyDMEnvPool,
    NativeOccupancyGymnasiumEnvPool,
    NativeTTC5DMEnvPool,
    NativeTTC5GymnasiumEnvPool,
    NativeTTC16DMEnvPool,
    NativeTTC16GymnasiumEnvPool,
):
    cast(Any, _env_cls).debug_states = _debug_states


__all__ = [
    "HighwayEnvSpec",
    "HighwayDMEnvPool",
    "HighwayGymnasiumEnvPool",
    "NativeAttributesEnvSpec",
    "NativeAttributesDMEnvPool",
    "NativeAttributesGymnasiumEnvPool",
    "NativeGoalEnvSpec",
    "NativeGoalDMEnvPool",
    "NativeGoalGymnasiumEnvPool",
    "NativeKinematics5EnvSpec",
    "NativeKinematics5DMEnvPool",
    "NativeKinematics5GymnasiumEnvPool",
    "NativeKinematics7Action3EnvSpec",
    "NativeKinematics7Action3DMEnvPool",
    "NativeKinematics7Action3GymnasiumEnvPool",
    "NativeKinematics7Action5EnvSpec",
    "NativeKinematics7Action5DMEnvPool",
    "NativeKinematics7Action5GymnasiumEnvPool",
    "NativeKinematics8ContinuousEnvSpec",
    "NativeKinematics8ContinuousDMEnvPool",
    "NativeKinematics8ContinuousGymnasiumEnvPool",
    "NativeMultiAgentEnvSpec",
    "NativeMultiAgentDMEnvPool",
    "NativeMultiAgentGymnasiumEnvPool",
    "NativeOccupancyEnvSpec",
    "NativeOccupancyDMEnvPool",
    "NativeOccupancyGymnasiumEnvPool",
    "NativeTTC5EnvSpec",
    "NativeTTC5DMEnvPool",
    "NativeTTC5GymnasiumEnvPool",
    "NativeTTC16EnvSpec",
    "NativeTTC16DMEnvPool",
    "NativeTTC16GymnasiumEnvPool",
    "_HighwayDebugState",
    "_HighwayVehicleDebugState",
]
