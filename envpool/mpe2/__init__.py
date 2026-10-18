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
"""Native MPE2 environments with discrete or continuous player actions."""

from typing import Any

from envpool.python.api import py_env

from .mpe2_envpool import (
    TASKS,
    _MPE2ContinuousEnvPool,
    _MPE2ContinuousEnvSpec,
    _MPE2DiscreteEnvPool,
    _MPE2DiscreteEnvSpec,
)

_DiscreteSpec, _DiscreteDM, _DiscreteGym = py_env(
    _MPE2DiscreteEnvSpec, _MPE2DiscreteEnvPool
)
_ContinuousSpec, _ContinuousDM, _ContinuousGym = py_env(
    _MPE2ContinuousEnvSpec, _MPE2ContinuousEnvPool
)


class MPE2EnvSpec:
    """Select the native spec for the requested action dtype at construction."""

    gen_config = _DiscreteSpec.gen_config

    def __new__(cls, config: Any) -> Any:
        """Construct the corresponding native spec."""
        return (
            _ContinuousSpec if config.continuous_actions else _DiscreteSpec
        )(config)


def MPE2DMEnvPool(spec: Any) -> Any:
    """Build the native dm_env wrapper with the spec's action dtype."""
    return (_ContinuousDM if spec.config.continuous_actions else _DiscreteDM)(
        spec
    )


def MPE2GymnasiumEnvPool(spec: Any) -> Any:
    """Build the native Gymnasium wrapper with the spec's action dtype."""
    return (_ContinuousGym if spec.config.continuous_actions else _DiscreteGym)(
        spec
    )


__all__ = ["TASKS", "MPE2EnvSpec", "MPE2DMEnvPool", "MPE2GymnasiumEnvPool"]
