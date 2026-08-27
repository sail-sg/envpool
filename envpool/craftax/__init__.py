# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native Craftax environments, pinned to the v1.6.1 oracle."""

from typing import Any, cast

try:
    from gymnasium.vector.vector_env import AutoresetMode
except ImportError:
    _autoreset_mode_type: Any = None
else:
    _autoreset_mode_type = AutoresetMode

from envpool.python.api import py_env

from .craftax_envpool import (
    _CraftaxClassicPixelsEnvPool,
    _CraftaxClassicPixelsEnvSpec,
    _CraftaxClassicSymbolicEnvPool,
    _CraftaxClassicSymbolicEnvSpec,
    _CraftaxPixelsEnvPool,
    _CraftaxPixelsEnvSpec,
    _CraftaxSymbolicEnvPool,
    _CraftaxSymbolicEnvSpec,
)

(
    CraftaxSymbolicEnvSpec,
    CraftaxSymbolicDMEnvPool,
    CraftaxSymbolicGymnasiumEnvPool,
) = py_env(_CraftaxSymbolicEnvSpec, _CraftaxSymbolicEnvPool)

(
    CraftaxPixelsEnvSpec,
    CraftaxPixelsDMEnvPool,
    CraftaxPixelsGymnasiumEnvPool,
) = py_env(_CraftaxPixelsEnvSpec, _CraftaxPixelsEnvPool)

(
    CraftaxClassicSymbolicEnvSpec,
    CraftaxClassicSymbolicDMEnvPool,
    CraftaxClassicSymbolicGymnasiumEnvPool,
) = py_env(_CraftaxClassicSymbolicEnvSpec, _CraftaxClassicSymbolicEnvPool)

(
    CraftaxClassicPixelsEnvSpec,
    CraftaxClassicPixelsDMEnvPool,
    CraftaxClassicPixelsGymnasiumEnvPool,
) = py_env(_CraftaxClassicPixelsEnvSpec, _CraftaxClassicPixelsEnvPool)


def _metadata(pool: Any) -> dict[str, Any]:
    return {
        "render_modes": ["rgb_array", "human"],
        "autoreset_mode": getattr(_autoreset_mode_type, "SAME_STEP", "SameStep")
        if pool.config["auto_reset"]
        else getattr(_autoreset_mode_type, "NEXT_STEP", "NextStep"),
    }


for _pool in (
    CraftaxSymbolicGymnasiumEnvPool,
    CraftaxPixelsGymnasiumEnvPool,
    CraftaxClassicSymbolicGymnasiumEnvPool,
    CraftaxClassicPixelsGymnasiumEnvPool,
):
    cast(Any, _pool).metadata = property(_metadata)
