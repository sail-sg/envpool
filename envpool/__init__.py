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
"""EnvPool package for efficient RL environment simulation."""

import os
import sys
from pathlib import Path

import numpy as np

_WINDOWS_DLL_HANDLES: list[object] = []


def _configure_windows_dll_search_path() -> None:
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    dll_dir = os.environ.get("ENVPOOL_DLL_DIR")
    if not dll_dir:
        return
    resolved_dir = Path(dll_dir).expanduser().resolve()
    if not resolved_dir.is_dir():
        raise FileNotFoundError(
            f"ENVPOOL_DLL_DIR does not exist: {resolved_dir}"
        )
    _WINDOWS_DLL_HANDLES.append(os.add_dll_directory(str(resolved_dir)))


_configure_windows_dll_search_path()

import envpool.entry  # noqa: F401
from envpool.python.protocol import (
    DMEnvPool,
    EnvPool,
    EnvSpec,
    GymEnvPool,
    GymnasiumEnvPool,
)
from envpool.registration import (
    list_all_envs,
    make,
    make_dm,
    make_gym,
    make_gymnasium,
    make_spec,
    make_thread_pool,
    register,
)

# Gym 0.26 still references np.bool8, which NumPy 2 removed.
if not hasattr(np, "bool8"):
    np.__dict__["bool8"] = np.bool_

__version__ = "1.0.1"
__all__ = [
    "register",
    "make_thread_pool",
    "make",
    "make_dm",
    "make_gym",
    "make_gymnasium",
    "make_spec",
    "list_all_envs",
    "EnvSpec",
    "EnvPool",
    "DMEnvPool",
    "GymEnvPool",
    "GymnasiumEnvPool",
]
