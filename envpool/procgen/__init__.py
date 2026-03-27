# Copyright 2023 Garena Online Private Limited
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
"""Procgen env Init."""

import os
import sys

from envpool.python.api import py_env

_WINDOWS_DLL_DIR_HANDLES: list[object] = []

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    # Procgen links against Qt on Windows, so register the Qt bin dir before
    # importing the extension module.
    for env_var in ("QT_ROOT_DIR", "BAZEL_RULES_QT_DIR"):
        qt_root = os.environ.get(env_var)
        if not qt_root:
            continue
        qt_bin = os.path.join(qt_root, "bin")
        if os.path.isdir(qt_bin):
            _WINDOWS_DLL_DIR_HANDLES.append(os.add_dll_directory(qt_bin))
            break

from .procgen_envpool import _ProcgenEnvPool, _ProcgenEnvSpec

(
    ProcgenEnvSpec,
    ProcgenDMEnvPool,
    ProcgenGymEnvPool,
    ProcgenGymnasiumEnvPool,
) = py_env(_ProcgenEnvSpec, _ProcgenEnvPool)

__all__ = [
    "ProcgenEnvSpec",
    "ProcgenDMEnvPool",
    "ProcgenGymEnvPool",
    "ProcgenGymnasiumEnvPool",
]
