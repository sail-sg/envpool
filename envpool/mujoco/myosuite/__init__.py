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
"""MyoSuite native MuJoCo envs."""

import os
import platform

from envpool.python.glfw_context import preload_windows_gl_dlls

if platform.system() == "Windows":
    preload_windows_gl_dlls(strict=bool(os.environ.get("ENVPOOL_DLL_DIR")))

from envpool.mujoco.myosuite_envpool import (
    _MyoSuiteEnvPool,
    _MyoSuiteEnvSpec,
    _MyoSuitePixelEnvPool,
    _MyoSuitePixelEnvSpec,
)

from envpool.python.api import py_env

MyoSuiteEnvSpec, MyoSuiteDMEnvPool, MyoSuiteGymnasiumEnvPool = py_env(
    _MyoSuiteEnvSpec, _MyoSuiteEnvPool
)
(
    MyoSuitePixelEnvSpec,
    MyoSuitePixelDMEnvPool,
    MyoSuitePixelGymnasiumEnvPool,
) = py_env(_MyoSuitePixelEnvSpec, _MyoSuitePixelEnvPool)

__all__ = [
    "MyoSuiteDMEnvPool",
    "MyoSuiteEnvSpec",
    "MyoSuiteGymnasiumEnvPool",
    "MyoSuitePixelDMEnvPool",
    "MyoSuitePixelEnvSpec",
    "MyoSuitePixelGymnasiumEnvPool",
]
