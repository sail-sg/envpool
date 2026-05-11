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
"""PGX game environments in EnvPool."""

from envpool.python.api import py_env

from .pgx_envpool import _GoEnvPool, _GoEnvSpec

(
    GoEnvSpec,
    GoDMEnvPool,
    GoGymnasiumEnvPool,
) = py_env(_GoEnvSpec, _GoEnvPool)

__all__ = [
    "GoEnvSpec",
    "GoDMEnvPool",
    "GoGymnasiumEnvPool",
]
