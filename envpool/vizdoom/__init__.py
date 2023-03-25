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
"""Vizdoom env in EnvPool."""

from envpool.python.api import py_env

from .vizdoom_envpool import _VizdoomEnvPool, _VizdoomEnvSpec

(VizdoomEnvSpec, VizdoomDMEnvPool, VizdoomGymEnvPool,
 VizdoomGymnasiumEnvPool) = py_env(_VizdoomEnvSpec, _VizdoomEnvPool)

__all__ = [
  "VizdoomEnvSpec",
  "VizdoomDMEnvPool",
  "VizdoomGymEnvPool",
  "VizdoomGymnasiumEnvPool",
]
