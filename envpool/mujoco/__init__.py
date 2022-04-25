# Copyright 2022 Garena Online Private Limited
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
"""Mujoco env in EnvPool."""

from envpool.python.api import py_env

from .mujoco_envpool import _AntEnvPool, _AntEnvSpec

AntEnvSpec, AntDMEnvPool, AntGymEnvPool = py_env(_AntEnvSpec, _AntEnvPool)

__all__ = [
  "AntEnvSpec",
  "AntDMEnvPool",
  "AntGymEnvPool",
]
