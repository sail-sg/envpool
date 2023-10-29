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

import envpool.entry  # noqa: F401
from envpool.registration import (
  list_all_envs,
  make,
  make_dm,
  make_gym,
  make_gymnasium,
  make_spec,
  register,
)

__version__ = "0.8.4"
__all__ = [
  "register",
  "make",
  "make_dm",
  "make_gym",
  "make_gymnasium",
  "make_spec",
  "list_all_envs",
]
