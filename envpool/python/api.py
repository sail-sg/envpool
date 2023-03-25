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
"""Api wrapper layer for EnvPool."""

from typing import Tuple, Type

from .dm_envpool import DMEnvPoolMeta
from .env_spec import EnvSpecMeta
from .gym_envpool import GymEnvPoolMeta
from .gymnasium_envpool import GymnasiumEnvPoolMeta
from .protocol import EnvPool, EnvSpec


def py_env(
  envspec: Type[EnvSpec], envpool: Type[EnvPool]
) -> Tuple[Type[EnvSpec], Type[EnvPool], Type[EnvPool], Type[EnvPool]]:
  """Initialize EnvPool for users."""
  # remove the _ prefix added when registering cpp class via pybind
  spec_name = envspec.__name__[1:]
  pool_name = envpool.__name__[1:]
  return (
    EnvSpecMeta(spec_name, (envspec,), {}),  # type: ignore[return-value]
    DMEnvPoolMeta(pool_name.replace("EnvPool", "DMEnvPool"), (envpool,), {}),
    GymEnvPoolMeta(pool_name.replace("EnvPool", "GymEnvPool"), (envpool,), {}),
    GymnasiumEnvPoolMeta(
      pool_name.replace("EnvPool", "GymnasiumEnvPool"), (envpool,), {}
    ),
  )
