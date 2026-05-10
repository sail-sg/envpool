# ruff: noqa
# fmt: off
# Copyright 2022 InstaDeep Ltd. All rights reserved.
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
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

from typing_extensions import TypeAlias

Cube: TypeAlias = object


@dataclass
class State:
    """
    cube: 3D array whose cells contain the index of the corresponding colour of the sticker in the
        scramble.
    step_count: specifies how many timesteps have elapsed since environment reset.
    key: random key used for auto-reset.
    """

    cube: Cube  # (6, cube_size, cube_size)
    step_count: Any  # ()
    key: Any  # (2,)


class Observation(NamedTuple):
    """
    cube: 3D array whose cells contain the index of the corresponding colour of the sticker in the
        scramble.
    step_count: specifies how many timesteps have elapsed since environment reset.
    """

    cube: Cube  # (6, cube_size, cube_size)
    step_count: Any  # ()
