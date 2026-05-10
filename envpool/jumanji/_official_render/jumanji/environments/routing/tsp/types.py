# ruff: noqa
# fmt: off
from __future__ import annotations
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

from typing import TYPE_CHECKING, NamedTuple


if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from dataclasses import dataclass


@dataclass
class State:
    """
    coordinates: array of 2D coordinates for all cities.
    position: index of current city.
    visited_mask: binary mask (False/True <--> unvisited/visited).
    trajectory: array of city indices defining the route (-1 --> not filled yet).
    num_visited: how many cities have been visited.
    key: random key used for auto-reset.
    """

    coordinates: Any  # (num_cities, 2)
    position: Any  # ()
    visited_mask: Any  # (num_cities,)
    trajectory: Any  # (num_cities,)
    num_visited: Any  # ()
    key: Any  # (2,)


class Observation(NamedTuple):
    """
    coordinates: array of 2D coordinates for all cities.
    position: index of current city.
    trajectory: array of city indices defining the route (-1 --> not filled yet).
    action_mask: binary mask (False/True <--> illegal/legal).
    """

    coordinates: Any  # (num_cities, 2)
    position: Any  # ()
    trajectory: Any  # (num_cities,)
    action_mask: Any  # (num_cities,)
