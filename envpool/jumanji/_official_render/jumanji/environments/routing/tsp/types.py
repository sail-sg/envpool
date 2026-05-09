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

import chex

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass


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

    coordinates: chex.Array  # (num_cities, 2)
    position: chex.Numeric  # ()
    visited_mask: chex.Array  # (num_cities,)
    trajectory: chex.Array  # (num_cities,)
    num_visited: chex.Numeric  # ()
    key: chex.PRNGKey  # (2,)


class Observation(NamedTuple):
    """
    coordinates: array of 2D coordinates for all cities.
    position: index of current city.
    trajectory: array of city indices defining the route (-1 --> not filled yet).
    action_mask: binary mask (False/True <--> illegal/legal).
    """

    coordinates: chex.Array  # (num_cities, 2)
    position: chex.Numeric  # ()
    trajectory: chex.Array  # (num_cities,)
    action_mask: chex.Array  # (num_cities,)
