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
    coordinates: array of 2D coordinates of all nodes (including depot).
    demands: array with the demands of all nodes (+ depot).
    position: index of the current node.
    capacity: current capacity of the vehicle.
    visited_mask: binary mask (False/True <--> unvisited/visited).
    trajectory: array of node indices denoting route (set to DEPOT_IDX if not filled yet).
    num_total_visits: number of performed visits (it can count depot multiple times).
    key: random key used for auto-reset.
    """

    coordinates: chex.Array  # (num_nodes + 1, 2)
    demands: chex.Array  # (num_nodes + 1,)
    position: chex.Numeric  # ()
    capacity: chex.Numeric  # ()
    visited_mask: chex.Array  # (num_nodes + 1,)
    trajectory: chex.Array  # (2 * num_nodes,)
    num_total_visits: chex.Numeric  # ()
    key: chex.PRNGKey  # (2,)


class Observation(NamedTuple):
    """
    coordinates: array of 2D coordinates of all nodes (including depot).
    demands: array with the demands of all nodes (including depot).
    unvisited_nodes: array of booleans that indicates nodes that remain to be visited.
    position: index of the current node.
    trajectory: array of node indices denoting route (set to DEPOT_IDX if not filled yet).
    capacity: current capacity of the vehicle.
    action_mask: binary mask (False/True <--> invalid/valid action).
    """

    coordinates: chex.Array  # (num_nodes + 1, 2)
    demands: chex.Array  # (num_nodes + 1,)
    unvisited_nodes: chex.Array  # (num_nodes + 1,)
    position: chex.Numeric  # ()
    trajectory: chex.Array  # (2 * num_nodes,)
    capacity: chex.Numeric  # ()
    action_mask: chex.Array  # (num_nodes + 1,)
