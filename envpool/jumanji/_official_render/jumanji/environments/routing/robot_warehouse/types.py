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

from typing import TYPE_CHECKING, NamedTuple, Union

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

from enum import IntEnum

import chex


class Action(IntEnum):
    """An enumeration of possible actions
    that an agent can take in the warehouse.

    NOOP - represents no operation.
    FORWARD - move forward.
    LEFT - turn left.
    RIGHT - turn right.
    TOGGLE_LOAD - toggle loading/offloading a shelf.
    """

    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    TOGGLE_LOAD = 4


class Direction(IntEnum):
    """An enumeration of possible directions
    that an agent can take in the warehouse.

    UP - move up.
    RIGHT - move right.
    DOWN - move down.
    LEFT - move left.
    """

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Position(NamedTuple):
    """A class to represent the 2D coordinate position of entities

    x: the x-position of the entity.
    y: the y-position of the entity.
    """

    x: chex.Array  # ()
    y: chex.Array  # ()


class Agent(NamedTuple):
    """A class to represent an Agent in the warehouse

    position: the (x,y) position of the agent.
    direction: the direction the agent is facing.
    is_carrying: whether the agent is carrying a shelf or not.
    """

    position: Position  # (2,)
    direction: chex.Array  # ()
    is_carrying: chex.Array  # ()


class Shelf(NamedTuple):
    """A class to represent a Shelf in the warehouse.

    position: the (x,y) position of the shelf.
    is_requested: whether the shelf is requested for delivery.
    """

    position: Position  # (2,)
    is_requested: chex.Array  # ()


Entity = Union[Agent, Shelf]


@dataclass
class State:
    """A dataclass representing the state of the simulated warehouse.

    grid: an array representing the warehouse floor as a 2D grid with two separate channels
        one for the agents, and one for the shelves.
    agents: a pytree of Agent type with per agent leaves: [position, direction, is_carrying]
    shelves: a pytree of Shelf type with per shelf leaves: [position, is_requested]
    request_queue : the queue of requested shelves (by ID).
    step_count: an integer representing the current step of the episode.
    key: a pseudorandom number generator key.
    """

    grid: chex.Array  # (2, grid_width, grid_height)
    agents: Agent  # (num_agents, ...)
    shelves: Shelf  # (num_shelves, ...)
    request_queue: chex.Array  # (num_requested,)
    step_count: chex.Array  # ()
    action_mask: chex.Array  # (num_agents, 5)
    key: chex.PRNGKey  # (2,)


class Observation(NamedTuple):
    """The observation that the agent sees.
    agents_view: the agents' view of other agents and shelves within their
        sensor range. The number of features in the observation array
        depends on the sensor range of the agent.
    action_mask: boolean array specifying, for each agent, which action
        (up, right, down, left) is legal.
    step_count: the number of steps elapsed since the beginning of the episode.
    """

    agents_view: chex.Array  # (num_agents, num_obs_features)
    action_mask: chex.Array  # (num_agents, 5)
    step_count: chex.Array  # ()
