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
class Entity:
    """
    An entity is something that can be placed in the LBF environment (agent or food).

    id: unique number representing only this food.
    position: the position of this food.
    level: the level of this food.
    """

    id: chex.Array  # ()
    position: chex.Array  # (2,)
    level: chex.Array  # ()


@dataclass
class Agent(Entity):
    """
    An agent is an entity that can move and load food.

    id: unique number representing only this food.
    position: the position of this food.
    level: the level of this food.
    loading: whether the agent is currently loading food.
    """

    loading: chex.Array  # () - bool: is loading food


@dataclass
class Food(Entity):
    """
    A food is an entity that can be eaten by an agent.

    id: unique number representing only this food.
    position: the position of this food.
    level: the level of this food.
    eaten: whether the food has been eaten.
    """

    eaten: chex.Array  # () - bool: has been eaten


@dataclass
class State:
    """
    Holds the dynamics of the LBF environment.

    agents: a stacked pytree of Agents - all the agents in the environment.
    food: a stacked pytree of Food - all the food in the environment.
    step_count: the index of the current step.
    key: random key used for auto-reset.
    """

    agents: Agent  # List of Agent entities (pytree structure)
    food_items: Food  # List of Food entities (pytree structure)
    step_count: chex.Array  # ()
    key: chex.PRNGKey  # (2,)


class Observation(NamedTuple):
    """
    The observation returned by the LBF environment.
    agents_view: (num_agents, grid_size, grid_size) int32 array
        representing the view of each agent.
    action_mask: boolean array representing which action (noop, up, right, down, left, load)
        is legal, for each agent.
    step_count: (int32) the current episode step.
    """

    agents_view: chex.Array  # (num_agents, 3 * (num_food + num_agents))
    action_mask: chex.Array  # (num_agents, 6)
    step_count: chex.Array  # ()
