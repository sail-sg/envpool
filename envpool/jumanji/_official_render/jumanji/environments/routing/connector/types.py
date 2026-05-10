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

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from dataclasses import dataclass


@dataclass
class Agent:
    """
    id: unique number representing only this agent.
    start: start position of this agent.
    target: goal position of this agent.
    position: the current position of this agent.
    """

    id: Any  # ()
    start: Any  # (2,)
    target: Any  # (2,)
    position: Any  # (2,)

    @property
    def connected(self) -> Any:
        """returns: True if the agent has reached its target."""
        return np.all(self.position == self.target, axis=-1)

    def __eq__(self: Agent, agent_2: Any) -> Any:
        if not isinstance(agent_2, Agent):
            return NotImplemented
        same_ids = (agent_2.id == self.id).all()
        same_starts = (agent_2.start == self.start).all()
        same_targets = (agent_2.target == self.target).all()
        same_position = (agent_2.position == self.position).all()
        return same_ids & same_starts & same_targets & same_position


@dataclass
class State:
    """
    grid: grid representing the position of all agents.
    step_count: the index of the current step.
    agents: a stacked pytree of type Agent.
    key: random key used for auto-reset.
    """

    grid: Any  # (grid_size, grid_size)
    step_count: Any  # ()
    agents: Agent  # (num_agents, ...)
    key: Any  # (2,)


class Observation(NamedTuple):
    """
    The observation returned by the connector environment.

    grid: int array representing the positions of all agents from the perspective of all agents.
    The current agent is represented as 1,2,3 subsequent agents are represented as 4,5,6 and so on.

    For example, with 1 agent you might have a grid like this:
    0 0 1
    0 0 1
    0 3 2
    Which means agent 1 has moved from the top right of the grid down and is currently in the
    bottom right corner and is aiming to get to the middle bottom cell

    action_mask: boolean array representing whether each of the 5 actions is legal, for each agent.
    step_count: (int32) the current episode step.
    """

    grid: Any  # (num_agents, grid_size, grid_size)
    action_mask: Any  # (num_agents, 5)
    step_count: Any  # ()
