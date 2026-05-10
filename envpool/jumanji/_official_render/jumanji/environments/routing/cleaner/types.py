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

from typing import TYPE_CHECKING, NamedTuple, Optional

import numpy as np

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from dataclasses import dataclass


@dataclass
class State:
    """The state of the environment.

    grid: 2D matrix representing the grid of tiles, each of which is either clean,
        dirty, or a wall.
    agents_locations: array which specifies the x and y coordinates of every agent.
    action_mask: boolean array specifying, for each agent, which action
        (up, right, down, left) is legal.
    step_count: the number of steps elapsed since the beginning of the episode.
    key: random key used for auto-reset.
    """

    grid: Any  # (num_rows, num_cols)
    agents_locations: Any  # (num_agents, 2)
    action_mask: Optional[Any]  # (num_agents, 4)
    step_count: np.int32  # ()
    key: Any  # (2,)


class Observation(NamedTuple):
    """The observation that the agent sees.

    grid: 2D matrix representing the grid of tiles, each of which is either clean,
        dirty, or a wall.
    agents_locations: array which specifies the x and y coordinates of every agent.
    action_mask: boolean array specifying, for each agent, which action
        (up, right, down, left) is legal.
    step_count: the number of steps elapsed since the beginning of the episode.
    """

    grid: Any  # (num_rows, num_cols)
    agents_locations: Any  # (num_agents, 2)
    action_mask: Any  # (num_agents, 4)
    step_count: np.int32  # ()
