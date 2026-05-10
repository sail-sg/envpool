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

import numpy as np


class Position(NamedTuple):
    row: np.int32
    col: np.int32

    def __eq__(self, other: object) -> Any:
        if not isinstance(other, Position):
            return NotImplemented
        return (self.row == other.row) & (self.col == other.col)


@dataclass
class State:
    """
    agent_position: current 2D Position of agent.
    target_position: 2D Position of target cell.
    walls: array (bool) whose values are `True` where walls are and `False` for empty cells.
    action_mask: array specifying which directions the agent can move in from its current position.
    step_count: (int32) step number of the episode.
    key: random key used for auto-reset.
    """

    agent_position: Position  # Position(row, col) each of shape ()
    target_position: Position  # Position(row, col) each of shape ()
    walls: Any  # (num_rows, num_cols)
    action_mask: Any  # (4,)
    step_count: np.int32  # ()
    key: Any  # (2,)


class Observation(NamedTuple):
    """The Maze observation that the agent sees.

    agent_position: current 2D Position of agent.
    target_position: 2D Position of target cell.
    walls: array (bool) whose values are `True` where walls are and `False` for empty cells.
    action_mask: array specifying which directions the agent can move in from its current position.
    step_count: (int32) step number of the episode.
    """

    agent_position: Position  # Position(row, col) each of shape ()
    target_position: Position  # Position(row, col) each of shape ()
    walls: Any  # (num_rows, num_cols)
    action_mask: Any  # (4,)
    step_count: np.int32  # ()
