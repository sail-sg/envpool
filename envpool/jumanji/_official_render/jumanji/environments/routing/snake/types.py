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

from enum import IntEnum
from typing import TYPE_CHECKING, NamedTuple

import chex

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


class Position(NamedTuple):
    row: chex.Array
    col: chex.Array

    def __eq__(self, other: Position) -> chex.Array:  # type: ignore[override]
        if not isinstance(other, Position):
            return NotImplemented
        return (self.row == other.row) & (self.col == other.col)

    def __add__(self, other: Position) -> Position:  # type: ignore[override]
        if not isinstance(other, Position):
            return NotImplemented
        return Position(row=self.row + other.row, col=self.col + other.col)


class Actions(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


@dataclass
class State:
    """
    body: array indicating the snake's body cells.
    body_state: array ordering the snake's body cells.
    head_position: position of the snake's head on the 2D grid.
    tail: array indicating the snake's tail.
    fruit_position: position of the fruit on the 2D grid.
    length: current length of the snake.
    step_count: current number of steps in the episode.
    action_mask: array specifying which directions the snake can move in from its current position.
    key: random key used to sample a new fruit when one is eaten and used for auto-reset.
    """

    body: chex.Array  # (num_rows, num_cols)
    body_state: chex.Array  # (num_rows, num_cols)
    head_position: Position  # leaves of shape ()
    tail: chex.Array  # (num_rows, num_cols)
    fruit_position: Position  # ()
    length: chex.Numeric  # ()
    step_count: chex.Numeric  # ()
    action_mask: chex.Array  # (4,)
    key: chex.PRNGKey  # (2,)


class Observation(NamedTuple):
    """
    grid: feature maps that include information about the fruit, the snake head, its body and tail.
    step_count: current number of steps in the episode.
    action_mask: array specifying which directions the snake can move in from its current position.
    """

    grid: chex.Array  # (num_rows, num_cols, 5)
    step_count: chex.Numeric  # Shape ()
    action_mask: chex.Array  # (4,)
