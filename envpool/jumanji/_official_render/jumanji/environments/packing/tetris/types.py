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

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class State:
    """
    grid_padded: the game grid, filled with zeros for the empty cells
        and with positive values for the filled cells. To allow for the placement of tetrominoes
        at the extreme right or bottom of the grid, the array has a padding of 3 columns on
        the right and 3 rows at the bottom. This padding enables the encoding of tetrominoes
        as 4x4 matrices, while ensuring that they can be placed without going out of bounds.
    grid_padded_old: similar to grid padded, used to save the grid before
        placing the last tetromino.
    tetromino_index: index to map the tetromino block.
    old_tetromino_rotated: a copy of the placed tetromino in the last step.
    new_tetromino: the new tetromino that needs to be placed.
    x_position: the selected x position for the last placed tetromino.
    y_position: the calculated y position for the last placed tetromino.
    action_mask: array of booleans that indicate the feasible actions, i.e. valid
        directions to move in.
    full_lines: saves the full lines in the last step.
    score: cumulative reward
    reward: instant reward
    key: random key used to generate random numbers at each step and for auto-reset.
    is_reset: True if the state is generated from a reset.
    step_count: current number of steps in the episode.
    """

    grid_padded: chex.Array  # (num_rows+3, num_cols+3)
    grid_padded_old: chex.Array  # (num_rows+3, num_cols+3)
    tetromino_index: chex.Numeric  # ()
    old_tetromino_rotated: chex.Array  # (4, 4)
    new_tetromino: chex.Array  # (4, 4)
    x_position: chex.Array  # ()
    y_position: chex.Array  # ()
    action_mask: chex.Array  # (4, num_cols)
    full_lines: chex.Array  # (num_rows,)
    score: chex.Array  # ()
    reward: chex.Array  # ()
    key: chex.PRNGKey  # (2,)
    is_reset: chex.Array  # ()
    step_count: chex.Numeric  # ()


class Observation(NamedTuple):
    """
    grid: the game grid, filled with zeros for the empty cells and with
        ones for the filled cells.
    tetromino: matrix of size (4x4) of zeros for the empty cells and with
        ones for the filled cells.
    action_mask: array of booleans that indicate the feasible actions, i.e. valid
        orientations and columns to select.
    step_count: current number of steps in the episode.
    """

    grid: chex.Array  # (num_rows, num_cols)
    tetromino: chex.Array  # (4, 4)
    action_mask: chex.Array  # (4, num_cols)
    step_count: chex.Numeric  # ()
