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


class Observation(NamedTuple):
    """
    grid: 2D array with the current state of grid.
    blocks: 3D array with the blocks to be placed on the board. Here each block is a
        2D array with shape (3, 3).
    action_mask: 4D array showing where blocks can be placed on the grid.
        this mask includes all possible rotations and possible placement locations
        for each block on the grid.
    """

    grid: chex.Array  # (num_rows, num_cols)
    blocks: chex.Array  # (num_blocks, 3, 3)
    action_mask: chex.Array  # (num_blocks, num_rotations, num_rows-3, num_cols-3)


@dataclass
class State:
    """
    grid: 2D array with the current state of grid.
    num_blocks: number of blocks in the full grid.
    blocks: 3D array with the blocks to be placed on the board. Here each block is a
        2D array with shape (3, 3).
    action_mask: 4D array showing where blocks can be placed on the grid.
        this mask includes all possible rotations and possible placement locations
        for each block on the grid.
    placed_blocks: 1D boolean array showing which blocks have been placed on the board.
    step_count: number of steps taken in the environment.
    key: random key used for board generation.
    """

    grid: chex.Array  # (num_rows, num_cols)
    num_blocks: chex.Numeric  # ()
    blocks: chex.Array  # (num_blocks, 3, 3)
    action_mask: chex.Array  # (num_blocks, num_rotations, num_rows-3, num_cols-3)
    placed_blocks: chex.Array  # (num_blocks,)
    step_count: chex.Numeric  # ()
    key: chex.PRNGKey  # (2,)
