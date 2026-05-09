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
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

Board: TypeAlias = chex.Array


@dataclass
class State:
    """
    board: grid whose cells contain -1 if not yet explored, otherwise the number of mines in the 8
        adjacent cells.
    step_count: specifies how many timesteps have elapsed since environment reset.
    flat_mine_locations: indicates the flat locations (i.e. 2D is flattened to 1D) of all the mines
        on the board, is of length num_mines.
    key: random key used for auto-reset.
    """

    board: Board  # (num_rows, num_cols)
    step_count: chex.Numeric  # ()
    flat_mine_locations: chex.Array  # (num_mines,)
    key: chex.PRNGKey  # (2,)


class Observation(NamedTuple):
    """
    board: grid whose cells contain -1 if not yet explored, otherwise the number of mines in the 8
        adjacent cells.
    action_mask: indicates which actions are valid (not yet explored squares).
    num_mines: indicates the number of mines to locate.
    step_count: specifies how many timesteps have elapsed since environment reset.
    """

    board: Board  # (num_rows, num_cols)
    action_mask: Board  # (num_rows, num_cols)
    num_mines: chex.Numeric  # ()
    step_count: chex.Numeric  # ()
