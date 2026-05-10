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

from dataclasses import dataclass
from typing import Any, NamedTuple

from typing_extensions import TypeAlias

Board: TypeAlias = object


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
    step_count: Any  # ()
    flat_mine_locations: Any  # (num_mines,)
    key: Any  # (2,)


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
    num_mines: Any  # ()
    step_count: Any  # ()
