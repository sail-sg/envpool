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

# Define a type alias for a board, which is an array.
Board: TypeAlias = chex.Array


@dataclass
class State:
    """
    board: the game board, each nonzero element in the array corresponds
        to a game tile and holds an exponent of 2. The actual value of the tile
        is obtained by raising 2 to the power of said non-zero exponent.
    step_count: the number of steps taken so far.
    action_mask: array of booleans that indicate the feasible actions, i.e. valid
        directions to move in.
    score: the current score of the game state.
    key: random key used to generate random numbers at each step and for auto-reset.
    """

    board: Board  # (board_size, board_size)
    step_count: chex.Numeric  # ()
    action_mask: chex.Array  # (4,)
    score: chex.Numeric  # ()
    key: chex.PRNGKey  # (2,)


class Observation(NamedTuple):
    """
    board: the game board, each nonzero element in the array corresponds
        to a game tile and holds an exponent of 2. The actual value of the tile
        is obtained by raising 2 to the power of said non-zero exponent.
    action_mask: array of booleans that indicate the feasible actions, i.e. valid
        directions to move in.
    """

    board: Board  # (board_size, board_size)
    action_mask: chex.Array  # (4,)
