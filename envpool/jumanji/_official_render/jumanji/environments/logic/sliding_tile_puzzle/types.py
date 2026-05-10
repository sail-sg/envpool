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


@dataclass
class State:
    """
    puzzle: 2D array representing the current state of the puzzle.
    empty_tile_position: the position of the empty tile in the puzzle.
    key: random key used for generating random numbers at each step.
    """

    puzzle: Any  # (N, N)
    empty_tile_position: Any  # (2,)
    key: Any  # (2,)
    step_count: Any  # (1,)


class Observation(NamedTuple):
    """
    puzzle: 2D array representing the current state of the puzzle.
    empty_tile_position: the position of the empty tile in the puzzle.
    action_mask: 1D array indicating the validity of each action.
    """

    puzzle: Any  # (N, N)
    empty_tile_position: Any  # (2,)
    action_mask: Any  # (4,)  # assuming 4 possible actions: up, down, left, right
    step_count: int  # Current timestep
