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


import numpy as np


class Position(NamedTuple):
    x: np.int32
    y: np.int32

    def __eq__(self, other: object) -> Any:
        if not isinstance(other, Position):
            return NotImplemented
        return (self.x == other.x) & (self.y == other.y)


@dataclass
class State:
    """The state of the environment.

    key: random key used for auto-reset.
    grid: array (int) of the ingame maze with walls.
    pellets: int tracking the number of pellets.
    frightened_state_time: array (int) of shape ()
        tracks number of steps for the scatter state.
    pellet_locations: array (int) of pellet locations.
    power_up_locations: array (int) of power-up locations
    player_locations: current 2D position of agent.
    ghost_locations: array (int) of current ghost positions.
    initial_player_locations: starting 2D position of agent.
    initial_ghost_positions: array (int) of initial ghost positions.
    ghost_init_targets: array (int) of initial ghost targets.
        used to direct ghosts on respawn.
    old_ghost_locations: array (int) of shape ghost positions from last step.
        used to prevent ghost backtracking.
    ghost_init_steps: array (int) of number of initial ghost steps.
        used to determine per ghost initialisation.
    ghost_actions: array (int) of ghost action at current step.
    last_direction: (int) tracking the last direction of the player.
    dead: (bool) used to track player death.
    visited_index: array (int) of visited locations.
        used to prevent repeated pellet points.
    ghost_starts: array (int) of reset positions for ghosts
        used to reset ghost positions if eaten
    scatter_targets: array (int) of scatter targets.
            target locations for ghosts when scatter behavior is active.
    step_count: (int32) of total steps taken from reset till current timestep.
    ghost_eaten: array (bool) tracking if ghost has been eaten before.
    score: (int32) of total points aquired.
    """

    key: Any  # (2,)
    grid: Any  # (31,28)
    pellets: np.int32  # ()
    frightened_state_time: np.int32  # ()
    pellet_locations: Any  # (316,2)
    power_up_locations: Any  # (4,2)
    player_locations: Position  # Position(row, col) each of shape ()
    ghost_locations: Any  # (4,2)
    initial_player_locations: Position  # Position(row, col) each of shape ()
    initial_ghost_positions: Any  # (4,2)
    ghost_init_targets: Any  # (4,2)
    old_ghost_locations: Any  # (4,2)
    ghost_init_steps: Any  # (4,)
    ghost_actions: Any  # (4,)
    last_direction: np.int32  # ()
    dead: bool  # ()
    visited_index: Any  # (320,2)
    ghost_starts: Any  # (4,2)
    scatter_targets: Any  # (4,2)
    step_count: np.int32  # ()
    ghost_eaten: Any  # (4,)
    score: np.int32  # ()


class Observation(NamedTuple):
    """The observation that the agent sees.

    grid: 2D matrix of the wall and movable areas on the map.
    player_locations: the current 2D position of the agent.
    ghost_locations: a 2D matrix of the current ghost locations.
    power_up_locations: a 2D matrix of the current power-up locations.
    frightened_state_time: (int32) number of steps left of scatter mode.
    pellet_locations: a 2D matrix of all pellet locations.
    action_mask: array specifying which directions the agent can move in from its current position.
    score: (int32) of total points aquired.
    """

    grid: Any  # (31, 28)
    player_locations: Position
    ghost_locations: Any
    power_up_locations: Any
    frightened_state_time: np.int32
    pellet_locations: Any
    action_mask: Any
    score: np.int32  # ()
