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

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex
import jax.numpy as jnp


class Position(NamedTuple):
    x: jnp.int32
    y: jnp.int32

    def __eq__(self, other: object) -> chex.Array:
        if not isinstance(other, Position):
            return NotImplemented
        return (self.x == other.x) & (self.y == other.y)


@dataclass
class State:
    """The state of the environment.

    key: random key used for auto-reset.
    grid: jax array (int) of the ingame maze with walls.
    pellets: int tracking the number of pellets.
    frightened_state_time: jax array (int) of shape ()
        tracks number of steps for the scatter state.
    pellet_locations: jax array (int) of pellet locations.
    power_up_locations: jax array (int) of power-up locations
    player_locations: current 2D position of agent.
    ghost_locations: jax array (int) of current ghost positions.
    initial_player_locations: starting 2D position of agent.
    initial_ghost_positions: jax array (int) of initial ghost positions.
    ghost_init_targets: jax array (int) of initial ghost targets.
        used to direct ghosts on respawn.
    old_ghost_locations: jax array (int) of shape ghost positions from last step.
        used to prevent ghost backtracking.
    ghost_init_steps: jax array (int) of number of initial ghost steps.
        used to determine per ghost initialisation.
    ghost_actions: jax array (int) of ghost action at current step.
    last_direction: (int) tracking the last direction of the player.
    dead: (bool) used to track player death.
    visited_index: jax array (int) of visited locations.
        used to prevent repeated pellet points.
    ghost_starts: jax array (int) of reset positions for ghosts
        used to reset ghost positions if eaten
    scatter_targets: jax array (int) of scatter targets.
            target locations for ghosts when scatter behavior is active.
    step_count: (int32) of total steps taken from reset till current timestep.
    ghost_eaten: jax array (bool) tracking if ghost has been eaten before.
    score: (int32) of total points aquired.
    """

    key: chex.PRNGKey  # (2,)
    grid: chex.Array  # (31,28)
    pellets: jnp.int32  # ()
    frightened_state_time: jnp.int32  # ()
    pellet_locations: chex.Array  # (316,2)
    power_up_locations: chex.Array  # (4,2)
    player_locations: Position  # Position(row, col) each of shape ()
    ghost_locations: chex.Array  # (4,2)
    initial_player_locations: Position  # Position(row, col) each of shape ()
    initial_ghost_positions: chex.Array  # (4,2)
    ghost_init_targets: chex.Array  # (4,2)
    old_ghost_locations: chex.Array  # (4,2)
    ghost_init_steps: chex.Array  # (4,)
    ghost_actions: chex.Array  # (4,)
    last_direction: jnp.int32  # ()
    dead: bool  # ()
    visited_index: chex.Array  # (320,2)
    ghost_starts: chex.Array  # (4,2)
    scatter_targets: chex.Array  # (4,2)
    step_count: jnp.int32  # ()
    ghost_eaten: chex.Array  # (4,)
    score: jnp.int32  # ()


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

    grid: chex.Array  # (31, 28)
    player_locations: Position
    ghost_locations: chex.Array
    power_up_locations: chex.Array
    frightened_state_time: jnp.int32
    pellet_locations: chex.Array
    action_mask: chex.Array
    score: jnp.int32  # ()
