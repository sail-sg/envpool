# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reset-time state transfer for XMiniGrid tests and render examples."""

from collections import deque
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from xminigrid.core.constants import Tiles
from xminigrid.core.goals import AgentHoldGoal, TileOnPositionGoal
from xminigrid.core.observation import transparent_field_of_view
from xminigrid.types import RuleSet as OracleRuleSet

from envpool.xminigrid import RuleSet

_fov = jax.jit(transparent_field_of_view, static_argnums=(2, 3))

# Blue pyramid + purple square -> red ball. Put it beside the green ball.
RECIPE = RuleSet(
    goal=(4, 3, 1, 3, 2),
    rules=[(3, 5, 3, 4, 4, 3, 1)],
    init_tiles=[(5, 3), (4, 4), (3, 2)],
)


def oracle_ruleset(task: RuleSet) -> Any:
    """Convert the encoded task into the upstream array container."""
    return OracleRuleSet(
        goal=jnp.asarray(task.goal, dtype=jnp.uint8),
        rules=jnp.asarray(task.rules, dtype=jnp.uint8).reshape(-1, 7),
        init_tiles=jnp.asarray(task.init_tiles, dtype=jnp.uint8).reshape(-1, 2),
    )


def _route(info: dict[str, Any], target: np.ndarray, action: int) -> list[int]:
    """Find a route ending with the requested interaction at a target cell."""
    grid = info["grid"][0, ..., 0]
    y, x = info["agent_pos"][0]
    d = int(info["direction"][0])
    directions = ((-1, 0), (0, 1), (1, 0), (0, -1))
    queue: deque[tuple[int, int, int, list[int]]] = deque([(y, x, d, [])])
    seen = set()
    while queue:
        y, x, d, path = queue.popleft()
        if (y, x, d) in seen:
            continue
        seen.add((y, x, d))
        dy, dx = directions[d]
        front = y + dy, x + dx
        tile = grid[front]
        if target[front] and (action != 4 or tile == Tiles.FLOOR):
            return [*path, action]
        if tile in (Tiles.FLOOR, Tiles.DOOR_OPEN):
            queue.append((*front, d, [*path, 0]))
        elif tile == Tiles.DOOR_CLOSED:
            queue.append((*front, d, [*path, 5, 0]))
        queue.append((y, x, (d + 1) % 4, [*path, 1]))
        queue.append((y, x, (d - 1) % 4, [*path, 2]))
    raise ValueError(f"Unreachable target for action {action}")


def solution_steps(
    pool: Any, info: dict[str, Any]
) -> Iterator[tuple[int, Any]]:
    """Solve the recipe or a locked room, routing from each updated grid."""
    name = pool.config["task"]
    grid = info["grid"][0, ..., 0]
    rows, cols = np.indices(grid.shape)
    if name.startswith("XLand-"):

        def beside(tile: int) -> np.ndarray:
            ((y, x),) = np.argwhere(grid == tile)
            return abs(rows - y) + abs(cols - x) == 1

        # The red ball appears at the original square's position.
        plan = [
            (grid == Tiles.PYRAMID, 3),
            (beside(Tiles.SQUARE), 4),
            (grid == Tiles.SQUARE, 3),
            (beside(Tiles.BALL), 4),
        ]
    else:
        lock = grid == Tiles.DOOR_LOCKED
        ((y, x),) = np.argwhere(lock)
        away = cols < x - 1  # Keep the path through the doorway clear.
        plan = []
        if "Blocked" in name:
            plan.extend([((rows == y) & (cols == x - 1), 3), (away, 4)])
        plan.extend([(grid == Tiles.KEY, 3), (lock, 5)])
        if "PickUp" in name:
            plan.extend([(away, 4), ((cols > x) & (grid > Tiles.WALL), 3)])
        elif name != "MiniGrid-Unlock":
            plan.append((grid == Tiles.GOAL, 0))

    for target, action in plan:
        for move in _route(info, target, action):
            result = pool.step(np.array([move], np.int32))
            info = result[-1]
            yield move, result


def synchronize_reset(name: str, t: Any, info: dict[str, Any]) -> Any:
    """Transfer the sampled grid and pose. Derive any random goal from the grid."""
    grid = info["grid"][0]
    mid = grid.shape[1] // 2
    state = t.state.replace(
        grid=jnp.asarray(grid),
        agent=t.state.agent.replace(
            position=jnp.asarray(info["agent_pos"][0]),
            direction=jnp.asarray(info["direction"][0]),
        ),
    )
    if name == "MiniGrid-Unlock":
        (row,) = np.flatnonzero(grid[:, mid, 0] == 8)
        goal = TileOnPositionGoal(
            tile=jnp.array([10, grid[row, mid, 1]]),
            position=jnp.array([row, mid]),
        )
        state = state.replace(goal_encoding=goal.encode())
    elif name.endswith("UnlockPickUp"):
        room = grid[:, mid + 1 :]
        (obj,) = room[room[..., 0] > 2]
        goal = AgentHoldGoal(tile=jnp.asarray(obj))
        state = state.replace(goal_encoding=goal.encode())
    elif name.startswith("MiniGrid-Memory"):
        top = np.array_equal(grid[2, 1], grid[1, -2])
        col = grid.shape[1] - 2
        state = state.replace(
            carry=state.carry.replace(
                success_pos=jnp.array([2 if top else 4, col]),
                failure_pos=jnp.array([4 if top else 2, col]),
            )
        )
    obs = _fov(state.grid, state.agent, *t.observation.shape[:2])
    return t.replace(state=state, observation=obs)
