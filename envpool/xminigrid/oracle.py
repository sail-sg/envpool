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
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
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


def recipe_actions(info: dict[str, Any]) -> list[int]:
    """Route the agent between the four interactions of the one-room example."""
    grid = info["grid"][0, ..., 0]
    floor = grid == 1
    y, x = info["agent_pos"][0]
    d = int(info["direction"][0])
    directions = ((-1, 0), (0, 1), (1, 0), (0, -1))
    actions: list[int] = []
    # Pick up the pyramid, drop by the square, pick up the product
    # where the square was, and drop by the green ball.
    for tile, action in ((5, 3), (4, 4), (4, 3), (3, 4)):
        ((ty, tx),) = np.argwhere(grid == tile)
        queue = deque([(y, x, d, actions)])
        seen = set()
        while queue:
            y, x, d, path = queue.popleft()
            if (y, x, d) in seen:
                continue
            seen.add((y, x, d))
            dy, dx = directions[d]
            front = y + dy, x + dx
            distance = abs(front[0] - ty) + abs(front[1] - tx)
            if (action == 3 and distance == 0) or (
                action == 4 and floor[front] and distance == 1
            ):
                actions = [*path, action]
                floor[front] = True
                break
            if floor[front]:
                queue.append((*front, d, [*path, 0]))
            queue.append((y, x, (d + 1) % 4, [*path, 1]))
            queue.append((y, x, (d - 1) % 4, [*path, 2]))
        else:
            raise ValueError("Unreachable recipe ingredient")
    return actions


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
