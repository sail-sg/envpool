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
"""Reset-only MPE2 oracle synchronization for tests and render documentation."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from envpool.mpe2 import TASKS


def make_oracle(name: str, **kwargs: Any) -> Any:
    """Construct a pinned upstream parallel environment."""
    alias = next(alias for task, _, alias in TASKS if task == name)
    module = importlib.import_module("mpe2." + alias.split("/")[1])
    return module.parallel_env(render_mode="rgb_array", **kwargs)


def render_frame(pool: Any, env_id: int = 0) -> np.ndarray:
    """Return one native RGB frame, checking that rendering is enabled."""
    frames = pool.render(env_id)
    assert frames is not None
    return frames[0]


def synchronize_reset(oracle: Any, info: dict[str, Any]) -> None:
    """Copy native bootstrap state once; subsequent steps use only actions."""
    raw = oracle.unwrapped
    world = raw.world
    name = raw.metadata["name"]
    record_size = 10 + world.dim_c
    data = info["snapshot"][0]
    offset = 0
    for i, entity in enumerate(world.entities):
        record = data[offset : offset + record_size]
        offset += record_size
        entity.state.p_pos = record[:2].copy()
        entity.state.p_vel = record[2:4].copy()
        entity.color = record[4:7].copy()
        if "simple_crypto" in name and i != 0 and i != 2:
            entity.color = np.append(entity.color, 0.0)
        if i < len(world.agents):
            entity.state.c = record[10:].copy()
        if "collect_treasure" in name:
            entity.holding = None if record[8] < 0 else int(record[8])
            if i >= len(world.agents):
                entity.alive = bool(record[7])
                entity.type = int(record[9])
    count = len(world.agents)
    goals = data[offset : offset + count].astype(int)
    key, original_range = data[offset + count : offset + count + 2]
    raw.original_cam_range = float(original_range)
    for i, agent in enumerate(world.agents):
        if goals[i] < 0:
            continue
        goal = world.landmarks[goals[i]]
        if "simple_reference" in name or "simple_speaker_listener" in name:
            agent.goal_a = world.agents[1 - i]
            agent.goal_b = goal
        else:
            agent.goal_a = goal
    if "simple_crypto" in name:
        world.agents[2].key = world.landmarks[int(key)].color.copy()
    if "simple_line" in name:
        raw.scenario._expected_positions = (
            data[offset + count + 2 :].reshape(count, 2).copy()
        )
    if "collect_treasure" in name:
        high, low, inc_high, inc_low, has_uint32, uinteger = map(
            int, info["respawn_rng"][0]
        )
        rng = np.random.PCG64()
        rng.state = {
            "bit_generator": "PCG64",
            "state": {
                "state": (high << 64) | low,
                "inc": (inc_high << 64) | inc_low,
            },
            "has_uint32": has_uint32,
            "uinteger": uinteger,
        }
        world.np_random = np.random.Generator(rng)


def actions_for(pool: Any, info: dict[str, Any], rng: Any) -> np.ndarray:
    """Sample valid player actions, including heterogeneous role spaces."""
    sizes = info["players"]["action_size"]
    if pool.config["continuous_actions"]:
        return rng.uniform(
            0, 1, (len(sizes), pool.action_space.shape[0])
        ).astype(np.float32)
    return np.asarray([rng.integers(int(n)) for n in sizes], dtype=np.int32)


def player_actions(info: dict[str, Any], actions: np.ndarray) -> dict[str, Any]:
    """Associate a player action batch with its world IDs."""
    return {"players": {"env_id": info["players"]["env_id"], "action": actions}}


def oracle_actions(oracle: Any, actions: np.ndarray) -> dict[str, Any]:
    """Remove role padding and key the same actions by upstream agent name."""
    return {
        agent: actions[i, : oracle.action_space(agent).shape[0]]
        if actions.ndim == 2
        else int(actions[i])
        for i, agent in enumerate(oracle.agents)
    }
