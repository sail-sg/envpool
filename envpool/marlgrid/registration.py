# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MarlGrid env registration."""

from __future__ import annotations

from typing import Any

from envpool.registration import register

_IMPORT_PATH = "envpool.marlgrid"
_SPEC_CLS = "MarlGridEnvSpec"
_DM_CLS = "MarlGridDMEnvPool"
_GYMNASIUM_CLS = "MarlGridGymnasiumEnvPool"


def _register(task_id: str, n_agents: int, **kwargs: Any) -> None:
    register(
        task_id=task_id,
        import_path=_IMPORT_PATH,
        spec_cls=_SPEC_CLS,
        dm_cls=_DM_CLS,
        gymnasium_cls=_GYMNASIUM_CLS,
        max_episode_steps=100,
        max_num_players=n_agents,
        n_agents=n_agents,
        **kwargs,
    )


def _clutter_from_density(grid_size: int, density: float) -> int:
    return int(density * (grid_size - 2) * (grid_size - 2))


_register(
    "MarlGrid-1AgentCluttered15x15-v0",
    n_agents=1,
    env_name="cluttered",
    grid_size=15,
    view_size=5,
    n_clutter=30,
)

_register(
    "MarlGrid-3AgentCluttered11x11-v0",
    n_agents=3,
    env_name="cluttered",
    grid_size=11,
    view_size=7,
    n_clutter=_clutter_from_density(11, 0.15),
)

_register(
    "MarlGrid-3AgentCluttered15x15-v0",
    n_agents=3,
    env_name="cluttered",
    grid_size=15,
    view_size=7,
    n_clutter=_clutter_from_density(15, 0.15),
)

for n_agents in (2, 3, 4):
    _register(
        f"MarlGrid-{n_agents}AgentEmpty9x9-v0",
        n_agents=n_agents,
        env_name="empty",
        grid_size=9,
        view_size=7,
    )

_register(
    "Goalcycle-demo-solo-v0",
    n_agents=1,
    env_name="goalcycle",
    grid_size=13,
    view_size=7,
    view_offset=1,
    n_clutter=_clutter_from_density(13, 0.1),
    n_bonus_tiles=3,
    reward_decay=False,
)
