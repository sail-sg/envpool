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
"""Bitwise alignment tests against the pinned official Jumanji oracle."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import warnings
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, cast


def _configure_matplotlib() -> None:
    """Configure Matplotlib without importing EnvPool's Jumanji package."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    if "MPLCONFIGDIR" not in os.environ:
        mpl_config = Path(
            os.environ.get(
                "ENVPOOL_JUMANJI_MPLCONFIGDIR",
                "/tmp/envpool-jumanji-matplotlib",
            )
        )
        mpl_config.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config)


def _shorten_windows_oracle_site_packages() -> None:
    """Copy oracle wheels to a short path before loading Windows extensions."""
    if os.name != "nt":
        return

    oracle_sites: list[Path] = []
    for entry in sys.path:
        if "jumanji_oracle_requirements" not in str(entry):
            continue
        path = Path(entry)
        if path.name == "site-packages":
            oracle_sites.append(path)
    if not oracle_sites:
        return

    key = hashlib.sha1(
        "|".join(str(path) for path in oracle_sites).encode("utf-8")
    ).hexdigest()[:12]
    short_site = Path(tempfile.gettempdir()) / "ej_o" / key / "s"
    marker = short_site / ".copied"
    if not marker.exists():
        short_site.mkdir(parents=True, exist_ok=True)
        for site in oracle_sites:
            for src in site.iterdir():
                if src.name == "__pycache__":
                    continue
                dst = short_site / src.name
                if src.is_dir():
                    shutil.copytree(
                        src,
                        dst,
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"),
                    )
                else:
                    shutil.copy2(src, dst)
        marker.write_text("ok")

    sys.path[:] = [str(short_site)] + [
        entry
        for entry in sys.path
        if "jumanji_oracle_requirements" not in str(entry)
    ]


_configure_matplotlib()
_shorten_windows_oracle_site_packages()
warnings.filterwarnings(
    "ignore",
    message="FigureCanvasAgg is non-interactive.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="scatter inputs have incompatible types.*",
    category=FutureWarning,
)
warnings.filterwarnings("ignore", category=ResourceWarning)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jumanji  # noqa: E402
import numpy as np  # noqa: E402
from absl.testing import absltest  # noqa: E402
from jumanji.environments.routing.sokoban.generator import (  # noqa: E402
    SimpleSolveGenerator,
)

import envpool.jumanji.registration as jumanji_registration  # noqa: E402
from envpool.registration import make_gymnasium  # noqa: E402

_RENDER_ALIGN_STEPS = 3
_LONG_ALIGN_STEPS = 32
_RENDER_SIZE = 128


def _as_numpy(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _csv_flat(value: Any) -> str:
    array = _as_numpy(value).reshape(-1)
    return ",".join(map(str, array.tolist()))


def _csv_flat_int(value: Any) -> str:
    array = _as_numpy(value).astype(np.int64, copy=False).reshape(-1)
    return ",".join(map(str, array.tolist()))


def _position_csv(position: Any) -> str:
    row = getattr(position, "row", getattr(position, "y", None))
    col = getattr(position, "col", getattr(position, "x", None))
    if row is None or col is None:
        array = _as_numpy(position).reshape(-1)
        row, col = array[0], array[1]
    return f"{int(row)},{int(col)}"


def _position_yx(position: Any) -> np.ndarray:
    y = getattr(position, "y", getattr(position, "row", None))
    x = getattr(position, "x", getattr(position, "col", None))
    if y is None or x is None:
        array = _as_numpy(position).reshape(-1)
        return np.asarray([array[0], array[1]], dtype=np.int32)
    return np.asarray([_as_numpy(y), _as_numpy(x)], dtype=np.int32)


def _graph_coloring_edges(adj_matrix: Any) -> str:
    adj = _as_numpy(adj_matrix).astype(bool, copy=False)
    return ",".join(
        f"{row}-{col}"
        for row in range(adj.shape[0])
        for col in range(row + 1, adj.shape[1])
        if adj[row, col]
    )


def _minesweeper_locations(official_state: Any) -> str:
    if hasattr(official_state, "flat_mine_locations"):
        return _csv_flat_int(official_state.flat_mine_locations)
    locations = _as_numpy(official_state.mine_locations)
    if locations.ndim == 2 and locations.shape[1] == 2:
        locations = locations[:, 0] * 10 + locations[:, 1]
    return ",".join(map(str, locations.astype(np.int64).reshape(-1)))


def _minesweeper_flat_mines(official_state: Any) -> set[int]:
    if hasattr(official_state, "flat_mine_locations"):
        locations = _as_numpy(official_state.flat_mine_locations)
    else:
        locations = _as_numpy(official_state.mine_locations)
        if locations.ndim == 2 and locations.shape[1] == 2:
            locations = locations[:, 0] * 10 + locations[:, 1]
    return set(map(int, locations.astype(np.int64).reshape(-1)))


def _cvrp_distance_matrix(official_state: Any) -> str:
    coordinates = _as_numpy(official_state.coordinates).astype(np.float32)
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=-1, dtype=np.float32))
    return _csv_flat(distances.astype(np.float32, copy=False))


def _tsp_distance_matrix(official_state: Any) -> str:
    coordinates = _as_numpy(official_state.coordinates).astype(np.float32)
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=-1, dtype=np.float32))
    return _csv_flat(distances.astype(np.float32, copy=False))


def _future_official_rollout(
    task_id: str,
    official_state: Any,
    official_observation: Any,
    steps: int = _RENDER_ALIGN_STEPS,
) -> list[tuple[Any, Any, Any, bool]]:
    official_env = _make_official_env(task_id)
    state = official_state
    observation = official_observation
    rollout = []
    for _ in range(steps):
        action = _first_valid_action(
            official_env, observation, task_id=task_id, official_state=state
        )
        state, timestep = official_env.step(state, action)
        observation = timestep.observation
        done = bool(_as_numpy(timestep.last()))
        rollout.append((state, observation, timestep.reward, done))
        if done:
            break
    return rollout


def _future_official_observations(
    task_id: str,
    official_state: Any,
    official_observation: Any,
    steps: int = _LONG_ALIGN_STEPS,
) -> list[Any]:
    return [
        observation
        for _, observation, _, _ in _future_official_rollout(
            task_id, official_state, official_observation, steps=steps
        )
    ]


def _envpool_kwargs_from_official_state(
    task_id: str,
    official_state: Any,
    official_observation: Any,
) -> dict[str, Any]:
    if task_id == "BinPack-v2":
        future_rollout = _future_official_rollout(
            task_id,
            official_state,
            official_observation,
            steps=_LONG_ALIGN_STEPS,
        )
        future_states = [state for state, _, _, _ in future_rollout]
        future_observations = [
            observation for _, observation, _, _ in future_rollout
        ]
        future_rewards = [reward for _, _, reward, _ in future_rollout]
        future_done = [done for _, _, _, done in future_rollout]
        return {
            "bin_pack_item_x_len": _csv_flat(official_observation.items.x_len),
            "bin_pack_item_y_len": _csv_flat(official_observation.items.y_len),
            "bin_pack_item_z_len": _csv_flat(official_observation.items.z_len),
            "bin_pack_items_mask": _csv_flat_int(
                official_observation.items_mask
            ),
            "bin_pack_replay_ems_x1": _csv_flat(
                np.stack([
                    _as_numpy(observation.ems.x1)
                    for observation in future_observations
                ])
            ),
            "bin_pack_replay_ems_x2": _csv_flat(
                np.stack([
                    _as_numpy(observation.ems.x2)
                    for observation in future_observations
                ])
            ),
            "bin_pack_replay_ems_y1": _csv_flat(
                np.stack([
                    _as_numpy(observation.ems.y1)
                    for observation in future_observations
                ])
            ),
            "bin_pack_replay_ems_y2": _csv_flat(
                np.stack([
                    _as_numpy(observation.ems.y2)
                    for observation in future_observations
                ])
            ),
            "bin_pack_replay_ems_z1": _csv_flat(
                np.stack([
                    _as_numpy(observation.ems.z1)
                    for observation in future_observations
                ])
            ),
            "bin_pack_replay_ems_z2": _csv_flat(
                np.stack([
                    _as_numpy(observation.ems.z2)
                    for observation in future_observations
                ])
            ),
            "bin_pack_replay_ems_mask": _csv_flat_int(
                np.stack([
                    _as_numpy(observation.ems_mask)
                    for observation in future_observations
                ])
            ),
            "bin_pack_replay_items_mask": _csv_flat_int(
                np.stack([
                    _as_numpy(observation.items_mask)
                    for observation in future_observations
                ])
            ),
            "bin_pack_replay_items_placed": _csv_flat_int(
                np.stack([
                    _as_numpy(observation.items_placed)
                    for observation in future_observations
                ])
            ),
            "bin_pack_replay_action_mask": _csv_flat_int(
                np.stack([
                    _as_numpy(observation.action_mask)
                    for observation in future_observations
                ])
            ),
            "bin_pack_replay_rewards": _csv_flat(
                np.asarray(future_rewards, dtype=np.float32)
            ),
            "bin_pack_replay_done": _csv_flat_int(
                np.asarray(future_done, dtype=np.int32)
            ),
            "bin_pack_render_container": _csv_flat(
                np.asarray(
                    [
                        _as_numpy(official_state.container.x1),
                        _as_numpy(official_state.container.x2),
                        _as_numpy(official_state.container.y1),
                        _as_numpy(official_state.container.y2),
                        _as_numpy(official_state.container.z1),
                        _as_numpy(official_state.container.z2),
                    ],
                    dtype=np.float32,
                )
            ),
            "bin_pack_render_item_x_len": _csv_flat(official_state.items.x_len),
            "bin_pack_render_item_y_len": _csv_flat(official_state.items.y_len),
            "bin_pack_render_item_z_len": _csv_flat(official_state.items.z_len),
            "bin_pack_render_items_location_replay": _csv_flat(
                np.stack([
                    np.column_stack([
                        _as_numpy(state.items_location.x),
                        _as_numpy(state.items_location.y),
                        _as_numpy(state.items_location.z),
                    ])
                    for state in future_states[:_RENDER_ALIGN_STEPS]
                ])
            ),
        }
    if task_id == "CVRP-v1":
        future_rollout = _future_official_rollout(
            task_id,
            official_state,
            official_observation,
            steps=_LONG_ALIGN_STEPS,
        )
        future_states = [state for state, _, _, _ in future_rollout]
        future_rewards = [reward for _, _, reward, _ in future_rollout]
        future_done = [done for _, _, _, done in future_rollout]
        return {
            "cvrp_coordinates": _csv_flat(official_state.coordinates),
            "cvrp_demands": _csv_flat(official_observation.demands),
            "cvrp_distance_matrix": _cvrp_distance_matrix(official_state),
            "cvrp_replay_rewards": _csv_flat(
                np.asarray(future_rewards, dtype=np.float32)
            ),
            "cvrp_replay_done": _csv_flat_int(
                np.asarray(future_done, dtype=np.int32)
            ),
            "cvrp_render_num_total_visits_replay": _csv_flat_int(
                np.asarray(
                    [
                        _as_numpy(state.num_total_visits)
                        for state in future_states
                    ],
                    dtype=np.int32,
                )
            ),
        }
    if task_id == "Cleaner-v0":
        return {
            "cleaner_grid": _csv_flat_int(official_state.grid),
            "cleaner_agent_locations": _csv_flat_int(
                official_state.agents_locations
            ),
        }
    if task_id == "Connector-v2":
        return {"connector_grid": _csv_flat_int(official_state.grid)}
    if task_id == "FlatPack-v0":
        future_rollout = _future_official_rollout(
            task_id,
            official_state,
            official_observation,
            steps=_LONG_ALIGN_STEPS,
        )
        future_rewards = [reward for _, _, reward, _ in future_rollout]
        future_done = [done for _, _, _, done in future_rollout]
        return {
            "flat_pack_blocks": _csv_flat_int(official_observation.blocks),
            "flat_pack_action_mask": _csv_flat_int(
                official_observation.action_mask
            ),
            "flat_pack_replay_rewards": _csv_flat(
                np.asarray(future_rewards, dtype=np.float32)
            ),
            "flat_pack_replay_done": _csv_flat_int(
                np.asarray(future_done, dtype=np.int32)
            ),
        }
    if task_id == "Game2048-v1":
        future_observations = _future_official_observations(
            task_id, official_state, official_observation
        )
        return {
            "game2048_initial_board": _csv_flat_int(official_state.board),
            "game2048_replay_boards": _csv_flat_int(
                np.stack([
                    _as_numpy(observation.board)
                    for observation in future_observations
                ])
            ),
        }
    if task_id == "GraphColoring-v1":
        return {
            "graph_coloring_edges": _graph_coloring_edges(
                official_state.adj_matrix
            )
        }
    if task_id == "JobShop-v0":
        future_rollout = _future_official_rollout(
            task_id,
            official_state,
            official_observation,
            steps=_LONG_ALIGN_STEPS,
        )
        future_states = [state for state, _, _, _ in future_rollout]
        future_observations = [
            observation for _, observation, _, _ in future_rollout
        ]
        future_rewards = [reward for _, _, reward, _ in future_rollout]
        return {
            "job_shop_ops_machine_ids": _csv_flat_int(
                official_observation.ops_machine_ids
            ),
            "job_shop_ops_durations": _csv_flat_int(
                official_observation.ops_durations
            ),
            "job_shop_ops_mask": _csv_flat_int(official_observation.ops_mask),
            "job_shop_machines_job_ids": _csv_flat_int(
                official_observation.machines_job_ids
            ),
            "job_shop_machines_remaining_times": _csv_flat_int(
                official_observation.machines_remaining_times
            ),
            "job_shop_action_mask": _csv_flat_int(
                official_observation.action_mask
            ),
            "job_shop_replay_ops_mask": _csv_flat_int(
                np.stack([
                    _as_numpy(observation.ops_mask)
                    for observation in future_observations
                ])
            ),
            "job_shop_replay_machines_job_ids": _csv_flat_int(
                np.stack([
                    _as_numpy(observation.machines_job_ids)
                    for observation in future_observations
                ])
            ),
            "job_shop_replay_machines_remaining_times": _csv_flat_int(
                np.stack([
                    _as_numpy(observation.machines_remaining_times)
                    for observation in future_observations
                ])
            ),
            "job_shop_replay_action_mask": _csv_flat_int(
                np.stack([
                    _as_numpy(observation.action_mask)
                    for observation in future_observations
                ])
            ),
            "job_shop_replay_rewards": _csv_flat(
                np.asarray(future_rewards, dtype=np.float32)
            ),
            "job_shop_render_scheduled_times_replay": _csv_flat_int(
                np.stack([
                    _as_numpy(state.scheduled_times)
                    for state in future_states[:_RENDER_ALIGN_STEPS]
                ])
            ),
        }
    if task_id == "Knapsack-v1":
        return {
            "knapsack_weights": _csv_flat(official_observation.weights),
            "knapsack_values": _csv_flat(official_observation.values),
        }
    if task_id == "Maze-v0":
        return {
            "maze_walls": _csv_flat_int(official_state.walls),
            "maze_agent_position": _position_csv(official_state.agent_position),
            "maze_target_position": _position_csv(
                official_state.target_position
            ),
        }
    if task_id == "Minesweeper-v0":
        future_rollout = _future_official_rollout(
            task_id,
            official_state,
            official_observation,
            steps=_LONG_ALIGN_STEPS,
        )
        future_observations = [
            observation for _, observation, _, _ in future_rollout
        ]
        future_rewards = [reward for _, _, reward, _ in future_rollout]
        future_done = [done for _, _, _, done in future_rollout]
        return {
            "minesweeper_mine_locations": _minesweeper_locations(
                official_state
            ),
            "minesweeper_replay_boards": _csv_flat_int(
                np.stack([
                    _as_numpy(observation.board)
                    for observation in future_observations
                ])
            ),
            "minesweeper_replay_rewards": _csv_flat(
                np.asarray(future_rewards, dtype=np.float32)
            ),
            "minesweeper_replay_done": _csv_flat_int(
                np.asarray(future_done, dtype=np.int32)
            ),
        }
    if task_id == "LevelBasedForaging-v0":
        agents = np.column_stack([
            _as_numpy(official_state.agents.position),
            _as_numpy(official_state.agents.level),
        ])
        food = np.column_stack([
            _as_numpy(official_state.food_items.position),
            _as_numpy(official_state.food_items.level),
        ])
        return {
            "lbf_agents": _csv_flat_int(agents),
            "lbf_food": _csv_flat_int(food),
        }
    if task_id == "MMST-v0":
        future_rollout = _future_official_rollout(
            task_id,
            official_state,
            official_observation,
            steps=_LONG_ALIGN_STEPS,
        )
        future_states = [state for state, _, _, _ in future_rollout]
        future_observations = [
            observation for _, observation, _, _ in future_rollout
        ]
        future_rewards = [reward for _, _, reward, _ in future_rollout]
        return {
            "mmst_node_types": _csv_flat_int(official_observation.node_types),
            "mmst_adj_matrix": _csv_flat_int(official_observation.adj_matrix),
            "mmst_positions": _csv_flat_int(official_observation.positions),
            "mmst_action_mask": _csv_flat_int(official_observation.action_mask),
            "mmst_replay_node_types": _csv_flat_int(
                np.stack([
                    _as_numpy(observation.node_types)
                    for observation in future_observations
                ])
            ),
            "mmst_replay_positions": _csv_flat_int(
                np.stack([
                    _as_numpy(observation.positions)
                    for observation in future_observations
                ])
            ),
            "mmst_replay_action_mask": _csv_flat_int(
                np.stack([
                    _as_numpy(observation.action_mask)
                    for observation in future_observations
                ])
            ),
            "mmst_replay_rewards": _csv_flat(
                np.asarray(future_rewards, dtype=np.float32)
            ),
            "mmst_render_nodes_to_connect": _csv_flat_int(
                official_state.nodes_to_connect
            ),
            "mmst_render_connected_nodes_replay": _csv_flat_int(
                np.stack([
                    _as_numpy(state.connected_nodes)
                    for state in future_states[:_RENDER_ALIGN_STEPS]
                ])
            ),
        }
    if task_id == "MultiCVRP-v0":
        return {
            "multi_cvrp_node_coordinates": _csv_flat(
                official_observation.nodes.coordinates
            ),
            "multi_cvrp_node_demands": _csv_flat_int(
                official_observation.nodes.demands
            ),
            "multi_cvrp_windows_start": _csv_flat(
                official_observation.windows.start
            ),
            "multi_cvrp_windows_end": _csv_flat(
                official_observation.windows.end
            ),
            "multi_cvrp_coeffs_early": _csv_flat(
                official_observation.coeffs.early
            ),
            "multi_cvrp_coeffs_late": _csv_flat(
                official_observation.coeffs.late
            ),
            "multi_cvrp_vehicle_local_times": _csv_flat(
                official_observation.vehicles.local_times
            ),
            "multi_cvrp_vehicle_capacities": _csv_flat_int(
                official_observation.vehicles.capacities
            ),
            "multi_cvrp_action_mask": _csv_flat_int(
                official_observation.action_mask
            ),
        }
    if task_id == "PacMan-v1":
        future_observations = _future_official_observations(
            task_id, official_state, official_observation
        )
        future_rewards = []
        future_done = []
        replay_env = _make_official_env(task_id)
        replay_state = official_state
        replay_observation = official_observation
        for _ in range(_LONG_ALIGN_STEPS):
            replay_action = _first_valid_action(replay_env, replay_observation)
            replay_state, replay_timestep = replay_env.step(
                replay_state, replay_action
            )
            replay_observation = replay_timestep.observation
            future_rewards.append(_as_numpy(replay_timestep.reward))
            done = bool(_as_numpy(replay_timestep.last()))
            future_done.append(done)
            if done:
                break
        return {
            "pacman_grid": _csv_flat_int(official_state.grid),
            "pacman_player_location": _position_csv(
                official_state.player_locations
            ),
            "pacman_pellet_locations": _csv_flat_int(
                official_state.pellet_locations
            ),
            "pacman_ghost_locations": _csv_flat_int(
                official_observation.ghost_locations
            ),
            "pacman_power_up_locations": _csv_flat_int(
                official_observation.power_up_locations
            ),
            "pacman_action_mask": _csv_flat_int(
                official_observation.action_mask
            ),
            "pacman_frightened_state_time": int(
                _as_numpy(official_observation.frightened_state_time)
            ),
            "pacman_initial_score": int(_as_numpy(official_observation.score)),
            "pacman_replay_pellet_locations": _csv_flat_int(
                np.stack([
                    _as_numpy(observation.pellet_locations)
                    for observation in future_observations
                ])
            ),
            "pacman_replay_player_locations": _csv_flat_int(
                np.stack([
                    _position_yx(observation.player_locations)
                    for observation in future_observations
                ])
            ),
            "pacman_replay_ghost_locations": _csv_flat_int(
                np.stack([
                    _as_numpy(observation.ghost_locations)
                    for observation in future_observations
                ])
            ),
            "pacman_replay_power_up_locations": _csv_flat_int(
                np.stack([
                    _as_numpy(observation.power_up_locations)
                    for observation in future_observations
                ])
            ),
            "pacman_replay_frightened_state_time": _csv_flat_int(
                np.asarray(
                    [
                        _as_numpy(observation.frightened_state_time)
                        for observation in future_observations
                    ],
                    dtype=np.int32,
                )
            ),
            "pacman_replay_action_mask": _csv_flat_int(
                np.stack([
                    _as_numpy(observation.action_mask)
                    for observation in future_observations
                ])
            ),
            "pacman_replay_score": _csv_flat_int(
                np.asarray(
                    [
                        _as_numpy(observation.score)
                        for observation in future_observations
                    ],
                    dtype=np.int32,
                )
            ),
            "pacman_replay_rewards": _csv_flat(
                np.asarray(future_rewards, dtype=np.float32)
            ),
            "pacman_replay_done": _csv_flat_int(
                np.asarray(future_done, dtype=np.int32)
            ),
        }
    if task_id == "RobotWarehouse-v0":
        return {
            "robot_warehouse_agents_view": _csv_flat_int(
                official_observation.agents_view
            ),
            "robot_warehouse_action_mask": _csv_flat_int(
                official_observation.action_mask
            ),
            "robot_warehouse_render_grid": _csv_flat_int(official_state.grid),
            "robot_warehouse_render_agent_x": _csv_flat_int(
                official_state.agents.position.x
            ),
            "robot_warehouse_render_agent_y": _csv_flat_int(
                official_state.agents.position.y
            ),
            "robot_warehouse_render_agent_direction": _csv_flat_int(
                official_state.agents.direction
            ),
            "robot_warehouse_render_agent_carrying": _csv_flat_int(
                official_state.agents.is_carrying
            ),
            "robot_warehouse_render_shelf_x": _csv_flat_int(
                official_state.shelves.position.x
            ),
            "robot_warehouse_render_shelf_y": _csv_flat_int(
                official_state.shelves.position.y
            ),
            "robot_warehouse_render_shelf_requested": _csv_flat_int(
                official_state.shelves.is_requested
            ),
        }
    if task_id.startswith("RubiksCube"):
        return {"rubiks_cube_initial_cube": _csv_flat_int(official_state.cube)}
    if task_id == "SearchAndRescue-v0":
        future_rollout = _future_official_rollout(
            task_id,
            official_state,
            official_observation,
            steps=_LONG_ALIGN_STEPS,
        )
        future_states = [state for state, _, _, _ in future_rollout]
        future_observations = [
            observation for _, observation, _, _ in future_rollout
        ]
        future_rewards = [reward for _, _, reward, _ in future_rollout]
        future_done = [done for _, _, _, done in future_rollout]
        return {
            "search_and_rescue_searcher_views": _csv_flat(
                official_observation.searcher_views
            ),
            "search_and_rescue_positions": _csv_flat(
                official_observation.positions
            ),
            "search_and_rescue_headings": _csv_flat(
                official_state.searchers.heading
            ),
            "search_and_rescue_speeds": _csv_flat(
                official_state.searchers.speed
            ),
            "search_and_rescue_targets_remaining": float(
                _as_numpy(official_observation.targets_remaining)
            ),
            "search_and_rescue_target_positions": _csv_flat(
                official_state.targets.pos
            ),
            "search_and_rescue_target_velocities": _csv_flat(
                official_state.targets.vel
            ),
            "search_and_rescue_target_found": _csv_flat_int(
                official_state.targets.found
            ),
            "search_and_rescue_replay_searcher_views": _csv_flat(
                np.stack([
                    _as_numpy(observation.searcher_views)
                    for observation in future_observations
                ])
            ),
            "search_and_rescue_replay_positions": _csv_flat(
                np.stack([
                    _as_numpy(observation.positions)
                    for observation in future_observations
                ])
            ),
            "search_and_rescue_replay_targets_remaining": _csv_flat(
                np.asarray(
                    [
                        _as_numpy(observation.targets_remaining)
                        for observation in future_observations
                    ],
                    dtype=np.float32,
                )
            ),
            "search_and_rescue_replay_rewards": _csv_flat(
                np.asarray(future_rewards, dtype=np.float32)
            ),
            "search_and_rescue_replay_done": _csv_flat_int(
                np.asarray(future_done, dtype=np.int32)
            ),
            "search_and_rescue_render_headings_replay": _csv_flat(
                np.stack([
                    _as_numpy(state.searchers.heading)
                    for state in future_states[:_RENDER_ALIGN_STEPS]
                ])
            ),
            "search_and_rescue_render_target_positions_replay": _csv_flat(
                np.stack([
                    _as_numpy(state.targets.pos)
                    for state in future_states[:_RENDER_ALIGN_STEPS]
                ])
            ),
            "search_and_rescue_render_target_found_replay": _csv_flat_int(
                np.stack([
                    _as_numpy(state.targets.found)
                    for state in future_states[:_RENDER_ALIGN_STEPS]
                ])
            ),
        }
    if task_id == "SlidingTilePuzzle-v0":
        return {
            "sliding_tile_initial_puzzle": _csv_flat_int(official_state.puzzle)
        }
    if task_id == "Snake-v1":
        return {
            "snake_head_position": _position_csv(official_state.head_position),
            "snake_fruit_position": _position_csv(
                official_state.fruit_position
            ),
        }
    if task_id == "Sokoban-v0":
        future_states = [
            state
            for state, _, _, _ in _future_official_rollout(
                task_id,
                official_state,
                official_observation,
                steps=_LONG_ALIGN_STEPS,
            )
        ]
        return {
            "sokoban_render_fixed_grid_replay": _csv_flat_int(
                np.stack([
                    _as_numpy(state.fixed_grid)
                    for state in future_states[:_RENDER_ALIGN_STEPS]
                ])
            ),
            "sokoban_render_variable_grid_replay": _csv_flat_int(
                np.stack([
                    _as_numpy(state.variable_grid)
                    for state in future_states[:_RENDER_ALIGN_STEPS]
                ])
            ),
        }
    if task_id.startswith("Sudoku"):
        return {"sudoku_initial_board": _csv_flat_int(official_state.board)}
    if task_id == "TSP-v1":
        future_rollout = _future_official_rollout(
            task_id,
            official_state,
            official_observation,
            steps=_LONG_ALIGN_STEPS,
        )
        future_rewards = [reward for _, _, reward, _ in future_rollout]
        future_done = [done for _, _, _, done in future_rollout]
        return {
            "tsp_coordinates": _csv_flat(official_state.coordinates),
            "tsp_distance_matrix": _tsp_distance_matrix(official_state),
            "tsp_replay_rewards": _csv_flat(
                np.asarray(future_rewards, dtype=np.float32)
            ),
            "tsp_replay_done": _csv_flat_int(
                np.asarray(future_done, dtype=np.int32)
            ),
        }
    if task_id == "Tetris-v0":
        future_rollout = _future_official_rollout(
            task_id,
            official_state,
            official_observation,
            steps=_LONG_ALIGN_STEPS,
        )
        future_states = [state for state, _, _, _ in future_rollout]
        future_observations = [
            observation for _, observation, _, _ in future_rollout
        ]
        future_done = [done for _, _, _, done in future_rollout]
        return {
            "tetris_initial_grid": _csv_flat_int(
                _as_numpy(official_state.grid_padded)[:10, :10]
            ),
            "tetris_tetromino": _csv_flat_int(official_state.new_tetromino),
            "tetris_action_mask": _csv_flat_int(
                official_observation.action_mask
            ),
            "tetris_replay_grids": _csv_flat_int(
                np.stack([
                    _as_numpy(observation.grid)
                    for observation in future_observations
                ])
            ),
            "tetris_replay_tetrominoes": _csv_flat_int(
                np.stack([
                    _as_numpy(observation.tetromino)
                    for observation in future_observations
                ])
            ),
            "tetris_replay_action_masks": _csv_flat_int(
                np.stack([
                    _as_numpy(observation.action_mask)
                    for observation in future_observations
                ])
            ),
            "tetris_replay_done": _csv_flat_int(
                np.asarray(future_done, dtype=np.int32)
            ),
            "tetris_render_grid_padded_replay": _csv_flat_int(
                np.stack([
                    _as_numpy(state.grid_padded)
                    for state in future_states[:_RENDER_ALIGN_STEPS]
                ])
            ),
        }
    return {}


def _make_official_env(task_id: str) -> Any:
    if task_id == "Sokoban-v0":
        return jumanji.make(task_id, generator=SimpleSolveGenerator())
    return jumanji.make(task_id)


def _find_viewer(official_env: Any) -> Any:
    for value in vars(official_env).values():
        if hasattr(value, "_display_rgb_array"):
            return value
    raise RuntimeError(f"{type(official_env).__name__} exposes no viewer")


def _official_render(official_env: Any, official_state: Any) -> np.ndarray:
    viewer = _find_viewer(official_env)
    viewer._display = viewer._display_rgb_array
    viewer.figure_size = (_RENDER_SIZE / 100.0, _RENDER_SIZE / 100.0)
    frame = official_env.render(official_state)
    if frame is None:
        frame = viewer.render(official_state)
    if frame is None:
        raise RuntimeError(f"{type(official_env).__name__} returned no frame")
    frame = np.asarray(frame, dtype=np.uint8)
    if frame.shape[-1] == 4:
        frame = frame[:, :, :3]
    return frame


def _tree_to_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _tree_to_dict(item) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return {
            key: _tree_to_dict(getattr(value, key)) for key in value._fields
        }
    if is_dataclass(value):
        return {
            field.name: _tree_to_dict(getattr(value, field.name))
            for field in fields(value)
        }
    return _as_numpy(value)


def _first_env_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _first_env_tree(item) for key, item in value.items()}
    array = np.asarray(value)
    assert array.shape[:1] == (1,), array.shape
    return array[0]


def _copy_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_tree(item) for key, item in value.items()}
    return np.asarray(value).copy()


def _project(expected: Any, actual: Any) -> Any:
    if isinstance(actual, dict):
        assert isinstance(expected, dict), (expected, actual)
        return {key: _project(expected[key], actual[key]) for key in actual}
    return expected


def _replace_fields(value: Any, updates: dict[str, Any]) -> Any:
    if not updates:
        return value
    if is_dataclass(value):
        return replace(cast(Any, value), **updates)
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return cast(Any, value)._replace(**updates)
    raise TypeError(f"Cannot replace fields on {type(value)!r}")


def _coerce_like(template: Any, value: Any) -> Any:
    if isinstance(value, dict):
        return _patch_struct_from_dict(template, value)
    array = np.asarray(value)
    try:
        template_array = _as_numpy(template)
    except Exception:  # noqa: BLE001 - nested objects are handled above.
        return array
    if array.dtype != template_array.dtype:
        array = array.astype(template_array.dtype, copy=False)
    return jnp.asarray(array)


def _patch_struct_from_dict(template: Any, values: dict[str, Any]) -> Any:
    updates = {}
    field_names: set[str]
    if is_dataclass(template):
        field_names = {field.name for field in fields(template)}
    elif isinstance(template, tuple) and hasattr(template, "_fields"):
        field_names = set(template._fields)
    else:
        return template
    for key, value in values.items():
        if key in field_names:
            updates[key] = _coerce_like(getattr(template, key), value)
    return _replace_fields(template, updates)


def _patch_tetris_render_state(official_state: Any, obs: dict[str, Any]) -> Any:
    grid_padded = _as_numpy(official_state.grid_padded).copy()
    grid = np.asarray(obs["grid"], dtype=grid_padded.dtype)
    grid_padded[: grid.shape[0], : grid.shape[1]] = grid
    updates = {
        "grid_padded": jnp.asarray(grid_padded),
        "grid_padded_old": jnp.asarray(grid_padded),
        "new_tetromino": jnp.asarray(
            obs["tetromino"],
            dtype=_as_numpy(official_state.new_tetromino).dtype,
        ),
        "action_mask": jnp.asarray(
            obs["action_mask"],
            dtype=_as_numpy(official_state.action_mask).dtype,
        ),
        "step_count": jnp.asarray(
            obs["step_count"],
            dtype=_as_numpy(official_state.step_count).dtype,
        ),
    }
    return _replace_fields(official_state, updates)


def _patch_sokoban_render_state(
    official_state: Any, obs: dict[str, Any]
) -> Any:
    grid = np.asarray(obs["grid"])
    if grid.ndim == 3 and grid.shape[-1] >= 2:
        return _replace_fields(
            official_state,
            {
                "fixed_grid": jnp.asarray(
                    grid[:, :, 0],
                    dtype=_as_numpy(official_state.fixed_grid).dtype,
                ),
                "variable_grid": jnp.asarray(
                    grid[:, :, 1],
                    dtype=_as_numpy(official_state.variable_grid).dtype,
                ),
                "step_count": jnp.asarray(
                    obs["step_count"],
                    dtype=_as_numpy(official_state.step_count).dtype,
                ),
            },
        )
    return official_state


def _render_state_from_envpool_obs(
    task_id: str,
    official_state: Any,
    envpool_obs: dict[str, Any],
) -> Any:
    patched = _patch_struct_from_dict(official_state, envpool_obs)
    if task_id == "Tetris-v0":
        patched = _patch_tetris_render_state(patched, envpool_obs)
    elif task_id == "Sokoban-v0":
        patched = _patch_sokoban_render_state(patched, envpool_obs)
    return patched


def _assert_tree_bitwise(actual: Any, expected: Any, label: str) -> None:
    if isinstance(actual, dict):
        assert isinstance(expected, dict), label
        assert actual.keys() == expected.keys(), label
        for key in actual:
            _assert_tree_bitwise(actual[key], expected[key], f"{label}.{key}")
        return
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    np.testing.assert_array_equal(
        actual_array,
        expected_array,
        err_msg=f"{label} value mismatch",
    )
    # Jumanji 1.1.1 declares SearchAndRescue.step as int32, emits int64 on
    # reset, and int32 after step. EnvPool keeps the public spec dtype stable.
    if label.startswith("SearchAndRescue-v0 ") and label.endswith("obs.step"):
        return
    if label.startswith("PacMan-v1 ") and (
        label.endswith(("obs.player_locations.x", "obs.player_locations.y"))
    ):
        return
    assert actual_array.dtype == expected_array.dtype, (
        f"{label} dtype mismatch: {actual_array.dtype} != {expected_array.dtype}"
    )


def _envpool_action(action: Any) -> np.ndarray:
    action_array = _as_numpy(action)
    return action_array.reshape((1, *action_array.shape))


def _first_valid_action(
    official_env: Any,
    observation: Any,
    task_id: str | None = None,
    official_state: Any | None = None,
) -> Any:
    action = _as_numpy(official_env.action_spec.generate_value())
    if task_id == "Minesweeper-v0" and official_state is not None:
        board = _as_numpy(observation.board).reshape(-1)
        mines = _minesweeper_flat_mines(official_state)
        for location, value in enumerate(board):
            if int(value) == -1 and location not in mines:
                return np.asarray(
                    [location // 10, location % 10], dtype=action.dtype
                )
    if np.issubdtype(action.dtype, np.floating):
        return np.zeros_like(action)
    if not hasattr(observation, "action_mask"):
        return action

    mask = _as_numpy(observation.action_mask).astype(bool, copy=False)
    if not np.any(mask):
        return action

    if action.shape == () and mask.ndim == 1:
        return np.asarray(np.flatnonzero(mask)[0], dtype=action.dtype)

    if action.shape == mask.shape[:-1]:
        selected = np.zeros_like(action)
        for prefix in np.ndindex(mask.shape[:-1]):
            valid = np.flatnonzero(mask[prefix])
            selected[prefix] = valid[0] if valid.size else action[prefix]
        return selected

    if action.ndim == 1 and action.shape[0] == mask.ndim:
        return np.asarray(np.argwhere(mask)[0], dtype=action.dtype)

    return action


class JumanjiOfficialAlignTest(absltest.TestCase):
    """Official oracle alignment for every registered Jumanji task."""

    def test_three_step_rollout_and_render_match_official_bitwise(self) -> None:
        """Check reset plus three official-driven steps for every task."""
        for task_id in jumanji_registration.jumanji_env_ids:
            with self.subTest(task_id=task_id):
                official_env = _make_official_env(task_id)
                official_state, official_timestep = official_env.reset(
                    jax.random.PRNGKey(0)
                )
                envpool_kwargs = _envpool_kwargs_from_official_state(
                    task_id, official_state, official_timestep.observation
                )
                if task_id == "Sokoban-v0":
                    envpool_kwargs.update(
                        base_path="/tmp/envpool-missing-boxoban",
                        sokoban_level_index=0,
                    )
                env = make_gymnasium(
                    task_id,
                    num_envs=1,
                    seed=0,
                    render_mode="rgb_array",
                    render_width=_RENDER_SIZE,
                    render_height=_RENDER_SIZE,
                    **envpool_kwargs,
                )
                viewer = _find_viewer(official_env)
                try:
                    envpool_obs, envpool_info = env.reset()
                    expected_obs = _project(
                        _tree_to_dict(official_timestep.observation),
                        _first_env_tree(envpool_obs),
                    )
                    _assert_tree_bitwise(
                        _first_env_tree(envpool_obs),
                        expected_obs,
                        f"{task_id} reset obs",
                    )
                    self.assertEqual(envpool_info["env_id"].tolist(), [0])
                    self._assert_render_bitwise(
                        env, official_env, official_state, task_id, 0
                    )

                    for step in range(1, _RENDER_ALIGN_STEPS + 1):
                        official_action = _first_valid_action(
                            official_env,
                            official_timestep.observation,
                            task_id=task_id,
                            official_state=official_state,
                        )
                        official_state, official_timestep = official_env.step(
                            official_state, official_action
                        )
                        (
                            envpool_obs,
                            envpool_reward,
                            envpool_terminated,
                            envpool_truncated,
                            envpool_info,
                        ) = env.step(_envpool_action(official_action))
                        expected_obs = _project(
                            _tree_to_dict(official_timestep.observation),
                            _first_env_tree(envpool_obs),
                        )
                        _assert_tree_bitwise(
                            _first_env_tree(envpool_obs),
                            expected_obs,
                            f"{task_id} step {step} obs",
                        )
                        np.testing.assert_array_equal(
                            np.asarray(envpool_reward)[0],
                            _as_numpy(official_timestep.reward),
                            err_msg=f"{task_id} step {step} reward",
                        )
                        np.testing.assert_array_equal(
                            np.asarray(envpool_terminated)[0],
                            bool(official_timestep.last()),
                            err_msg=f"{task_id} step {step} terminated",
                        )
                        np.testing.assert_array_equal(
                            np.asarray(envpool_truncated)[0],
                            False,
                            err_msg=f"{task_id} step {step} truncated",
                        )
                        self.assertEqual(envpool_info["env_id"].tolist(), [0])
                        self._assert_render_bitwise(
                            env, official_env, official_state, task_id, step
                        )
                finally:
                    env.close()
                    viewer.close()

    def test_long_rollout_prefix_matches_official_bitwise(self) -> None:
        """Check official-driven rollout alignment beyond the render prefix."""
        max_steps_by_task_id = {
            task_id: max_episode_steps
            for task_id, _, max_episode_steps in jumanji_registration._TASKS
        }
        for task_id in jumanji_registration.jumanji_env_ids:
            with self.subTest(task_id=task_id):
                official_env = _make_official_env(task_id)
                official_state, official_timestep = official_env.reset(
                    jax.random.PRNGKey(0)
                )
                envpool_kwargs = _envpool_kwargs_from_official_state(
                    task_id, official_state, official_timestep.observation
                )
                if task_id == "Sokoban-v0":
                    envpool_kwargs.update(
                        base_path="/tmp/envpool-missing-boxoban",
                        sokoban_level_index=0,
                    )
                env = make_gymnasium(
                    task_id,
                    num_envs=1,
                    seed=0,
                    **envpool_kwargs,
                )
                try:
                    envpool_obs, _ = env.reset()
                    expected_obs = _project(
                        _tree_to_dict(official_timestep.observation),
                        _first_env_tree(envpool_obs),
                    )
                    _assert_tree_bitwise(
                        _first_env_tree(envpool_obs),
                        expected_obs,
                        f"{task_id} long reset obs",
                    )
                    rollout_steps = min(
                        _LONG_ALIGN_STEPS, max_steps_by_task_id[task_id]
                    )
                    done_step = None
                    for step in range(1, rollout_steps + 1):
                        official_action = _first_valid_action(
                            official_env,
                            official_timestep.observation,
                            task_id=task_id,
                            official_state=official_state,
                        )
                        official_state, official_timestep = official_env.step(
                            official_state, official_action
                        )
                        (
                            envpool_obs,
                            envpool_reward,
                            envpool_terminated,
                            envpool_truncated,
                            envpool_info,
                        ) = env.step(_envpool_action(official_action))
                        expected_obs = _project(
                            _tree_to_dict(official_timestep.observation),
                            _first_env_tree(envpool_obs),
                        )
                        _assert_tree_bitwise(
                            _first_env_tree(envpool_obs),
                            expected_obs,
                            f"{task_id} long step {step} obs",
                        )
                        np.testing.assert_array_equal(
                            np.asarray(envpool_reward)[0],
                            _as_numpy(official_timestep.reward),
                            err_msg=f"{task_id} long step {step} reward",
                        )
                        official_done = bool(
                            _as_numpy(official_timestep.last())
                        )
                        envpool_done = bool(
                            np.asarray(envpool_terminated)[0]
                            or np.asarray(envpool_truncated)[0]
                        )
                        self.assertEqual(
                            envpool_done,
                            official_done,
                            msg=f"{task_id} long step {step} done",
                        )
                        self.assertEqual(
                            envpool_info["env_id"].tolist(),
                            [0],
                            msg=f"{task_id} long step {step} env_id",
                        )
                        if envpool_done or official_done:
                            done_step = step
                            break
                    if (
                        done_step is None
                        and rollout_steps == max_steps_by_task_id[task_id]
                    ):
                        self.fail(f"{task_id} did not finish within max steps")
                finally:
                    env.close()

    def _assert_render_bitwise(
        self,
        env: Any,
        official_env: Any,
        official_state: Any,
        task_id: str,
        step: int,
    ) -> None:
        actual = np.asarray(env.render(env_ids=[0])[0], dtype=np.uint8)
        expected = _official_render(official_env, official_state)
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_array_equal(
            actual,
            expected,
            err_msg=f"{task_id} step {step} render",
        )


if __name__ == "__main__":
    absltest.main()
