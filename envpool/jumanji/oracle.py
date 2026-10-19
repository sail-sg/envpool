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
"""Pinned Jumanji reset-state and rendering helpers for tests and tooling."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any


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
import jumanji  # noqa: E402
import numpy as np  # noqa: E402
from jumanji.environments.routing.sokoban.generator import (  # noqa: E402
    SimpleSolveGenerator,
)


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


def _envpool_kwargs_from_official_state(
    task_id: str, official_state: Any, official_observation: Any
) -> dict[str, Any]:
    if task_id == "BinPack-v2":
        return {
            "bin_pack_item_x_len": _csv_flat(official_observation.items.x_len),
            "bin_pack_item_y_len": _csv_flat(official_observation.items.y_len),
            "bin_pack_item_z_len": _csv_flat(official_observation.items.z_len),
            "bin_pack_items_mask": _csv_flat_int(
                official_observation.items_mask
            ),
        }
    if task_id == "CVRP-v1":
        return {
            "cvrp_max_capacity": int(_as_numpy(official_state.capacity)),
            "cvrp_coordinates": _csv_flat(official_state.coordinates),
            "cvrp_demands": _csv_flat(official_observation.demands),
            "cvrp_distance_matrix": _cvrp_distance_matrix(official_state),
        }
    if task_id == "Cleaner-v0":
        return {
            "cleaner_grid": _csv_flat_int(official_state.grid),
            "cleaner_agent_locations": _csv_flat_int(
                official_state.agents_locations
            ),
        }
    if task_id == "Connector-v3":
        return {"connector_grid": _csv_flat_int(official_state.grid)}
    if task_id == "FlatPack-v0":
        return {
            "flat_pack_blocks": _csv_flat_int(official_observation.blocks),
            "flat_pack_action_mask": _csv_flat_int(
                official_observation.action_mask
            ),
        }
    if task_id == "Game2048-v1":
        return {"game2048_initial_board": _csv_flat_int(official_state.board)}
    if task_id == "GraphColoring-v1":
        return {
            "graph_coloring_edges": _graph_coloring_edges(
                official_state.adj_matrix
            )
        }
    if task_id == "JobShop-v0":
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
        return {
            "minesweeper_mine_locations": _minesweeper_locations(official_state)
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
        return {
            "mmst_node_types": _csv_flat_int(official_observation.node_types),
            "mmst_adj_matrix": _csv_flat_int(official_observation.adj_matrix),
            "mmst_positions": _csv_flat_int(official_observation.positions),
            "mmst_action_mask": _csv_flat_int(official_observation.action_mask),
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
        }
    if task_id == "RobotWarehouse-v0":
        return {
            "robot_warehouse_agents_view": _csv_flat_int(
                official_observation.agents_view
            ),
            "robot_warehouse_action_mask": _csv_flat_int(
                official_observation.action_mask
            ),
            "robot_warehouse_initial_agent_x": _csv_flat_int(
                official_state.agents.position.x
            ),
            "robot_warehouse_initial_agent_y": _csv_flat_int(
                official_state.agents.position.y
            ),
            "robot_warehouse_initial_agent_direction": _csv_flat_int(
                official_state.agents.direction
            ),
            "robot_warehouse_initial_agent_carrying": _csv_flat_int(
                official_state.agents.is_carrying
            ),
            "robot_warehouse_initial_shelf_x": _csv_flat_int(
                official_state.shelves.position.x
            ),
            "robot_warehouse_initial_shelf_y": _csv_flat_int(
                official_state.shelves.position.y
            ),
            "robot_warehouse_initial_shelf_requested": _csv_flat_int(
                official_state.shelves.is_requested
            ),
        }
    if task_id.startswith("RubiksCube"):
        return {"rubiks_cube_initial_cube": _csv_flat_int(official_state.cube)}
    if task_id == "SearchAndRescue-v0":
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
        return {}
    if task_id.startswith("Sudoku"):
        return {"sudoku_initial_board": _csv_flat_int(official_state.board)}
    if task_id == "TSP-v1":
        return {
            "tsp_coordinates": _csv_flat(official_state.coordinates),
            "tsp_distance_matrix": _tsp_distance_matrix(official_state),
        }
    if task_id == "Tetris-v0":
        return {
            "tetris_initial_grid": _csv_flat_int(
                _as_numpy(official_state.grid_padded)[:10, :10]
            ),
            "tetris_tetromino": _csv_flat_int(official_state.new_tetromino),
            "tetris_action_mask": _csv_flat_int(
                official_observation.action_mask
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


def _official_render(
    official_env: Any, official_state: Any, width: int = 128, height: int = 128
) -> np.ndarray:
    viewer = _find_viewer(official_env)
    viewer._display = viewer._display_rgb_array
    viewer.figure_size = (width / 100.0, height / 100.0)
    frame = official_env.render(official_state)
    if frame is None:
        frame = viewer.render(official_state)
    if frame is None:
        raise RuntimeError(f"{type(official_env).__name__} returned no frame")
    frame = np.asarray(frame, dtype=np.uint8)
    if frame.shape[-1] == 4:
        frame = frame[:, :, :3]
    return frame
