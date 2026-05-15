#!/usr/bin/env python3

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
"""Generate render comparison and sample images for docs."""

from __future__ import annotations

import argparse
import math
import os
import platform
import subprocess
import sys
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envpool.python.glfw_context import preload_windows_gl_dlls

preload_windows_gl_dlls(strict=True)


def _configure_linux_mujoco_gl() -> None:
    """Pick an offscreen MuJoCo GL backend on Linux before importing mujoco."""
    if platform.system() != "Linux" or os.environ.get("MUJOCO_GL"):
        return

    for backend in ("egl", "osmesa"):
        probe_env = os.environ.copy()
        probe_env["MUJOCO_GL"] = backend
        code = "import mujoco; mujoco.GLContext(8, 8).free()"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=probe_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if proc.returncode == 0:
            os.environ["MUJOCO_GL"] = backend
            return


_configure_linux_mujoco_gl()


@dataclass(frozen=True)
class RenderCompareConfig:
    """Shared image generation and comparison settings."""

    family: str
    tile_width: int
    tile_height: int
    source_width: int
    source_height: int
    columns: int
    seed: int
    camera_id: int
    max_mean_abs_diff: float
    max_mismatch_ratio: float
    max_ignored_abs_diff: int
    require_bitwise: bool
    flip_vertical: bool


RenderPairFn = Callable[
    [str, RenderCompareConfig], tuple[np.ndarray, np.ndarray | None]
]


@dataclass(frozen=True)
class RenderItem:
    """One family-specific render key and its display label."""

    key: str
    label: str


@dataclass(frozen=True)
class RenderFamily:
    """Family-specific render integration."""

    items: tuple[RenderItem, ...]
    default_output: Path
    render_pair: RenderPairFn
    left_title: str = "EnvPool"
    right_title: str | None = "Official"
    default_flip_vertical: bool = False
    compare_frames: bool = True


_PAIR_GAP = 4
_CELL_GAP = 12
_HEADER_HEIGHT = 28
_MARGIN = 16


def _make_metaworld_family() -> RenderFamily:
    """Build the MetaWorld render comparison adapter."""
    import mujoco
    from metaworld.env_dict import ALL_V3_ENVIRONMENTS

    import envpool.mujoco.metaworld.registration as metaworld_registration
    from envpool.registration import make_gymnasium

    def make_oracle(task_name: str, cfg: RenderCompareConfig) -> Any:
        env = ALL_V3_ENVIRONMENTS[task_name](
            render_mode="rgb_array",
            camera_id=cfg.camera_id,
        )
        env._set_task_called = True
        env._partially_observable = True
        env.mujoco_renderer.width = cfg.source_width
        env.mujoco_renderer.height = cfg.source_height
        return env

    def sync_reset_state(oracle: Any, info: dict[str, Any]) -> None:
        required_keys = (
            "rand_vec0",
            "qpos0",
            "qvel0",
            "mocap_pos0",
            "mocap_quat0",
            "qacc0",
            "qacc_warmstart0",
            "init_tcp0",
            "init_left_pad0",
            "init_right_pad0",
        )
        missing = [key for key in required_keys if key not in info]
        if missing:
            raise RuntimeError(
                "MetaWorld render comparison must be generated with "
                "`bazel run --config=debug //scripts:render_compare -- "
                "--family=metaworld` so EnvPool exposes reset-sync info. "
                f"Missing keys: {missing}"
            )

        rand_vec = np.asarray(info["rand_vec0"][0], dtype=np.float64)
        random_dim = int(oracle._random_reset_space.low.size)
        oracle._freeze_rand_vec = True
        oracle._last_rand_vec = rand_vec[:random_dim].copy()
        oracle.reset()

        qpos = np.asarray(info["qpos0"][0], dtype=np.float64)[
            : oracle.data.qpos.size
        ]
        qvel = np.asarray(info["qvel0"][0], dtype=np.float64)[
            : oracle.data.qvel.size
        ]
        oracle.set_state(qpos, qvel)
        oracle.data.mocap_pos[0] = np.asarray(
            info["mocap_pos0"][0], dtype=np.float64
        )
        oracle.data.mocap_quat[0] = np.asarray(
            info["mocap_quat0"][0], dtype=np.float64
        )
        oracle.data.qacc[:] = np.asarray(info["qacc0"][0], dtype=np.float64)[
            : oracle.data.qacc.size
        ]
        oracle.data.qacc_warmstart[:] = np.asarray(
            info["qacc_warmstart0"][0], dtype=np.float64
        )[: oracle.data.qacc_warmstart.size]
        mujoco.mj_forward(oracle.model, oracle.data)
        oracle.init_tcp = np.asarray(
            info["init_tcp0"][0], dtype=np.float64
        ).copy()
        oracle.init_left_pad = np.asarray(
            info["init_left_pad0"][0], dtype=np.float64
        ).copy()
        oracle.init_right_pad = np.asarray(
            info["init_right_pad0"][0], dtype=np.float64
        ).copy()
        if hasattr(oracle, "_handle_init_pos"):
            oracle._handle_init_pos = oracle._get_pos_objects().copy()

        curr_obs = oracle._get_curr_obs_combined_no_goal()
        oracle._prev_obs = curr_obs.copy()
        obs = oracle._get_obs().astype(np.float64)
        oracle._last_stable_obs = obs.copy()

    def render_pair(
        task_name: str,
        cfg: RenderCompareConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        task_id = metaworld_registration.metaworld_task_id(task_name)
        env = make_gymnasium(
            task_id,
            num_envs=1,
            seed=cfg.seed,
            render_mode="rgb_array",
            render_width=cfg.source_width,
            render_height=cfg.source_height,
            render_camera_id=cfg.camera_id,
        )
        oracle = make_oracle(task_name, cfg)
        try:
            _, info = env.reset()
            sync_reset_state(oracle, info)
            envpool_frame = env.render()
            if envpool_frame is None:
                raise RuntimeError(f"{task_id} returned no EnvPool frame")
            return envpool_frame[0], oracle.render()
        finally:
            env.close()
            oracle.close()

    return RenderFamily(
        items=tuple(
            RenderItem(
                key=task_name,
                label=metaworld_registration.metaworld_public_task_name(
                    task_name
                ),
            )
            for task_name in metaworld_registration.metaworld_v3_envs
        ),
        default_output=Path(
            "docs/_static/render_samples/metaworld_official_compare.png"
        ),
        render_pair=render_pair,
        default_flip_vertical=True,
    )


def _make_jumanji_family() -> RenderFamily:
    """Build the Jumanji official render comparison adapter."""
    from envpool.jumanji.jumanji_official_render import configure_matplotlib

    configure_matplotlib()
    warnings.filterwarnings(
        "ignore",
        message="FigureCanvasAgg is non-interactive.*",
        category=UserWarning,
    )

    import jax
    import jax.numpy as jnp
    import jumanji
    from jumanji.environments.routing.sokoban.generator import (
        SimpleSolveGenerator,
    )

    import envpool.jumanji.registration as jumanji_registration
    from envpool.registration import make_gymnasium

    def find_viewer(official_env: Any) -> Any:
        for value in vars(official_env).values():
            if hasattr(value, "_display_rgb_array"):
                return value
        raise RuntimeError(
            f"{type(official_env).__name__} exposes no Matplotlib viewer"
        )

    def make_official_env(task_id: str) -> Any:
        if task_id == "Sokoban-v0":
            return jumanji.make(task_id, generator=SimpleSolveGenerator())
        return jumanji.make(task_id)

    def as_numpy(value: Any) -> np.ndarray:
        return np.asarray(jax.device_get(value))

    def csv_flat(value: Any) -> str:
        array = as_numpy(value).reshape(-1)
        return ",".join(map(str, array.tolist()))

    def csv_flat_int(value: Any) -> str:
        array = as_numpy(value).astype(np.int64, copy=False).reshape(-1)
        return ",".join(map(str, array.tolist()))

    def position_csv(position: Any) -> str:
        row = getattr(position, "row", getattr(position, "y", None))
        col = getattr(position, "col", getattr(position, "x", None))
        if row is None or col is None:
            array = as_numpy(position).reshape(-1)
            row, col = array[0], array[1]
        return f"{int(row)},{int(col)}"

    def replace_fields(value: Any, updates: dict[str, Any]) -> Any:
        if not updates:
            return value
        if is_dataclass(value):
            return replace(value, **updates)
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return value._replace(**updates)
        raise TypeError(f"Cannot replace fields on {type(value)!r}")

    def coerce_like(template: Any, value: Any) -> Any:
        if isinstance(value, dict):
            return patch_struct_from_dict(template, value)
        array = np.asarray(value)
        try:
            template_array = as_numpy(template)
        except Exception:  # noqa: BLE001 - nested objects are handled above.
            return array
        if array.dtype != template_array.dtype:
            array = array.astype(template_array.dtype, copy=False)
        return jnp.asarray(array)

    def patch_struct_from_dict(template: Any, values: dict[str, Any]) -> Any:
        updates = {}
        if is_dataclass(template):
            field_names = {field.name for field in fields(template)}
        elif isinstance(template, tuple) and hasattr(template, "_fields"):
            field_names = set(template._fields)
        else:
            return template
        for key, value in values.items():
            if key in field_names:
                updates[key] = coerce_like(getattr(template, key), value)
        return replace_fields(template, updates)

    def first_env_tree(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: first_env_tree(item) for key, item in value.items()}
        array = np.asarray(value)
        if array.shape[:1] != (1,):
            raise RuntimeError(f"Expected one-env batch, got {array.shape}")
        return array[0]

    def patch_tetris_render_state(
        official_state: Any,
        obs: dict[str, Any],
    ) -> Any:
        grid_padded = as_numpy(official_state.grid_padded).copy()
        grid = np.asarray(obs["grid"], dtype=grid_padded.dtype)
        grid_padded[: grid.shape[0], : grid.shape[1]] = grid
        return replace_fields(
            official_state,
            {
                "grid_padded": jnp.asarray(grid_padded),
                "grid_padded_old": jnp.asarray(grid_padded),
                "new_tetromino": jnp.asarray(
                    obs["tetromino"],
                    dtype=as_numpy(official_state.new_tetromino).dtype,
                ),
                "action_mask": jnp.asarray(
                    obs["action_mask"],
                    dtype=as_numpy(official_state.action_mask).dtype,
                ),
                "step_count": jnp.asarray(
                    obs["step_count"],
                    dtype=as_numpy(official_state.step_count).dtype,
                ),
            },
        )

    def patch_sokoban_render_state(
        official_state: Any,
        obs: dict[str, Any],
    ) -> Any:
        grid = np.asarray(obs["grid"])
        if grid.ndim != 3 or grid.shape[-1] < 2:
            return official_state
        return replace_fields(
            official_state,
            {
                "fixed_grid": jnp.asarray(
                    grid[:, :, 0],
                    dtype=as_numpy(official_state.fixed_grid).dtype,
                ),
                "variable_grid": jnp.asarray(
                    grid[:, :, 1],
                    dtype=as_numpy(official_state.variable_grid).dtype,
                ),
                "step_count": jnp.asarray(
                    obs["step_count"],
                    dtype=as_numpy(official_state.step_count).dtype,
                ),
            },
        )

    def render_state_from_envpool_obs(
        task_id: str,
        official_state: Any,
        envpool_obs: dict[str, Any],
    ) -> Any:
        patched = patch_struct_from_dict(official_state, envpool_obs)
        if task_id == "Tetris-v0":
            patched = patch_tetris_render_state(patched, envpool_obs)
        elif task_id == "Sokoban-v0":
            patched = patch_sokoban_render_state(patched, envpool_obs)
        return patched

    def official_render(
        official_env: Any,
        official_viewer: Any,
        official_state: Any,
    ) -> np.ndarray:
        frame = official_env.render(official_state)
        if frame is None:
            frame = official_viewer.render(official_state)
        if frame is None:
            raise RuntimeError(
                f"{type(official_env).__name__} returned no frame"
            )
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        return frame

    def graph_coloring_edges(adj_matrix: Any) -> str:
        adj = as_numpy(adj_matrix).astype(bool, copy=False)
        edges = [
            f"{row}-{col}"
            for row in range(adj.shape[0])
            for col in range(row + 1, adj.shape[1])
            if adj[row, col]
        ]
        return ",".join(edges)

    def minesweeper_locations(official_state: Any) -> str:
        if hasattr(official_state, "flat_mine_locations"):
            return csv_flat_int(official_state.flat_mine_locations)
        locations = as_numpy(official_state.mine_locations)
        if locations.ndim == 2 and locations.shape[1] == 2:
            locations = locations[:, 0] * 10 + locations[:, 1]
        return ",".join(map(str, locations.astype(np.int64).reshape(-1)))

    def cvrp_distance_matrix(official_state: Any) -> str:
        coordinates = as_numpy(official_state.coordinates).astype(np.float32)
        deltas = coordinates[:, None, :] - coordinates[None, :, :]
        distances = np.sqrt(np.sum(deltas * deltas, axis=-1, dtype=np.float32))
        return csv_flat(distances.astype(np.float32, copy=False))

    def envpool_kwargs_from_official_state(
        task_id: str,
        official_state: Any,
        official_observation: Any,
    ) -> dict[str, Any]:
        if task_id == "BinPack-v2":
            return {
                "bin_pack_item_x_len": csv_flat(
                    official_observation.items.x_len
                ),
                "bin_pack_item_y_len": csv_flat(
                    official_observation.items.y_len
                ),
                "bin_pack_item_z_len": csv_flat(
                    official_observation.items.z_len
                ),
                "bin_pack_items_mask": csv_flat_int(
                    official_observation.items_mask
                ),
                "bin_pack_render_container": csv_flat(
                    np.asarray(
                        [
                            as_numpy(official_state.container.x1),
                            as_numpy(official_state.container.x2),
                            as_numpy(official_state.container.y1),
                            as_numpy(official_state.container.y2),
                            as_numpy(official_state.container.z1),
                            as_numpy(official_state.container.z2),
                        ],
                        dtype=np.float32,
                    )
                ),
                "bin_pack_render_item_x_len": csv_flat(
                    official_state.items.x_len
                ),
                "bin_pack_render_item_y_len": csv_flat(
                    official_state.items.y_len
                ),
                "bin_pack_render_item_z_len": csv_flat(
                    official_state.items.z_len
                ),
            }
        if task_id == "CVRP-v1":
            return {
                "cvrp_coordinates": csv_flat(official_state.coordinates),
                "cvrp_demands": csv_flat(official_observation.demands),
                "cvrp_distance_matrix": cvrp_distance_matrix(official_state),
            }
        if task_id == "Cleaner-v0":
            return {
                "cleaner_grid": csv_flat_int(official_state.grid),
                "cleaner_agent_locations": csv_flat_int(
                    official_state.agents_locations
                ),
            }
        if task_id == "Connector-v2":
            return {"connector_grid": csv_flat_int(official_state.grid)}
        if task_id == "FlatPack-v0":
            return {
                "flat_pack_blocks": csv_flat_int(official_observation.blocks),
                "flat_pack_action_mask": csv_flat_int(
                    official_observation.action_mask
                ),
            }
        if task_id == "Game2048-v1":
            return {
                "game2048_initial_board": csv_flat_int(official_state.board)
            }
        if task_id == "GraphColoring-v1":
            return {
                "graph_coloring_edges": graph_coloring_edges(
                    official_state.adj_matrix
                )
            }
        if task_id == "JobShop-v0":
            return {
                "job_shop_ops_machine_ids": csv_flat_int(
                    official_observation.ops_machine_ids
                ),
                "job_shop_ops_durations": csv_flat_int(
                    official_observation.ops_durations
                ),
                "job_shop_ops_mask": csv_flat_int(
                    official_observation.ops_mask
                ),
                "job_shop_machines_job_ids": csv_flat_int(
                    official_observation.machines_job_ids
                ),
                "job_shop_machines_remaining_times": csv_flat_int(
                    official_observation.machines_remaining_times
                ),
                "job_shop_action_mask": csv_flat_int(
                    official_observation.action_mask
                ),
            }
        if task_id == "Knapsack-v1":
            return {
                "knapsack_weights": csv_flat(official_observation.weights),
                "knapsack_values": csv_flat(official_observation.values),
            }
        if task_id == "Maze-v0":
            return {
                "maze_walls": csv_flat_int(official_state.walls),
                "maze_agent_position": position_csv(
                    official_state.agent_position
                ),
                "maze_target_position": position_csv(
                    official_state.target_position
                ),
            }
        if task_id == "Minesweeper-v0":
            return {
                "minesweeper_mine_locations": minesweeper_locations(
                    official_state
                )
            }
        if task_id == "LevelBasedForaging-v0":
            agents = np.column_stack([
                as_numpy(official_state.agents.position),
                as_numpy(official_state.agents.level),
            ])
            food = np.column_stack([
                as_numpy(official_state.food_items.position),
                as_numpy(official_state.food_items.level),
            ])
            return {
                "lbf_agents": csv_flat_int(agents),
                "lbf_food": csv_flat_int(food),
            }
        if task_id == "MMST-v0":
            return {
                "mmst_node_types": csv_flat_int(
                    official_observation.node_types
                ),
                "mmst_adj_matrix": csv_flat_int(
                    official_observation.adj_matrix
                ),
                "mmst_positions": csv_flat_int(official_observation.positions),
                "mmst_action_mask": csv_flat_int(
                    official_observation.action_mask
                ),
                "mmst_render_nodes_to_connect": csv_flat_int(
                    official_state.nodes_to_connect
                ),
            }
        if task_id == "MultiCVRP-v0":
            return {
                "multi_cvrp_node_coordinates": csv_flat(
                    official_observation.nodes.coordinates
                ),
                "multi_cvrp_node_demands": csv_flat_int(
                    official_observation.nodes.demands
                ),
                "multi_cvrp_windows_start": csv_flat(
                    official_observation.windows.start
                ),
                "multi_cvrp_windows_end": csv_flat(
                    official_observation.windows.end
                ),
                "multi_cvrp_coeffs_early": csv_flat(
                    official_observation.coeffs.early
                ),
                "multi_cvrp_coeffs_late": csv_flat(
                    official_observation.coeffs.late
                ),
                "multi_cvrp_vehicle_local_times": csv_flat(
                    official_observation.vehicles.local_times
                ),
                "multi_cvrp_vehicle_capacities": csv_flat_int(
                    official_observation.vehicles.capacities
                ),
                "multi_cvrp_action_mask": csv_flat_int(
                    official_observation.action_mask
                ),
            }
        if task_id == "PacMan-v1":
            return {
                "pacman_grid": csv_flat_int(official_state.grid),
                "pacman_player_location": position_csv(
                    official_state.player_locations
                ),
                "pacman_pellet_locations": csv_flat_int(
                    official_state.pellet_locations
                ),
                "pacman_ghost_locations": csv_flat_int(
                    official_observation.ghost_locations
                ),
                "pacman_power_up_locations": csv_flat_int(
                    official_observation.power_up_locations
                ),
                "pacman_action_mask": csv_flat_int(
                    official_observation.action_mask
                ),
                "pacman_frightened_state_time": int(
                    as_numpy(official_observation.frightened_state_time)
                ),
                "pacman_initial_score": int(
                    as_numpy(official_observation.score)
                ),
            }
        if task_id == "RobotWarehouse-v0":
            return {
                "robot_warehouse_agents_view": csv_flat_int(
                    official_observation.agents_view
                ),
                "robot_warehouse_action_mask": csv_flat_int(
                    official_observation.action_mask
                ),
                "robot_warehouse_render_grid": csv_flat_int(
                    official_state.grid
                ),
                "robot_warehouse_render_agent_x": csv_flat_int(
                    official_state.agents.position.x
                ),
                "robot_warehouse_render_agent_y": csv_flat_int(
                    official_state.agents.position.y
                ),
                "robot_warehouse_render_agent_direction": csv_flat_int(
                    official_state.agents.direction
                ),
                "robot_warehouse_render_agent_carrying": csv_flat_int(
                    official_state.agents.is_carrying
                ),
                "robot_warehouse_render_shelf_x": csv_flat_int(
                    official_state.shelves.position.x
                ),
                "robot_warehouse_render_shelf_y": csv_flat_int(
                    official_state.shelves.position.y
                ),
                "robot_warehouse_render_shelf_requested": csv_flat_int(
                    official_state.shelves.is_requested
                ),
            }
        if task_id.startswith("RubiksCube"):
            return {
                "rubiks_cube_initial_cube": csv_flat_int(official_state.cube)
            }
        if task_id == "SearchAndRescue-v0":
            return {
                "search_and_rescue_searcher_views": csv_flat(
                    official_observation.searcher_views
                ),
                "search_and_rescue_positions": csv_flat(
                    official_observation.positions
                ),
                "search_and_rescue_headings": csv_flat(
                    official_state.searchers.heading
                ),
                "search_and_rescue_speeds": csv_flat(
                    official_state.searchers.speed
                ),
                "search_and_rescue_targets_remaining": float(
                    as_numpy(official_observation.targets_remaining)
                ),
                "search_and_rescue_target_positions": csv_flat(
                    official_state.targets.pos
                ),
                "search_and_rescue_target_velocities": csv_flat(
                    official_state.targets.vel
                ),
                "search_and_rescue_target_found": csv_flat_int(
                    official_state.targets.found
                ),
            }
        if task_id == "SlidingTilePuzzle-v0":
            return {
                "sliding_tile_initial_puzzle": csv_flat_int(
                    official_state.puzzle
                )
            }
        if task_id == "Snake-v1":
            return {
                "snake_head_position": position_csv(
                    official_state.head_position
                ),
                "snake_fruit_position": position_csv(
                    official_state.fruit_position
                ),
            }
        if task_id.startswith("Sudoku"):
            return {"sudoku_initial_board": csv_flat_int(official_state.board)}
        if task_id == "TSP-v1":
            return {"tsp_coordinates": csv_flat(official_state.coordinates)}
        if task_id == "Tetris-v0":
            return {
                "tetris_initial_grid": csv_flat_int(
                    as_numpy(official_state.grid_padded)[:10, :10]
                ),
                "tetris_tetromino": csv_flat_int(official_state.new_tetromino),
                "tetris_action_mask": csv_flat_int(
                    official_observation.action_mask
                ),
            }
        return {}

    def render_pair(
        task_id: str,
        cfg: RenderCompareConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        official_env = make_official_env(task_id)
        official_viewer = find_viewer(official_env)
        official_viewer._display = official_viewer._display_rgb_array
        official_viewer.figure_size = (
            cfg.source_width / 100.0,
            cfg.source_height / 100.0,
        )
        env = None
        try:
            official_state, official_timestep = official_env.reset(
                jax.random.PRNGKey(cfg.seed)
            )
            official_frame = official_render(
                official_env,
                official_viewer,
                official_state,
            )
            envpool_kwargs: dict[str, Any] = {
                "num_envs": 1,
                "seed": cfg.seed,
                "render_mode": "rgb_array",
                "render_width": cfg.source_width,
                "render_height": cfg.source_height,
            }
            envpool_kwargs.update(
                envpool_kwargs_from_official_state(
                    task_id, official_state, official_timestep.observation
                )
            )
            if task_id == "Sokoban-v0":
                envpool_kwargs.update(
                    base_path="/tmp/envpool-missing-boxoban",
                    sokoban_level_index=0,
                )
            env = make_gymnasium(
                task_id,
                **envpool_kwargs,
            )
            envpool_obs, _ = env.reset()
            del envpool_obs
            envpool_frame = env.render()[0]
            return envpool_frame, official_frame
        finally:
            if env is not None:
                env.close()
            official_viewer.close()

    return RenderFamily(
        items=tuple(
            RenderItem(key=task_id, label=task_id)
            for task_id in jumanji_registration.jumanji_env_ids
        ),
        default_output=Path(
            "docs/_static/render_samples/jumanji_official_compare.png"
        ),
        render_pair=render_pair,
        left_title="EnvPool",
        right_title="Official",
        compare_frames=True,
    )


def _candidate_runfile_roots() -> Iterator[Path]:
    seen: set[Path] = set()
    raw_roots = (
        os.environ.get("RUNFILES_DIR"),
        os.environ.get("TEST_SRCDIR"),
        os.environ.get("PYTHON_RUNFILES"),
    )
    for raw_root in raw_roots:
        if raw_root:
            root = Path(raw_root)
            if root not in seen:
                seen.add(root)
                yield root

    for current in (Path(__file__), Path(__file__).resolve()):
        for parent in current.parents:
            if parent.name.endswith(".runfiles") and parent not in seen:
                seen.add(parent)
                yield parent


@lru_cache
def _runfiles_manifest_entries() -> tuple[tuple[str, Path], ...]:
    manifest = os.environ.get("RUNFILES_MANIFEST_FILE")
    if not manifest:
        return ()
    entries = []
    with Path(manifest).open(encoding="utf-8") as f:
        for line in f:
            logical, sep, physical = line.rstrip("\n").partition(" ")
            entries.append((logical, Path(physical if sep else logical)))
    return tuple(entries)


def _find_manifest_runfile(path: str) -> Path | None:
    path = path.replace("\\", "/")
    candidates = [path]
    workspace = os.environ.get("TEST_WORKSPACE")
    if workspace:
        candidates.append(f"{workspace}/{path}")
    for logical, physical in _runfiles_manifest_entries():
        for candidate in candidates:
            if logical == candidate:
                return physical
            prefix = f"{candidate.rstrip('/')}/"
            if logical.startswith(prefix):
                result = physical
                for _ in logical[len(prefix) :].split("/"):
                    result = result.parent
                return result
    return None


def _find_runfile(path: str) -> Path:
    manifest_path = _find_manifest_runfile(path)
    if manifest_path is not None:
        return manifest_path
    for root in _candidate_runfile_roots():
        direct = root / path
        if direct.exists():
            return direct
        if root.exists():
            for workspace_dir in root.iterdir():
                candidate = workspace_dir / path
                if candidate.exists():
                    return candidate
    raise FileNotFoundError(f"Could not find Bazel runfile: {path}")


def _make_mujoco_playground_family() -> RenderFamily:
    """Build the MuJoCo Playground official render comparison adapter."""
    from etils import epath
    from jax import config as jax_config

    jax_config.update("jax_enable_x64", True)

    import mujoco
    import mujoco_playground._src.mjx_env as oracle_mjx_env
    from mujoco_playground._src.locomotion.apollo import (
        joystick as oracle_apollo,
    )
    from mujoco_playground._src.locomotion.barkour import (
        joystick as oracle_barkour,
    )
    from mujoco_playground._src.locomotion.berkeley_humanoid import (
        joystick as oracle_berkeley,
    )
    from mujoco_playground._src.locomotion.g1 import joystick as oracle_g1
    from mujoco_playground._src.locomotion.go1 import getup as oracle_getup
    from mujoco_playground._src.locomotion.go1 import (
        handstand as oracle_handstand,
    )
    from mujoco_playground._src.locomotion.go1 import joystick as oracle_go1
    from mujoco_playground._src.locomotion.h1 import (
        inplace_gait_tracking as oracle_h1_inplace,
    )
    from mujoco_playground._src.locomotion.h1 import (
        joystick_gait_tracking as oracle_h1_joystick,
    )
    from mujoco_playground._src.locomotion.op3 import joystick as oracle_op3
    from mujoco_playground._src.locomotion.spot import (
        getup as oracle_spot_getup,
    )
    from mujoco_playground._src.locomotion.spot import (
        joystick as oracle_spot_joystick,
    )
    from mujoco_playground._src.locomotion.spot import (
        joystick_gait_tracking as oracle_spot_gait,
    )
    from mujoco_playground._src.locomotion.t1 import joystick as oracle_t1
    from mujoco_playground._src.manipulation.aero_hand import (
        rotate_z as oracle_aero_rotate,
    )
    from mujoco_playground._src.manipulation.aloha import (
        handover as oracle_aloha_handover,
    )
    from mujoco_playground._src.manipulation.aloha import (
        single_peg_insertion as oracle_aloha_peg,
    )
    from mujoco_playground._src.manipulation.franka_emika_panda import (
        open_cabinet as oracle_panda_open_cabinet,
    )
    from mujoco_playground._src.manipulation.franka_emika_panda import (
        pick as oracle_panda_pick,
    )
    from mujoco_playground._src.manipulation.franka_emika_panda import (
        pick_cartesian as oracle_panda_pick_cartesian,
    )
    from mujoco_playground._src.manipulation.franka_emika_panda_robotiq import (
        push_cube as oracle_panda_robotiq,
    )
    from mujoco_playground._src.manipulation.leap_hand import (
        reorient as oracle_leap_reorient,
    )
    from mujoco_playground._src.manipulation.leap_hand import (
        rotate_z as oracle_leap_rotate,
    )

    import envpool.mujoco.playground.registration as playground_registration
    from envpool.registration import make_gymnasium

    oracle_mjx_env.MENAGERIE_PATH = epath.Path(
        _find_runfile("mujoco_menagerie_playground")
    )

    def make_oracle(task_id: str) -> Any:
        if task_id == "AlohaHandOver-v1":
            config = oracle_aloha_handover.default_config()
            config.impl = "jax"
            return oracle_aloha_handover.HandOver(config=config)
        if task_id == "AlohaSinglePegInsertion-v1":
            config = oracle_aloha_peg.default_config()
            config.impl = "jax"
            return oracle_aloha_peg.SinglePegInsertion(config=config)
        if task_id == "ApolloJoystickFlatTerrain-v1":
            config = oracle_apollo.default_config()
            config.impl = "jax"
            config.noise_config.level = 0.0
            config.push_config.enable = False
            config.command_config.min = [0.0, 0.0, 0.0]
            config.command_config.max = [0.0, 0.0, 0.0]
            config.command_config.zero_prob = [1.0, 1.0, 1.0]
            return oracle_apollo.Joystick(task="flat_terrain", config=config)
        if task_id == "BarkourJoystick-v1":
            config = oracle_barkour.default_config()
            config.impl = "jax"
            config.obs_noise = -1.0
            config.lin_vel_x = [0.0, 0.0]
            config.lin_vel_y = [0.0, 0.0]
            config.ang_vel_yaw = [0.0, 0.0]
            config.kick_wait_steps = [100000, 100001]
            return oracle_barkour.Joystick(config=config)
        if task_id == "Op3Joystick-v1":
            config = oracle_op3.default_config()
            config.impl = "jax"
            config.obs_noise = -1.0
            config.lin_vel_x = [0.0, 0.0]
            config.lin_vel_y = [0.0, 0.0]
            config.ang_vel_yaw = [0.0, 0.0]
            return oracle_op3.Joystick(config=config)
        if task_id.startswith("BerkeleyHumanoid"):
            config = oracle_berkeley.default_config()
            config.impl = "jax"
            config.noise_config.level = 0.0
            config.push_config.enable = False
            config.lin_vel_x = [0.0, 0.0]
            config.lin_vel_y = [0.0, 0.0]
            config.ang_vel_yaw = [0.0, 0.0]
            task = (
                "rough_terrain" if "RoughTerrain" in task_id else "flat_terrain"
            )
            return oracle_berkeley.Joystick(task=task, config=config)
        if task_id in (
            "G1JoystickFlatTerrain-v1",
            "G1JoystickRoughTerrain-v1",
        ):
            config = oracle_g1.default_config()
            config.impl = "jax"
            config.noise_config.level = 0.0
            config.push_config.enable = False
            config.lin_vel_x = [0.0, 0.0]
            config.lin_vel_y = [0.0, 0.0]
            config.ang_vel_yaw = [0.0, 0.0]
            task = (
                "rough_terrain" if "RoughTerrain" in task_id else "flat_terrain"
            )
            return oracle_g1.Joystick(task=task, config=config)
        if task_id in (
            "T1JoystickFlatTerrain-v1",
            "T1JoystickRoughTerrain-v1",
        ):
            config = oracle_t1.default_config()
            config.impl = "jax"
            config.noise_config.level = 0.0
            config.push_config.enable = False
            config.lin_vel_x = [0.0, 0.0]
            config.lin_vel_y = [0.0, 0.0]
            config.ang_vel_yaw = [0.0, 0.0]
            task = (
                "rough_terrain" if "RoughTerrain" in task_id else "flat_terrain"
            )
            return oracle_t1.Joystick(task=task, config=config)
        if task_id == "H1InplaceGaitTracking-v1":
            config = oracle_h1_inplace.default_config()
            config.impl = "jax"
            config.obs_noise.level = 0.0
            return oracle_h1_inplace.InplaceGaitTracking(config=config)
        if task_id == "H1JoystickGaitTracking-v1":
            config = oracle_h1_joystick.default_config()
            config.impl = "jax"
            config.obs_noise.level = 0.0
            config.lin_vel_x = [0.0, 0.0]
            config.lin_vel_y = [0.0, 0.0]
            config.ang_vel_yaw = [0.0, 0.0]
            return oracle_h1_joystick.JoystickGaitTracking(config=config)
        if task_id == "SpotFlatTerrainJoystick-v1":
            config = oracle_spot_joystick.default_config()
            config.impl = "jax"
            config.obs_noise.scales.joint_pos = 0.0
            config.obs_noise.scales.gyro = 0.0
            config.obs_noise.scales.gravity = 0.0
            config.obs_noise.scales.feet_pos = [0.0, 0.0, 0.0]
            config.pert_config.enable = False
            config.command_config.lin_vel_x = [0.0, 0.0]
            config.command_config.lin_vel_y = [0.0, 0.0]
            config.command_config.ang_vel_yaw = [0.0, 0.0]
            return oracle_spot_joystick.Joystick(
                task="flat_terrain", config=config
            )
        if task_id == "SpotGetup-v1":
            config = oracle_spot_getup.default_config()
            config.impl = "jax"
            config.obs_noise.level = 0.0
            return oracle_spot_getup.Getup(config=config)
        if task_id == "SpotJoystickGaitTracking-v1":
            config = oracle_spot_gait.default_config()
            config.impl = "jax"
            config.obs_noise.scales.joint_pos = 0.0
            config.obs_noise.scales.gyro = 0.0
            config.obs_noise.scales.gravity = 0.0
            config.obs_noise.scales.feet_pos = [0.0, 0.0, 0.0]
            config.command_config.lin_vel_x = [0.0, 0.0]
            config.command_config.lin_vel_y = [0.0, 0.0]
            config.command_config.ang_vel_yaw = [0.0, 0.0]
            return oracle_spot_gait.JoystickGaitTracking(config=config)
        if task_id == "PandaPickCube-v1":
            config = oracle_panda_pick.default_config()
            config.impl = "jax"
            return oracle_panda_pick.PandaPickCube(config=config)
        if task_id == "PandaPickCubeCartesian-v1":
            config = oracle_panda_pick_cartesian.default_config()
            config.impl = "jax"
            return oracle_panda_pick_cartesian.PandaPickCubeCartesian(
                config=config
            )
        if task_id == "PandaPickCubeOrientation-v1":
            config = oracle_panda_pick.default_config()
            config.impl = "jax"
            return oracle_panda_pick.PandaPickCubeOrientation(config=config)
        if task_id == "PandaOpenCabinet-v1":
            config = oracle_panda_open_cabinet.default_config()
            config.impl = "jax"
            return oracle_panda_open_cabinet.PandaOpenCabinet(config=config)
        if task_id == "PandaRobotiqPushCube-v1":
            config = oracle_panda_robotiq.default_config()
            config.impl = "jax"
            config.noise_config.action_min_delay = 0
            config.noise_config.action_max_delay = 1
            config.noise_config.obs_min_delay = 0
            config.noise_config.obs_max_delay = 1
            config.noise_config.noise_scales.obj_pos = 0.0
            config.noise_config.noise_scales.obj_angle = 0.0
            config.noise_config.noise_scales.robot_qpos = 0.0
            config.noise_config.noise_scales.robot_qvel = 0.0
            config.noise_config.noise_scales.eef_pos = 0.0
            config.noise_config.noise_scales.eef_angle = 0.0
            return oracle_panda_robotiq.PandaRobotiqPushCube(config=config)
        if task_id == "LeapCubeRotateZAxis-v1":
            config = oracle_leap_rotate.default_config()
            config.impl = "jax"
            config.noise_config.level = 0.0
            return oracle_leap_rotate.CubeRotateZAxis(config=config)
        if task_id == "LeapCubeReorient-v1":
            config = oracle_leap_reorient.default_config()
            config.impl = "jax"
            config.obs_noise.level = 0.0
            config.pert_config.enable = False
            return oracle_leap_reorient.CubeReorient(config=config)
        if task_id == "AeroCubeRotateZAxis-v1":
            config = oracle_aero_rotate.default_config()
            config.noise_config.level = 0.0
            return oracle_aero_rotate.CubeRotateZAxis(config=config)
        if task_id == "Go1Getup-v1":
            config = oracle_getup.default_config()
            config.impl = "jax"
            config.noise_config.level = 0.0
            return oracle_getup.Getup(config=config)
        if task_id == "Go1Handstand-v1":
            config = oracle_handstand.default_config()
            config.impl = "jax"
            config.noise_config.level = 0.0
            return oracle_handstand.Handstand(config=config)
        if task_id == "Go1Footstand-v1":
            config = oracle_handstand.default_config()
            config.impl = "jax"
            config.noise_config.level = 0.0
            return oracle_handstand.Footstand(config=config)
        config = oracle_go1.default_config()
        config.impl = "jax"
        config.noise_config.level = 0.0
        config.command_config.a = [0.0, 0.0, 0.0]
        config.command_config.b = [0.0, 0.0, 0.0]
        task = "rough_terrain" if "RoughTerrain" in task_id else "flat_terrain"
        return oracle_go1.Joystick(task=task, config=config)

    def make_render_data(model_or_oracle: Any, info: dict[str, Any]) -> Any:
        model = (
            model_or_oracle
            if isinstance(model_or_oracle, mujoco.MjModel)
            else model_or_oracle.mj_model
        )
        data = mujoco.MjData(model)
        data.qpos[:] = info["qpos"][0, : model.nq]
        data.qvel[:] = info["qvel"][0, : model.nv]
        if "ctrl" in info:
            data.ctrl[:] = info["ctrl"][0, : model.nu]
        if "mocap_pos" in info:
            data.mocap_pos[:] = info["mocap_pos"][
                0, : model.nmocap * 3
            ].reshape(model.nmocap, 3)
        if "mocap_quat" in info:
            data.mocap_quat[:] = info["mocap_quat"][
                0, : model.nmocap * 4
            ].reshape(model.nmocap, 4)
        if "qacc_warmstart" in info:
            data.qacc_warmstart[:] = info["qacc_warmstart"][0, : model.nv]
        mujoco.mj_forward(model, data)
        if "qacc" in info:
            data.qacc[:] = info["qacc"][0, : model.nv]
        if "qacc_warmstart" in info:
            data.qacc_warmstart[:] = info["qacc_warmstart"][0, : model.nv]
        if "sensordata" in info:
            data.sensordata[:] = info["sensordata"][0, : data.sensordata.size]
        return data

    op3_render_model: mujoco.MjModel | None = None

    def official_render_model(task_id: str, oracle: Any) -> Any:
        nonlocal op3_render_model
        if task_id != "Op3Joystick-v1":
            return oracle
        if op3_render_model is None:
            xml_path = _find_runfile(
                "envpool/mujoco/playground/assets/mujoco_playground/_src/"
                "locomotion/op3/xmls/scene_mjx_feetonly.xml"
            )
            op3_render_model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        return op3_render_model

    def render_official_frame(
        model_or_oracle: Any,
        data: Any,
        render_kwargs: dict[str, Any],
    ) -> np.ndarray:
        model = (
            model_or_oracle
            if isinstance(model_or_oracle, mujoco.MjModel)
            else model_or_oracle.mj_model
        )
        renderer = mujoco.Renderer(
            model,
            height=render_kwargs["height"],
            width=render_kwargs["width"],
        )
        try:
            renderer.update_scene(data, camera=render_kwargs.get("camera", -1))
            return np.asarray(renderer.render())
        finally:
            renderer.close()

    def zero_command_kwargs(
        *,
        noise_key: str | None = "noise_level",
        noise_value: float = 0.0,
        disable_key: str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "lin_vel_x_min": 0.0,
            "lin_vel_x_max": 0.0,
            "lin_vel_y_min": 0.0,
            "lin_vel_y_max": 0.0,
            "ang_vel_yaw_min": 0.0,
            "ang_vel_yaw_max": 0.0,
        }
        if noise_key is not None:
            kwargs[noise_key] = noise_value
        if disable_key is not None:
            kwargs[disable_key] = 0.0
        return kwargs

    def envpool_kwargs(task_id: str) -> dict[str, Any]:
        if task_id.startswith("Apollo"):
            return {
                "noise_level": 0.0,
                "push_enable": 0.0,
                "command_min0": 0.0,
                "command_min1": 0.0,
                "command_min2": 0.0,
                "command_max0": 0.0,
                "command_max1": 0.0,
                "command_max2": 0.0,
                "command_zero_prob0": 1.0,
                "command_zero_prob1": 1.0,
                "command_zero_prob2": 1.0,
            }
        if task_id.startswith("Barkour"):
            return {
                **zero_command_kwargs(noise_key="obs_noise", noise_value=-1.0),
                "kick_wait_steps_min": 100000,
                "kick_wait_steps_max": 100001,
            }
        if task_id.startswith("Op3"):
            return zero_command_kwargs(noise_key="obs_noise", noise_value=-1.0)
        if task_id.startswith(("BerkeleyHumanoid", "G1", "T1")):
            return zero_command_kwargs(disable_key="push_enable")
        if task_id.startswith("H1"):
            return zero_command_kwargs(noise_key="obs_noise_level")
        if task_id == "SpotFlatTerrainJoystick-v1":
            return zero_command_kwargs(disable_key="pert_enable")
        if task_id == "SpotGetup-v1":
            return {"noise_level": 0.0}
        if task_id == "SpotJoystickGaitTracking-v1":
            return zero_command_kwargs()
        if task_id == "PandaPickCubeCartesian-v1":
            return {"guide_sample_prob": 0.0}
        if task_id == "PandaRobotiqPushCube-v1":
            return {
                "action_min_delay": 0,
                "action_max_delay": 1,
                "obs_min_delay": 0,
                "obs_max_delay": 1,
                "noise_obj_pos": 0.0,
                "noise_obj_angle": 0.0,
                "noise_robot_qpos": 0.0,
                "noise_robot_qvel": 0.0,
                "noise_eef_pos": 0.0,
                "noise_eef_angle": 0.0,
            }
        if task_id in (
            "LeapCubeRotateZAxis-v1",
            "LeapCubeReorient-v1",
            "AeroCubeRotateZAxis-v1",
        ):
            return {"noise_level": 0.0, "pert_enable": 0.0}
        return {}

    def render_pair(
        task_id: str,
        cfg: RenderCompareConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        oracle = make_oracle(task_id)
        env = make_gymnasium(
            task_id,
            num_envs=1,
            seed=cfg.seed,
            render_mode="rgb_array",
            render_width=cfg.source_width,
            render_height=cfg.source_height,
            **envpool_kwargs(task_id),
        )
        try:
            _, info = env.reset()
            render_model = official_render_model(task_id, oracle)
            render_data = make_render_data(render_model, info)
            render_kwargs: dict[str, Any] = {
                "height": cfg.source_height,
                "width": cfg.source_width,
            }
            if task_id.startswith((
                "Apollo",
                "Barkour",
                "Berkeley",
                "G1",
                "Go1",
                "H1",
                "Op3",
                "Spot",
                "T1",
            )):
                render_kwargs["camera"] = "track"
            official_frame = render_official_frame(
                render_model, render_data, render_kwargs
            )
            envpool_frame = env.render(env_ids=[0])
            if envpool_frame is None:
                raise RuntimeError(f"{task_id} returned no EnvPool frame")
            return np.asarray(envpool_frame[0]), official_frame
        finally:
            env.close()

    return RenderFamily(
        items=tuple(
            RenderItem(
                key=f"{task_name}-v1",
                label=task_name
                .replace("BerkeleyHumanoid", "Berkeley ")
                .replace("G1", "G1 ")
                .replace("Go1", "Go1 ")
                .replace("Op3", "OP3"),
            )
            for task_name in playground_registration.PLAYGROUND_ENVS
        ),
        default_output=Path(
            "docs/_static/render_samples/mujoco_playground_official_compare.png"
        ),
        render_pair=render_pair,
    )


_FAMILY_BUILDERS: dict[str, Callable[[], RenderFamily]] = {
    "jumanji": _make_jumanji_family,
    "metaworld": _make_metaworld_family,
    "mujoco_playground": _make_mujoco_playground_family,
}


def _make_display_image(
    frame: np.ndarray, cfg: RenderCompareConfig
) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    if image.mode != "RGB":
        image = image.convert("RGB")
    if cfg.flip_vertical:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    size = (cfg.tile_width, cfg.tile_height)
    if image.size == size:
        return image
    return image.resize(size, Image.Resampling.LANCZOS)


def _render_mismatch_message(
    label: str,
    diff: np.ndarray,
    mismatch_ratio: float,
    max_ignored_abs_diff: int,
) -> str:
    return (
        f"{label} render mismatch: "
        f"mean_abs_diff={float(diff.mean()):.6f}, "
        f"max_abs_diff={int(diff.max())}, "
        f"mismatch_ratio={mismatch_ratio:.6%}, "
        f"ignored_abs_diff={max_ignored_abs_diff}"
    )


def _assert_frames_match(
    label: str,
    envpool_frame: np.ndarray,
    official_frame: np.ndarray,
    cfg: RenderCompareConfig,
) -> None:
    if envpool_frame.shape != official_frame.shape:
        raise RuntimeError(
            f"{label} render shape mismatch: "
            f"{envpool_frame.shape} != {official_frame.shape}"
        )
    if np.array_equal(envpool_frame, official_frame):
        return

    diff = np.abs(
        envpool_frame.astype(np.int16) - official_frame.astype(np.int16)
    )
    mismatch_ratio = float(np.count_nonzero(diff)) / float(diff.size)
    if cfg.require_bitwise:
        raise RuntimeError(
            _render_mismatch_message(label, diff, mismatch_ratio, 0)
        )

    mean_abs_diff = float(diff.mean())
    significant = diff > cfg.max_ignored_abs_diff
    mismatch_ratio = float(np.count_nonzero(significant)) / float(diff.size)
    if (
        mean_abs_diff > cfg.max_mean_abs_diff
        or mismatch_ratio > cfg.max_mismatch_ratio
    ):
        raise RuntimeError(
            _render_mismatch_message(
                label, diff, mismatch_ratio, cfg.max_ignored_abs_diff
            )
        )


def _draw_panel(
    canvas: Image.Image,
    family: RenderFamily,
    label: str,
    envpool_image: Image.Image,
    official_image: Image.Image | None,
    index: int,
    cfg: RenderCompareConfig,
    font: Any,
) -> None:
    draw = ImageDraw.Draw(canvas)
    has_official = official_image is not None
    cell_width = cfg.tile_width * (2 if has_official else 1)
    if has_official:
        cell_width += _PAIR_GAP
    cell_height = _HEADER_HEIGHT + cfg.tile_height
    col = index % cfg.columns
    row = index // cfg.columns
    left = _MARGIN + col * (cell_width + _CELL_GAP)
    top = _MARGIN + row * (cell_height + _CELL_GAP)
    frame_top = top + _HEADER_HEIGHT

    draw.text((left, top), label, fill=(30, 30, 30), font=font)
    draw.text((left, top + 14), family.left_title, fill=(80, 80, 80), font=font)
    canvas.paste(envpool_image, (left, frame_top))
    draw.rectangle(
        [
            left,
            frame_top,
            left + cfg.tile_width - 1,
            frame_top + cfg.tile_height - 1,
        ],
        outline=(205, 205, 205),
    )
    if has_official:
        official_left = left + cfg.tile_width + _PAIR_GAP
        assert family.right_title is not None
        draw.text(
            (official_left, top + 14),
            family.right_title,
            fill=(80, 80, 80),
            font=font,
        )
        canvas.paste(official_image, (official_left, frame_top))
        draw.rectangle(
            [
                official_left,
                frame_top,
                official_left + cfg.tile_width - 1,
                frame_top + cfg.tile_height - 1,
            ],
            outline=(205, 205, 205),
        )


def generate(
    output: Path, family: RenderFamily, cfg: RenderCompareConfig
) -> None:
    """Generate and write one docs render image."""
    rows = math.ceil(len(family.items) / cfg.columns)
    has_official = family.right_title is not None
    cell_width = cfg.tile_width * (2 if has_official else 1)
    if has_official:
        cell_width += _PAIR_GAP
    cell_height = _HEADER_HEIGHT + cfg.tile_height
    width = _MARGIN * 2 + cfg.columns * cell_width
    width += (cfg.columns - 1) * _CELL_GAP
    height = _MARGIN * 2 + rows * cell_height + (rows - 1) * _CELL_GAP
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    font = ImageFont.load_default()

    for index, item in enumerate(family.items):
        envpool_frame, official_frame = family.render_pair(item.key, cfg)
        if official_frame is not None and family.compare_frames:
            _assert_frames_match(item.label, envpool_frame, official_frame, cfg)
        envpool_image = _make_display_image(envpool_frame, cfg)
        official_image = (
            None
            if official_frame is None
            else _make_display_image(official_frame, cfg)
        )
        _draw_panel(
            canvas,
            family,
            item.label,
            envpool_image,
            official_image,
            index,
            cfg,
            font,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        choices=sorted(_FAMILY_BUILDERS),
        default="metaworld",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tile-width", type=int, default=96)
    parser.add_argument("--tile-height", type=int, default=72)
    parser.add_argument(
        "--source-width",
        type=int,
        default=480,
        help="Source render width before docs thumbnail downsampling.",
    )
    parser.add_argument(
        "--source-height",
        type=int,
        default=480,
        help="Source render height before docs thumbnail downsampling.",
    )
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--camera-id",
        type=int,
        default=1,
        help="Renderer camera id for families that support camera selection.",
    )
    parser.add_argument("--max-mean-abs-diff", type=float, default=0.25)
    parser.add_argument("--max-mismatch-ratio", type=float, default=0.005)
    parser.add_argument(
        "--max-ignored-abs-diff",
        type=int,
        default=0,
        help=(
            "RGB channel deltas up to this value are ignored when computing "
            "mismatch ratio. Mean absolute diff still uses raw deltas."
        ),
    )
    parser.add_argument("--require-bitwise", action="store_true")
    parser.add_argument(
        "--flip-vertical",
        action="store_true",
        default=None,
        help="Flip both EnvPool and oracle frames vertically in the output.",
    )
    parser.add_argument(
        "--no-flip-vertical",
        action="store_false",
        dest="flip_vertical",
        help="Do not vertically flip output frames.",
    )
    return parser.parse_args()


def main() -> None:
    """Parse command-line arguments and generate the comparison image."""
    args = _parse_args()
    family = _FAMILY_BUILDERS[args.family]()
    cfg = RenderCompareConfig(
        family=args.family,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        source_width=args.source_width,
        source_height=args.source_height,
        columns=args.columns,
        seed=args.seed,
        camera_id=args.camera_id,
        max_mean_abs_diff=args.max_mean_abs_diff,
        max_mismatch_ratio=args.max_mismatch_ratio,
        max_ignored_abs_diff=args.max_ignored_abs_diff,
        require_bitwise=args.require_bitwise,
        flip_vertical=(
            family.default_flip_vertical
            if args.flip_vertical is None
            else args.flip_vertical
        ),
    )

    output = args.output or family.default_output
    if not output.is_absolute():
        output = Path(
            os.environ.get("BUILD_WORKSPACE_DIRECTORY", ".")
        ).joinpath(output)
    generate(output, family, cfg)


if __name__ == "__main__":
    main()
