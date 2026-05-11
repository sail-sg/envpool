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
"""Bitwise Jumanji rendering via vendored Jumanji Matplotlib viewers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def configure_matplotlib() -> None:
    """Configure Matplotlib before any vendored viewer imports pyplot."""
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


configure_matplotlib()

from envpool.jumanji._official_render import (  # noqa: E402
    logic,
    packing,
    routing,
    swarms,
)


def _asarray(value: Any, dtype: Any | None = None) -> Any:
    return np.asarray(value, dtype=dtype)


def _default_array_dtype(value: Any, dtype: Any | None) -> Any | None:
    if dtype is not None:
        return dtype
    array = np.asarray(value)
    if array.dtype == np.dtype("float64"):
        return np.float32
    if array.dtype == np.dtype("int64"):
        return np.int32
    return None


def _a(value: Any, dtype: Any | None = None) -> Any:
    return np.asarray(value, dtype=_default_array_dtype(value, dtype))


def _key() -> Any:
    return _a([0, 0], np.uint32)


def _get(tree: Mapping[str, Any], path: str) -> Any:
    value: Any = tree
    for key in path.split("."):
        value = value[key]
    return value


def _config_array(
    config: Mapping[str, Any],
    key: str,
    shape: tuple[int, ...],
    dtype: Any,
    default: Any,
) -> Any:
    value = config.get(key, "")
    if value == "" or value is None:
        return np.asarray(default, dtype=dtype).reshape(shape)
    if isinstance(value, str):
        sep = "," if "," in value else " "
        array = np.fromstring(value, dtype=dtype, sep=sep)
    else:
        array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.size != int(np.prod(shape)):
        return np.asarray(default, dtype=dtype).reshape(shape)
    return array.reshape(shape)


def _config_step_array(
    config: Mapping[str, Any],
    key: str,
    step: int,
    shape: tuple[int, ...],
    dtype: Any,
    steps: int = 3,
) -> Any | None:
    value = config.get(key, "")
    if value == "" or value is None or step <= 0 or step > steps:
        return None
    if isinstance(value, str):
        sep = "," if "," in value else " "
        array = np.fromstring(value, dtype=dtype, sep=sep)
    else:
        array = np.asarray(value, dtype=dtype).reshape(-1)
    full_shape = (steps, *shape)
    if array.size != int(np.prod(full_shape)):
        return None
    return array.reshape(full_shape)[step - 1]


def _csv_bool_array(
    config: Mapping[str, Any],
    key: str,
    shape: tuple[int, ...],
    default: Any,
) -> NDArray[np.bool_]:
    return _config_array(config, key, shape, np.int32, default).astype(bool)


def _viewer_frame(
    viewer: Any,
    state: Any,
    width: int,
    height: int,
    passes: int = 1,
) -> NDArray[np.uint8]:
    viewer._display = viewer._display_rgb_array
    viewer.figure_size = (width / 100.0, height / 100.0)
    frame = None
    for _ in range(passes):
        frame = viewer.render(state)
    if frame is None:
        raise RuntimeError(f"{type(viewer).__name__} returned no frame")
    array = np.asarray(frame, dtype=np.uint8)
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    return np.array(array, copy=True)


def _cached_viewer(
    aux: Mapping[str, Any],
    key: str,
    factory: Any,
) -> Any:
    if not isinstance(aux, dict):
        return factory()
    cache = aux.setdefault("_viewer_cache", {})
    viewer = cache.get(key)
    if viewer is None:
        viewer = factory()
        viewer._figure_name = f"EnvPoolJumanji:{id(aux)}:{key}"
        cache[key] = viewer
    return viewer


def _path_counts(trajectory: NDArray[np.generic], missing: int) -> int:
    values = np.asarray(trajectory).reshape(-1)
    return int(np.count_nonzero(values != missing))


def _bin_pack_aux(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "_render_step": 0,
        "items_location": np.zeros((20, 3), dtype=np.float32),
        "container": _config_array(
            config,
            "bin_pack_render_container",
            (6,),
            np.float32,
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        ),
    }


def _init_aux(
    task_id: str, obs: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    if task_id == "BinPack-v2":
        return _bin_pack_aux(config)
    if task_id == "CVRP-v1":
        return {"_render_step": 0, "num_total_visits": 1}
    if task_id == "JobShop-v0":
        return {
            "_render_step": 0,
            "scheduled_times": np.full((20, 8), -1, dtype=np.int32),
            "step_count": 0,
        }
    if task_id == "MMST-v0":
        connected = np.full((3, 70), -1, dtype=np.int32)
        positions = _asarray(obs["positions"], np.int32).reshape(3)
        connected[:, 0] = positions
        return {
            "_render_step": 0,
            "connected_nodes": connected,
            "position_index": np.zeros(3, np.int32),
        }
    if task_id == "MultiCVRP-v0":
        order = np.zeros((2, 40), dtype=np.int16)
        return {"_render_step": 0, "order": order, "step_count": 1}
    return {"_render_step": 0}


def _update_aux(
    task_id: str,
    aux: dict[str, Any],
    previous_obs: Mapping[str, Any] | None,
    action: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if previous_obs is None or action is None:
        return aux
    render_step = int(aux.get("_render_step", 0)) + 1
    aux["_render_step"] = render_step
    if task_id == "BinPack-v2":
        action_array = np.asarray(action, dtype=np.int32).reshape(-1)
        if action_array.size >= 2:
            ems, item = int(action_array[0]), int(action_array[1])
            ems_obs = previous_obs["ems"]
            container = np.asarray(aux["container"], dtype=np.float32)
            scale = np.asarray(
                [
                    container[1] - container[0],
                    container[3] - container[2],
                    container[5] - container[4],
                ],
                dtype=np.float32,
            )
            loc = np.asarray(
                [
                    _asarray(ems_obs["x1"])[ems],
                    _asarray(ems_obs["y1"])[ems],
                    _asarray(ems_obs["z1"])[ems],
                ],
                dtype=np.float32,
            )
            aux["items_location"][item] = loc * scale
        replay = _config_step_array(
            config,
            "bin_pack_render_items_location_replay",
            render_step,
            (20, 3),
            np.float32,
        )
        if replay is not None:
            aux["items_location"] = replay
    elif task_id == "CVRP-v1":
        aux["num_total_visits"] = int(aux.get("num_total_visits", 1)) + 1
        replay = _config_step_array(
            config,
            "cvrp_render_num_total_visits_replay",
            render_step,
            (),
            np.int32,
        )
        if replay is not None:
            aux["num_total_visits"] = int(replay)
    elif task_id == "JobShop-v0":
        actions = np.asarray(action, dtype=np.int32).reshape(-1)
        scheduled = np.asarray(aux["scheduled_times"], dtype=np.int32)
        ops_mask = _asarray(previous_obs["ops_mask"], bool)
        machine_ids = _asarray(previous_obs["ops_machine_ids"], np.int32)
        step_count = int(aux.get("step_count", 0))
        for machine, job in enumerate(actions.tolist()):
            if 0 <= job < scheduled.shape[0]:
                ops = np.flatnonzero(ops_mask[job])
                if ops.size > 0 and int(machine_ids[job, ops[0]]) == machine:
                    scheduled[job, ops[0]] = step_count
        aux["scheduled_times"] = scheduled
        aux["step_count"] = step_count + 1
        replay = _config_step_array(
            config,
            "job_shop_render_scheduled_times_replay",
            render_step,
            (20, 8),
            np.int32,
        )
        if replay is not None:
            aux["scheduled_times"] = replay
            aux["step_count"] = render_step
    elif task_id == "MMST-v0":
        actions = np.asarray(action, dtype=np.int32).reshape(-1)
        connected = np.asarray(aux["connected_nodes"], dtype=np.int32)
        index = np.asarray(aux["position_index"], dtype=np.int32)
        for agent, node in enumerate(actions.tolist()):
            next_index = int(index[agent]) + 1
            if next_index < connected.shape[1]:
                connected[agent, next_index] = int(node)
                index[agent] = next_index
        aux["connected_nodes"] = connected
        aux["position_index"] = index
        replay = _config_step_array(
            config,
            "mmst_render_connected_nodes_replay",
            render_step,
            (3, 70),
            np.int32,
        )
        if replay is not None:
            aux["connected_nodes"] = replay
            aux["position_index"] = np.maximum(
                np.count_nonzero(replay >= 0, axis=1) - 1, 0
            ).astype(np.int32)
    elif task_id == "MultiCVRP-v0":
        vehicle_actions = np.asarray(action, dtype=np.int16).reshape(-1)
        step_count = int(aux.get("step_count", 1))
        order = np.asarray(aux["order"], dtype=np.int16)
        if (
            vehicle_actions.size >= order.shape[0]
            and step_count < order.shape[1]
        ):
            order[:, step_count] = vehicle_actions[: order.shape[0]]
        aux["order"] = order
        aux["step_count"] = step_count + 1
    elif task_id == "Sokoban-v0":
        fixed = _config_step_array(
            config,
            "sokoban_render_fixed_grid_replay",
            render_step,
            (10, 10),
            np.int32,
        )
        variable = _config_step_array(
            config,
            "sokoban_render_variable_grid_replay",
            render_step,
            (10, 10),
            np.int32,
        )
        if fixed is not None and variable is not None:
            aux["fixed_grid"] = fixed
            aux["variable_grid"] = variable
    elif task_id == "Tetris-v0":
        grid_padded = _config_step_array(
            config,
            "tetris_render_grid_padded_replay",
            render_step,
            (13, 13),
            np.int32,
        )
        if grid_padded is not None:
            aux["grid_padded"] = grid_padded
            aux["grid_padded_old"] = grid_padded
    elif task_id == "SearchAndRescue-v0":
        headings = _config_step_array(
            config,
            "search_and_rescue_render_headings_replay",
            render_step,
            (2,),
            np.float32,
        )
        target_positions = _config_step_array(
            config,
            "search_and_rescue_render_target_positions_replay",
            render_step,
            (40, 2),
            np.float32,
        )
        target_found = _config_step_array(
            config,
            "search_and_rescue_render_target_found_replay",
            render_step,
            (40,),
            np.int32,
        )
        if headings is not None:
            aux["search_and_rescue_headings"] = headings
        if target_positions is not None:
            aux["search_and_rescue_target_positions"] = target_positions
        if target_found is not None:
            aux["search_and_rescue_target_found"] = target_found.astype(bool)
    return aux


def update_render_aux(
    task_id: str,
    aux: dict[str, Any] | None,
    obs: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    reset: bool,
    previous_obs: Mapping[str, Any] | None = None,
    action: Any = None,
) -> dict[str, Any]:
    """Update per-env renderer state from the latest EnvPool transition."""
    if reset or aux is None:
        return _init_aux(task_id, obs, config)
    return _update_aux(task_id, aux, previous_obs, action, config)


def _make_bin_pack_state(
    obs: Mapping[str, Any], config: Mapping[str, Any], aux: Mapping[str, Any]
) -> Any:
    ems = obs["ems"]
    container = _config_array(
        config,
        "bin_pack_render_container",
        (6,),
        np.float32,
        aux.get("container", [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
    )
    item_x = _config_array(
        config,
        "bin_pack_render_item_x_len",
        (20,),
        np.float32,
        _get(obs, "items.x_len"),
    )
    item_y = _config_array(
        config,
        "bin_pack_render_item_y_len",
        (20,),
        np.float32,
        _get(obs, "items.y_len"),
    )
    item_z = _config_array(
        config,
        "bin_pack_render_item_z_len",
        (20,),
        np.float32,
        _get(obs, "items.z_len"),
    )
    loc = np.asarray(aux.get("items_location", np.zeros((20, 3))), np.float32)
    return packing.BinPackState(
        container=packing.Space(*[_a(v) for v in container.tolist()]),
        ems=packing.Space(
            _a(ems["x1"]),
            _a(ems["x2"]),
            _a(ems["y1"]),
            _a(ems["y2"]),
            _a(ems["z1"]),
            _a(ems["z2"]),
        ),
        ems_mask=_a(obs["ems_mask"]),
        items=packing.Item(_a(item_x), _a(item_y), _a(item_z)),
        items_mask=_a(obs["items_mask"]),
        items_placed=_a(obs["items_placed"]),
        items_location=packing.Location(
            _a(loc[:, 0]), _a(loc[:, 1]), _a(loc[:, 2])
        ),
        action_mask=_a(obs["action_mask"]),
        sorted_ems_indexes=_a(np.arange(40, dtype=np.int32)),
        key=_key(),
    )


def _make_cvrp_state(obs: Mapping[str, Any], aux: Mapping[str, Any]) -> Any:
    return routing.CVRPState(
        coordinates=_a(obs["coordinates"]),
        demands=_a(obs["demands"]),
        position=_a(obs["position"]),
        capacity=_a(obs["capacity"]),
        visited_mask=_a(~_asarray(obs["unvisited_nodes"], bool)),
        trajectory=_a(obs["trajectory"]),
        num_total_visits=_a(aux.get("num_total_visits", 1), np.int32),
        key=_key(),
    )


def _make_cleaner_state(obs: Mapping[str, Any]) -> Any:
    return routing.CleanerState(
        grid=_a(obs["grid"]),
        agents_locations=_a(obs["agents_locations"]),
        action_mask=_a(obs["action_mask"]),
        step_count=_a(obs["step_count"], np.int32),
        key=_key(),
    )


def _make_flat_pack_state(obs: Mapping[str, Any]) -> Any:
    blocks = _asarray(obs["blocks"])
    return packing.FlatPackState(
        grid=_a(obs["grid"]),
        num_blocks=_a(blocks.shape[0], np.int32),
        blocks=_a(blocks),
        action_mask=_a(obs["action_mask"]),
        placed_blocks=_a(np.zeros(blocks.shape[0], dtype=bool)),
        step_count=_a(0, np.int32),
        key=_key(),
    )


def _make_game2048_state(obs: Mapping[str, Any], score: float) -> Any:
    return logic.Game2048State(
        board=_a(obs["board"]),
        step_count=_a(0, np.int32),
        action_mask=_a(obs["action_mask"]),
        score=_a(score, np.float32),
        key=_key(),
    )


def _make_graph_coloring_state(obs: Mapping[str, Any]) -> Any:
    return logic.GraphColoringState(
        adj_matrix=_a(obs["adj_matrix"]),
        colors=_a(obs["colors"]),
        current_node_index=_a(obs["current_node_index"], np.int32),
        action_mask=_a(obs["action_mask"]),
        key=_key(),
    )


def _make_job_shop_state(obs: Mapping[str, Any], aux: Mapping[str, Any]) -> Any:
    return packing.JobShopState(
        ops_machine_ids=_a(obs["ops_machine_ids"]),
        ops_durations=_a(obs["ops_durations"]),
        ops_mask=_a(obs["ops_mask"]),
        machines_job_ids=_a(obs["machines_job_ids"]),
        machines_remaining_times=_a(obs["machines_remaining_times"]),
        action_mask=_a(obs["action_mask"]),
        step_count=_a(aux.get("step_count", 0), np.int32),
        scheduled_times=_a(aux.get("scheduled_times", np.full((20, 8), -1))),
        key=_key(),
    )


def _make_knapsack_state(obs: Mapping[str, Any]) -> Any:
    return packing.KnapsackState(
        weights=_a(obs["weights"]),
        values=_a(obs["values"]),
        packed_items=_a(obs["packed_items"]),
        remaining_budget=_a(0.0, np.float32),
        key=_key(),
    )


def _make_lbf_state(obs: Mapping[str, Any]) -> Any:
    view = _asarray(obs["agents_view"], np.int32).reshape(2, -1, 3)[0]
    food = view[:2]
    agents = view[2:4]
    return routing.LBFState(
        agents=routing.LBFAgent(
            id=_a(np.arange(agents.shape[0], dtype=np.int32)),
            position=_a(agents[:, :2]),
            level=_a(agents[:, 2]),
            loading=_a(np.zeros(agents.shape[0], dtype=bool)),
        ),
        food_items=routing.Food(
            id=_a(np.arange(food.shape[0], dtype=np.int32)),
            position=_a(food[:, :2]),
            level=_a(food[:, 2]),
            eaten=_a(np.all(food[:, :2] < 0, axis=1)),
        ),
        step_count=_a(obs["step_count"], np.int32),
        key=_key(),
    )


def _make_maze_state(obs: Mapping[str, Any]) -> Any:
    agent = obs["agent_position"]
    target = obs["target_position"]
    return routing.MazeState(
        agent_position=routing.MazePosition(_a(agent["row"]), _a(agent["col"])),
        target_position=routing.MazePosition(
            _a(target["row"]), _a(target["col"])
        ),
        walls=_a(obs["walls"]),
        action_mask=_a(obs["action_mask"]),
        step_count=_a(obs["step_count"], np.int32),
        key=_key(),
    )


def _make_minesweeper_state(
    obs: Mapping[str, Any], config: Mapping[str, Any]
) -> Any:
    mines = _config_array(
        config,
        "minesweeper_mine_locations",
        (int(obs["num_mines"]),),
        np.int32,
        np.arange(int(obs["num_mines"]), dtype=np.int32),
    )
    return logic.MinesweeperState(
        board=_a(obs["board"]),
        step_count=_a(obs["step_count"], np.int32),
        flat_mine_locations=_a(mines),
        key=_key(),
    )


def _mmst_nodes_to_connect(
    obs: Mapping[str, Any], config: Mapping[str, Any]
) -> NDArray[np.int32]:
    configured = _config_array(
        config,
        "mmst_render_nodes_to_connect",
        (3, 4),
        np.int32,
        np.full((3, 4), -1, dtype=np.int32),
    )
    if np.any(configured >= 0):
        return configured
    node_types = _asarray(obs["node_types"], np.int32)
    groups = []
    for agent in range(3):
        values = (2 * agent, 2 * agent + 1)
        nodes = np.flatnonzero(np.isin(node_types, values)).astype(np.int32)
        padded = np.full(4, -1, dtype=np.int32)
        padded[: min(4, nodes.size)] = nodes[:4]
        groups.append(padded)
    return np.stack(groups, axis=0)


def _make_mmst_state(
    obs: Mapping[str, Any], config: Mapping[str, Any], aux: Mapping[str, Any]
) -> Any:
    connected = np.asarray(
        aux.get("connected_nodes", np.full((3, 70), -1)), dtype=np.int32
    )
    positions = _asarray(obs["positions"], np.int32)
    position_index = np.maximum(
        np.count_nonzero(connected >= 0, axis=1) - 1, 0
    ).astype(np.int32)
    return routing.MMSTState(
        node_types=_a(obs["node_types"]),
        adj_matrix=_a(obs["adj_matrix"]),
        connected_nodes=_a(connected),
        connected_nodes_index=_a(np.full((3, 36), -1, dtype=np.int32)),
        nodes_to_connect=_a(_mmst_nodes_to_connect(obs, config)),
        node_edges=_a(np.full((3, 36, 36), -1, dtype=np.int32)),
        positions=_a(positions),
        position_index=_a(position_index),
        action_mask=_a(obs["action_mask"]),
        finished_agents=_a(np.zeros(3, dtype=bool)),
        step_count=_a(obs["step_count"], np.int32),
        key=_key(),
    )


def _make_multi_cvrp_state(
    obs: Mapping[str, Any], aux: Mapping[str, Any]
) -> Any:
    nodes = obs["nodes"]
    windows = obs["windows"]
    coeffs = obs["coeffs"]
    vehicles = obs["vehicles"]
    step_count = int(aux.get("step_count", 1))
    positions = np.zeros(2, dtype=np.int16)
    coords = _asarray(nodes["coordinates"], np.float32)
    vehicle_coords = _asarray(vehicles["coordinates"], np.float32)
    for vehicle in range(2):
        distances = np.sum((coords - vehicle_coords[vehicle]) ** 2, axis=1)
        positions[vehicle] = int(np.argmin(distances))
    return routing.MultiCVRPState(
        nodes=routing.Node(_a(nodes["coordinates"]), _a(nodes["demands"])),
        windows=routing.TimeWindow(_a(windows["start"]), _a(windows["end"])),
        coeffs=routing.PenalityCoeff(_a(coeffs["early"]), _a(coeffs["late"])),
        vehicles=routing.StateVehicle(
            local_times=_a(vehicles["local_times"]),
            capacities=_a(vehicles["capacities"]),
            positions=_a(positions),
            distances=_a(np.zeros(2, dtype=np.float32)),
            time_penalties=_a(np.zeros(2, dtype=np.float32)),
        ),
        order=_a(aux.get("order", np.zeros((2, 40), dtype=np.int16))),
        step_count=_a(step_count, np.int16),
        action_mask=_a(obs["action_mask"]),
        key=_key(),
    )


def _make_pac_man_state(obs: Mapping[str, Any]) -> Any:
    player = obs["player_locations"]
    return routing.PacManObservation(
        grid=_a(obs["grid"]),
        player_locations=routing.PacManPosition(
            _a(player["x"]), _a(player["y"])
        ),
        ghost_locations=_a(obs["ghost_locations"]),
        power_up_locations=_a(obs["power_up_locations"]),
        frightened_state_time=_a(obs["frightened_state_time"], np.int32),
        pellet_locations=_a(obs["pellet_locations"]),
        action_mask=_a(obs["action_mask"]),
        score=_a(obs["score"], np.int32),
    )


def _make_robot_warehouse_state(
    obs: Mapping[str, Any], config: Mapping[str, Any]
) -> Any:
    grid = _config_array(
        config,
        "robot_warehouse_render_grid",
        (2, 20, 10),
        np.int32,
        np.zeros((2, 20, 10), dtype=np.int32),
    )
    agent_x = _config_array(
        config,
        "robot_warehouse_render_agent_x",
        (4,),
        np.int32,
        np.zeros(4, dtype=np.int32),
    )
    agent_y = _config_array(
        config,
        "robot_warehouse_render_agent_y",
        (4,),
        np.int32,
        np.arange(4, dtype=np.int32),
    )
    shelf_x = _config_array(
        config,
        "robot_warehouse_render_shelf_x",
        (80,),
        np.int32,
        np.zeros(80, dtype=np.int32),
    )
    shelf_y = _config_array(
        config,
        "robot_warehouse_render_shelf_y",
        (80,),
        np.int32,
        np.zeros(80, dtype=np.int32),
    )
    return routing.RobotWarehouseState(
        grid=_a(grid),
        agents=routing.RobotAgent(
            position=routing.RobotPosition(_a(agent_x), _a(agent_y)),
            direction=_a(
                _config_array(
                    config,
                    "robot_warehouse_render_agent_direction",
                    (4,),
                    np.int32,
                    np.zeros(4, dtype=np.int32),
                )
            ),
            is_carrying=_a(
                _csv_bool_array(
                    config,
                    "robot_warehouse_render_agent_carrying",
                    (4,),
                    np.zeros(4, dtype=bool),
                )
            ),
        ),
        shelves=routing.Shelf(
            position=routing.RobotPosition(_a(shelf_x), _a(shelf_y)),
            is_requested=_a(
                _csv_bool_array(
                    config,
                    "robot_warehouse_render_shelf_requested",
                    (80,),
                    np.zeros(80, dtype=bool),
                )
            ),
        ),
        request_queue=_a(np.zeros(8, dtype=np.int32)),
        step_count=_a(obs["step_count"], np.int32),
        action_mask=_a(obs["action_mask"]),
        key=_key(),
    )


def _make_rubiks_state(obs: Mapping[str, Any]) -> Any:
    return logic.RubiksCubeState(
        cube=_a(obs["cube"]),
        step_count=_a(obs["step_count"], np.int32),
        key=_key(),
    )


def _make_search_and_rescue_state(
    obs: Mapping[str, Any], config: Mapping[str, Any], aux: Mapping[str, Any]
) -> Any:
    target_pos = _config_array(
        config,
        "search_and_rescue_target_positions",
        (40, 2),
        np.float32,
        np.zeros((40, 2), dtype=np.float32),
    )
    target_pos = np.asarray(
        aux.get("search_and_rescue_target_positions", target_pos),
        dtype=np.float32,
    )
    headings = np.asarray(
        aux.get(
            "search_and_rescue_headings",
            _config_array(
                config,
                "search_and_rescue_headings",
                (2,),
                np.float32,
                np.zeros(2, dtype=np.float32),
            ),
        ),
        dtype=np.float32,
    )
    target_found = np.asarray(
        aux.get(
            "search_and_rescue_target_found",
            _csv_bool_array(
                config,
                "search_and_rescue_target_found",
                (40,),
                np.zeros(40, dtype=bool),
            ),
        ),
        dtype=bool,
    )
    return swarms.SearchAndRescueState(
        searchers=swarms.AgentState(
            pos=_a(obs["positions"]),
            heading=_a(headings),
            speed=_a(
                _config_array(
                    config,
                    "search_and_rescue_speeds",
                    (2,),
                    np.float32,
                    np.zeros(2, dtype=np.float32),
                )
            ),
        ),
        targets=swarms.TargetState(
            pos=_a(target_pos),
            vel=_a(
                _config_array(
                    config,
                    "search_and_rescue_target_velocities",
                    (40, 2),
                    np.float32,
                    np.zeros((40, 2), dtype=np.float32),
                )
            ),
            found=_a(target_found),
        ),
        key=_key(),
        step=int(obs["step"]),
    )


def _make_sliding_tile_state(obs: Mapping[str, Any]) -> Any:
    puzzle = _asarray(obs["puzzle"], np.int32)
    empty = np.argwhere(puzzle == 0)
    return logic.SlidingTilePuzzleState(
        puzzle=_a(puzzle),
        empty_tile_position=_a(empty[0] if empty.size else [0, 0], np.int32),
        key=_key(),
        step_count=_a(obs["step_count"], np.int32),
    )


def _make_snake_state(obs: Mapping[str, Any]) -> Any:
    grid = _asarray(obs["grid"], np.float32)
    body_state = grid[:, :, 4].astype(np.int32)
    body = body_state > 0
    head = np.argwhere(grid[:, :, 1] > 0)
    fruit = np.argwhere(grid[:, :, 3] > 0)
    length = max(int(np.max(body_state)), 1)
    return routing.SnakeState(
        body=_a(body),
        body_state=_a(body_state),
        head_position=routing.SnakePosition(
            _a(head[0, 0] if head.size else 0, np.int32),
            _a(head[0, 1] if head.size else 0, np.int32),
        ),
        tail=_a(body_state == 1),
        fruit_position=routing.SnakePosition(
            _a(fruit[0, 0] if fruit.size else 0, np.int32),
            _a(fruit[0, 1] if fruit.size else 0, np.int32),
        ),
        length=_a(length, np.int32),
        step_count=_a(obs["step_count"], np.int32),
        action_mask=_a(obs["action_mask"]),
        key=_key(),
    )


def _make_sokoban_state(obs: Mapping[str, Any], aux: Mapping[str, Any]) -> Any:
    if "fixed_grid" in aux and "variable_grid" in aux:
        return type(
            "SokobanState",
            (),
            {
                "fixed_grid": _a(aux["fixed_grid"]),
                "variable_grid": _a(aux["variable_grid"]),
            },
        )()
    grid = _asarray(obs["grid"], np.int32)
    if grid.ndim == 3:
        return type(
            "SokobanState",
            (),
            {
                "fixed_grid": _a(grid[:, :, 0]),
                "variable_grid": _a(grid[:, :, 1]),
            },
        )()
    return type(
        "SokobanState",
        (),
        {"fixed_grid": _a(grid), "variable_grid": _a(np.zeros_like(grid))},
    )()


def _make_sudoku_state(obs: Mapping[str, Any]) -> Any:
    return logic.SudokuState(board=_a(obs["board"]))


def _make_tetris_state(
    obs: Mapping[str, Any], score: float, aux: Mapping[str, Any]
) -> Any:
    grid = _asarray(obs["grid"], np.int32)
    if "grid_padded" in aux:
        grid_padded = np.asarray(aux["grid_padded"], dtype=np.int32)
    else:
        grid_padded = np.zeros((13, 13), dtype=np.int32)
        grid_padded[: grid.shape[0], : grid.shape[1]] = grid
    return packing.TetrisState(
        grid_padded=_a(grid_padded),
        grid_padded_old=_a(aux.get("grid_padded_old", grid_padded)),
        tetromino_index=_a(0, np.int32),
        old_tetromino_rotated=_a(np.zeros((4, 4), dtype=np.int32)),
        new_tetromino=_a(obs["tetromino"]),
        x_position=_a(3, np.int32),
        y_position=_a(-1, np.int32),
        action_mask=_a(obs["action_mask"]),
        full_lines=_a(np.zeros(13, dtype=bool)),
        score=_a(score, np.float32),
        reward=_a(0.0, np.float32),
        key=_key(),
        is_reset=_a(True),
        step_count=_a(obs["step_count"], np.int32),
    )


def _make_tsp_state(obs: Mapping[str, Any]) -> Any:
    trajectory = _asarray(obs["trajectory"], np.int32)
    return routing.TSPState(
        coordinates=_a(obs["coordinates"]),
        position=_a(obs["position"], np.int32),
        visited_mask=_a(~_asarray(obs["action_mask"], bool)),
        trajectory=_a(trajectory),
        num_visited=_a(_path_counts(trajectory, -1), np.int32),
        key=_key(),
    )


def _sokoban_combine(variable_grid: Any, fixed_grid: Any) -> Any:
    variable = np.asarray(variable_grid)
    fixed = np.asarray(fixed_grid)
    grid = np.maximum(variable, fixed).astype(np.uint8, copy=False)
    target_agent = np.logical_and(fixed == 2, variable == 3)
    target_box = np.logical_and(fixed == 2, variable == 4)
    grid[target_agent] = 5
    grid[target_box] = 6
    return _a(grid)


def render_official_frame(
    task_id: str,
    obs: Mapping[str, Any],
    info: Mapping[str, Any],
    config: Mapping[str, Any],
    aux: Mapping[str, Any],
    width: int,
    height: int,
    score: float,
) -> NDArray[np.uint8]:
    """Render a Jumanji EnvPool observation through vendored official viewers."""
    del info
    render_passes = 1
    if task_id == "BinPack-v2":
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: packing.BinPackViewer("BinPack", "rgb_array"),
        )
        state = _make_bin_pack_state(obs, config, aux)
    elif task_id == "CVRP-v1":
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: routing.CVRPViewer("CVRP", 20, "rgb_array"),
        )
        state = _make_cvrp_state(obs, aux)
    elif task_id == "Cleaner-v0":
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: routing.CleanerViewer("Cleaner", "rgb_array"),
        )
        state = _make_cleaner_state(obs)
    elif task_id == "Connector-v2":
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: routing.ConnectorViewer("Connector", 10, "rgb_array"),
        )
        state = _a(obs["grid"])
    elif task_id == "FlatPack-v0":
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: packing.FlatPackViewer(
                "FlatPack", _asarray(obs["blocks"]).shape[0], "rgb_array"
            ),
        )
        state = _make_flat_pack_state(obs)
    elif task_id == "Game2048-v1":
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: logic.Game2048Viewer("Game2048", 4, "rgb_array"),
        )
        state = _make_game2048_state(obs, score)
    elif task_id == "GraphColoring-v1":
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: logic.GraphColoringViewer("GraphColoring", "rgb_array"),
        )
        state = _make_graph_coloring_state(obs)
    elif task_id == "JobShop-v0":
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: packing.JobShopViewer("JobShop", 20, 10, 8, 6, "rgb_array"),
        )
        state = _make_job_shop_state(obs, aux)
    elif task_id == "Knapsack-v1":
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: packing.KnapsackViewer("Knapsack", "rgb_array", 12.5),
        )
        state = _make_knapsack_state(obs)
    elif task_id == "LevelBasedForaging-v0":
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: routing.LevelBasedForagingViewer(
                8, "LevelBasedForaging", "rgb_array"
            ),
        )
        state = _make_lbf_state(obs)
    elif task_id == "MMST-v0":
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: routing.MMSTViewer(3, "MMST", "rgb_array"),
        )
        state = _make_mmst_state(obs, config, aux)
    elif task_id == "Maze-v0":
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: routing.MazeEnvViewer("Maze", "rgb_array"),
        )
        state = _make_maze_state(obs)
    elif task_id == "Minesweeper-v0":
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: logic.MinesweeperViewer(10, 10, "rgb_array"),
        )
        state = _make_minesweeper_state(obs, config)
    elif task_id == "MultiCVRP-v0":
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: routing.MultiCVRPViewer(
                "MultiCVRP", 2, 20, 10, "rgb_array"
            ),
        )
        state = _make_multi_cvrp_state(obs, aux)
    elif task_id == "PacMan-v1":
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: routing.PacManViewer("PacMan", "rgb_array"),
        )
        state = _make_pac_man_state(obs)
    elif task_id == "RobotWarehouse-v0":
        goals = _a(np.asarray([[4, 19], [5, 19]], dtype=np.int32))
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: routing.RobotWarehouseViewer(
                (20, 10), goals, "RobotWarehouse", "rgb_array"
            ),
        )
        state = _make_robot_warehouse_state(obs, config)
    elif task_id.startswith("RubiksCube"):
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: logic.RubiksCubeViewer(
                ["white", "green", "red", "blue", "orange", "yellow"],
                _asarray(obs["cube"]).shape[1],
                "rgb_array",
            ),
        )
        state = _make_rubiks_state(obs)
    elif task_id == "SearchAndRescue-v0":
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: swarms.SearchAndRescueViewer(
                "SearchAndRescue", render_mode="rgb_array"
            ),
        )
        state = _make_search_and_rescue_state(obs, config, aux)
        render_passes = 2
    elif task_id == "SlidingTilePuzzle-v0":
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: logic.SlidingTilePuzzleViewer(
                "SlidingTilePuzzle",
                "rgb_array",
            ),
        )
        state = _make_sliding_tile_state(obs)
    elif task_id == "Snake-v1":
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: routing.SnakeViewer("Snake", "rgb_array"),
        )
        state = _make_snake_state(obs)
    elif task_id == "Sokoban-v0":
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: routing.BoxViewer("Sokoban", _sokoban_combine, "rgb_array"),
        )
        state = _make_sokoban_state(obs, aux)
    elif task_id.startswith("Sudoku"):
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: logic.SudokuViewer("Sudoku", "rgb_array"),
        )
        state = _make_sudoku_state(obs)
    elif task_id == "TSP-v1":
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: routing.TSPViewer("TSP", "rgb_array"),
        )
        state = _make_tsp_state(obs)
    elif task_id == "Tetris-v0":
        viewer = _cached_viewer(
            aux,
            task_id,
            lambda: packing.TetrisViewer(10, 10, "rgb_array"),
        )
        state = _make_tetris_state(obs, score, aux)
    else:
        raise NotImplementedError(
            f"no Jumanji renderer registered for {task_id}"
        )
    return _viewer_frame(viewer, state, width, height, render_passes)
