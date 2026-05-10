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
"""Python/Matplotlib rendering override for EnvPool's Jumanji tasks.

The renderers in this file are adapted from the Apache-2.0 Jumanji v1.1.1
Matplotlib viewers, but consume EnvPool observations and do not import the
official runtime packages.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from envpool.jumanji.jumanji_official_render import (
    configure_matplotlib,
    render_official_frame,
    update_render_aux,
)
from envpool.python.envpool import _normalize_render_env_ids

configure_matplotlib()

_Color = tuple[float, float, float]
_ColorLike = str | _Color
_GymEnvT = TypeVar("_GymEnvT", bound=type)
plt_Rectangle: Any

_PALETTE: tuple[_Color, ...] = (
    (0.121, 0.466, 0.705),
    (1.000, 0.498, 0.054),
    (0.172, 0.627, 0.172),
    (0.839, 0.153, 0.157),
    (0.580, 0.404, 0.741),
    (0.549, 0.337, 0.294),
    (0.890, 0.467, 0.761),
    (0.498, 0.498, 0.498),
    (0.737, 0.741, 0.133),
    (0.090, 0.745, 0.811),
)


def _asarray(value: Any) -> NDArray[np.generic]:
    return np.asarray(value)


def _get(tree: Mapping[str, Any], path: str) -> Any:
    value: Any = tree
    for key in path.split("."):
        value = value[key]
    return value


def _maybe_get(tree: Mapping[str, Any], path: str, default: Any = None) -> Any:
    try:
        return _get(tree, path)
    except KeyError:
        return default


def _slice_tree(value: Any, index: int) -> Any:
    if isinstance(value, Mapping):
        return {key: _slice_tree(item, index) for key, item in value.items()}
    array = np.asarray(value)
    if array.shape[:1] == (0,):
        return array
    if array.ndim == 0:
        return array
    return np.array(array[index], copy=True)


def _slice_info(info: Mapping[str, Any], index: int) -> dict[str, Any]:
    sliced = {}
    for key, value in info.items():
        array = np.asarray(value)
        if array.ndim > 0 and array.shape[0] > index:
            sliced[key] = np.array(array[index], copy=True)
        else:
            sliced[key] = value
    return sliced


def _empty_cache(env: Any) -> None:
    if not hasattr(env, "_jumanji_render_obs_cache"):
        env._jumanji_render_obs_cache = {}
        env._jumanji_render_info_cache = {}
        env._jumanji_render_score_cache = {}
        env._jumanji_render_aux_cache = {}


def _close_render_aux(aux: Any) -> None:
    if not isinstance(aux, Mapping):
        return
    for viewer in aux.get("_viewer_cache", {}).values():
        close = getattr(viewer, "close", None)
        if callable(close):
            close()


def _slice_action(action: Any, batch_index: int, batch_size: int) -> Any:
    if action is None:
        return None
    if isinstance(action, Mapping):
        return _slice_tree(action, batch_index)
    array = np.asarray(action)
    if array.ndim > 0 and array.shape[0] == batch_size:
        return np.array(array[batch_index], copy=True)
    return np.array(array, copy=True)


def _cache_gymnasium_output(
    env: Any,
    output: Any,
    reset: bool,
    action: Any = None,
) -> None:
    _empty_cache(env)
    if reset:
        obs, info = output
        reward = None
    else:
        obs, reward, _, _, info = output
    env_ids = np.asarray(info["env_id"], dtype=np.int32).reshape(-1)
    rewards = None if reward is None else np.asarray(reward).reshape(-1)
    batch_size = int(env_ids.shape[0])
    config = getattr(env, "config", {})
    for batch_index, env_id in enumerate(env_ids.tolist()):
        env_id_int = int(env_id)
        obs_slice = _slice_tree(obs, batch_index)
        info_slice = _slice_info(info, batch_index)
        previous_obs = env._jumanji_render_obs_cache.get(env_id_int)
        if reset:
            _close_render_aux(env._jumanji_render_aux_cache.get(env_id_int))
        action_slice = (
            None if reset else _slice_action(action, batch_index, batch_size)
        )
        env._jumanji_render_aux_cache[env_id_int] = update_render_aux(
            env._jumanji_task_id,
            env._jumanji_render_aux_cache.get(env_id_int),
            obs_slice,
            config,
            reset=reset,
            previous_obs=previous_obs,
            action=action_slice,
        )
        env._jumanji_render_obs_cache[env_id_int] = obs_slice
        env._jumanji_render_info_cache[env_id_int] = info_slice
        if reset:
            env._jumanji_render_score_cache[env_id_int] = 0.0
        elif rewards is not None:
            env._jumanji_render_score_cache[env_id_int] = float(
                env._jumanji_render_score_cache.get(env_id_int, 0.0)
            ) + float(rewards[batch_index])


def _figure_to_rgb(fig: Any) -> NDArray[np.uint8]:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return np.array(rgba[:, :, :3], copy=True)


def _render_figure(
    width: int,
    height: int,
    draw: Callable[[Any, Any], None],
    *,
    projection: str | None = None,
    facecolor: str = "white",
) -> NDArray[np.uint8]:
    import matplotlib.pyplot as plt
    from matplotlib.layout_engine import ConstrainedLayoutEngine

    dpi = 100
    fig = plt.figure(
        figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=facecolor
    )
    fig.set_layout_engine(
        layout=ConstrainedLayoutEngine(h_pad=0.05, w_pad=0.05)
    )
    ax = fig.add_subplot(111, projection=projection)
    try:
        draw(fig, ax)
        return _figure_to_rgb(fig)
    finally:
        plt.close(fig)


def _draw_grid_lines(
    ax: Any, rows: int, cols: int, color: str = "black"
) -> None:
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color=color, linewidth=0.8)
    ax.tick_params(
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )


def _imshow_grid(
    ax: Any,
    grid: NDArray[np.generic],
    colors: Sequence[_ColorLike],
    *,
    grid_lines: bool = True,
) -> None:
    from matplotlib.colors import ListedColormap

    ax.imshow(grid, cmap=ListedColormap(colors), interpolation="nearest")
    ax.set_aspect("equal")
    if grid_lines:
        _draw_grid_lines(ax, int(grid.shape[0]), int(grid.shape[1]), "0.35")


def _plot_cell_marker(
    ax: Any, row: int, col: int, color: _ColorLike, marker: str = "o"
) -> None:
    ax.scatter(
        [col],
        [row],
        color=color,
        marker=marker,
        s=180,
        edgecolors="black",
        linewidths=0.8,
    )


def _render_matrix_env(
    obs: Mapping[str, Any],
    width: int,
    height: int,
    grid_path: str,
    title: str,
    colors: Sequence[_ColorLike],
) -> NDArray[np.uint8]:
    grid = _asarray(_get(obs, grid_path))

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title(title)
        _imshow_grid(ax, grid, colors)

    return _render_figure(width, height, draw)


def _render_bin_pack(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    ems = cast(Mapping[str, Any], obs["ems"])
    mask = _asarray(obs["ems_mask"]).astype(bool)
    placed = int(np.count_nonzero(_asarray(obs["items_placed"])))

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title(f"Placed: {placed} / {len(_asarray(obs['items_placed']))}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        xs1, xs2 = _asarray(ems["x1"]), _asarray(ems["x2"])
        ys1, ys2 = _asarray(ems["y1"]), _asarray(ems["y2"])
        zs1, zs2 = _asarray(ems["z1"]), _asarray(ems["z2"])
        for idx in np.flatnonzero(mask)[:8]:
            ax.bar3d(
                xs1[idx],
                ys1[idx],
                zs1[idx],
                max(float(xs2[idx] - xs1[idx]), 0.01),
                max(float(ys2[idx] - ys1[idx]), 0.01),
                max(float(zs2[idx] - zs1[idx]), 0.01),
                alpha=0.18,
                color=_PALETTE[idx % len(_PALETTE)],
                edgecolor="black",
            )

    return _render_figure(width, height, draw, projection="3d")


def _render_cleaner(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    grid = _asarray(obs["grid"])
    agents = _asarray(obs["agents_locations"])

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title("Cleaner")
        _imshow_grid(ax, grid, ["white", "limegreen", "black"])
        for idx, (row, col) in enumerate(agents):
            _plot_cell_marker(
                ax, int(row), int(col), _PALETTE[idx % len(_PALETTE)]
            )

    return _render_figure(width, height, draw)


def _render_connector(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    grid = _asarray(obs["grid"])

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title("Connector")
        ax.imshow(grid, cmap="tab20", interpolation="nearest")
        _draw_grid_lines(ax, grid.shape[0], grid.shape[1], "0.25")

    return _render_figure(width, height, draw)


def _render_flat_pack(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    return _render_matrix_env(
        obs, width, height, "grid", "FlatPack", ["white", *_PALETTE]
    )


def _render_game2048(
    obs: Mapping[str, Any],
    width: int,
    height: int,
    info: Mapping[str, Any],
    score: float,
) -> NDArray[np.uint8]:
    board = _asarray(obs["board"]).astype(int)
    tile_colors = {
        0: "#cdc1b4",
        1: "#eee4da",
        2: "#ede0c8",
        3: "#f2b179",
        4: "#f59563",
        5: "#f67c5f",
        6: "#f65e3b",
        7: "#edcf72",
        8: "#edcc61",
        9: "#edc850",
        10: "#edc53f",
        11: "#edc22e",
    }

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title(f"2048    Score: {int(score)}")
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(
            plt_Rectangle((0, 0), 4, 4, facecolor="#bbada0", edgecolor="none")
        )
        for row in range(4):
            for col in range(4):
                value = int(board[row, col])
                color = tile_colors.get(value, "#3c3a32")
                ax.add_patch(
                    plt_Rectangle(
                        (col + 0.06, 3 - row + 0.06),
                        0.88,
                        0.88,
                        facecolor=color,
                    )
                )
                if value > 0:
                    ax.text(
                        col + 0.5,
                        3 - row + 0.5,
                        str(1 << value),
                        ha="center",
                        va="center",
                        fontsize=18,
                        fontweight="bold",
                        color="#776e65" if value <= 2 else "white",
                    )

    return _render_figure(width, height, draw)


def _graph_layout(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    return np.cos(angles), np.sin(angles)


def _render_graph_coloring(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    adj = _asarray(obs["adj_matrix"]).astype(bool)
    colors = _asarray(obs["colors"]).astype(int)
    current = int(_asarray(obs["current_node_index"]))
    n = adj.shape[0]
    xs, ys = _graph_layout(n)

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title("GraphColoring")
        ax.axis("off")
        ax.set_aspect("equal")
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    ax.plot(
                        [xs[i], xs[j]],
                        [ys[i], ys[j]],
                        color="0.65",
                        linewidth=0.8,
                    )
        for i in range(n):
            face = (
                "white"
                if colors[i] < 0
                else _PALETTE[int(colors[i]) % len(_PALETTE)]
            )
            edge = "red" if i == current else "black"
            ax.scatter(
                [xs[i]],
                [ys[i]],
                s=190,
                c=[face],
                edgecolors=edge,
                linewidths=1.5,
            )
            ax.text(xs[i], ys[i], str(i), ha="center", va="center", fontsize=8)

    return _render_figure(width, height, draw)


def _render_job_shop(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    machines_job_ids = _asarray(obs["machines_job_ids"]).astype(int)
    remaining = _asarray(obs["machines_remaining_times"]).astype(int)

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title("Scheduled Jobs")
        ax.set_xlabel("Time")
        ax.set_ylabel("Machine")
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, len(machines_job_ids) - 0.5)
        ax.grid(True, alpha=0.2)
        for machine, (job, duration) in enumerate(
            zip(machines_job_ids, remaining, strict=True)
        ):
            if duration > 0:
                ax.barh(
                    machine,
                    duration * 8,
                    left=0,
                    color=_PALETTE[job % len(_PALETTE)],
                )

    return _render_figure(width, height, draw)


def _render_knapsack(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    values = _asarray(obs["values"])
    packed = _asarray(obs["packed_items"]).astype(bool)
    total = float(np.sum(values[packed]))
    budget = float(_asarray(info.get("remaining_budget", 0.0)))

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title(f"Total value: {total:.2f}    Budget: {budget:.2f}")
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_patch(
            plt_Rectangle(
                (0.28, 0.15),
                0.44,
                0.58,
                facecolor="#0b8f6a",
                edgecolor="black",
                linewidth=2,
            )
        )
        ax.add_patch(
            plt_Rectangle(
                (0.30, 0.56),
                0.40,
                0.23,
                facecolor="#f0302d",
                edgecolor="black",
                linewidth=1.5,
            )
        )
        ax.add_patch(
            plt_Rectangle(
                (0.42, 0.18),
                0.27,
                0.25,
                facecolor="#ffa52f",
                edgecolor="black",
                linewidth=1,
            )
        )
        ax.plot(
            [0.30, 0.20, 0.20, 0.28],
            [0.58, 0.65, 0.28, 0.20],
            color="#245f9f",
            linewidth=4,
        )
        ax.text(0.5, 0.06, f"{np.count_nonzero(packed)} packed", ha="center")

    return _render_figure(width, height, draw)


def _render_lbf(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    view = _asarray(obs["agents_view"]).astype(int)
    base = view[0]
    foods = base[:6].reshape(2, 3)
    agents = np.vstack([base[6:9], base[9:12]])

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title("LevelBasedForaging")
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(7.5, -0.5)
        ax.set_aspect("equal")
        _draw_grid_lines(ax, 8, 8, "0.35")
        for row, col, level in foods:
            if row >= 0:
                _plot_cell_marker(ax, int(row), int(col), "red")
                ax.text(
                    col,
                    row,
                    str(int(level)),
                    ha="center",
                    va="center",
                    color="white",
                )
        for idx, (row, col, level) in enumerate(agents):
            _plot_cell_marker(ax, int(row), int(col), _PALETTE[idx])
            ax.text(
                col,
                row,
                str(int(level)),
                ha="center",
                va="center",
                color="white",
            )

    return _render_figure(width, height, draw, facecolor="#111111")


def _render_maze(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    walls = _asarray(obs["walls"]).astype(int)
    agent = (
        int(_get(obs, "agent_position.row")),
        int(_get(obs, "agent_position.col")),
    )
    target = (
        int(_get(obs, "target_position.row")),
        int(_get(obs, "target_position.col")),
    )

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title("Maze")
        _imshow_grid(ax, walls, ["white", "black"])
        _plot_cell_marker(ax, target[0], target[1], "limegreen", "s")
        _plot_cell_marker(ax, agent[0], agent[1], "red")

    return _render_figure(width, height, draw)


def _render_minesweeper(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    board = _asarray(obs["board"]).astype(int)

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title(f"{board.shape[0]}x{board.shape[1]} Minesweeper")
        hidden = board < 0
        ax.imshow(
            np.where(hidden, 0, 1),
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        _draw_grid_lines(ax, board.shape[0], board.shape[1])
        for (row, col), value in np.ndenumerate(board):
            if value > 0:
                ax.text(
                    col,
                    row,
                    str(int(value)),
                    ha="center",
                    va="center",
                    color="tab:blue",
                )

    return _render_figure(width, height, draw)


def _render_mmst(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    adj = _asarray(obs["adj_matrix"]).astype(bool)
    node_types = _asarray(obs["node_types"]).astype(int)
    positions = {
        int(x) for x in _asarray(obs["positions"]).reshape(-1) if int(x) >= 0
    }
    n = adj.shape[0]
    xs, ys = _graph_layout(n)

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title("MMST")
        ax.axis("off")
        ax.set_aspect("equal")
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    ax.plot(
                        [xs[i], xs[j]],
                        [ys[i], ys[j]],
                        color="0.65",
                        linewidth=0.7,
                    )
        for i in range(n):
            color = (
                "red"
                if i in positions
                else _PALETTE[node_types[i] % len(_PALETTE)]
            )
            ax.scatter(
                [xs[i]],
                [ys[i]],
                s=75,
                c=[color],
                edgecolors="black",
                linewidths=0.5,
            )
            ax.text(xs[i], ys[i], str(i), ha="center", va="center", fontsize=5)

    return _render_figure(width, height, draw)


def _render_multi_cvrp(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    nodes = cast(Mapping[str, Any], obs["nodes"])
    vehicles = cast(Mapping[str, Any], obs["vehicles"])
    coords = _asarray(nodes["coordinates"])
    vehicle_coords = _asarray(vehicles["coordinates"])

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title("MultiCVRP")
        ax.scatter(coords[:, 0], coords[:, 1], c="black", s=20)
        ax.scatter(coords[:1, 0], coords[:1, 1], marker="s", c="black", s=50)
        ax.scatter(
            vehicle_coords[:, 0],
            vehicle_coords[:, 1],
            c=["tab:red", "tab:blue"],
            s=60,
        )
        ax.set_aspect("equal", adjustable="box")

    return _render_figure(width, height, draw)


def _render_pac_man(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    grid = _asarray(obs["grid"]).astype(int)
    pellets = _asarray(obs["pellet_locations"]).astype(int)
    ghosts = _asarray(obs["ghost_locations"]).astype(int)
    power = _asarray(obs["power_up_locations"]).astype(int)
    player = (
        int(_get(obs, "player_locations.y")),
        int(_get(obs, "player_locations.x")),
    )
    score = int(_asarray(obs["score"]))

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title(f"PacMan    Score: {score}")
        _imshow_grid(ax, grid, ["black", "#1c28be"], grid_lines=False)
        ax.axis("off")
        for row, col in pellets:
            if row >= 0 and col >= 0:
                ax.scatter([col], [row], c="#f5d2aa", s=6)
        for row, col in power:
            if row >= 0 and col >= 0:
                ax.scatter([col], [row], c="#ffd2ff", s=30)
        for idx, (row, col) in enumerate(ghosts):
            if row >= 0 and col >= 0:
                ax.scatter(
                    [col], [row], c=[_PALETTE[idx + 3]], s=70, marker="s"
                )
        _plot_cell_marker(ax, player[0], player[1], "yellow")

    return _render_figure(width, height, draw)


def _render_robot_warehouse(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    agents = _asarray(obs["agents_view"]).astype(int)

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title("RobotWarehouse")
        ax.imshow(
            np.zeros((10, 10)),
            cmap="Greys",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        _draw_grid_lines(ax, 10, 10, "0.45")
        for col in (2, 5):
            ax.axvspan(col - 0.5, col + 0.5, color="#645a96", alpha=0.65)
        for idx, row_view in enumerate(agents):
            row, col = int(row_view[0]), int(row_view[1])
            carrying = bool(row_view[2])
            _plot_cell_marker(ax, row, col, _PALETTE[idx % len(_PALETTE)])
            if carrying:
                ax.scatter([col], [row], c="orange", s=35, marker="s")

    return _render_figure(width, height, draw)


def _render_rubiks(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    cube = _asarray(obs["cube"]).astype(int)
    face_names = ("UP", "FRONT", "RIGHT", "BACK", "LEFT", "DOWN")
    colors = ["white", "yellow", "red", "orange", "green", "blue"]

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title("3x3x3 Rubik's Cube")
        ax.axis("off")
        positions = [(1, 2), (1, 1), (2, 1), (3, 1), (0, 1), (1, 0)]
        for face, (ox, oy) in enumerate(positions):
            for row in range(3):
                for col in range(3):
                    ax.add_patch(
                        plt_Rectangle(
                            (ox * 3 + col, oy * 3 + (2 - row)),
                            1,
                            1,
                            facecolor=colors[
                                int(cube[face, row, col]) % len(colors)
                            ],
                            edgecolor="black",
                        )
                    )
            ax.text(
                ox * 3 + 1.5,
                oy * 3 + 3.15,
                face_names[face],
                ha="center",
                fontsize=6,
            )
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")

    return _render_figure(width, height, draw)


def _render_search_and_rescue(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    positions = _asarray(obs["positions"])
    target_visible = float(_asarray(obs["targets_remaining"])) > 0.0

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title("SearchAndRescue")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        if target_visible:
            ax.scatter([0.1], [0.0], c="tab:orange", s=80)
        ax.scatter(
            positions[:, 0], positions[:, 1], c=["tab:blue", "tab:green"], s=70
        )

    return _render_figure(width, height, draw)


def _render_sliding_tile(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    puzzle = _asarray(obs["puzzle"]).astype(int)

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title("SlidingTilePuzzle")
        ax.imshow(puzzle, cmap="coolwarm", interpolation="nearest")
        _draw_grid_lines(ax, puzzle.shape[0], puzzle.shape[1])
        for (row, col), value in np.ndenumerate(puzzle):
            if value:
                ax.text(
                    col,
                    row,
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    return _render_figure(width, height, draw)


def _render_snake(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    grid = np.asarray(obs["grid"], dtype=np.float32)

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title("Snake")
        ax.imshow(
            np.zeros(grid.shape[:2]),
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        _draw_grid_lines(ax, grid.shape[0], grid.shape[1], "0.75")
        body = np.argwhere(grid[:, :, 0] > 0.5)
        for row, col in body:
            ax.scatter([col], [row], c="limegreen", s=45, marker="s")
        for row, col in np.argwhere(grid[:, :, 3] > 0.5):
            ax.scatter([col], [row], c="tab:red", s=60)
        for row, col in np.argwhere(grid[:, :, 1] > 0.5):
            _plot_cell_marker(ax, int(row), int(col), "green")

    return _render_figure(width, height, draw)


def _render_sokoban(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    grid = _asarray(obs["grid"]).astype(int)
    variable = grid[:, :, 0]
    fixed = grid[:, :, 1]

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title("Sokoban")
        base = np.where(fixed == 1, 1, 0)
        _imshow_grid(ax, base, ["#e8e8e8", "#823618"])
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                if fixed[row, col] == 2:
                    ax.scatter(
                        [col], [row], facecolors="none", edgecolors="red", s=120
                    )
                if variable[row, col] == 4:
                    ax.scatter(
                        [col],
                        [row],
                        c="#ffb000",
                        marker="s",
                        s=120,
                        edgecolors="black",
                    )
                if variable[row, col] == 3:
                    _plot_cell_marker(ax, row, col, "limegreen")

    return _render_figure(width, height, draw)


def _render_sudoku(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    board = _asarray(obs["board"]).astype(int)

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title("Sudoku")
        ax.imshow(
            np.ones_like(board),
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        _draw_grid_lines(ax, 9, 9)
        for x in (2.5, 5.5):
            ax.axvline(x, color="black", linewidth=2)
            ax.axhline(x, color="black", linewidth=2)
        for (row, col), value in np.ndenumerate(board):
            if value >= 0:
                ax.text(
                    col,
                    row,
                    str(int(value) + 1),
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    return _render_figure(width, height, draw)


def _render_tetris(
    obs: Mapping[str, Any],
    width: int,
    height: int,
    info: Mapping[str, Any],
    score: float,
) -> NDArray[np.uint8]:
    grid = _asarray(obs["grid"]).astype(int)
    tetromino = _asarray(obs["tetromino"]).astype(int)

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title(f"Tetris    Score: {int(score)}")
        board = np.zeros((14, 10), dtype=int)
        board[4:, :] = grid
        for row in range(min(4, tetromino.shape[0])):
            for col in range(min(4, tetromino.shape[1])):
                if tetromino[row, col]:
                    board[row, 3 + col] = 2
        _imshow_grid(ax, board, ["white", "black", "orange"])

    return _render_figure(width, height, draw)


def _render_tsp(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    coords = _asarray(obs["coordinates"])
    trajectory = _asarray(obs["trajectory"]).astype(int)
    path = [int(node) for node in trajectory if int(node) >= 0]

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title("TSP")
        ax.scatter(coords[:, 0], coords[:, 1], c="0.35", s=50)
        if len(path) > 1:
            pts = coords[path]
            ax.plot(pts[:, 0], pts[:, 1], color="tab:blue", linewidth=1)
        ax.set_aspect("equal", adjustable="box")

    return _render_figure(width, height, draw)


def _render_cvrp(
    obs: Mapping[str, Any], width: int, height: int, info: Mapping[str, Any]
) -> NDArray[np.uint8]:
    coords = _asarray(obs["coordinates"])
    trajectory = _asarray(obs["trajectory"]).astype(int)
    path = [
        int(node) for node in trajectory if 0 <= int(node) < coords.shape[0]
    ]

    def draw(fig: Any, ax: Any) -> None:
        del fig
        ax.set_title("CVRP")
        ax.scatter(coords[:, 0], coords[:, 1], c="black", s=20)
        ax.scatter(coords[:1, 0], coords[:1, 1], marker="s", c="black", s=50)
        if len(path) > 1:
            pts = coords[path]
            ax.plot(pts[:, 0], pts[:, 1], color="tab:blue", linewidth=1)
        ax.set_aspect("equal", adjustable="box")

    return _render_figure(width, height, draw)


def _render_frame(
    task_id: str,
    obs: Mapping[str, Any],
    info: Mapping[str, Any],
    config: Mapping[str, Any],
    aux: Mapping[str, Any],
    width: int,
    height: int,
    score: float,
) -> NDArray[np.uint8]:
    return render_official_frame(
        task_id, obs, info, config, aux, width, height, score
    )


def _render_approx_frame(
    task_id: str,
    obs: Mapping[str, Any],
    info: Mapping[str, Any],
    width: int,
    height: int,
    score: float,
) -> NDArray[np.uint8]:
    if task_id == "BinPack-v2":
        return _render_bin_pack(obs, width, height, info)
    if task_id == "CVRP-v1":
        return _render_cvrp(obs, width, height, info)
    if task_id == "Cleaner-v0":
        return _render_cleaner(obs, width, height, info)
    if task_id == "Connector-v2":
        return _render_connector(obs, width, height, info)
    if task_id == "FlatPack-v0":
        return _render_flat_pack(obs, width, height, info)
    if task_id == "Game2048-v1":
        return _render_game2048(obs, width, height, info, score)
    if task_id == "GraphColoring-v1":
        return _render_graph_coloring(obs, width, height, info)
    if task_id == "JobShop-v0":
        return _render_job_shop(obs, width, height, info)
    if task_id == "Knapsack-v1":
        return _render_knapsack(obs, width, height, info)
    if task_id == "LevelBasedForaging-v0":
        return _render_lbf(obs, width, height, info)
    if task_id == "MMST-v0":
        return _render_mmst(obs, width, height, info)
    if task_id == "Maze-v0":
        return _render_maze(obs, width, height, info)
    if task_id == "Minesweeper-v0":
        return _render_minesweeper(obs, width, height, info)
    if task_id == "MultiCVRP-v0":
        return _render_multi_cvrp(obs, width, height, info)
    if task_id == "PacMan-v1":
        return _render_pac_man(obs, width, height, info)
    if task_id == "RobotWarehouse-v0":
        return _render_robot_warehouse(obs, width, height, info)
    if task_id.startswith("RubiksCube"):
        return _render_rubiks(obs, width, height, info)
    if task_id == "SearchAndRescue-v0":
        return _render_search_and_rescue(obs, width, height, info)
    if task_id == "SlidingTilePuzzle-v0":
        return _render_sliding_tile(obs, width, height, info)
    if task_id == "Snake-v1":
        return _render_snake(obs, width, height, info)
    if task_id == "Sokoban-v0":
        return _render_sokoban(obs, width, height, info)
    if task_id.startswith("Sudoku"):
        return _render_sudoku(obs, width, height, info)
    if task_id == "TSP-v1":
        return _render_tsp(obs, width, height, info)
    if task_id == "Tetris-v0":
        return _render_tetris(obs, width, height, info, score)
    raise NotImplementedError(f"no Jumanji renderer registered for {task_id}")


def _jumanji_render(
    self: Any,
    env_ids: Any = None,
    camera_id: int | None = None,
) -> NDArray[np.uint8] | None:
    del camera_id
    render_mode, default_env_id, width, height, _ = self._render_config()
    width = 256 if width <= 0 else width
    height = 256 if height <= 0 else height
    if render_mode not in {"rgb_array", "human"}:
        raise RuntimeError(
            "render_mode must be set to 'rgb_array' or 'human' when creating this env"
        )
    env_ids_arr = _normalize_render_env_ids(env_ids, default_env_id)
    _empty_cache(self)
    frames = []
    config = getattr(self, "config", {})
    for env_id in env_ids_arr.tolist():
        if int(env_id) not in self._jumanji_render_obs_cache:
            raise RuntimeError(
                "Jumanji render requires reset() before render()."
            )
        obs = self._jumanji_render_obs_cache[int(env_id)]
        info = self._jumanji_render_info_cache.get(int(env_id), {})
        score = float(self._jumanji_render_score_cache.get(int(env_id), 0.0))
        aux = self._jumanji_render_aux_cache.get(int(env_id), {})
        frames.append(
            _render_frame(
                self._jumanji_task_id,
                obs,
                info,
                config,
                aux,
                width,
                height,
                score,
            )
        )
    batch = np.stack(frames, axis=0).astype(np.uint8, copy=False)
    if render_mode == "human":
        if batch.shape[0] != 1:
            raise ValueError(
                "render_mode='human' only supports a single env_id"
            )
        self._show_human_frame(batch[0])
        return None
    return batch


def with_jumanji_python_render(cls: _GymEnvT, task_id: str) -> _GymEnvT:
    """Install a task-specific Python render override on a GymnasiumEnvPool class."""
    env_cls = cast(Any, cls)
    original_reset = cast(Callable[..., Any], env_cls.reset)
    original_step = cast(Callable[..., Any], env_cls.step)
    original_recv = cast(Callable[..., Any], env_cls.recv)
    original_close = getattr(env_cls, "close", None)

    def reset(self: Any, *args: Any, **kwargs: Any) -> Any:
        output = original_reset(self, *args, **kwargs)
        _cache_gymnasium_output(self, output, reset=True)
        return output

    def step(self: Any, *args: Any, **kwargs: Any) -> Any:
        action = args[0] if args else kwargs.get("action", None)
        output = original_step(self, *args, **kwargs)
        _cache_gymnasium_output(self, output, reset=False, action=action)
        return output

    def recv(self: Any, *args: Any, **kwargs: Any) -> Any:
        reset_output = bool(kwargs.get("reset", False))
        output = original_recv(self, *args, **kwargs)
        _cache_gymnasium_output(self, output, reset=reset_output)
        return output

    def close(self: Any, *args: Any, **kwargs: Any) -> Any:
        for aux in getattr(self, "_jumanji_render_aux_cache", {}).values():
            _close_render_aux(aux)
        if callable(original_close):
            return original_close(self, *args, **kwargs)
        return None

    env_cls._jumanji_task_id = task_id
    env_cls.reset = reset
    env_cls.step = step
    env_cls.recv = recv
    env_cls.close = close
    env_cls.render = _jumanji_render
    return cls


def _load_patches() -> None:
    global plt_Rectangle
    from matplotlib.patches import Rectangle as plt_Rectangle


_load_patches()
