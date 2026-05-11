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
"""Flat renderers for Jumanji packing environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d
import numpy as np
from numpy.typing import NDArray

from envpool.jumanji._official_render.base import MatplotlibViewer, asset_path


@dataclass
class Space:
    x1: Any
    x2: Any
    y1: Any
    y2: Any
    z1: Any
    z2: Any


class Item(NamedTuple):
    x_len: Any
    y_len: Any
    z_len: Any


class Location(NamedTuple):
    x: Any
    y: Any
    z: Any


def item_from_space(space: Space) -> Item:
    return Item(
        x_len=space.x2 - space.x1,
        y_len=space.y2 - space.y1,
        z_len=space.z2 - space.z1,
    )


@dataclass
class BinPackState:
    container: Space
    ems: Space
    ems_mask: Any
    items: Item
    items_mask: Any
    items_placed: Any
    items_location: Location
    action_mask: Any
    sorted_ems_indexes: Any
    key: Any


@dataclass
class FlatPackState:
    grid: Any
    num_blocks: Any
    blocks: Any
    action_mask: Any
    placed_blocks: Any
    step_count: Any
    key: Any


@dataclass
class JobShopState:
    ops_machine_ids: Any
    ops_durations: Any
    ops_mask: Any
    machines_job_ids: Any
    machines_remaining_times: Any
    action_mask: Any
    step_count: Any
    scheduled_times: Any
    key: Any


@dataclass
class KnapsackState:
    weights: Any
    values: Any
    packed_items: Any
    remaining_budget: Any
    key: Any


@dataclass
class TetrisState:
    grid_padded: Any
    grid_padded_old: Any
    tetromino_index: Any
    old_tetromino_rotated: Any
    new_tetromino: Any
    x_position: Any
    y_position: Any
    action_mask: Any
    full_lines: Any
    score: Any
    reward: Any
    key: Any
    is_reset: Any
    step_count: Any


class BinPackViewer(MatplotlibViewer):
    FONT_STYLE = "monospace"

    def render(
        self, state: BinPackState, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        for entity in self._create_entities(state):
            ax.add_collection3d(entity)
        self._add_overlay(fig, ax, state)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def _get_fig_ax(
        self,
        name_suffix: str | None = None,
        show: bool = True,
        padding: float = 0.05,
        **fig_kwargs: Any,
    ) -> tuple[plt.Figure, Any]:
        del padding, fig_kwargs
        figure_name = getattr(self, "_figure_name", self._name)
        name = figure_name if name_suffix is None else figure_name + name_suffix
        recreate = not plt.fignum_exists(name)
        fig = plt.figure(name, figsize=self.figure_size)
        if recreate:
            fig.tight_layout()
            if not plt.isinteractive() and show:
                fig.show()
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.get_axes()[0]
        return fig, ax

    def _create_entities(
        self, state: BinPackState
    ) -> list[mpl_toolkits.mplot3d.art3d.Poly3DCollection]:
        entities = []
        n_items = len(state.items_mask)
        cmap = plt.get_cmap("hsv", n_items)
        for i in range(n_items):
            if state.items_placed[i]:
                entities.append(
                    self._create_box(
                        (
                            state.items_location.x[i],
                            state.items_location.y[i],
                            state.items_location.z[i],
                        ),
                        (
                            state.items.x_len[i],
                            state.items.y_len[i],
                            state.items.z_len[i],
                        ),
                        cmap(i),
                        0.3,
                    )
                )

        container = item_from_space(state.container)
        entities.append(
            self._create_box(
                (0.0, 0.0, 0.0),
                (container.x_len, container.y_len, container.z_len),
                "cyan",
                0.05,
            )
        )
        return entities

    def _create_box(
        self,
        pos: tuple[float, float, float],
        lens: tuple[float, float, float],
        colour: Any,
        alpha: float,
    ) -> mpl_toolkits.mplot3d.art3d.Poly3DCollection:
        return mpl_toolkits.mplot3d.art3d.Poly3DCollection(
            self._create_box_vertices(pos, lens),
            linewidths=1,
            edgecolors="black",
            facecolors=colour,
            alpha=alpha,
        )

    def _add_overlay(
        self, fig: plt.Figure, ax: Any, state: BinPackState
    ) -> None:
        eps = 0.05
        container = item_from_space(state.container)
        ax.set(
            xlim=(-container.x_len * eps, container.x_len * (1 + eps)),
            ylim=(-container.y_len * eps, container.y_len * (1 + eps)),
            zlim=(-container.z_len * eps, container.z_len * (1 + eps)),
        )
        ax.set_xlabel("x", font=self.FONT_STYLE)
        ax.set_ylabel("y", font=self.FONT_STYLE)
        ax.set_zlabel("z", font=self.FONT_STYLE)

        n_items = sum(state.items_mask)
        placed_items = sum(state.items_placed)
        container_volume = (
            float(container.x_len)
            * float(container.y_len)
            * float(container.z_len)
        )
        used_volume = sum(
            float(state.items.x_len[i])
            * float(state.items.y_len[i])
            * float(state.items.z_len[i])
            for i, placed in enumerate(state.items_placed)
            if placed
        )
        metrics = [
            ("Placed", f"{placed_items:{len(str(n_items))}}/{n_items}"),
            ("Used Volume", f"{used_volume / container_volume:6.1%}"),
        ]
        fig.suptitle(
            " | ".join(key + ": " + value for key, value in metrics),
            font=self.FONT_STYLE,
        )

    def _create_box_vertices(
        self, pos: tuple[float, float, float], lens: tuple[float, float, float]
    ) -> list[list[tuple[float, float, float]]]:
        verts = [
            (pos[0], pos[1], pos[2]),
            (pos[0] + lens[0], pos[1], pos[2]),
            (pos[0] + lens[0], pos[1] + lens[1], pos[2]),
            (pos[0] + lens[0], pos[1] + lens[1], pos[2] + lens[2]),
            (pos[0], pos[1] + lens[1], pos[2] + lens[2]),
            (pos[0], pos[1], pos[2] + lens[2]),
            (pos[0] + lens[0], pos[1], pos[2] + lens[2]),
            (pos[0], pos[1] + lens[1], pos[2]),
        ]
        faces = [
            [0, 1, 2, 7],
            [1, 2, 3, 6],
            [0, 1, 6, 5],
            [0, 7, 4, 5],
            [2, 7, 4, 3],
            [6, 3, 4, 5],
        ]
        return [[verts[i] for i in face] for face in faces]


class FlatPackViewer(MatplotlibViewer):
    def __init__(
        self, name: str, num_blocks: int, render_mode: str = "human"
    ) -> None:
        colormap_indices = np.arange(0, 1, 1 / num_blocks)
        colormap = plt.get_cmap("hsv", num_blocks + 1)
        self.colors = [(1.0, 1.0, 1.0, 1.0)]
        for colormap_idx in colormap_indices:
            r, g, b, _ = colormap(colormap_idx)
            self.colors.append((r, g, b, 0.7))
        super().__init__(name, render_mode)

    def render(
        self, state: FlatPackState, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._add_grid_image(state.grid, ax)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def _add_grid_image(self, grid: Any, ax: plt.Axes) -> None:
        self._draw_grid(grid, ax)
        ax.set_axis_off()
        ax.set_aspect(1)
        ax.relim()
        ax.autoscale_view()

    def _draw_grid(self, grid: Any, ax: plt.Axes) -> None:
        grid = np.flipud(grid)
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                cell_value = int(grid[row, col])
                cell = plt.Rectangle(
                    (col, row),
                    1,
                    1,
                    facecolor=self.colors[cell_value],
                    edgecolor="black",
                    linewidth=1,
                )
                ax.add_patch(cell)
                if cell_value != 0:
                    ax.text(
                        col + 0.5,
                        row + 0.5,
                        str(cell_value),
                        color="#606060",
                        ha="center",
                        va="center",
                        fontsize="xx-large",
                    )


class JobShopViewer(MatplotlibViewer):
    COLORMAP_NAME = "hsv"

    def __init__(
        self,
        name: str,
        num_jobs: int,
        num_machines: int,
        max_num_ops: int,
        max_op_duration: int,
        render_mode: str = "human",
    ) -> None:
        self._num_jobs = num_jobs
        self._num_machines = num_machines
        self._max_num_ops = max_num_ops
        self._max_op_duration = max_op_duration
        self._cmap = plt.get_cmap(self.COLORMAP_NAME, self._num_jobs + 1)
        super().__init__(name, render_mode)

    def render(
        self, state: JobShopState, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        ax.set_title(f"Scheduled Jobs at Time={state.step_count}")
        ax.axvline(state.step_count, ls="--", color="red", lw=0.5)
        self._prepare_figure(ax)
        self._add_scheduled_ops(ax, state)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def _prepare_figure(self, ax: plt.Axes) -> None:
        ax.set_xlabel("Time")
        ax.set_ylabel("Machine ID")
        xlim = (
            self._num_jobs
            * self._max_num_ops
            * self._max_op_duration
            // self._num_machines
        )
        ax.set_xlim(0, xlim)
        ax.set_ylim(-0.9, self._num_machines)
        cast(Any, ax.xaxis.get_major_locator()).set_params(integer=True)
        cast(Any, ax.yaxis.get_major_locator()).set_params(integer=True)
        major_ticks = np.arange(0, xlim, 10)
        minor_ticks = np.arange(0, xlim, 1)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.grid(axis="x", linewidth=0.25)
        ax.set_axisbelow(True)

    def _add_scheduled_ops(self, ax: plt.Axes, state: JobShopState) -> None:
        for job_id in range(self._num_jobs):
            for op_id in range(self._max_num_ops):
                start_time = state.scheduled_times[job_id, op_id]
                machine_id = state.ops_machine_ids[job_id, op_id]
                duration = state.ops_durations[job_id, op_id]
                if start_time < 0:
                    continue
                rectangle = matplotlib.patches.Rectangle(
                    (start_time, machine_id - 0.4),
                    width=duration,
                    height=0.8,
                    linewidth=1,
                    facecolor=self._cmap(job_id),
                    edgecolor="black",
                )
                ax.add_patch(rectangle)
                rx, ry = rectangle.get_xy()
                ax.annotate(
                    f"J{job_id}",
                    (
                        rx + rectangle.get_width() / 2.0,
                        ry + rectangle.get_height() / 2.0,
                    ),
                    color="black",
                    fontsize=10,
                    ha="center",
                    va="center",
                )


class KnapsackViewer(MatplotlibViewer):
    def __init__(
        self, name: str, render_mode: str = "human", total_budget: float = 2.0
    ) -> None:
        self._total_budget = total_budget
        super().__init__(name, render_mode)

    def render(
        self, state: KnapsackState, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        self._prepare_figure(ax)
        budget_used: np.ndarray = np.sum(
            state.weights, where=state.packed_items
        )
        total_value: np.ndarray = np.sum(state.values, where=state.packed_items)
        ax.set_title(
            f"Total value: {round(float(total_value), 2):.2f}. "
            f"Budget used: {round(float(budget_used), 2):.2f}/{self._total_budget}."
        )
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def _prepare_figure(self, ax: plt.Axes) -> None:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(plt.imread(asset_path("knapsack.png")), extent=(0, 1, 0, 1))


class TetrisViewer(MatplotlibViewer):
    def __init__(
        self, num_rows: int, num_cols: int, render_mode: str = "human"
    ) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.n_colors = 10
        colormap_indices = np.arange(0, 1, 1 / self.n_colors)
        colormap = plt.get_cmap("hsv", self.n_colors + 1)
        self.colors: list[Any] = [(1.0, 1.0, 1.0, 1.0)]
        for colormap_idx in colormap_indices:
            self.colors.append(colormap(colormap_idx))
        self.edgecolors = [(0.0, 0.0, 0.0), (0.9, 0.9, 0.9)]
        super().__init__(f"{num_rows}x{num_cols} Tetris", render_mode)

    def render(
        self, state: TetrisState, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        fig.suptitle(f"Tetris    Score: {int(state.score)}", size=20)
        ax.invert_yaxis()
        self._add_grid_image(ax, self._create_rendering_grid(state))
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def _create_rendering_grid(self, state: TetrisState) -> Any:
        grid = state.grid_padded[: self.num_rows, : self.num_cols]
        tetromino = np.zeros((4, self.num_cols))
        center_position = self.num_cols - 4
        tetromino_color_id = state.grid_padded.max() + 1
        tetromino[:, center_position : center_position + 4] = (
            state.new_tetromino * tetromino_color_id
        )
        return np.vstack((tetromino, grid))

    def _add_grid_image(self, ax: plt.Axes, grid: Any) -> None:
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                cell_value = int(grid[row, col])
                color_id = (
                    cell_value
                    if cell_value == 0
                    else cell_value % (len(self.colors) - 1) + 1
                )
                ax.add_patch(
                    plt.Rectangle(
                        (col, row),
                        1,
                        1,
                        facecolor=self.colors[color_id],
                        edgecolor=self.edgecolors[row < 4],
                        linewidth=1,
                    )
                )
        ax.set_axis_off()
        ax.set_aspect(1)
        ax.relim()
        ax.autoscale_view()
