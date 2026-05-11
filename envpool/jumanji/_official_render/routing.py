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
"""Flat renderers for Jumanji routing environments."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import IntEnum
from itertools import groupby
from typing import Any, Callable, NamedTuple, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch
from matplotlib.quiver import Quiver
from numpy.typing import NDArray
from PIL import Image

from envpool.jumanji._official_render.base import (
    MatplotlibViewer,
    asset_path,
    spring_layout,
)


def _tree_slice(tree: Any, index: int) -> Any:
    if is_dataclass(tree):
        field_values = {
            field.name: _tree_slice(getattr(tree, field.name), index)
            for field in fields(tree)
        }
        return cast(Any, type(tree))(**field_values)
    if isinstance(tree, tuple) and hasattr(tree, "_fields"):
        tuple_values = (
            _tree_slice(getattr(tree, name), index) for name in tree._fields
        )
        return cast(Any, type(tree))(*tuple_values)
    return tree[index]


class MazeViewer(MatplotlibViewer):
    COLORS = {0: [1, 1, 1], 1: [0, 0, 0]}

    def render(
        self, maze: Any, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._add_grid_image(maze, ax)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def _add_grid_image(self, maze: Any, ax: plt.Axes) -> image.AxesImage:
        img = self._create_grid_image(maze)
        ax.set_axis_off()
        return ax.imshow(img)

    def _create_grid_image(self, maze: Any) -> NDArray[np.generic]:
        img = np.zeros((*maze.shape, 3))
        for tile_value, color in self.COLORS.items():
            img[np.where(maze == tile_value)] = color
        return np.pad(img, ((1, 1), (1, 1), (0, 0)))


@dataclass
class CleanerState:
    grid: Any
    agents_locations: Any
    action_mask: Any
    step_count: Any
    key: Any


class CleanerViewer(MazeViewer):
    COLORS = {1: [1, 1, 1], 2: [0, 0, 0], 0: [0, 1, 0]}
    AGENT_COLOR = ([1, 0, 0],)
    ALPHA = 0.5

    def render(
        self, state: CleanerState, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._add_grid_image(state, ax)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def _add_grid_image(
        self, state: CleanerState, ax: plt.Axes
    ) -> image.AxesImage:
        img = self._create_grid_image(state)
        ax.set_axis_off()
        return ax.imshow(img)

    def _create_grid_image(self, state: CleanerState) -> NDArray[np.generic]:
        grid = state.grid
        img = np.zeros((*grid.shape, 3))
        for tile_value, color in self.COLORS.items():
            img[np.where(grid == tile_value)] = color
        img = np.pad(img, ((0, 0), (0, 0), (0, 1)), constant_values=1)
        img = self._set_agents_colors(img, state.agents_locations)
        return self._draw_black_frame_around(img)

    def _set_agents_colors(
        self, img: NDArray[np.generic], agents_locations: NDArray[np.generic]
    ) -> NDArray[np.generic]:
        unique_locations, counts = np.unique(
            agents_locations, return_counts=True, axis=0
        )
        for location, count in zip(unique_locations, counts, strict=False):
            img[location[0], location[1], :3] = np.array(self.AGENT_COLOR)
            img[location[0], location[1], 3] = 1 - self.ALPHA**count
        return img

    def _draw_black_frame_around(
        self, img: NDArray[np.generic]
    ) -> NDArray[np.generic]:
        img = np.pad(img, ((1, 1), (1, 1), (0, 0)))
        img[0, :, 3] = 1
        img[-1, :, 3] = 1
        img[:, 0, 3] = 1
        img[:, -1, 3] = 1
        return img


def _connector_agent_id(value: int) -> int:
    return 0 if value == 0 else (value - 1) // 3 + 1


def _connector_is_path(value: int) -> bool:
    return (value > 0) and ((value - 1) % 3 == 0)


def _connector_is_target(value: int) -> bool:
    return (value > 0) and ((value - 3) % 3 == 0)


class ConnectorViewer(MatplotlibViewer):
    def __init__(
        self, name: str, num_agents: int, render_mode: str = "human"
    ) -> None:
        colormap_indices = np.arange(0, 1, 1 / num_agents)
        colormap = plt.get_cmap("hsv", num_agents + 1)
        self.colors = [(1.0, 1.0, 1.0, 1.0)]
        for colormap_idx in colormap_indices:
            self.colors.append(colormap(float(colormap_idx)))
        super().__init__(name, render_mode)

    def render(
        self, grid: Any, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._add_grid_image(grid, ax)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def _add_grid_image(self, grid: Any, ax: plt.Axes) -> None:
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                self._draw_grid_cell(grid[row, col], row, col, ax)
        ax.set_axis_off()
        ax.set_aspect(1)
        ax.relim()
        ax.autoscale_view()

    def _draw_grid_cell(
        self, cell_value: int, row: int, col: int, ax: plt.Axes
    ) -> None:
        cell = plt.Rectangle(
            (col, row), 1, 1, **self._get_cell_attributes(cell_value)
        )
        ax.add_patch(cell)
        if _connector_is_target(cell_value):
            ax.add_patch(
                plt.Rectangle(
                    (col + 0.25, row + 0.25),
                    0.5,
                    0.5,
                    **self._get_inner_cell_attributes(cell_value),
                )
            )

    def _get_cell_attributes(self, cell_value: int) -> dict[str, Any]:
        agent_id = _connector_agent_id(cell_value)
        color = self.colors[agent_id]
        if _connector_is_target(cell_value):
            color = (1.0, 1.0, 1.0, 1.0)
        elif _connector_is_path(cell_value):
            color = (*self.colors[agent_id][:3], 0.25)
        return {"facecolor": color, "edgecolor": "black", "linewidth": 1}

    def _get_inner_cell_attributes(self, cell_value: int) -> dict[str, Any]:
        return {"facecolor": self.colors[_connector_agent_id(cell_value)]}


@dataclass
class CVRPState:
    coordinates: Any
    demands: Any
    position: Any
    capacity: Any
    visited_mask: Any
    trajectory: Any
    num_total_visits: Any
    key: Any


class CVRPViewer(MatplotlibViewer):
    NODE_COLOUR = "black"
    COLORMAP_NAME = "hsv"
    NODE_SIZE = 0.01
    ROUTE_NODE_SIZE = 100
    DEPOT_SIZE = 0.04
    ARROW_WIDTH = 0.004

    def __init__(
        self, name: str, num_cities: int, render_mode: str = "human"
    ) -> None:
        self._num_cities = num_cities
        self._cmap = plt.get_cmap(self.COLORMAP_NAME, self._num_cities)
        super().__init__(name, render_mode)

    def render(
        self, state: CVRPState, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._prepare_figure(ax)
        self._add_tour(ax, state)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def _prepare_figure(self, ax: plt.Axes) -> None:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(
            plt.imread(asset_path("city_map.jpeg")),
            extent=(0, 1, 0, 1),
        )

    def _group_tour(
        self, tour: NDArray[np.generic]
    ) -> list[NDArray[np.generic]]:
        depot = tour[0]
        check_depot_fn = lambda x: (x != depot).all()
        tour_grouped = [
            np.array([depot, *list(g), depot])
            for k, g in groupby(tour, key=check_depot_fn)
            if k
        ]
        if (tour[-1] != tour[0]).all():
            tour_grouped[-1] = tour_grouped[-1][:-1]
        return tour_grouped

    def _draw_route(
        self, ax: plt.Axes, coords: NDArray[np.generic], col_id: int
    ) -> tuple[Quiver, PathCollection]:
        x, y = coords.T
        quiver = ax.quiver(
            x[:-1],
            y[:-1],
            x[1:] - x[:-1],
            y[1:] - y[:-1],
            scale_units="xy",
            angles="xy",
            scale=1,
            width=self.ARROW_WIDTH,
            headwidth=5,
            color=self._cmap(col_id),
        )
        scatter = ax.scatter(
            x, y, s=self.ROUTE_NODE_SIZE, color=self._cmap(col_id)
        )
        return quiver, scatter

    def _draw_cities(
        self, ax: plt.Axes, state: CVRPState
    ) -> list[plt.Circle | plt.Rectangle]:
        nodes: list[plt.Circle | plt.Rectangle] = []
        x_coords, y_coords = state.coordinates.T
        depot = plt.Rectangle(
            (
                x_coords[0] - 0.5 * self.DEPOT_SIZE,
                y_coords[0] - 0.5 * self.DEPOT_SIZE,
            ),
            self.DEPOT_SIZE,
            self.DEPOT_SIZE,
            color=self.NODE_COLOUR,
        )
        ax.add_artist(depot)
        nodes.append(depot)
        for i in range(1, x_coords.shape[0]):
            node = plt.Circle(
                (x_coords[i], y_coords[i]),
                self.NODE_SIZE,
                color=self.NODE_COLOUR,
            )
            ax.add_artist(node)
            nodes.append(node)
        return nodes

    def _add_tour(
        self, ax: plt.Axes, state: CVRPState
    ) -> tuple[
        list[plt.Circle | plt.Rectangle], list[tuple[Quiver, PathCollection]]
    ]:
        nodes = self._draw_cities(ax, state)
        routes = []
        if state.num_total_visits > 1:
            coords = state.coordinates[
                state.trajectory[: state.num_total_visits]
            ]
            for col_id, coords_route in enumerate(self._group_tour(coords)):
                routes.append(self._draw_route(ax, coords_route, col_id))
        return nodes, routes


@dataclass
class Entity:
    id: Any
    position: Any
    level: Any


@dataclass
class LBFAgent(Entity):
    loading: Any


@dataclass
class Food(Entity):
    eaten: Any


@dataclass
class LBFState:
    agents: LBFAgent
    food_items: Food
    step_count: Any
    key: Any


class LevelBasedForagingViewer(MatplotlibViewer):
    GRID_COLOR = (0, 0, 0)
    LINE_COLOR = (1, 1, 1)

    def __init__(
        self,
        grid_size: int,
        name: str = "LevelBasedForaging",
        render_mode: str = "human",
    ) -> None:
        self.rows, self.cols = (grid_size, grid_size)
        self.grid_size = 30
        self.icon_size = self.grid_size * 5 / self.rows
        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 1 + self.rows * (self.grid_size + 1)
        super().__init__(name, render_mode)

    def render(
        self, state: LBFState, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax(facecolor=self.GRID_COLOR)
        ax.clear()
        self._prepare_figure(ax)
        self._draw_state(ax, state)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def _prepare_figure(self, ax: plt.Axes) -> None:
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.patch.set_alpha(0.0)
        ax.set_axis_off()
        ax.set_aspect("equal", "box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    def _draw_state(self, ax: plt.Axes, state: LBFState) -> None:
        self._draw_grid(ax)
        self._draw_food(state.food_items, ax)
        self._draw_agents(state.agents, ax)

    def _draw_grid(self, ax: plt.Axes) -> None:
        lines = [
            [
                (0, (self.grid_size + 1) * r + 1),
                (
                    (self.grid_size + 1) * self.cols,
                    (self.grid_size + 1) * r + 1,
                ),
            ]
            for r in range(self.rows + 1)
        ]
        lines.extend(
            [
                ((self.grid_size + 1) * c + 1, 0),
                (
                    (self.grid_size + 1) * c + 1,
                    (self.grid_size + 1) * self.rows,
                ),
            ]
            for c in range(self.cols + 1)
        )
        ax.add_collection(LineCollection(lines, colors=(self.LINE_COLOR,)))

    def _draw_agents(self, agents: LBFAgent, ax: plt.Axes) -> None:
        img = plt.imread(asset_path("lbf_agent.png"))
        for i in range(len(agents.level)):
            agent = _tree_slice(agents, i)
            cell_center = self._entity_position(agent)
            imagebox = OffsetImage(img, zoom=self.icon_size / self.grid_size)
            ax.add_artist(
                AnnotationBbox(imagebox, cell_center, frameon=False, zorder=0)
            )
            self.draw_badge(agent.level, cell_center, ax)

    def _draw_food(self, food_items: Food, ax: plt.Axes) -> None:
        img = plt.imread(asset_path("lbf_apple.png"))
        for i in range(len(food_items.level)):
            food = _tree_slice(food_items, i)
            if food.eaten:
                continue
            cell_center = self._entity_position(food)
            self.draw_badge(food.level, cell_center, ax)
            imagebox = OffsetImage(img, zoom=self.icon_size / self.grid_size)
            ax.add_artist(
                AnnotationBbox(imagebox, cell_center, frameon=False, zorder=0)
            )

    def _entity_position(self, entity: Entity) -> tuple[float, float]:
        row, col = entity.position
        row = self.rows - row - 1
        return (
            (self.grid_size + 1) * col + self.grid_size // 2 + 1,
            (self.grid_size + 1) * row + self.grid_size // 2 + 1,
        )

    def draw_badge(
        self, level: int, anchor_point: tuple[float, float], ax: plt.Axes
    ) -> None:
        resolution = 6
        radius = self.grid_size / 6
        badge_center_x = anchor_point[0] + self.grid_size / 3 - 3
        badge_center_y = anchor_point[1] - self.grid_size / 3
        verts = []
        for i in range(resolution):
            angle = 2 * np.pi * i / resolution
            verts.append([
                radius * np.cos(angle) + badge_center_x + 1,
                radius * np.sin(angle) + 1 + badge_center_y,
            ])
        circle = plt.Polygon(
            verts, edgecolor="white", facecolor=self.GRID_COLOR
        )
        ax.add_patch(circle)
        fontsize = 10 if self.rows <= 10 else (6 if 10 < self.rows < 15 else 5)
        ax.annotate(
            str(level),
            xy=(badge_center_x + 1, badge_center_y + 1),
            color="white",
            ha="center",
            va="center",
            zorder=10,
            fontsize=fontsize,
            weight="bold",
        )


class MazePosition(NamedTuple):
    row: Any
    col: Any


@dataclass
class MazeState:
    agent_position: MazePosition
    target_position: MazePosition
    walls: Any
    action_mask: Any
    step_count: Any
    key: Any


class MazeEnvViewer(MazeViewer):
    AGENT = 2
    TARGET = 3
    COLORS = {0: [1, 1, 1], 1: [0, 0, 0], AGENT: [0, 1, 0], TARGET: [1, 0, 0]}

    def render(
        self, state: MazeState, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        return super().render(self._overlay_agent_and_target(state), save_path)

    def _overlay_agent_and_target(self, state: MazeState) -> Any:
        maze = state.walls.astype(int, copy=True)
        maze[tuple(state.agent_position)] = self.AGENT
        maze[tuple(state.target_position)] = self.TARGET
        return maze


@dataclass
class MMSTState:
    node_types: Any
    adj_matrix: Any
    connected_nodes: Any
    connected_nodes_index: Any
    nodes_to_connect: Any
    node_edges: Any
    positions: Any
    position_index: Any
    action_mask: Any
    finished_agents: Any
    step_count: Any
    key: Any


class MMSTViewer(MatplotlibViewer):
    GREY = (100 / 255, 100 / 255, 100 / 255)
    YELLOW = (200 / 255, 200 / 255, 0 / 255)
    BLACK = (0 / 255, 0 / 255, 0 / 255)
    BLUE = (50 / 255, 50 / 255, 160 / 255)

    def __init__(
        self, num_agents: int, name: str = "MMST", render_mode: str = "human"
    ) -> None:
        self.num_agents = num_agents
        np.random.seed(0)
        self.palette = [
            (
                np.random.randint(0, 192) / 255,
                np.random.randint(0, 192) / 255,
                np.random.randint(0, 192) / 255,
            )
            for _ in range(num_agents)
        ]
        super().__init__(name, render_mode)

    def render(
        self, state: MMSTState, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._draw_graph(state, ax)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def _draw_graph(
        self, state: MMSTState, ax: plt.Axes
    ) -> tuple[
        dict[tuple[int, ...], Line2D],
        list[tuple[plt.Circle, plt.Circle]],
        list[plt.Text],
    ]:
        num_nodes = state.adj_matrix.shape[0]
        positions = spring_layout(state.adj_matrix, num_nodes)
        node_radius = 0.05 * 5 / (5 + int(np.sqrt(num_nodes)))
        lines = {}
        for key, edge in self.build_edges(
            state.adj_matrix, state.connected_nodes
        ).items():
            (n1, n2), color = edge
            n1, n2 = int(n1), int(n2)
            line = Line2D(
                [positions[n1][0], positions[n2][0]],
                [positions[n1][1], positions[n2][1]],
                c=color,
                linewidth=2,
            )
            ax.add_artist(line)
            lines[key] = line

        circles = []
        labels = []
        for node in range(num_nodes):
            pos = np.where(state.nodes_to_connect == node)[0]
            fill_color = self.palette[pos[0]] if len(pos) == 1 else self.BLACK
            line_color = self.YELLOW if node in state.positions else self.BLUE
            circles.append(
                self.circle_fill(
                    positions[node],
                    line_color,
                    fill_color,
                    node_radius,
                    0.2 * node_radius,
                    ax,
                )
            )
            label = plt.Text(
                positions[node][0],
                positions[node][1],
                str(node),
                color="white",
                ha="center",
                va="center",
                weight="bold",
                zorder=200,
            )
            ax.add_artist(label)
            labels.append(label)

        ax.set_axis_off()
        ax.set_aspect(1)
        ax.relim()
        ax.autoscale_view()
        return lines, circles, labels

    def build_edges(
        self, adj_matrix: Any, connected_nodes: Any
    ) -> dict[tuple[int, ...], list[Any]]:
        def edge_id(n1: int, n2: int) -> tuple[int, ...]:
            return tuple(sorted((n1, n2)))

        edges: dict[tuple[int, ...], list[Any]] = {}
        connected_nodes = np.asarray(connected_nodes)
        row_indices, col_indices = np.nonzero(adj_matrix)
        for row, col in zip(row_indices, col_indices, strict=False):
            key = edge_id(int(row), int(col))
            if key not in edges:
                edges[key] = [(int(row), int(col)), self.GREY]

        for agent in range(self.num_agents):
            conn_group = connected_nodes[agent]
            len_conn = np.where(conn_group != -1)[0][-1]
            for i in range(len_conn):
                key = edge_id(conn_group[i], conn_group[i + 1])
                edges[key] = [
                    (conn_group[i], conn_group[i + 1]),
                    self.palette[agent],
                ]
        return edges

    def circle_fill(
        self,
        xy: tuple[float, float],
        line_color: tuple[float, float, float],
        fill_color: tuple[float, float, float],
        radius: float,
        thickness: float,
        ax: plt.Axes,
    ) -> tuple[plt.Circle, plt.Circle]:
        outer = plt.Circle(xy, radius, color=line_color, zorder=100)
        inner = plt.Circle(xy, radius - thickness, color=fill_color, zorder=100)
        ax.add_artist(outer)
        ax.add_artist(inner)
        return outer, inner


@dataclass
class Node:
    coordinates: Any
    demands: Any


@dataclass
class TimeWindow:
    start: Any
    end: Any


@dataclass
class PenalityCoeff:
    early: Any
    late: Any


@dataclass
class StateVehicle:
    local_times: Any
    capacities: Any
    positions: Any
    distances: Any
    time_penalties: Any


@dataclass
class MultiCVRPState:
    nodes: Node
    windows: TimeWindow
    coeffs: PenalityCoeff
    vehicles: StateVehicle
    order: Any
    step_count: Any
    action_mask: Any
    key: Any


class MultiCVRPViewer(CVRPViewer):
    ROUTE_NODES_SIZE = 100

    def __init__(
        self,
        name: str,
        num_vehicles: int,
        num_customers: int,
        map_max: int,
        render_mode: str = "human",
    ) -> None:
        super().__init__(name, num_customers + 1, render_mode)
        self._num_vehicles = num_vehicles
        self._num_customers = num_customers
        self._map_max = map_max
        self._cmap = plt.get_cmap(self.COLORMAP_NAME, self._num_vehicles + 1)

    def render(
        self, state: Any, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._prepare_figure(ax)
        self._add_tour(ax, state)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def _draw_route(
        self, ax: plt.Axes, coords: Any, col_id: int
    ) -> tuple[Quiver, PathCollection]:
        x, y = coords[:, 0], coords[:, 1]
        arrows = ax.quiver(
            x[:-1],
            y[:-1],
            x[1:] - x[:-1],
            y[1:] - y[:-1],
            scale_units="xy",
            angles="xy",
            scale=1,
            width=self.ARROW_WIDTH,
            headwidth=5,
            color=self._cmap(col_id),
        )
        nodes = ax.scatter(
            x, y, s=self.ROUTE_NODES_SIZE, color=self._cmap(col_id)
        )
        return arrows, nodes

    def _draw_all_routes(
        self, ax: plt.Axes, state: MultiCVRPState
    ) -> list[tuple[Quiver, PathCollection]]:
        routes = []
        if state.step_count > 0:
            for i in range(len(state.order)):
                coords = (
                    state.nodes.coordinates[state.order[i, : state.step_count]]
                    / self._map_max
                )
                for coords_route in self._group_tour(coords):
                    routes.append(self._draw_route(ax, coords_route, i))
        return routes

    def _add_tour(
        self, ax: plt.Axes, state: Any
    ) -> tuple[
        list[plt.Circle | plt.Rectangle], list[tuple[Quiver, PathCollection]]
    ]:
        x_coords = state.nodes.coordinates[:, 0] / self._map_max
        y_coords = state.nodes.coordinates[:, 1] / self._map_max
        depot = plt.Rectangle(
            (
                x_coords[0] - 0.5 * self.DEPOT_SIZE,
                y_coords[0] - 0.5 * self.DEPOT_SIZE,
            ),
            self.DEPOT_SIZE,
            self.DEPOT_SIZE,
            color=self.NODE_COLOUR,
        )
        ax.add_artist(depot)
        nodes: list[plt.Circle | plt.Rectangle] = [depot]
        for i in range(1, x_coords.shape[0]):
            node = plt.Circle(
                (x_coords[i], y_coords[i]),
                self.NODE_SIZE,
                color=self.NODE_COLOUR,
            )
            ax.add_artist(node)
            nodes.append(node)
        return nodes, self._draw_all_routes(ax, state)


class PacManPosition(NamedTuple):
    x: Any
    y: Any


class PacManObservation(NamedTuple):
    grid: Any
    player_locations: PacManPosition
    ghost_locations: Any
    power_up_locations: Any
    frightened_state_time: Any
    pellet_locations: Any
    action_mask: Any
    score: Any


class PacManViewer(MazeViewer):
    def render(
        self, state: PacManObservation, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        fig.suptitle(f"PacMan    Score: {int(state.score)}", size=15)
        ax.set_axis_off()
        ax.imshow(create_pac_man_grid_image(state))
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)


def create_pac_man_grid_image(observation: PacManObservation) -> Any:
    grid = np.asarray(1 - observation.grid, dtype=np.float32)
    layer_1 = grid * np.float32(0.0)
    layer_2 = grid * np.float32(0.0)
    layer_3 = grid * np.float32(0.6)
    player_loc = observation.player_locations
    ghost_pos = observation.ghost_locations
    pellets_loc = observation.power_up_locations
    is_scared = observation.frightened_state_time
    idx = observation.pellet_locations
    n = 3

    for i in range(len(pellets_loc)):
        p = pellets_loc[i]
        layer_1[p[1], p[0]] = 1.0
        layer_2[p[1], p[0]] = 0.8
        layer_3[p[1], p[0]] = 0.6

    layer_1[player_loc.x, player_loc.y] = 1
    layer_2[player_loc.x, player_loc.y] = 1
    layer_3[player_loc.x, player_loc.y] = 0

    cr = np.array([1, 1, 0, 1], dtype=np.float32)
    cg = np.array([0, 0.7, 1, 0.5], dtype=np.float32)
    cb = np.array([0, 1, 1, 0.0], dtype=np.float32)
    layers = (layer_1, layer_2, layer_3)

    if is_scared > 0:
        for i in range(4):
            y = ghost_pos[i][0]
            x = ghost_pos[i][1]
            layer_1[x, y] = 0
            layer_2[x, y] = 0
            layer_3[x, y] = 1
    else:
        for i in range(4):
            y = ghost_pos[i][0]
            x = ghost_pos[i][1]
            layer_1[x, y] = cr[i]
            layer_2[x, y] = cg[i]
            layer_3[x, y] = cb[i]

    layer_1, layer_2, layer_3 = layers
    layer_1[0, 0] = 0
    layer_2[0, 0] = 0
    layer_3[0, 0] = 0.6

    expand_rgb = np.kron(
        np.stack([layer_1, layer_2, layer_3], axis=-1), np.ones((n, n, 1))
    )
    layer_1 = expand_rgb[:, :, 0]
    layer_2 = expand_rgb[:, :, 1]
    layer_3 = expand_rgb[:, :, 2]

    for i in range(len(idx)):
        # Jumanji v1.1.1 compares the sum method object instead of calling it.
        # Keep the typo so EnvPool render remains bitwise with the oracle.
        if cast(Any, np.array(idx[i]).sum) != 0:
            loc = idx[i]
            c = loc[1] * n + 1
            r = loc[0] * n + 1
            layer_1[c, r] = 1.0
            layer_2[c, r] = 0.8
            layer_3[c, r] = 0.6

    if is_scared > 0:
        for i in range(4):
            y = ghost_pos[i][0]
            x = ghost_pos[i][1]
            c = x * n + 1
            r = y * n + 1
            layer_1[x * n + 1, y * n + 1] = 0
            layer_2[x * n + 1, y * n + 1] = 0
            layer_3[x * n + 1, y * n + 1] = 1
            layer_1[c - 1, r - 1] = 0.0
            layer_2[c - 1, r - 1] = 0.0
            layer_3[c - 1, r - 1] = 0.0
            layer_1[c - 1, r + 1] = 0.0
            layer_2[c - 1, r + 1] = 0.0
            layer_3[c - 1, r + 1] = 0.0
            layer_1[c, r + 1] = 1
            layer_2[c, r + 1] = 0.6
            layer_3[c, r + 1] = 0.2
            layer_1[c, r - 1] = 1
            layer_2[c, r - 1] = 0.6
            layer_3[c, r - 1] = 0.2
    else:
        for i in range(4):
            y = ghost_pos[i][0]
            x = ghost_pos[i][1]
            c = x * n + 1
            r = y * n + 1
            layer_1[c, r] = cr[i]
            layer_2[c, r] = cg[i]
            layer_3[c, r] = cb[i]
            layer_1[c - 1, r - 1] = 0.0
            layer_2[c - 1, r - 1] = 0.0
            layer_3[c - 1, r - 1] = 0.0
            layer_1[c - 1, r + 1] = 0.0
            layer_2[c - 1, r + 1] = 0.0
            layer_3[c - 1, r + 1] = 0.0
            layer_1[c, r + 1] = 1
            layer_2[c, r + 1] = 1
            layer_3[c, r + 1] = 1
            layer_1[c, r - 1] = 1
            layer_2[c, r - 1] = 1
            layer_3[c, r - 1] = 1

    for i in range(len(pellets_loc)):
        p = pellets_loc[i]
        layer_1[p[1] * n + 2, p[0] * n + 1] = 1
        layer_2[p[1] * n + 1, p[0] * n + 1] = 0.8
        layer_3[p[1] * n + 1, p[0] * n + 1] = 0.6

    layer_1[player_loc.x * n + 1, player_loc.y * n + 1] = 1
    layer_2[player_loc.x * n + 1, player_loc.y * n + 1] = 1
    layer_3[player_loc.x * n + 1, player_loc.y * n + 1] = 0
    return np.stack([layer_1, layer_2, layer_3], axis=-1)


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class RobotPosition(NamedTuple):
    x: Any
    y: Any


class RobotAgent(NamedTuple):
    position: RobotPosition
    direction: Any
    is_carrying: Any


class Shelf(NamedTuple):
    position: RobotPosition
    is_requested: Any


@dataclass
class RobotWarehouseState:
    grid: Any
    agents: RobotAgent
    shelves: Shelf
    request_queue: Any
    step_count: Any
    action_mask: Any
    key: Any


class RobotWarehouseViewer(MatplotlibViewer):
    SHELF_PADDING = 2
    GRID_COLOR = (0, 0, 0)
    SHELF_COLOR = (72 / 255.0, 61 / 255.0, 139 / 255.0)
    SHELF_REQ_COLOR = (0, 128 / 255.0, 128 / 255.0)
    AGENT_COLOR = (1, 140 / 255.0, 0)
    AGENT_LOADED_COLOR = (1, 0, 0)
    AGENT_DIR_COLOR = (0, 0, 0)
    GOAL_COLOR = (60 / 255.0, 60 / 255.0, 60 / 255.0)

    def __init__(
        self,
        grid_size: tuple[int, int],
        goals: Any,
        name: str = "RobotWarehouse",
        render_mode: str = "human",
    ) -> None:
        self.goals = goals
        self.rows, self.cols = grid_size
        self.grid_size = 30
        self.icon_size = 20
        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 1 + self.rows * (self.grid_size + 1)
        super().__init__(name, render_mode)

    def render(
        self, state: RobotWarehouseState, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._prepare_figure(ax)
        self._draw_state(ax, state)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def _prepare_figure(self, ax: plt.Axes) -> None:
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.patch.set_alpha(0.0)
        ax.set_axis_off()
        ax.set_aspect("equal", "box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    def _draw_state(self, ax: plt.Axes, state: RobotWarehouseState) -> None:
        self.n_agents = state.agents.position.x.shape[0]
        self.n_shelves = state.shelves.position.x.shape[0]
        self._draw_grid(ax)
        self._draw_goals(ax)
        self._draw_shelves(ax, state.shelves)
        self._draw_agents(ax, state.agents)

    def _draw_grid(self, ax: plt.Axes) -> None:
        lines = []
        for r in range(self.rows + 1):
            lines.append([
                (0, (self.grid_size + 1) * r + 1),
                (
                    (self.grid_size + 1) * self.cols,
                    (self.grid_size + 1) * r + 1,
                ),
            ])
        for c in range(self.cols + 1):
            lines.append([
                ((self.grid_size + 1) * c + 1, 0),
                (
                    (self.grid_size + 1) * c + 1,
                    (self.grid_size + 1) * self.rows,
                ),
            ])
        ax.add_collection(LineCollection(lines, colors=(self.GRID_COLOR,)))

    def _draw_goals(self, ax: plt.Axes) -> None:
        for goal in self.goals:
            x, y = goal
            y = self.rows - y - 1
            ax.fill(
                [
                    x * (self.grid_size + 1) + 1,
                    (x + 1) * (self.grid_size + 1),
                    (x + 1) * (self.grid_size + 1),
                    x * (self.grid_size + 1) + 1,
                ],
                [
                    y * (self.grid_size + 1) + 1,
                    y * (self.grid_size + 1) + 1,
                    (y + 1) * (self.grid_size + 1),
                    (y + 1) * (self.grid_size + 1),
                ],
                color=self.GOAL_COLOR,
                alpha=1,
            )

    def _draw_shelves(self, ax: plt.Axes, shelves: Shelf) -> None:
        for shelf_id in range(self.n_shelves):
            shelf = _tree_slice(shelves, shelf_id)
            y, x = shelf.position.x, shelf.position.y
            y = self.rows - y - 1
            shelf_color = (
                self.SHELF_REQ_COLOR if shelf.is_requested else self.SHELF_COLOR
            )
            x_points = [
                (self.grid_size + 1) * x + self.SHELF_PADDING + 1,
                (self.grid_size + 1) * (x + 1) - self.SHELF_PADDING,
                (self.grid_size + 1) * (x + 1) - self.SHELF_PADDING,
                (self.grid_size + 1) * x + self.SHELF_PADDING + 1,
            ]
            y_points = [
                (self.grid_size + 1) * y + self.SHELF_PADDING + 1,
                (self.grid_size + 1) * y + self.SHELF_PADDING + 1,
                (self.grid_size + 1) * (y + 1) - self.SHELF_PADDING,
                (self.grid_size + 1) * (y + 1) - self.SHELF_PADDING,
            ]
            ax.fill(x_points, y_points, color=shelf_color)

    def _draw_agents(self, ax: plt.Axes, agents: RobotAgent) -> None:
        radius = self.grid_size / 3
        resolution = 6
        for agent_id in range(self.n_agents):
            agent = _tree_slice(agents, agent_id)
            row, col = agent.position.x, agent.position.y
            row = self.rows - row - 1
            x_center = (self.grid_size + 1) * col + self.grid_size // 2 + 1
            y_center = (self.grid_size + 1) * row + self.grid_size // 2 + 1
            verts = []
            for i in range(resolution):
                angle = 2 * np.pi * i / resolution
                verts.append([
                    radius * np.cos(angle) + x_center + 1,
                    radius * np.sin(angle) + 1 + y_center,
                ])
            facecolor = (
                self.AGENT_LOADED_COLOR
                if agent.is_carrying
                else self.AGENT_COLOR
            )
            circle = plt.Polygon(verts, edgecolor="none", facecolor=facecolor)
            ax.add_patch(circle)

            agent_dir = agent.direction
            x_dir = (
                x_center
                + (radius if agent_dir == Direction.RIGHT.value else 0)
                - (radius if agent_dir == Direction.LEFT.value else 0)
            )
            y_dir = (
                y_center
                + (radius if agent_dir == Direction.UP.value else 0)
                - (radius if agent_dir == Direction.DOWN.value else 0)
            )
            ax.plot(
                [x_center, x_dir],
                [y_center, y_dir],
                color=self.AGENT_DIR_COLOR,
                linewidth=2,
            )


class SnakePosition(NamedTuple):
    row: Any
    col: Any


@dataclass
class SnakeState:
    body: Any
    body_state: Any
    head_position: SnakePosition
    tail: Any
    fruit_position: SnakePosition
    length: Any
    step_count: Any
    action_mask: Any
    key: Any


class SnakeViewer(MatplotlibViewer):
    def render(
        self, state: SnakeState, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        self._draw(ax, state)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def _draw(self, ax: plt.Axes, state: SnakeState) -> None:
        ax.clear()
        self._draw_board(ax, state)
        for patch in self._create_entities(state):
            ax.add_patch(patch)

    def _draw_board(self, ax: plt.Axes, state: SnakeState) -> None:
        num_rows, num_cols = state.body_state.shape[-2:]
        ax.axis("off")
        ax.plot([0, 0], [0, num_rows], "-k", lw=2)
        ax.plot([0, num_cols], [num_rows, num_rows], "-k", lw=2)
        ax.plot([num_cols, num_cols], [num_rows, 0], "-k", lw=2)
        ax.plot([num_cols, 0], [0, 0], "-k", lw=2)

    def _create_entities(self, state: SnakeState) -> list[Patch]:
        num_rows, num_cols = state.body_state.shape[-2:]
        linewidth = (
            min(
                n * size
                for n, size in zip(
                    (num_rows, num_cols), self.figure_size, strict=False
                )
            )
            / 44.0
        )
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "", ["yellowgreen", "forestgreen"]
        )
        patches: list[Patch] = []
        for row in range(num_rows):
            for col in range(num_cols):
                if state.body_state[row, col]:
                    patches.append(
                        plt.Rectangle(
                            (col, num_rows - 1 - row),
                            1,
                            1,
                            edgecolor=cmap(1),
                            facecolor=cmap(
                                state.body_state[row, col] / state.length
                            ),
                            fill=True,
                            lw=linewidth,
                        )
                    )
        patches.append(
            plt.Circle(
                (
                    state.head_position[1] + 0.5,
                    num_rows - 1 - state.head_position[0] + 0.5,
                ),
                0.3,
                edgecolor=cmap(0.5),
                facecolor=cmap(0),
                fill=True,
                lw=linewidth,
            )
        )
        patches.append(
            plt.Circle(
                (
                    state.fruit_position[1] + 0.5,
                    num_rows - 1 - state.fruit_position[0] + 0.5,
                ),
                0.2,
                edgecolor="brown",
                facecolor="lightcoral",
                fill=True,
                lw=linewidth,
            )
        )
        return patches


class BoxViewer(MatplotlibViewer):
    def __init__(
        self,
        name: str,
        grid_combine: Callable[..., Any],
        render_mode: str = "human",
    ) -> None:
        self.grid_combine = grid_combine
        image_names = [
            "floor",
            "wall",
            "box_target",
            "agent",
            "box",
            "agent_on_target",
            "box_on_target",
        ]
        self.images = [
            Image.open(asset_path(f"sokoban_{image_name}.png"))
            for image_name in image_names
        ]
        super().__init__(name, render_mode)

    def render(
        self, state: Any, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._add_grid_image(state, ax)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def _add_grid_image(self, state: Any, ax: plt.Axes) -> None:
        self._draw_grid(
            self.grid_combine(state.variable_grid, state.fixed_grid), ax
        )
        ax.set_axis_off()
        ax.set_aspect(1)
        ax.relim()
        ax.autoscale_view()

    def _draw_grid(self, grid: Any, ax: plt.Axes) -> None:
        cols, rows = grid.shape
        for col in range(cols):
            for row in range(rows):
                ax.imshow(
                    self.images[int(grid[row, col])],
                    extent=(col, col + 1, 9 - row, 10 - row),
                )


@dataclass
class TSPState:
    coordinates: Any
    position: Any
    visited_mask: Any
    trajectory: Any
    num_visited: Any
    key: Any


class TSPViewer(MatplotlibViewer):
    NODE_COLOUR = "dimgray"
    NODE_SIZE = 150
    ARROW_WIDTH = 0.004

    def render(
        self, state: TSPState, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._prepare_figure(ax)
        self._add_tour(ax, state)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def _prepare_figure(self, ax: plt.Axes) -> None:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(
            plt.imread(asset_path("city_map.jpeg")),
            extent=(0, 1, 0, 1),
        )

    def _draw_route(
        self, ax: plt.Axes, state: TSPState
    ) -> tuple[Quiver, PathCollection]:
        xs, ys = state.coordinates[state.trajectory[: state.num_visited]].T
        route = ax.quiver(
            xs[:-1],
            ys[:-1],
            xs[1:] - xs[:-1],
            ys[1:] - ys[:-1],
            scale_units="xy",
            angles="xy",
            scale=1,
            width=self.ARROW_WIDTH,
            headwidth=5,
        )
        route_nodes = ax.scatter(xs, ys, s=self.NODE_SIZE, color="black")
        return route, route_nodes

    def _add_tour(
        self, ax: plt.Axes, state: TSPState
    ) -> tuple[PathCollection, Quiver, PathCollection]:
        x_coords, y_coords = state.coordinates.T
        cities = ax.scatter(
            x_coords, y_coords, s=self.NODE_SIZE, color=self.NODE_COLOUR
        )
        route, route_nodes = self._draw_route(ax, state)
        return cities, route, route_nodes
