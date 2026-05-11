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
"""Flat renderers for Jumanji logic environments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage
from matplotlib.text import Text
from numpy.typing import NDArray

from envpool.jumanji._official_render.base import (
    MatplotlibViewer,
    spring_layout,
)


@dataclass
class Game2048State:
    board: Any
    step_count: Any
    action_mask: Any
    score: Any
    key: Any


@dataclass
class GraphColoringState:
    adj_matrix: Any
    colors: Any
    current_node_index: Any
    action_mask: Any
    key: Any


@dataclass
class MinesweeperState:
    board: Any
    step_count: Any
    flat_mine_locations: Any
    key: Any


@dataclass
class RubiksCubeState:
    cube: Any
    step_count: Any
    key: Any


@dataclass
class SlidingTilePuzzleState:
    puzzle: Any
    empty_tile_position: Any
    key: Any
    step_count: Any


@dataclass
class SudokuState:
    board: Any
    action_mask: Any | None = None
    key: Any | None = None


class Game2048Viewer(MatplotlibViewer):
    COLORS: ClassVar[dict[int | str, str]] = {
        1: "#ccc0b3",
        2: "#eee4da",
        4: "#ede0c8",
        8: "#f59563",
        16: "#f59563",
        32: "#f67c5f",
        64: "#f65e3b",
        128: "#edcf72",
        256: "#edcc61",
        512: "#edc651",
        1024: "#eec744",
        2048: "#ecc22e",
        4096: "#b784ab",
        8192: "#b784ab",
        16384: "#aa60a6",
        "other": "#f8251d",
        "light_text": "#f9f6f2",
        "dark_text": "#766d64",
        "edge": "#bbada0",
        "bg": "#faf8ef",
    }

    def __init__(
        self,
        name: str = "2048",
        board_size: int = 4,
        render_mode: str = "human",
    ) -> None:
        self._board_size = board_size
        super().__init__(name, render_mode)

    def render(
        self, state: Game2048State, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        fig.suptitle(f"2048    Score: {int(state.score)}", size=20)
        self.draw_board(ax, state)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def render_tile(
        self, tile_value: int, ax: plt.Axes, row: int, col: int
    ) -> None:
        if tile_value <= 16384:
            color = self.COLORS[int(tile_value)]
        else:
            color = self.COLORS["other"]
        ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color=color))

        if tile_value in [2, 4]:
            color, size = self.COLORS["dark_text"], 30
        elif tile_value < 1024:
            color, size = self.COLORS["light_text"], 30
        elif 1024 <= tile_value < 16384:
            color, size = self.COLORS["light_text"], 25
        else:
            color, size = self.COLORS["light_text"], 20
        if tile_value != 1:
            ax.text(
                col,
                row,
                str(tile_value),
                color=color,
                ha="center",
                va="center",
                size=size,
                weight="bold",
            )

    def draw_board(self, ax: plt.Axes, state: Game2048State) -> None:
        ax.clear()
        ax.set_xticks(np.arange(-0.5, 4 - 1, 1))
        ax.set_yticks(np.arange(-0.5, 4 - 1, 1))
        ax.tick_params(
            top=False,
            bottom=False,
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
            labeltop=False,
            labelright=False,
        )
        board = np.power(2, state.board)
        for row in range(self._board_size):
            for col in range(self._board_size):
                self.render_tile(board[row, col], ax, row, col)
        ax.imshow(board)
        ax.grid(color=self.COLORS["edge"], linestyle="-", linewidth=7)


class GraphColoringViewer(MatplotlibViewer):
    def render(
        self, state: GraphColoringState, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        self._set_params(state)
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._prepare_figure(ax, state)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def _set_params(self, state: GraphColoringState) -> None:
        self.num_nodes = state.adj_matrix.shape[0]
        self.node_scale = 5 + int(np.sqrt(self.num_nodes))
        self._color_mapping = self._create_color_mapping(self.num_nodes)

    def _prepare_figure(self, ax: plt.Axes, state: GraphColoringState) -> None:
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        pos = spring_layout(state.adj_matrix, self.num_nodes)
        self._render_edges(ax, pos, state.adj_matrix, self.num_nodes)
        self._render_nodes(ax, pos, state.colors)

    def _render_nodes(
        self, ax: plt.Axes, pos: list[tuple[float, float]], colors: Any
    ) -> None:
        node_radius = 0.05 * 5 / self.node_scale
        for i, (x, y) in enumerate(pos):
            ax.add_artist(
                plt.Circle(
                    (x, y),
                    node_radius,
                    color=self._color_mapping[colors[i]],
                    fill=True,
                    zorder=100,
                )
            )
            ax.add_artist(
                plt.Text(
                    x,
                    y,
                    str(i),
                    color="white",
                    ha="center",
                    va="center",
                    weight="bold",
                    zorder=200,
                )
            )

    def _render_edges(
        self,
        ax: plt.Axes,
        pos: list[tuple[float, float]],
        adj_matrix: Any,
        num_nodes: int,
    ) -> None:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                ax.add_artist(
                    plt.Line2D(
                        [pos[i][0], pos[j][0]],
                        [pos[i][1], pos[j][1]],
                        color=self._color_mapping[-1],
                        linewidth=0.5,
                        visible=adj_matrix[i, j],
                    )
                )

    def _create_color_mapping(
        self,
        num_nodes: int,
    ) -> list[tuple[float, float, float, float]]:
        colormap_indices = np.arange(0, 1, 1 / num_nodes)
        colormap = plt.get_cmap("hsv", num_nodes + 1)
        color_mapping = [colormap(float(index)) for index in colormap_indices]
        color_mapping.append((0.0, 0.0, 0.0, 1.0))
        return color_mapping


UNEXPLORED_ID = -1
MINESWEEPER_COLORS = [
    "orange",
    "blue",
    "green",
    "red",
    "purple",
    "maroon",
    "teal",
    "black",
    "gray",
]


def _explored_mine(state: MinesweeperState, action: np.ndarray) -> bool:
    row, col = np.asarray(action, dtype=np.int64)
    index = int(col + row * state.board.shape[-1])
    locations = np.asarray(state.flat_mine_locations, dtype=np.int64).reshape(
        -1
    )
    return bool(index in set(locations.tolist()))


class MinesweeperViewer(MatplotlibViewer):
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        render_mode: str = "human",
        color_mapping: list[str] | None = None,
    ) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cmap = color_mapping or MINESWEEPER_COLORS
        super().__init__(f"{num_rows}x{num_cols} Minesweeper", render_mode)

    def render(
        self, state: MinesweeperState, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        fig.suptitle(self._name)
        self._draw(ax, state)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def _draw(self, ax: plt.Axes, state: MinesweeperState) -> None:
        ax.clear()
        ax.set_xticks(np.arange(-0.5, self.num_cols - 1, 1))
        ax.set_yticks(np.arange(-0.5, self.num_rows - 1, 1))
        ax.tick_params(
            top=False,
            bottom=False,
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
            labeltop=False,
            labelright=False,
        )
        background = np.ones_like(state.board)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                board_value = state.board[i, j]
                if board_value == UNEXPLORED_ID:
                    continue
                if _explored_mine(state, np.array([i, j], dtype=np.int32)):
                    background = np.array(background, copy=True)
                    background[i, j] = 0
                else:
                    ax.text(
                        j,
                        i,
                        str(board_value),
                        color=self.cmap[board_value],
                        ha="center",
                        va="center",
                        fontsize="xx-large",
                    )
        ax.imshow(background, cmap="gray", vmin=0, vmax=1)
        ax.grid(color="black", linestyle="-", linewidth=2)


class Face(Enum):
    UP = 0
    FRONT = 1
    RIGHT = 2
    BACK = 3
    LEFT = 4
    DOWN = 5


class RubiksCubeViewer(MatplotlibViewer):
    def __init__(
        self,
        sticker_colors: list[str],
        cube_size: int,
        render_mode: str = "human",
    ) -> None:
        self.cube_size = cube_size
        self.sticker_colors_cmap = matplotlib.colors.ListedColormap(
            sticker_colors
        )
        super().__init__(
            f"{cube_size}x{cube_size}x{cube_size} Rubik's Cube",
            render_mode,
        )

    def render(
        self, state: RubiksCubeState, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        self._draw(ax, state)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def _get_fig_ax(
        self,
        name_suffix: str | None = None,
        show: bool = True,
        padding: float = 0.05,
        **fig_kwargs: Any,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        del padding
        fig_name = (
            self._name if name_suffix is None else self._name + name_suffix
        )
        if plt.fignum_exists(fig_name):
            fig = plt.figure(fig_name)
            ax = fig.get_axes()
        else:
            fig, axes = plt.subplots(
                nrows=3,
                ncols=2,
                figsize=self.figure_size,
                num=fig_name,
                **fig_kwargs,
            )
            fig.suptitle(self._name)
            ax = list(axes.flatten())
            plt.tight_layout()
            plt.axis("off")
            if not plt.isinteractive() and show:
                fig.show()
        return fig, ax

    def _draw(
        self, ax: list[plt.Axes], state: RubiksCubeState
    ) -> list[AxesImage]:
        images = []
        for i, face in enumerate(Face):
            ax[i].clear()
            ax[i].set_title(label=f"{face}")
            ax[i].set_xticks(np.arange(-0.5, self.cube_size - 1, 1))
            ax[i].set_yticks(np.arange(-0.5, self.cube_size - 1, 1))
            ax[i].tick_params(
                top=False,
                bottom=False,
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
                labeltop=False,
                labelright=False,
            )
            image = ax[i].imshow(
                state.cube[i],
                cmap=self.sticker_colors_cmap,
                vmin=0,
                vmax=len(Face) - 1,
            )
            images.append(image)
            ax[i].grid(color="black", linestyle="-", linewidth=2)
        return images


class SlidingTilePuzzleViewer(MatplotlibViewer):
    EMPTY_TILE_COLOR = "#ccc0b3"

    def __init__(
        self, name: str = "SlidingTilePuzzle", render_mode: str = "human"
    ) -> None:
        self._color_map = mcolors.LinearSegmentedColormap.from_list(
            "", ["white", "blue"]
        )
        super().__init__(name, render_mode)

    def render(
        self, state: SlidingTilePuzzleState, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        self.draw_puzzle(ax, state)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def draw_puzzle(self, ax: plt.Axes, state: SlidingTilePuzzleState) -> None:
        ax.clear()
        grid_size = state.puzzle.shape[0]
        ax.set_xticks(np.arange(-0.5, grid_size - 1, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size - 1, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=2)

        for row in range(grid_size):
            for col in range(grid_size):
                tile_value = state.puzzle[row, col]
                if tile_value == 0:
                    ax.add_patch(
                        plt.Rectangle(
                            (col - 0.5, row - 0.5),
                            1,
                            1,
                            color=self.EMPTY_TILE_COLOR,
                        )
                    )
                else:
                    ax.text(col, row, str(tile_value), ha="center", va="center")
        ax.imshow(state.puzzle, cmap=self._color_map)


BOARD_WIDTH = 9


class SudokuViewer(MatplotlibViewer):
    def render(
        self, state: SudokuState, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        self._draw(ax, state)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def _draw_board(self, ax: plt.Axes) -> None:
        ax.axis("off")
        linewidth = 2.5
        ax.plot([0, 0], [0, BOARD_WIDTH], "-k", lw=linewidth)
        ax.plot(
            [0, BOARD_WIDTH], [BOARD_WIDTH, BOARD_WIDTH], "-k", lw=linewidth
        )
        ax.plot(
            [BOARD_WIDTH, BOARD_WIDTH], [BOARD_WIDTH, 0], "-k", lw=linewidth
        )
        ax.plot([BOARD_WIDTH, 0], [0, 0], "-k", lw=linewidth)
        for i in range(1, BOARD_WIDTH):
            if i % int(BOARD_WIDTH**0.5) == 0:
                linewidth = 2.5
            else:
                linewidth = 1
            ax.add_line(
                matplotlib.lines.Line2D(
                    [0, BOARD_WIDTH],
                    [i, i],
                    color="k",
                    linewidth=linewidth,
                )
            )
            ax.add_line(
                matplotlib.lines.Line2D(
                    [i, i],
                    [0, BOARD_WIDTH],
                    color="k",
                    linewidth=linewidth,
                )
            )

    def _draw(self, ax: plt.Axes, state: SudokuState) -> list[list[Text]]:
        ax.clear()
        self._draw_board(ax)
        return self._draw_figures(ax, state)

    def _draw_figures(
        self, ax: plt.Axes, state: SudokuState
    ) -> list[list[Text]]:
        board = state.board
        artists: list[list[Text]] = []
        for i in range(board.shape[0]):
            artists.append([])
            for j in range(board.shape[1]):
                element = board[i, j]
                text = "" if element == -1 else str(element + 1)
                artist = Text(
                    x=j + 0.5,
                    y=board.shape[0] - i - 0.5,
                    text=text,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=16,
                )
                ax.add_artist(artist)
                artists[-1].append(artist)
        return artists
