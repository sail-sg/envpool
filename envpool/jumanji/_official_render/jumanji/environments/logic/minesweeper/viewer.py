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

from typing import List, Optional, Sequence, Tuple

import chex
import jax.numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from numpy.typing import NDArray

from jumanji.environments.logic.minesweeper.constants import (
    DEFAULT_COLOR_MAPPING,
    UNEXPLORED_ID,
)
from jumanji.environments.logic.minesweeper.types import State
from jumanji.environments.logic.minesweeper.utils import explored_mine
from jumanji.viewer import MatplotlibViewer


class MinesweeperViewer(MatplotlibViewer[State]):
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        render_mode: str = "human",
        color_mapping: Optional[List[str]] = None,
    ):
        """
        Args:
            num_rows: number of rows, i.e. height of the board.
            num_cols: number of columns, i.e. width of the board.
            render_mode: Figure rendering mode, either "human" or "rgb_array".
            color_mapping: colors used in rendering the cells in `Minesweeper`.
                Defaults to `DEFAULT_COLOR_MAPPING`.
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cmap = color_mapping or DEFAULT_COLOR_MAPPING
        super().__init__(f"{num_rows}x{num_cols} Minesweeper", render_mode)

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Render the given environment state using matplotlib.

        Args:
            state: environment state to be rendered.
            save_path: Optional path to save the rendered environment image to.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        fig.suptitle(self._name)
        self._draw(ax, state)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

        return self._display(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> FuncAnimation:
        """Create an animation from a sequence of environment states.

        Args:
            states: sequence of environment states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation object that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig, ax = self._get_fig_ax(name_suffix="_animation", show=False)
        plt.close(fig=fig)
        ax.set_title(self._name)

        def make_frame(state: State) -> Tuple[Artist]:
            self._draw(ax, state)
            return (ax,)

        # Create the animation object.
        self._animation = FuncAnimation(
            fig,
            make_frame,
            frames=states,
            interval=interval,
            save_count=len(states),
        )

        # Save the animation as a GIF.
        if save_path:
            self._animation.save(save_path)

        return self._animation

    def _draw(self, ax: plt.Axes, state: State) -> None:
        ax.clear()
        ax.set_xticks(jnp.arange(-0.5, self.num_cols - 1, 1))
        ax.set_yticks(jnp.arange(-0.5, self.num_rows - 1, 1))
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
        background = jnp.ones_like(state.board)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                background = self._render_grid_square(
                    state=state, ax=ax, i=i, j=j, background=background
                )
        ax.imshow(background, cmap="gray", vmin=0, vmax=1)
        ax.grid(color="black", linestyle="-", linewidth=2)

    def _render_grid_square(
        self, state: State, ax: plt.Axes, i: int, j: int, background: chex.Array
    ) -> chex.Array:
        board_value = state.board[i, j]
        if board_value != UNEXPLORED_ID:
            if explored_mine(state=state, action=jnp.array([i, j], dtype=jnp.int32)):
                background = background.at[i, j].set(0)
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
        return background
