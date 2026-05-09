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

from typing import ClassVar, Dict, Optional, Sequence, Tuple

import jax.numpy as jnp
import matplotlib.animation
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from numpy.typing import NDArray

from jumanji.environments.logic.game_2048.types import State
from jumanji.viewer import MatplotlibViewer


class Game2048Viewer(MatplotlibViewer[State]):
    COLORS: ClassVar[Dict[int | str, str]] = {
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
        """Viewer for the 2048 environment.

        Args:
            name: the window name to be used when initialising the window.
            board_size: size of the board.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        self._board_size = board_size
        super().__init__(name, render_mode)

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Renders the current state of the game board.

        Args:
            state: is the current game state to be rendered.
            save_path: Optional path to save the rendered environment image to.
        """
        self._clear_display()
        # Get the figure and axes for the game board.
        fig, ax = self._get_fig_ax()
        # Set the figure title to display the current score.
        fig.suptitle(f"2048    Score: {int(state.score)}", size=20)
        # Draw the game board
        self.draw_board(ax, state)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

        return self._display(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Creates an animated gif of the 2048 game board based on the sequence of game states.

        Args:
            states: is a list of `State` objects representing the sequence of game states.
            interval: the delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation object that can be saved as a GIF, MP4, or rendered with HTML.
        """
        # Set up the figure and axes for the game board.
        fig, ax = self._get_fig_ax(name_suffix="_animation", show=False)
        plt.close(fig=fig)
        ax.set_title("2048    Score: 0", size=20)

        # Define a function to animate a single game state.
        def make_frame(state: State) -> Tuple[Artist]:
            self.draw_board(ax, state)
            ax.set_title(f"2048    Score: {int(state.score)}", size=20)
            return (ax,)

        # Create the animation object.
        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=states,
            interval=interval,
        )

        # Save the animation as a gif.
        if save_path:
            self._animation.save(save_path)

        return self._animation

    def render_tile(self, tile_value: int, ax: plt.Axes, row: int, col: int) -> None:
        """Renders a single tile on the game board.

        Args:
            tile_value: is the value of the tile on the game board.
            ax: the axes on which to draw the tile.
            row: the row index of the tile on the board.
            col: the col index of the tile on the board.
        """
        # Set the background color of the tile based on its value.
        if tile_value <= 16384:
            rect = plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color=self.COLORS[int(tile_value)])
        else:
            rect = plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color=self.COLORS["other"])
        ax.add_patch(rect)

        if tile_value in [2, 4]:
            color = self.COLORS["dark_text"]
            size = 30
        elif tile_value < 1024:
            color = self.COLORS["light_text"]
            size = 30
        elif 1024 <= tile_value < 16384:
            color = self.COLORS["light_text"]
            size = 25
        else:  # tile_value >= 16384:
            color = self.COLORS["light_text"]
            size = 20
        # Empty tiles (each corresponding to the number 1) are not rendered.
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

    def draw_board(self, ax: plt.Axes, state: State) -> None:
        """Draw the game board with the current state.

        Args:
            ax: the axis to draw the board on.
            state: the current state of the game.
        """
        ax.clear()
        ax.set_xticks(jnp.arange(-0.5, 4 - 1, 1))
        ax.set_yticks(jnp.arange(-0.5, 4 - 1, 1))
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
        # Get the tile values from the exponents.
        board = jnp.power(2, state.board)

        # Iterate through each cell and render tiles.
        for row in range(0, self._board_size):
            for col in range(0, self._board_size):
                self.render_tile(tile_value=board[row, col], ax=ax, row=row, col=col)

        # Show the image of the board.
        ax.imshow(board)

        # Draw the grid lines.
        ax.grid(color=self.COLORS["edge"], linestyle="-", linewidth=7)
