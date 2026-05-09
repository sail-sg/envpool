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

# Copyright 2023 InstaDeep Ltd. All rights reserved.
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

from typing import Optional, Sequence, Tuple

import matplotlib.animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray

from jumanji.environments.logic.sliding_tile_puzzle.types import State
from jumanji.viewer import MatplotlibViewer


class SlidingTilePuzzleViewer(MatplotlibViewer[State]):
    EMPTY_TILE_COLOR = "#ccc0b3"

    def __init__(self, name: str = "SlidingTilePuzzle", render_mode: str = "human") -> None:
        """Viewer for the Sliding Tile Puzzle environment.

        Args:
            name: the window name to be used when initialising the window.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        self._color_map = mcolors.LinearSegmentedColormap.from_list("", ["white", "blue"])
        super().__init__(name, render_mode)

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Renders the current state of the game puzzle.

        Args:
            state: is the current game state to be rendered.
            save_path: Optional path to save the rendered environment image to.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        self.draw_puzzle(ax, state)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

        return self._display(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Creates an animated gif of the sliding tiles puzzle game based on a sequence of states.

        Args:
            states: is a list of `State` objects representing the sequence of game states.
            interval: the delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation object that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig, ax = self._get_fig_ax(name_suffix="_animation", show=False)
        plt.close(fig=fig)

        def make_frame(state: State) -> Tuple[Artist]:
            self.draw_puzzle(ax, state)
            return (ax,)

        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=states,
            interval=interval,
        )

        if save_path:
            self._animation.save(save_path)

        return self._animation

    def draw_puzzle(self, ax: plt.Axes, state: State) -> None:
        """Draw the game puzzle with the current state.

        Args:
            ax: the axis to draw the puzzle on.
            state: the current state of the game.
        """
        ax.clear()
        grid_size = state.puzzle.shape[0]
        ax.set_xticks(np.arange(-0.5, grid_size - 1, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size - 1, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=2)

        # Render the puzzle
        for row in range(grid_size):
            for col in range(grid_size):
                tile_value = state.puzzle[row, col]
                if tile_value == 0:
                    # Render the empty tile
                    rect = plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color=self.EMPTY_TILE_COLOR)
                    ax.add_patch(rect)
                else:
                    # Render the numbered tile
                    ax.text(col, row, str(tile_value), ha="center", va="center")

        # Show the image of the puzzle.
        ax.imshow(state.puzzle, cmap=self._color_map)
