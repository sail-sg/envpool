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

from typing import Any, Dict, Optional, Sequence, Tuple

import chex
import matplotlib.animation
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray

from jumanji.environments.packing.flat_pack.types import State
from jumanji.viewer import MatplotlibViewer


class FlatPackViewer(MatplotlibViewer[State]):
    def __init__(self, name: str, num_blocks: int, render_mode: str = "human") -> None:
        """Viewer for a `FlatPack` environment.

        Args:
            name: the window name to be used when initialising the window.
            num_blocks: number of blocks in the environment.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        # Create a color for each block.
        colormap_indices = np.arange(0, 1, 1 / num_blocks)
        colormap = plt.get_cmap("hsv", num_blocks + 1)

        self.colors = [(1.0, 1.0, 1.0, 1.0)]  # Empty grid colour should be white.
        for colormap_idx in colormap_indices:
            # Give the blocks an alpha of 0.7.
            r, g, b, _ = colormap(colormap_idx)
            self.colors.append((r, g, b, 0.7))

        super().__init__(name, render_mode)

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Render a FlatPack environment state.

        Args:
            state: the flat_pack environment state to be rendered.
            save_path: Optional path to save the rendered environment image to.

        Returns:
            RGB array if the render_mode is 'rgb_array'.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._add_grid_image(state.grid, ax)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

        return self._display(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of FlatPack states.

        Args:
            states: sequence of FlatPack states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig, ax = self._get_fig_ax(name_suffix="_animation", show=False)
        plt.close(fig=fig)

        def make_frame(state: State) -> Tuple[Artist]:
            ax.clear()
            self._add_grid_image(state.grid, ax)
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

    def _add_grid_image(self, grid: chex.Array, ax: plt.Axes) -> None:
        self._draw_grid(grid, ax)
        ax.set_axis_off()
        ax.set_aspect(1)
        ax.relim()
        ax.autoscale_view()

    def _draw_grid(self, grid: chex.Array, ax: plt.Axes) -> None:
        # Flip the grid upside down to match the coordinate system of matplotlib.
        grid = np.flipud(grid)
        rows, cols = grid.shape

        for row in range(rows):
            for col in range(cols):
                self._draw_grid_cell(grid[row, col], row, col, ax)

    def _draw_grid_cell(self, cell_value: int, row: int, col: int, ax: plt.Axes) -> None:
        cell = plt.Rectangle((col, row), 1, 1, **self._get_cell_attributes(cell_value))
        ax.add_patch(cell)
        if cell_value != 0:
            ax.text(
                col + 0.5,
                row + 0.5,
                str(int(cell_value)),
                color="#606060",
                ha="center",
                va="center",
                fontsize="xx-large",
            )

    def _get_cell_attributes(self, cell_value: int) -> Dict[str, Any]:
        color = self.colors[int(cell_value)]
        return {"facecolor": color, "edgecolor": "black", "linewidth": 1}
