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

from importlib import resources
from typing import Callable, Optional, Sequence, Tuple

import chex
import matplotlib.animation
import matplotlib.cm
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from numpy.typing import NDArray
from PIL import Image

import jumanji.environments
from jumanji.viewer import MatplotlibViewer


class BoxViewer(MatplotlibViewer):
    def __init__(self, name: str, grid_combine: Callable, render_mode: str = "human") -> None:
        """
        Viewer for a `Sokoban` environment using images from
        https://github.com/mpSchrader/gym-sokoban.

        Args:
            name: the window name to be used when initialising the window.
            grid_combine: function for combining fixed_grid and variable grid.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """

        self.NUM_COLORS = 10
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

        def get_image(image_name: str) -> Image.Image:
            img_path = (
                resources.files(jumanji.environments.routing.sokoban) / f"imgs/{image_name}.png"
            )
            return Image.open(str(img_path))

        self.images = [get_image(image_name) for image_name in image_names]

        super().__init__(name, render_mode)

    def render(self, state: chex.Array, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Render the given state of the `Sokoban` environment.

        Args:
            state: the environment state to render.
            save_path: Optional path to save the rendered environment image to.

        Return:
            RGB array if the render_mode is 'rgb_array'.
        """

        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._add_grid_image(state, ax)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

        return self._display(fig)

    def animate(
        self,
        states: Sequence[chex.Array],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of environment states.

        Args:
            states: sequence of environment states corresponding to
            consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If
            it is None, the plot will not be saved.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig, ax = self._get_fig_ax(name_suffix="_animation", show=False)
        plt.close(fig=fig)

        def make_frame(state: chex.Array) -> Tuple[Artist]:
            ax.clear()
            self._add_grid_image(state, ax)
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

    def _add_grid_image(self, state: chex.Array, ax: plt.Axes) -> None:
        """
        Add a grid image to the provided axes.

        Args:
            state: 'State' object representing a state of Sokoban.
            ax: (plt.Axes) object where the state image will be added.
        """
        grid = self.grid_combine(state.variable_grid, state.fixed_grid)

        self._draw_grid(grid, ax)
        ax.set_axis_off()
        ax.set_aspect(1)
        ax.relim()
        ax.autoscale_view()

    def _draw_grid(self, grid: chex.Array, ax: plt.Axes) -> None:
        """
        Draw a grid onto provided axes.

        Args:
            grid: Array () of shape ().
            ax: (plt.Axes) The axes on which to draw the grid.
        """

        cols, rows = grid.shape

        for col in range(cols):
            for row in range(rows):
                self._draw_grid_cell(grid[row, col], 9 - row, col, ax)

    def _draw_grid_cell(self, cell_value: int, row: int, col: int, ax: plt.Axes) -> None:
        """
        Draw a single cell of the grid.

        Args:
            cell_value: int representing the cell's value determining its image.
            row: int representing the cell's row index.
            col: int representing the cell's col index.
            ax: (plt.Axes) The axes on which to draw the cell.
        """
        cell_value = int(cell_value)
        image = self.images[cell_value]
        ax.imshow(image, extent=(col, col + 1, row, row + 1))
