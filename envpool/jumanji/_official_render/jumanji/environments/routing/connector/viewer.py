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

from jumanji.environments.routing.connector.utils import (
    get_agent_id,
    is_path,
    is_target,
)
from jumanji.viewer import MatplotlibViewer


class ConnectorViewer(MatplotlibViewer):
    def __init__(self, name: str, num_agents: int, render_mode: str = "human") -> None:
        """
        Viewer for a `Connector` environment.

        Args:
            name: the window name to be used when initialising the window.
            num_agents: Number of environment agents
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        # Pick colors.
        colormap_indices = np.arange(0, 1, 1 / num_agents)
        colormap = plt.get_cmap("hsv", num_agents + 1)

        self.colors = [(1.0, 1.0, 1.0, 1.0)]  # Initial color must be white.
        for colormap_idx in colormap_indices:
            self.colors.append(colormap(float(colormap_idx)))

        super().__init__(name, render_mode)

    def render(self, grid: chex.Array, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Render Connector.

        Args:
            grid: the grid of the Connector environment to render.
            save_path: Optional path to save the rendered environment image to.

        Returns:
            RGB array if the render_mode is 'rgb_array'.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._add_grid_image(grid, ax)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

        return self._display(fig)

    def animate(
        self,
        grids: Sequence[chex.Array],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of Connector grids.

        Args:
            grids: sequence of Connector grids corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig, ax = self._get_fig_ax(name_suffix="_animation", show=False)
        plt.close(fig=fig)

        def make_frame(grid: chex.Array) -> Tuple[Artist]:
            ax.clear()
            self._add_grid_image(grid, ax)
            return (ax,)

        # Create the animation object.
        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=grids,
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
        rows, cols = grid.shape

        for row in range(rows):
            for col in range(cols):
                self._draw_grid_cell(grid[row, col], row, col, ax)

    def _draw_grid_cell(self, cell_value: int, row: int, col: int, ax: plt.Axes) -> None:
        cell = plt.Rectangle((col, row), 1, 1, **self._get_cell_attributes(cell_value))
        ax.add_patch(cell)

        if is_target(cell_value):
            pos = (col + 0.25, row + 0.25)
            size = 0.5
            attribs = self._get_inner_cell_attributes(cell_value)

            cell = plt.Rectangle(pos, size, size, **attribs)
            ax.add_patch(cell)

    def _get_cell_attributes(self, cell_value: int) -> Dict[str, Any]:
        agent_id = get_agent_id(cell_value)

        color = self.colors[agent_id]
        if is_target(cell_value):
            color = (1.0, 1.0, 1.0, 1.0)
        elif is_path(cell_value):
            color = (*self.colors[agent_id][:3], 0.25)

        return {"facecolor": color, "edgecolor": "black", "linewidth": 1}

    def _get_inner_cell_attributes(self, cell_value: int) -> Dict[str, Any]:
        assert is_target(cell_value)
        agent_id = get_agent_id(cell_value)

        return {"facecolor": self.colors[agent_id]}
