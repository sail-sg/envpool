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

from typing import ClassVar, Dict, List, Optional, Sequence, Tuple

import chex
import matplotlib.animation
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from numpy.typing import NDArray

from jumanji.environments.commons.maze_utils.maze_generation import EMPTY, WALL
from jumanji.viewer import MatplotlibViewer


class MazeViewer(MatplotlibViewer):
    FONT_STYLE = "monospace"
    # EMPTY is white, WALL is black
    COLORS: ClassVar[Dict[int, List[int]]] = {EMPTY: [1, 1, 1], WALL: [0, 0, 0]}

    def __init__(self, name: str, render_mode: str = "human") -> None:
        """Viewer for a maze environment.

        Args:
            name: the window name to be used when initialising the window.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        super().__init__(name, render_mode)

    def render(self, maze: chex.Array, save_path: Optional[str] = None) -> Optional[NDArray]:
        """
        Render maze.

        Args:
            maze: the maze to render.
            save_path: Optional path to save the rendered environment image to.

        Returns:
            RGB array if the render_mode is 'rgb_array'.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._add_grid_image(maze, ax)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

        return self._display(fig)

    def animate(
        self,
        mazes: Sequence[chex.Array],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of mazes.

        Args:
            mazes: sequence of `Maze` corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig, ax = self._get_fig_ax(name_suffix="_animation", show=False)
        plt.close(fig=fig)

        def make_frame(maze: chex.Array) -> Tuple[Artist]:
            ax.clear()
            self._add_grid_image(maze, ax)
            return (ax,)

        # Create the animation object.
        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=mazes,
            interval=interval,
        )

        # Save the animation as a gif.
        if save_path:
            self._animation.save(save_path)

        return self._animation

    def _add_grid_image(self, maze: chex.Array, ax: Axes) -> image.AxesImage:
        img = self._create_grid_image(maze)
        ax.set_axis_off()
        return ax.imshow(img)

    def _create_grid_image(self, maze: chex.Array) -> NDArray:
        img = np.zeros((*maze.shape, 3))
        for tile_value, color in self.COLORS.items():
            img[np.where(maze == tile_value)] = color
        # Draw black frame around maze by padding axis 0 and 1
        img = np.pad(img, ((1, 1), (1, 1), (0, 0)))  # type: ignore
        return img
