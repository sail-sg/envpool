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

import matplotlib.animation
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from numpy.typing import NDArray

from jumanji.environments.commons.maze_utils.maze_rendering import MazeViewer
from jumanji.environments.routing.cleaner.constants import CLEAN, DIRTY, WALL
from jumanji.environments.routing.cleaner.types import State


class CleanerViewer(MazeViewer):
    AGENT = 3
    COLORS: ClassVar[Dict[int, List[int]]] = {
        CLEAN: [1, 1, 1],  # White
        WALL: [0, 0, 0],  # Black
        DIRTY: [0, 1, 0],  # Green
    }
    AGENT_COLOR = ([1, 0, 0],)  # Red
    ALPHA = 0.5

    def __init__(self, name: str, render_mode: str = "human") -> None:
        """
        Viewer for the `Cleaner` environment.

        Args:
            name: the window name to be used when initialising the window.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        super().__init__(name, render_mode)

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Render the given state of the `Cleaner` environment.

        Args:
            state: the environment state to render.
            save_path: Optional path to save the rendered environment image to.

        Returns:
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
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of environment states.

        Args:
            states: sequence of environment states corresponding to consecutive timesteps.
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
            return (self._add_grid_image(state, ax),)

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

    def _add_grid_image(self, state: State, ax: Axes) -> image.AxesImage:
        img = self._create_grid_image(state)
        ax.set_axis_off()
        return ax.imshow(img)

    def _create_grid_image(self, state: State) -> NDArray:
        grid = state.grid
        img = np.zeros((*grid.shape, 3))
        for tile_value, color in self.COLORS.items():
            img[np.where(grid == tile_value)] = color
        # Add a channel for transparency
        img = np.pad(img, ((0, 0), (0, 0), (0, 1)), constant_values=1)
        img = self._set_agents_colors(img, state.agents_locations)
        img = self._draw_black_frame_around(img)
        return img

    def _set_agents_colors(self, img: NDArray, agents_locations: NDArray) -> NDArray:
        unique_locations, counts = np.unique(agents_locations, return_counts=True, axis=0)
        for location, count in zip(unique_locations, counts, strict=False):
            img[location[0], location[1], :3] = np.array(self.AGENT_COLOR)
            img[location[0], location[1], 3] = 1 - self.ALPHA**count
        return img

    def _draw_black_frame_around(self, img: NDArray) -> NDArray:
        # Draw black frame around maze by padding axis 0 and 1
        img = np.pad(img, ((1, 1), (1, 1), (0, 0)))  # type: ignore
        # Ensure the black frame is not transparent
        img[0, :, 3] = 1
        img[-1, :, 3] = 1
        img[:, 0, 3] = 1
        img[:, -1, 3] = 1
        return img
