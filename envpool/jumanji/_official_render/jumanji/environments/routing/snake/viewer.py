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

import matplotlib.animation
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.patches import Circle, Patch, Rectangle
from numpy.typing import NDArray

from jumanji.environments.routing.snake.types import State
from jumanji.viewer import MatplotlibViewer


class SnakeViewer(MatplotlibViewer[State]):
    def __init__(self, name: str = "Snake", render_mode: str = "human") -> None:
        """Viewer for the `Snake` environment.

        Args:
            name: the window name to be used when initialising the window.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        super().__init__(name, render_mode)

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Render frames of the environment for a given state using matplotlib.

        Args:
            state: State object containing the current dynamics of the environment.
            save_path: Optional path to save the rendered environment image to.

        Return:
            RGB array if the render_mode is 'rgb_array'.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        self._draw(ax, state)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

        return self._display(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of states.

        Args:
            states: sequence of `State` corresponding to subsequent timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation object that can be saved as a GIF, MP4, or rendered with HTML.
        """
        if not states:
            raise ValueError(f"The states argument has to be non-empty, got {states}.")

        fig, ax = self._get_fig_ax(
            "_animation",
            show=False,
        )
        plt.close(fig=fig)
        self._draw_board(ax, states[0])

        patches: List[matplotlib.patches.Patch] = []

        def make_frame(state: State) -> Tuple[Artist]:
            while patches:
                patches.pop().remove()
            patches.extend(self._create_entities(state))
            for patch in patches:
                ax.add_patch(patch)
            return (ax,)

        # Create the animation object.
        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=states[1:],
            interval=interval,
        )

        # Save the animation as a gif.
        if save_path:
            self._animation.save(save_path)

        return self._animation

    def _draw(self, ax: plt.Axes, state: State) -> None:
        ax.clear()
        self._draw_board(ax, state)
        for patch in self._create_entities(state):
            ax.add_patch(patch)

    def _draw_board(self, ax: plt.Axes, state: State) -> None:
        num_rows, num_cols = state.body_state.shape[-2:]
        # Draw the square box that delimits the board.
        ax.axis("off")
        ax.plot([0, 0], [0, num_rows], "-k", lw=2)
        ax.plot([0, num_cols], [num_rows, num_rows], "-k", lw=2)
        ax.plot([num_cols, num_cols], [num_rows, 0], "-k", lw=2)
        ax.plot([num_cols, 0], [0, 0], "-k", lw=2)

    def _create_entities(self, state: State) -> Sequence[Patch]:
        """Loop over the different cells and draws corresponding shapes in the ax object."""
        num_rows, num_cols = state.body_state.shape[-2:]

        patches: List[Patch] = list()
        linewidth = (
            min(n * size for n, size in zip((num_rows, num_cols), self.figure_size, strict=False))
            / 44.0
        )
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "", ["yellowgreen", "forestgreen"]
        )
        for row in range(num_rows):
            for col in range(num_cols):
                if state.body_state[row, col]:
                    body_cell_patch = Rectangle(
                        (col, num_rows - 1 - row),
                        1,
                        1,
                        edgecolor=cmap(1),
                        facecolor=cmap(state.body_state[row, col] / state.length),
                        fill=True,
                        lw=linewidth,
                    )
                    patches.append(body_cell_patch)
        head_patch = Circle(
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
        patches.append(head_patch)
        fruit_patch = Circle(
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
        patches.append(fruit_patch)
        return patches
