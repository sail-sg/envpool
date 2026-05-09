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

from typing import Any, List, Optional, Sequence, Tuple

import jax.numpy as jnp
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from jumanji.environments.logic.rubiks_cube.constants import Face
from jumanji.environments.logic.rubiks_cube.types import State
from jumanji.viewer import MatplotlibViewer


class RubiksCubeViewer(MatplotlibViewer[State]):
    def __init__(self, sticker_colors: Optional[list], cube_size: int, render_mode: str = "human"):
        """
        Args:
            sticker_colors: colors used in rendering the faces of the Rubik's cube.
            cube_size: size of cube to view.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        self.cube_size = cube_size
        self.sticker_colors_cmap = matplotlib.colors.ListedColormap(sticker_colors)
        super().__init__(f"{cube_size}x{cube_size}x{cube_size} Rubik's Cube", render_mode)

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Render frames of the environment for a given state using matplotlib.

        Args:
            state: `State` object corresponding to the new state of the environment.
            save_path: Path to save the rendered environment image to.
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
    ) -> FuncAnimation:
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

        faces = self._draw(ax, states[0])

        def make_frame(state: State) -> Sequence[Artist]:
            for i, face in enumerate(faces):
                face.set_data(state.cube[i])
            return faces

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

    def _get_fig_ax(
        self,
        name_suffix: Optional[str] = None,
        show: bool = True,
        padding: float = 0.05,
        **fig_kwargs: Any,
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        fig_name = self._name if name_suffix is None else self._name + name_suffix
        exists = plt.fignum_exists(fig_name)
        if exists:
            fig = plt.figure(fig_name)
            ax = fig.get_axes()
        else:
            fig, ax = plt.subplots(nrows=3, ncols=2, figsize=self.figure_size, num=fig_name)
            fig.suptitle(self._name)
            ax = ax.flatten()
            plt.tight_layout()
            plt.axis("off")
            if not plt.isinteractive() and show:
                fig.show()

        return fig, ax

    def _draw(self, ax: List[plt.Axes], state: State) -> List[AxesImage]:
        images = []

        for i, face in enumerate(Face):
            ax[i].clear()
            ax[i].set_title(label=f"{face}")
            ax[i].set_xticks(jnp.arange(-0.5, self.cube_size - 1, 1))
            ax[i].set_yticks(jnp.arange(-0.5, self.cube_size - 1, 1))
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
