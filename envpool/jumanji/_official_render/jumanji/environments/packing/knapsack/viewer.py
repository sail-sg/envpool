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
from typing import Optional, Sequence, Tuple

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray

import jumanji.environments
from jumanji.environments.packing.knapsack.types import State
from jumanji.viewer import MatplotlibViewer


class KnapsackViewer(MatplotlibViewer[State]):
    def __init__(self, name: str, render_mode: str = "human", total_budget: float = 2.0) -> None:
        """Viewer for the `Knapsack` environment.

        Args:
            name: the window name to be used when initialising the window.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
            total_budget: the capacity of the knapsack.
        """
        self._total_budget = total_budget
        super().__init__(name, render_mode)

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Render the given state of the `Knapsack` environment.

        Args:
            state: the environment state to render.
            save_path: Optional path to save the rendered environment image to.

        Returns:
            RGB array if the render_mode is 'rgb_array'.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        self._prepare_figure(ax)
        self._show_value_and_budget(ax, state)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

        return self._display(fig)

    def _show_value_and_budget(self, ax: plt.Axes, state: State) -> None:
        # Initially, no items have been picked
        budget_used: np.ndarray = np.sum(state.weights, where=state.packed_items)
        total_value: np.ndarray = np.sum(state.values, where=state.packed_items)

        ax.set_title(
            f"Total value: {round(float(total_value), 2):.2f}. "
            f"Budget used: {round(float(budget_used), 2):.2f}/{self._total_budget}."
        )

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
        self._prepare_figure(ax)

        def make_frame(state: State) -> Tuple[Artist]:
            self._show_value_and_budget(ax, state)
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

    def _prepare_figure(self, ax: plt.Axes) -> None:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        img_path = resources.files(jumanji.environments.packing.knapsack) / "img/knapsack.png"
        sack_img = plt.imread(img_path)
        ax.imshow(sack_img, extent=(0, 1, 0, 1))
