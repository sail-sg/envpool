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

from typing import Optional, Sequence, Tuple

import jax.numpy as jnp
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray

from jumanji.environments.swarms.common.viewer import draw_agents, format_plot
from jumanji.environments.swarms.search_and_rescue.types import State
from jumanji.viewer import MatplotlibViewer


class SearchAndRescueViewer(MatplotlibViewer[State]):
    def __init__(
        self,
        name: str = "SearchAndRescue",
        env_size: Tuple[float, float] = (1.0, 1.0),
        searcher_color: str = "#282B28",  # black
        target_found_color: str = "#B3B6BC",  # light grey
        target_lost_color: str = "#E98449",  # orange
        render_mode: str = "human",
    ) -> None:
        """Viewer for the `SearchAndRescue` environment.

        Args:
            name: The window name to be used when initialising the window.
            searcher_color: Searching agents color
            target_found_color: Target node color when found
            target_lost_color: Target node color when not found
            env_size: Tuple environment spatial dimensions, used to set the plot region.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        self.searcher_color = searcher_color
        self.target_colors = np.array([target_lost_color, target_found_color])
        self.env_size = env_size
        super().__init__(name, render_mode)

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Render a frame of the environment for a given state using matplotlib.

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
        self, states: Sequence[State], interval: int, save_path: Optional[str] = None
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
        fig, ax = self._get_fig_ax(name_suffix="_animation", show=False)
        fig, ax = format_plot(fig, ax, self.env_size)
        plt.close(fig=fig)

        searcher_quiver = draw_agents(ax, states[0].searchers, self.searcher_color)
        target_scatter = ax.scatter(
            states[0].targets.pos[:, 0], states[0].targets.pos[:, 1], marker="o"
        )

        def make_frame(state: State) -> Tuple[Artist, Artist]:
            searcher_quiver.set_offsets(state.searchers.pos)
            searcher_quiver.set_UVC(
                jnp.cos(state.searchers.heading), jnp.sin(state.searchers.heading)
            )
            target_colors = self.target_colors[state.targets.found.astype(jnp.int32)]
            target_scatter.set_offsets(state.targets.pos)
            target_scatter.set_color(target_colors)
            return searcher_quiver, target_scatter

        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=states,
            interval=interval,
            blit=True,
        )

        if save_path:
            self._animation.save(save_path)

        return self._animation

    def _draw(self, ax: plt.Axes, state: State) -> None:
        ax.clear()
        draw_agents(ax, state.searchers, self.searcher_color)
        target_colors = self.target_colors[state.targets.found.astype(jnp.int32)]
        ax.scatter(
            state.targets.pos[:, 0], state.targets.pos[:, 1], marker="o", color=target_colors
        )

    def _get_fig_ax(
        self,
        name_suffix: Optional[str] = None,
        show: bool = True,
        padding: float = 0.05,
        **fig_kwargs: str,
    ) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = super()._get_fig_ax(name_suffix=name_suffix, show=show, padding=padding)
        fig, ax = format_plot(fig, ax, self.env_size)
        return fig, ax
