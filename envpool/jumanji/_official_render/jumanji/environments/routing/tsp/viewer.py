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
from itertools import pairwise
from typing import List, Optional, Sequence, Tuple

import jax.numpy as jnp
import matplotlib.animation
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.collections import PathCollection
from matplotlib.quiver import Quiver
from numpy.typing import NDArray

import jumanji.environments
from jumanji.environments.routing.tsp.types import State
from jumanji.viewer import MatplotlibViewer


class TSPViewer(MatplotlibViewer[State]):
    NODE_COLOUR = "dimgray"
    NODE_SIZE = 150
    ARROW_WIDTH = 0.004

    def __init__(self, name: str, render_mode: str = "human") -> None:
        """Viewer for the TSP environment.

        Args:
            name: the window name to be used when initialising the window.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        super().__init__(name, render_mode)

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Render the given state of the `TSP` environment.

        Args:
            state: the environment state to render.
            save_path: Optional path to save the rendered environment image to.

        Return:
            RGB array if the render_mode is 'rgb_array'.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._prepare_figure(ax)
        self._add_tour(ax, state)

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
        self._prepare_figure(ax)
        cities, route, route_nodes = self._add_tour(ax, states[0])
        routes = [(route, route_nodes)]

        def make_frame(state_pair: Tuple[State, State]) -> List[Artist]:
            old_state, new_state = state_pair
            updated: List[Artist] = []

            if not jnp.array_equal(old_state.coordinates, new_state.coordinates):
                cities.set_offsets(new_state.coordinates)
                updated.append(cities)

            old_route, old_route_nodes = routes.pop()
            old_route.remove()
            old_route_nodes.remove()
            updated.append(old_route)
            updated.append(old_route_nodes)

            new_route, new_route_nodes = self._draw_route(ax, new_state)
            updated.append(new_route)
            updated.append(new_route_nodes)
            routes.append((new_route, new_route_nodes))

            return updated

        # Create the animation object.
        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=pairwise(states),
            interval=interval,
            save_count=len(states) - 1,
            blit=True,
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
        img_path = resources.files(jumanji.environments.routing.tsp) / "img/city_map.jpeg"
        map_img = plt.imread(img_path)
        ax.imshow(map_img, extent=(0, 1, 0, 1))

    def _draw_route(self, ax: plt.Axes, state: State) -> Tuple[Quiver, PathCollection]:
        xs, ys = state.coordinates[state.trajectory[: state.num_visited]].T
        dx = xs[1:] - xs[:-1]
        dy = ys[1:] - ys[:-1]
        quiver = ax.quiver(
            xs[:-1],
            ys[:-1],
            dx,
            dy,
            scale_units="xy",
            angles="xy",
            scale=1,
            width=self.ARROW_WIDTH,
            headwidth=5,
        )
        scatter = ax.scatter(xs, ys, s=self.NODE_SIZE, color="black")
        return quiver, scatter

    def _add_tour(
        self, ax: plt.Axes, state: State
    ) -> Tuple[PathCollection, Quiver, PathCollection]:
        """Add all the cities and the current tour between the visited cities to the plot."""
        x_coords, y_coords = state.coordinates.T

        # Draw the cities as nodes
        cities = ax.scatter(x_coords, y_coords, s=self.NODE_SIZE, color=self.NODE_COLOUR)

        # Draw the arrows between cities
        # if state.num_visited > 1:
        route, route_nodes = self._draw_route(ax, state)

        return cities, route, route_nodes
