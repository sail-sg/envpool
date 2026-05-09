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
from itertools import groupby, pairwise
from typing import List, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from chex import Array
from matplotlib.artist import Artist
from matplotlib.collections import PathCollection
from matplotlib.quiver import Quiver
from numpy.typing import NDArray

import jumanji.environments
from jumanji.environments.routing.cvrp.types import State
from jumanji.viewer import MatplotlibViewer


class CVRPViewer(MatplotlibViewer):
    NODE_COLOUR = "black"
    COLORMAP_NAME = "hsv"
    NODE_SIZE = 0.01
    ROUTE_NODE_SIZE = 100
    DEPOT_SIZE = 0.04
    ARROW_WIDTH = 0.004

    def __init__(self, name: str, num_cities: int, render_mode: str = "human") -> None:
        """Viewer for the `CVRP` environment.

        Args:
            name: the window name to be used when initialising the window.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        self._num_cities = num_cities
        # Each route to and from depot has a different color
        self._cmap = plt.get_cmap(self.COLORMAP_NAME, self._num_cities)
        super().__init__(name, render_mode)

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Render the given state of the `CVRP` environment.

        Args:
            state: the environment state to render.
            save_path: Optional path to save the rendered environment image to.

        Returns:
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
        nodes, routes = self._add_tour(ax, states[0])

        def make_frame(state_pair: Tuple[State, State]) -> List[Artist]:
            old_state, new_state = state_pair
            updated = []

            # If new episode then update node  artist positions
            if not jnp.array_equal(old_state.coordinates, new_state.coordinates):
                x_coords, y_coords = new_state.coordinates.T
                nodes[0].set(
                    x=x_coords[0] - 0.5 * self.DEPOT_SIZE, y=y_coords[0] - 0.5 * self.DEPOT_SIZE
                )
                for i in range(1, x_coords.shape[0]):
                    nodes[i].set_center((x_coords[i], y_coords[i]))
                updated.extend(nodes)

            # Pop existing route artists from the list, remove them from
            #  the plot, and add them to the updated artist list
            while routes:
                quiver, route_nodes = routes.pop()
                quiver.remove()
                route_nodes.remove()
                updated.append(quiver)
                updated.append(route_nodes)

            # Redraw the route (if it exists)
            if new_state.num_total_visits > 1:
                coords = new_state.coordinates[new_state.trajectory[: new_state.num_total_visits]]
                coords_grouped = self._group_tour(coords)

                for coords_route, col_id in zip(
                    coords_grouped, range(0, len(coords_grouped)), strict=False
                ):
                    new_route = self._draw_route(ax, coords_route, col_id)
                    routes.append(new_route)
                    updated.extend(new_route)

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
        img_path = resources.files(jumanji.environments.routing.cvrp) / "img/city_map.jpeg"
        map_img = plt.imread(img_path)
        ax.imshow(map_img, extent=(0, 1, 0, 1))

    def _group_tour(self, tour: Array) -> list:
        """Group the tour into routes that either (1) start and end at the depot, or, (2) start at
        the depot and end at the current city.

        Args:
            tour: x and y coordinates of the cities in the tour.

        Returns:
            tour_grouped: list of x and y coordinates that are grouped based on the above.
        """
        depot = tour[0]
        check_depot_fn = lambda x: (x != depot).all()
        tour_grouped = [
            np.array([depot, *list(g), depot]) for k, g in groupby(tour, key=check_depot_fn) if k
        ]
        if (tour[-1] != tour[0]).all():
            tour_grouped[-1] = tour_grouped[-1][:-1]
        return tour_grouped

    def _draw_route(
        self, ax: plt.Axes, coords: Array, col_id: int
    ) -> Tuple[Quiver, PathCollection]:
        """Draw the arrows and nodes for each route in the given colour."""
        x, y = coords.T

        # Compute the difference in the x- and y-coordinates to determine the distance between
        # consecutive cities.
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        quiver = ax.quiver(
            x[:-1],
            y[:-1],
            dx,
            dy,
            scale_units="xy",
            angles="xy",
            scale=1,
            width=self.ARROW_WIDTH,
            headwidth=5,
            color=self._cmap(col_id),
        )
        scatter = ax.scatter(x, y, s=self.ROUTE_NODE_SIZE, color=self._cmap(col_id))
        return quiver, scatter

    def _draw_cities(self, ax: plt.Axes, state: State) -> List[Union[plt.Circle, plt.Rectangle]]:
        nodes = []

        x_coords, y_coords = state.coordinates.T

        depot = plt.Rectangle(
            (x_coords[0] - 0.5 * self.DEPOT_SIZE, y_coords[0] - 0.5 * self.DEPOT_SIZE),
            self.DEPOT_SIZE,
            self.DEPOT_SIZE,
            color=self.NODE_COLOUR,
        )
        ax.add_artist(depot)
        nodes.append(depot)

        for i in range(1, x_coords.shape[0]):
            node = plt.Circle((x_coords[i], y_coords[i]), self.NODE_SIZE, color=self.NODE_COLOUR)
            ax.add_artist(node)
            nodes.append(node)

        return nodes

    def _add_tour(
        self, ax: plt.Axes, state: State
    ) -> Tuple[List[Union[plt.Circle, plt.Rectangle]], List[Tuple[Quiver, PathCollection]]]:
        """Add the cities and the depot to the plot, and draw each route in the tour in a different
        colour. The tour is the entire trajectory between the visited cities and a route is a
        trajectory either starting and ending at the depot or starting at the depot and ending at
        the current city."""
        x_coords, y_coords = state.coordinates.T

        nodes = self._draw_cities(ax, state)
        routes = []

        # Draw the arrows between cities
        if state.num_total_visits > 1:
            coords = state.coordinates[state.trajectory[: state.num_total_visits]]
            coords_grouped = self._group_tour(coords)

            # Draw each route in different colour
            for coords_route, col_id in zip(
                coords_grouped, range(0, len(coords_grouped)), strict=False
            ):
                routes.append(self._draw_route(ax, coords_route, col_id))

        return nodes, routes
