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

from itertools import pairwise
from typing import Dict, List, Optional, Sequence, Tuple

import chex
import jax.numpy as jnp
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from jumanji.environments.commons.graph_view_utils import spring_layout
from jumanji.environments.routing.mmst.types import State
from jumanji.viewer import MatplotlibViewer

grey = (100 / 255, 100 / 255, 100 / 255)
white = (255 / 255, 255 / 255, 255 / 255)
yellow = (200 / 255, 200 / 255, 0 / 255)
red = (200 / 255, 0 / 255, 0 / 255)
black = (0 / 255, 0 / 255, 0 / 255)
blue = (50 / 255, 50 / 255, 160 / 255)


class MMSTViewer(MatplotlibViewer[State]):
    """Viewer class for the MMST environment."""

    def __init__(
        self,
        num_agents: int,
        name: str = "MMST",
        render_mode: str = "human",
    ) -> None:
        """Create a `MMSTViewer` instance for rendering the `MMST` environment.

        Args:
            num_agents: Number of agents in the environment.
        """

        self.num_agents = num_agents

        np.random.seed(0)

        self.palette: List[Tuple[float, float, float]] = []

        for _ in range(num_agents):
            colour = (
                np.random.randint(0, 192) / 255,
                np.random.randint(0, 192) / 255,
                np.random.randint(0, 192) / 255,
            )
            self.palette.append(colour)

        super().__init__(name, render_mode)

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Render the state of the environment.

        Args:
            state: the current state of the environment to render.
            save_path: Optional path to save the rendered environment image to.

        Return:
            RGB array if the render_mode is 'rgb_array'.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._draw_graph(state, ax)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

        return self._display(fig)

    def _draw_graph(
        self, state: State, ax: plt.Axes
    ) -> Tuple[
        Dict[Tuple[int, ...], plt.Line2D], List[Tuple[plt.Circle, plt.Circle]], List[plt.Text]
    ]:
        """Draw the different nodes and edges in the graph

        Args:
            state: current state of the environment.
            ax: figure axes on which to plot.
        """
        num_nodes = state.adj_matrix.shape[0]
        positions = spring_layout(state.adj_matrix, num_nodes)
        node_scale = 5 + int(np.sqrt(num_nodes))
        node_radius = 0.05 * 5 / node_scale

        edges = self.build_edges(state.adj_matrix, state.connected_nodes)

        lines = {}

        # Draw edges.
        for k, e in edges.items():
            (n1, n2), color = e
            n1, n2 = int(n1), int(n2)
            x_values = [positions[n1][0], positions[n2][0]]
            y_values = [positions[n1][1], positions[n2][1]]
            edge = Line2D(x_values, y_values, c=color, linewidth=2)
            ax.add_artist(edge)
            lines[k] = edge

        circles = []
        labels = []

        # Draw nodes.
        for node in range(num_nodes):
            pos = np.where(state.nodes_to_connect == node)[0]
            if len(pos) == 1:
                fcolor = self.palette[pos[0]]
            else:
                fcolor = black

            if node in state.positions:
                lcolor = yellow
            else:
                lcolor = blue

            c = self.circle_fill(
                positions[node],
                lcolor,
                fcolor,
                node_radius,
                0.2 * node_radius,
                ax,
            )

            circles.append(c)

            txt = plt.Text(
                positions[node][0],
                positions[node][1],
                str(node),
                color="white",
                ha="center",
                va="center",
                weight="bold",
                zorder=200,
            )
            ax.add_artist(txt)
            labels.append(txt)

        ax.set_axis_off()
        ax.set_aspect(1)
        ax.relim()
        ax.autoscale_view()

        return lines, circles, labels

    def build_edges(
        self, adj_matrix: chex.Array, connected_nodes: chex.Array
    ) -> Dict[Tuple[int, ...], List[Tuple[float, ...]]]:
        # Normalize id for either order.
        def edge_id(n1: int, n2: int) -> Tuple[int, ...]:
            return tuple(sorted((n1, n2)))

        # Might be slow but for now we will always build all the edges.
        edges: Dict[Tuple[int, ...], List[Tuple[float, ...]]] = {}

        # Convert to numpy
        connected_nodes = np.asarray(connected_nodes)
        row_indices, col_indices = jnp.nonzero(adj_matrix)
        # Create the edge list as a list of tuples (source, target)
        edges_list = [
            (int(row), int(col)) for row, col in zip(row_indices, col_indices, strict=False)
        ]

        for edge in edges_list:
            n1, n2 = edge
            eid = edge_id(n1, n2)
            if eid not in edges:
                edges[eid] = [(n1, n2), grey]

        for agent in range(self.num_agents):
            conn_group = connected_nodes[agent]
            len_conn = np.where(conn_group != -1)[0][-1]  # Get last index where node is not -1.
            for i in range(len_conn):
                eid = edge_id(conn_group[i], conn_group[i + 1])
                edges[eid] = [(conn_group[i], conn_group[i + 1]), self.palette[agent]]

        return edges

    def circle_fill(
        self,
        xy: Tuple[float, float],
        line_color: Tuple[float, float, float],
        fill_color: Tuple[float, float, float],
        radius: float,
        thickness: float,
        ax: plt.Axes,
    ) -> Tuple[plt.Circle, plt.Circle]:
        ca = plt.Circle(xy, radius, color=line_color, zorder=100)
        cb = plt.Circle(xy, radius - thickness, color=fill_color, zorder=100)
        ax.add_artist(ca)
        ax.add_artist(cb)
        return ca, cb

    def animate(
        self,
        states: Sequence[State],
        interval: int = 2000,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of Connector grids.

        Args:
            states: sequence of states to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 2000.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """

        fig, ax = self._get_fig_ax(name_suffix="_animation", show=False)
        plt.close(fig=fig)
        edges, circles, labels = self._draw_graph(states[0], ax)

        def make_frame(state_pair: Tuple[State, State]) -> List[Artist]:
            old_state, new_state = state_pair

            updated = []

            if not jnp.array_equal(old_state.adj_matrix, new_state.adj_matrix):
                while circles:
                    circle = circles.pop()
                    circle[0].remove()
                    circle[1].remove()
                    updated.append(circle[0])
                    updated.append(circle[1])
                while labels:
                    label = labels.pop()
                    label.remove()
                    updated.append(label)
                for k in list(edges.keys()):
                    edge = edges.pop(k)
                    edge.remove()
                    updated.append(edge)

                new_edges, new_circles, new_labels = self._draw_graph(new_state, ax)
                edges.update(new_edges)
                circles.extend(new_circles)
                labels.extend(new_labels)

            else:
                edge_updates = self.build_edges(new_state.adj_matrix, new_state.connected_nodes)

                for k, (_, color) in edge_updates.items():
                    edge = edges[k]
                    edge.set(color=color)
                    updated.append(edge)

                for i, (ca, cb) in enumerate(circles):
                    pos = np.where(new_state.nodes_to_connect == i)[0]
                    if len(pos) == 1:
                        fcolor = self.palette[pos[0]]
                    else:
                        fcolor = black

                    if i in new_state.positions:
                        lcolor = yellow
                    else:
                        lcolor = blue

                    ca.set(color=lcolor)
                    cb.set(color=fcolor)
                    updated.append(ca)
                    updated.append(cb)

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
