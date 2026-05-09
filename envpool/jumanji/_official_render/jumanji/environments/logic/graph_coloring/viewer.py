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
from typing import List, Optional, Sequence, Tuple

import chex
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray

from jumanji.environments.commons.graph_view_utils import spring_layout
from jumanji.environments.logic.graph_coloring.types import State
from jumanji.viewer import MatplotlibViewer


class GraphColoringViewer(MatplotlibViewer[State]):
    def __init__(
        self,
        name: str = "GraphColoring",
        render_mode: str = "human",
    ) -> None:
        """
        Viewer for the `GraphColoring` environment.

        Args:
            name: the window name to be used when initializing the window.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        super().__init__(name, render_mode)

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Renders the current state of the graph.

        Args:
            state: The current game state to be rendered.
            save_path: Optional path to save the rendered environment image to.
        """
        self._clear_display()
        self._set_params(state)
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._prepare_figure(ax, state)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

        return self._display(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> animation.FuncAnimation:
        """Creates an animation of the graph from a sequence of game states.

        Args:
            states: Sequence of `State` objects.
            interval: Delay between frames in milliseconds, default to 200.
            save_path: Path to save the animation to. If None, the plot
                will not be saved.

        Returns:
            Animation object that can be saved as a GIF, MP4, or rendered with HTML.
        """
        self._set_params(states[0])
        fig, ax = self._get_fig_ax(name_suffix="_animation", show=False)
        plt.close(fig=fig)
        nodes, labels, edges = self._prepare_figure(ax, states[0])

        def make_frame(state_pair: Tuple[State, State]) -> List[Artist]:
            prev_state, state = state_pair

            for circle, color in zip(nodes, state.colors, strict=False):
                circle.set(color=self._color_mapping[color])
            # Update node and edges if new episode
            if not np.array_equal(prev_state.adj_matrix, state.adj_matrix):
                pos = spring_layout(state.adj_matrix, self.num_nodes)
                for circle, label, xy in zip(nodes, labels, pos, strict=False):
                    circle.set_center(xy)
                    label.set(x=xy[0], y=xy[1])
                n = 0
                for i in range(self.num_nodes):
                    for j in range(i + 1, self.num_nodes):
                        edges[n].set(
                            xdata=[pos[i][0], pos[j][0]],
                            ydata=[pos[i][1], pos[j][1]],
                            visible=state.adj_matrix[i, j],
                        )
                        n += 1

                return nodes + edges

            else:
                return nodes

        _animation = animation.FuncAnimation(
            fig,
            make_frame,
            frames=pairwise(states),
            interval=interval,
            save_count=len(states) - 1,
            blit=True,
        )

        if save_path:
            _animation.save(save_path)

        return _animation

    def _set_params(self, state: State) -> None:
        self.num_nodes = state.adj_matrix.shape[0]
        self.node_scale = self._calculate_node_scale(self.num_nodes)
        self._color_mapping = self._create_color_mapping(self.num_nodes)

    def _prepare_figure(
        self, ax: plt.Axes, state: State
    ) -> Tuple[List[plt.Circle], List[plt.Text], List[plt.Line2D]]:
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        pos = spring_layout(state.adj_matrix, self.num_nodes)
        edges = self._render_edges(ax, pos, state.adj_matrix, self.num_nodes)
        nodes, labels = self._render_nodes(ax, pos, state.colors)
        return nodes, labels, edges

    def _render_nodes(
        self, ax: plt.Axes, pos: List[Tuple[float, float]], colors: chex.Array
    ) -> Tuple[List[plt.Circle], List[plt.Text]]:
        # Set the radius of the nodes as a fraction of the scale,
        # so nodes appear smaller when there are more of them.
        node_radius = 0.05 * 5 / self.node_scale
        circles = []
        labels = []

        for i, (x, y) in enumerate(pos):
            c = plt.Circle(
                (x, y),
                node_radius,
                color=self._color_mapping[colors[i]],
                fill=True,
                zorder=100,
            )
            circles.append(c)
            ax.add_artist(c)
            label = plt.Text(
                x,
                y,
                str(i),
                color="white",
                ha="center",
                va="center",
                weight="bold",
                zorder=200,
            )
            labels.append(label)
            ax.add_artist(label)

        return circles, labels

    def _render_edges(
        self,
        ax: plt.Axes,
        pos: List[Tuple[float, float]],
        adj_matrix: chex.Array,
        num_nodes: int,
    ) -> List[plt.Line2D]:
        edges = []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge = plt.Line2D(
                    [pos[i][0], pos[j][0]],
                    [pos[i][1], pos[j][1]],
                    color=self._color_mapping[-1],
                    linewidth=0.5,
                    visible=adj_matrix[i, j],
                )
                ax.add_artist(edge)
                edges.append(edge)

        return edges

    def _calculate_node_scale(self, num_nodes: int) -> int:
        # Set the scale of the graph based on the number of nodes,
        # so the graph grows (at a decelerating rate) with more nodes.
        return 5 + int(np.sqrt(num_nodes))

    def _create_color_mapping(
        self,
        num_nodes: int,
    ) -> List[Tuple[float, float, float, float]]:
        colormap_indices = np.arange(0, 1, 1 / num_nodes)
        colormap = plt.get_cmap("hsv", num_nodes + 1)
        color_mapping = []
        for colormap_idx in colormap_indices:
            color_mapping.append(colormap(float(colormap_idx)))
        color_mapping.append((0.0, 0.0, 0.0, 1.0))  # Adding black to the color mapping
        return color_mapping
