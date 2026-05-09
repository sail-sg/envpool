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

"""File adapted from [Rware](https://github.com/semitable/robotic-warehouse). More specifically,
the rendering code is copied from the original Rware environment and should be ignored from the
copyright."""

# flake8: noqa: CCR001

from typing import Optional, Sequence, Tuple

import chex
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection
from numpy.typing import NDArray

import jumanji.environments.routing.robot_warehouse.constants as constants
from jumanji.environments.routing.robot_warehouse.types import Agent, Direction, Shelf, State
from jumanji.tree_utils import tree_slice
from jumanji.viewer import MatplotlibViewer


class RobotWarehouseViewer(MatplotlibViewer[State]):
    def __init__(
        self,
        grid_size: Tuple[int, int],
        goals: chex.Array,
        name: str = "RobotWarehouse",
        render_mode: str = "human",
    ) -> None:
        """Viewer for the RobotWarehouse environment.

        Args:
            grid_size: the size of the warehouse floor grid (width, height)
            goals: x,y coordinates of goal locations (where shelves
                should be delivered)
            name: custom name for the Viewer. Defaults to `RobotWarehouse`.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        self.goals = goals
        self.rows, self.cols = grid_size

        self.grid_size = 30
        self.icon_size = 20

        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 1 + self.rows * (self.grid_size + 1)

        super().__init__(name, render_mode)

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Render the given state of the `RobotWarehouse` environment.

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
        self._draw_state(ax, state)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

        return self._display(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> animation.FuncAnimation:
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
            ax.clear()
            self._prepare_figure(ax)
            self._draw_state(ax, state)
            return (ax,)

        # Create the animation object.
        self._animation = animation.FuncAnimation(
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
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.patch.set_alpha(0.0)
        ax.set_axis_off()

        ax.set_aspect("equal", "box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    def _draw_state(self, ax: plt.Axes, state: State) -> None:
        self.n_agents = state.agents.position.x.shape[0]
        self.n_shelves = state.shelves.position.x.shape[0]
        self._draw_grid(ax)
        self._draw_goals(ax)
        self._draw_shelves(ax, state.shelves)
        self._draw_agents(ax, state.agents)

    def _draw_grid(self, ax: plt.Axes) -> None:
        """Draw grid of warehouse floor."""
        lines = []
        # VERTICAL LINES
        for r in range(self.rows + 1):
            lines.append(
                [
                    (0, (self.grid_size + 1) * r + 1),
                    ((self.grid_size + 1) * self.cols, (self.grid_size + 1) * r + 1),
                ]
            )

        # HORIZONTAL LINES
        for c in range(self.cols + 1):
            lines.append(
                [
                    ((self.grid_size + 1) * c + 1, 0),
                    ((self.grid_size + 1) * c + 1, (self.grid_size + 1) * self.rows),
                ]
            )

        lc = LineCollection(lines, colors=(constants._GRID_COLOR,))
        ax.add_collection(lc)

    def _draw_goals(self, ax: plt.Axes) -> None:
        """Draw goals, i.e. positions where shelves should be delivered."""
        for goal in self.goals:
            x, y = goal
            y = self.rows - y - 1  # pyglet rendering is reversed
            ax.fill(  # changed to ax, from plt, check if still works!
                [
                    x * (self.grid_size + 1) + 1,
                    (x + 1) * (self.grid_size + 1),
                    (x + 1) * (self.grid_size + 1),
                    x * (self.grid_size + 1) + 1,
                ],
                [
                    y * (self.grid_size + 1) + 1,
                    y * (self.grid_size + 1) + 1,
                    (y + 1) * (self.grid_size + 1),
                    (y + 1) * (self.grid_size + 1),
                ],
                color=constants._GOAL_COLOR,
                alpha=1,
            )

    def _draw_shelves(self, ax: plt.Axes, shelves: Shelf) -> None:
        """Draw shelves at their respective positions.

        Args:
            shelves: a pytree of Shelf type containing shelves information.
        """
        for shelf_id in range(self.n_shelves):
            shelf = tree_slice(shelves, shelf_id)
            y, x = shelf.position.x, shelf.position.y
            y = self.rows - y - 1  # pyglet rendering is reversed
            shelf_color = (
                constants._SHELF_REQ_COLOR if shelf.is_requested else constants._SHELF_COLOR
            )
            shelf_padding = constants._SHELF_PADDING

            x_points = [
                (self.grid_size + 1) * x + shelf_padding + 1,
                (self.grid_size + 1) * (x + 1) - shelf_padding,
                (self.grid_size + 1) * (x + 1) - shelf_padding,
                (self.grid_size + 1) * x + shelf_padding + 1,
            ]

            y_points = [
                (self.grid_size + 1) * y + shelf_padding + 1,
                (self.grid_size + 1) * y + shelf_padding + 1,
                (self.grid_size + 1) * (y + 1) - shelf_padding,
                (self.grid_size + 1) * (y + 1) - shelf_padding,
            ]

            ax.fill(x_points, y_points, color=shelf_color)

    def _draw_agents(self, ax: plt.Axes, agents: Agent) -> None:
        """Draw agents at their respective positions.

        Args:
            agents: a pytree of Shelf type containing agents information.
        """
        radius = self.grid_size / 3

        resolution = 6

        for agent_id in range(self.n_agents):
            agent = tree_slice(agents, agent_id)
            row, col = agent.position.x, agent.position.y
            row = self.rows - row - 1  # pyglet rendering is reversed
            x_center = (self.grid_size + 1) * col + self.grid_size // 2 + 1
            y_center = (self.grid_size + 1) * row + self.grid_size // 2 + 1

            # make a circle
            verts = []
            for i in range(resolution):
                angle = 2 * np.pi * i / resolution

                x_radius = radius * np.cos(angle)
                x = x_radius + x_center + 1

                y_radius = radius * np.sin(angle) + 1
                y = y_radius + y_center
                verts += [[x, y]]
                facecolor = (
                    constants._AGENT_LOADED_COLOR if agent.is_carrying else constants._AGENT_COLOR
                )
                circle = plt.Polygon(
                    verts,
                    edgecolor="none",
                    facecolor=facecolor,
                )

            ax.add_patch(circle)

            agent_dir = agent.direction

            x_dir = (
                x_center
                + (radius if agent_dir == Direction.RIGHT.value else 0)
                - (radius if agent_dir == Direction.LEFT.value else 0)
            )
            y_dir = (
                y_center
                + (radius if agent_dir == Direction.UP.value else 0)
                - (radius if agent_dir == Direction.DOWN.value else 0)
            )

            ax.plot(
                [x_center, x_dir],
                [y_center, y_dir],
                color=constants._AGENT_DIR_COLOR,
                linewidth=2,
            )
