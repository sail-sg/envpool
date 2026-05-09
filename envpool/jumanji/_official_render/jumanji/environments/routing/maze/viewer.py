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

from typing import ClassVar, Optional, Sequence

import chex
from matplotlib import animation
from numpy.typing import NDArray

from jumanji.environments.commons.maze_utils.maze_rendering import MazeViewer
from jumanji.environments.routing.maze.types import State
from jumanji.viewer import Viewer

EMPTY = 0
WALL = 1


class MazeEnvViewer(MazeViewer):
    AGENT = 2
    TARGET = 3
    COLORS: ClassVar = {
        EMPTY: [1, 1, 1],  # White
        WALL: [0, 0, 0],  # Black
        AGENT: [0, 1, 0],  # Green
        TARGET: [1, 0, 0],  # Red
    }

    def __init__(
        self,
        name: str,
        render_mode: str = "human",
        viewer: Optional[Viewer[State]] = None,
    ) -> None:
        """Viewer for the Maze environment.

        Args:
            name: the window name to be used when initialising the matplotlib window.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        super().__init__(name, render_mode)

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Render the given state of the `Maze` environment.

        Args:
            state: the environment state to render.
            save_path: Optional path to save the rendered environment image to.

        Returns:
            RGB array if the render_mode is 'rgb_array'.
        """
        maze = self._overlay_agent_and_target(state)

        return super().render(maze, save_path=save_path)

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
        mazes = [self._overlay_agent_and_target(state) for state in states]
        return super().animate(mazes, interval, save_path)

    def _overlay_agent_and_target(self, state: State) -> chex.Array:
        maze = state.walls.astype(int)
        maze = maze.at[tuple(state.agent_position)].set(self.AGENT)
        return maze.at[tuple(state.target_position)].set(self.TARGET)
