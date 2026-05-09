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

from typing import List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.text import Text
from numpy.typing import NDArray

from jumanji.environments.logic.sudoku.constants import BOARD_WIDTH
from jumanji.environments.logic.sudoku.env import State
from jumanji.viewer import MatplotlibViewer


class SudokuViewer(MatplotlibViewer[State]):
    def __init__(self, name: str = "Sudoku", render_mode: str = "human") -> None:
        """Viewer for the `Sudoku` environment.

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
            state: `State` object corresponding to the new state of the environment.
            save_path: Optional path to save the rendered environment image to.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        self._draw(ax, state)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

        return self._display(fig)

    def _draw_board(self, ax: plt.Axes) -> None:
        # Draw the square box that delimits the board.
        ax.axis("off")

        _linewidth = 2.5
        ax.plot([0, 0], [0, BOARD_WIDTH], "-k", lw=_linewidth)
        ax.plot([0, BOARD_WIDTH], [BOARD_WIDTH, BOARD_WIDTH], "-k", lw=_linewidth)
        ax.plot([BOARD_WIDTH, BOARD_WIDTH], [BOARD_WIDTH, 0], "-k", lw=_linewidth)
        ax.plot([BOARD_WIDTH, 0], [0, 0], "-k", lw=_linewidth)
        for i in range(1, BOARD_WIDTH):
            if i % int(BOARD_WIDTH**0.5) == 0:
                _linewidth = 2.5
            else:
                _linewidth = 1

            hline = matplotlib.lines.Line2D(
                [0, BOARD_WIDTH], [i, i], color="k", linewidth=_linewidth
            )
            vline = matplotlib.lines.Line2D(
                [i, i], [0, BOARD_WIDTH], color="k", linewidth=_linewidth
            )
            ax.add_line(hline)
            ax.add_line(vline)

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
        fig.suptitle(f"{self._name}")
        texts = self._draw(ax, states[0])

        board_shape = states[0].board.shape

        def make_frame(state: State) -> List[plt.Text]:
            updated = []
            for i in range(board_shape[0]):
                for j in range(board_shape[1]):
                    element = state.board[i, j]
                    text = "" if element == -1 else str(element + 1)
                    txt_artist = texts[i][j]
                    txt_artist.set(text=text)
                    updated.append(txt_artist)

            return updated

        animation = FuncAnimation(
            fig,
            make_frame,
            frames=states,
            interval=interval,
            blit=True,
            save_count=len(states),
        )

        if save_path:
            animation.save(save_path)

        return animation

    def _draw(self, ax: plt.Axes, state: State) -> List[List[Text]]:
        ax.clear()
        self._draw_board(ax)
        return self._draw_figures(ax, state)

    def _draw_figures(self, ax: plt.Axes, state: State) -> List[List[Text]]:
        """Loop over the different cells and draws corresponding shapes in the ax object."""
        board = state.board
        board_shape = board.shape
        artists: List[List[Text]] = list()

        for i in range(board_shape[0]):
            artists.append([])
            for j in range(board_shape[1]):
                x_pos = j + 0.5
                y_pos = board_shape[0] - i - 0.5
                element = board[i, j]
                txt = "" if element == -1 else str(element + 1)
                txt = Text(
                    x=x_pos,
                    y=y_pos,
                    text=txt,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=16,
                )
                ax.add_artist(txt)
                artists[-1].append(txt)

        return artists
