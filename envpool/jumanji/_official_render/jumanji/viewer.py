# ruff: noqa
# fmt: off
from __future__ import annotations
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

"""Abstract environment viewer class."""

import abc
from typing import Any, Callable, Generic, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.layout_engine import ConstrainedLayoutEngine
from numpy.typing import NDArray

import jumanji.environments
from jumanji.env import State


class Viewer(abc.ABC, Generic[State]):
    """Abstract viewer class to support rendering and animation. This interface assumes
    that matplotlib is used for rendering the environment in question.
    """

    @abc.abstractmethod
    def render(self, state: State, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Render frames of the environment for a given state using matplotlib.

        Args:
            state: `State` object corresponding to the new state of the environment.
            save_path: Path to save the rendered environment image to.
        """

    @abc.abstractmethod
    def animate(
        self,
        states: Sequence[State],
        interval: int,
        save_path: Optional[str] = None,
    ) -> animation.FuncAnimation:
        """Create an animation from a sequence of environment states.

        Args:
            states: sequence of environment states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Perform any necessary cleanup. Environments will automatically :meth:`close()`
        themselves when garbage collected or when the program exits.
        """


class MatplotlibViewer(Viewer, abc.ABC, Generic[State]):
    """Abstract viewer class extending `Viewer` with some common
    matplotib figure creation and display functionality
    """

    def __init__(
        self,
        name: str,
        render_mode: str,
        figure_size: Tuple[float, float] = (10.0, 10.0),
    ):
        """
        Initialise a matplotlib viewer

        Args:
            name: Figure name.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
            figure_size: Figure size, default (10.0, 10.0)
        """
        self._name = name
        self._animation: Optional[animation.Animation] = None
        self.figure_size = figure_size

        # Render interactive animations in Jupyter
        plt.rcParams["animation.html"] = "jshtml"

        self._display: Callable[[plt.Figure], Optional[NDArray]]
        if render_mode == "rgb_array":
            self._display = self._display_rgb_array
        elif render_mode == "human":
            self._display = self._display_human
        else:
            raise ValueError(f"Invalid render mode: {render_mode}")

    def close(self) -> None:
        plt.close(getattr(self, "_figure_name", self._name))

    def _clear_display(self) -> None:
        if jumanji.environments.is_colab():
            import IPython.display

            IPython.display.clear_output(True)

    def _get_fig_ax(
        self,
        name_suffix: Optional[str] = None,
        show: bool = True,
        padding: float = 0.05,
        **fig_kwargs: Any,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Get axes figure, either retrieving existing axes with the same name
        or otherwise creating new axes.

        Args:
            name_suffix: Suffix to append to figure name, for example appending
                "_animation" to animation axes.
            show: Whether to display the figure, default `False`.
            padding: Padding around the figure axes, in fractions of the figure size.
                Default value is 0.05.
            **fig_kwargs: Keyword arguments to forward to the figure creation method.

        Returns:
            Matplotlib figure and axes.
        """
        figure_name = getattr(self, "_figure_name", self._name)
        name = figure_name if name_suffix is None else figure_name + name_suffix
        exists = plt.fignum_exists(name)

        if (not plt.isinteractive()) or (not show):
            plt.ioff()

        if exists:
            fig = plt.figure(name)
            ax = fig.get_axes()[0]
        else:
            fig = plt.figure(name, figsize=self.figure_size, **fig_kwargs)
            h_pad, w_pad = padding * self.figure_size[0], padding * self.figure_size[1]
            fig.set_layout_engine(layout=ConstrainedLayoutEngine(h_pad=h_pad, w_pad=w_pad))
            ax = fig.add_subplot(111)

            if (not plt.isinteractive()) and show:
                fig.show()

        return fig, ax

    def _display_human(self, fig: plt.Figure) -> None:
        if plt.isinteractive():
            # Required to update render when using Jupyter Notebook.
            fig.canvas.draw()
            if jumanji.environments.is_colab():
                plt.show(self._name)
        else:
            # Required to update render when not using Jupyter Notebook.
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    def _display_rgb_array(self, fig: plt.Figure) -> NDArray:
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())
