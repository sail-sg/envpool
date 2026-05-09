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

import matplotlib.animation
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray

from jumanji.environments.packing.job_shop.types import State
from jumanji.viewer import MatplotlibViewer


class JobShopViewer(MatplotlibViewer[State]):
    COLORMAP_NAME = "hsv"

    def __init__(
        self,
        name: str,
        num_jobs: int,
        num_machines: int,
        max_num_ops: int,
        max_op_duration: int,
        render_mode: str = "human",
    ) -> None:
        """Viewer for the `JobShop` environment.

        Args:
            name: the window name to be used when initialising the window.
            num_jobs: the number of jobs that need to be scheduled.
            num_machines: the number of machines that the jobs can be scheduled on.
            max_num_ops: the maximum number of operations for any given job.
            max_op_duration: the maximum processing time of any given operation.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        self._num_jobs = num_jobs
        self._num_machines = num_machines
        self._max_num_ops = max_num_ops
        self._max_op_duration = max_op_duration

        # Have additional color to avoid two jobs having same color when using hsv colormap
        self._cmap = plt.get_cmap(self.COLORMAP_NAME, self._num_jobs + 1)

        super().__init__(name, render_mode)

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Render the given state of the `JobShop` environment.

        Args:
            state: the environment state to render.
            save_path: Optional path to save the rendered environment image to.

        Returns:
            RGB array if the render_mode is 'rgb_array'.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        ax.set_title(f"Scheduled Jobs at Time={state.step_count}")
        ax.axvline(state.step_count, ls="--", color="red", lw=0.5)
        self._prepare_figure(ax)
        self._add_scheduled_ops(ax, state)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

        return self._display(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of states.

        Args:
            states: sequence of environment states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation object that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig, ax = self._get_fig_ax(name_suffix="_animation", show=False)
        plt.close(fig=fig)
        self._prepare_figure(ax)

        def make_frame(state: State) -> Tuple[Artist]:
            ax.clear()
            self._prepare_figure(ax)
            ax.set_title(rf"Scheduled Jobs at Time={state.step_count}")
            self._add_scheduled_ops(ax, state)
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
        ax.set_xlabel("Time")
        ax.set_ylabel("Machine ID")
        xlim = self._num_jobs * self._max_num_ops * self._max_op_duration // self._num_machines
        ax.set_xlim(0, xlim)
        ax.set_ylim(-0.9, self._num_machines)
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.get_major_locator().set_params(integer=True)
        major_ticks = np.arange(0, xlim, 10)
        minor_ticks = np.arange(0, xlim, 1)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.grid(axis="x", linewidth=0.25)
        ax.set_axisbelow(True)

    def _add_scheduled_ops(self, ax: plt.Axes, state: State) -> None:
        """Add the scheduled operations to the plot."""
        for job_id in range(self._num_jobs):
            for op_id in range(self._max_num_ops):
                start_time = state.scheduled_times[job_id, op_id]
                machine_id = state.ops_machine_ids[job_id, op_id]
                duration = state.ops_durations[job_id, op_id]
                colour = self._cmap(job_id)
                line_height = 0.8
                if start_time >= 0:
                    rectangle = matplotlib.patches.Rectangle(
                        (start_time, machine_id - line_height / 2),
                        width=duration,
                        height=line_height,
                        linewidth=1,
                        facecolor=colour,
                        edgecolor="black",
                    )
                    ax.add_patch(rectangle)

                    # Annotate the operation with the job id
                    rx, ry = rectangle.get_xy()
                    cx = rx + rectangle.get_width() / 2.0
                    cy = ry + rectangle.get_height() / 2.0
                    ax.annotate(
                        f"J{job_id}",
                        (cx, cy),
                        color="black",
                        fontsize=10,
                        ha="center",
                        va="center",
                    )
