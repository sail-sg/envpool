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
"""Small Matplotlib viewer base for EnvPool's Jumanji renderers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.layout_engine import ConstrainedLayoutEngine
from numpy.typing import NDArray


def asset_path(name: str) -> Path:
    return Path(__file__).with_name("img") / name


class MatplotlibViewer:
    def __init__(
        self,
        name: str,
        render_mode: str,
        figure_size: tuple[float, float] = (10.0, 10.0),
    ) -> None:
        self._name = name
        self.figure_size = figure_size
        plt.rcParams["animation.html"] = "jshtml"

        self._display: Callable[[plt.Figure], NDArray[np.generic] | None]
        if render_mode == "rgb_array":
            self._display = self._display_rgb_array
        elif render_mode == "human":
            self._display = self._display_human
        else:
            raise ValueError(f"Invalid render mode: {render_mode}")

    def close(self) -> None:
        plt.close(getattr(self, "_figure_name", self._name))

    def _clear_display(self) -> None:
        return None

    def _get_fig_ax(
        self,
        name_suffix: str | None = None,
        show: bool = True,
        padding: float = 0.05,
        **fig_kwargs: Any,
    ) -> tuple[plt.Figure, Any]:
        figure_name = getattr(self, "_figure_name", self._name)
        name = figure_name if name_suffix is None else figure_name + name_suffix

        if (not plt.isinteractive()) or (not show):
            plt.ioff()

        if plt.fignum_exists(name):
            fig = plt.figure(name)
            ax = fig.get_axes()[0]
        else:
            fig = plt.figure(name, figsize=self.figure_size, **fig_kwargs)
            h_pad = padding * self.figure_size[0]
            w_pad = padding * self.figure_size[1]
            fig.set_layout_engine(
                layout=ConstrainedLayoutEngine(h_pad=h_pad, w_pad=w_pad)
            )
            ax = fig.add_subplot(111)
            if (not plt.isinteractive()) and show:
                fig.show()

        return fig, ax

    def _display_human(self, fig: plt.Figure) -> None:
        if plt.isinteractive():
            fig.canvas.draw()
        else:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    def _display_rgb_array(self, fig: plt.Figure) -> NDArray[np.generic]:
        fig.canvas.draw()
        canvas = cast(Any, fig.canvas)
        return np.asarray(canvas.buffer_rgba())


def _compute_repulsive_forces(
    repulsive_forces: np.ndarray, pos: np.ndarray, k: float, num_nodes: int
) -> np.ndarray:
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            delta = pos[i] - pos[j]
            distance = np.linalg.norm(delta)
            direction = delta / (distance + 1e-6)
            force = k * k / (distance + 1e-6)
            repulsive_forces[i] += direction * force
            repulsive_forces[j] -= direction * force
    return repulsive_forces


def _compute_attractive_forces(
    graph: Any,
    attractive_forces: np.ndarray,
    pos: np.ndarray,
    k: float,
    num_nodes: int,
) -> np.ndarray:
    for i in range(num_nodes):
        for j in range(num_nodes):
            if graph[i, j]:
                delta = pos[i] - pos[j]
                distance = np.linalg.norm(delta)
                direction = delta / (distance + 1e-6)
                force = distance * distance / k
                attractive_forces[i] -= direction * force
                attractive_forces[j] += direction * force
    return attractive_forces


def spring_layout(
    graph: Any, num_nodes: int, seed: int = 42, iterations: int = 100
) -> list[tuple[float, float]]:
    rng = np.random.default_rng(seed)
    pos = rng.random((num_nodes, 2)) * 2 - 1
    k = np.sqrt(5 / num_nodes)
    temperature = 2.0

    for _ in range(iterations):
        repulsive_forces = _compute_repulsive_forces(
            np.zeros((num_nodes, 2)), pos, k, num_nodes
        )
        attractive_forces = _compute_attractive_forces(
            graph, np.zeros((num_nodes, 2)), pos, k, num_nodes
        )
        pos += (repulsive_forces + attractive_forces) * temperature
        temperature *= 0.9
        pos = np.clip(pos, -1, 1)

    pos_max = np.max(pos, axis=0)
    pos_min = np.min(pos, axis=0)
    pos = 0.05 + (pos - pos_min) / (1.1 * (pos_max - pos_min))
    return [(float(p[0]), float(p[1])) for p in pos]
