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
"""Flat renderer for Jumanji swarm environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from envpool.jumanji._official_render.base import MatplotlibViewer


@dataclass
class AgentState:
    pos: Any
    heading: Any
    speed: Any


@dataclass
class TargetState:
    pos: Any
    vel: Any
    found: Any


@dataclass
class SearchAndRescueState:
    searchers: AgentState
    targets: TargetState
    key: Any
    step: int = 0


class SearchAndRescueViewer(MatplotlibViewer):
    def __init__(
        self,
        name: str = "SearchAndRescue",
        env_size: tuple[float, float] = (1.0, 1.0),
        searcher_color: str = "#282B28",
        target_found_color: str = "#B3B6BC",
        target_lost_color: str = "#E98449",
        render_mode: str = "human",
    ) -> None:
        self.searcher_color = searcher_color
        self.target_colors = np.array([target_lost_color, target_found_color])
        self.env_size = env_size
        super().__init__(name, render_mode)

    def render(
        self, state: SearchAndRescueState, save_path: str | None = None
    ) -> NDArray[np.generic] | None:
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        ax.quiver(
            state.searchers.pos[:, 0],
            state.searchers.pos[:, 1],
            np.cos(state.searchers.heading),
            np.sin(state.searchers.heading),
            color=self.searcher_color,
            pivot="middle",
            width=0.005,
            headwidth=5,
            headlength=8,
            headaxislength=8,
        )
        ax.scatter(
            state.targets.pos[:, 0],
            state.targets.pos[:, 1],
            marker="o",
            color=self.target_colors[state.targets.found.astype(np.int32)],
        )
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return self._display(fig)

    def _get_fig_ax(
        self,
        name_suffix: str | None = None,
        show: bool = True,
        padding: float = 0.05,
        **fig_kwargs: Any,
    ) -> Any:
        fig, ax = super()._get_fig_ax(name_suffix, show, padding, **fig_kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, self.env_size[0])
        ax.set_ylim(0, self.env_size[1])
        return fig, ax
