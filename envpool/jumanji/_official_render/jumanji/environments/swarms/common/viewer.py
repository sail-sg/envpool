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

from typing import Tuple, Union

import jax.numpy as jnp
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.quiver import Quiver

from jumanji.environments.swarms.common.types import AgentState


def draw_agents(
    ax: Axes, agent_states: AgentState, color: Union[str, Tuple[int, int, int, int]]
) -> Quiver:
    """Draw a flock/swarm of agent using a matplotlib quiver

    Args:
        ax: Plot axes.
        agent_states: Flock/swarm agent states.
        color: Fill color of agents.

    Returns:
        `Quiver`: Matplotlib quiver, can also be used and
            updated when animating.
    """
    q = ax.quiver(
        agent_states.pos[:, 0],
        agent_states.pos[:, 1],
        jnp.cos(agent_states.heading),
        jnp.sin(agent_states.heading),
        color=color,
        pivot="middle",
        width=0.005,
        headwidth=5,
        headlength=8,
        headaxislength=8,
    )
    return q


def format_plot(
    fig: Figure, ax: Axes, env_dims: Tuple[float, float], border: float = 0.01
) -> Tuple[Figure, Axes]:
    """Format a flock/swarm plot, remove ticks and bound to the environment dimensions.

    Args:
        fig: Matplotlib figure.
        ax: Matplotlib axes.
        env_dims: Environment dimensions (i.e. its boundaries).
        border: Border padding to apply around plot.

    Returns:
        Figure: Formatted figure.
        Axes: Formatted axes.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, env_dims[0])
    ax.set_ylim(0, env_dims[1])

    return fig, ax
