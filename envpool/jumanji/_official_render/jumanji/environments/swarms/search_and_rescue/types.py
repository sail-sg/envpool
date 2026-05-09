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
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex

from jumanji.environments.swarms.common.types import AgentState


@dataclass
class TargetState:
    """
    The state of the rescue targets.

    pos: 2d position of the target agents.
    velocity: 2d velocity of the target agents.
    found: Boolean flag indicating if the
        target has been located by a searcher.
    """

    pos: chex.Array  # (num_targets, 2)
    vel: chex.Array  # (num_targets, 2)
    found: chex.Array  # (num_targets,)


@dataclass
class State:
    """
    searchers: Searcher agent states.
    targets: Search target state.
    key: JAX random key.
    step: Environment step number
    """

    searchers: AgentState
    targets: TargetState
    key: chex.PRNGKey
    step: int = 0


class Observation(NamedTuple):
    """
    Individual observations for searching agents and information
    on number of remaining steps and ratio of targets to be found.

    Each agent generates an independent observation, an array of
    values representing the distance along a ray from the agent to
    the nearest neighbour, with each cell representing a ray angle
    (with `num_vision` rays evenly distributed over the agents
    field of vision).

    The co-ordinates of each agent are also included in the
    observation for debug and use in global observations.

    For example if an agent sees another agent straight ahead and
    `num_vision = 5` then the observation array could be

    ```
    [-1.0, -1.0, 0.5, -1.0, -1.0]
    ```

    where `-1.0` indicates there is no agents along that ray,
    and `0.5` is the normalised distance to the other agent.
    """

    searcher_views: chex.Array  # (num_searchers, num_vision)
    targets_remaining: chex.Numeric  # ()
    step: chex.Numeric  # ()
    positions: chex.Array  # (num_searchers, 2)
