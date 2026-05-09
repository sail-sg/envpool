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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex


@dataclass(frozen=True)
class AgentParams:
    """
    max_rotate: Max angle an agent can rotate during a step (a fraction of pi)
    max_accelerate: Max change in speed during a step
    min_speed: Minimum agent speed
    max_speed: Maximum agent speed
    view_angle: Agent view angle, as a fraction of pi either side of its heading
    """

    max_rotate: float
    max_accelerate: float
    min_speed: float
    max_speed: float


@dataclass
class AgentState:
    """
    State of multiple agents of a single type

    pos: 2d position of the (centre of the) agents
    heading: Heading of the agents (in radians)
    speed: Speed of the agents
    """

    pos: chex.Array  # (num_agents, 2)
    heading: chex.Array  # (num_agents,)
    speed: chex.Array  # (num_agents,)
