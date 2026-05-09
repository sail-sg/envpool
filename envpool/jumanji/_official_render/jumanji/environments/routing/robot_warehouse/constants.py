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

import jax.numpy as jnp

from jumanji.environments.routing.robot_warehouse.types import Direction

# grid channels
_SHELVES = 0
_AGENTS = 1

# agent directions
_POSSIBLE_DIRECTIONS = jnp.array([d.value for d in Direction])

# viewer constants
_FIGURE_SIZE = (5, 5)
_SHELF_PADDING = 2

# colors
_GRID_COLOR = (0, 0, 0)  # black
_SHELF_COLOR = (72 / 255.0, 61 / 255.0, 139 / 255.0)  # dark slate blue
_SHELF_REQ_COLOR = (0, 128 / 255.0, 128 / 255.0)  # teal
_AGENT_COLOR = (1, 140 / 255.0, 0)  # dark orange
_AGENT_LOADED_COLOR = (1, 0, 0)  # red
_AGENT_DIR_COLOR = (0, 0, 0)  # black
_GOAL_COLOR = (60 / 255.0, 60 / 255.0, 60 / 255.0)
