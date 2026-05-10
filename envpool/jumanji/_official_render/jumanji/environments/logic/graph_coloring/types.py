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

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from dataclasses import dataclass
from typing import NamedTuple


@dataclass
class State:
    """
    adj_matrix: adjacency matrix used to represent the graph.
    colors: array giving the color index of each node.
    current_node_index: current node being colored.
    action_mask: binary mask indicating the validity of assigning a color to the current node.
    key: random key used for auto-reset.
    """

    adj_matrix: Any  # (num_nodes, num_nodes)
    colors: Any  # (num_nodes,)
    current_node_index: Any  # ()
    action_mask: Any  # (num_colors,)
    key: Any  # (2,)


class Observation(NamedTuple):
    """
    adj_matrix: adjacency matrix used to represent the graph.
    colors: array giving the color index of each node.
    current_node_index: current node being colored.
    action_mask: binary mask indicating the validity of assigning a color to the current node.
    """

    adj_matrix: Any  # (num_nodes, num_nodes)
    colors: Any  # (num_nodes,)
    current_node_index: Any  # ()
    action_mask: Any  # (num_colors,)
