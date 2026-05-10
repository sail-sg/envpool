# ruff: noqa
# fmt: off
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
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np


@dataclass
class State:
    """
    node_types: Array with node types (-1 represents utility nodes).
    adj_matrix: Array with adjacency matrix.
    connected_nodes: Array of node indices denoting route (-1 --> not filled yet).
    connected_nodes_index: Array tracking connected nodes.
    nodes_to_connect: array with the nodes that each agent needs to connect.
    node_edges: Array used to track active edges.
    position: Array with agents current positions.
    position_index: Array with current index position in connected_nodes.
    action_mask: array with current action mask for each agent.
    finished_agents: Array indicating if an agent's nodes are fully connected.
    step_count: integer to keep track of the number of steps.
    key: state PRNGkey.
    """

    node_types: Any  # (num_nodes,)
    adj_matrix: Any  # (num_nodes, num_nodes)
    connected_nodes: Any  # (num_agents, time_limit)
    connected_nodes_index: Any  # (num_agents, num_nodes)
    nodes_to_connect: Any  # (num_agents, num_nodes_to_connect_per_agent)
    node_edges: Any  # (num_agents, num_nodes, num_nodes)
    positions: Any  # (num_agents,)
    position_index: Any  # (num_agents,)
    action_mask: Any  # (num_agents, num_nodes)
    finished_agents: Any  # (num_agents,)
    step_count: np.int32  # ()
    key: Any  # (2,)


class Observation(NamedTuple):
    """
    node_types: Array representing the types of nodes in the problem.
        For example, if we have 12 nodes, their indices are 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11.
        Let's consider we have 2 agents. Agent 0 wants to connect nodes (0, 1, 9),
        and agent 1 wants to connect nodes (3, 5, 8).
        The remaining nodes are considered utility nodes.
        Therefore, in the state view, the node_types are
        represented as [0, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, -1].
        When generating the problem, each agent starts from one of its nodes.
        So, if agent 0 starts on node 1 and agent 1 on node 3,
        the connected_nodes array will have values [1, -1, ...] and [3, -1, ...] respectively.
        The agent's observation is represented using the following rules:
        - Each agent should see its connected nodes on the path as 0.
        - Nodes that the agent still needs to connect are represented as 1.
        - The next agent's nodes are represented by 2 and 3, the next by 4 and 5, and so on.
        - Utility unconnected nodes are represented by -1.
        For the 12 node example mentioned above,
        the expected observation view node_types will have the following values:
        node_types = np.array(
            [
                [1, 0, -1, 2, -1, 3, 1, -1, 3, 1, -1, -1],
                [3, 2, -1, 0, -1, 1, 3, -1, 1, 3, -1, -1],
            ],
            dtype=np.int32,
        )
        Note: to make the environment single agent, we use the first agent's observation.

    adj_matrix: Adjacency matrix representing the connections between nodes.

    positions: Current node positions of the agents.
        In our current problem, this will be represented as np.array([1, 3]).

    step_count: integer to keep track of the number of steps.

    action_mask: Binary mask indicating the validity of each action.
        Given the current node on which the agent is located,
        this mask determines if there is a valid edge to every other node.
    """

    node_types: Any  # (num_nodes)
    adj_matrix: Any  # (num_nodes, num_nodes)
    positions: Any  # (num_agents,)
    step_count: np.int32  # ()
    action_mask: Any  # (num_agents, num_nodes)


@dataclass
class Graph:
    """
    nodes: Array with node indices (np.arange(number of nodes)).
    edges: Array with all egdes in the graph.
    edge_codes: Array with edge codes.
    max_degree: (int).
    node_degree: Array with the degree of every node.
    edge_index: (int) index location for the next edge.
    """

    nodes: Any  # (num_nodes,)
    edges: Any  # (num_edges, 2)
    edge_codes: Any  # (num_edges,)
    max_degree: np.int32  # ()
    node_degree: Any  # (num_nodes,)
    edge_index: np.int32  # ()
    node_edges: Any  # (num_nodes, num_nodes)
