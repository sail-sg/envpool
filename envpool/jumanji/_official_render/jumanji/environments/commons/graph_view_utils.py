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

from typing import List, Tuple

import chex
import numpy as np


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
    graph: chex.Array,
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
    graph: chex.Array, num_nodes: int, seed: int = 42, iterations: int = 100
) -> List[Tuple[float, float]]:
    """
    Compute a 2D spring layout for the given graph using
    the Fruchterman-Reingold force-directed algorithm.

    The algorithm computes a layout by simulating the graph as a physical system,
    where nodes are repelling each other and edges are attracting connected nodes.
    The method minimizes the energy of the system over several iterations.

    Args:
        graph: A Graph object representing the adjacency matrix of the graph.
        num_nodes: Number of graph nodes.
        seed: An integer used to seed the random number generator for reproducibility.
        iterations: Number of layout refining iterations.

    Returns:
        A list of tuples representing the 2D positions of nodes in the graph.
    """
    rng = np.random.default_rng(seed)
    pos = rng.random((num_nodes, 2)) * 2 - 1

    k = np.sqrt(5 / num_nodes)
    temperature = 2.0  # Added a temperature variable

    for _ in range(iterations):
        repulsive_forces = _compute_repulsive_forces(np.zeros((num_nodes, 2)), pos, k, num_nodes)
        attractive_forces = _compute_attractive_forces(
            graph, np.zeros((num_nodes, 2)), pos, k, num_nodes
        )

        pos += (repulsive_forces + attractive_forces) * temperature
        # Reduce the temperature (cooling factor) to refine the layout.
        temperature *= 0.9

        pos = np.clip(pos, -1, 1)  # Keep positions within the [-1, 1] range

    # Scale positions to fill figure
    pos_max = np.max(pos, axis=0)
    pos_min = np.min(pos, axis=0)
    pos = 0.05 + (pos - pos_min) / (1.1 * (pos_max - pos_min))

    return [(float(p[0]), float(p[1])) for p in pos]
