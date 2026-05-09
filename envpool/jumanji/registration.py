# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Jumanji v1.1.1 env registration."""

from __future__ import annotations

from envpool.registration import register

JUMANJI_ORACLE_VERSION = "1.1.1"
JUMANJI_ORACLE_COMMIT = "b668afc08a14a71d7eed3f618456d5bda2eea06e"

_COMMON = {
    "import_path": "envpool.jumanji",
}

_TASKS: tuple[tuple[str, str, int], ...] = (
    ("BinPack-v2", "BinPack", 20),
    ("CVRP-v1", "CVRP", 40),
    ("Cleaner-v0", "Cleaner", 100),
    ("Connector-v2", "Connector", 50),
    ("FlatPack-v0", "FlatPack", 25),
    ("Game2048-v1", "Game2048", 1000),
    ("GraphColoring-v1", "GraphColoring", 20),
    ("JobShop-v0", "JobShop", 1000),
    ("Knapsack-v1", "Knapsack", 50),
    ("LevelBasedForaging-v0", "LevelBasedForaging", 100),
    ("MMST-v0", "MMST", 70),
    ("Maze-v0", "Maze", 100),
    ("Minesweeper-v0", "Minesweeper", 90),
    ("MultiCVRP-v0", "MultiCVRP", 40),
    ("PacMan-v1", "PacMan", 1000),
    ("RobotWarehouse-v0", "RobotWarehouse", 500),
    ("RubiksCube-partly-scrambled-v0", "RubiksCubePartlyScrambled", 20),
    ("RubiksCube-v0", "RubiksCube", 200),
    ("SearchAndRescue-v0", "SearchAndRescue", 400),
    ("SlidingTilePuzzle-v0", "SlidingTilePuzzle", 500),
    ("Snake-v1", "Snake", 4000),
    ("Sokoban-v0", "Sokoban", 120),
    ("Sudoku-v0", "Sudoku", 81),
    ("Sudoku-very-easy-v0", "Sudoku", 81),
    ("TSP-v1", "TSP", 20),
    ("Tetris-v0", "Tetris", 400),
)

jumanji_env_ids = [task_id for task_id, _, _ in _TASKS]
jumanji_envpool_task_ids = [f"Jumanji/{task_id}" for task_id in jumanji_env_ids]

for task_id, class_prefix, max_episode_steps in _TASKS:
    register(
        task_id=task_id,
        aliases=(f"Jumanji/{task_id}",),
        spec_cls=f"{class_prefix}EnvSpec",
        dm_cls=f"{class_prefix}DMEnvPool",
        gymnasium_cls=f"{class_prefix}GymnasiumEnvPool",
        max_episode_steps=max_episode_steps,
        **_COMMON,
    )
