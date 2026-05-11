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
"""Jumanji environments in EnvPool."""

from envpool.python.api import py_env

from .jumanji_envpool import (
    _BinPackEnvPool,
    _BinPackEnvSpec,
    _CleanerEnvPool,
    _CleanerEnvSpec,
    _ConnectorEnvPool,
    _ConnectorEnvSpec,
    _CVRPEnvPool,
    _CVRPEnvSpec,
    _FlatPackEnvPool,
    _FlatPackEnvSpec,
    _Game2048EnvPool,
    _Game2048EnvSpec,
    _GraphColoringEnvPool,
    _GraphColoringEnvSpec,
    _JobShopEnvPool,
    _JobShopEnvSpec,
    _KnapsackEnvPool,
    _KnapsackEnvSpec,
    _LevelBasedForagingEnvPool,
    _LevelBasedForagingEnvSpec,
    _MazeEnvPool,
    _MazeEnvSpec,
    _MinesweeperEnvPool,
    _MinesweeperEnvSpec,
    _MMSTEnvPool,
    _MMSTEnvSpec,
    _MultiCVRPEnvPool,
    _MultiCVRPEnvSpec,
    _PacManEnvPool,
    _PacManEnvSpec,
    _RobotWarehouseEnvPool,
    _RobotWarehouseEnvSpec,
    _RubiksCubeEnvPool,
    _RubiksCubeEnvSpec,
    _RubiksCubePartlyScrambledEnvPool,
    _RubiksCubePartlyScrambledEnvSpec,
    _SearchAndRescueEnvPool,
    _SearchAndRescueEnvSpec,
    _SlidingTilePuzzleEnvPool,
    _SlidingTilePuzzleEnvSpec,
    _SnakeEnvPool,
    _SnakeEnvSpec,
    _SokobanEnvPool,
    _SokobanEnvSpec,
    _SudokuEnvPool,
    _SudokuEnvSpec,
    _TetrisEnvPool,
    _TetrisEnvSpec,
    _TSPEnvPool,
    _TSPEnvSpec,
)
from .jumanji_render import with_jumanji_python_render

(
    Game2048EnvSpec,
    Game2048DMEnvPool,
    Game2048GymnasiumEnvPool,
) = py_env(_Game2048EnvSpec, _Game2048EnvPool)
(
    GraphColoringEnvSpec,
    GraphColoringDMEnvPool,
    GraphColoringGymnasiumEnvPool,
) = py_env(_GraphColoringEnvSpec, _GraphColoringEnvPool)
(
    MinesweeperEnvSpec,
    MinesweeperDMEnvPool,
    MinesweeperGymnasiumEnvPool,
) = py_env(_MinesweeperEnvSpec, _MinesweeperEnvPool)
(
    RubiksCubeEnvSpec,
    RubiksCubeDMEnvPool,
    RubiksCubeGymnasiumEnvPool,
) = py_env(_RubiksCubeEnvSpec, _RubiksCubeEnvPool)
(
    RubiksCubePartlyScrambledEnvSpec,
    RubiksCubePartlyScrambledDMEnvPool,
    RubiksCubePartlyScrambledGymnasiumEnvPool,
) = py_env(
    _RubiksCubePartlyScrambledEnvSpec,
    _RubiksCubePartlyScrambledEnvPool,
)
(
    SudokuEnvSpec,
    SudokuDMEnvPool,
    SudokuGymnasiumEnvPool,
) = py_env(_SudokuEnvSpec, _SudokuEnvPool)
(
    BinPackEnvSpec,
    BinPackDMEnvPool,
    BinPackGymnasiumEnvPool,
) = py_env(_BinPackEnvSpec, _BinPackEnvPool)
(
    FlatPackEnvSpec,
    FlatPackDMEnvPool,
    FlatPackGymnasiumEnvPool,
) = py_env(_FlatPackEnvSpec, _FlatPackEnvPool)
(
    JobShopEnvSpec,
    JobShopDMEnvPool,
    JobShopGymnasiumEnvPool,
) = py_env(_JobShopEnvSpec, _JobShopEnvPool)
(
    KnapsackEnvSpec,
    KnapsackDMEnvPool,
    KnapsackGymnasiumEnvPool,
) = py_env(_KnapsackEnvSpec, _KnapsackEnvPool)
(
    TetrisEnvSpec,
    TetrisDMEnvPool,
    TetrisGymnasiumEnvPool,
) = py_env(_TetrisEnvSpec, _TetrisEnvPool)
(
    CleanerEnvSpec,
    CleanerDMEnvPool,
    CleanerGymnasiumEnvPool,
) = py_env(_CleanerEnvSpec, _CleanerEnvPool)
(
    ConnectorEnvSpec,
    ConnectorDMEnvPool,
    ConnectorGymnasiumEnvPool,
) = py_env(_ConnectorEnvSpec, _ConnectorEnvPool)
(
    CVRPEnvSpec,
    CVRPDMEnvPool,
    CVRPGymnasiumEnvPool,
) = py_env(_CVRPEnvSpec, _CVRPEnvPool)
(
    MultiCVRPEnvSpec,
    MultiCVRPDMEnvPool,
    MultiCVRPGymnasiumEnvPool,
) = py_env(_MultiCVRPEnvSpec, _MultiCVRPEnvPool)
(
    MazeEnvSpec,
    MazeDMEnvPool,
    MazeGymnasiumEnvPool,
) = py_env(_MazeEnvSpec, _MazeEnvPool)
(
    MMSTEnvSpec,
    MMSTDMEnvPool,
    MMSTGymnasiumEnvPool,
) = py_env(_MMSTEnvSpec, _MMSTEnvPool)
(
    RobotWarehouseEnvSpec,
    RobotWarehouseDMEnvPool,
    RobotWarehouseGymnasiumEnvPool,
) = py_env(_RobotWarehouseEnvSpec, _RobotWarehouseEnvPool)
(
    SnakeEnvSpec,
    SnakeDMEnvPool,
    SnakeGymnasiumEnvPool,
) = py_env(_SnakeEnvSpec, _SnakeEnvPool)
(
    TSPEnvSpec,
    TSPDMEnvPool,
    TSPGymnasiumEnvPool,
) = py_env(_TSPEnvSpec, _TSPEnvPool)
(
    PacManEnvSpec,
    PacManDMEnvPool,
    PacManGymnasiumEnvPool,
) = py_env(_PacManEnvSpec, _PacManEnvPool)
(
    SlidingTilePuzzleEnvSpec,
    SlidingTilePuzzleDMEnvPool,
    SlidingTilePuzzleGymnasiumEnvPool,
) = py_env(_SlidingTilePuzzleEnvSpec, _SlidingTilePuzzleEnvPool)
(
    LevelBasedForagingEnvSpec,
    LevelBasedForagingDMEnvPool,
    LevelBasedForagingGymnasiumEnvPool,
) = py_env(_LevelBasedForagingEnvSpec, _LevelBasedForagingEnvPool)
(
    SearchAndRescueEnvSpec,
    SearchAndRescueDMEnvPool,
    SearchAndRescueGymnasiumEnvPool,
) = py_env(_SearchAndRescueEnvSpec, _SearchAndRescueEnvPool)
(
    SokobanEnvSpec,
    SokobanDMEnvPool,
    SokobanGymnasiumEnvPool,
) = py_env(_SokobanEnvSpec, _SokobanEnvPool)

BinPackGymnasiumEnvPool = with_jumanji_python_render(
    BinPackGymnasiumEnvPool, "BinPack-v2"
)
CVRPGymnasiumEnvPool = with_jumanji_python_render(
    CVRPGymnasiumEnvPool, "CVRP-v1"
)
CleanerGymnasiumEnvPool = with_jumanji_python_render(
    CleanerGymnasiumEnvPool, "Cleaner-v0"
)
ConnectorGymnasiumEnvPool = with_jumanji_python_render(
    ConnectorGymnasiumEnvPool, "Connector-v2"
)
FlatPackGymnasiumEnvPool = with_jumanji_python_render(
    FlatPackGymnasiumEnvPool, "FlatPack-v0"
)
Game2048GymnasiumEnvPool = with_jumanji_python_render(
    Game2048GymnasiumEnvPool, "Game2048-v1"
)
GraphColoringGymnasiumEnvPool = with_jumanji_python_render(
    GraphColoringGymnasiumEnvPool, "GraphColoring-v1"
)
JobShopGymnasiumEnvPool = with_jumanji_python_render(
    JobShopGymnasiumEnvPool, "JobShop-v0"
)
KnapsackGymnasiumEnvPool = with_jumanji_python_render(
    KnapsackGymnasiumEnvPool, "Knapsack-v1"
)
LevelBasedForagingGymnasiumEnvPool = with_jumanji_python_render(
    LevelBasedForagingGymnasiumEnvPool, "LevelBasedForaging-v0"
)
MMSTGymnasiumEnvPool = with_jumanji_python_render(
    MMSTGymnasiumEnvPool, "MMST-v0"
)
MazeGymnasiumEnvPool = with_jumanji_python_render(
    MazeGymnasiumEnvPool, "Maze-v0"
)
MinesweeperGymnasiumEnvPool = with_jumanji_python_render(
    MinesweeperGymnasiumEnvPool, "Minesweeper-v0"
)
MultiCVRPGymnasiumEnvPool = with_jumanji_python_render(
    MultiCVRPGymnasiumEnvPool, "MultiCVRP-v0"
)
PacManGymnasiumEnvPool = with_jumanji_python_render(
    PacManGymnasiumEnvPool, "PacMan-v1"
)
RobotWarehouseGymnasiumEnvPool = with_jumanji_python_render(
    RobotWarehouseGymnasiumEnvPool, "RobotWarehouse-v0"
)
RubiksCubeGymnasiumEnvPool = with_jumanji_python_render(
    RubiksCubeGymnasiumEnvPool, "RubiksCube-v0"
)
RubiksCubePartlyScrambledGymnasiumEnvPool = with_jumanji_python_render(
    RubiksCubePartlyScrambledGymnasiumEnvPool,
    "RubiksCube-partly-scrambled-v0",
)
SearchAndRescueGymnasiumEnvPool = with_jumanji_python_render(
    SearchAndRescueGymnasiumEnvPool, "SearchAndRescue-v0"
)
SlidingTilePuzzleGymnasiumEnvPool = with_jumanji_python_render(
    SlidingTilePuzzleGymnasiumEnvPool, "SlidingTilePuzzle-v0"
)
SnakeGymnasiumEnvPool = with_jumanji_python_render(
    SnakeGymnasiumEnvPool, "Snake-v1"
)
SokobanGymnasiumEnvPool = with_jumanji_python_render(
    SokobanGymnasiumEnvPool, "Sokoban-v0"
)
SudokuGymnasiumEnvPool = with_jumanji_python_render(
    SudokuGymnasiumEnvPool, "Sudoku-v0"
)
TetrisGymnasiumEnvPool = with_jumanji_python_render(
    TetrisGymnasiumEnvPool, "Tetris-v0"
)
TSPGymnasiumEnvPool = with_jumanji_python_render(TSPGymnasiumEnvPool, "TSP-v1")
