/*
 * Copyright 2026 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "envpool/core/py_envpool.h"
#include "envpool/jumanji/jumanji_env.h"

using Game2048EnvSpec = PyEnvSpec<jumanji::Game2048EnvSpec>;
using Game2048EnvPool = PyEnvPool<jumanji::Game2048EnvPool>;
using GraphColoringEnvSpec = PyEnvSpec<jumanji::GraphColoringEnvSpec>;
using GraphColoringEnvPool = PyEnvPool<jumanji::GraphColoringEnvPool>;
using MinesweeperEnvSpec = PyEnvSpec<jumanji::MinesweeperEnvSpec>;
using MinesweeperEnvPool = PyEnvPool<jumanji::MinesweeperEnvPool>;
using RubiksCubeEnvSpec = PyEnvSpec<jumanji::RubiksCubeEnvSpec>;
using RubiksCubeEnvPool = PyEnvPool<jumanji::RubiksCubeEnvPool>;
using RubiksCubePartlyScrambledEnvSpec =
    PyEnvSpec<jumanji::RubiksCubePartlyScrambledEnvSpec>;
using RubiksCubePartlyScrambledEnvPool =
    PyEnvPool<jumanji::RubiksCubePartlyScrambledEnvPool>;
using SudokuEnvSpec = PyEnvSpec<jumanji::SudokuEnvSpec>;
using SudokuEnvPool = PyEnvPool<jumanji::SudokuEnvPool>;
using BinPackEnvSpec = PyEnvSpec<jumanji::BinPackEnvSpec>;
using BinPackEnvPool = PyEnvPool<jumanji::BinPackEnvPool>;
using FlatPackEnvSpec = PyEnvSpec<jumanji::FlatPackEnvSpec>;
using FlatPackEnvPool = PyEnvPool<jumanji::FlatPackEnvPool>;
using JobShopEnvSpec = PyEnvSpec<jumanji::JobShopEnvSpec>;
using JobShopEnvPool = PyEnvPool<jumanji::JobShopEnvPool>;
using KnapsackEnvSpec = PyEnvSpec<jumanji::KnapsackEnvSpec>;
using KnapsackEnvPool = PyEnvPool<jumanji::KnapsackEnvPool>;
using TetrisEnvSpec = PyEnvSpec<jumanji::TetrisEnvSpec>;
using TetrisEnvPool = PyEnvPool<jumanji::TetrisEnvPool>;
using CleanerEnvSpec = PyEnvSpec<jumanji::CleanerEnvSpec>;
using CleanerEnvPool = PyEnvPool<jumanji::CleanerEnvPool>;
using ConnectorEnvSpec = PyEnvSpec<jumanji::ConnectorEnvSpec>;
using ConnectorEnvPool = PyEnvPool<jumanji::ConnectorEnvPool>;
using CVRPEnvSpec = PyEnvSpec<jumanji::CVRPEnvSpec>;
using CVRPEnvPool = PyEnvPool<jumanji::CVRPEnvPool>;
using MultiCVRPEnvSpec = PyEnvSpec<jumanji::MultiCVRPEnvSpec>;
using MultiCVRPEnvPool = PyEnvPool<jumanji::MultiCVRPEnvPool>;
using MazeEnvSpec = PyEnvSpec<jumanji::MazeEnvSpec>;
using MazeEnvPool = PyEnvPool<jumanji::MazeEnvPool>;
using MMSTEnvSpec = PyEnvSpec<jumanji::MMSTEnvSpec>;
using MMSTEnvPool = PyEnvPool<jumanji::MMSTEnvPool>;
using RobotWarehouseEnvSpec = PyEnvSpec<jumanji::RobotWarehouseEnvSpec>;
using RobotWarehouseEnvPool = PyEnvPool<jumanji::RobotWarehouseEnvPool>;
using SnakeEnvSpec = PyEnvSpec<jumanji::SnakeEnvSpec>;
using SnakeEnvPool = PyEnvPool<jumanji::SnakeEnvPool>;
using TSPEnvSpec = PyEnvSpec<jumanji::TSPEnvSpec>;
using TSPEnvPool = PyEnvPool<jumanji::TSPEnvPool>;
using PacManEnvSpec = PyEnvSpec<jumanji::PacManEnvSpec>;
using PacManEnvPool = PyEnvPool<jumanji::PacManEnvPool>;
using SlidingTilePuzzleEnvSpec = PyEnvSpec<jumanji::SlidingTilePuzzleEnvSpec>;
using SlidingTilePuzzleEnvPool = PyEnvPool<jumanji::SlidingTilePuzzleEnvPool>;
using LevelBasedForagingEnvSpec = PyEnvSpec<jumanji::LevelBasedForagingEnvSpec>;
using LevelBasedForagingEnvPool = PyEnvPool<jumanji::LevelBasedForagingEnvPool>;
using SearchAndRescueEnvSpec = PyEnvSpec<jumanji::SearchAndRescueEnvSpec>;
using SearchAndRescueEnvPool = PyEnvPool<jumanji::SearchAndRescueEnvPool>;
using SokobanEnvSpec = PyEnvSpec<jumanji::SokobanEnvSpec>;
using SokobanEnvPool = PyEnvPool<jumanji::SokobanEnvPool>;

PYBIND11_MODULE(jumanji_envpool, m) {
  REGISTER(m, Game2048EnvSpec, Game2048EnvPool)
  REGISTER(m, GraphColoringEnvSpec, GraphColoringEnvPool)
  REGISTER(m, MinesweeperEnvSpec, MinesweeperEnvPool)
  REGISTER(m, RubiksCubeEnvSpec, RubiksCubeEnvPool)
  REGISTER(m, RubiksCubePartlyScrambledEnvSpec,
           RubiksCubePartlyScrambledEnvPool)
  REGISTER(m, SudokuEnvSpec, SudokuEnvPool)
  REGISTER(m, BinPackEnvSpec, BinPackEnvPool)
  REGISTER(m, FlatPackEnvSpec, FlatPackEnvPool)
  REGISTER(m, JobShopEnvSpec, JobShopEnvPool)
  REGISTER(m, KnapsackEnvSpec, KnapsackEnvPool)
  REGISTER(m, TetrisEnvSpec, TetrisEnvPool)
  REGISTER(m, CleanerEnvSpec, CleanerEnvPool)
  REGISTER(m, ConnectorEnvSpec, ConnectorEnvPool)
  REGISTER(m, CVRPEnvSpec, CVRPEnvPool)
  REGISTER(m, MultiCVRPEnvSpec, MultiCVRPEnvPool)
  REGISTER(m, MazeEnvSpec, MazeEnvPool)
  REGISTER(m, MMSTEnvSpec, MMSTEnvPool)
  REGISTER(m, RobotWarehouseEnvSpec, RobotWarehouseEnvPool)
  REGISTER(m, SnakeEnvSpec, SnakeEnvPool)
  REGISTER(m, TSPEnvSpec, TSPEnvPool)
  REGISTER(m, PacManEnvSpec, PacManEnvPool)
  REGISTER(m, SlidingTilePuzzleEnvSpec, SlidingTilePuzzleEnvPool)
  REGISTER(m, LevelBasedForagingEnvSpec, LevelBasedForagingEnvPool)
  REGISTER(m, SearchAndRescueEnvSpec, SearchAndRescueEnvPool)
  REGISTER(m, SokobanEnvSpec, SokobanEnvPool)
}
