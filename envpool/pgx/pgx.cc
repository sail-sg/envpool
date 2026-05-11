// Copyright 2026 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "envpool/core/py_envpool.h"
#include "envpool/pgx/animal_shogi.h"
#include "envpool/pgx/backgammon.h"
#include "envpool/pgx/board_games.h"
#include "envpool/pgx/card_games.h"
#include "envpool/pgx/chess_games.h"
#include "envpool/pgx/go.h"
#include "envpool/pgx/play2048.h"
#include "envpool/pgx/shogi.h"
#include "envpool/pgx/sparrow_mahjong.h"

using GoEnvSpec = PyEnvSpec<pgx::GoEnvSpec>;
using GoEnvPool = PyEnvPool<pgx::GoEnvPool>;
using TicTacToeEnvSpec = PyEnvSpec<pgx::TicTacToeEnvSpec>;
using TicTacToeEnvPool = PyEnvPool<pgx::TicTacToeEnvPool>;
using ConnectFourEnvSpec = PyEnvSpec<pgx::ConnectFourEnvSpec>;
using ConnectFourEnvPool = PyEnvPool<pgx::ConnectFourEnvPool>;
using HexEnvSpec = PyEnvSpec<pgx::HexEnvSpec>;
using HexEnvPool = PyEnvPool<pgx::HexEnvPool>;
using OthelloEnvSpec = PyEnvSpec<pgx::OthelloEnvSpec>;
using OthelloEnvPool = PyEnvPool<pgx::OthelloEnvPool>;
using KuhnPokerEnvSpec = PyEnvSpec<pgx::KuhnPokerEnvSpec>;
using KuhnPokerEnvPool = PyEnvPool<pgx::KuhnPokerEnvPool>;
using LeducHoldemEnvSpec = PyEnvSpec<pgx::LeducHoldemEnvSpec>;
using LeducHoldemEnvPool = PyEnvPool<pgx::LeducHoldemEnvPool>;
using Play2048EnvSpec = PyEnvSpec<pgx::Play2048EnvSpec>;
using Play2048EnvPool = PyEnvPool<pgx::Play2048EnvPool>;
using AnimalShogiEnvSpec = PyEnvSpec<pgx::AnimalShogiEnvSpec>;
using AnimalShogiEnvPool = PyEnvPool<pgx::AnimalShogiEnvPool>;
using BackgammonEnvSpec = PyEnvSpec<pgx::BackgammonEnvSpec>;
using BackgammonEnvPool = PyEnvPool<pgx::BackgammonEnvPool>;
using ChessEnvSpec = PyEnvSpec<pgx::ChessEnvSpec>;
using ChessEnvPool = PyEnvPool<pgx::ChessEnvPool>;
using GardnerChessEnvSpec = PyEnvSpec<pgx::GardnerChessEnvSpec>;
using GardnerChessEnvPool = PyEnvPool<pgx::GardnerChessEnvPool>;
using ShogiEnvSpec = PyEnvSpec<pgx::ShogiEnvSpec>;
using ShogiEnvPool = PyEnvPool<pgx::ShogiEnvPool>;
using SparrowMahjongEnvSpec = PyEnvSpec<pgx::SparrowMahjongEnvSpec>;
using SparrowMahjongEnvPool = PyEnvPool<pgx::SparrowMahjongEnvPool>;

PYBIND11_MODULE(pgx_envpool, m) {
  REGISTER(m, GoEnvSpec, GoEnvPool)
  REGISTER(m, TicTacToeEnvSpec, TicTacToeEnvPool)
  REGISTER(m, ConnectFourEnvSpec, ConnectFourEnvPool)
  REGISTER(m, HexEnvSpec, HexEnvPool)
  REGISTER(m, OthelloEnvSpec, OthelloEnvPool)
  REGISTER(m, KuhnPokerEnvSpec, KuhnPokerEnvPool)
  REGISTER(m, LeducHoldemEnvSpec, LeducHoldemEnvPool)
  REGISTER(m, Play2048EnvSpec, Play2048EnvPool)
  REGISTER(m, AnimalShogiEnvSpec, AnimalShogiEnvPool)
  REGISTER(m, BackgammonEnvSpec, BackgammonEnvPool)
  REGISTER(m, ChessEnvSpec, ChessEnvPool)
  REGISTER(m, GardnerChessEnvSpec, GardnerChessEnvPool)
  REGISTER(m, ShogiEnvSpec, ShogiEnvPool)
  REGISTER(m, SparrowMahjongEnvSpec, SparrowMahjongEnvPool)
}
