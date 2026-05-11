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
"""PGX game environments in EnvPool."""

from envpool.python.api import py_env

from .pgx_envpool import (
    _AnimalShogiEnvPool,
    _AnimalShogiEnvSpec,
    _BackgammonEnvPool,
    _BackgammonEnvSpec,
    _ChessEnvPool,
    _ChessEnvSpec,
    _ConnectFourEnvPool,
    _ConnectFourEnvSpec,
    _GardnerChessEnvPool,
    _GardnerChessEnvSpec,
    _GoEnvPool,
    _GoEnvSpec,
    _HexEnvPool,
    _HexEnvSpec,
    _KuhnPokerEnvPool,
    _KuhnPokerEnvSpec,
    _LeducHoldemEnvPool,
    _LeducHoldemEnvSpec,
    _OthelloEnvPool,
    _OthelloEnvSpec,
    _Play2048EnvPool,
    _Play2048EnvSpec,
    _ShogiEnvPool,
    _ShogiEnvSpec,
    _SparrowMahjongEnvPool,
    _SparrowMahjongEnvSpec,
    _TicTacToeEnvPool,
    _TicTacToeEnvSpec,
)

(
    GoEnvSpec,
    GoDMEnvPool,
    GoGymnasiumEnvPool,
) = py_env(_GoEnvSpec, _GoEnvPool)
(
    TicTacToeEnvSpec,
    TicTacToeDMEnvPool,
    TicTacToeGymnasiumEnvPool,
) = py_env(_TicTacToeEnvSpec, _TicTacToeEnvPool)
(
    ConnectFourEnvSpec,
    ConnectFourDMEnvPool,
    ConnectFourGymnasiumEnvPool,
) = py_env(_ConnectFourEnvSpec, _ConnectFourEnvPool)
(
    HexEnvSpec,
    HexDMEnvPool,
    HexGymnasiumEnvPool,
) = py_env(_HexEnvSpec, _HexEnvPool)
(
    OthelloEnvSpec,
    OthelloDMEnvPool,
    OthelloGymnasiumEnvPool,
) = py_env(_OthelloEnvSpec, _OthelloEnvPool)
(
    KuhnPokerEnvSpec,
    KuhnPokerDMEnvPool,
    KuhnPokerGymnasiumEnvPool,
) = py_env(_KuhnPokerEnvSpec, _KuhnPokerEnvPool)
(
    LeducHoldemEnvSpec,
    LeducHoldemDMEnvPool,
    LeducHoldemGymnasiumEnvPool,
) = py_env(_LeducHoldemEnvSpec, _LeducHoldemEnvPool)
(
    Play2048EnvSpec,
    Play2048DMEnvPool,
    Play2048GymnasiumEnvPool,
) = py_env(_Play2048EnvSpec, _Play2048EnvPool)
(
    AnimalShogiEnvSpec,
    AnimalShogiDMEnvPool,
    AnimalShogiGymnasiumEnvPool,
) = py_env(_AnimalShogiEnvSpec, _AnimalShogiEnvPool)
(
    BackgammonEnvSpec,
    BackgammonDMEnvPool,
    BackgammonGymnasiumEnvPool,
) = py_env(_BackgammonEnvSpec, _BackgammonEnvPool)
(
    ChessEnvSpec,
    ChessDMEnvPool,
    ChessGymnasiumEnvPool,
) = py_env(_ChessEnvSpec, _ChessEnvPool)
(
    GardnerChessEnvSpec,
    GardnerChessDMEnvPool,
    GardnerChessGymnasiumEnvPool,
) = py_env(_GardnerChessEnvSpec, _GardnerChessEnvPool)
(
    ShogiEnvSpec,
    ShogiDMEnvPool,
    ShogiGymnasiumEnvPool,
) = py_env(_ShogiEnvSpec, _ShogiEnvPool)
(
    SparrowMahjongEnvSpec,
    SparrowMahjongDMEnvPool,
    SparrowMahjongGymnasiumEnvPool,
) = py_env(_SparrowMahjongEnvSpec, _SparrowMahjongEnvPool)

__all__ = [
    "AnimalShogiDMEnvPool",
    "AnimalShogiEnvSpec",
    "AnimalShogiGymnasiumEnvPool",
    "BackgammonDMEnvPool",
    "BackgammonEnvSpec",
    "BackgammonGymnasiumEnvPool",
    "ChessDMEnvPool",
    "ChessEnvSpec",
    "ChessGymnasiumEnvPool",
    "ConnectFourDMEnvPool",
    "ConnectFourEnvSpec",
    "ConnectFourGymnasiumEnvPool",
    "GardnerChessDMEnvPool",
    "GardnerChessEnvSpec",
    "GardnerChessGymnasiumEnvPool",
    "GoEnvSpec",
    "GoDMEnvPool",
    "GoGymnasiumEnvPool",
    "HexDMEnvPool",
    "HexEnvSpec",
    "HexGymnasiumEnvPool",
    "KuhnPokerDMEnvPool",
    "KuhnPokerEnvSpec",
    "KuhnPokerGymnasiumEnvPool",
    "LeducHoldemDMEnvPool",
    "LeducHoldemEnvSpec",
    "LeducHoldemGymnasiumEnvPool",
    "OthelloDMEnvPool",
    "OthelloEnvSpec",
    "OthelloGymnasiumEnvPool",
    "Play2048DMEnvPool",
    "Play2048EnvSpec",
    "Play2048GymnasiumEnvPool",
    "ShogiDMEnvPool",
    "ShogiEnvSpec",
    "ShogiGymnasiumEnvPool",
    "SparrowMahjongDMEnvPool",
    "SparrowMahjongEnvSpec",
    "SparrowMahjongGymnasiumEnvPool",
    "TicTacToeDMEnvPool",
    "TicTacToeEnvSpec",
    "TicTacToeGymnasiumEnvPool",
]
