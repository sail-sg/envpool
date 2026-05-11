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
"""PGX env registration."""

from typing import Any

from envpool.registration import register

_COMMON: dict[str, Any] = {
    "import_path": "envpool.pgx",
    "spec_cls": "GoEnvSpec",
    "dm_cls": "GoDMEnvPool",
    "gymnasium_cls": "GoGymnasiumEnvPool",
    "max_num_players": 2,
    "komi": 7.5,
    "history_length": 8,
    "max_terminal_steps": 0,
    "rules": "pgx",
}

register(
    task_id="Go9x9-v1",
    board_size=9,
    task="go_9x9",
    **_COMMON,
)

register(
    task_id="Go13x13-v1",
    board_size=13,
    task="go_13x13",
    **_COMMON,
)

register(
    task_id="Go19x19-v1",
    board_size=19,
    task="go_19x19",
    **_COMMON,
)

_CHINESE_COMMON: dict[str, Any] = {
    **_COMMON,
    "rules": "chinese",
}

register(
    task_id="ChineseGo9x9-v1",
    board_size=9,
    task="go_chinese_9x9",
    **_CHINESE_COMMON,
)

register(
    task_id="ChineseGo13x13-v1",
    board_size=13,
    task="go_chinese_13x13",
    **_CHINESE_COMMON,
)

register(
    task_id="ChineseGo19x19-v1",
    board_size=19,
    task="go_chinese_19x19",
    **_CHINESE_COMMON,
)

_BOARD_GAMES: tuple[dict[str, Any], ...] = (
    {
        "task_id": "TicTacToe-v1",
        "task": "tic_tac_toe",
        "spec_cls": "TicTacToeEnvSpec",
        "dm_cls": "TicTacToeDMEnvPool",
        "gymnasium_cls": "TicTacToeGymnasiumEnvPool",
        "max_num_players": 2,
    },
    {
        "task_id": "ConnectFour-v1",
        "task": "connect_four",
        "spec_cls": "ConnectFourEnvSpec",
        "dm_cls": "ConnectFourDMEnvPool",
        "gymnasium_cls": "ConnectFourGymnasiumEnvPool",
        "max_num_players": 2,
    },
    {
        "task_id": "Hex-v1",
        "task": "hex",
        "spec_cls": "HexEnvSpec",
        "dm_cls": "HexDMEnvPool",
        "gymnasium_cls": "HexGymnasiumEnvPool",
        "max_num_players": 2,
    },
    {
        "task_id": "Othello-v1",
        "task": "othello",
        "spec_cls": "OthelloEnvSpec",
        "dm_cls": "OthelloDMEnvPool",
        "gymnasium_cls": "OthelloGymnasiumEnvPool",
        "max_num_players": 2,
    },
    {
        "task_id": "KuhnPoker-v1",
        "task": "kuhn_poker",
        "spec_cls": "KuhnPokerEnvSpec",
        "dm_cls": "KuhnPokerDMEnvPool",
        "gymnasium_cls": "KuhnPokerGymnasiumEnvPool",
        "max_num_players": 2,
    },
    {
        "task_id": "LeducHoldem-v1",
        "task": "leduc_holdem",
        "spec_cls": "LeducHoldemEnvSpec",
        "dm_cls": "LeducHoldemDMEnvPool",
        "gymnasium_cls": "LeducHoldemGymnasiumEnvPool",
        "max_num_players": 2,
    },
    {
        "task_id": "Play2048-v1",
        "task": "2048",
        "spec_cls": "Play2048EnvSpec",
        "dm_cls": "Play2048DMEnvPool",
        "gymnasium_cls": "Play2048GymnasiumEnvPool",
        "max_num_players": 1,
    },
    {
        "task_id": "AnimalShogi-v1",
        "task": "animal_shogi",
        "spec_cls": "AnimalShogiEnvSpec",
        "dm_cls": "AnimalShogiDMEnvPool",
        "gymnasium_cls": "AnimalShogiGymnasiumEnvPool",
        "max_num_players": 2,
    },
    {
        "task_id": "Backgammon-v1",
        "task": "backgammon",
        "spec_cls": "BackgammonEnvSpec",
        "dm_cls": "BackgammonDMEnvPool",
        "gymnasium_cls": "BackgammonGymnasiumEnvPool",
        "max_num_players": 2,
    },
    {
        "task_id": "Chess-v1",
        "task": "chess",
        "spec_cls": "ChessEnvSpec",
        "dm_cls": "ChessDMEnvPool",
        "gymnasium_cls": "ChessGymnasiumEnvPool",
        "max_num_players": 2,
    },
    {
        "task_id": "GardnerChess-v1",
        "task": "gardner_chess",
        "spec_cls": "GardnerChessEnvSpec",
        "dm_cls": "GardnerChessDMEnvPool",
        "gymnasium_cls": "GardnerChessGymnasiumEnvPool",
        "max_num_players": 2,
    },
    {
        "task_id": "Shogi-v1",
        "task": "shogi",
        "spec_cls": "ShogiEnvSpec",
        "dm_cls": "ShogiDMEnvPool",
        "gymnasium_cls": "ShogiGymnasiumEnvPool",
        "max_num_players": 2,
    },
    {
        "task_id": "SparrowMahjong-v1",
        "task": "sparrow_mahjong",
        "spec_cls": "SparrowMahjongEnvSpec",
        "dm_cls": "SparrowMahjongDMEnvPool",
        "gymnasium_cls": "SparrowMahjongGymnasiumEnvPool",
        "max_num_players": 3,
    },
)

for kwargs in _BOARD_GAMES:
    register(import_path="envpool.pgx", **kwargs)
