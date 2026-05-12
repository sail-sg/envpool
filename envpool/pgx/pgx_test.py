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
"""Tests for PGX environments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from absl.testing import absltest

import envpool.pgx.registration  # noqa: F401
from envpool.registration import make_gymnasium, make_spec

_BOARD_GAME_CASES = (
    ("TicTacToe-v1", (3, 3, 2), 9),
    ("ConnectFour-v1", (6, 7, 2), 7),
    ("Hex-v1", (11, 11, 4), 122),
    ("Othello-v1", (8, 8, 2), 65),
)
_CARD_GAME_CASES = (
    ("KuhnPoker-v1", (7,), 2),
    ("LeducHoldem-v1", (34,), 3),
)
_SINGLE_PLAYER_CASES = (("Play2048-v1", (4, 4, 31), 4),)
_FLOAT_OBS_CASES = (
    ("AnimalShogi-v1", (4, 3, 194), 132),
    ("Chess-v1", (8, 8, 119), 4672),
    ("GardnerChess-v1", (5, 5, 115), 1225),
)
_SHOGI_CASES = (("Shogi-v1", (9, 9, 119), 2187),)
_MAHJONG_CASES = (("SparrowMahjong-v1", (11, 15), 11),)
_INT_OBS_CASES = (("Backgammon-v1", (34,), 156),)


def _make_small_go(**kwargs: Any) -> Any:
    return make_gymnasium(
        "Go9x9-v1",
        board_size=5,
        num_envs=1,
        seed=0,
        **kwargs,
    )


def _make_small_chinese_go(**kwargs: Any) -> Any:
    return make_gymnasium(
        "ChineseGo9x9-v1",
        board_size=5,
        num_envs=1,
        seed=0,
        **kwargs,
    )


def _step(env: Any, action: int) -> Any:
    return env.step(np.asarray([action], dtype=np.int32))


class _PgxGoTest(absltest.TestCase):
    def test_registered_specs_match_pgx_shape(self) -> None:
        cases = (
            ("Go9x9-v1", 9, "pgx"),
            ("Go13x13-v1", 13, "pgx"),
            ("Go19x19-v1", 19, "pgx"),
            ("ChineseGo9x9-v1", 9, "chinese"),
            ("ChineseGo13x13-v1", 13, "chinese"),
            ("ChineseGo19x19-v1", 19, "chinese"),
        )
        for task_id, size, rules in cases:
            with self.subTest(task_id=task_id):
                spec = make_spec(task_id)
                self.assertEqual(spec.config.max_num_players, 2)
                self.assertEqual(spec.config.board_size, size)
                self.assertEqual(spec.config.rules, rules)
                self.assertIsInstance(
                    spec.gymnasium_observation_space, gym.spaces.MultiBinary
                )
                self.assertEqual(
                    spec.gymnasium_observation_space.shape,
                    (size, size, 17),
                )
                self.assertIsInstance(
                    spec.gymnasium_action_space, gym.spaces.Discrete
                )
                self.assertEqual(spec.gymnasium_action_space.n, size * size + 1)

    def test_registered_board_game_specs_match_pgx_shape(self) -> None:
        for task_id, obs_shape, num_actions in (
            *_BOARD_GAME_CASES,
            *_CARD_GAME_CASES,
            *_SINGLE_PLAYER_CASES,
            *_FLOAT_OBS_CASES,
            *_SHOGI_CASES,
            *_MAHJONG_CASES,
            *_INT_OBS_CASES,
        ):
            with self.subTest(task_id=task_id):
                spec = make_spec(task_id)
                self.assertEqual(
                    spec.config.max_num_players,
                    1
                    if task_id == "Play2048-v1"
                    else 3
                    if task_id == "SparrowMahjong-v1"
                    else 2,
                )
                if task_id in (
                    "AnimalShogi-v1",
                    "Backgammon-v1",
                    "Chess-v1",
                    "GardnerChess-v1",
                ):
                    self.assertIsInstance(
                        spec.gymnasium_observation_space,
                        gym.spaces.Box,
                    )
                else:
                    self.assertIsInstance(
                        spec.gymnasium_observation_space,
                        gym.spaces.MultiBinary,
                    )
                self.assertEqual(
                    spec.gymnasium_observation_space.shape,
                    obs_shape,
                )
                self.assertIsInstance(
                    spec.gymnasium_action_space,
                    gym.spaces.Discrete,
                )
                self.assertEqual(spec.gymnasium_action_space.n, num_actions)

    def test_reset_shapes_and_player_metadata(self) -> None:
        env = _make_small_go()
        obs, info = env.reset()
        self.assertEqual(obs.shape, (2, 5, 5, 17))
        self.assertEqual(obs.dtype, np.bool_)
        np.testing.assert_array_equal(info["players"]["id"], [0, 1])
        self.assertIn(int(info["current_player"][0]), (0, 1))
        self.assertEqual(info["legal_action_mask"].shape, (1, 26))
        np.testing.assert_array_equal(info["legal_action_mask"], True)
        self.assertEqual(info["black_area"][0], 25)
        self.assertEqual(info["white_area"][0], 25)

    def test_board_game_reset_shapes_and_player_metadata(self) -> None:
        for task_id, obs_shape, num_actions in (
            *_BOARD_GAME_CASES,
            *_CARD_GAME_CASES,
            *_SINGLE_PLAYER_CASES,
            *_FLOAT_OBS_CASES,
            *_SHOGI_CASES,
            *_MAHJONG_CASES,
            *_INT_OBS_CASES,
        ):
            with self.subTest(task_id=task_id):
                env = make_gymnasium(task_id, num_envs=1, seed=0)
                obs, info = env.reset()
                if task_id == "Play2048-v1":
                    self.assertEqual(obs.shape, (1, *obs_shape))
                elif task_id == "SparrowMahjong-v1":
                    self.assertEqual(obs.shape, (3, *obs_shape))
                else:
                    self.assertEqual(obs.shape, (2, *obs_shape))
                self.assertEqual(
                    obs.dtype,
                    (
                        np.float32
                        if task_id
                        in ("AnimalShogi-v1", "Chess-v1", "GardnerChess-v1")
                        else np.int32
                        if task_id == "Backgammon-v1"
                        else np.bool_
                    ),
                )
                if task_id != "Play2048-v1":
                    expected_players = (
                        [0, 1, 2] if task_id == "SparrowMahjong-v1" else [0, 1]
                    )
                    np.testing.assert_array_equal(
                        info["players"]["id"],
                        expected_players,
                    )
                    self.assertIn(
                        int(info["current_player"][0]),
                        tuple(expected_players),
                    )
                self.assertEqual(
                    info["legal_action_mask"].shape,
                    (1, num_actions),
                )
                if task_id == "Backgammon-v1":
                    self.assertEqual(info["board"].shape, (1, 28))
                elif task_id == "SparrowMahjong-v1":
                    self.assertEqual(info["hands"].shape, (1, 3, 11))
                elif task_id not in ("KuhnPoker-v1", "LeducHoldem-v1"):
                    self.assertEqual(info["board"].shape, (1, *obs_shape[:2]))

    def test_board_game_initial_legal_actions(self) -> None:
        cases = (
            ("TicTacToe-v1", np.ones(9, dtype=np.bool_)),
            ("ConnectFour-v1", np.ones(7, dtype=np.bool_)),
            (
                "Hex-v1",
                np.asarray([True] * 121 + [False], dtype=np.bool_),
            ),
            (
                "Othello-v1",
                np.asarray(
                    [i in (19, 26, 37, 44) for i in range(65)],
                    dtype=np.bool_,
                ),
            ),
            ("KuhnPoker-v1", np.asarray([True, True], dtype=np.bool_)),
            (
                "LeducHoldem-v1",
                np.asarray([True, True, False], dtype=np.bool_),
            ),
        )
        for task_id, expected_mask in cases:
            with self.subTest(task_id=task_id):
                env = make_gymnasium(task_id, num_envs=1, seed=0)
                _, info = env.reset()
                np.testing.assert_array_equal(
                    info["legal_action_mask"][0],
                    expected_mask,
                )

    def test_play2048_reset_has_two_tiles(self) -> None:
        env = make_gymnasium("Play2048-v1", num_envs=1, seed=0)
        obs, info = env.reset()
        board = info["board"][0]
        self.assertEqual(int(np.count_nonzero(board)), 2)
        self.assertEqual(obs.shape, (1, 4, 4, 31))
        for row in range(4):
            for col in range(4):
                self.assertTrue(bool(obs[0, row, col, board[row, col]]))

    def test_end_by_two_consecutive_passes(self) -> None:
        env = _make_small_go()
        env.reset()
        _, _, terminated, truncated, info = _step(env, 25)
        self.assertFalse(bool(terminated[0]))
        self.assertFalse(bool(truncated[0]))
        self.assertEqual(info["consecutive_pass_count"][0], 1)
        _, reward, terminated, truncated, info = _step(env, 25)
        self.assertTrue(bool(terminated[0]))
        self.assertFalse(bool(truncated[0]))
        self.assertEqual(info["consecutive_pass_count"][0], 2)
        np.testing.assert_array_equal(np.sort(reward), [-1.0, 1.0])

    def test_pgx_go_five_by_five_rollout_board(self) -> None:
        env = _make_small_go()
        env.reset()
        actions = [
            12,
            11,
            17,
            7,
            8,
            1,
            3,
            16,
            21,
            2,
            10,
            5,
            14,
            15,
            23,
            20,
            25,
            22,
            19,
            21,
            25,
            25,
        ]
        for action in actions:
            obs, reward, terminated, truncated, info = _step(env, action)

        del obs
        expected_board = np.asarray(
            [
                [0, -1, -1, 1, 0],
                [-1, 0, -1, 1, 0],
                [0, -1, 1, 0, 1],
                [-1, -1, 1, 0, 1],
                [-1, -1, -1, 1, 0],
            ],
            dtype=np.int32,
        )
        self.assertTrue(bool(terminated[0]))
        self.assertFalse(bool(truncated[0]))
        np.testing.assert_array_equal(info["board"][0], expected_board)
        np.testing.assert_array_equal(np.sort(reward), [-1.0, 1.0])

    def test_ko_retake_is_illegal_and_loses(self) -> None:
        env = _make_small_go()
        env.reset()
        for action in (2, 17, 6, 13, 8, 11, 12, 7):
            obs, reward, terminated, truncated, info = _step(env, action)

        del obs, reward, truncated
        self.assertFalse(bool(terminated[0]))
        self.assertEqual(info["ko"][0], 12)
        self.assertFalse(bool(info["legal_action_mask"][0, 12]))
        loser = int(info["current_player"][0])

        _, reward, terminated, truncated, _ = _step(env, 12)
        self.assertTrue(bool(terminated[0]))
        self.assertFalse(bool(truncated[0]))
        self.assertEqual(reward[loser], -1.0)
        self.assertEqual(float(np.sum(reward)), 0.0)

    def test_chinese_rules_empty_board_area_is_neutral(self) -> None:
        env = _make_small_chinese_go()
        _, info = env.reset()
        black_player = int(info["current_player"][0])
        self.assertEqual(info["black_area"][0], 0)
        self.assertEqual(info["white_area"][0], 0)

        _step(env, 25)
        _, reward, terminated, truncated, info = _step(env, 25)
        self.assertTrue(bool(terminated[0]))
        self.assertFalse(bool(truncated[0]))
        self.assertEqual(info["black_area"][0], 0)
        self.assertEqual(info["white_area"][0], 0)
        self.assertEqual(reward[black_player], -1.0)
        self.assertEqual(reward[1 - black_player], 1.0)

    def test_chinese_rules_mask_positional_superko(self) -> None:
        env = make_gymnasium(
            "ChineseGo9x9-v1",
            board_size=3,
            max_terminal_steps=200,
            num_envs=1,
            seed=0,
        )
        actions = [
            5,
            0,
            8,
            3,
            1,
            4,
            6,
            7,
            6,
            3,
            7,
            0,
            2,
            4,
            5,
            7,
            1,
            8,
            6,
            0,
            8,
            3,
            7,
        ]
        _, info = env.reset()
        for action in actions[:-1]:
            self.assertTrue(bool(info["legal_action_mask"][0, action]))
            _, _, terminated, _, info = _step(env, action)
            self.assertFalse(bool(terminated[0]))

        repeated_action = actions[-1]
        loser = int(info["current_player"][0])
        self.assertFalse(bool(info["legal_action_mask"][0, repeated_action]))
        _, reward, terminated, truncated, info = _step(env, repeated_action)
        self.assertTrue(bool(terminated[0]))
        self.assertFalse(bool(truncated[0]))
        self.assertFalse(bool(info["is_psk"][0]))
        self.assertEqual(reward[loser], -1.0)

    def test_render_reset_and_step(self) -> None:
        env = _make_small_go(render_mode="rgb_array")
        env.reset()
        reset_frame = env.render()
        self.assertEqual(reset_frame.shape, (1, 160, 160, 3))
        self.assertEqual(reset_frame.dtype, np.uint8)
        _step(env, 12)
        step_frame = env.render()
        self.assertEqual(step_frame.shape, reset_frame.shape)
        self.assertGreater(
            int(np.count_nonzero(step_frame != reset_frame)),
            0,
        )


if __name__ == "__main__":
    absltest.main()
