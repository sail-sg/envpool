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
"""Determinism tests for PGX environments."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from absl.testing import absltest

import envpool.pgx.registration  # noqa: F401
from envpool.registration import make_gymnasium

_GO_TASKS = (
    ("Go9x9-v1", 9),
    ("Go13x13-v1", 13),
    ("Go19x19-v1", 19),
    ("ChineseGo9x9-v1", 9),
    ("ChineseGo13x13-v1", 13),
    ("ChineseGo19x19-v1", 19),
)
_BOARD_GAME_TASKS = (
    "TicTacToe-v1",
    "ConnectFour-v1",
    "Hex-v1",
    "Othello-v1",
    "AnimalShogi-v1",
    "Backgammon-v1",
    "Chess-v1",
    "GardnerChess-v1",
    "Shogi-v1",
    "SparrowMahjong-v1",
)
_CARD_GAME_TASKS = (
    "KuhnPoker-v1",
    "LeducHoldem-v1",
)
_SINGLE_PLAYER_TASKS = ("Play2048-v1",)


def _make_env(task_id: str) -> Any:
    return make_gymnasium(
        task_id,
        num_envs=1,
        seed=123,
        render_mode="rgb_array",
        render_width=96,
        render_height=96,
    )


def _render_array(env: Any) -> np.ndarray:
    frame = env.render(env_ids=0)
    assert frame is not None
    return cast(np.ndarray, frame)


def _unique_board_actions(size: int) -> list[int]:
    coords = (
        (0, 0),
        (0, size - 1),
        (size - 1, size - 1),
        (size - 1, 0),
        (size // 2, size // 2),
        (0, size // 2),
        (size - 1, size // 2),
        (size // 2, 0),
        (size // 2, size - 1),
        (size // 3, size // 3),
        (size // 3, (2 * size) // 3),
        ((2 * size) // 3, size // 3),
        ((2 * size) // 3, (2 * size) // 3),
        (size // 4, size // 2),
        ((3 * size) // 4, size // 2),
    )
    actions: list[int] = []
    for row, col in coords:
        action = row * size + col
        if action not in actions:
            actions.append(action)
    return actions


def _action_sequence(size: int) -> list[int]:
    board_area = size * size
    actions = _unique_board_actions(size)
    return [
        *actions[:10],
        board_area,
        actions[10],
        board_area,
        board_area,
    ]


def _pick_legal(mask: np.ndarray, step: int) -> int:
    actions = np.flatnonzero(mask)
    if actions.size == 0:
        raise AssertionError("environment returned no legal actions")
    return int(actions[(step * 3 + 1) % actions.size])


def _board_game_action(task_id: str, info: dict[str, Any], step: int) -> int:
    sequences = {
        "TicTacToe-v1": [0, 3, 1, 4, 2],
        "ConnectFour-v1": [0, 1, 0, 1, 0, 1, 0],
        "KuhnPoker-v1": [0, 0],
        "LeducHoldem-v1": [1, 0, 1, 0],
    }
    if task_id in sequences and step < len(sequences[task_id]):
        return sequences[task_id][step]
    return _pick_legal(np.asarray(info["legal_action_mask"][0]), step)


def _assert_info_equal(
    test_case: absltest.TestCase,
    lhs: dict[str, Any],
    rhs: dict[str, Any],
    keys: tuple[str, ...],
) -> None:
    for key in keys:
        np.testing.assert_array_equal(lhs[key], rhs[key], err_msg=key)
    if "id" in lhs.get("players", {}):
        test_case.assertIn("players", rhs)
        np.testing.assert_array_equal(
            lhs["players"]["id"],
            rhs["players"]["id"],
            err_msg="players.id",
        )


class PgxDeterministicTest(absltest.TestCase):
    """Same-seed PGX rollouts must be reproducible end to end."""

    def test_go_same_seed_rollout_and_render_are_deterministic(self) -> None:
        """Reset, steps, terminal state, info, and render should match."""
        info_keys = (
            "board",
            "current_player",
            "legal_action_mask",
            "ko",
            "is_psk",
            "consecutive_pass_count",
            "black_area",
            "white_area",
        )
        for task_id, size in _GO_TASKS:
            with self.subTest(task_id=task_id):
                env0 = _make_env(task_id)
                env1 = _make_env(task_id)
                obs0, info0 = env0.reset()
                obs1, info1 = env1.reset()
                np.testing.assert_array_equal(obs0, obs1)
                _assert_info_equal(self, info0, info1, info_keys)
                np.testing.assert_array_equal(
                    _render_array(env0),
                    _render_array(env1),
                )

                terminated = np.asarray([False])
                truncated = np.asarray([False])
                for action in _action_sequence(size):
                    actions = np.asarray([action], dtype=np.int32)
                    step0 = env0.step(actions)
                    step1 = env1.step(actions)
                    for lhs, rhs in zip(step0[:4], step1[:4], strict=True):
                        np.testing.assert_array_equal(lhs, rhs)
                    _assert_info_equal(self, step0[4], step1[4], info_keys)
                    np.testing.assert_array_equal(
                        _render_array(env0),
                        _render_array(env1),
                    )
                    terminated = np.asarray(step0[2], dtype=np.bool_)
                    truncated = np.asarray(step0[3], dtype=np.bool_)
                    if bool(np.all(terminated | truncated)):
                        break

                np.testing.assert_array_equal(terminated, [True])
                np.testing.assert_array_equal(truncated, [False])

    def test_board_game_same_seed_rollout_and_render_are_deterministic(
        self,
    ) -> None:
        """Native PGX board games should replay the same actions exactly."""
        for task_id in (*_BOARD_GAME_TASKS, *_CARD_GAME_TASKS):
            with self.subTest(task_id=task_id):
                if task_id == "AnimalShogi-v1":
                    info_keys = (
                        "board",
                        "current_player",
                        "hand",
                        "legal_action_mask",
                        "turn",
                    )
                elif task_id == "Backgammon-v1":
                    info_keys = (
                        "board",
                        "current_player",
                        "dice",
                        "legal_action_mask",
                        "playable_dice",
                        "played_dice_num",
                        "turn",
                    )
                elif task_id == "GardnerChess-v1":
                    info_keys = (
                        "board",
                        "current_player",
                        "fullmove_count",
                        "halfmove_count",
                        "legal_action_mask",
                        "turn",
                    )
                elif task_id == "Chess-v1":
                    info_keys = (
                        "board",
                        "castling_rights",
                        "current_player",
                        "en_passant",
                        "fullmove_count",
                        "halfmove_count",
                        "legal_action_mask",
                        "turn",
                    )
                elif task_id == "Shogi-v1":
                    info_keys = (
                        "board",
                        "current_player",
                        "hand",
                        "legal_action_mask",
                        "turn",
                    )
                elif task_id == "SparrowMahjong-v1":
                    info_keys = (
                        "current_player",
                        "dora",
                        "draw_ix",
                        "hands",
                        "is_red_in_river",
                        "last_discard",
                        "legal_action_mask",
                        "n_red_in_hands",
                        "rivers",
                        "scores",
                        "shuffled_players",
                        "turn",
                        "wall",
                    )
                elif task_id in _BOARD_GAME_TASKS:
                    info_keys = ("board", "current_player", "legal_action_mask")
                elif task_id == "KuhnPoker-v1":
                    info_keys = (
                        "cards",
                        "current_player",
                        "last_action",
                        "legal_action_mask",
                        "pot",
                    )
                else:
                    info_keys = (
                        "cards",
                        "chips",
                        "current_player",
                        "first_player",
                        "last_action",
                        "legal_action_mask",
                        "raise_count",
                        "round",
                    )
                env0 = _make_env(task_id)
                env1 = _make_env(task_id)
                obs0, info0 = env0.reset()
                obs1, info1 = env1.reset()
                np.testing.assert_array_equal(obs0, obs1)
                _assert_info_equal(self, info0, info1, info_keys)
                np.testing.assert_array_equal(
                    _render_array(env0),
                    _render_array(env1),
                )

                max_steps = 30 if task_id in _BOARD_GAME_TASKS else 6
                terminated = np.asarray([False])
                truncated = np.asarray([False])
                for step in range(max_steps):
                    action = _board_game_action(task_id, info0, step)
                    actions = np.asarray([action], dtype=np.int32)
                    step0 = env0.step(actions)
                    step1 = env1.step(actions)
                    for lhs, rhs in zip(step0[:4], step1[:4], strict=True):
                        np.testing.assert_array_equal(lhs, rhs)
                    _assert_info_equal(self, step0[4], step1[4], info_keys)
                    np.testing.assert_array_equal(
                        _render_array(env0),
                        _render_array(env1),
                    )
                    terminated = np.asarray(step0[2], dtype=np.bool_)
                    truncated = np.asarray(step0[3], dtype=np.bool_)
                    info0 = step0[4]
                    if bool(np.all(terminated | truncated)):
                        break

                np.testing.assert_array_equal(truncated, [False])
                min_steps = 2 if task_id == "KuhnPoker-v1" else 4
                if task_id in _BOARD_GAME_TASKS:
                    min_steps = 5
                if task_id == "SparrowMahjong-v1":
                    min_steps = 2
                self.assertGreaterEqual(
                    int(info0["elapsed_step"][0]), min_steps
                )

    def test_single_player_same_seed_rollout_and_render_are_deterministic(
        self,
    ) -> None:
        """Single-player PGX tasks should replay the same rollout exactly."""
        for task_id in _SINGLE_PLAYER_TASKS:
            with self.subTest(task_id=task_id):
                info_keys = ("board", "legal_action_mask")
                env0 = _make_env(task_id)
                env1 = _make_env(task_id)
                obs0, info0 = env0.reset()
                obs1, info1 = env1.reset()
                np.testing.assert_array_equal(obs0, obs1)
                _assert_info_equal(
                    self,
                    info0,
                    info1,
                    info_keys,
                )
                np.testing.assert_array_equal(
                    _render_array(env0),
                    _render_array(env1),
                )

                for step in range(12):
                    action = _pick_legal(
                        np.asarray(info0["legal_action_mask"][0]),
                        step,
                    )
                    actions = np.asarray([action], dtype=np.int32)
                    step0 = env0.step(actions)
                    step1 = env1.step(actions)
                    for lhs, rhs in zip(step0[:4], step1[:4], strict=True):
                        np.testing.assert_array_equal(lhs, rhs)
                    _assert_info_equal(
                        self,
                        step0[4],
                        step1[4],
                        info_keys,
                    )
                    np.testing.assert_array_equal(
                        _render_array(env0),
                        _render_array(env1),
                    )
                    info0 = step0[4]
                    if bool(
                        np.all(np.asarray(step0[2]) | np.asarray(step0[3]))
                    ):
                        break


if __name__ == "__main__":
    absltest.main()
