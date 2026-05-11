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
"""Alignment tests against the official PGX 2.6.0 oracle."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pgx
from absl.testing import absltest
from pgx.go import Go

import envpool.pgx.registration  # noqa: F401
from envpool.registration import make_gymnasium

_PGX_VERSION = "2.6.0"
_PUBLIC_CASES = (
    ("Go9x9-v1", "go_9x9", 9, 30),
    ("Go13x13-v1", "go_13x13", 13, 30),
    ("Go19x19-v1", "go_19x19", 19, 30),
)
_BOARD_GAME_CASES = (
    ("TicTacToe-v1", "tic_tac_toe", (3, 3), False, 9, 9),
    ("ConnectFour-v1", "connect_four", (6, 7), False, 7, 20),
    ("Hex-v1", "hex", (11, 11), True, 122, 30),
    ("Othello-v1", "othello", (8, 8), False, 65, 30),
)
_CARD_GAME_CASES = (
    ("KuhnPoker-v1", "kuhn_poker", (0, 0)),
    ("LeducHoldem-v1", "leduc_holdem", (1, 0, 1, 0)),
)
_FIVE_BY_FIVE_ACTIONS = (
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
)
_KO_ACTIONS = (2, 17, 6, 13, 8, 11, 12, 7)
_PSK_ACTIONS = (
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
)


def _make_envpool_go(task_id: str, size: int, **kwargs: Any) -> Any:
    env_kwargs: dict[str, Any] = {"num_envs": 1, "seed": 0}
    if size not in (9, 19):
        env_kwargs["board_size"] = size
    env_kwargs.update(kwargs)
    return make_gymnasium(task_id, **env_kwargs)


def _make_pgx_go(oracle_id: str, size: int, **kwargs: Any) -> Any:
    if size in (9, 19) and not kwargs:
        return pgx.make(oracle_id)
    return Go(size=size, **kwargs)


def _init_oracle_for_current_player(oracle: Any, current_player: int) -> Any:
    for seed in range(64):
        state = oracle.init(jax.random.PRNGKey(seed))
        if int(np.asarray(state.current_player)) == current_player:
            return state
    raise AssertionError(f"could not sync PGX current player {current_player}")


def _oracle_board(state: Any, size: int) -> np.ndarray:
    return np.sign(np.asarray(state._x.board, dtype=np.int32)).reshape(
        size,
        size,
    )


def _pick_legal_non_pass(mask: np.ndarray, board_area: int, step: int) -> int:
    start = (step * 17 + 5) % board_area
    for offset in range(board_area):
        action = (start + offset * 7) % board_area
        if bool(mask[action]):
            return action
    return board_area


def _pick_legal(mask: np.ndarray, step: int) -> int:
    actions = np.flatnonzero(mask)
    if actions.size == 0:
        raise AssertionError("oracle returned no legal actions")
    return int(actions[(step * 5 + 1) % actions.size])


class PgxGoAlignTest(absltest.TestCase):
    """Step-level alignment with the pinned PGX Go oracle."""

    def setUp(self) -> None:
        """Verify the oracle package stays pinned to the expected version."""
        super().setUp()
        self.assertEqual(pgx.__version__, _PGX_VERSION)

    def _assert_reset_matches(
        self,
        oracle: Any,
        oracle_state: Any,
        env_obs: np.ndarray,
        env_info: dict[str, Any],
        size: int,
    ) -> None:
        current_player = int(np.asarray(oracle_state.current_player))
        self.assertEqual(int(env_info["env_id"][0]), 0)
        np.testing.assert_array_equal(env_info["players"]["env_id"], [0, 0])
        np.testing.assert_array_equal(env_info["players"]["id"], [0, 1])
        self.assertEqual(
            int(env_info["elapsed_step"][0]),
            int(np.asarray(oracle_state._x.step_count)),
        )
        self.assertEqual(int(env_info["current_player"][0]), current_player)
        np.testing.assert_array_equal(
            env_info["legal_action_mask"][0],
            np.asarray(oracle_state.legal_action_mask),
        )
        np.testing.assert_array_equal(
            env_info["board"][0],
            _oracle_board(oracle_state, size),
        )
        self.assertEqual(
            int(env_info["ko"][0]), int(np.asarray(oracle_state._x.ko))
        )
        self.assertEqual(
            bool(env_info["is_psk"][0]),
            bool(np.asarray(oracle_state._x.is_psk)),
        )
        self.assertEqual(
            int(env_info["consecutive_pass_count"][0]),
            int(np.asarray(oracle_state._x.consecutive_pass_count)),
        )
        for player in range(2):
            np.testing.assert_array_equal(
                env_obs[player],
                np.asarray(oracle._observe(oracle_state, jnp.int32(player))),
            )

    def _assert_step_matches(
        self,
        oracle: Any,
        oracle_state: Any,
        env_step: tuple[Any, Any, Any, Any, dict[str, Any]],
        size: int,
    ) -> None:
        obs, reward, terminated, truncated, info = env_step
        self._assert_reset_matches(oracle, oracle_state, obs, info, size)
        np.testing.assert_array_equal(
            reward,
            np.asarray(oracle_state.rewards, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            terminated,
            np.asarray([oracle_state.terminated], dtype=np.bool_),
        )
        np.testing.assert_array_equal(
            truncated,
            np.asarray([oracle_state.truncated], dtype=np.bool_),
        )

    def _step_and_assert(
        self,
        env: Any,
        oracle: Any,
        oracle_state: Any,
        action: int,
        size: int,
    ) -> Any:
        env_step = env.step(np.asarray([action], dtype=np.int32))
        oracle_state = oracle.step(oracle_state, jnp.int32(action))
        self._assert_step_matches(oracle, oracle_state, env_step, size)
        return oracle_state

    def _reset_pair(
        self,
        task_id: str,
        oracle_id: str,
        size: int,
        **kwargs: Any,
    ) -> tuple[Any, Any, Any]:
        env = _make_envpool_go(task_id, size, **kwargs)
        oracle = _make_pgx_go(oracle_id, size, **kwargs)
        obs, info = env.reset()
        oracle_state = _init_oracle_for_current_player(
            oracle,
            int(info["current_player"][0]),
        )
        self._assert_reset_matches(oracle, oracle_state, obs, info, size)
        return env, oracle, oracle_state

    def test_public_go_tasks_align_with_pgx_oracle(self) -> None:
        """Registered Go tasks should match PGX for nontrivial rollouts."""
        for task_id, oracle_id, size, rollout_steps in _PUBLIC_CASES:
            with self.subTest(task_id=task_id):
                env, oracle, oracle_state = self._reset_pair(
                    task_id,
                    oracle_id,
                    size,
                )
                board_area = size * size
                for step in range(rollout_steps):
                    mask = np.asarray(oracle_state.legal_action_mask)
                    action = _pick_legal_non_pass(mask, board_area, step)
                    oracle_state = self._step_and_assert(
                        env,
                        oracle,
                        oracle_state,
                        action,
                        size,
                    )
                    if bool(np.asarray(oracle_state.terminated)):
                        break
                if not bool(np.asarray(oracle_state.terminated)):
                    oracle_state = self._step_and_assert(
                        env,
                        oracle,
                        oracle_state,
                        board_area,
                        size,
                    )
                    self.assertFalse(bool(np.asarray(oracle_state.terminated)))
                    oracle_state = self._step_and_assert(
                        env,
                        oracle,
                        oracle_state,
                        board_area,
                        size,
                    )
                    self.assertTrue(bool(np.asarray(oracle_state.terminated)))

    def test_five_by_five_capture_rollout_aligns_with_pgx_oracle(self) -> None:
        """A capture-heavy upstream 5x5 rollout should match step by step."""
        env, oracle, oracle_state = self._reset_pair(
            "Go9x9-v1",
            "go_9x9",
            5,
        )
        for action in _FIVE_BY_FIVE_ACTIONS:
            oracle_state = self._step_and_assert(
                env,
                oracle,
                oracle_state,
                action,
                5,
            )
        self.assertTrue(bool(np.asarray(oracle_state.terminated)))

    def test_ko_and_illegal_retake_align_with_pgx_oracle(self) -> None:
        """SSK ko masking and illegal retake penalties should match PGX."""
        env, oracle, oracle_state = self._reset_pair(
            "Go9x9-v1",
            "go_9x9",
            5,
        )
        for action in _KO_ACTIONS:
            oracle_state = self._step_and_assert(
                env,
                oracle,
                oracle_state,
                action,
                5,
            )
        self.assertEqual(int(np.asarray(oracle_state._x.ko)), 12)
        self.assertFalse(bool(np.asarray(oracle_state.legal_action_mask)[12]))
        oracle_state = self._step_and_assert(env, oracle, oracle_state, 12, 5)
        self.assertTrue(bool(np.asarray(oracle_state.terminated)))

    def test_occupied_illegal_action_aligns_with_pgx_oracle(self) -> None:
        """Ordinary illegal occupied-point moves should match PGX."""
        for task_id, oracle_id, size, _ in _PUBLIC_CASES:
            with self.subTest(task_id=task_id):
                env, oracle, oracle_state = self._reset_pair(
                    task_id,
                    oracle_id,
                    size,
                )
                oracle_state = self._step_and_assert(
                    env,
                    oracle,
                    oracle_state,
                    0,
                    size,
                )
                self.assertFalse(bool(np.asarray(oracle_state.terminated)))
                self.assertFalse(
                    bool(np.asarray(oracle_state.legal_action_mask)[0])
                )
                oracle_state = self._step_and_assert(
                    env,
                    oracle,
                    oracle_state,
                    0,
                    size,
                )
                self.assertTrue(bool(np.asarray(oracle_state.terminated)))

    def test_positional_superko_aligns_with_pgx_oracle(self) -> None:
        """PSK terminal state and rewards should match PGX exactly."""
        env, oracle, oracle_state = self._reset_pair(
            "Go9x9-v1",
            "go_9x9",
            3,
            max_terminal_steps=200,
        )
        for action in _PSK_ACTIONS:
            oracle_state = self._step_and_assert(
                env,
                oracle,
                oracle_state,
                action,
                3,
            )
        self.assertTrue(bool(np.asarray(oracle_state._x.is_psk)))
        self.assertTrue(bool(np.asarray(oracle_state.terminated)))

    def test_max_terminal_steps_aligns_with_pgx_oracle(self) -> None:
        """Time-limit terminal logic should match PGX Go."""
        env, oracle, oracle_state = self._reset_pair(
            "Go9x9-v1",
            "go_9x9",
            9,
            max_terminal_steps=3,
        )
        for action in (0, 10, 20):
            oracle_state = self._step_and_assert(
                env,
                oracle,
                oracle_state,
                action,
                9,
            )
        self.assertTrue(bool(np.asarray(oracle_state.terminated)))


class PgxBoardGameAlignTest(absltest.TestCase):
    """Step-level alignment for native PGX board games."""

    def setUp(self) -> None:
        """Verify the oracle package stays pinned to the expected version."""
        super().setUp()
        self.assertEqual(pgx.__version__, _PGX_VERSION)

    def _assert_state_matches(
        self,
        oracle: Any,
        oracle_state: Any,
        env_obs: np.ndarray,
        env_info: dict[str, Any],
        board_shape: tuple[int, int],
        sign_board: bool,
    ) -> None:
        current_player = int(np.asarray(oracle_state.current_player))
        expected_board = np.asarray(oracle_state._x.board, dtype=np.int32)
        if sign_board:
            expected_board = np.sign(expected_board)
        self.assertEqual(int(env_info["env_id"][0]), 0)
        np.testing.assert_array_equal(env_info["players"]["env_id"], [0, 0])
        np.testing.assert_array_equal(env_info["players"]["id"], [0, 1])
        self.assertEqual(
            int(env_info["elapsed_step"][0]),
            int(np.asarray(oracle_state._step_count)),
        )
        self.assertEqual(int(env_info["current_player"][0]), current_player)
        np.testing.assert_array_equal(
            env_info["legal_action_mask"][0],
            np.asarray(oracle_state.legal_action_mask),
        )
        np.testing.assert_array_equal(
            env_info["board"][0],
            expected_board.reshape(board_shape),
        )
        for player in range(2):
            np.testing.assert_array_equal(
                env_obs[player],
                np.asarray(oracle._observe(oracle_state, jnp.int32(player))),
            )

    def _assert_step_matches(
        self,
        oracle: Any,
        oracle_state: Any,
        env_step: tuple[Any, Any, Any, Any, dict[str, Any]],
        board_shape: tuple[int, int],
        sign_board: bool,
    ) -> None:
        obs, reward, terminated, truncated, info = env_step
        self._assert_state_matches(
            oracle,
            oracle_state,
            obs,
            info,
            board_shape,
            sign_board,
        )
        np.testing.assert_array_equal(
            reward,
            np.asarray(oracle_state.rewards, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            terminated,
            np.asarray([oracle_state.terminated], dtype=np.bool_),
        )
        np.testing.assert_array_equal(
            truncated,
            np.asarray([oracle_state.truncated], dtype=np.bool_),
        )

    def _reset_pair(
        self,
        task_id: str,
        oracle_id: str,
        board_shape: tuple[int, int],
        sign_board: bool,
    ) -> tuple[Any, Any, Any]:
        env = make_gymnasium(task_id, num_envs=1, seed=0)
        oracle = pgx.make(oracle_id)
        obs, info = env.reset()
        oracle_state = _init_oracle_for_current_player(
            oracle,
            int(info["current_player"][0]),
        )
        self._assert_state_matches(
            oracle,
            oracle_state,
            obs,
            info,
            board_shape,
            sign_board,
        )
        return env, oracle, oracle_state

    def _step_and_assert(
        self,
        env: Any,
        oracle: Any,
        oracle_state: Any,
        action: int,
        board_shape: tuple[int, int],
        sign_board: bool,
    ) -> Any:
        env_step = env.step(np.asarray([action], dtype=np.int32))
        oracle_state = oracle.step(oracle_state, jnp.int32(action))
        self._assert_step_matches(
            oracle,
            oracle_state,
            env_step,
            board_shape,
            sign_board,
        )
        return oracle_state

    def test_board_game_rollouts_align_with_pgx_oracle(self) -> None:
        """Board-game rollouts should match PGX state, masks, and rewards."""
        for (
            task_id,
            oracle_id,
            board_shape,
            sign_board,
            _,
            rollout_steps,
        ) in _BOARD_GAME_CASES:
            with self.subTest(task_id=task_id):
                env, oracle, oracle_state = self._reset_pair(
                    task_id,
                    oracle_id,
                    board_shape,
                    sign_board,
                )
                for step in range(rollout_steps):
                    action = _pick_legal(
                        np.asarray(oracle_state.legal_action_mask),
                        step,
                    )
                    oracle_state = self._step_and_assert(
                        env,
                        oracle,
                        oracle_state,
                        action,
                        board_shape,
                        sign_board,
                    )
                    if bool(np.asarray(oracle_state.terminated)):
                        break

    def test_board_game_illegal_actions_align_with_pgx_oracle(self) -> None:
        """Illegal board-game actions should match PGX terminal penalties."""
        cases = (
            ("TicTacToe-v1", "tic_tac_toe", (3, 3), False, (0, 0)),
            ("ConnectFour-v1", "connect_four", (6, 7), False, (0,) * 7),
            ("Hex-v1", "hex", (11, 11), True, (0, 0)),
            ("Othello-v1", "othello", (8, 8), False, (0,)),
        )
        for task_id, oracle_id, board_shape, sign_board, actions in cases:
            with self.subTest(task_id=task_id):
                env, oracle, oracle_state = self._reset_pair(
                    task_id,
                    oracle_id,
                    board_shape,
                    sign_board,
                )
                for action in actions:
                    oracle_state = self._step_and_assert(
                        env,
                        oracle,
                        oracle_state,
                        action,
                        board_shape,
                        sign_board,
                    )
                self.assertTrue(bool(np.asarray(oracle_state.terminated)))


class PgxCardGameAlignTest(absltest.TestCase):
    """Step-level alignment for native PGX card games."""

    def setUp(self) -> None:
        """Verify the oracle package stays pinned to the expected version."""
        super().setUp()
        self.assertEqual(pgx.__version__, _PGX_VERSION)

    def _init_oracle_for_deal(
        self,
        oracle: Any,
        current_player: int,
        cards: np.ndarray,
    ) -> Any:
        for seed in range(16384):
            state = oracle.init(jax.random.PRNGKey(seed))
            if int(np.asarray(state.current_player)) != current_player:
                continue
            if np.array_equal(np.asarray(state._cards), cards):
                return state
        raise AssertionError("could not sync PGX card deal")

    def _assert_state_matches(
        self,
        oracle: Any,
        oracle_state: Any,
        env_obs: np.ndarray,
        env_info: dict[str, Any],
    ) -> None:
        del oracle
        self.assertEqual(int(env_info["env_id"][0]), 0)
        np.testing.assert_array_equal(env_info["players"]["env_id"], [0, 0])
        np.testing.assert_array_equal(env_info["players"]["id"], [0, 1])
        self.assertEqual(
            int(env_info["elapsed_step"][0]),
            int(np.asarray(oracle_state._step_count)),
        )
        self.assertEqual(
            int(env_info["current_player"][0]),
            int(np.asarray(oracle_state.current_player)),
        )
        np.testing.assert_array_equal(
            env_info["legal_action_mask"][0],
            np.asarray(oracle_state.legal_action_mask),
        )
        np.testing.assert_array_equal(
            env_info["cards"][0],
            np.asarray(oracle_state._cards),
        )
        self.assertEqual(
            int(env_info["last_action"][0]),
            int(np.asarray(oracle_state._last_action)),
        )
        if "pot" in env_info:
            np.testing.assert_array_equal(
                env_info["pot"][0],
                np.asarray(oracle_state._pot),
            )
        if "chips" in env_info:
            np.testing.assert_array_equal(
                env_info["chips"][0],
                np.asarray(oracle_state._chips),
            )
            self.assertEqual(
                int(env_info["first_player"][0]),
                int(np.asarray(oracle_state._first_player)),
            )
            self.assertEqual(
                int(env_info["round"][0]),
                int(np.asarray(oracle_state._round)),
            )
            self.assertEqual(
                int(env_info["raise_count"][0]),
                int(np.asarray(oracle_state._raise_count)),
            )

    def _assert_observations_match(
        self,
        oracle: Any,
        oracle_state: Any,
        env_obs: np.ndarray,
    ) -> None:
        for player in range(2):
            np.testing.assert_array_equal(
                env_obs[player],
                np.asarray(oracle._observe(oracle_state, jnp.int32(player))),
            )

    def _assert_step_matches(
        self,
        oracle: Any,
        oracle_state: Any,
        env_step: tuple[Any, Any, Any, Any, dict[str, Any]],
    ) -> None:
        obs, reward, terminated, truncated, info = env_step
        self._assert_state_matches(oracle, oracle_state, obs, info)
        self._assert_observations_match(oracle, oracle_state, obs)
        np.testing.assert_array_equal(
            reward,
            np.asarray(oracle_state.rewards, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            terminated,
            np.asarray([oracle_state.terminated], dtype=np.bool_),
        )
        np.testing.assert_array_equal(
            truncated,
            np.asarray([oracle_state.truncated], dtype=np.bool_),
        )

    def _reset_pair(self, task_id: str, oracle_id: str) -> tuple[Any, Any, Any]:
        env = make_gymnasium(task_id, num_envs=1, seed=0)
        oracle = pgx.make(oracle_id)
        obs, info = env.reset()
        oracle_state = self._init_oracle_for_deal(
            oracle,
            int(info["current_player"][0]),
            np.asarray(info["cards"][0]),
        )
        self._assert_state_matches(oracle, oracle_state, obs, info)
        self._assert_observations_match(oracle, oracle_state, obs)
        return env, oracle, oracle_state

    def test_card_game_rollouts_align_with_pgx_oracle(self) -> None:
        """Card-game rollouts should match PGX deals, states, and rewards."""
        for task_id, oracle_id, actions in _CARD_GAME_CASES:
            with self.subTest(task_id=task_id):
                env, oracle, oracle_state = self._reset_pair(
                    task_id,
                    oracle_id,
                )
                for action in actions:
                    env_step = env.step(np.asarray([action], dtype=np.int32))
                    oracle_state = oracle.step(oracle_state, jnp.int32(action))
                    self._assert_step_matches(oracle, oracle_state, env_step)
                self.assertTrue(bool(np.asarray(oracle_state.terminated)))


class PgxAnimalShogiAlignTest(absltest.TestCase):
    """Step-level alignment for native PGX Animal Shogi."""

    def setUp(self) -> None:
        """Verify the oracle package stays pinned to the expected version."""
        super().setUp()
        self.assertEqual(pgx.__version__, _PGX_VERSION)

    def _expected_board(self, oracle_state: Any) -> np.ndarray:
        board = np.empty((4, 3), dtype=np.int32)
        flat = np.asarray(oracle_state._board, dtype=np.int32)
        for sq, value in enumerate(flat):
            row = sq % 4
            col = 2 - sq // 4
            board[row, col] = value
        return board

    def _assert_state_matches(
        self,
        oracle: Any,
        oracle_state: Any,
        env_obs: np.ndarray,
        env_info: dict[str, Any],
    ) -> None:
        self.assertEqual(int(env_info["env_id"][0]), 0)
        np.testing.assert_array_equal(env_info["players"]["env_id"], [0, 0])
        np.testing.assert_array_equal(env_info["players"]["id"], [0, 1])
        self.assertEqual(
            int(env_info["elapsed_step"][0]),
            int(np.asarray(oracle_state._step_count)),
        )
        self.assertEqual(
            int(env_info["current_player"][0]),
            int(np.asarray(oracle_state.current_player)),
        )
        self.assertEqual(
            int(env_info["turn"][0]),
            int(np.asarray(oracle_state._turn)),
        )
        np.testing.assert_array_equal(
            env_info["board"][0],
            self._expected_board(oracle_state),
        )
        np.testing.assert_array_equal(
            env_info["hand"][0],
            np.asarray(oracle_state._hand),
        )
        np.testing.assert_array_equal(
            env_info["legal_action_mask"][0],
            np.asarray(oracle_state.legal_action_mask),
        )
        for player in range(2):
            np.testing.assert_array_equal(
                env_obs[player],
                np.asarray(oracle._observe(oracle_state, jnp.int32(player))),
            )

    def test_animal_shogi_rollout_aligns_with_pgx_oracle(self) -> None:
        """Animal Shogi rollout should match the PGX oracle step by step."""
        env = make_gymnasium("AnimalShogi-v1", num_envs=1, seed=0)
        oracle = pgx.make("animal_shogi")
        obs, info = env.reset()
        oracle_state = _init_oracle_for_current_player(
            oracle,
            int(info["current_player"][0]),
        )
        self._assert_state_matches(oracle, oracle_state, obs, info)
        for step in range(20):
            action = _pick_legal(
                np.asarray(oracle_state.legal_action_mask),
                step,
            )
            env_step = env.step(np.asarray([action], dtype=np.int32))
            oracle_state = oracle.step(oracle_state, jnp.int32(action))
            obs, reward, terminated, truncated, info = env_step
            self._assert_state_matches(oracle, oracle_state, obs, info)
            np.testing.assert_array_equal(
                reward,
                np.asarray(oracle_state.rewards, dtype=np.float32),
            )
            np.testing.assert_array_equal(
                terminated,
                np.asarray([oracle_state.terminated], dtype=np.bool_),
            )
            np.testing.assert_array_equal(
                truncated,
                np.asarray([oracle_state.truncated], dtype=np.bool_),
            )
            if bool(np.asarray(oracle_state.terminated)):
                break


class PgxGardnerChessAlignTest(absltest.TestCase):
    """Step-level alignment for native PGX Gardner Chess."""

    def setUp(self) -> None:
        """Verify the oracle package stays pinned to the expected version."""
        super().setUp()
        self.assertEqual(pgx.__version__, _PGX_VERSION)

    def _expected_board(self, oracle_state: Any) -> np.ndarray:
        return np.rot90(
            np.asarray(oracle_state._board, dtype=np.int32).reshape(5, 5),
            k=1,
        )

    def _init_oracle_for_current_player(
        self,
        oracle: Any,
        current_player: int,
    ) -> Any:
        for seed in range(64):
            state = oracle.init(jax.random.PRNGKey(seed))
            if int(np.asarray(state.current_player)) == current_player:
                return state
        raise AssertionError("could not sync PGX Gardner Chess current player")

    def _assert_state_matches(
        self,
        oracle: Any,
        oracle_state: Any,
        env_obs: np.ndarray,
        env_info: dict[str, Any],
    ) -> None:
        self.assertEqual(int(env_info["env_id"][0]), 0)
        np.testing.assert_array_equal(env_info["players"]["env_id"], [0, 0])
        np.testing.assert_array_equal(env_info["players"]["id"], [0, 1])
        self.assertEqual(
            int(env_info["elapsed_step"][0]),
            int(np.asarray(oracle_state._step_count)),
        )
        self.assertEqual(
            int(env_info["current_player"][0]),
            int(np.asarray(oracle_state.current_player)),
        )
        self.assertEqual(
            int(env_info["turn"][0]), int(np.asarray(oracle_state._turn))
        )
        self.assertEqual(
            int(env_info["halfmove_count"][0]),
            int(np.asarray(oracle_state._halfmove_count)),
        )
        self.assertEqual(
            int(env_info["fullmove_count"][0]),
            int(np.asarray(oracle_state._fullmove_count)),
        )
        np.testing.assert_array_equal(
            env_info["board"][0],
            self._expected_board(oracle_state),
        )
        np.testing.assert_array_equal(
            env_info["legal_action_mask"][0],
            np.asarray(oracle_state.legal_action_mask),
        )
        for player in range(2):
            np.testing.assert_array_equal(
                env_obs[player],
                np.asarray(oracle._observe(oracle_state, jnp.int32(player))),
            )

    def test_gardner_chess_rollout_aligns_with_pgx_oracle(self) -> None:
        """Gardner Chess rollout should match the PGX oracle step by step."""
        env = make_gymnasium("GardnerChess-v1", num_envs=1, seed=0)
        oracle = pgx.make("gardner_chess")
        obs, info = env.reset()
        oracle_state = self._init_oracle_for_current_player(
            oracle,
            int(info["current_player"][0]),
        )
        self._assert_state_matches(oracle, oracle_state, obs, info)
        for step in range(24):
            action = _pick_legal(
                np.asarray(oracle_state.legal_action_mask),
                step,
            )
            env_step = env.step(np.asarray([action], dtype=np.int32))
            oracle_state = oracle.step(oracle_state, jnp.int32(action))
            obs, reward, terminated, truncated, info = env_step
            self._assert_state_matches(oracle, oracle_state, obs, info)
            np.testing.assert_array_equal(
                reward,
                np.asarray(oracle_state.rewards, dtype=np.float32),
            )
            np.testing.assert_array_equal(
                terminated,
                np.asarray([oracle_state.terminated], dtype=np.bool_),
            )
            np.testing.assert_array_equal(
                truncated,
                np.asarray([oracle_state.truncated], dtype=np.bool_),
            )
            if bool(np.asarray(oracle_state.terminated)):
                break


class PgxChessAlignTest(absltest.TestCase):
    """Step-level alignment for native PGX Chess."""

    def setUp(self) -> None:
        """Verify the oracle package stays pinned to the expected version."""
        super().setUp()
        self.assertEqual(pgx.__version__, _PGX_VERSION)

    def _expected_board(self, oracle_state: Any) -> np.ndarray:
        return np.rot90(
            np.asarray(oracle_state._x.board, dtype=np.int32).reshape(8, 8),
            k=1,
        )

    def _init_oracle_for_current_player(
        self,
        oracle: Any,
        current_player: int,
    ) -> Any:
        for seed in range(64):
            state = oracle.init(jax.random.PRNGKey(seed))
            if int(np.asarray(state.current_player)) == current_player:
                return state
        raise AssertionError("could not sync PGX Chess current player")

    def _assert_state_matches(
        self,
        oracle: Any,
        oracle_state: Any,
        env_obs: np.ndarray,
        env_info: dict[str, Any],
    ) -> None:
        self.assertEqual(int(env_info["env_id"][0]), 0)
        np.testing.assert_array_equal(env_info["players"]["env_id"], [0, 0])
        np.testing.assert_array_equal(env_info["players"]["id"], [0, 1])
        self.assertEqual(
            int(env_info["elapsed_step"][0]),
            int(np.asarray(oracle_state._step_count)),
        )
        self.assertEqual(
            int(env_info["current_player"][0]),
            int(np.asarray(oracle_state.current_player)),
        )
        self.assertEqual(
            int(env_info["turn"][0]),
            int(np.asarray(oracle_state._x.color)),
        )
        self.assertEqual(
            int(env_info["en_passant"][0]),
            int(np.asarray(oracle_state._x.en_passant)),
        )
        self.assertEqual(
            int(env_info["halfmove_count"][0]),
            int(np.asarray(oracle_state._x.halfmove_count)),
        )
        self.assertEqual(
            int(env_info["fullmove_count"][0]),
            int(np.asarray(oracle_state._x.fullmove_count)),
        )
        np.testing.assert_array_equal(
            env_info["board"][0],
            self._expected_board(oracle_state),
        )
        np.testing.assert_array_equal(
            env_info["castling_rights"][0],
            np.asarray(oracle_state._x.castling_rights),
        )
        np.testing.assert_array_equal(
            env_info["legal_action_mask"][0],
            np.asarray(oracle_state.legal_action_mask),
        )
        for player in range(2):
            np.testing.assert_array_equal(
                env_obs[player],
                np.asarray(oracle._observe(oracle_state, jnp.int32(player))),
            )

    def test_chess_rollout_aligns_with_pgx_oracle(self) -> None:
        """Chess rollout should match the PGX oracle step by step."""
        env = make_gymnasium("Chess-v1", num_envs=1, seed=0)
        oracle = pgx.make("chess")
        obs, info = env.reset()
        oracle_state = self._init_oracle_for_current_player(
            oracle,
            int(info["current_player"][0]),
        )
        self._assert_state_matches(oracle, oracle_state, obs, info)
        for step in range(24):
            action = _pick_legal(
                np.asarray(oracle_state.legal_action_mask),
                step,
            )
            env_step = env.step(np.asarray([action], dtype=np.int32))
            oracle_state = oracle.step(oracle_state, jnp.int32(action))
            obs, reward, terminated, truncated, info = env_step
            self._assert_state_matches(oracle, oracle_state, obs, info)
            np.testing.assert_array_equal(
                reward,
                np.asarray(oracle_state.rewards, dtype=np.float32),
            )
            np.testing.assert_array_equal(
                terminated,
                np.asarray([oracle_state.terminated], dtype=np.bool_),
            )
            np.testing.assert_array_equal(
                truncated,
                np.asarray([oracle_state.truncated], dtype=np.bool_),
            )
            if bool(np.asarray(oracle_state.terminated)):
                break


class PgxShogiAlignTest(absltest.TestCase):
    """Step-level alignment for native PGX Shogi."""

    def setUp(self) -> None:
        """Verify the oracle package stays pinned to the expected version."""
        super().setUp()
        self.assertEqual(pgx.__version__, _PGX_VERSION)

    def _expected_board(self, oracle_state: Any) -> np.ndarray:
        return np.rot90(
            np.asarray(oracle_state._x.board, dtype=np.int32).reshape(9, 9),
            k=3,
        )

    def _init_oracle_for_current_player(
        self,
        oracle: Any,
        current_player: int,
    ) -> Any:
        for seed in range(64):
            state = oracle.init(jax.random.PRNGKey(seed))
            if int(np.asarray(state.current_player)) == current_player:
                return state
        raise AssertionError("could not sync PGX Shogi current player")

    def _assert_state_matches(
        self,
        oracle: Any,
        oracle_state: Any,
        env_obs: np.ndarray,
        env_info: dict[str, Any],
    ) -> None:
        self.assertEqual(int(env_info["env_id"][0]), 0)
        np.testing.assert_array_equal(env_info["players"]["env_id"], [0, 0])
        np.testing.assert_array_equal(env_info["players"]["id"], [0, 1])
        self.assertEqual(
            int(env_info["elapsed_step"][0]),
            int(np.asarray(oracle_state._x.step_count)),
        )
        self.assertEqual(
            int(env_info["current_player"][0]),
            int(np.asarray(oracle_state.current_player)),
        )
        self.assertEqual(
            int(env_info["turn"][0]),
            int(np.asarray(oracle_state._x.color)),
        )
        np.testing.assert_array_equal(
            env_info["board"][0],
            self._expected_board(oracle_state),
        )
        np.testing.assert_array_equal(
            env_info["hand"][0],
            np.asarray(oracle_state._x.hand),
        )
        np.testing.assert_array_equal(
            env_info["legal_action_mask"][0],
            np.asarray(oracle_state.legal_action_mask),
        )
        for player in range(2):
            np.testing.assert_array_equal(
                env_obs[player],
                np.asarray(oracle._observe(oracle_state, jnp.int32(player))),
            )

    def test_shogi_rollout_aligns_with_pgx_oracle(self) -> None:
        """Shogi rollout should match the PGX oracle step by step."""
        env = make_gymnasium("Shogi-v1", num_envs=1, seed=0)
        oracle = pgx.make("shogi")
        obs, info = env.reset()
        oracle_state = self._init_oracle_for_current_player(
            oracle,
            int(info["current_player"][0]),
        )
        self._assert_state_matches(oracle, oracle_state, obs, info)
        for step in range(16):
            action = _pick_legal(
                np.asarray(oracle_state.legal_action_mask),
                step,
            )
            env_step = env.step(np.asarray([action], dtype=np.int32))
            oracle_state = oracle.step(oracle_state, jnp.int32(action))
            obs, reward, terminated, truncated, info = env_step
            self._assert_state_matches(oracle, oracle_state, obs, info)
            np.testing.assert_array_equal(
                reward,
                np.asarray(oracle_state.rewards, dtype=np.float32),
            )
            np.testing.assert_array_equal(
                terminated,
                np.asarray([oracle_state.terminated], dtype=np.bool_),
            )
            np.testing.assert_array_equal(
                truncated,
                np.asarray([oracle_state.truncated], dtype=np.bool_),
            )
            if bool(np.asarray(oracle_state.terminated)):
                break


class PgxBackgammonAlignTest(absltest.TestCase):
    """Step-level alignment for native PGX Backgammon."""

    def setUp(self) -> None:
        """Verify the oracle package stays pinned to the expected version."""
        super().setUp()
        self.assertEqual(pgx.__version__, _PGX_VERSION)

    def _state_signature(self, state: Any) -> np.ndarray:
        return np.concatenate((
            np.asarray(state._board, dtype=np.int32),
            np.asarray(state._dice, dtype=np.int32),
            np.asarray(state._playable_dice, dtype=np.int32),
            np.asarray(
                [state.current_player, state._played_dice_num, state._turn],
                dtype=np.int32,
            ),
        ))

    def _info_signature(self, info: dict[str, Any]) -> np.ndarray:
        return np.concatenate((
            np.asarray(info["board"][0], dtype=np.int32),
            np.asarray(info["dice"][0], dtype=np.int32),
            np.asarray(info["playable_dice"][0], dtype=np.int32),
            np.asarray(
                [
                    info["current_player"][0],
                    info["played_dice_num"][0],
                    info["turn"][0],
                ],
                dtype=np.int32,
            ),
        ))

    def _find_reset_state(self, oracle: Any, info: dict[str, Any]) -> Any:
        target = self._info_signature(info)
        for seed in range(4096):
            state = oracle.init(jax.random.PRNGKey(seed))
            if np.array_equal(self._state_signature(state), target):
                return state
        raise AssertionError("could not sync PGX Backgammon reset state")

    def _find_step_state(
        self,
        oracle: Any,
        oracle_state: Any,
        action: int,
        info: dict[str, Any],
    ) -> Any:
        target = self._info_signature(info)
        seeds = jnp.arange(4096, dtype=jnp.uint32)

        def step_signature(seed: Any) -> Any:
            state = oracle.step(
                oracle_state,
                jnp.int32(action),
                jax.random.PRNGKey(seed),
            )
            return jnp.concatenate((
                state._board,
                state._dice,
                state._playable_dice,
                jnp.asarray(
                    [
                        state.current_player,
                        state._played_dice_num,
                        state._turn,
                    ],
                    dtype=jnp.int32,
                ),
            ))

        signatures = np.asarray(jax.jit(jax.vmap(step_signature))(seeds))
        matches = np.flatnonzero(np.all(signatures == target, axis=1))
        if matches.size == 0:
            raise AssertionError("could not sync PGX Backgammon step key")
        return oracle.step(
            oracle_state,
            jnp.int32(action),
            jax.random.PRNGKey(int(matches[0])),
        )

    def _assert_state_matches(
        self,
        oracle: Any,
        oracle_state: Any,
        env_obs: np.ndarray,
        env_info: dict[str, Any],
    ) -> None:
        self.assertEqual(int(env_info["env_id"][0]), 0)
        np.testing.assert_array_equal(env_info["players"]["env_id"], [0, 0])
        np.testing.assert_array_equal(env_info["players"]["id"], [0, 1])
        self.assertEqual(
            int(env_info["elapsed_step"][0]),
            int(np.asarray(oracle_state._step_count)),
        )
        np.testing.assert_array_equal(
            self._info_signature(env_info),
            self._state_signature(oracle_state),
        )
        np.testing.assert_array_equal(
            env_info["legal_action_mask"][0],
            np.asarray(oracle_state.legal_action_mask),
        )
        for player in range(2):
            np.testing.assert_array_equal(
                env_obs[player],
                np.asarray(oracle._observe(oracle_state, jnp.int32(player))),
            )

    def test_backgammon_rollout_aligns_with_pgx_oracle(self) -> None:
        """Backgammon rollout should match PGX with synchronized dice keys."""
        env = make_gymnasium("Backgammon-v1", num_envs=1, seed=0)
        oracle = pgx.make("backgammon")
        obs, info = env.reset()
        oracle_state = self._find_reset_state(oracle, info)
        self._assert_state_matches(oracle, oracle_state, obs, info)
        for step in range(12):
            action = _pick_legal(
                np.asarray(oracle_state.legal_action_mask),
                step,
            )
            env_step = env.step(np.asarray([action], dtype=np.int32))
            obs, reward, terminated, truncated, info = env_step
            oracle_state = self._find_step_state(
                oracle,
                oracle_state,
                action,
                info,
            )
            self._assert_state_matches(oracle, oracle_state, obs, info)
            np.testing.assert_array_equal(
                reward,
                np.asarray(oracle_state.rewards, dtype=np.float32),
            )
            np.testing.assert_array_equal(
                terminated,
                np.asarray([oracle_state.terminated], dtype=np.bool_),
            )
            np.testing.assert_array_equal(
                truncated,
                np.asarray([oracle_state.truncated], dtype=np.bool_),
            )


class PgxSparrowMahjongAlignTest(absltest.TestCase):
    """Step-level alignment for native PGX Sparrow Mahjong."""

    def setUp(self) -> None:
        """Verify the oracle package stays pinned to the expected version."""
        super().setUp()
        self.assertEqual(pgx.__version__, _PGX_VERSION)

    def _oracle_state_from_info(self, oracle: Any, info: dict[str, Any]) -> Any:
        state = oracle.init(jax.random.PRNGKey(0))
        return state.replace(
            current_player=jnp.int32(info["current_player"][0]),
            legal_action_mask=jnp.asarray(
                info["legal_action_mask"][0],
                dtype=jnp.bool_,
            ),
            _step_count=jnp.int32(info["elapsed_step"][0]),
            _turn=jnp.int32(info["turn"][0]),
            _rivers=jnp.asarray(info["rivers"][0], dtype=jnp.int32),
            _last_discard=jnp.int32(info["last_discard"][0]),
            _hands=jnp.asarray(info["hands"][0], dtype=jnp.int32),
            _n_red_in_hands=jnp.asarray(
                info["n_red_in_hands"][0],
                dtype=jnp.int32,
            ),
            _is_red_in_river=jnp.asarray(
                info["is_red_in_river"][0],
                dtype=jnp.bool_,
            ),
            _wall=jnp.asarray(info["wall"][0], dtype=jnp.int32),
            _draw_ix=jnp.int32(info["draw_ix"][0]),
            _shuffled_players=jnp.asarray(
                info["shuffled_players"][0],
                dtype=jnp.int32,
            ),
            _dora=jnp.int32(info["dora"][0]),
            _scores=jnp.asarray(info["scores"][0], dtype=jnp.int32),
        )

    def _assert_state_matches(
        self,
        oracle: Any,
        oracle_state: Any,
        env_obs: np.ndarray,
        env_info: dict[str, Any],
    ) -> None:
        self.assertEqual(int(env_info["env_id"][0]), 0)
        np.testing.assert_array_equal(env_info["players"]["env_id"], [0, 0, 0])
        np.testing.assert_array_equal(env_info["players"]["id"], [0, 1, 2])
        self.assertEqual(
            int(env_info["elapsed_step"][0]),
            int(np.asarray(oracle_state._step_count)),
        )
        self.assertEqual(
            int(env_info["current_player"][0]),
            int(np.asarray(oracle_state.current_player)),
        )
        for key in (
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
        ):
            oracle_value = getattr(oracle_state, f"_{key}", None)
            if key == "legal_action_mask":
                oracle_value = oracle_state.legal_action_mask
            np.testing.assert_array_equal(
                env_info[key][0],
                np.asarray(oracle_value),
                err_msg=key,
            )
        for player in range(3):
            np.testing.assert_array_equal(
                env_obs[player],
                np.asarray(oracle._observe(oracle_state, jnp.int32(player))),
            )

    def test_sparrow_mahjong_rollout_aligns_with_pgx_oracle(self) -> None:
        """Sparrow Mahjong rollout should match a reset-synced PGX oracle."""
        env = make_gymnasium("SparrowMahjong-v1", num_envs=1, seed=0)
        oracle = pgx.make("sparrow_mahjong")
        obs, info = env.reset()
        oracle_state = self._oracle_state_from_info(oracle, info)
        self._assert_state_matches(oracle, oracle_state, obs, info)
        for step in range(12):
            action = _pick_legal(
                np.asarray(oracle_state.legal_action_mask),
                step,
            )
            env_step = env.step(np.asarray([action], dtype=np.int32))
            oracle_state = oracle.step(oracle_state, jnp.int32(action))
            obs, reward, terminated, truncated, info = env_step
            self._assert_state_matches(oracle, oracle_state, obs, info)
            np.testing.assert_array_equal(
                reward,
                np.asarray(oracle_state.rewards, dtype=np.float32),
            )
            np.testing.assert_array_equal(
                terminated,
                np.asarray([oracle_state.terminated], dtype=np.bool_),
            )
            np.testing.assert_array_equal(
                truncated,
                np.asarray([oracle_state.truncated], dtype=np.bool_),
            )
            if bool(np.asarray(oracle_state.terminated)):
                break


class PgxPlay2048AlignTest(absltest.TestCase):
    """Alignment for PGX 2048 with matched stochastic keys."""

    def setUp(self) -> None:
        """Verify the oracle package stays pinned to the expected version."""
        super().setUp()
        self.assertEqual(pgx.__version__, _PGX_VERSION)

    def _find_reset_state(self, oracle: Any, board: np.ndarray) -> Any:
        seeds = jnp.arange(65536, dtype=jnp.uint32)
        boards = np.asarray(
            jax.jit(
                jax.vmap(
                    lambda seed: oracle.init(jax.random.PRNGKey(seed))._board
                )
            )(seeds)
        )
        matches = np.flatnonzero(np.all(boards == board.reshape(-1), axis=1))
        if matches.size == 0:
            raise AssertionError("could not sync PGX 2048 reset board")
        return oracle.init(jax.random.PRNGKey(int(matches[0])))

    def _find_step_state(
        self,
        oracle: Any,
        oracle_state: Any,
        action: int,
        board: np.ndarray,
    ) -> Any:
        seeds = jnp.arange(65536, dtype=jnp.uint32)
        boards = np.asarray(
            jax.jit(
                jax.vmap(
                    lambda seed: (
                        oracle.step(
                            oracle_state,
                            jnp.int32(action),
                            jax.random.PRNGKey(seed),
                        )._board
                    )
                )
            )(seeds)
        )
        matches = np.flatnonzero(np.all(boards == board.reshape(-1), axis=1))
        if matches.size == 0:
            raise AssertionError("could not sync PGX 2048 step key")
        return oracle.step(
            oracle_state,
            jnp.int32(action),
            jax.random.PRNGKey(int(matches[0])),
        )

    def _assert_state_matches(
        self,
        oracle: Any,
        oracle_state: Any,
        env_obs: np.ndarray,
        env_info: dict[str, Any],
    ) -> None:
        self.assertEqual(int(env_info["env_id"][0]), 0)
        self.assertEqual(
            int(env_info["elapsed_step"][0]),
            int(np.asarray(oracle_state._step_count)),
        )
        np.testing.assert_array_equal(
            env_info["board"][0],
            np.asarray(oracle_state._board).reshape(4, 4),
        )
        np.testing.assert_array_equal(
            env_info["legal_action_mask"][0],
            np.asarray(oracle_state.legal_action_mask),
        )
        np.testing.assert_array_equal(
            env_obs[0],
            np.asarray(oracle._observe(oracle_state, jnp.int32(0))),
        )

    def test_play2048_aligns_with_pgx_oracle(self) -> None:
        """2048 rollout should match PGX with synchronized tile-spawn keys."""
        env = make_gymnasium("Play2048-v1", num_envs=1, seed=0)
        oracle = pgx.make("2048")
        obs, info = env.reset()
        oracle_state = self._find_reset_state(oracle, info["board"][0])
        self._assert_state_matches(oracle, oracle_state, obs, info)

        for step in range(4):
            mask = np.asarray(info["legal_action_mask"][0])
            action = _pick_legal(mask, step)
            obs, reward, terminated, truncated, info = env.step(
                np.asarray([action], dtype=np.int32),
            )
            oracle_state = self._find_step_state(
                oracle,
                oracle_state,
                action,
                info["board"][0],
            )
            self._assert_state_matches(oracle, oracle_state, obs, info)
            np.testing.assert_array_equal(
                reward,
                np.asarray(oracle_state.rewards, dtype=np.float32),
            )
            np.testing.assert_array_equal(
                terminated,
                np.asarray([oracle_state.terminated], dtype=np.bool_),
            )
            np.testing.assert_array_equal(
                truncated,
                np.asarray([oracle_state.truncated], dtype=np.bool_),
            )


if __name__ == "__main__":
    absltest.main()
