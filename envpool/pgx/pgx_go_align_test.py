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
"""Alignment tests against the official PGX 2.6.0 Go oracle."""

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
    ("Go19x19-v1", "go_19x19", 19, 30),
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


if __name__ == "__main__":
    absltest.main()
