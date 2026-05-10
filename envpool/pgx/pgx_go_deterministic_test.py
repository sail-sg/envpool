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
"""Determinism tests for PGX Go environments."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from absl.testing import absltest

import envpool.pgx.registration  # noqa: F401
from envpool.registration import make_gymnasium

_TASKS = (
    ("Go9x9-v1", 9),
    ("Go19x19-v1", 19),
    ("ChineseGo9x9-v1", 9),
    ("ChineseGo19x19-v1", 19),
)


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


def _assert_info_equal(
    test_case: absltest.TestCase,
    lhs: dict[str, Any],
    rhs: dict[str, Any],
) -> None:
    for key in (
        "board",
        "current_player",
        "legal_action_mask",
        "ko",
        "is_psk",
        "consecutive_pass_count",
        "black_area",
        "white_area",
    ):
        np.testing.assert_array_equal(lhs[key], rhs[key], err_msg=key)
    test_case.assertIn("players", lhs)
    test_case.assertIn("players", rhs)
    np.testing.assert_array_equal(
        lhs["players"]["id"],
        rhs["players"]["id"],
        err_msg="players.id",
    )


class PgxGoDeterministicTest(absltest.TestCase):
    """Same-seed PGX Go rollouts must be reproducible end to end."""

    def test_same_seed_rollout_and_render_are_deterministic(self) -> None:
        """Reset, steps, terminal state, info, and render should match."""
        for task_id, size in _TASKS:
            with self.subTest(task_id=task_id):
                env0 = _make_env(task_id)
                env1 = _make_env(task_id)
                obs0, info0 = env0.reset()
                obs1, info1 = env1.reset()
                np.testing.assert_array_equal(obs0, obs1)
                _assert_info_equal(self, info0, info1)
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
                    _assert_info_equal(self, step0[4], step1[4])
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


if __name__ == "__main__":
    absltest.main()
