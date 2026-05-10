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
"""Native Sudoku rule tests."""

# ruff: noqa: D102

from __future__ import annotations

from typing import Any

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium

_SOLVED = (
    np.asarray(
        [
            [2, 3, 7, 8, 4, 1, 5, 6, 9],
            [1, 8, 6, 7, 9, 5, 2, 4, 3],
            [5, 9, 4, 3, 2, 6, 7, 1, 8],
            [3, 1, 5, 6, 7, 4, 8, 9, 2],
            [4, 6, 9, 5, 8, 2, 1, 3, 7],
            [7, 2, 8, 1, 3, 9, 4, 5, 6],
            [6, 4, 2, 9, 1, 8, 3, 7, 5],
            [8, 5, 3, 4, 6, 7, 9, 2, 1],
            [9, 7, 1, 2, 5, 3, 6, 8, 4],
        ],
        dtype=np.int32,
    )
    - 1
)


def _make_env(task_id: str = "Sudoku-v0") -> Any:
    board = np.array(_SOLVED, copy=True)
    board[8, 8] = -1
    return make_gymnasium(
        task_id,
        num_envs=1,
        seed=0,
        sudoku_initial_board=",".join(map(str, board.reshape(-1))),
        render_mode="rgb_array",
    )


class JumanjiSudokuTest(absltest.TestCase):
    """Checks native Sudoku transitions for both registered IDs."""

    def test_default_reset_uses_seeded_database(self) -> None:
        env0 = make_gymnasium("Sudoku-v0", num_envs=1, seed=0)
        env1 = make_gymnasium("Sudoku-v0", num_envs=1, seed=1)
        easy = make_gymnasium("Sudoku-very-easy-v0", num_envs=1, seed=0)
        try:
            obs0, _ = env0.reset()
            obs1, _ = env1.reset()
            obs_easy, _ = easy.reset()
            self.assertFalse(np.array_equal(obs0["board"], obs1["board"]))
            self.assertFalse(np.array_equal(obs0["board"], obs_easy["board"]))
        finally:
            env0.close()
            env1.close()
            easy.close()

    def test_final_valid_action_solves(self) -> None:
        for task_id in ("Sudoku-v0", "Sudoku-very-easy-v0"):
            with self.subTest(task_id=task_id):
                env = _make_env(task_id)
                try:
                    obs, _ = env.reset()
                    self.assertTrue(bool(obs["action_mask"][0, 8, 8, 3]))
                    self.assertFalse(bool(obs["action_mask"][0, 8, 8, 0]))

                    obs, reward, terminated, truncated, _ = env.step(
                        np.asarray([[8, 8, 3]], dtype=np.int32)
                    )
                    np.testing.assert_array_equal(obs["board"][0], _SOLVED)
                    self.assertEqual(float(reward[0]), 1.0)
                    self.assertTrue(bool(terminated[0]))
                    self.assertFalse(bool(truncated[0]))
                    self.assertFalse(bool(np.any(obs["action_mask"][0])))

                    frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

                    assert frame is not None
                    self.assertEqual(frame.shape, (1, 256, 256, 3))
                    self.assertGreater(int(frame.max() - frame.min()), 0)
                finally:
                    env.close()

    def test_invalid_action_terminates_without_reward(self) -> None:
        env = _make_env()
        try:
            env.reset()
            obs, reward, terminated, truncated, _ = env.step(
                np.asarray([[8, 8, 0]], dtype=np.int32)
            )
            self.assertEqual(int(obs["board"][0, 8, 8]), 0)
            self.assertEqual(float(reward[0]), 0.0)
            self.assertTrue(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
