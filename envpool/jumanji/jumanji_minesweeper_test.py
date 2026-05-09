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
"""Native Minesweeper rule tests."""

# ruff: noqa: D102

from __future__ import annotations

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium

_UNEXPLORED = -1


def _count_adjacent(mines: set[int], row: int, col: int) -> int:
    count = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r = row + dr
            c = col + dc
            if 0 <= r < 10 and 0 <= c < 10 and r * 10 + c in mines:
                count += 1
    return count


class _ReferenceMinesweeper:
    def __init__(self, mines: set[int]) -> None:
        self.mines = mines
        self.board = np.full((10, 10), _UNEXPLORED, dtype=np.int32)
        self.step_count = 0

    @property
    def action_mask(self) -> np.ndarray:
        return self.board == _UNEXPLORED

    def step(self, row: int, col: int) -> tuple[float, bool]:
        valid = self.board[row, col] == _UNEXPLORED
        hit_mine = row * 10 + col in self.mines
        reward = 0.0
        if valid:
            self.board[row, col] = _count_adjacent(self.mines, row, col)
            reward = 0.0 if hit_mine else 1.0
        self.step_count += 1
        solved = int(np.sum(self.board >= 0)) == 100 - len(self.mines)
        return reward, (not valid) or hit_mine or solved


class JumanjiMinesweeperTest(absltest.TestCase):
    """Checks native Minesweeper against the official rule structure."""

    def test_rollout_matches_reference(self) -> None:
        mines = {0, 11, 22}
        env = make_gymnasium(
            "Minesweeper-v0",
            num_envs=1,
            seed=0,
            minesweeper_mine_locations="0,11,22",
            render_mode="rgb_array",
        )
        reference = _ReferenceMinesweeper(mines)
        try:
            obs, _ = env.reset()
            np.testing.assert_array_equal(obs["board"][0], reference.board)
            np.testing.assert_array_equal(
                obs["action_mask"][0], reference.action_mask
            )
            self.assertEqual(int(obs["num_mines"][0]), 3)
            self.assertEqual(int(obs["step_count"][0]), 0)

            for action in [(0, 1), (2, 2)]:
                expected_reward, expected_done = reference.step(*action)
                obs, reward, terminated, truncated, _ = env.step(
                    np.asarray([action], dtype=np.int32)
                )
                np.testing.assert_array_equal(obs["board"][0], reference.board)
                np.testing.assert_array_equal(
                    obs["action_mask"][0], reference.action_mask
                )
                self.assertEqual(
                    int(obs["step_count"][0]), reference.step_count
                )
                self.assertEqual(float(reward[0]), expected_reward)
                self.assertEqual(bool(terminated[0]), expected_done)
                self.assertFalse(bool(truncated[0]))

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()

    def test_seeded_reset_is_deterministic(self) -> None:
        env0 = make_gymnasium("Minesweeper-v0", num_envs=4, seed=9)
        env1 = make_gymnasium("Minesweeper-v0", num_envs=4, seed=9)
        try:
            obs0, _ = env0.reset()
            obs1, _ = env1.reset()
            np.testing.assert_array_equal(obs0["board"], obs1["board"])
            np.testing.assert_array_equal(
                obs0["action_mask"], obs1["action_mask"]
            )
        finally:
            env0.close()
            env1.close()


if __name__ == "__main__":
    absltest.main()
