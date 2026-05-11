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
"""Native Game2048 rule tests."""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium


def _move_line_left(line: np.ndarray) -> tuple[np.ndarray, float]:
    compact = [int(value) for value in line if int(value) != 0]
    moved: list[int] = []
    reward = 0.0
    index = 0
    while index < len(compact):
        if index + 1 < len(compact) and compact[index] == compact[index + 1]:
            merged = compact[index] + 1
            moved.append(merged)
            reward += float(2**merged)
            index += 2
        else:
            moved.append(compact[index])
            index += 1
    moved.extend([0] * (4 - len(moved)))
    return np.asarray(moved, dtype=np.int32), reward


def _move(board: np.ndarray, action: int) -> tuple[np.ndarray, float]:
    board = np.asarray(board, dtype=np.int32)
    moved = np.array(board, copy=True)
    reward = 0.0
    for i in range(4):
        if action == 0:
            line = board[:, i]
        elif action == 1:
            line = board[i, ::-1]
        elif action == 2:
            line = board[::-1, i]
        else:
            line = board[i, :]
        next_line, line_reward = _move_line_left(line)
        reward += line_reward
        if action == 0:
            moved[:, i] = next_line
        elif action == 1:
            moved[i, ::-1] = next_line
        elif action == 2:
            moved[::-1, i] = next_line
        else:
            moved[i, :] = next_line
    return moved, reward


def _action_mask(board: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            not np.array_equal(_move(board, action)[0], board)
            for action in range(4)
        ],
        dtype=bool,
    )


class JumanjiGame2048Test(absltest.TestCase):
    """Checks native Game2048 transitions against a rule reference."""

    def test_rollout_matches_reference_without_random_cell(self) -> None:
        """Checks a deterministic rollout against a rule reference."""
        initial = np.asarray(
            [
                [1, 1, 2, 2],
                [3, 4, 0, 0],
                [0, 2, 0, 0],
                [0, 5, 0, 0],
            ],
            dtype=np.int32,
        )
        env = make_gymnasium(
            "Game2048-v1",
            num_envs=1,
            seed=0,
            game2048_initial_board=",".join(map(str, initial.reshape(-1))),
            game2048_add_random_cell=False,
            render_mode="rgb_array",
        )
        try:
            board = np.array(initial, copy=True)
            obs, info = env.reset()
            np.testing.assert_array_equal(obs["board"][0], board)
            np.testing.assert_array_equal(
                obs["action_mask"][0], _action_mask(board)
            )
            self.assertEqual(int(info["highest_tile"][0]), 32)

            for action in [3, 0, 1, 2, 3, 0]:
                board, expected_reward = _move(board, action)
                obs, reward, terminated, truncated, info = env.step(
                    np.asarray([action], dtype=np.int32)
                )
                np.testing.assert_array_equal(obs["board"][0], board)
                np.testing.assert_array_equal(
                    obs["action_mask"][0], _action_mask(board)
                )
                self.assertEqual(float(reward[0]), expected_reward)
                self.assertEqual(
                    bool(terminated[0]), not _action_mask(board).any()
                )
                self.assertFalse(bool(truncated[0]))
                self.assertEqual(
                    int(info["highest_tile"][0]), int(2 ** board.max())
                )

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()

    def test_seeded_reset_is_deterministic(self) -> None:
        """Checks seeded resets produce identical boards."""
        env0 = make_gymnasium("Game2048-v1", num_envs=4, seed=7)
        env1 = make_gymnasium("Game2048-v1", num_envs=4, seed=7)
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
