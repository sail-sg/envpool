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
"""Native Sokoban rule tests."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium

_EMPTY = np.uint8(0)
_WALL = np.uint8(1)
_TARGET = np.uint8(2)
_AGENT = np.uint8(3)
_BOX = np.uint8(4)
_MOVES = np.asarray(
    [
        [-1, 0],
        [0, 1],
        [1, 0],
        [0, -1],
    ],
    dtype=np.int32,
)

_SIMPLE_SOLVE_LEVEL = (
    "##########",
    "#       ##",
    "# ....   #",
    "# $$$$  ##",
    "# @    # #",
    "#   #   # ",
    "#        #",
    "##########",
    "##########",
    "##########",
)


@dataclasses.dataclass
class _ReferenceSokoban:
    fixed: np.ndarray
    variable: np.ndarray
    agent_location: np.ndarray
    step_count: int = 0

    @classmethod
    def simple_solve(cls) -> "_ReferenceSokoban":
        fixed = np.zeros((10, 10), dtype=np.uint8)
        variable = np.zeros((10, 10), dtype=np.uint8)
        agent_location = np.zeros((2,), dtype=np.int32)
        for row, line in enumerate(_SIMPLE_SOLVE_LEVEL):
            for col, cell in enumerate(line):
                if cell == "#":
                    fixed[row, col] = _WALL
                elif cell == ".":
                    fixed[row, col] = _TARGET
                elif cell == "@":
                    variable[row, col] = _AGENT
                    agent_location[:] = (row, col)
                elif cell == "$":
                    variable[row, col] = _BOX
        return cls(
            fixed=fixed, variable=variable, agent_location=agent_location
        )

    @property
    def observation(self) -> dict[str, Any]:
        return {
            "grid": np.stack([self.variable, self.fixed], axis=-1),
            "step_count": np.asarray(self.step_count, dtype=np.int32),
        }

    @property
    def boxes_on_targets(self) -> int:
        return int(np.sum((self.fixed == _TARGET) & (self.variable == _BOX)))

    def step(self, action: int) -> tuple[float, bool]:
        previous_targets = self.boxes_on_targets
        action = self._detect_noop(action)
        if action != -1:
            self._move_agent(action)
        self.step_count += 1
        next_targets = self.boxes_on_targets
        solved = next_targets == 4
        reward = float(next_targets - previous_targets)
        reward += 10.0 if solved else 0.0
        reward -= 0.1
        return reward, solved or self.step_count >= 120

    def _detect_noop(self, action: int) -> int:
        next_location = self.agent_location + _MOVES[action]
        if not _in_grid(next_location):
            return -1
        row, col = next_location
        if self.fixed[row, col] == _WALL:
            return -1
        if self.variable[row, col] != _BOX:
            return action
        box_location = next_location + _MOVES[action]
        if not _in_grid(box_location):
            return -1
        box_row, box_col = box_location
        if self.variable[box_row, box_col] == _BOX:
            return -1
        if self.fixed[box_row, box_col] == _WALL:
            return -1
        return action

    def _move_agent(self, action: int) -> None:
        next_location = self.agent_location + _MOVES[action]
        row, col = self.agent_location
        next_row, next_col = next_location
        pushes_box = self.variable[next_row, next_col] == _BOX
        self.variable[row, col] = _EMPTY
        if pushes_box:
            box_row, box_col = next_location + _MOVES[action]
            self.variable[box_row, box_col] = _BOX
        self.variable[next_row, next_col] = _AGENT
        self.agent_location[:] = next_location


def _in_grid(location: np.ndarray) -> bool:
    return bool(np.all((0 <= location) & (location < 10)))


class JumanjiSokobanTest(absltest.TestCase):
    """Checks native Sokoban against the Jumanji SimpleSolve rules."""

    def test_simple_solve_rollout_matches_reference(self) -> None:
        """Solves the official simple level with step-level checks."""
        env = make_gymnasium(
            "Sokoban-v0",
            num_envs=1,
            seed=0,
            base_path="/tmp/envpool-missing-boxoban",
            sokoban_level_index=0,
            render_mode="rgb_array",
        )
        reference = _ReferenceSokoban.simple_solve()
        try:
            obs, info = env.reset()
            np.testing.assert_array_equal(
                obs["grid"][0], reference.observation["grid"]
            )
            self.assertEqual(int(obs["step_count"][0]), 0)
            self.assertEqual(float(info["prop_correct_boxes"][0]), 0.0)
            self.assertFalse(bool(info["solved"][0]))

            actions = [0, 2, 1] * 3 + [0]
            for i, action in enumerate(actions):
                reward_ref, done_ref = reference.step(action)
                obs, reward, terminated, truncated, info = env.step(
                    np.asarray([action], dtype=np.int32)
                )
                np.testing.assert_array_equal(
                    obs["grid"][0], reference.observation["grid"]
                )
                self.assertEqual(int(obs["step_count"][0]), i + 1)
                self.assertAlmostEqual(float(reward[0]), reward_ref, places=6)
                self.assertEqual(bool(terminated[0]), done_ref)
                self.assertFalse(bool(truncated[0]))
                self.assertEqual(
                    float(info["prop_correct_boxes"][0]),
                    reference.boxes_on_targets / 4.0,
                )
                self.assertEqual(bool(info["solved"][0]), done_ref)

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()

    def test_reset_is_deterministic_for_same_seed(self) -> None:
        """Checks reset determinism on the packaged Boxoban shard."""
        env0 = make_gymnasium(
            "Sokoban-v0", num_envs=2, seed=123, render_mode="rgb_array"
        )
        env1 = make_gymnasium(
            "Sokoban-v0", num_envs=2, seed=123, render_mode="rgb_array"
        )
        try:
            obs0, info0 = env0.reset()
            obs1, info1 = env1.reset()
            np.testing.assert_array_equal(obs0["grid"], obs1["grid"])
            np.testing.assert_array_equal(info0["env_id"], info1["env_id"])
        finally:
            env0.close()
            env1.close()


if __name__ == "__main__":
    absltest.main()
