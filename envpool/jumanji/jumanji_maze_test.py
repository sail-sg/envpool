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
"""Native Maze rule tests."""

# ruff: noqa: D102

from __future__ import annotations

from typing import Any

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium


def _make_env() -> Any:
    return make_gymnasium(
        "Maze-v0",
        num_envs=1,
        seed=0,
        maze_walls=",".join(["0"] * 100),
        maze_agent_position="0,0",
        maze_target_position="0,2",
        render_mode="rgb_array",
    )


class JumanjiMazeTest(absltest.TestCase):
    """Checks native Maze transitions."""

    def test_invalid_move_noops_and_target_terminates(self) -> None:
        env = _make_env()
        try:
            obs, _ = env.reset()
            self.assertEqual(int(obs["agent_position"]["row"][0]), 0)
            self.assertEqual(int(obs["agent_position"]["col"][0]), 0)
            self.assertEqual(int(obs["target_position"]["row"][0]), 0)
            self.assertEqual(int(obs["target_position"]["col"][0]), 2)
            np.testing.assert_array_equal(
                obs["action_mask"][0], [False, True, True, False]
            )

            obs, reward, terminated, truncated, _ = env.step(
                np.asarray([0], dtype=np.int32)
            )
            self.assertEqual(int(obs["agent_position"]["row"][0]), 0)
            self.assertEqual(int(obs["agent_position"]["col"][0]), 0)
            self.assertEqual(float(reward[0]), 0.0)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertEqual(int(obs["step_count"][0]), 1)

            obs, reward, terminated, truncated, _ = env.step(
                np.asarray([1], dtype=np.int32)
            )
            self.assertEqual(int(obs["agent_position"]["col"][0]), 1)
            self.assertEqual(float(reward[0]), 0.0)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))

            obs, reward, terminated, truncated, _ = env.step(
                np.asarray([1], dtype=np.int32)
            )
            self.assertEqual(int(obs["agent_position"]["col"][0]), 2)
            self.assertEqual(float(reward[0]), 1.0)
            self.assertTrue(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()

    def test_seeded_reset_is_deterministic(self) -> None:
        env0 = make_gymnasium("Maze-v0", num_envs=3, seed=5)
        env1 = make_gymnasium("Maze-v0", num_envs=3, seed=5)
        try:
            obs0, _ = env0.reset()
            obs1, _ = env1.reset()
            np.testing.assert_array_equal(obs0["walls"], obs1["walls"])
        finally:
            env0.close()
            env1.close()


if __name__ == "__main__":
    absltest.main()
