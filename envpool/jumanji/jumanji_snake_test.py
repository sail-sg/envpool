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
"""Native Snake rule tests."""

from __future__ import annotations

from typing import Any

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium


def _make_env() -> Any:
    return make_gymnasium(
        "Snake-v1",
        num_envs=1,
        seed=0,
        snake_head_position="0,0",
        snake_fruit_position="0,2",
        render_mode="rgb_array",
    )


class JumanjiSnakeTest(absltest.TestCase):
    """Checks native Snake transitions."""

    def test_move_and_eat_fruit(self) -> None:
        """Checks movement, fruit collection, and scoring."""
        env = _make_env()
        try:
            obs, _ = env.reset()
            np.testing.assert_array_equal(
                obs["action_mask"][0], [False, True, True, False]
            )
            self.assertEqual(float(obs["grid"][0, 0, 0, 1]), 1.0)
            self.assertEqual(float(obs["grid"][0, 0, 2, 3]), 1.0)

            obs, reward, terminated, truncated, _ = env.step(
                np.asarray([1], dtype=np.int32)
            )
            self.assertEqual(float(reward[0]), 0.0)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertEqual(int(obs["step_count"][0]), 1)
            self.assertEqual(float(obs["grid"][0, 0, 1, 1]), 1.0)
            self.assertEqual(float(obs["grid"][0, 0, 1, 2]), 1.0)

            obs, reward, terminated, truncated, _ = env.step(
                np.asarray([1], dtype=np.int32)
            )
            self.assertEqual(float(reward[0]), 1.0)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertEqual(float(obs["grid"][0, 0, 2, 1]), 1.0)
            self.assertEqual(float(obs["grid"][0, 0, 1, 2]), 1.0)
            self.assertEqual(float(obs["grid"][0, 0, 2, 4]), 1.0)
            self.assertEqual(float(obs["grid"][0, 0, 1, 4]), 0.5)

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()

    def test_partial_initial_positions(self) -> None:
        """Keep an explicit head or fruit while sampling the other free cell."""
        for key, position, channel in (
            ("snake_head_position", "0,1", 1),
            ("snake_fruit_position", "7,8", 3),
        ):
            with self.subTest(key=key):
                env = make_gymnasium(
                    "Snake-v1", num_envs=32, seed=17, **{key: position}
                )
                try:
                    for _ in range(3):
                        obs, _ = env.reset()
                        grid = obs["grid"]
                        row, col = map(int, position.split(","))
                        np.testing.assert_array_equal(
                            grid[:, row, col, channel], 1
                        )
                        np.testing.assert_array_equal(
                            grid[..., 0] * grid[..., 3], 0
                        )
                        sampled = np.argwhere(grid[..., 4 - channel] == 1)[
                            :, 1:
                        ]
                        self.assertGreater(len(np.unique(sampled, axis=0)), 1)
                finally:
                    env.close()

    def test_invalid_wall_move_terminates(self) -> None:
        """Checks moving into a wall terminates."""
        env = _make_env()
        try:
            env.reset()
            obs, reward, terminated, truncated, _ = env.step(
                np.asarray([0], dtype=np.int32)
            )
            self.assertEqual(float(reward[0]), 0.0)
            self.assertTrue(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            # Jumanji's terminal array state wraps a negative head coordinate.
            np.testing.assert_array_equal(obs["grid"][0, 11, 0, :3], 1)
            self.assertEqual(float(obs["grid"][0, :, :, 0].sum()), 1)
            self.assertEqual(float(obs["grid"][0, 11, 0, 4]), 1)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
