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
"""Native Tetris rule tests."""

from __future__ import annotations

import numpy as np
from absl.testing import absltest, parameterized

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium


class JumanjiTetrisTest(parameterized.TestCase):
    """Checks native Tetris transitions."""

    @parameterized.named_parameters(
        ("square", 2, 2, 0, 2, 2),
        ("horizontal_line_reaches_floor", 4, 1, 1, 1, 4),
    )
    def test_drop_piece(
        self,
        rows: int,
        cols: int,
        rotation: int,
        landed_rows: int,
        landed_cols: int,
    ) -> None:
        """Drop pieces, including a one-row rotated piece, onto the floor."""
        square = np.zeros((4, 4), dtype=np.int32)
        square[:rows, :cols] = 1
        env = make_gymnasium(
            "Tetris-v0",
            num_envs=1,
            seed=0,
            render_mode="rgb_array",
            tetris_tetromino=",".join(map(str, square.ravel())),
        )
        try:
            obs, _ = env.reset()
            np.testing.assert_array_equal(
                obs["tetromino"][0, :rows, :cols],
                np.ones((rows, cols), dtype=obs["tetromino"].dtype),
            )
            self.assertTrue(bool(obs["action_mask"][0, 0, 0]))
            self.assertFalse(
                bool(obs["action_mask"][0, rotation, 11 - landed_cols])
            )

            obs, reward, terminated, truncated, _ = env.step(
                np.asarray([[rotation, 0]], dtype=np.int32)
            )
            self.assertAlmostEqual(float(reward[0]), 0.0)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertEqual(int(obs["step_count"][0]), 0)
            np.testing.assert_array_equal(
                obs["grid"][0, -landed_rows:, :landed_cols],
                np.ones((landed_rows, landed_cols), dtype=obs["grid"].dtype),
            )
            self.assertTrue(bool(obs["action_mask"][0, 0, 0]))

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
