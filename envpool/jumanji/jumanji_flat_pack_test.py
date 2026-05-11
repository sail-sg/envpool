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
"""Native FlatPack rule tests."""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium


class JumanjiFlatPackTest(absltest.TestCase):
    """Checks native FlatPack transitions."""

    def test_place_first_block(self) -> None:
        """Checks placing the first block updates the grid."""
        env = make_gymnasium(
            "FlatPack-v0", num_envs=1, seed=0, render_mode="rgb_array"
        )
        try:
            obs, _ = env.reset()
            np.testing.assert_array_equal(
                obs["blocks"][0, 0],
                np.asarray([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
            )
            self.assertTrue(bool(obs["action_mask"][0, 0, 0, 0, 0]))

            obs, reward, terminated, truncated, _ = env.step(
                np.asarray([[0, 0, 0, 0]], dtype=np.int32)
            )
            self.assertAlmostEqual(float(reward[0]), 4.0 / 121.0)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            np.testing.assert_array_equal(
                obs["grid"][0, :2, :2],
                np.ones((2, 2), dtype=obs["grid"].dtype),
            )
            self.assertFalse(bool(obs["action_mask"][0, 0, 0, 0, 0]))

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
