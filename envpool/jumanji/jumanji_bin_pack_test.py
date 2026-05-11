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
"""Native BinPack rule tests."""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium


class JumanjiBinPackTest(absltest.TestCase):
    """Checks native BinPack transitions."""

    def test_place_item_in_empty_space(self) -> None:
        """Checks placement updates item and EMS state."""
        env = make_gymnasium(
            "BinPack-v2", num_envs=1, seed=0, render_mode="rgb_array"
        )
        try:
            obs, _ = env.reset()
            self.assertTrue(bool(obs["ems_mask"][0, 0]))
            self.assertAlmostEqual(float(obs["ems"]["x1"][0, 0]), 0.0)
            self.assertAlmostEqual(float(obs["items"]["x_len"][0, 0]), 0.5)
            self.assertTrue(bool(obs["items_mask"][0, 0]))
            self.assertTrue(bool(obs["action_mask"][0, 0, 0]))

            obs, reward, terminated, truncated, _ = env.step(
                np.asarray([[0, 0]], dtype=np.int32)
            )
            self.assertAlmostEqual(float(reward[0]), 0.125, places=6)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertTrue(bool(obs["items_placed"][0, 0]))
            self.assertFalse(bool(obs["items_mask"][0, 0]))
            self.assertFalse(bool(obs["action_mask"][0, 0, 0]))
            self.assertTrue(bool(np.any(obs["action_mask"][0])))

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
