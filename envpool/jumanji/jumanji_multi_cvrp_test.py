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
"""Native MultiCVRP rule tests."""

# ruff: noqa: D102

from __future__ import annotations

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium


class JumanjiMultiCVRPTest(absltest.TestCase):
    """Checks native MultiCVRP transitions."""

    def test_two_vehicles_serve_customers(self) -> None:
        env = make_gymnasium(
            "MultiCVRP-v0", num_envs=1, seed=0, render_mode="rgb_array"
        )
        try:
            obs, _ = env.reset()
            self.assertEqual(int(obs["nodes"]["demands"][0, 1]), 10)
            self.assertEqual(int(obs["vehicles"]["capacities"][0, 0]), 60)
            self.assertTrue(bool(obs["action_mask"][0, 0, 1]))
            self.assertTrue(bool(obs["action_mask"][0, 1, 2]))

            obs, reward, terminated, truncated, _ = env.step(
                np.asarray([[1, 2]], dtype=np.int16)
            )
            self.assertAlmostEqual(float(reward[0]), -1.5, places=6)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertEqual(int(obs["nodes"]["demands"][0, 1]), 0)
            self.assertEqual(int(obs["nodes"]["demands"][0, 2]), 0)
            self.assertEqual(int(obs["vehicles"]["capacities"][0, 0]), 50)
            self.assertEqual(int(obs["vehicles"]["capacities"][0, 1]), 50)
            self.assertFalse(bool(obs["action_mask"][0, 0, 1]))
            self.assertFalse(bool(obs["action_mask"][0, 1, 2]))

            obs, reward, terminated, truncated, _ = env.step(
                np.asarray([[0, 0]], dtype=np.int16)
            )
            self.assertAlmostEqual(float(reward[0]), -1.5, places=6)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertEqual(int(obs["vehicles"]["capacities"][0, 0]), 60)
            self.assertEqual(int(obs["vehicles"]["capacities"][0, 1]), 60)

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
