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
"""Native Connector rule tests."""

# ruff: noqa: D102

from __future__ import annotations

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium


class JumanjiConnectorTest(absltest.TestCase):
    """Checks native Connector transitions."""

    def test_parallel_connections_on_default_layout(self) -> None:
        env = make_gymnasium(
            "Connector-v2", num_envs=1, seed=0, render_mode="rgb_array"
        )
        try:
            obs, info = env.reset()
            self.assertEqual(int(obs["grid"][0, 0, 0]), 2)
            self.assertEqual(int(obs["grid"][0, 0, 9]), 3)
            self.assertEqual(int(obs["grid"][0, 9, 0]), 29)
            self.assertEqual(int(obs["grid"][0, 9, 9]), 30)
            self.assertTrue(bool(obs["action_mask"][0, 0, 2]))
            self.assertEqual(int(info["num_connections"][0]), 0)

            action = np.full((1, 10), 2, dtype=np.int32)
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertAlmostEqual(float(reward[0]), -0.03, places=6)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertEqual(int(obs["grid"][0, 0, 0]), 1)
            self.assertEqual(int(obs["grid"][0, 0, 1]), 2)
            self.assertFalse(bool(obs["action_mask"][0, 0, 4]))
            self.assertEqual(int(info["total_path_length"][0]), 20)

            for _ in range(8):
                obs, reward, terminated, truncated, info = env.step(action)
            self.assertAlmostEqual(float(reward[0]), 1.0, places=6)
            self.assertTrue(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertEqual(int(info["num_connections"][0]), 10)
            self.assertAlmostEqual(float(info["ratio_connections"][0]), 1.0)
            self.assertEqual(int(info["total_path_length"][0]), 100)

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
