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
"""Native CVRP rule tests."""

# ruff: noqa: D102

from __future__ import annotations

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium


class JumanjiCVRPTest(absltest.TestCase):
    """Checks native CVRP transitions."""

    def test_visit_customer_and_return_to_depot(self) -> None:
        env = make_gymnasium(
            "CVRP-v1", num_envs=1, seed=0, render_mode="rgb_array"
        )
        try:
            obs, _ = env.reset()
            self.assertEqual(int(obs["position"][0]), 0)
            self.assertFalse(bool(obs["action_mask"][0, 0]))
            self.assertTrue(bool(obs["action_mask"][0, 1]))
            self.assertAlmostEqual(float(obs["coordinates"][0, 1, 0]), 0.05)
            self.assertAlmostEqual(float(obs["capacity"][0]), 1.0)

            obs, reward, terminated, truncated, _ = env.step(
                np.asarray([1], dtype=np.int32)
            )
            self.assertAlmostEqual(float(reward[0]), -0.05, places=6)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertEqual(int(obs["position"][0]), 1)
            self.assertFalse(bool(obs["unvisited_nodes"][0, 1]))
            self.assertFalse(bool(obs["action_mask"][0, 1]))
            self.assertTrue(bool(obs["action_mask"][0, 0]))
            self.assertAlmostEqual(float(obs["capacity"][0]), 0.95, places=6)
            self.assertEqual(int(obs["trajectory"][0, 0]), 0)
            self.assertEqual(int(obs["trajectory"][0, 1]), 1)

            obs, reward, terminated, truncated, _ = env.step(
                np.asarray([0], dtype=np.int32)
            )
            self.assertAlmostEqual(float(reward[0]), -0.05, places=6)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertEqual(int(obs["position"][0]), 0)
            self.assertAlmostEqual(float(obs["capacity"][0]), 1.0)
            self.assertEqual(int(obs["trajectory"][0, 2]), 0)

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
