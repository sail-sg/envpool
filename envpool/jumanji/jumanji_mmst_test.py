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
"""Native MMST rule tests."""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium


class JumanjiMMSTTest(absltest.TestCase):
    """Checks native MMST transitions."""

    def test_agents_expand_tree_frontiers(self) -> None:
        """Checks agents expand tree frontiers."""
        env = make_gymnasium(
            "MMST-v0", num_envs=1, seed=0, render_mode="rgb_array"
        )
        try:
            obs, _ = env.reset()
            np.testing.assert_array_equal(
                obs["positions"][0], np.asarray([0, 12, 24])
            )
            self.assertTrue(bool(obs["action_mask"][0, 0, 1]))
            self.assertFalse(bool(obs["action_mask"][0, 0, 0]))

            obs, reward, terminated, truncated, _ = env.step(
                np.asarray([[1, 13, 25]], dtype=np.int32)
            )
            self.assertAlmostEqual(float(reward[0]), 3.0)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            np.testing.assert_array_equal(
                obs["positions"][0], np.asarray([1, 13, 25])
            )
            self.assertEqual(int(obs["node_types"][0, 1]), 5)
            self.assertEqual(int(obs["step_count"][0]), 1)

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
