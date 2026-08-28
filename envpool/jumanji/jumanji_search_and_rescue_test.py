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
"""Native SearchAndRescue rule tests."""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium


class JumanjiSearchAndRescueTest(absltest.TestCase):
    """Checks native SearchAndRescue transitions."""

    def test_searcher_finds_target(self) -> None:
        """Checks a searcher finds a target."""
        env = make_gymnasium(
            "SearchAndRescue-v0",
            num_envs=1,
            seed=0,
            render_mode="rgb_array",
            search_and_rescue_positions="0,0,0.5,0.5",
            search_and_rescue_headings="0,0",
            search_and_rescue_speeds="0.005,0.005",
            search_and_rescue_target_positions=",".join(
                map(str, [0.025, 0.0] + [0.75, 0.75] * 39)
            ),
            search_and_rescue_target_found=",".join(map(str, [0] + [1] * 39)),
            search_and_rescue_target_acc_std=0.0,
        )
        try:
            obs, _ = env.reset()
            self.assertAlmostEqual(float(obs["positions"][0, 0, 0]), 0.0)
            self.assertAlmostEqual(float(obs["positions"][0, 0, 1]), 0.0)
            self.assertAlmostEqual(
                float(obs["searcher_views"][0, 0, 2, 64]), 0.25
            )
            self.assertAlmostEqual(float(obs["targets_remaining"][0]), 1.0 / 40)

            action = np.asarray([[[0.0, 1.0], [0.0, 0.0]]], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertAlmostEqual(float(reward[0]), 1.0)
            np.testing.assert_array_equal(
                info["searcher_rewards"][0], [1.0, 0.0]
            )
            self.assertTrue(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertAlmostEqual(float(obs["positions"][0, 0, 0]), 0.01)
            self.assertAlmostEqual(float(obs["targets_remaining"][0]), 0.0)
            self.assertEqual(int(obs["step"][0]), 1)

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
