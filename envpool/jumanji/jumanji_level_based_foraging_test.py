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
"""Native LevelBasedForaging rule tests."""

# ruff: noqa: D102

from __future__ import annotations

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium


class JumanjiLevelBasedForagingTest(absltest.TestCase):
    """Checks native LevelBasedForaging transitions."""

    def test_two_agents_load_adjacent_food(self) -> None:
        env = make_gymnasium(
            "LevelBasedForaging-v0",
            num_envs=1,
            seed=0,
            render_mode="rgb_array",
        )
        try:
            obs, info = env.reset()
            np.testing.assert_array_equal(
                obs["agents_view"][0, 0],
                np.asarray([1, 0, 2, 7, 7, 2, 0, 0, 1, 0, 1, 1]),
            )
            np.testing.assert_array_equal(
                obs["action_mask"][0, 0],
                np.asarray([True, False, False, False, False, True]),
            )
            self.assertEqual(int(obs["step_count"][0]), 0)
            self.assertEqual(float(info["percent_eaten"][0]), 0.0)

            obs, reward, terminated, truncated, info = env.step(
                np.asarray([[0, 2]], dtype=np.int32)
            )
            self.assertAlmostEqual(float(reward[0]), 0.0, places=6)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            np.testing.assert_array_equal(
                obs["agents_view"][0, 0],
                np.asarray([1, 0, 2, 7, 7, 2, 0, 0, 1, 1, 1, 1]),
            )
            self.assertEqual(float(info["percent_eaten"][0]), 0.0)

            obs, reward, terminated, truncated, info = env.step(
                np.asarray([[5, 5]], dtype=np.int32)
            )
            self.assertAlmostEqual(float(reward[0]), 0.5, places=6)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            np.testing.assert_array_equal(
                obs["agents_view"][0, 0],
                np.asarray([-1, -1, 0, 7, 7, 2, 0, 0, 1, 1, 1, 1]),
            )
            self.assertAlmostEqual(float(info["percent_eaten"][0]), 50.0)

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
