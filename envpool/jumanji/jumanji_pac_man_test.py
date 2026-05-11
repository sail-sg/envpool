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
"""Native PacMan rule tests."""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium


class JumanjiPacManTest(absltest.TestCase):
    """Checks native PacMan transitions."""

    def test_player_eats_pellet(self) -> None:
        """Checks the player eats a pellet and scores."""
        env = make_gymnasium(
            "PacMan-v1", num_envs=1, seed=0, render_mode="rgb_array"
        )
        try:
            obs, _ = env.reset()
            self.assertEqual(int(obs["player_locations"]["y"][0]), 1)
            self.assertEqual(int(obs["player_locations"]["x"][0]), 1)
            np.testing.assert_array_equal(
                obs["pellet_locations"][0, 0], np.asarray([1, 2])
            )
            self.assertTrue(bool(obs["action_mask"][0, 2]))
            self.assertFalse(bool(obs["action_mask"][0, 4]))

            obs, reward, terminated, truncated, _ = env.step(
                np.asarray([2], dtype=np.int32)
            )
            self.assertAlmostEqual(float(reward[0]), 10.0)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertEqual(int(obs["player_locations"]["x"][0]), 2)
            self.assertEqual(int(obs["score"][0]), 10)
            np.testing.assert_array_equal(
                obs["pellet_locations"][0, 0], np.asarray([-1, -1])
            )

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
