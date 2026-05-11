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
"""Native SlidingTilePuzzle rule tests."""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium


class JumanjiSlidingTilePuzzleTest(absltest.TestCase):
    """Checks native SlidingTilePuzzle transitions."""

    def test_one_move_solution_matches_dense_reward(self) -> None:
        """Checks a one-move solution receives dense reward."""
        initial = np.arange(1, 26, dtype=np.int32)
        initial[-2:] = [0, 24]
        initial = initial.reshape(5, 5)
        solved = np.arange(1, 26, dtype=np.int32)
        solved[-1] = 0
        solved = solved.reshape(5, 5)

        env = make_gymnasium(
            "SlidingTilePuzzle-v0",
            num_envs=1,
            seed=0,
            sliding_tile_initial_puzzle=",".join(map(str, initial.reshape(-1))),
            render_mode="rgb_array",
        )
        try:
            obs, info = env.reset()
            np.testing.assert_array_equal(obs["puzzle"][0], initial)
            np.testing.assert_array_equal(obs["empty_tile_position"][0], [4, 3])
            np.testing.assert_array_equal(
                obs["action_mask"][0], [True, True, False, True]
            )
            self.assertEqual(int(obs["step_count"][0]), 0)
            self.assertAlmostEqual(
                float(info["prop_correctly_placed"][0]), 23 / 25
            )

            obs, reward, terminated, truncated, info = env.step(
                np.asarray([2], dtype=np.int32)
            )
            np.testing.assert_array_equal(obs["puzzle"][0], initial)
            self.assertEqual(float(reward[0]), 0.0)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertEqual(int(obs["step_count"][0]), 1)
            self.assertAlmostEqual(
                float(info["prop_correctly_placed"][0]), 23 / 25
            )

            obs, reward, terminated, truncated, info = env.step(
                np.asarray([1], dtype=np.int32)
            )
            np.testing.assert_array_equal(obs["puzzle"][0], solved)
            np.testing.assert_array_equal(obs["empty_tile_position"][0], [4, 4])
            self.assertEqual(float(reward[0]), 2.0)
            self.assertTrue(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertAlmostEqual(float(info["prop_correctly_placed"][0]), 1.0)

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()

    def test_seeded_reset_is_deterministic(self) -> None:
        """Checks seeded resets produce identical puzzles."""
        env0 = make_gymnasium("SlidingTilePuzzle-v0", num_envs=4, seed=17)
        env1 = make_gymnasium("SlidingTilePuzzle-v0", num_envs=4, seed=17)
        try:
            obs0, _ = env0.reset()
            obs1, _ = env1.reset()
            np.testing.assert_array_equal(obs0["puzzle"], obs1["puzzle"])
            np.testing.assert_array_equal(
                obs0["empty_tile_position"], obs1["empty_tile_position"]
            )
        finally:
            env0.close()
            env1.close()


if __name__ == "__main__":
    absltest.main()
