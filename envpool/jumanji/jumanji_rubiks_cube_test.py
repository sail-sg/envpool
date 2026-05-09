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
"""Native RubiksCube rule tests."""

# ruff: noqa: D102

from __future__ import annotations

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium


def _is_solved(cube: np.ndarray) -> bool:
    return bool(np.all(cube == cube[:, :1, :1]))


class JumanjiRubiksCubeTest(absltest.TestCase):
    """Checks native RubiksCube moves for both registered IDs."""

    def test_move_and_inverse_restore_solved_cube(self) -> None:
        for task_id in ("RubiksCube-v0", "RubiksCube-partly-scrambled-v0"):
            with self.subTest(task_id=task_id):
                env = make_gymnasium(
                    task_id,
                    num_envs=1,
                    seed=0,
                    rubiks_cube_num_scrambles=0,
                    render_mode="rgb_array",
                )
                try:
                    obs, _ = env.reset()
                    self.assertTrue(_is_solved(obs["cube"][0]))

                    obs, reward, terminated, truncated, _ = env.step(
                        np.asarray([[0, 0, 0]], dtype=np.int32)
                    )
                    cube = obs["cube"][0]
                    self.assertFalse(_is_solved(cube))
                    np.testing.assert_array_equal(cube[1, 0], [2, 2, 2])
                    np.testing.assert_array_equal(cube[4, 0], [1, 1, 1])
                    np.testing.assert_array_equal(cube[3, 0], [4, 4, 4])
                    np.testing.assert_array_equal(cube[2, 0], [3, 3, 3])
                    self.assertEqual(float(reward[0]), 0.0)
                    self.assertFalse(bool(terminated[0]))
                    self.assertFalse(bool(truncated[0]))

                    obs, reward, terminated, truncated, _ = env.step(
                        np.asarray([[0, 0, 1]], dtype=np.int32)
                    )
                    self.assertTrue(_is_solved(obs["cube"][0]))
                    self.assertEqual(float(reward[0]), 1.0)
                    self.assertTrue(bool(terminated[0]))
                    self.assertFalse(bool(truncated[0]))

                    frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

                    assert frame is not None
                    self.assertEqual(frame.shape, (1, 256, 256, 3))
                    self.assertGreater(int(frame.max() - frame.min()), 0)
                finally:
                    env.close()


if __name__ == "__main__":
    absltest.main()
