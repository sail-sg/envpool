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
"""Native Cleaner rule tests."""

from __future__ import annotations

from typing import Any

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium

_DIRTY = 0
_CLEAN = 1


def _make_env() -> Any:
    grid = [_DIRTY] * 100
    return make_gymnasium(
        "Cleaner-v0",
        num_envs=1,
        seed=0,
        cleaner_grid=",".join(map(str, grid)),
        render_mode="rgb_array",
    )


class JumanjiCleanerTest(absltest.TestCase):
    """Checks native Cleaner multi-agent transitions."""

    def test_valid_moves_clean_unique_tiles(self) -> None:
        """Checks valid moves clean each tile once."""
        env = _make_env()
        try:
            obs, info = env.reset()
            self.assertEqual(int(obs["grid"][0, 0, 0]), _CLEAN)
            np.testing.assert_array_equal(
                obs["agents_locations"][0], [[0, 0]] * 3
            )
            np.testing.assert_array_equal(
                obs["action_mask"][0],
                [[False, True, True, False]] * 3,
            )
            self.assertAlmostEqual(
                float(info["ratio_dirty_tiles"][0]), 99 / 100
            )
            self.assertEqual(int(info["num_dirty_tiles"][0]), 99)

            obs, reward, terminated, truncated, info = env.step(
                np.asarray([[1, 2, 1]], dtype=np.int32)
            )
            np.testing.assert_array_equal(
                obs["agents_locations"][0], [[0, 1], [1, 0], [0, 1]]
            )
            self.assertEqual(int(obs["grid"][0, 0, 1]), _CLEAN)
            self.assertEqual(int(obs["grid"][0, 1, 0]), _CLEAN)
            self.assertAlmostEqual(float(reward[0]), 1.5)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertEqual(int(info["num_dirty_tiles"][0]), 97)

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()

    def test_invalid_action_terminates_after_valid_agents_move(self) -> None:
        """Checks invalid actions terminate after valid agents move."""
        env = _make_env()
        try:
            env.reset()
            obs, reward, terminated, truncated, info = env.step(
                np.asarray([[1, 2, 0]], dtype=np.int32)
            )
            np.testing.assert_array_equal(
                obs["agents_locations"][0], [[0, 1], [1, 0], [0, 0]]
            )
            self.assertAlmostEqual(float(reward[0]), 1.5)
            self.assertTrue(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertEqual(int(info["num_dirty_tiles"][0]), 97)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
