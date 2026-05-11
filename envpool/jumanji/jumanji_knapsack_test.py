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
"""Native Knapsack rule tests."""

from __future__ import annotations

from typing import Any

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium


def _items() -> tuple[list[float], list[float]]:
    weights = [0.2, 0.4, 0.7] + [1.0] * 47
    values = [0.3, 0.5, 0.9] + [0.0] * 47
    return weights, values


def _make_env() -> Any:
    weights, values = _items()
    return make_gymnasium(
        "Knapsack-v1",
        num_envs=1,
        seed=0,
        knapsack_weights=",".join(map(str, weights)),
        knapsack_values=",".join(map(str, values)),
        knapsack_total_budget=0.6,
        render_mode="rgb_array",
    )


class JumanjiKnapsackTest(absltest.TestCase):
    """Checks native Knapsack transitions."""

    def test_dense_reward_and_action_mask(self) -> None:
        """Checks dense reward and mask updates for valid items."""
        env = _make_env()
        try:
            obs, info = env.reset()
            np.testing.assert_allclose(obs["weights"][0, :3], [0.2, 0.4, 0.7])
            np.testing.assert_allclose(obs["values"][0, :3], [0.3, 0.5, 0.9])
            np.testing.assert_array_equal(obs["packed_items"][0], [False] * 50)
            expected_mask = np.asarray([True, True, False] + [False] * 47)
            np.testing.assert_array_equal(obs["action_mask"][0], expected_mask)
            self.assertAlmostEqual(float(info["remaining_budget"][0]), 0.6)

            obs, reward, terminated, truncated, info = env.step(
                np.asarray([0], dtype=np.int32)
            )
            self.assertAlmostEqual(float(reward[0]), 0.3)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertTrue(bool(obs["packed_items"][0, 0]))
            expected_mask = np.asarray([False, True, False] + [False] * 47)
            np.testing.assert_array_equal(obs["action_mask"][0], expected_mask)
            self.assertAlmostEqual(float(info["remaining_budget"][0]), 0.4)

            obs, reward, terminated, truncated, info = env.step(
                np.asarray([1], dtype=np.int32)
            )
            self.assertAlmostEqual(float(reward[0]), 0.5)
            self.assertTrue(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            np.testing.assert_array_equal(obs["action_mask"][0], [False] * 50)
            self.assertAlmostEqual(float(info["remaining_budget"][0]), 0.0)

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()

    def test_invalid_item_terminates_with_zero_reward(self) -> None:
        """Checks invalid items terminate with zero reward."""
        env = _make_env()
        try:
            env.reset()
            obs, reward, terminated, truncated, _ = env.step(
                np.asarray([2], dtype=np.int32)
            )
            self.assertEqual(float(reward[0]), 0.0)
            self.assertTrue(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertFalse(bool(obs["packed_items"][0, 2]))
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
