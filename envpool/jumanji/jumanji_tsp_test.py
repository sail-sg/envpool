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
"""Native TSP rule tests."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium


def _line_coordinates() -> np.ndarray:
    coordinates = np.zeros((20, 2), dtype=np.float32)
    coordinates[:, 0] = np.linspace(0.0, 1.0, 20, dtype=np.float32)
    return coordinates


def _make_env() -> Any:
    coordinates = _line_coordinates()
    return make_gymnasium(
        "TSP-v1",
        num_envs=1,
        seed=0,
        tsp_coordinates=",".join(map(str, coordinates.reshape(-1))),
        render_mode="rgb_array",
    )


class JumanjiTSPTest(absltest.TestCase):
    """Checks native TSP dense rewards and masks."""

    def test_line_tour_matches_dense_reward(self) -> None:
        """Checks a line tour receives the dense distance reward."""
        coordinates = _line_coordinates()
        env = _make_env()
        try:
            obs, _ = env.reset()
            np.testing.assert_allclose(obs["coordinates"][0], coordinates)
            self.assertEqual(int(obs["position"][0]), -1)
            np.testing.assert_array_equal(obs["trajectory"][0], [-1] * 20)
            np.testing.assert_array_equal(obs["action_mask"][0], [True] * 20)

            for city in range(20):
                obs, reward, terminated, truncated, _ = env.step(
                    np.asarray([city], dtype=np.int32)
                )
                expected = 0.0 if city == 0 else -1.0 / 19.0
                if city == 19:
                    expected -= 1.0
                self.assertAlmostEqual(float(reward[0]), expected, places=6)
                self.assertEqual(int(obs["position"][0]), city)
                self.assertEqual(int(obs["trajectory"][0, city]), city)
                self.assertFalse(bool(obs["action_mask"][0, city]))
                self.assertEqual(bool(terminated[0]), city == 19)
                self.assertFalse(bool(truncated[0]))

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()

    def test_revisiting_city_terminates_with_penalty(self) -> None:
        """Checks revisiting a city terminates with penalty."""
        env = _make_env()
        try:
            env.reset()
            env.step(np.asarray([0], dtype=np.int32))
            obs, reward, terminated, truncated, _ = env.step(
                np.asarray([0], dtype=np.int32)
            )
            self.assertAlmostEqual(float(reward[0]), -20.0 * math.sqrt(2.0))
            self.assertTrue(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertEqual(int(obs["position"][0]), 0)
            self.assertEqual(int(obs["trajectory"][0, 0]), 0)
            self.assertEqual(int(obs["trajectory"][0, 1]), -1)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
