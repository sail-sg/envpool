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
"""Native GraphColoring rule tests."""

# ruff: noqa: D102

from __future__ import annotations

from typing import Any

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium


def _make_env() -> Any:
    return make_gymnasium(
        "GraphColoring-v1",
        num_envs=1,
        seed=0,
        graph_coloring_edges="0-1",
        render_mode="rgb_array",
    )


class JumanjiGraphColoringTest(absltest.TestCase):
    """Checks native GraphColoring transitions."""

    def test_valid_coloring_finishes_with_unique_color_penalty(self) -> None:
        env = _make_env()
        try:
            obs, _ = env.reset()
            self.assertTrue(bool(obs["adj_matrix"][0, 0, 1]))
            self.assertTrue(bool(obs["adj_matrix"][0, 1, 0]))
            self.assertEqual(int(obs["current_node_index"][0]), 0)
            np.testing.assert_array_equal(obs["colors"][0], [-1] * 20)
            np.testing.assert_array_equal(obs["action_mask"][0], [True] * 20)

            actions = [0, 1] + [0] * 18
            for i, action in enumerate(actions):
                obs, reward, terminated, truncated, _ = env.step(
                    np.asarray([action], dtype=np.int32)
                )
                self.assertFalse(bool(truncated[0]))
                self.assertEqual(
                    int(obs["current_node_index"][0]), (i + 1) % 20
                )
                if i == 0:
                    self.assertFalse(bool(obs["action_mask"][0, 0]))
                    self.assertEqual(float(reward[0]), 0.0)
                    self.assertFalse(bool(terminated[0]))
                elif i == 19:
                    self.assertEqual(float(reward[0]), -2.0)
                    self.assertTrue(bool(terminated[0]))
                else:
                    self.assertEqual(float(reward[0]), 0.0)
                    self.assertFalse(bool(terminated[0]))

            np.testing.assert_array_equal(obs["colors"][0], actions)
            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))
            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()

    def test_invalid_neighbor_color_terminates(self) -> None:
        env = _make_env()
        try:
            env.reset()
            env.step(np.asarray([0], dtype=np.int32))
            obs, reward, terminated, truncated, _ = env.step(
                np.asarray([0], dtype=np.int32)
            )
            self.assertEqual(float(reward[0]), -20.0)
            self.assertTrue(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertEqual(obs["colors"][0, 0], 0)
            self.assertEqual(obs["colors"][0, 1], 0)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
