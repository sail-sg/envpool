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
"""Native RobotWarehouse rule tests."""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium


class JumanjiRobotWarehouseTest(absltest.TestCase):
    """Checks native RobotWarehouse transitions."""

    def test_agents_turn_move_and_load_shelves(self) -> None:
        """Checks heading-relative moves and shelf pickup after reset sync."""
        env = make_gymnasium(
            "RobotWarehouse-v0",
            num_envs=1,
            seed=0,
            render_mode="rgb_array",
            robot_warehouse_render_agent_x="0,3,6,9",
            robot_warehouse_render_agent_y="0,0,0,0",
            robot_warehouse_render_agent_direction="1,1,1,1",
        )
        try:
            obs, _ = env.reset()
            self.assertEqual(int(obs["agents_view"][0, 0, 0]), 0)
            self.assertEqual(int(obs["agents_view"][0, 0, 1]), 0)
            self.assertTrue(bool(obs["action_mask"][0, 0, 2]))
            self.assertTrue(bool(obs["action_mask"][0, 0, 4]))

            obs, reward, terminated, truncated, _ = env.step(
                np.asarray([[1, 1, 1, 1]], dtype=np.int32)
            )
            self.assertAlmostEqual(float(reward[0]), 0.0)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertEqual(int(obs["agents_view"][0, 0, 1]), 1)
            self.assertEqual(int(obs["agents_view"][0, 3, 1]), 1)
            self.assertEqual(int(obs["step_count"][0]), 1)

            obs, _, terminated, _, _ = env.step(
                np.full((1, 4), 4, dtype=np.int32)
            )
            np.testing.assert_array_equal(
                obs["agents_view"][0, :, 2], [0, 1, 1, 0]
            )
            # Shelves block a loaded robot's next forward action.
            np.testing.assert_array_equal(
                obs["action_mask"][0, :, 1], [True, False, False, True]
            )
            obs, _, _, _, _ = env.step(np.full((1, 4), 2, dtype=np.int32))
            np.testing.assert_array_equal(
                obs["agents_view"][0, :, 3], [1, 1, 1, 1]
            )
            self.assertFalse(bool(terminated[0]))

            for _ in range(3):
                obs, reward, terminated, _, _ = env.step(
                    np.asarray([[0, 0, 0, 1]], dtype=np.int32)
                )
            self.assertTrue(bool(terminated[0]))
            self.assertEqual(float(reward[0]), 0.0)
            np.testing.assert_array_equal(
                obs["agents_view"][0, 2, :2], obs["agents_view"][0, 3, :2]
            )

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
