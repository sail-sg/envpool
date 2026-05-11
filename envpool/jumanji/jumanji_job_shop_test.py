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
"""Native JobShop rule tests."""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium


class JumanjiJobShopTest(absltest.TestCase):
    """Checks native JobShop transitions."""

    def test_schedule_two_jobs(self) -> None:
        """Checks scheduling advances both jobs."""
        env = make_gymnasium(
            "JobShop-v0", num_envs=1, seed=0, render_mode="rgb_array"
        )
        try:
            obs, _ = env.reset()
            self.assertEqual(int(obs["ops_machine_ids"][0, 0, 0]), 0)
            self.assertEqual(int(obs["ops_machine_ids"][0, 1, 0]), 1)
            self.assertTrue(bool(obs["action_mask"][0, 0, 0]))
            self.assertTrue(bool(obs["action_mask"][0, 1, 1]))

            action = np.full((1, 10), 20, dtype=np.int32)
            action[0, 0] = 0
            action[0, 1] = 1
            obs, reward, terminated, truncated, _ = env.step(action)
            self.assertAlmostEqual(float(reward[0]), -1.0)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertEqual(int(obs["machines_job_ids"][0, 0]), 0)
            self.assertEqual(int(obs["machines_job_ids"][0, 1]), 1)
            self.assertEqual(int(obs["machines_remaining_times"][0, 0]), 1)
            self.assertEqual(int(obs["machines_remaining_times"][0, 1]), 2)
            self.assertFalse(bool(obs["action_mask"][0, 0, 0]))

            action.fill(20)
            obs, _, terminated, truncated, _ = env.step(action)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertEqual(int(obs["machines_job_ids"][0, 0]), 20)
            self.assertEqual(int(obs["machines_remaining_times"][0, 1]), 1)

            obs, _, terminated, truncated, _ = env.step(action)
            self.assertTrue(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertEqual(int(obs["machines_job_ids"][0, 1]), 20)

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
