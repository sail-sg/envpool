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
"""Render tests for Box2D environments."""

import gymnasium as gym
import numpy as np
from absl.testing import absltest

import envpool.box2d.registration  # noqa: F401
from envpool.registration import make_gym


class Box2DRenderTest(absltest.TestCase):
    """Render regression tests for Box2D environments."""

    def _assert_batch_consistent_render(self, task_id: str) -> None:
        env = make_gym(
            task_id,
            num_envs=2,
            render_mode="rgb_array",
            render_width=64,
            render_height=48,
        )
        try:
            env.reset()
            frame0 = env.render()
            frame1 = env.render(env_ids=1)
            frames = env.render(env_ids=[0, 1])
            frame0_again = env.render()
            self.assertEqual(frame0.shape, (1, 48, 64, 3))
            self.assertEqual(frame1.shape, (1, 48, 64, 3))
            self.assertEqual(frames.shape, (2, 48, 64, 3))
            self.assertEqual(frame0.dtype, np.uint8)
            self.assertEqual(frames.dtype, np.uint8)
            np.testing.assert_array_equal(frame0[0], frames[0])
            np.testing.assert_array_equal(frame1[0], frames[1])
            np.testing.assert_array_equal(frame0, frame0_again)
        finally:
            env.close()

    def test_car_racing_render(self) -> None:
        """CarRacing should support consistent batched rendering."""
        self._assert_batch_consistent_render("CarRacing-v3")

    def test_bipedal_walker_render(self) -> None:
        """BipedalWalker should support consistent batched rendering."""
        self._assert_batch_consistent_render("BipedalWalker-v3")

    def test_bipedal_walker_render_matches_official_ground_profile(
        self,
    ) -> None:
        """BipedalWalker renders should preserve the official ground profile."""
        for task_id in ("BipedalWalker-v3", "BipedalWalkerHardcore-v3"):
            with self.subTest(task_id=task_id):
                env = make_gym(
                    task_id,
                    num_envs=1,
                    seed=0,
                    render_mode="rgb_array",
                    render_width=600,
                    render_height=400,
                )
                oracle = gym.make(task_id, render_mode="rgb_array")
                try:
                    env.reset()
                    oracle.reset(seed=0)
                    frame = env.render()[0].astype(np.int16)
                    expected = np.asarray(oracle.render(), dtype=np.int16)
                    lower_band = slice(-96, None)
                    diff = np.abs(
                        frame[lower_band] - expected[lower_band]
                    ).mean()
                    agent_crop = np.abs(
                        frame[100:320, :220] - expected[100:320, :220]
                    ).mean()
                    self.assertLess(diff, 40.0)
                    self.assertLess(agent_crop, 15.0)
                finally:
                    env.close()
                    oracle.close()

    def test_render_matches_official_first_frame(self) -> None:
        """Representative Box2D renders should stay close to Gymnasium."""
        thresholds = {
            "CarRacing-v3": 15.0,
            "LunarLander-v3": 30.0,
            "LunarLanderContinuous-v3": 30.0,
        }
        for task_id, threshold in thresholds.items():
            with self.subTest(task_id=task_id):
                env = make_gym(
                    task_id,
                    num_envs=1,
                    seed=0,
                    render_mode="rgb_array",
                )
                oracle = gym.make(task_id, render_mode="rgb_array")
                try:
                    env.reset()
                    oracle.reset(seed=0)
                    frame = env.render()[0].astype(np.int16)
                    expected = np.asarray(oracle.render(), dtype=np.int16)
                    self.assertEqual(frame.shape, expected.shape)
                    self.assertLess(np.abs(frame - expected).mean(), threshold)
                finally:
                    env.close()
                    oracle.close()

    def test_lunar_lander_render(self) -> None:
        """LunarLander should support consistent batched rendering."""
        self._assert_batch_consistent_render("LunarLander-v3")


if __name__ == "__main__":
    absltest.main()
