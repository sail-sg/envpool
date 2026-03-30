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
"""Render tests for classic control environments."""

from __future__ import annotations

from typing import Any

import numpy as np
from absl.testing import absltest

import envpool.classic_control.registration  # noqa: F401
from envpool.registration import make_gym


_TASK_SIZES = {
    "CartPole-v1": (400, 600),
    "Pendulum-v1": (500, 500),
    "MountainCar-v0": (400, 600),
    "MountainCarContinuous-v0": (400, 600),
    "Acrobot-v1": (500, 500),
}


def _batched_action(space: Any, num_envs: int) -> np.ndarray:
    sample = space.sample()
    sample_arr = np.asarray(sample)
    if sample_arr.ndim == 0:
      return np.full((num_envs,), sample, dtype=sample_arr.dtype)
    return np.repeat(sample_arr[np.newaxis, ...], num_envs, axis=0)


class ClassicControlRenderTest(absltest.TestCase):
    def test_render_is_batch_consistent_and_state_invariant(self) -> None:
        for task_id, (height, width) in _TASK_SIZES.items():
            with self.subTest(task_id=task_id):
                env = make_gym(task_id, num_envs=2, render_mode="rgb_array")
                try:
                    env.reset()
                    frame0 = env.render()
                    frame1 = env.render(env_ids=1)
                    frames = env.render(env_ids=[0, 1])
                    frame0_again = env.render()

                    self.assertEqual(frame0.shape, (1, height, width, 3))
                    self.assertEqual(frame1.shape, (1, height, width, 3))
                    self.assertEqual(frames.shape, (2, height, width, 3))
                    self.assertEqual(frame0.dtype, np.uint8)
                    self.assertEqual(frames.dtype, np.uint8)
                    np.testing.assert_array_equal(frame0[0], frames[0])
                    np.testing.assert_array_equal(frame1[0], frames[1])
                    np.testing.assert_array_equal(frame0, frame0_again)

                    action = _batched_action(env.action_space, 2)
                    env.step(action)
                    stepped0 = env.render()
                    stepped_frames = env.render(env_ids=[0, 1])
                    stepped0_again = env.render()

                    np.testing.assert_array_equal(stepped0[0], stepped_frames[0])
                    np.testing.assert_array_equal(stepped0, stepped0_again)
                finally:
                    env.close()


if __name__ == "__main__":
    absltest.main()
