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
"""Tests for the batched MuJoCo render API."""

import numpy as np
from absl.testing import absltest

import envpool.mujoco.gym.registration  # noqa: F401
from envpool.registration import make_gym


def _maybe_skip_render_error(testcase: absltest.TestCase, exc: RuntimeError) -> None:
    message = str(exc)
    if any(
        needle in message
        for needle in (
            "failed to initialize EGL",
            "failed to get EGL display",
            "unsupported on this platform/build",
            "failed to create CGL",
            "failed to create EGL",
        )
    ):
        testcase.skipTest(message)
    raise exc


class MujocoRenderTest(absltest.TestCase):
    """Render regression tests for Gym-style MuJoCo tasks."""

    def test_rgb_array_render_is_batch_consistent_and_state_invariant(self) -> None:
        """RGB renders should be batch-consistent and free of side effects."""
        env = make_gym(
            "Ant-v5",
            num_envs=2,
            render_mode="rgb_array",
            render_width=64,
            render_height=48,
        )
        try:
            env.reset()
            try:
                frame0 = env.render()
                frame1 = env.render(env_ids=1)
                frames = env.render(env_ids=[0, 1])
                frame0_again = env.render()
            except RuntimeError as exc:
                _maybe_skip_render_error(self, exc)

            self.assertEqual(frame0.shape, (1, 48, 64, 3))
            self.assertEqual(frame1.shape, (1, 48, 64, 3))
            self.assertEqual(frame0.dtype, np.uint8)
            self.assertEqual(frames.shape, (2, 48, 64, 3))
            self.assertEqual(frames.dtype, np.uint8)
            np.testing.assert_array_equal(frame0[0], frames[0])
            np.testing.assert_array_equal(frame1[0], frames[1])
            np.testing.assert_array_equal(frame0, frame0_again)
        finally:
            env.close()

    def test_human_render_uses_python_viewer(self) -> None:
        """Human mode should route rendered frames through the Python viewer."""
        env = make_gym(
            "Ant-v5",
            num_envs=1,
            render_mode="human",
            render_width=32,
            render_height=24,
        )
        shown: list[np.ndarray] = []
        env._show_human_frame = lambda frame: shown.append(np.array(frame))  # type: ignore[method-assign]
        try:
            env.reset()
            try:
                result = env.render()
            except RuntimeError as exc:
                _maybe_skip_render_error(self, exc)

            self.assertIsNone(result)
            self.assertLen(shown, 1)
            self.assertEqual(shown[0].shape, (24, 32, 3))
            self.assertEqual(shown[0].dtype, np.uint8)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
