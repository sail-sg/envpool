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
"""Render tests for VizDoom environments."""

import numpy as np
from absl.testing import absltest

import envpool.vizdoom.registration  # noqa: F401
from envpool.registration import make_gym


class VizdoomRenderTest(absltest.TestCase):
    def test_render_matches_screen_buffer_and_is_batch_consistent(self) -> None:
        env = make_gym(
            "D1Basic-v1",
            num_envs=2,
            render_mode="rgb_array",
            use_combined_action=True,
            stack_num=1,
            img_width=320,
            img_height=240,
        )
        try:
            obs, _ = env.reset()
            frame0 = env.render()
            frame1 = env.render(env_ids=1)
            frames = env.render(env_ids=[0, 1])
            frame0_again = env.render()

            gray0 = obs[0, 0]
            gray1 = obs[1, 0]

            self.assertEqual(frame0.shape, (1, 240, 320, 3))
            self.assertEqual(frame1.shape, (1, 240, 320, 3))
            self.assertEqual(frames.shape, (2, 240, 320, 3))
            self.assertEqual(frame0.dtype, np.uint8)
            self.assertEqual(frames.dtype, np.uint8)
            for channel in range(3):
                np.testing.assert_array_equal(frame0[0, :, :, channel], gray0)
                np.testing.assert_array_equal(frame1[0, :, :, channel], gray1)
            np.testing.assert_array_equal(frame0[0], frames[0])
            np.testing.assert_array_equal(frame1[0], frames[1])
            np.testing.assert_array_equal(frame0, frame0_again)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
