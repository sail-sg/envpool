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
"""Render tests for Atari environments."""

import numpy as np
from absl.testing import absltest

import envpool.atari.registration as reg
import envpool.atari.registration  # noqa: F401
from envpool.registration import make_gym

_TASK_IDS = tuple(
    f"{''.join(piece.capitalize() for piece in game.split('_'))}-v5"
    for game in reg.atari_game_list
)


class AtariRenderTest(absltest.TestCase):
    def test_render_matches_raw_rgb_obs_first_frame_for_all_tasks(self) -> None:
        for task_id in _TASK_IDS:
            with self.subTest(task_id=task_id):
                env = make_gym(
                    task_id,
                    num_envs=1,
                    render_mode="rgb_array",
                    gray_scale=False,
                    stack_num=1,
                    img_height=210,
                    img_width=160,
                    use_inter_area_resize=False,
                )
                try:
                    obs, _ = env.reset()
                    frame = env.render()

                    expected = obs[0].transpose(1, 2, 0)

                    self.assertEqual(frame.shape, (1, 210, 160, 3))
                    self.assertEqual(frame.dtype, np.uint8)
                    np.testing.assert_array_equal(frame[0], expected)
                finally:
                    env.close()

    def test_render_is_batch_consistent_for_representative_task(self) -> None:
        env = make_gym(
            "Pong-v5",
            num_envs=2,
            render_mode="rgb_array",
            gray_scale=False,
            stack_num=1,
            img_height=210,
            img_width=160,
            use_inter_area_resize=False,
        )
        try:
            obs, _ = env.reset()
            frame0 = env.render()
            frame1 = env.render(env_ids=1)
            frames = env.render(env_ids=[0, 1])
            frame0_again = env.render()

            expected0 = obs[0].transpose(1, 2, 0)
            expected1 = obs[1].transpose(1, 2, 0)

            self.assertEqual(frame0.shape, (1, 210, 160, 3))
            self.assertEqual(frame1.shape, (1, 210, 160, 3))
            self.assertEqual(frames.shape, (2, 210, 160, 3))
            self.assertEqual(frame0.dtype, np.uint8)
            self.assertEqual(frames.dtype, np.uint8)
            np.testing.assert_array_equal(frame0[0], expected0)
            np.testing.assert_array_equal(frame1[0], expected1)
            np.testing.assert_array_equal(frame0[0], frames[0])
            np.testing.assert_array_equal(frame1[0], frames[1])
            np.testing.assert_array_equal(frame0, frame0_again)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
