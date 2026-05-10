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
"""Render tests for PGX Go environments."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from absl.testing import absltest

import envpool.pgx.registration  # noqa: F401
from envpool.registration import make_gymnasium

_TASKS = (
    ("Go9x9-v1", 9),
    ("Go19x19-v1", 19),
    ("ChineseGo9x9-v1", 9),
    ("ChineseGo19x19-v1", 19),
)


def _render_array(env: Any, env_ids: Any = None) -> np.ndarray:
    frame = env.render(env_ids=env_ids)
    assert frame is not None
    return cast(np.ndarray, frame)


class PgxGoRenderTest(absltest.TestCase):
    """RGB render coverage for native PGX Go."""

    def test_render_reset_and_multiple_steps_for_all_tasks(self) -> None:
        """Every public Go task should render reset and multi-step frames."""
        for task_id, size in _TASKS:
            with self.subTest(task_id=task_id):
                env = make_gymnasium(
                    task_id,
                    num_envs=1,
                    seed=0,
                    render_mode="rgb_array",
                    render_width=96,
                    render_height=96,
                )
                env.reset()
                reset_frame = _render_array(env)
                reset_again = _render_array(env)
                self.assertEqual(reset_frame.shape, (1, 96, 96, 3))
                self.assertEqual(reset_frame.dtype, np.uint8)
                np.testing.assert_array_equal(reset_frame, reset_again)
                self.assertGreater(
                    int(reset_frame.max()) - int(reset_frame.min()),
                    0,
                )

                env.step(np.asarray([size * size // 2], dtype=np.int32))
                first_stone = _render_array(env)
                self.assertGreater(
                    int(np.count_nonzero(first_stone != reset_frame)),
                    0,
                )
                env.step(np.asarray([0], dtype=np.int32))
                second_stone = _render_array(env)
                self.assertGreater(
                    int(np.count_nonzero(second_stone != first_stone)),
                    0,
                )

    def test_batched_render_env_id_selection(self) -> None:
        """Batched render should respect explicit env-id selection."""
        env = make_gymnasium(
            "Go9x9-v1",
            board_size=5,
            num_envs=2,
            seed=0,
            render_mode="rgb_array",
            render_width=80,
            render_height=80,
        )
        env.reset()
        env.step(np.asarray([12, 0], dtype=np.int32))

        frame0 = _render_array(env, env_ids=0)
        frame1 = _render_array(env, env_ids=1)
        frames = _render_array(env, env_ids=[0, 1])
        default_frame = _render_array(env)
        self.assertEqual(frame0.shape, (1, 80, 80, 3))
        self.assertEqual(frame1.shape, (1, 80, 80, 3))
        self.assertEqual(frames.shape, (2, 80, 80, 3))
        np.testing.assert_array_equal(frame0[0], frames[0])
        np.testing.assert_array_equal(frame1[0], frames[1])
        np.testing.assert_array_equal(default_frame, frame0)
        self.assertGreater(int(np.count_nonzero(frame0 != frame1)), 0)

    def test_render_marks_stone_colors_at_grid_points(self) -> None:
        """Rendered stones should use stable black and white colors."""
        env = make_gymnasium(
            "Go9x9-v1",
            board_size=5,
            num_envs=1,
            seed=0,
            render_mode="rgb_array",
            render_width=160,
            render_height=160,
        )
        env.reset()
        env.step(np.asarray([12], dtype=np.int32))
        black_frame = _render_array(env)
        np.testing.assert_array_less(black_frame[0, 80, 80], 40)

        env.step(np.asarray([0], dtype=np.int32))
        black_white_frame = _render_array(env)
        self.assertTrue(bool(np.all(black_white_frame[0, 12, 12] > 200)))


if __name__ == "__main__":
    absltest.main()
