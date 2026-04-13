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
"""Render tests for native MetaWorld v3 tasks."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from absl.testing import absltest

from envpool.python.glfw_context import preload_windows_gl_dlls

preload_windows_gl_dlls(strict=True)

import envpool.mujoco.metaworld.registration as metaworld_registration  # noqa: E402
from envpool.registration import make_gymnasium  # noqa: E402

_RENDER_WIDTH = 64
_RENDER_HEIGHT = 48
_RENDER_STEPS = 3
_TASK_IDS = tuple(
    f"MetaWorld/{task_name}"
    for task_name in metaworld_registration.metaworld_v3_envs
)


def _zero_action(space: Any, num_envs: int) -> np.ndarray:
    sample = np.asarray(space.sample())
    zero = np.zeros_like(sample)
    if sample.ndim == 0:
        return np.full((num_envs,), zero.item(), dtype=sample.dtype)
    return np.repeat(zero[np.newaxis, ...], num_envs, axis=0)


def _render_array(env: Any, env_ids: Any = None) -> np.ndarray:
    frame = env.render(env_ids=env_ids)
    assert frame is not None
    return cast(np.ndarray, frame)


class MetaWorldRenderTest(absltest.TestCase):
    """Render regression tests for native MetaWorld tasks."""

    def test_rgb_array_render_all_v3_tasks_reset_and_multistep(self) -> None:
        """Every v3 task should render nonblank reset and stepped frames."""
        for task_id in _TASK_IDS:
            with self.subTest(task_id=task_id):
                env = make_gymnasium(
                    task_id,
                    num_envs=1,
                    seed=0,
                    render_mode="rgb_array",
                    render_width=_RENDER_WIDTH,
                    render_height=_RENDER_HEIGHT,
                )
                try:
                    env.reset()
                    action = _zero_action(env.action_space, 1)
                    for step_idx in range(_RENDER_STEPS):
                        frame = _render_array(env)
                        self.assertEqual(
                            frame.shape,
                            (1, _RENDER_HEIGHT, _RENDER_WIDTH, 3),
                        )
                        self.assertEqual(frame.dtype, np.uint8)
                        self.assertGreater(
                            int(frame.max()) - int(frame.min()), 0
                        )
                        if step_idx + 1 < _RENDER_STEPS:
                            env.step(action)
                finally:
                    env.close()

    def test_rgb_array_render_is_batch_consistent_and_state_invariant(
        self,
    ) -> None:
        """Batch rendering should match env-id rendering without side effects."""
        env = make_gymnasium(
            "MetaWorld/reach-v3",
            num_envs=2,
            seed=0,
            render_mode="rgb_array",
            render_width=_RENDER_WIDTH,
            render_height=_RENDER_HEIGHT,
        )
        try:
            env.reset()
            action = _zero_action(env.action_space, 2)
            for step_idx in range(_RENDER_STEPS):
                frame0 = _render_array(env)
                frame1 = _render_array(env, env_ids=1)
                frames = _render_array(env, env_ids=[0, 1])
                frame0_again = _render_array(env)

                self.assertEqual(
                    frame0.shape,
                    (1, _RENDER_HEIGHT, _RENDER_WIDTH, 3),
                )
                self.assertEqual(
                    frame1.shape,
                    (1, _RENDER_HEIGHT, _RENDER_WIDTH, 3),
                )
                self.assertEqual(
                    frames.shape,
                    (2, _RENDER_HEIGHT, _RENDER_WIDTH, 3),
                )
                self.assertEqual(frame0.dtype, np.uint8)
                self.assertEqual(frame1.dtype, np.uint8)
                self.assertEqual(frames.dtype, np.uint8)
                np.testing.assert_array_equal(frame0[0], frames[0])
                np.testing.assert_array_equal(frame1[0], frames[1])
                np.testing.assert_array_equal(frame0, frame0_again)
                if step_idx + 1 < _RENDER_STEPS:
                    env.step(action)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
