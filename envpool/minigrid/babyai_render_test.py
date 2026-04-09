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
"""Render tests for BabyAI environments."""

from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import numpy as np
from absl.testing import absltest

from envpool.minigrid.babyai_test_utils import (
    babyai_task_ids,
    debug_state,
    patch_render_state,
)
from envpool.registration import make_gymnasium

_RENDER_STEPS = 3
_REPRESENTATIVE_TASK_IDS = (
    "BabyAI-BossLevel-v0",
    "BabyAI-GoToObj-v0",
)


def _render_array(env: Any, env_ids: Any = None) -> np.ndarray:
    frame = env.render(env_ids=env_ids)
    assert frame is not None
    return cast(np.ndarray, frame)


def _zero_action(num_envs: int) -> np.ndarray:
    return np.zeros((num_envs,), dtype=np.int32)


class BabyAIRenderTest(absltest.TestCase):
    """Render regression tests for BabyAI environments."""

    def _oracle_frame(
        self,
        task_id: str,
        state: Any,
        step_count: int,
    ) -> np.ndarray:
        oracle = gym.make(task_id, render_mode="rgb_array")
        try:
            oracle.reset(seed=0)
            patch_render_state(oracle.unwrapped, state, step_count)
            return cast(
                np.ndarray,
                cast(Any, oracle.unwrapped).get_frame(
                    highlight=False,
                    tile_size=32,
                    agent_pov=False,
                ),
            )
        finally:
            oracle.close()

    def test_render_matches_upstream_oracle_for_multiple_steps_for_all_tasks(
        self,
    ) -> None:
        """Rendered frames should match the upstream BabyAI oracle."""
        for task_id in babyai_task_ids():
            with self.subTest(task_id=task_id):
                print(f"render {task_id}", flush=True)
                env = make_gymnasium(
                    task_id,
                    num_envs=1,
                    render_mode="rgb_array",
                )
                try:
                    env.reset()
                    step_count = 0
                    for step_idx in range(_RENDER_STEPS):
                        frame = _render_array(env)
                        frame_again = _render_array(env)
                        expected = self._oracle_frame(
                            task_id,
                            debug_state(env, 0),
                            step_count,
                        )

                        self.assertEqual(frame.shape, (1,) + expected.shape)
                        self.assertEqual(frame.dtype, np.uint8)
                        np.testing.assert_array_equal(frame[0], expected)
                        np.testing.assert_array_equal(frame, frame_again)
                        if step_idx + 1 < _RENDER_STEPS:
                            env.step(_zero_action(1))
                            step_count += 1
                finally:
                    env.close()

    def test_render_is_batch_consistent_and_state_invariant(self) -> None:
        """Rendering should be batch-consistent and side-effect free."""
        for task_id in _REPRESENTATIVE_TASK_IDS:
            with self.subTest(task_id=task_id):
                env = make_gymnasium(
                    task_id,
                    num_envs=2,
                    render_mode="rgb_array",
                )
                try:
                    env.reset()
                    step_count = 0
                    for step_idx in range(_RENDER_STEPS):
                        frame0 = _render_array(env)
                        frame1 = _render_array(env, env_ids=1)
                        frames = _render_array(env, env_ids=[0, 1])
                        frame0_again = _render_array(env)

                        expected0 = self._oracle_frame(
                            task_id,
                            debug_state(env, 0),
                            step_count,
                        )
                        expected1 = self._oracle_frame(
                            task_id,
                            debug_state(env, 1),
                            step_count,
                        )

                        self.assertEqual(frame0.shape, (1,) + expected0.shape)
                        self.assertEqual(frame1.shape, (1,) + expected1.shape)
                        self.assertEqual(frames.shape, (2,) + expected0.shape)
                        self.assertEqual(frame0.dtype, np.uint8)
                        self.assertEqual(frames.dtype, np.uint8)
                        np.testing.assert_array_equal(frame0[0], expected0)
                        np.testing.assert_array_equal(frame1[0], expected1)
                        np.testing.assert_array_equal(frame0[0], frames[0])
                        np.testing.assert_array_equal(frame1[0], frames[1])
                        np.testing.assert_array_equal(frame0, frame0_again)

                        if step_idx + 1 < _RENDER_STEPS:
                            env.step(_zero_action(2))
                            step_count += 1
                finally:
                    env.close()


if __name__ == "__main__":
    absltest.main()
