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
"""Render tests for Procgen environments."""

from typing import Any, cast

import numpy as np
from absl.testing import absltest

import envpool.procgen.registration as reg
from envpool.registration import make_gym

_TASK_IDS = tuple(
    sorted([
        f"{env_name.capitalize()}{reg.distribution[dist_mode]}-v0"
        for env_name, _, dist_modes in reg.procgen_game_config
        for dist_mode in dist_modes
    ])
)
_RENDER_STEPS = 3


def _render_array(env: Any, env_ids: Any = None) -> np.ndarray:
    frame = env.render(env_ids=env_ids)
    assert frame is not None
    return cast(np.ndarray, frame)


def _zero_action(space: Any, num_envs: int) -> np.ndarray:
    sample = np.asarray(space.sample())
    zero = np.zeros_like(sample)
    if sample.ndim == 0:
        return np.full((num_envs,), zero.item(), dtype=sample.dtype)
    return np.repeat(zero[np.newaxis, ...], num_envs, axis=0)


class ProcgenRenderTest(absltest.TestCase):
    """Render regression tests for Procgen environments."""

    def test_render_matches_obs_for_multiple_steps_for_all_tasks(self) -> None:
        """Rendered frames should match the channel-last observation."""
        for task_id in _TASK_IDS:
            with self.subTest(task_id=task_id):
                env = make_gym(
                    task_id,
                    num_envs=1,
                    render_mode="rgb_array",
                    channel_first=False,
                )
                try:
                    obs, _ = env.reset()
                    for step_idx in range(_RENDER_STEPS):
                        frame = _render_array(env)
                        frame_again = _render_array(env)

                        self.assertEqual(frame.shape, (1, 64, 64, 3))
                        self.assertEqual(frame.dtype, np.uint8)
                        np.testing.assert_array_equal(frame[0], obs[0])
                        np.testing.assert_array_equal(frame, frame_again)
                        if step_idx + 1 < _RENDER_STEPS:
                            obs, _, _, _, _ = env.step(
                                _zero_action(env.action_space, 1)
                            )
                finally:
                    env.close()

    def test_render_is_batch_consistent_for_representative_task(self) -> None:
        """Batch render output should match per-env renders for CoinRun."""
        env = make_gym(
            "CoinrunHard-v0",
            num_envs=2,
            render_mode="rgb_array",
            channel_first=False,
        )
        try:
            obs, _ = env.reset()
            for step_idx in range(_RENDER_STEPS):
                frame0 = _render_array(env)
                frame1 = _render_array(env, env_ids=1)
                frames = _render_array(env, env_ids=[0, 1])
                frame0_again = _render_array(env)

                self.assertEqual(frame0.shape, (1, 64, 64, 3))
                self.assertEqual(frame1.shape, (1, 64, 64, 3))
                self.assertEqual(frame0.dtype, np.uint8)
                self.assertEqual(frames.shape, (2, 64, 64, 3))
                self.assertEqual(frames.dtype, np.uint8)
                np.testing.assert_array_equal(frame0[0], obs[0])
                np.testing.assert_array_equal(frame1[0], obs[1])
                np.testing.assert_array_equal(frame0[0], frames[0])
                np.testing.assert_array_equal(frame1[0], frames[1])
                np.testing.assert_array_equal(frame0, frame0_again)
                if step_idx + 1 < _RENDER_STEPS:
                    obs, _, _, _, _ = env.step(
                        _zero_action(env.action_space, 2)
                    )
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
