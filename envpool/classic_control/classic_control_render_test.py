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

from typing import Any, cast

import gymnasium as gym
import numpy as np
from absl.testing import absltest

import envpool.classic_control.registration  # noqa: F401
from envpool.registration import make_gymnasium

_TASK_SIZES = {
    "CartPole-v1": (400, 600),
    "Pendulum-v1": (500, 500),
    "MountainCar-v0": (400, 600),
    "MountainCarContinuous-v0": (400, 600),
    "Acrobot-v1": (500, 500),
}
_FIRST_FRAME_MEAN_DIFF_THRESHOLD = 4.0


def _batched_action(space: Any, num_envs: int) -> np.ndarray:
    sample = space.sample()
    sample_arr = np.asarray(sample)
    if sample_arr.ndim == 0:
        return np.full((num_envs,), sample, dtype=sample_arr.dtype)
    return np.repeat(sample_arr[np.newaxis, ...], num_envs, axis=0)


def _render_array(env: Any, env_ids: Any = None) -> np.ndarray:
    frame = env.render(env_ids=env_ids)
    assert frame is not None
    return cast(np.ndarray, frame)


def _sync_oracle_state(
    task_id: str, oracle: gym.Env, obs: np.ndarray, info: dict[str, Any]
) -> None:
    unwrapped = cast(Any, oracle.unwrapped)
    if task_id == "CartPole-v1":
        unwrapped.state = np.asarray(obs, dtype=np.float64).copy()
    elif task_id == "Pendulum-v1":
        theta = float(np.arctan2(obs[1], obs[0]))
        unwrapped.state = np.asarray([theta, obs[2]], dtype=np.float64)
        unwrapped.last_u = None
    elif task_id in {"MountainCar-v0", "MountainCarContinuous-v0"}:
        unwrapped.state = np.asarray(obs, dtype=np.float64).copy()
    elif task_id == "Acrobot-v1":
        unwrapped.state = np.concatenate([
            np.asarray(info["state"][0], dtype=np.float64),
            np.asarray(obs[-2:], dtype=np.float64),
        ])
    else:
        raise KeyError(task_id)


class ClassicControlRenderTest(absltest.TestCase):
    """Render regression tests for classic control environments."""

    def test_render_is_batch_consistent_and_state_invariant(self) -> None:
        """Rendering should be batch-consistent and free of side effects."""
        for task_id, (height, width) in _TASK_SIZES.items():
            with self.subTest(task_id=task_id):
                env = make_gymnasium(
                    task_id, num_envs=2, render_mode="rgb_array"
                )
                try:
                    env.reset()
                    frame0 = _render_array(env)
                    frame1 = _render_array(env, env_ids=1)
                    frames = _render_array(env, env_ids=[0, 1])
                    frame0_again = _render_array(env)

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
                    stepped0 = _render_array(env)
                    stepped_frames = _render_array(env, env_ids=[0, 1])
                    stepped0_again = _render_array(env)

                    np.testing.assert_array_equal(
                        stepped0[0], stepped_frames[0]
                    )
                    np.testing.assert_array_equal(stepped0, stepped0_again)
                finally:
                    env.close()

    def test_render_matches_upstream_first_frame(self) -> None:
        """The first rendered frame should stay close to Gymnasium output."""
        for task_id, (height, width) in _TASK_SIZES.items():
            with self.subTest(task_id=task_id):
                env = make_gymnasium(
                    task_id, num_envs=1, seed=0, render_mode="rgb_array"
                )
                oracle = gym.make(task_id, render_mode="rgb_array")
                try:
                    obs, info = env.reset()
                    oracle.reset(seed=0)
                    _sync_oracle_state(
                        task_id, oracle, np.asarray(obs[0]), info
                    )

                    frame = _render_array(env)[0].astype(np.int16)
                    expected = np.asarray(oracle.render(), dtype=np.int16)

                    self.assertEqual(frame.shape, (height, width, 3))
                    self.assertEqual(frame.dtype, np.int16)
                    self.assertEqual(expected.shape, (height, width, 3))
                    self.assertLess(
                        np.abs(frame - expected).mean(),
                        _FIRST_FRAME_MEAN_DIFF_THRESHOLD,
                    )
                finally:
                    env.close()
                    oracle.close()


if __name__ == "__main__":
    absltest.main()
