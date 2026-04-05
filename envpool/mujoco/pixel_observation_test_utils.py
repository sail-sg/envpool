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
"""Shared helpers for native MuJoCo pixel-observation tests."""

from typing import Any

import gymnasium
import numpy as np
from absl.testing import absltest

from envpool.python.glfw_context import preload_windows_gl_dlls
from envpool.registration import make_gymnasium, make_spec, registry

preload_windows_gl_dlls(strict=True)

RENDER_WIDTH = 64
RENDER_HEIGHT = 48
NUM_STEPS = 3


def task_ids_for_import_path(import_path: str) -> list[str]:
    """Returns all registered task ids for one MuJoCo family."""
    return sorted(
        task_id
        for task_id, (task_import_path, _, _) in registry.specs.items()
        if task_import_path == import_path
    )


def first_task_id_for_import_path(import_path: str) -> str:
    """Returns one representative task id for a MuJoCo family."""
    task_ids = task_ids_for_import_path(import_path)
    if not task_ids:
        raise ValueError(f"No MuJoCo tasks registered for {import_path}.")
    return task_ids[0]


def _zero_action(space: Any, num_envs: int) -> Any:
    if isinstance(space, gymnasium.spaces.Dict):
        return {
            key: _zero_action(subspace, num_envs)
            for key, subspace in space.spaces.items()
        }
    sample = np.asarray(space.sample())
    zero = np.zeros_like(sample)
    if sample.ndim == 0:
        return np.full((num_envs,), zero.item(), dtype=sample.dtype)
    return np.repeat(zero[np.newaxis, ...], num_envs, axis=0)


def _render_to_bchw(frame: Any) -> np.ndarray:
    return np.transpose(np.asarray(frame), (0, 3, 1, 2))


def _assert_pixels_match(obs: np.ndarray, render: Any) -> None:
    np.testing.assert_array_equal(obs, _render_to_bchw(render))


def assert_make_spec_exposes_bchw_pixel_specs(
    test: absltest.TestCase, import_path: str
) -> None:
    """Checks that `make_spec` exposes channel-first pixel observations."""
    task_id = first_task_id_for_import_path(import_path)
    with test.subTest(task_id=task_id):
        spec = make_spec(
            task_id,
            from_pixels=True,
            frame_stack=3,
            render_width=RENDER_WIDTH,
            render_height=RENDER_HEIGHT,
        )
        test.assertEqual(
            spec.observation_space.shape,
            (9, RENDER_HEIGHT, RENDER_WIDTH),
        )
        test.assertEqual(spec.observation_space.dtype, np.uint8)
        test.assertEqual(
            spec.gymnasium_observation_space.shape,
            (9, RENDER_HEIGHT, RENDER_WIDTH),
        )
        dm_obs_spec = spec.observation_spec()
        test.assertEqual(
            dm_obs_spec.pixels.shape,
            (9, RENDER_HEIGHT, RENDER_WIDTH),
        )
        test.assertEqual(dm_obs_spec.pixels.dtype, np.uint8)
        test.assertTrue(hasattr(dm_obs_spec, "env_id"))


def assert_frame_stack_rolls_in_channel_dimension(
    test: absltest.TestCase, import_path: str
) -> None:
    """Checks that frame stacking shifts along the channel dimension."""
    task_id = first_task_id_for_import_path(import_path)
    with test.subTest(task_id=task_id):
        env = make_gymnasium(
            task_id,
            num_envs=1,
            seed=0,
            from_pixels=True,
            frame_stack=3,
            render_mode="rgb_array",
            render_width=RENDER_WIDTH,
            render_height=RENDER_HEIGHT,
        )
        try:
            obs0, _ = env.reset()
            test.assertEqual(
                obs0.shape,
                (1, 9, RENDER_HEIGHT, RENDER_WIDTH),
            )
            np.testing.assert_array_equal(obs0[:, 0:3], obs0[:, 3:6])
            np.testing.assert_array_equal(obs0[:, 3:6], obs0[:, 6:9])
            render0 = env.render(env_ids=[0])
            assert render0 is not None
            _assert_pixels_match(obs0[:, -3:], render0)

            prev_obs = obs0
            action = _zero_action(env.action_space, 1)
            for _ in range(2):
                obs, _, _, _, _ = env.step(action)
                render = env.render(env_ids=[0])
                assert render is not None
                np.testing.assert_array_equal(obs[:, :-3], prev_obs[:, 3:])
                _assert_pixels_match(obs[:, -3:], render)
                prev_obs = obs
        finally:
            env.close()


def assert_tasks_align_with_render_for_three_steps(
    test: absltest.TestCase, import_path: str
) -> None:
    """Checks that each task matches `render()` for reset + 3 steps."""
    for task_id in task_ids_for_import_path(import_path):
        with test.subTest(task_id=task_id):
            env = make_gymnasium(
                task_id,
                num_envs=1,
                seed=0,
                from_pixels=True,
                render_mode="rgb_array",
                render_width=RENDER_WIDTH,
                render_height=RENDER_HEIGHT,
            )
            try:
                obs, _ = env.reset()
                test.assertEqual(
                    obs.shape, (1, 3, RENDER_HEIGHT, RENDER_WIDTH)
                )
                test.assertEqual(obs.dtype, np.uint8)
                render = env.render(env_ids=[0])
                assert render is not None
                _assert_pixels_match(obs, render)

                action = _zero_action(env.action_space, 1)
                for _ in range(NUM_STEPS):
                    obs, _, _, _, _ = env.step(action)
                    test.assertEqual(
                        obs.shape, (1, 3, RENDER_HEIGHT, RENDER_WIDTH)
                    )
                    render = env.render(env_ids=[0])
                    assert render is not None
                    _assert_pixels_match(obs, render)
            finally:
                env.close()
