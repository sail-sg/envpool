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

import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

import gymnasium
import numpy as np
from absl import logging
from absl.testing import absltest

from envpool.python import glfw_context as envpool_glfw_context
from envpool.python.glfw_context import preload_windows_gl_dlls
from envpool.registration import make_gymnasium, make_spec, registry

preload_windows_gl_dlls(strict=True)

RENDER_WIDTH = 64
RENDER_HEIGHT = 48
NUM_STEPS = 3
EGL_TEARDOWN_RENDER_WIDTH = 84
EGL_TEARDOWN_RENDER_HEIGHT = 84
EGL_TEARDOWN_FRAME_STACK = 3
EGL_TEARDOWN_SUBPROCESS_TIMEOUT_SECONDS = 180
EGL_ASYNC_CONTEXT_STEPS = 64
EglTeardownCase = tuple[str, str, str]


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


def _assert_nested_equal(lhs: Any, rhs: Any) -> None:
    if isinstance(lhs, dict):
        assert isinstance(rhs, dict)
        assert lhs.keys() == rhs.keys()
        for key in lhs:
            _assert_nested_equal(lhs[key], rhs[key])
        return
    np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs))


def _subprocess_output_to_text(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output


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


def assert_pixel_env_preserves_gym_info_fields(test: absltest.TestCase) -> None:
    """Checks that Gym pixel envs keep task-specific info specs and values."""
    task_id = "Ant-v4"
    info_keys = [
        "info:reward_forward",
        "info:reward_ctrl",
        "info:reward_contact",
        "info:reward_survive",
        "info:x_velocity",
        "info:y_velocity",
    ]
    pixel_spec = make_spec(
        task_id,
        from_pixels=True,
        render_width=RENDER_WIDTH,
        render_height=RENDER_HEIGHT,
    )
    for key in info_keys:
        test.assertIn(key, pixel_spec.state_array_spec)

    state_env = make_gymnasium(task_id, num_envs=1, seed=0)
    pixel_env = make_gymnasium(
        task_id,
        num_envs=1,
        seed=0,
        from_pixels=True,
        render_mode="rgb_array",
        render_width=RENDER_WIDTH,
        render_height=RENDER_HEIGHT,
    )
    try:
        _, state_info = state_env.reset()
        _, pixel_info = pixel_env.reset()
        _assert_nested_equal(state_info, pixel_info)

        action = _zero_action(state_env.action_space, 1)
        _, _, _, _, state_info = state_env.step(action)
        _, _, _, _, pixel_info = pixel_env.step(action)
        _assert_nested_equal(state_info, pixel_info)
    finally:
        state_env.close()
        pixel_env.close()


def assert_tasks_align_with_render_for_three_steps(
    test: absltest.TestCase, import_path: str
) -> None:
    """Checks that each task matches `render()` for reset + 3 steps."""
    for task_id in task_ids_for_import_path(import_path):
        with test.subTest(task_id=task_id):
            logging.info("creating pixel spec for %s", task_id)
            make_spec(
                task_id,
                from_pixels=True,
                render_width=RENDER_WIDTH,
                render_height=RENDER_HEIGHT,
            )
            logging.info("created pixel spec for %s", task_id)
            logging.info("creating pixel env for %s", task_id)
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
                logging.info("created pixel env for %s", task_id)
                obs, _ = env.reset()
                logging.info("reset pixel env for %s", task_id)
                test.assertEqual(obs.shape, (1, 3, RENDER_HEIGHT, RENDER_WIDTH))
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


def assert_egl_pixel_env_teardown_exits_cleanly(
    test: absltest.TestCase,
    cases: Sequence[EglTeardownCase],
) -> None:
    """Checks EGL pixel envs from multiple families exit without GL noise.

    The subprocess intentionally keeps envs alive until interpreter shutdown:
    issue #401 happens in teardown, after the rollout itself has succeeded.
    """
    if platform.system() != "Linux":
        test.skipTest("EGL teardown regression is Linux-specific.")
    env = dict(os.environ)
    env["MUJOCO_GL"] = "egl"
    env.setdefault("EGL_PLATFORM", "surfaceless")

    package_parent = os.path.dirname(
        os.path.dirname(os.path.dirname(envpool_glfw_context.__file__))
    )
    python_paths = [package_parent] + [
        path for path in sys.path if path and path != package_parent
    ]
    python_path = os.pathsep.join(python_paths)
    if env.get("PYTHONPATH"):
        python_path = f"{python_path}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = python_path

    code = f"""
import importlib
import sys

sys.path.insert(0, {package_parent!r})
sys.modules.pop("envpool", None)
from envpool.registration import make

envs = []
for label, registration_module, task_id in {tuple(cases)!r}:
    importlib.import_module(registration_module)
    pixels = make(
        task_id,
        env_type="gymnasium",
        num_envs=2,
        from_pixels=True,
        frame_stack={EGL_TEARDOWN_FRAME_STACK},
        render_width={EGL_TEARDOWN_RENDER_WIDTH},
        render_height={EGL_TEARDOWN_RENDER_HEIGHT},
    )
    obs, info = pixels.reset()
    assert obs.shape == (
        2,
        {3 * EGL_TEARDOWN_FRAME_STACK},
        {EGL_TEARDOWN_RENDER_HEIGHT},
        {EGL_TEARDOWN_RENDER_WIDTH},
    ), (label, task_id, obs.shape)
    envs.append(pixels)
    print("successful", label, task_id)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=EGL_TEARDOWN_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _subprocess_output_to_text(exc.stdout)
        stderr = _subprocess_output_to_text(exc.stderr)
        test.fail(
            "EGL teardown subprocess timed out after "
            f"{EGL_TEARDOWN_SUBPROCESS_TIMEOUT_SECONDS} seconds.\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    test.assertEqual(
        result.returncode,
        0,
        msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )
    test.assertNotIn(
        "OpenGL error 0x502 in or before mjr_makeContext",
        result.stderr,
        msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )


def assert_egl_async_pixel_envs_do_not_share_context_concurrently(
    test: absltest.TestCase,
) -> None:
    """Checks async pixel rendering does not reuse one EGL context in workers."""
    if platform.system() != "Linux":
        test.skipTest("EGL context regression is Linux-specific.")
    env = dict(os.environ)
    env["MUJOCO_GL"] = "egl"
    env.setdefault("EGL_PLATFORM", "surfaceless")

    package_parent = os.path.dirname(
        os.path.dirname(os.path.dirname(envpool_glfw_context.__file__))
    )
    python_paths = [package_parent] + [
        path for path in sys.path if path and path != package_parent
    ]
    python_path = os.pathsep.join(python_paths)
    if env.get("PYTHONPATH"):
        python_path = f"{python_path}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = python_path

    code = f"""
import importlib
import sys

import numpy as np

sys.path.insert(0, {package_parent!r})
sys.modules.pop("envpool", None)
importlib.import_module("envpool.mujoco.dmc.registration")
from envpool.registration import make

pixels = make(
    "WalkerWalk-v1",
    env_type="gymnasium",
    num_envs=4,
    batch_size=2,
    num_threads=2,
    from_pixels=True,
    render_width={RENDER_WIDTH},
    render_height={RENDER_HEIGHT},
)
pixels.async_reset()
for _ in range(2):
    obs, reward, term, trunc, info = pixels.recv()
    assert obs.shape == (2, 3, {RENDER_HEIGHT}, {RENDER_WIDTH}), obs.shape

action = np.zeros((2,) + pixels.action_space.shape, dtype=np.float64)
pairs = [
    np.asarray([0, 2], dtype=np.int32),
    np.asarray([1, 3], dtype=np.int32),
    np.asarray([0, 3], dtype=np.int32),
    np.asarray([1, 2], dtype=np.int32),
]
for i in range({EGL_ASYNC_CONTEXT_STEPS}):
    pixels.send(action, pairs[i % len(pairs)])
    obs, reward, term, trunc, info = pixels.recv()
    assert obs.shape == (2, 3, {RENDER_HEIGHT}, {RENDER_WIDTH}), obs.shape

pixels.close()
print("successful async egl context stress")
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=EGL_TEARDOWN_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _subprocess_output_to_text(exc.stdout)
        stderr = _subprocess_output_to_text(exc.stderr)
        test.fail(
            "EGL async context subprocess timed out after "
            f"{EGL_TEARDOWN_SUBPROCESS_TIMEOUT_SECONDS} seconds.\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    test.assertEqual(
        result.returncode,
        0,
        msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )
    test.assertNotIn(
        "failed to make EGL context current",
        result.stderr,
        msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )
