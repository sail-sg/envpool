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

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import numpy as np
from absl.testing import absltest

import envpool.vizdoom.registration as reg
from envpool.registration import make_gym

_UNSTABLE_TASK_IDS = {
    # These scenarios are not reliably renderable in the current local test
    # setup: CIG and multi-duel can segfault during initialization, and the
    # custom task has no bundled scenario config.
    "Cig-v1",
    "MultiDuel-v1",
    "VizdoomCustom-v1",
}
_TASK_IDS = tuple(
    sorted(
        f"{''.join(piece.capitalize() for piece in game.split('_'))}-v1"
        for game in reg._vizdoom_game_list()
        if f"{''.join(piece.capitalize() for piece in game.split('_'))}-v1"
        not in _UNSTABLE_TASK_IDS
    )
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


def _cleanup_runtime_dir() -> None:
    if os.path.isdir("_vizdoom"):
        shutil.rmtree("_vizdoom")
    elif os.path.exists("_vizdoom"):
        os.remove("_vizdoom")


@contextmanager
def _temporary_workdir() -> Iterator[None]:
    prev_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="vizdoom-render-") as tempdir:
        os.chdir(tempdir)
        try:
            yield
        finally:
            os.chdir(prev_cwd)


class VizdoomRenderTest(absltest.TestCase):
    """Render regression tests for VizDoom environments."""

    def _expected_frame(self, obs: np.ndarray) -> np.ndarray:
        if obs.shape[0] == 1:
            gray = obs[0]
            return np.repeat(gray[:, :, np.newaxis], 3, axis=2)
        return obs.transpose(1, 2, 0)

    def test_render_matches_screen_buffer_for_multiple_steps_for_stable_tasks(
        self,
    ) -> None:
        """Stable scenarios should render the same pixels across steps."""
        for task_id in _TASK_IDS:
            with self.subTest(task_id=task_id):
                with _temporary_workdir():
                    _cleanup_runtime_dir()
                    env = make_gym(
                        task_id,
                        num_envs=1,
                        render_mode="rgb_array",
                        render_width=320,
                        render_height=240,
                        use_combined_action=True,
                        stack_num=1,
                        img_width=320,
                        img_height=240,
                    )
                    try:
                        obs, _ = env.reset()
                        for step_idx in range(_RENDER_STEPS):
                            frame = _render_array(env)
                            frame_again = _render_array(env)
                            expected = self._expected_frame(obs[0])

                            self.assertEqual(frame.shape, (1, 240, 320, 3))
                            self.assertEqual(frame.dtype, np.uint8)
                            np.testing.assert_array_equal(frame[0], expected)
                            np.testing.assert_array_equal(frame, frame_again)
                            if step_idx + 1 < _RENDER_STEPS:
                                obs, _, _, _, _ = env.step(
                                    _zero_action(env.action_space, 1)
                                )
                    finally:
                        env.close()
                        _cleanup_runtime_dir()


if __name__ == "__main__":
    absltest.main()
